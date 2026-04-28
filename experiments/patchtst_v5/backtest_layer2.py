"""
backtest_layer2.py — Couche 2 trading method : TP adaptatif + trail break-even.

Pour chaque event sélectionné (top-K% par score modèle), simule bar-par-bar
un trade :

  - TP = premier pivot dans la direction tel que |tp − entry|/entry ≥ min_edge_pct
         (ex: H1 si déjà à ≥0.10%, sinon H2, sinon H3, sinon H4)
  - SL initial = pivot rang 4 opposé (L4 LONG / H4 SHORT) — inchangé vs label
  - Trail break-even : si le pivot immédiat (H1 LONG / L1 SHORT) est touché
    avant TP/SL, SL passe à entry_price → exit ≈ break-even si renversement
  - Walk-forward sur high/low entre idx+1 et idx+time_barrier
  - Tie-break same-bar TP+SL → SL (conservateur)
  - SKIP_NO_PROFITABLE_TP si aucun pivot ne dépasse min_edge_pct avant L4/H4

Exit reasons : TP_RANK0..3 (rang TP atteint), SL_INIT (full loss),
SL_BE (break-even après confirmation H1), TIMEOUT, AMBIGUOUS.

Avec --min-edge-pct 0.0 + --no-breakeven-trail → strict-H1 (réplique label).
Défaut : --min-edge-pct 0.10 + trail ON.

  Time barrier :
    - --time-barrier (default 24) bars max

Inputs:
    --predictions  : NPZ from predict_ensemble (scores, direction, feature_idx, timestamp)
    --features     : parquet OHLC + timestamps (mêmes que used by pivot_labeler_levels)
    --top-k-pct    : sélectionner top X% events par score (default 10)
    --time-to-h1   : bars avant time-stop conditionnel (default 8)
    --time-barrier : max bars total du trade (default 24)
    --fees-pct     : one-way fee % (default 0.02 = maker)
    --output-dir   : dir de sortie pour summary.json + trades_detail.csv

Usage:
    python -m experiments.patchtst_v5.backtest_layer2 \\
        --predictions models/patchtst_v5_pivot_sl4_btc_xgb_short_ensemble/predictions_test.npz \\
        --features data/patchtst_v5/features_btc.parquet \\
        --top-k-pct 10 \\
        --output-dir models/patchtst_v5_pivot_sl4_btc_xgb_short_ensemble/backtest_layer2_test/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .pivot_labeler_levels import compute_camarilla_5min

logger = logging.getLogger("patchtst_v5.backtest_layer2")

PIVOT_COLS = ["h1", "h2", "h3", "h4", "l1", "l2", "l3", "l4"]


# ---------------------------------------------------------------------------
# Trailing simulation
# ---------------------------------------------------------------------------

def _sorted_pivots_around(entry: float, pivots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (above_sorted_asc, below_sorted_desc) i.e. levels strictly above/below entry,
    nearest first in each list.
    """
    valid = np.sort(pivots[~np.isnan(pivots)])
    above = valid[valid > entry]                    # ascending : nearest first
    below = valid[valid < entry][::-1]              # descending : nearest first
    return above, below


def simulate_adaptive_tp(
    direction: int,
    entry_idx: int,
    entry_price: float,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    pivots_at_entry: np.ndarray,    # 8 Camarilla pivots [h1,h2,h3,h4,l1,l2,l3,l4]
    time_barrier: int,
    fees_pct: float,
    min_edge_pct: float,
    breakeven_trail: bool = True,
) -> dict:
    """Simule un trade avec TP adaptatif au premier pivot rentable.

    TP = premier pivot dans la direction tel que |tp − entry|/entry ≥ min_edge_pct
         (ex: H1 si distance ≥ 0.10%, sinon H2, sinon H3, sinon H4)
    SL = sl_pool[3] (L4 LONG / H4 SHORT) — inchangé vs label sl_level=4
    Walk-forward bar par bar sur [entry_idx+1, entry_idx+time_barrier].
    Tie-break (same-bar TP+SL) → SL conservateur (identique label).

    breakeven_trail=True : si le pivot immédiat (H1 LONG / L1 SHORT) est touché
    avant TP/SL, on déplace SL à entry_price (break-even). Évite les renversements
    catastrophiques sur les trades qui ont confirmé la direction sans atteindre TP.
    Nouvelles exit reasons : SL_BE (break-even hit après H1 touché).

    Avec min_edge_pct=0.0 → comportement identique au strict H1 du label.
    Avec min_edge_pct>0 → grimpe les pivots jusqu'à dépasser le seuil.
    """
    above, below = _sorted_pivots_around(entry_price, pivots_at_entry)
    if direction > 0:
        targets = above
        sl_pool = below
    else:
        targets = below
        sl_pool = above

    if len(targets) < 1 or len(sl_pool) < 4:
        return {
            "pnl_net": np.nan, "exit_bars": -1,
            "exit_reason": "SKIP_NOT_ENOUGH_PIVOTS",
            "exit_price": np.nan, "trail_level": -1,
        }

    # Choix du TP : premier target dont la distance dépasse min_edge_pct
    tp = None
    tp_rank = -1
    for rank, candidate in enumerate(targets):
        dist_pct = 100.0 * abs(float(candidate) - entry_price) / entry_price
        if dist_pct >= min_edge_pct:
            tp = float(candidate)
            tp_rank = rank
            break

    if tp is None:
        return {
            "pnl_net": np.nan, "exit_bars": -1,
            "exit_reason": "SKIP_NO_PROFITABLE_TP",
            "exit_price": np.nan, "trail_level": -1,
        }

    sl_init = float(sl_pool[3])
    sl = sl_init
    h1_pivot = float(targets[0])         # pivot immédiat (peut être == tp si tp_rank=0)
    h1_touched = False                    # devient True quand on a confirmé la direction

    n_bars = len(high)
    end = min(entry_idx + 1 + time_barrier, n_bars)
    horizon = end - (entry_idx + 1)

    if horizon <= 0:
        return {
            "pnl_net": np.nan, "exit_bars": -1,
            "exit_reason": "OUT_OF_DATA",
            "exit_price": np.nan, "trail_level": -1,
        }

    # Bar-by-bar : nécessaire pour gérer le trail break-even
    for k in range(horizon):
        bar = entry_idx + 1 + k
        bar_h = high[bar]
        bar_l = low[bar]

        if direction > 0:
            sl_hit = bar_l <= sl
            tp_hit = bar_h >= tp
            h1_now = bar_h >= h1_pivot
        else:
            sl_hit = bar_h >= sl
            tp_hit = bar_l <= tp
            h1_now = bar_l <= h1_pivot

        # Tie-break conservateur : SL prioritaire si même bar
        if sl_hit:
            if h1_touched:
                # SL trail à entry → exit ≈ break-even (perte ≈ fees seulement)
                return _exit(direction, entry_price, sl, k + 1, "SL_BE",
                             fees_pct, trail_level=-3)
            # Sinon SL initial = perte pleine
            return _exit(direction, entry_price, sl, k + 1, "SL_INIT",
                         fees_pct, trail_level=-2)
        if tp_hit:
            return _exit(direction, entry_price, tp, k + 1,
                         f"TP_RANK{tp_rank}", fees_pct, trail_level=tp_rank)

        # Pas d'exit cette bar : si H1 touché et trail activé, déplacer SL à entry
        if breakeven_trail and h1_now and not h1_touched:
            h1_touched = True
            sl = entry_price

    last_close = float(close[end - 1])
    return _exit(direction, entry_price, last_close, horizon, "TIMEOUT",
                 fees_pct, trail_level=-2)


def _exit(direction: int, entry: float, exit_p: float, bars: int,
          reason: str, fees_pct: float, trail_level: int) -> dict:
    if direction > 0:
        pnl = 100.0 * (exit_p - entry) / entry
    else:
        pnl = 100.0 * (entry - exit_p) / entry
    pnl_net = pnl - 2 * fees_pct
    return {
        "pnl_net": pnl_net,
        "exit_bars": bars,
        "exit_reason": reason,
        "exit_price": exit_p,
        "trail_level": trail_level,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predictions(npz_path: Path) -> dict:
    """Load predictions NPZ (scores, direction, feature_idx, timestamp, etc.)."""
    data = np.load(npz_path, allow_pickle=False)
    out = {k: data[k] for k in data.files}
    if "feature_idx" not in out:
        raise SystemExit(
            f"feature_idx missing from {npz_path}. Re-run predict_ensemble after "
            "the asset_id propagation commit (7a09dd3) so feature_idx is preserved."
        )
    return out


def load_features_with_pivots(features_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load OHLC features + recompute Camarilla pivots (causal). Returns (df, pivots_8col)."""
    df = pd.read_parquet(features_path, columns=["timestamp", "high", "low", "close"])
    pivot_levels = compute_camarilla_5min(df["timestamp"],
                                          df["high"].values.astype("float64"),
                                          df["low"].values.astype("float64"),
                                          df["close"].values.astype("float64"))
    return df, pivot_levels[PIVOT_COLS].values  # (n_bars, 8)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_results(trades: pd.DataFrame, span_days: float) -> dict:
    """Compute aggregated metrics for a set of simulated trades."""
    n = len(trades)
    if n == 0:
        return {"n_trades": 0}

    pnl = trades["pnl_net"].dropna().values
    wins = (pnl > 0).sum()
    losses = (pnl <= 0).sum()
    wr = wins / len(pnl) if len(pnl) > 0 else 0.0

    cumul = pnl.sum()
    avg_net = pnl.mean()
    std_net = pnl.std()
    span_years = span_days / 365.25 if span_days > 0 else 1.0
    ann_ret = cumul / span_years
    ann_std = std_net * np.sqrt(len(pnl) / span_years) if std_net > 0 else 0.0
    sharpe = ann_ret / ann_std if ann_std > 1e-9 else 0.0

    # Equity curve & MaxDD
    eq = np.cumsum(pnl)
    running_max = np.maximum.accumulate(eq)
    dd = eq - running_max
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-9 else 0.0

    # Exit reason distribution
    exit_reasons = trades["exit_reason"].value_counts().to_dict()

    # Trail level distribution
    trail_dist = trades["trail_level"].value_counts().sort_index().to_dict()

    return {
        "n_trades": int(n),
        "win_rate": float(wr),
        "wins": int(wins),
        "losses": int(losses),
        "avg_net_pct": float(avg_net),
        "std_net_pct": float(std_net),
        "cumul_net_pct": float(cumul),
        "ann_ret_pct": float(ann_ret),
        "ann_std_pct": float(ann_std),
        "sharpe": float(sharpe),
        "max_dd_pct": float(max_dd),
        "calmar": float(calmar),
        "exit_reasons": exit_reasons,
        "trail_level_distribution": trail_dist,
    }


def log_summary(name: str, metrics: dict) -> None:
    if metrics.get("n_trades", 0) == 0:
        logger.info("%-40s : 0 trades", name)
        return
    logger.info(
        "%-40s n=%5d  WR=%.3f  AvgNet=%+.4f%%  Cumul=%+8.2f%%  AnnRet=%+7.2f%%  "
        "Sharpe=%+.2f  MaxDD=%+7.2f%%  Calmar=%+.2f",
        name, metrics["n_trades"], metrics["win_rate"], metrics["avg_net_pct"],
        metrics["cumul_net_pct"], metrics["ann_ret_pct"], metrics["sharpe"],
        metrics["max_dd_pct"], metrics["calmar"],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, required=True,
                   help="NPZ from predict_ensemble (single direction)")
    p.add_argument("--features", type=Path, required=True,
                   help="Features parquet for OHLC + Camarilla recomputation")
    p.add_argument("--top-k-pct", type=float, default=10.0,
                   help="Select top X%% events by score (default: 10)")
    p.add_argument("--time-barrier", type=int, default=24,
                   help="Max bars per trade (default: 24)")
    p.add_argument("--fees-pct", type=float, default=0.02,
                   help="One-way fee %% (default: 0.02 = maker)")
    p.add_argument("--min-edge-pct", type=float, default=0.10,
                   help="TP adaptatif : skip pivots dont distance < min-edge-pct, "
                        "viser le premier pivot rentable. Recommandé : "
                        "2*fees + edge_min (ex: 0.10 pour fees=0.02). "
                        "Mettre 0.0 pour comportement strict-H1.")
    p.add_argument("--no-breakeven-trail", action="store_true",
                   help="Désactive le trail SL→entry après touche du pivot immédiat. "
                        "Par défaut le trail est activé : si H1 (LONG) ou L1 (SHORT) "
                        "est touché avant TP/SL, SL passe à entry → exit break-even "
                        "au lieu de SL_INIT en cas de renversement.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--also-sweep-topk", action="store_true",
                   help="Also report results for several top-K%% (1, 2, 5, 10, 25, 50)")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading predictions: %s", args.predictions)
    pred = load_predictions(args.predictions)
    n_total = len(pred["scores"])

    span = (pd.to_datetime(pred["timestamp"]).max() - pd.to_datetime(pred["timestamp"]).min())
    span_days = span.total_seconds() / 86400.0

    logger.info("Loading features + Camarilla pivots: %s", args.features)
    features_df, pivots_8col = load_features_with_pivots(args.features)
    high = features_df["high"].values.astype("float64")
    low = features_df["low"].values.astype("float64")
    close = features_df["close"].values.astype("float64")

    logger.info("Total events available: %d  | span: %.1f days (%.2f years)",
                n_total, span_days, span_days / 365.25)
    logger.info("Adaptive-TP config: time_barrier=%d bars  fees=%.3f%% one-way  "
                "min_edge_pct=%.3f%%  breakeven_trail=%s",
                args.time_barrier, args.fees_pct, args.min_edge_pct,
                "OFF" if args.no_breakeven_trail else "ON")

    # ------------------------------------------------------------------
    # Helper to run sim for a given top-K%% threshold
    # ------------------------------------------------------------------
    def run_for_topk(top_k_pct: float) -> tuple[pd.DataFrame, dict]:
        n_top = max(1, int(n_total * top_k_pct / 100))
        sorted_idx = np.argsort(-pred["scores"])
        selected = sorted_idx[:n_top]

        rows: list[dict] = []
        for k_idx in selected:
            entry_idx = int(pred["feature_idx"][k_idx])
            direction = int(pred["direction"][k_idx])
            entry_price = float(close[entry_idx])
            piv = pivots_8col[entry_idx]
            res = simulate_adaptive_tp(
                direction, entry_idx, entry_price,
                high, low, close, piv,
                args.time_barrier, args.fees_pct,
                args.min_edge_pct,
                breakeven_trail=not args.no_breakeven_trail,
            )
            res["score"] = float(pred["scores"][k_idx])
            res["direction"] = direction
            res["timestamp"] = pred["timestamp"][k_idx]
            res["entry_idx"] = entry_idx
            res["entry_price"] = entry_price
            res["y_true"] = int(pred["y_true"][k_idx]) if "y_true" in pred else -1
            # Sanity-check : pnl du label (calculé par pivot_labeler_levels avec
            # exactement la même logique strict-H1) doit matcher notre pnl_net.
            res["label_pnl_net"] = (float(pred["pnl_after_fees_pct"][k_idx])
                                    if "pnl_after_fees_pct" in pred else np.nan)
            rows.append(res)
        df = pd.DataFrame(rows)
        m = report_results(df, span_days)
        return df, m

    # ------------------------------------------------------------------
    # Main run @ requested top-K
    # ------------------------------------------------------------------
    logger.info("=" * 110)
    logger.info("ADAPTIVE-TP BACKTEST  (top %.1f%%, min_edge=%.3f%%)",
                args.top_k_pct, args.min_edge_pct)
    logger.info("=" * 110)
    main_df, main_metrics = run_for_topk(args.top_k_pct)
    log_summary(f"top_{args.top_k_pct:.0f}pct", main_metrics)

    # Save trades_detail
    trades_path = args.output_dir / f"trades_top{int(args.top_k_pct)}pct.csv"
    main_df.to_csv(trades_path, index=False)
    logger.info("Trades detail saved: %s", trades_path)

    # Exit reason breakdown
    logger.info("Exit reasons (top %.1f%%):", args.top_k_pct)
    for reason, count in sorted(main_metrics.get("exit_reasons", {}).items(),
                                 key=lambda x: -x[1]):
        logger.info("  %-20s : %5d (%.1f%%)",
                    reason, count, 100 * count / max(main_metrics["n_trades"], 1))

    # ------------------------------------------------------------------
    # Diagnostic overlap : combien de trades chevauchent en concurrence ?
    # IMPORTANT : le backtest simule chaque event indépendamment (capital
    # infini, multi-positions illimitées). Cette mesure indique l'écart
    # avec une politique single-position réaliste.
    # ------------------------------------------------------------------
    if "entry_idx" in main_df.columns and "exit_bars" in main_df.columns:
        traded = main_df[(main_df["exit_bars"] >= 0) &
                         (main_df["pnl_net"].notna())].copy()
        traded["start"] = traded["entry_idx"].astype(int) + 1
        traded["end"] = traded["start"] + traded["exit_bars"].astype(int) - 1
        traded = traded.sort_values("start").reset_index(drop=True)
        n_traded = len(traded)
        if n_traded > 0:
            # Sweep-line : compter les overlaps via events
            starts = traded["start"].values
            ends = traded["end"].values
            events = np.concatenate([starts, ends + 1])
            kinds = np.concatenate([np.ones(n_traded, dtype=int),
                                    -np.ones(n_traded, dtype=int)])
            order = np.argsort(events, kind="stable")
            running = np.cumsum(kinds[order])
            max_concurrent = int(running.max()) if running.size else 0
            mean_concurrent = float(running.mean()) if running.size else 0.0

            # Pour chaque trade, nombre de trades précédents non encore exits
            n_overlap = np.zeros(n_traded, dtype=int)
            for i in range(n_traded):
                # Combien des trades 0..i-1 ont end >= starts[i] ?
                if i == 0:
                    continue
                prev_ends = ends[:i]
                n_overlap[i] = int((prev_ends >= starts[i]).sum())
            n_with_overlap = int((n_overlap > 0).sum())

            logger.info("Overlap analysis (top %.1f%%) :", args.top_k_pct)
            logger.info("  Trades : %d  |  Avec >= 1 trade actif au moment de l'entrée : "
                        "%d (%.1f%%)",
                        n_traded, n_with_overlap, 100 * n_with_overlap / n_traded)
            logger.info("  Max concurrent trades : %d", max_concurrent)
            logger.info("  Mean concurrent trades : %.2f", mean_concurrent)
            # Si politique single-position appliquée : combien de trades
            # auraient été ignorés ?
            n_ignored_if_single = n_with_overlap
            n_kept_if_single = n_traded - n_ignored_if_single
            logger.info("  → Single-position policy : %d trades gardés "
                        "(%d ignorés, -%.1f%%)",
                        n_kept_if_single, n_ignored_if_single,
                        100 * n_ignored_if_single / n_traded)

    logger.info("TP rank distribution (rang du pivot atteint à l'exit) :")
    rank_labels = {-3: "SL_BE (break-even)", -2: "SL_INIT/TIMEOUT/AMBIG",
                   -1: "skipped",
                   0: "TP_H1 (rang 1)", 1: "TP_H2 (rang 2)",
                   2: "TP_H3 (rang 3)", 3: "TP_H4 (rang 4)"}
    for level, count in sorted(main_metrics.get("trail_level_distribution", {}).items()):
        logger.info("  rank %3d (%s) : %d (%.1f%%)",
                    level, rank_labels.get(level, "?"), count,
                    100 * count / max(main_metrics["n_trades"], 1))

    # ------------------------------------------------------------------
    # Distribution distance entry→TP (sur les exits TP_RANK*) : utile pour
    # vérifier que le min_edge_pct fait son travail.
    # ------------------------------------------------------------------
    if "entry_price" in main_df.columns and "exit_price" in main_df.columns:
        tp_dist_pct = []
        for _, r in main_df.iterrows():
            if not str(r["exit_reason"]).startswith("TP_RANK"):
                continue
            entry_p = r["entry_price"]
            d = int(r["direction"])
            if d > 0:
                tp_dist_pct.append(100 * (r["exit_price"] - entry_p) / entry_p)
            else:
                tp_dist_pct.append(100 * (entry_p - r["exit_price"]) / entry_p)
        if tp_dist_pct:
            arr = np.array(tp_dist_pct)
            logger.info("TP exit distance entry→TP : n=%d  mean=%.4f%%  median=%.4f%%  "
                        "P10=%.4f%%  P90=%.4f%%  min=%.4f%%  max=%.4f%%",
                        len(arr), arr.mean(), np.median(arr),
                        np.percentile(arr, 10), np.percentile(arr, 90),
                        arr.min(), arr.max())
            logger.info("Fraction TP exits au-dessus de min_edge_pct (%.3f%%) : %.1f%%",
                        args.min_edge_pct,
                        100 * (arr >= args.min_edge_pct).mean())

    # ------------------------------------------------------------------
    # Sanity check : compare backtest pnl_net vs label pnl_after_fees_pct
    # Le label couche-1 (pivot_labeler_levels sl_level=4) calcule exactement
    # la même chose. Toute divergence > seuil ⇒ bug dans le backtest.
    # ------------------------------------------------------------------
    if "label_pnl_net" in main_df.columns and main_df["label_pnl_net"].notna().any():
        cmp = main_df.dropna(subset=["pnl_net", "label_pnl_net"]).copy()
        cmp["delta"] = cmp["pnl_net"] - cmp["label_pnl_net"]
        n_cmp = len(cmp)
        n_match = int((cmp["delta"].abs() < 1e-4).sum())
        logger.info("Sanity check vs label pnl_after_fees_pct  (n=%d) : "
                    "match=%d (%.1f%%)  mean|Δ|=%.4f%%  max|Δ|=%.4f%%  "
                    "label_WR=%.3f  backtest_WR=%.3f",
                    n_cmp, n_match, 100 * n_match / max(n_cmp, 1),
                    float(cmp["delta"].abs().mean()),
                    float(cmp["delta"].abs().max()),
                    float((cmp["label_pnl_net"] > 0).mean()),
                    float((cmp["pnl_net"] > 0).mean()))
        # Échantillon des divergences les plus grandes pour debug
        if (cmp["delta"].abs() >= 1e-4).any():
            worst = cmp.assign(abs_delta=cmp["delta"].abs())\
                       .nlargest(5, "abs_delta")\
                       [["timestamp", "direction", "entry_price",
                         "exit_price", "exit_reason", "pnl_net",
                         "label_pnl_net", "delta", "y_true"]]
            logger.warning("Top 5 divergences (premier signe de bug) :\n%s",
                           worst.to_string(index=False))

    # ------------------------------------------------------------------
    # Optional sweep
    # ------------------------------------------------------------------
    sweep_results: dict = {}
    if args.also_sweep_topk:
        logger.info("=" * 110)
        logger.info("SWEEP top-K%%")
        logger.info("=" * 110)
        for tk in (1.0, 2.0, 5.0, 10.0, 25.0, 50.0):
            _, m = run_for_topk(tk)
            sweep_results[f"top_{tk:.0f}pct"] = m
            log_summary(f"top_{tk:.0f}pct", m)

    # ------------------------------------------------------------------
    # Save summary JSON
    # ------------------------------------------------------------------
    summary = {
        "config": {
            "predictions": str(args.predictions),
            "features": str(args.features),
            "top_k_pct": args.top_k_pct,
            "time_barrier": args.time_barrier,
            "fees_pct": args.fees_pct,
            "min_edge_pct": args.min_edge_pct,
            "breakeven_trail": not args.no_breakeven_trail,
            "mode": "adaptive_tp",
        },
        "n_total_events": int(n_total),
        "span_days": float(span_days),
        "main_run": main_metrics,
        "sweep": sweep_results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Summary saved: %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
