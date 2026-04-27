"""
backtest_layer2.py — Couche 2 trading method : strict close at H1 (réplique label sl_level=4).

Pour chaque event sélectionné (top-K% par score modèle), simule bar-par-bar
exactement la même logique que pivot_labeler_levels.label_events() avec
sl_level=4 :

  - TP = pivot rang 1 dans la direction (H1 pour LONG, L1 pour SHORT)
  - SL = pivot rang 4 opposé (L4 pour LONG, H4 pour SHORT)
  - Walk-forward sur high/low entre idx+1 et idx+time_barrier
  - argmax tie-break : si TP et SL touchés dans la même bar → AMBIGUOUS → SL
    (conservateur, identique au label)
  - Sinon : premier touché gagne (TP_H1 ou SL_INIT)
  - Aucun touch sur l'horizon → TIMEOUT (exit au last_close)

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


def simulate_strict_h1(
    direction: int,
    entry_idx: int,
    entry_price: float,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    pivots_at_entry: np.ndarray,    # 8 Camarilla pivots [h1,h2,h3,h4,l1,l2,l3,l4]
    time_barrier: int,
    fees_pct: float,
) -> dict:
    """Simule un trade strict H1 (réplique label sl_level=4).

    TP = targets[0] (H1 LONG / L1 SHORT)
    SL = sl_pool[3] (L4 LONG / H4 SHORT)
    Walk-forward bar par bar sur [entry_idx+1, entry_idx+time_barrier].
    Convention tie-break identique au label : same-bar TP+SL → AMBIGUOUS=SL.
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

    tp = float(targets[0])
    sl = float(sl_pool[3])

    n_bars = len(high)
    end = min(entry_idx + 1 + time_barrier, n_bars)
    sub_high = high[entry_idx + 1: end]
    sub_low = low[entry_idx + 1: end]
    horizon = sub_high.size

    if horizon == 0:
        return {
            "pnl_net": np.nan, "exit_bars": -1,
            "exit_reason": "OUT_OF_DATA",
            "exit_price": np.nan, "trail_level": -1,
        }

    if direction > 0:
        tp_hit = sub_high >= tp
        sl_hit = sub_low <= sl
    else:
        tp_hit = sub_low <= tp
        sl_hit = sub_high >= sl

    first_tp = int(np.argmax(tp_hit)) if tp_hit.any() else horizon
    first_sl = int(np.argmax(sl_hit)) if sl_hit.any() else horizon

    if first_tp < first_sl:
        return _exit(direction, entry_price, tp, first_tp + 1, "TP_H1",
                     fees_pct, trail_level=1)
    if first_sl < first_tp:
        return _exit(direction, entry_price, sl, first_sl + 1, "SL_INIT",
                     fees_pct, trail_level=0)
    if first_tp == first_sl and first_tp < horizon:
        # Convention conservative identique au label : same-bar tie → SL
        return _exit(direction, entry_price, sl, first_tp + 1, "AMBIGUOUS",
                     fees_pct, trail_level=0)

    # TIMEOUT : exit au last close
    last_close = float(close[end - 1])
    return _exit(direction, entry_price, last_close, horizon, "TIMEOUT",
                 fees_pct, trail_level=0)


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
    logger.info("Strict-H1 config: time_barrier=%d bars  fees=%.3f%% one-way",
                args.time_barrier, args.fees_pct)

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
            res = simulate_strict_h1(
                direction, entry_idx, entry_price,
                high, low, close, piv,
                args.time_barrier, args.fees_pct,
            )
            res["score"] = float(pred["scores"][k_idx])
            res["direction"] = direction
            res["timestamp"] = pred["timestamp"][k_idx]
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
    logger.info("TRAILING BACKTEST  (top %.1f%%)", args.top_k_pct)
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

    logger.info("Outcome distribution:")
    for level, count in sorted(main_metrics.get("trail_level_distribution", {}).items()):
        labels = {-1: "skipped", 0: "loss/timeout/ambig", 1: "TP_H1 win"}
        logger.info("  level %d (%s) : %d (%.1f%%)",
                    level, labels.get(level, "?"), count,
                    100 * count / max(main_metrics["n_trades"], 1))

    # ------------------------------------------------------------------
    # Distribution distance TP/SL → entry (en %), pour comprendre si la
    # rentabilité dépend de la proximité du pivot cible.
    # ------------------------------------------------------------------
    if "entry_price" in main_df.columns and "exit_price" in main_df.columns:
        # Recompute tp/sl distances depuis les pivots à l'entry (independent de exit_p)
        tp_dist_pct = []
        sl_dist_pct = []
        for _, r in main_df.iterrows():
            if r["exit_reason"] in ("SKIP_NOT_ENOUGH_PIVOTS", "OUT_OF_DATA"):
                continue
            entry_idx = None
            # Find the matching feature_idx via timestamp (cheap proxy)
            ts = r["timestamp"]
            entry_p = r["entry_price"]
            d = int(r["direction"])
            # use the saved label tp/sl computation: we don't have direct access,
            # so recompute via pivots_at_entry
            # (timestamp lookup is costly, instead we just report from exit_price for TP_H1)
            if r["exit_reason"] == "TP_H1":
                if d > 0:
                    tp_dist_pct.append(100 * (r["exit_price"] - entry_p) / entry_p)
                else:
                    tp_dist_pct.append(100 * (entry_p - r["exit_price"]) / entry_p)
        if tp_dist_pct:
            arr = np.array(tp_dist_pct)
            logger.info("TP_H1 distance entry→TP : n=%d  mean=%.4f%%  median=%.4f%%  "
                        "P10=%.4f%%  P90=%.4f%%  min=%.4f%%  max=%.4f%%",
                        len(arr), arr.mean(), np.median(arr),
                        np.percentile(arr, 10), np.percentile(arr, 90),
                        arr.min(), arr.max())
            logger.info("Fraction TP_H1 avec distance > 2×fees (%.3f%%) : %.1f%%",
                        2 * args.fees_pct,
                        100 * (arr > 2 * args.fees_pct).mean())

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
            "mode": "strict_h1",
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
