"""
backtest_layer2.py — Couche 2 trading method : trailing Camarilla 4 niveaux + time-stop H1.

Pour chaque event sélectionné (top-K% par score modèle), simule bar-par-bar un
trade avec gestion dynamique du stop-loss :

  Initial state:
    - SL initial = pivot Camarilla rang 4 opposé (L4 pour LONG, H4 pour SHORT)
    - Cible = pivot 1 dans la direction (H1 LONG, L1 SHORT)

  Trailing rules:
    - Atteinte H1 (LONG) ou L1 (SHORT) : SL → entry_price (lock break-even),
      cible suivante = H2/L2
    - Atteinte H2/L2 : SL → H1/L1, cible = H3/L3
    - Atteinte H3/L3 : SL → H2/L2, cible = H4/L4
    - Atteinte H4/L4 : exit

  Time-stop conditionnel (Idée 3 user) :
    - Si après --time-to-h1 bars (default 8), H1 toujours pas atteint → exit au close
    - Sinon le trade continue jusqu'à TP/SL/timeout

  Time barrier final :
    - --time-barrier (default 24) bars max → exit au close si rien ne s'est passé

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


def simulate_trailing(
    direction: int,
    entry_idx: int,
    entry_price: float,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    pivots_at_entry: np.ndarray,    # 8 Camarilla pivots [h1,h2,h3,h4,l1,l2,l3,l4]
    time_to_h1: int,
    time_barrier: int,
    fees_pct: float,
) -> dict:
    """Simule un trade avec trailing Camarilla 4 niveaux + time-stop H1."""
    above, below = _sorted_pivots_around(entry_price, pivots_at_entry)

    if direction > 0:
        targets = above        # [pivot_immediat, ..., pivot_lointain]
        sl_pool = below        # [pivot_immediat sous, ..., pivot lointain sous]
    else:
        targets = below        # SHORT : on vise les pivots EN-DESSOUS
        sl_pool = above        # SL en cas de mouvement contraire (au-dessus)

    # Need at least 4 targets in trade direction AND 4 sl_pool to set initial SL=4th opposé
    if len(targets) < 4 or len(sl_pool) < 4:
        return {
            "pnl_net": np.nan, "exit_bars": -1,
            "exit_reason": "SKIP_NOT_ENOUGH_PIVOTS",
            "exit_price": np.nan, "trail_level": -1,
        }

    sl = sl_pool[3]                # L4 (LONG) ou H4 (SHORT)
    target_idx = 0                  # current target index dans `targets`
    h1_reached = False
    n_bars = len(high)

    for k in range(time_barrier):
        bar = entry_idx + 1 + k
        if bar >= n_bars:
            break

        bar_h, bar_l, bar_c = high[bar], low[bar], close[bar]

        # 1) Check SL hit (priorité, vérifié AVANT TP par convention conservative)
        sl_hit = (direction > 0 and bar_l <= sl) or (direction < 0 and bar_h >= sl)
        if sl_hit:
            reason = "SL_TRAIL" if h1_reached else "SL_INIT"
            return _exit(direction, entry_price, sl, k + 1, reason, fees_pct,
                         trail_level=target_idx)

        # 2) Check TP cascade (peut traverser plusieurs niveaux dans la même bar)
        while target_idx < 4:
            tgt = targets[target_idx]
            tp_hit = (direction > 0 and bar_h >= tgt) or (direction < 0 and bar_l <= tgt)
            if not tp_hit:
                break
            if target_idx == 3:
                # Niveau 4 = TP final, exit
                return _exit(direction, entry_price, tgt, k + 1, "TP_FINAL",
                             fees_pct, trail_level=4)
            # Trail SL au niveau précédent (entry pour target_idx=0)
            sl = entry_price if target_idx == 0 else targets[target_idx - 1]
            if target_idx == 0:
                h1_reached = True
            target_idx += 1

        # 3) Time-stop conditionnel : si pas d'atteinte H1/L1 après time_to_h1 bars → exit close
        if (k + 1) == time_to_h1 and not h1_reached:
            return _exit(direction, entry_price, bar_c, k + 1, "TIMESTOP_H1",
                         fees_pct, trail_level=0)

    # Time barrier full
    last_bar = min(entry_idx + time_barrier, n_bars - 1)
    return _exit(direction, entry_price, close[last_bar], time_barrier, "TIMEOUT",
                 fees_pct, trail_level=target_idx)


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
    p.add_argument("--time-to-h1", type=int, default=8,
                   help="Bars before conditional time-stop if H1/L1 not reached (default: 8)")
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
    logger.info("Trailing config: time_to_h1=%d bars  time_barrier=%d bars  fees=%.3f%% one-way",
                args.time_to_h1, args.time_barrier, args.fees_pct)

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
            res = simulate_trailing(
                direction, entry_idx, entry_price,
                high, low, close, piv,
                args.time_to_h1, args.time_barrier, args.fees_pct,
            )
            res["score"] = float(pred["scores"][k_idx])
            res["direction"] = direction
            res["timestamp"] = pred["timestamp"][k_idx]
            res["entry_price"] = entry_price
            res["y_true"] = int(pred["y_true"][k_idx]) if "y_true" in pred else -1
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

    logger.info("Trail level reached distribution:")
    for level, count in sorted(main_metrics.get("trail_level_distribution", {}).items()):
        labels = {0: "no H1", 1: "H1 reached", 2: "H2 reached",
                  3: "H3 reached", 4: "H4 final TP"}
        logger.info("  level %d (%s) : %d (%.1f%%)",
                    level, labels.get(level, "?"), count,
                    100 * count / max(main_metrics["n_trades"], 1))

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
            "time_to_h1": args.time_to_h1,
            "time_barrier": args.time_barrier,
            "fees_pct": args.fees_pct,
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
