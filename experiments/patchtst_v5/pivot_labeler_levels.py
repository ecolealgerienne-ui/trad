"""
pivot_labeler_levels.py — Triple Barrier avec niveaux pivot Camarilla comme TP/SL.

Pour chaque event:
  - Trouve le niveau pivot Camarilla IMMÉDIATEMENT au-dessus et IMMÉDIATEMENT
    en-dessous de l'entry close.
  - LONG: TP = level_above, SL = level_below
  - SHORT: TP = level_below, SL = level_above
  - Skip events où il n'y a pas de niveau dans la direction (extrêmes hors structure)
  - Time barrier configurable (default 24 bars = 2h)

Distances TP/SL varient naturellement selon position du trade dans la structure.
RR effectif varie par trade.

Usage:
    python -m experiments.patchtst_v5.pivot_labeler_levels \\
        --features data/patchtst_v5/features_btc.parquet \\
        --events data/patchtst_v5/events_btc.parquet \\
        --output data/patchtst_v5/labels_btc_pivot_levels.parquet \\
        --time-barrier 24 --fees-pct 0.02
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger("patchtst_v5.pivot_labeler_levels")

PIVOT_COLS = ["h1", "h2", "h3", "h4", "l1", "l2", "l3", "l4"]


def find_neighbor_levels(entry: float, levels: np.ndarray) -> tuple[float, float]:
    """Trouve niveau immédiatement au-dessus et en-dessous de entry."""
    valid = levels[~np.isnan(levels)]
    above_mask = valid > entry
    below_mask = valid < entry
    above = valid[above_mask].min() if above_mask.any() else np.nan
    below = valid[below_mask].max() if below_mask.any() else np.nan
    return above, below


def label_events(events: pd.DataFrame, features: pd.DataFrame,
                 high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 time_barrier: int, sl_buffer_atr: float, fees_pct: float) -> pd.DataFrame:
    """Triple Barrier avec niveaux Camarilla comme TP/SL dynamiques."""
    n_events = len(events)
    n_bars = len(high)
    direction_arr = events["direction"].values.astype("int8")
    feature_idx_arr = events["feature_idx"].values.astype("int64")
    signal_close = events["close"].values.astype("float64")
    signal_atr = events["atr_14"].values.astype("float64")

    # Niveaux Camarilla au moment de chaque event
    levels_at_event = features[PIVOT_COLS].iloc[feature_idx_arr].values  # (n_events, 8)

    label = np.full(n_events, -1, dtype="int8")
    tp_price = np.full(n_events, np.nan, dtype="float64")
    sl_price = np.full(n_events, np.nan, dtype="float64")
    pnl_net = np.full(n_events, np.nan, dtype="float64")
    exit_bars = np.full(n_events, -1, dtype="int16")
    exit_reason = np.empty(n_events, dtype=object)
    rr_ratio = np.full(n_events, np.nan, dtype="float64")
    skipped_reason = np.empty(n_events, dtype=object)

    n_skipped_no_target = 0
    n_skipped_oof = 0

    for k in range(n_events):
        idx = feature_idx_arr[k]
        end = idx + 1 + time_barrier
        if end > n_bars:
            n_skipped_oof += 1
            skipped_reason[k] = "OUT_OF_DATA"
            continue

        direction = direction_arr[k]
        entry = signal_close[k]
        atr_t = signal_atr[k]

        above, below = find_neighbor_levels(entry, levels_at_event[k])

        # Direction LONG → TP au-dessus, SL en dessous
        # Direction SHORT → TP en-dessous, SL au-dessus
        if direction > 0:
            tp_lvl = above
            sl_lvl = below
        else:
            tp_lvl = below
            sl_lvl = above

        if np.isnan(tp_lvl) or np.isnan(sl_lvl):
            n_skipped_no_target += 1
            skipped_reason[k] = "NO_PIVOT_TARGET"
            continue

        # SL avec buffer optionnel (en éloignement du level pour anti stop hunt)
        if direction > 0:
            sl = sl_lvl - sl_buffer_atr * atr_t
            tp = tp_lvl
        else:
            sl = sl_lvl + sl_buffer_atr * atr_t
            tp = tp_lvl

        tp_price[k] = tp
        sl_price[k] = sl

        # RR effectif (récompense/risque)
        if direction > 0:
            reward = abs(tp - entry)
            risk = abs(entry - sl)
        else:
            reward = abs(entry - tp)
            risk = abs(sl - entry)
        rr_ratio[k] = reward / risk if risk > 1e-12 else np.nan

        # Walk-forward exit detection
        sub_high = high[idx + 1: end]
        sub_low = low[idx + 1: end]
        if direction > 0:
            tp_hit = sub_high >= tp
            sl_hit = sub_low <= sl
        else:
            tp_hit = sub_low <= tp
            sl_hit = sub_high >= sl

        first_tp = int(np.argmax(tp_hit)) if tp_hit.any() else time_barrier
        first_sl = int(np.argmax(sl_hit)) if sl_hit.any() else time_barrier

        if first_tp < first_sl:
            exit_p = tp
            label[k] = 1
            exit_reason[k] = "TP"
            exit_bars[k] = first_tp + 1
        elif first_sl < first_tp:
            exit_p = sl
            label[k] = 0
            exit_reason[k] = "SL"
            exit_bars[k] = first_sl + 1
        elif first_tp == first_sl and first_tp < time_barrier:
            exit_p = sl
            label[k] = 0
            exit_reason[k] = "AMBIGUOUS"
            exit_bars[k] = first_tp + 1
        else:
            last_close = close[end - 1]
            exit_p = last_close
            exit_reason[k] = "TIMEOUT"
            exit_bars[k] = time_barrier
            if direction > 0:
                label[k] = 1 if last_close > entry else 0
            else:
                label[k] = 1 if last_close < entry else 0

        if direction > 0:
            pnl = 100.0 * (exit_p - entry) / entry
        else:
            pnl = 100.0 * (entry - exit_p) / entry
        pnl_net[k] = pnl - 2 * fees_pct

    valid = label != -1
    out = events.copy()
    out["tp_price"] = tp_price.astype("float32")
    out["sl_price"] = sl_price.astype("float32")
    out["rr_effective"] = rr_ratio.astype("float32")
    out["label"] = label
    out["exit_bars"] = exit_bars
    out["exit_reason"] = exit_reason
    out["pnl_after_fees_pct"] = pnl_net.astype("float32")
    out = out.loc[valid].reset_index(drop=True)

    logger.info("Skipped: %d OUT_OF_DATA, %d NO_PIVOT_TARGET (extremes)", n_skipped_oof, n_skipped_no_target)
    return out


def report(labels: pd.DataFrame) -> None:
    n = len(labels)
    n_pos = int((labels["label"] == 1).sum())
    pos_rate = 100 * n_pos / n if n else 0.0

    logger.info("=" * 110)
    logger.info("LABEL SUMMARY (Pivot levels TP/SL)")
    logger.info("=" * 110)
    logger.info("Total labeled events : %d", n)
    logger.info("Class balance        : Label=1 %d (%.1f%%) | Label=0 %d (%.1f%%)",
                n_pos, pos_rate, n - n_pos, 100 - pos_rate)

    rr_stats = labels["rr_effective"].describe()
    logger.info("RR effectif (reward/risk):")
    logger.info("  min=%.2f q25=%.2f median=%.2f mean=%.2f q75=%.2f max=%.2f",
                rr_stats["min"], rr_stats["25%"], rr_stats["50%"],
                rr_stats["mean"], rr_stats["75%"], rr_stats["max"])

    reason_counts = labels["exit_reason"].value_counts()
    logger.info("Exit reasons:")
    for reason, count in reason_counts.items():
        logger.info("  %-12s : %6d (%.1f%%)", reason, count, 100 * count / n)

    pnl = labels["pnl_after_fees_pct"]
    logger.info("PnL net per-trade: mean=%+.4f%% median=%+.4f%% std=%.4f%%",
                pnl.mean(), pnl.median(), pnl.std())
    logger.info("Cumul PnL net    : %+.2f%%", pnl.sum())

    pos_mask = labels["label"] == 1
    neg_mask = labels["label"] == 0
    if pos_mask.any() and neg_mask.any():
        mean_win = labels.loc[pos_mask, "pnl_after_fees_pct"].mean()
        mean_loss = labels.loc[neg_mask, "pnl_after_fees_pct"].mean()
        breakeven = abs(mean_loss) / (mean_win + abs(mean_loss)) if mean_win > 0 else float("nan")
        oracle = labels.loc[pos_mask, "pnl_after_fees_pct"].sum()
        logger.info("Mean win net  : %+.4f%%   Mean loss net : %+.4f%%", mean_win, mean_loss)
        logger.info("Breakeven WR  : %.1f%%   Oracle cumul  : %+.1f%%", breakeven * 100, oracle)

        # Span estimé
        ts = pd.to_datetime(labels["timestamp"])
        span_years = (ts.max() - ts.min()).total_seconds() / (365.25 * 86400)
        logger.info("Span: %.2f years   Oracle annualisé: %+.1f%%/an", span_years, oracle / span_years)
    logger.info("=" * 110)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=Path("data/patchtst_v5/features_btc.parquet"))
    p.add_argument("--events", type=Path, default=Path("data/patchtst_v5/events_btc.parquet"))
    p.add_argument("--output", type=Path, default=Path("data/patchtst_v5/labels_btc_pivot_levels.parquet"))
    p.add_argument("--time-barrier", type=int, default=24, help="Bars max (24 = 2h)")
    p.add_argument("--sl-buffer-atr", type=float, default=0.0,
                   help="Buffer ATR au-delà du niveau pivot pour le SL (0=strict, 0.3=anti stop hunt)")
    p.add_argument("--fees-pct", type=float, default=0.02, help="One-way fee %% (0.02 = maker)")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    logger.info("Loading features: %s", args.features)
    cols = ["high", "low", "close"] + PIVOT_COLS
    features = pd.read_parquet(args.features, columns=cols)
    high = features["high"].values.astype("float64")
    low = features["low"].values.astype("float64")
    close = features["close"].values.astype("float64")

    logger.info("Loading events: %s", args.events)
    events = pd.read_parquet(args.events)
    logger.info("Events: %d", len(events))
    logger.info("Time barrier: %d bars (~%d min)", args.time_barrier, args.time_barrier * 5)
    logger.info("SL buffer: %.2f × ATR au-delà du level", args.sl_buffer_atr)
    logger.info("Fees: %.3f%% one-way (round-trip = %.3f%%)", args.fees_pct, 2 * args.fees_pct)

    labeled = label_events(events, features, high, low, close,
                           args.time_barrier, args.sl_buffer_atr, args.fees_pct)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(args.output, compression="snappy", index=False)
    logger.info("Output: %s (%d events)", args.output, len(labeled))
    report(labeled)
    return 0


if __name__ == "__main__":
    sys.exit(main())
