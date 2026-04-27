"""
analyze_pivot_distances.py — Quantify Camarilla pivot distances vs entry price.

For each event in events_btc.parquet, computes the distance from entry close
to each of the 8 Camarilla pivots (H1-H4 above, L1-L4 below) in % of the
entry price. Aggregates statistics across all events to understand:

  1. Average distance per pivot level (raw, direction-agnostic)
  2. Per-trade economics by sl_level (TP vs SL distance, R/R, breakeven WR)
  3. Optional cross-tabulation by event direction

Useful to compare distances against fees (~0.04% round-trip) and to design
trailing/scaling targets for the trading method (Layer 2).

Usage:
    python -m experiments.patchtst_v5.analyze_pivot_distances \\
        --features data/patchtst_v5/features_btc.parquet \\
        --events data/patchtst_v5/events_btc.parquet \\
        --output data/patchtst_v5/pivot_distances_report.json
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

from .pivot_labeler_levels import compute_camarilla_5min, find_neighbor_levels

logger = logging.getLogger("patchtst_v5.analyze_pivot_distances")


PIVOT_COLS_ABOVE = ["h1", "h2", "h3", "h4"]
PIVOT_COLS_BELOW = ["l1", "l2", "l3", "l4"]


def percentiles_dict(arr: np.ndarray) -> dict:
    """Return mean, std, min, q25, median, q75, max of a 1D array, NaN-safe."""
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {k: float("nan") for k in
                ("n", "mean", "std", "min", "q25", "median", "q75", "max")}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "max": float(arr.max()),
    }


def compute_distances(entry: np.ndarray, pivots: np.ndarray) -> dict:
    """
    Returns absolute % distance from entry to each pivot, signed: positive
    above entry, negative below.

    entry: (n_events,) close at signal
    pivots: (n_events, 8) ordered as [h1, h2, h3, h4, l1, l2, l3, l4]
    """
    abs_dist_pct = (pivots - entry[:, None]) / entry[:, None] * 100.0  # (n, 8)
    return abs_dist_pct


def report_per_level(dist_pct: np.ndarray, level_names: list[str]) -> dict:
    """For each of the 8 pivot levels, compute distance stats."""
    out = {}
    for i, name in enumerate(level_names):
        out[name] = percentiles_dict(dist_pct[:, i])
    return out


def report_per_trade_economics(features: pd.DataFrame, events: pd.DataFrame,
                                feature_idx: np.ndarray, entry: np.ndarray,
                                direction: np.ndarray) -> dict:
    """For each (direction, sl_level), compute mean TP/SL distance and R/R."""
    levels_at_event = features[
        ["h1", "h2", "h3", "h4", "l1", "l2", "l3", "l4"]
    ].iloc[feature_idx].values  # (n, 8)

    # For each event, find the 4 levels (immediate above/below + nth above/below)
    n = len(events)
    out = {}

    for sl_level in (2, 3, 4):
        for label in ("LONG", "SHORT"):
            tp_dist = np.full(n, np.nan, dtype="float64")
            sl_dist = np.full(n, np.nan, dtype="float64")

            target_dir = 1 if label == "LONG" else -1
            mask = direction == target_dir
            indices = np.where(mask)[0]

            for idx in indices:
                above, below, beyond_above, beyond_below = find_neighbor_levels(
                    entry[idx], levels_at_event[idx], n_beyond=sl_level
                )
                if label == "LONG":
                    tp = above
                    sl = beyond_below
                else:  # SHORT
                    tp = below
                    sl = beyond_above
                if not np.isnan(tp):
                    tp_dist[idx] = abs(tp - entry[idx]) / entry[idx] * 100.0
                if not np.isnan(sl):
                    sl_dist[idx] = abs(sl - entry[idx]) / entry[idx] * 100.0

            tp_stats = percentiles_dict(tp_dist[mask])
            sl_stats = percentiles_dict(sl_dist[mask])
            valid = ~np.isnan(tp_dist) & ~np.isnan(sl_dist)
            rr = tp_dist[valid] / sl_dist[valid]
            rr_stats = percentiles_dict(rr)

            mean_tp = tp_stats["mean"]
            mean_sl = sl_stats["mean"]
            if mean_tp + mean_sl > 0:
                breakeven_wr = mean_sl / (mean_tp + mean_sl) * 100.0
            else:
                breakeven_wr = float("nan")

            key = f"{label}_sl{sl_level}"
            out[key] = {
                "tp_distance_pct": tp_stats,
                "sl_distance_pct": sl_stats,
                "rr_ratio": rr_stats,
                "breakeven_wr_pct": breakeven_wr,
                "n_events_valid": int(valid.sum()),
                "n_events_skipped_no_pivot": int(mask.sum() - valid.sum()),
            }
    return out


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path,
                   default=Path("data/patchtst_v5/features_btc.parquet"))
    p.add_argument("--events", type=Path,
                   default=Path("data/patchtst_v5/events_btc.parquet"))
    p.add_argument("--output", type=Path,
                   default=Path("data/patchtst_v5/pivot_distances_report.json"))
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    logger.info("Loading features: %s", args.features)
    features = pd.read_parquet(args.features, columns=["timestamp", "high", "low", "close"])
    high = features["high"].values.astype("float64")
    low = features["low"].values.astype("float64")
    close = features["close"].values.astype("float64")

    logger.info("Computing Camarilla pivots ...")
    pivot_levels = compute_camarilla_5min(features["timestamp"], high, low, close)
    features = pd.concat([features.reset_index(drop=True), pivot_levels], axis=1)

    logger.info("Loading events: %s", args.events)
    events = pd.read_parquet(args.events)
    n_events = len(events)
    logger.info("Events: %d", n_events)

    feature_idx = events["feature_idx"].values.astype("int64")
    entry = events["close"].values.astype("float64")
    direction = events["direction"].values.astype("int8")

    # 8 distance signed (positive above entry, negative below)
    pivots_at_event = features[
        ["h1", "h2", "h3", "h4", "l1", "l2", "l3", "l4"]
    ].iloc[feature_idx].values
    dist_pct = compute_distances(entry, pivots_at_event)

    # Per-level report (raw distances)
    per_level = report_per_level(
        dist_pct,
        ["H1", "H2", "H3", "H4", "L1", "L2", "L3", "L4"],
    )

    # Per-trade economics report
    per_trade = report_per_trade_economics(
        features, events, feature_idx, entry, direction
    )

    # Console output
    logger.info("=" * 110)
    logger.info("DISTANCES PIVOT CAMARILLA (en %% du prix d'entry)")
    logger.info("=" * 110)
    logger.info("%-6s | %6s | %7s | %7s | %7s | %7s | %7s | %7s | %7s",
                "Level", "n", "mean", "std", "min", "q25", "median", "q75", "max")
    logger.info("-" * 110)
    for name, stats in per_level.items():
        logger.info("%-6s | %6d | %+7.4f | %7.4f | %+7.4f | %+7.4f | %+7.4f | %+7.4f | %+7.4f",
                    name, stats["n"], stats["mean"], stats["std"],
                    stats["min"], stats["q25"], stats["median"], stats["q75"], stats["max"])
    logger.info("=" * 110)

    logger.info("")
    logger.info("=" * 110)
    logger.info("ECONOMIE PAR TRADE selon (direction, sl_level)")
    logger.info("=" * 110)
    logger.info(
        "%-12s | %5s | %8s | %8s | %6s | %6s | %6s",
        "Trade", "n", "TP %", "SL %", "RR", "BE WR%", "Skip"
    )
    logger.info("-" * 110)
    for key in sorted(per_trade.keys()):
        d = per_trade[key]
        logger.info(
            "%-12s | %5d | %+8.4f | %+8.4f | %6.3f | %6.2f | %6d",
            key,
            d["n_events_valid"],
            d["tp_distance_pct"]["mean"],
            d["sl_distance_pct"]["mean"],
            d["rr_ratio"]["mean"],
            d["breakeven_wr_pct"],
            d["n_events_skipped_no_pivot"],
        )
    logger.info("=" * 110)

    # Save full JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "n_events_total": int(n_events),
        "per_level_distance_pct": per_level,
        "per_trade_economics_by_sl_level": per_trade,
    }
    args.output.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Full JSON saved: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
