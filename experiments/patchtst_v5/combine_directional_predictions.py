"""
combine_directional_predictions.py — Combine LONG-only + SHORT-only ensemble
predictions into a single "dual directional portfolio" predictions file.

For each split (train, val, test), reads predictions from two separate ensemble
directories and concatenates them into a single predictions_{split}.npz that
backtest_realistic.py can consume directly.

Use this when you train one ensemble on LONG-only events and another on
SHORT-only events, and want to evaluate the COMBINED portfolio — each event
keeps its own model's score, calibrated within its direction.

Score normalization (`--rank-normalize`):
  Without it, raw scores from the two models are concatenated directly. Top-K
  ranking will then favor whichever model produces higher-magnitude scores.
  With `--rank-normalize`, each model's scores are converted to per-direction
  percentile ranks [0, 1] before concatenation, balancing the two portfolios.

Usage:
    python -m experiments.patchtst_v5.combine_directional_predictions \\
        --long-dir  models/patchtst_v5_pivot_buf05_xgb_long_multi_ensemble_bagging/ \\
        --short-dir models/patchtst_v5_pivot_buf05_xgb_short_multi_ensemble_bagging/ \\
        --output-dir models/patchtst_v5_pivot_buf05_xgb_dual_directional/

Then:
    python -m experiments.patchtst_v5.backtest_realistic \\
        --predictions <output-dir>/predictions_test.npz \\
        --output-dir  <output-dir>/backtest_test/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

logger = logging.getLogger("patchtst_v5.combine_directional")


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    """Convert scores to per-array percentile ranks in [0, 1]."""
    n = len(scores)
    if n == 0:
        return scores
    order = np.argsort(scores)
    ranks = np.empty(n, dtype="float32")
    ranks[order] = np.arange(n, dtype="float32") / max(n - 1, 1)
    return ranks


def combine_split(long_npz: Path, short_npz: Path, out_npz: Path,
                   rank_norm: bool) -> None:
    long_data = np.load(long_npz, allow_pickle=False)
    short_data = np.load(short_npz, allow_pickle=False)

    # Sanity: expect direction matches (long: +1 only, short: -1 only)
    if (long_data["direction"] != 1).any():
        logger.warning("LONG file %s contains non-LONG events", long_npz)
    if (short_data["direction"] != -1).any():
        logger.warning("SHORT file %s contains non-SHORT events", short_npz)

    long_scores = long_data["scores"].astype("float32")
    short_scores = short_data["scores"].astype("float32")

    if rank_norm:
        long_scores_used = rank_normalize(long_scores)
        short_scores_used = rank_normalize(short_scores)
        logger.info("  rank-normalized scores per direction")
    else:
        long_scores_used = long_scores
        short_scores_used = short_scores

    # Common keys between the two files
    common_keys = set(long_data.files) & set(short_data.files)

    combined: dict = {}
    for key in common_keys:
        if key == "scores":
            combined["scores"] = np.concatenate([long_scores_used, short_scores_used])
        elif key == "scores_std":
            combined["scores_std"] = np.concatenate([
                long_data["scores_std"].astype("float32"),
                short_data["scores_std"].astype("float32"),
            ])
        else:
            combined[key] = np.concatenate([long_data[key], short_data[key]])

    # Sort all arrays by timestamp to restore chronological order
    if "timestamp" in combined:
        order = np.argsort(combined["timestamp"])
        combined = {k: v[order] for k, v in combined.items()}

    # Optional: keep raw scores too if rank_norm was applied (for debugging)
    if rank_norm:
        raw_scores = np.concatenate([long_scores, short_scores])
        if "timestamp" in combined:
            raw_scores = raw_scores[order]
        combined["scores_raw"] = raw_scores

    np.savez_compressed(out_npz, **combined)

    n_long = len(long_data["scores"])
    n_short = len(short_data["scores"])
    logger.info("  saved %s: n=%d (%d LONG + %d SHORT), scores [%.3f, %.3f]",
                out_npz, n_long + n_short, n_long, n_short,
                combined["scores"].min(), combined["scores"].max())


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--long-dir", type=Path, required=True,
                   help="Dir containing predictions_{train,val,test}.npz from LONG ensemble")
    p.add_argument("--short-dir", type=Path, required=True,
                   help="Dir containing predictions_{train,val,test}.npz from SHORT ensemble")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--rank-normalize", action="store_true",
                   help="Convert each direction's scores to percentile ranks [0,1] "
                        "before concatenation (balances the two portfolios)")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("LONG dir : %s", args.long_dir)
    logger.info("SHORT dir: %s", args.short_dir)
    logger.info("Rank normalize: %s", args.rank_normalize)

    for split in ("train", "val", "test"):
        long_npz = args.long_dir / f"predictions_{split}.npz"
        short_npz = args.short_dir / f"predictions_{split}.npz"
        out_npz = args.output_dir / f"predictions_{split}.npz"
        if not long_npz.exists():
            raise SystemExit(f"Missing LONG predictions: {long_npz}")
        if not short_npz.exists():
            raise SystemExit(f"Missing SHORT predictions: {short_npz}")
        logger.info("Combining %s ...", split)
        combine_split(long_npz, short_npz, out_npz, args.rank_normalize)

    logger.info("Done. Run backtest_realistic on %s/predictions_{train,val,test}.npz",
                args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
