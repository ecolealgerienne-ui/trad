"""
train_ensemble.py — Train N XGBoost models with different seeds for variance reduction.

Each model is saved in <output-dir>/seed_<S>/xgboost_model.json.
Use predict_ensemble.py to generate averaged predictions across all members.

All `train_xgboost.py` CLI args are forwarded EXCEPT `--seed` (replaced by `--seeds`)
and `--output-dir` (interpreted as PARENT dir; each seed gets its own subdir).

Usage:
    python -m experiments.patchtst_v5.train_ensemble \\
        --train data/patchtst_v5_pivot_buf05/train.npz \\
        --val   data/patchtst_v5_pivot_buf05/val.npz \\
        --test  data/patchtst_v5_pivot_buf05/test.npz \\
        --output-dir models/patchtst_v5_pivot_buf05_xgb_short_multi_pushed_ensemble/ \\
        --seeds 42,7,13,100,999 \\
        --feature-mode last-plus-multi-aggs \\
        --direction-filter short \\
        --max-depth 10 --learning-rate 0.03 --n-estimators 3000 \\
        --min-child-weight 1 --subsample 1.0 --colsample-bytree 1.0 \\
        --reg-lambda 0.0 --reg-alpha 0.0 --no-early-stopping
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

from .train_xgboost import main as train_main

logger = logging.getLogger("patchtst_v5.train_ensemble")


def parse_args(argv: Iterable[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                add_help=False)
    p.add_argument("--seeds", type=str, default="42,7,13,100,999",
                   help="Comma-separated seeds (default: 42,7,13,100,999)")
    p.add_argument("-h", "--help", action="store_true")
    args, remaining = p.parse_known_args(argv)
    if args.help:
        p.print_help()
        # Forward to show train_xgboost help too
        print("\n--- train_xgboost.py CLI args (all forwarded) ---")
        train_main(["--help"])
        sys.exit(0)
    return args, remaining


def main(argv: Iterable[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args, remaining = parse_args(argv)

    logging.basicConfig(level="INFO",
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")

    # Extract --output-dir from remaining (will become PARENT dir)
    parent_output: Path | None = None
    for i, arg in enumerate(remaining):
        if arg == "--output-dir" and i + 1 < len(remaining):
            parent_output = Path(remaining[i + 1])
            break
    if parent_output is None:
        raise SystemExit("--output-dir is required")
    parent_output.mkdir(parents=True, exist_ok=True)

    logger.info("Ensemble: %d seeds → %s", len(seeds), parent_output)
    logger.info("Seeds: %s", seeds)

    for seed in seeds:
        seed_dir = parent_output / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Rebuild sub_argv: replace --output-dir, drop any --seed, then append our seed
        sub_argv: list[str] = []
        i = 0
        while i < len(remaining):
            if remaining[i] == "--output-dir":
                sub_argv.extend(["--output-dir", str(seed_dir)])
                i += 2
            elif remaining[i] == "--seed":
                # Drop user's --seed if any (we override)
                i += 2
            else:
                sub_argv.append(remaining[i])
                i += 1
        sub_argv.extend(["--seed", str(seed)])

        logger.info("=" * 80)
        logger.info("Training seed=%d → %s", seed, seed_dir)
        logger.info("=" * 80)
        rc = train_main(sub_argv)
        if rc != 0:
            logger.error("Training failed for seed=%d (rc=%d)", seed, rc)
            return rc

    logger.info("=" * 80)
    logger.info("Ensemble training done: %d members in %s", len(seeds), parent_output)
    logger.info("Next: python -m experiments.patchtst_v5.predict_ensemble "
                "--ensemble-dir %s --train ... --val ... --test ... --output-dir ...",
                parent_output)
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
