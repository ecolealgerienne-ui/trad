"""
predict_all_splits.py — Génère predictions sur train/val/test depuis modèle XGBoost.

Permet de backtester sur chaque split pour distinguer:
  - Overfit pur: train PnL ≫ val PnL ≈ test PnL
  - Distribution shift: train PnL bon, val moyen, test bas (tendance)
  - Pas de signal: train PnL ≈ val PnL ≈ test PnL ≈ 0

Usage:
    python -m experiments.patchtst_v5.predict_all_splits \\
        --model models/patchtst_v5_pivot_buf05_xgb/xgboost_model.json \\
        --train data/patchtst_v5_pivot_buf05/train.npz \\
        --val data/patchtst_v5_pivot_buf05/val.npz \\
        --test data/patchtst_v5_pivot_buf05/test.npz \\
        --output-dir models/patchtst_v5_pivot_buf05_xgb/ \\
        --feature-mode last-plus-aggs

Puis backtest sur chaque:
    python -m experiments.patchtst_v5.backtest_realistic \\
        --predictions models/.../predictions_train.npz \\
        --output-dir models/.../backtest_train/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import xgboost as xgb

from .train_xgboost import load_split, build_features, feature_names

logger = logging.getLogger("patchtst_v5.predict_all_splits")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--metadata", type=Path, default=None)
    p.add_argument("--feature-mode", type=str, default="last-plus-aggs",
                   choices=["last-only", "last-plus-aggs"])
    p.add_argument("--agg-window", type=int, default=24)
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = args.metadata or (args.train.parent / "dataset_metadata.json")
    metadata = json.loads(metadata_path.read_text())
    channels = metadata["channels"]
    fnames = feature_names(channels, args.feature_mode)

    booster = xgb.Booster()
    booster.load_model(str(args.model))
    logger.info("Loaded model: %s (best iter=%d)", args.model, booster.best_iteration)

    for split_name, npz_path in [("train", args.train), ("val", args.val), ("test", args.test)]:
        data = load_split(npz_path)
        X = build_features(data["X"], args.feature_mode, args.agg_window)
        d = xgb.DMatrix(X, feature_names=fnames)
        scores = booster.predict(d, iteration_range=(0, booster.best_iteration + 1))

        out_path = args.output_dir / f"predictions_{split_name}.npz"
        np.savez_compressed(
            out_path,
            scores=scores.astype("float32"),
            y_true=data["y"].astype("int8"),
            direction=data["direction"],
            timestamp=data["timestamp"],
            pnl_after_fees_pct=data["pnl_after_fees_pct"],
        )
        logger.info("Saved %s: %s (n=%d, scores [%.3f, %.3f])",
                    split_name, out_path, len(scores), scores.min(), scores.max())

    return 0


if __name__ == "__main__":
    sys.exit(main())
