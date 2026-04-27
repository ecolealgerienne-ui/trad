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

from .train_xgboost import (
    load_split, build_features, feature_names, filter_by_direction,
    parse_multi_windows, DEFAULT_MULTI_AGG_WINDOWS,
)

logger = logging.getLogger("patchtst_v5.predict_all_splits")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--metadata", type=Path, default=None)
    p.add_argument("--feature-mode", type=str, default=None,
                   choices=["last-only", "last-plus-aggs", "last-plus-multi-aggs",
                            "last-plus-multi-aggs-rich"],
                   help="Override model's stored feature_mode (default: read from model attrs)")
    p.add_argument("--agg-window", type=int, default=None,
                   help="Override model's stored agg_window (default: read from model attrs)")
    p.add_argument("--multi-agg-windows", type=str, default=None,
                   help="Override model's stored multi_agg_windows (default: read from model attrs)")
    p.add_argument("--direction-filter", type=str, default=None,
                   choices=["long", "short", "both"],
                   help="Override model's stored direction_filter (default: read from model attrs)")
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
    seq_len = int(metadata.get("window", 96))

    booster = xgb.Booster()
    booster.load_model(str(args.model))
    try:
        end_iter = booster.best_iteration + 1
        logger.info("Loaded model: %s (best iter=%d, using %d trees)",
                    args.model, booster.best_iteration, end_iter)
    except AttributeError:
        end_iter = booster.num_boosted_rounds()
        logger.info("Loaded model: %s (no early stopping, using all %d trees)",
                    args.model, end_iter)

    # Read model-stored training config and reconcile with CLI overrides.
    stored = {
        "feature_mode": booster.attr("feature_mode"),
        "agg_window": booster.attr("agg_window"),
        "multi_agg_windows": booster.attr("multi_agg_windows"),
        "direction_filter": booster.attr("direction_filter"),
        "n_features": booster.attr("n_features"),
    }
    logger.info("Model stored attrs: %s", stored)

    def resolve(cli_value, stored_value, default, name):
        if cli_value is not None and stored_value is not None and str(cli_value) != str(stored_value):
            logger.warning("CLI --%s=%s differs from model stored=%s — using CLI value",
                           name, cli_value, stored_value)
            return cli_value
        if cli_value is not None:
            return cli_value
        if stored_value is not None:
            return stored_value
        logger.warning("No %s in model attrs and no CLI override — falling back to default %s",
                       name, default)
        return default

    feature_mode = resolve(args.feature_mode, stored["feature_mode"], "last-plus-aggs", "feature-mode")
    agg_window = int(resolve(args.agg_window, stored["agg_window"], 24, "agg-window"))
    multi_windows_spec = resolve(args.multi_agg_windows, stored["multi_agg_windows"],
                                  ",".join(str(w) for w in DEFAULT_MULTI_AGG_WINDOWS), "multi-agg-windows")
    multi_windows = parse_multi_windows(multi_windows_spec)
    direction_filter = resolve(args.direction_filter, stored["direction_filter"], "both", "direction-filter")
    logger.info("Resolved config: feature_mode=%s agg_window=%d multi_windows=%s direction_filter=%s",
                feature_mode, agg_window, multi_windows, direction_filter)

    fnames = feature_names(channels, feature_mode, seq_len, multi_windows)
    if stored["n_features"] is not None and int(stored["n_features"]) != len(fnames):
        raise RuntimeError(
            f"Feature count mismatch: model trained with {stored['n_features']} features but "
            f"current pipeline produces {len(fnames)}. Check --feature-mode / --multi-agg-windows / channels."
        )

    for split_name, npz_path in [("train", args.train), ("val", args.val), ("test", args.test)]:
        data = load_split(npz_path)
        if direction_filter != "both":
            n0 = len(data["y"])
            data = filter_by_direction(data, direction_filter)
            logger.info("Direction filter %s on %s: %d → %d",
                        direction_filter, split_name, n0, len(data["y"]))
        X = build_features(data["X"], feature_mode, agg_window, multi_windows)
        d = xgb.DMatrix(X, feature_names=fnames)
        scores = booster.predict(d, iteration_range=(0, end_iter))

        save_kwargs = dict(
            scores=scores.astype("float32"),
            y_true=data["y"].astype("int8"),
            direction=data["direction"],
            timestamp=data["timestamp"],
            pnl_after_fees_pct=data["pnl_after_fees_pct"],
        )
        if "feature_idx" in data:
            save_kwargs["feature_idx"] = data["feature_idx"]
        out_path = args.output_dir / f"predictions_{split_name}.npz"
        np.savez_compressed(out_path, **save_kwargs)
        logger.info("Saved %s: %s (n=%d, scores [%.3f, %.3f])",
                    split_name, out_path, len(scores), scores.min(), scores.max())

    return 0


if __name__ == "__main__":
    sys.exit(main())
