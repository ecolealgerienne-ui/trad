"""
predict_ensemble.py — Average predictions from multiple XGBoost ensemble members.

Loads all xgboost_model.json found under <ensemble-dir>/seed_*/, runs predict
on train/val/test, averages the scores, saves predictions_{split}.npz that
backtest_realistic.py can consume directly.

Configuration (feature_mode, agg_window, multi_agg_windows, direction_filter)
is read from the FIRST model's stored attrs and applied uniformly. All members
must share the same training config.

Usage:
    python -m experiments.patchtst_v5.predict_ensemble \\
        --ensemble-dir models/patchtst_v5_pivot_buf05_xgb_short_multi_pushed_ensemble/ \\
        --train data/patchtst_v5_pivot_buf05/train.npz \\
        --val   data/patchtst_v5_pivot_buf05/val.npz \\
        --test  data/patchtst_v5_pivot_buf05/test.npz \\
        --output-dir models/patchtst_v5_pivot_buf05_xgb_short_multi_pushed_ensemble/

Then:
    python -m experiments.patchtst_v5.backtest_realistic \\
        --predictions <output-dir>/predictions_test.npz \\
        --output-dir <output-dir>/backtest_test/
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
    parse_multi_windows, DEFAULT_MULTI_AGG_WINDOWS, compute_metrics,
)

logger = logging.getLogger("patchtst_v5.predict_ensemble")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ensemble-dir", type=Path, required=True,
                   help="Parent dir containing seed_*/xgboost_model.json")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--metadata", type=Path, default=None)
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover ensemble members
    model_paths = sorted(args.ensemble_dir.glob("seed_*/xgboost_model.json"))
    if not model_paths:
        raise SystemExit(f"No models found in {args.ensemble_dir}/seed_*/")
    logger.info("Found %d ensemble members:", len(model_paths))
    for mp in model_paths:
        logger.info("  - %s", mp)

    # Load all boosters and read config from FIRST one
    boosters: list[xgb.Booster] = []
    end_iters: list[int] = []
    for mp in model_paths:
        b = xgb.Booster()
        b.load_model(str(mp))
        try:
            ei = b.best_iteration + 1
        except AttributeError:
            ei = b.num_boosted_rounds()
        boosters.append(b)
        end_iters.append(ei)

    # Read shared config from first booster
    b0 = boosters[0]
    feature_mode = b0.attr("feature_mode") or "last-plus-aggs"
    agg_window = int(b0.attr("agg_window") or 24)
    multi_windows_spec = b0.attr("multi_agg_windows") or \
        ",".join(str(w) for w in DEFAULT_MULTI_AGG_WINDOWS)
    multi_windows = parse_multi_windows(multi_windows_spec)
    direction_filter = b0.attr("direction_filter") or "both"
    n_features_stored = b0.attr("n_features")

    logger.info("Ensemble config (from first model): feature_mode=%s agg_window=%d "
                "multi_windows=%s direction_filter=%s",
                feature_mode, agg_window, multi_windows, direction_filter)

    # Sanity: verify all members share the same config
    for i, b in enumerate(boosters[1:], 1):
        for attr_name in ("feature_mode", "agg_window", "multi_agg_windows", "direction_filter", "n_features"):
            v0 = b0.attr(attr_name)
            vi = b.attr(attr_name)
            if v0 != vi:
                raise SystemExit(
                    f"Ensemble member #{i} has {attr_name}={vi!r} but first member has {v0!r}. "
                    f"All members must share the same training config."
                )

    # Load metadata for channels/seq_len
    metadata_path = args.metadata or (args.train.parent / "dataset_metadata.json")
    metadata = json.loads(metadata_path.read_text())
    channels = metadata["channels"]
    seq_len = int(metadata.get("window", 96))

    fnames = feature_names(channels, feature_mode, seq_len, multi_windows)
    if n_features_stored is not None and int(n_features_stored) != len(fnames):
        raise RuntimeError(
            f"Feature count mismatch: models trained with {n_features_stored} features but "
            f"current pipeline produces {len(fnames)}."
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

        # Predict with each member, average
        scores_per_member = []
        for b, ei in zip(boosters, end_iters):
            scores_per_member.append(b.predict(d, iteration_range=(0, ei)))
        scores_arr = np.stack(scores_per_member, axis=0)   # (n_members, n_events)
        avg_scores = scores_arr.mean(axis=0).astype("float32")
        std_scores = scores_arr.std(axis=0).astype("float32")

        # Quick stats per member vs ensemble (for logging only)
        if len(np.unique(data["y"])) > 1:
            ens_metrics = compute_metrics(data["y"], avg_scores)
            per_member_top1 = []
            for s in scores_per_member:
                m = compute_metrics(data["y"], s)
                per_member_top1.append(m.get("precision_top_1pct", float("nan")))
            logger.info(
                "%s | ensemble AUC=%.4f top1%%=%.4f | members top1%% mean=%.4f std=%.4f range=[%.4f, %.4f]",
                split_name.upper(),
                ens_metrics.get("roc_auc", float("nan")),
                ens_metrics.get("precision_top_1pct", float("nan")),
                float(np.mean(per_member_top1)),
                float(np.std(per_member_top1)),
                float(np.min(per_member_top1)),
                float(np.max(per_member_top1)),
            )

        save_kwargs = dict(
            scores=avg_scores,
            scores_std=std_scores,                        # variance per event across members
            y_true=data["y"].astype("int8"),
            direction=data["direction"],
            timestamp=data["timestamp"],
            pnl_after_fees_pct=data["pnl_after_fees_pct"],
        )
        if "feature_idx" in data:
            save_kwargs["feature_idx"] = data["feature_idx"]
        out_path = args.output_dir / f"predictions_{split_name}.npz"
        np.savez_compressed(out_path, **save_kwargs)
        logger.info("Saved %s: %s (n=%d, %d-model ensemble, scores [%.3f, %.3f])",
                    split_name, out_path, len(avg_scores), len(boosters),
                    avg_scores.min(), avg_scores.max())

    return 0


if __name__ == "__main__":
    sys.exit(main())
