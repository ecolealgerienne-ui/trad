"""
train_xgboost.py — Baseline XGBoost sur les features à l'event (et agrégats récents).

Hypothèse: PatchTST overfit du bruit sur fenêtre 96 timesteps. Un modèle plus
simple sur les features event-time uniquement devrait mieux capturer le signal
modeste détecté par le diagnostic de séparabilité.

Features modes:
  --feature-mode last-only       : 19 features (event-time uniquement)
  --feature-mode last-plus-aggs  : 19 × 4 = 76 features (last + mean/std/first sur 24 bars)

Usage:
    python -m experiments.patchtst_v5.train_xgboost \\
        --train data/patchtst_v5_pivot_buf05/train.npz \\
        --val data/patchtst_v5_pivot_buf05/val.npz \\
        --test data/patchtst_v5_pivot_buf05/test.npz \\
        --output-dir models/patchtst_v5_pivot_buf05_xgb/ \\
        --feature-mode last-plus-aggs
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
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger("patchtst_v5.train_xgboost")
TOP_K_PERCENTS = [1, 2, 5, 10, 25]


def load_split(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=False)
    return {
        "X": data["X"].astype("float32"),
        "y": data["y"].astype("int8"),
        "direction": data["direction"],
        "timestamp": data["timestamp"],
        "pnl_after_fees_pct": data["pnl_after_fees_pct"],
    }


def build_features(X: np.ndarray, mode: str, agg_window: int = 24) -> np.ndarray:
    """X shape (n, T, C). Retourne (n, F) features 2D."""
    n, T, C = X.shape
    if mode == "last-only":
        return X[:, -1, :]
    elif mode == "last-plus-aggs":
        sub = X[:, -agg_window:, :]                          # (n, 24, C)
        last = X[:, -1, :]                                   # (n, C)
        mean = sub.mean(axis=1)                              # (n, C)
        std = sub.std(axis=1)                                # (n, C)
        first = X[:, -agg_window, :]                         # (n, C)
        return np.concatenate([last, mean, std, first], axis=1)  # (n, 4C)
    raise ValueError(f"Unknown mode: {mode}")


def feature_names(channels: list[str], mode: str) -> list[str]:
    if mode == "last-only":
        return [f"{c}_last" for c in channels]
    return [f"{c}_last" for c in channels] + \
           [f"{c}_mean24" for c in channels] + \
           [f"{c}_std24" for c in channels] + \
           [f"{c}_first24" for c in channels]


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    metrics = {"accuracy": float((scores >= 0.5).astype(int).__eq__(y_true).mean())}
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        metrics["pr_auc"] = float(average_precision_score(y_true, scores))
    sorted_idx = np.argsort(-scores)
    n = len(scores)
    for k in TOP_K_PERCENTS:
        n_top = max(1, int(n * k / 100))
        metrics[f"precision_top_{k}pct"] = float(y_true[sorted_idx[:n_top]].mean())
    return metrics


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--metadata", type=Path, default=None,
                   help="dataset_metadata.json (default: <train>/.../dataset_metadata.json)")
    p.add_argument("--feature-mode", type=str, default="last-plus-aggs",
                   choices=["last-only", "last-plus-aggs"])
    p.add_argument("--agg-window", type=int, default=24)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--early-stopping-rounds", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
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
    logger.info("Channels: %d", len(channels))

    train = load_split(args.train)
    val = load_split(args.val)
    test = load_split(args.test)
    logger.info("Train: %d (Class1=%.1f%%) | Val: %d | Test: %d",
                len(train["y"]), 100 * train["y"].mean(), len(val["y"]), len(test["y"]))

    X_train = build_features(train["X"], args.feature_mode, args.agg_window)
    X_val = build_features(val["X"], args.feature_mode, args.agg_window)
    X_test = build_features(test["X"], args.feature_mode, args.agg_window)
    fnames = feature_names(channels, args.feature_mode)
    logger.info("Feature mode: %s → %d features", args.feature_mode, X_train.shape[1])

    pos_train = train["y"].mean()
    scale_pos_weight = (1 - pos_train) / max(pos_train, 1e-6)
    logger.info("Class1 ratio train: %.3f → scale_pos_weight = %.3f", pos_train, scale_pos_weight)

    dtrain = xgb.DMatrix(X_train, label=train["y"], feature_names=fnames)
    dval = xgb.DMatrix(X_val, label=val["y"], feature_names=fnames)
    dtest = xgb.DMatrix(X_test, label=test["y"], feature_names=fnames)

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss"],
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "scale_pos_weight": scale_pos_weight,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "seed": args.seed,
        "verbosity": 1,
        "tree_method": "hist",
    }

    logger.info("Training XGBoost (n_estimators=%d, max_depth=%d, lr=%.3f) ...",
                args.n_estimators, args.max_depth, args.learning_rate)
    booster = xgb.train(
        params, dtrain,
        num_boost_round=args.n_estimators,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=20,
    )
    best_iter = booster.best_iteration
    logger.info("Best iteration: %d (val AUC = %.4f)", best_iter, booster.best_score)

    # Predict on all splits
    train_scores = booster.predict(dtrain, iteration_range=(0, best_iter + 1))
    val_scores = booster.predict(dval, iteration_range=(0, best_iter + 1))
    test_scores = booster.predict(dtest, iteration_range=(0, best_iter + 1))

    train_m = compute_metrics(train["y"], train_scores)
    val_m = compute_metrics(val["y"], val_scores)
    test_m = compute_metrics(test["y"], test_scores)

    logger.info("=" * 110)
    logger.info("RESULTS")
    logger.info("=" * 110)
    logger.info("Split | n      | Class1 |   AUC | PR AUC | p@1%% | p@5%% | p@10%% | p@25%%")
    for split_name, y, m in [("train", train["y"], train_m),
                              ("val", val["y"], val_m),
                              ("test", test["y"], test_m)]:
        logger.info("%-5s | %6d | %.3f  | %.3f | %.4f  | %.3f | %.3f | %.4f | %.4f",
                    split_name.upper(), len(y), y.mean(),
                    m["roc_auc"], m["pr_auc"],
                    m["precision_top_1pct"], m["precision_top_5pct"],
                    m["precision_top_10pct"], m["precision_top_25pct"])
    logger.info("=" * 110)

    # Feature importance
    importance = booster.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])[:15]
    logger.info("Top 15 features par gain:")
    for name, gain in sorted_imp:
        logger.info("  %-40s %.2f", name, gain)
    logger.info("=" * 110)

    # Save
    booster.save_model(str(args.output_dir / "xgboost_model.json"))

    np.savez_compressed(
        args.output_dir / "predictions_test.npz",
        scores=test_scores.astype("float32"),
        y_true=test["y"].astype("int8"),
        direction=test["direction"],
        timestamp=test["timestamp"],
        pnl_after_fees_pct=test["pnl_after_fees_pct"],
    )
    logger.info("Predictions saved: %s", args.output_dir / "predictions_test.npz")

    report = {
        "feature_mode": args.feature_mode,
        "n_features": X_train.shape[1],
        "best_iteration": best_iter,
        "best_val_auc": float(booster.best_score),
        "train_metrics": train_m,
        "val_metrics": val_m,
        "test_metrics": test_m,
        "feature_importance_top15": dict(sorted_imp),
        "config": {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
    }
    (args.output_dir / "test_report.json").write_text(json.dumps(report, indent=2, default=str))
    logger.info("Report saved: %s", args.output_dir / "test_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
