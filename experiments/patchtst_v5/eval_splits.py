"""
eval_splits.py — Évaluer le modèle entraîné sur tous les splits (train, val, test).

Diagnostic critique : si train >> val/test → distribution shift / overfitting
                      si train ≈ val ≈ test ≈ baseline → pas de signal appris

Usage:
    python -m experiments.patchtst_v5.eval_splits \\
        --checkpoint models/patchtst_v5_rr2_indicators/best_model.pth \\
        --data-dir data/patchtst_v5_rr2/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score

from .model import PatchTSTClassifier

logger = logging.getLogger("patchtst_v5.eval_splits")
TOP_K_PERCENTS = [1, 2, 5, 10, 25]


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    metrics: dict = {}
    preds = (scores >= 0.5).astype("int8")
    metrics["accuracy"] = float((preds == y_true).mean())
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        metrics["pr_auc"] = float(average_precision_score(y_true, scores))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    n = len(scores)
    sorted_idx = np.argsort(-scores)
    for k in TOP_K_PERCENTS:
        n_top = max(1, int(n * k / 100))
        metrics[f"p_top_{k}pct"] = float(y_true[sorted_idx[:n_top]].mean())
    metrics["baseline_class"] = float(y_true.mean())
    metrics["score_mean"] = float(scores.mean())
    metrics["score_std"] = float(scores.std())
    metrics["score_min"] = float(scores.min())
    metrics["score_max"] = float(scores.max())
    return metrics


def evaluate_split(
    model: nn.Module,
    npz_path: Path,
    device: torch.device,
    batch_size: int = 256,
) -> dict:
    data = np.load(npz_path, allow_pickle=False)
    X = torch.from_numpy(data["X"].astype("float32"))
    y = data["y"].astype("int8")
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False, num_workers=0)
    all_scores: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
    scores = np.concatenate(all_scores)
    return compute_metrics(y, scores)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to best_model.pth from train.py")
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Directory with train.npz, val.npz, test.npz")
    p.add_argument("--patch-len", type=int, default=12)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--no-revin", action="store_true")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else (args.device if args.device != "auto" else "cpu"))
    logger.info("Device: %s", device)

    # Detect dimensions from train.npz
    train_data = np.load(args.data_dir / "train.npz", allow_pickle=False)
    n_train, seq_len, n_channels = train_data["X"].shape
    logger.info("Detected: seq_len=%d, n_channels=%d", seq_len, n_channels)

    # Build model with same architecture
    model = PatchTSTClassifier(
        n_channels=n_channels, seq_len=seq_len,
        patch_len=args.patch_len, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout, use_revin=not args.no_revin,
    ).to(device)

    # Load checkpoint
    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Checkpoint epoch=%d val_auc=%.4f", ckpt.get("epoch", -1),
                ckpt.get("val_auc", float("nan")))

    # Evaluate each split
    splits = ["train", "val", "test"]
    results = {}
    for split in splits:
        npz = args.data_dir / f"{split}.npz"
        if not npz.exists():
            logger.warning("Missing %s, skipping", npz)
            continue
        logger.info("Evaluating %s ...", split)
        results[split] = evaluate_split(model, npz, device)

    # Comparative table
    logger.info("=" * 110)
    logger.info("DIAGNOSTIC — METRICS PER SPLIT")
    logger.info("=" * 110)
    logger.info(
        "%-7s %8s %10s %10s %10s %10s %10s %10s %10s %12s",
        "Split", "n", "Class=1", "Acc", "ROC AUC", "PR AUC",
        "p@1%", "p@5%", "p@10%", "Score range",
    )
    logger.info("-" * 110)
    for split in splits:
        if split not in results:
            continue
        m = results[split]
        # Need to reload n
        n = np.load(args.data_dir / f"{split}.npz", allow_pickle=False)["y"].shape[0]
        logger.info(
            "%-7s %8d %10.3f %10.3f %10.4f %10.4f %10.3f %10.3f %10.3f   [%.2f, %.2f]",
            split.upper(), n, m["baseline_class"], m["accuracy"],
            m["roc_auc"], m["pr_auc"],
            m["p_top_1pct"], m["p_top_5pct"], m["p_top_10pct"],
            m["score_min"], m["score_max"],
        )
    logger.info("=" * 110)

    # Diagnostic interprétation
    if "train" in results and "test" in results:
        train_auc = results["train"]["roc_auc"]
        test_auc = results["test"]["roc_auc"]
        train_p1 = results["train"]["p_top_1pct"]
        test_p1 = results["test"]["p_top_1pct"]
        train_baseline = results["train"]["baseline_class"]
        test_baseline = results["test"]["baseline_class"]

        logger.info("INTERPRETATION:")
        gap_auc = train_auc - test_auc
        gap_p1 = train_p1 - test_p1
        logger.info("  Train AUC %.4f vs Test AUC %.4f → gap = %+.4f", train_auc, test_auc, gap_auc)
        logger.info("  Train p@1%% %.3f vs Test p@1%% %.3f → gap = %+.3f", train_p1, test_p1, gap_p1)
        logger.info("  Train baseline %.3f vs Test baseline %.3f → drift = %+.3f",
                    train_baseline, test_baseline, train_baseline - test_baseline)

        if train_auc - 0.5 < 0.02 and test_auc - 0.5 < 0.02:
            logger.info("  ⚠️  AUCUN SIGNAL APPRIS (train AUC ≈ test AUC ≈ 0.5)")
            logger.info("      → Le modèle n'apprend rien, même sur train. Pas un problème de généralisation.")
        elif gap_auc > 0.10:
            logger.info("  ⚠️  OVERFITTING / DISTRIBUTION SHIFT")
            logger.info("      → Train AUC bien > Test AUC : le modèle a appris du signal qui ne généralise pas.")
            logger.info("      → Possibilité : régime de marché test ≠ train, ou overfit pur.")
        elif gap_auc > 0.05:
            logger.info("  ⚠️  Léger overfitting (gap %.3f)", gap_auc)
        else:
            logger.info("  ✓  Train ≈ Test : pas d'overfitting majeur, signal stable mais faible.")
    logger.info("=" * 110)
    return 0


if __name__ == "__main__":
    sys.exit(main())
