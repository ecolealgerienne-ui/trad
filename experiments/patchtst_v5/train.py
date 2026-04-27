"""
train.py — Entraînement PatchTST v5.0 (étape 6 roadmap).

Charge les NPZ de dataset_builder.py, entraîne PatchTSTClassifier avec early
stopping sur val AUC, sauvegarde best_model.pth + history JSON + predictions
test NPZ pour backtest_realistic.py.

Usage:
    python -m experiments.patchtst_v5.train \\
        --train data/patchtst_v5/train.npz \\
        --val data/patchtst_v5/val.npz \\
        --test data/patchtst_v5/test.npz \\
        --output-dir models/patchtst_v5/

Voir STATUS_v5.0.md et experiments/patchtst_v5/README.md.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score

from .model import PatchTSTClassifier, count_parameters

logger = logging.getLogger("patchtst_v5.train")


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 15
DEFAULT_LR_REDUCE_FACTOR = 0.5
DEFAULT_LR_REDUCE_PATIENCE = 5
TOP_K_PERCENTS = [1, 5, 10, 25]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class SplitData:
    X: torch.Tensor
    y: torch.Tensor
    direction: np.ndarray
    timestamp: np.ndarray
    pnl_after_fees_pct: np.ndarray


def load_split(npz_path: Path) -> SplitData:
    data = np.load(npz_path, allow_pickle=False)
    X = torch.from_numpy(data["X"].astype("float32"))
    y = torch.from_numpy(data["y"].astype("float32"))
    return SplitData(
        X=X, y=y,
        direction=data["direction"],
        timestamp=data["timestamp"],
        pnl_after_fees_pct=data["pnl_after_fees_pct"],
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Binary classification metrics including precision @ top-K%."""
    metrics = {}
    preds = (scores >= 0.5).astype("int8")
    metrics["accuracy"] = float((preds == y_true).mean())
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        metrics["pr_auc"] = float(average_precision_score(y_true, scores))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    # Precision @ top-K% confidence
    n = len(scores)
    sorted_idx = np.argsort(-scores)  # descending
    for k in TOP_K_PERCENTS:
        n_top = max(1, int(n * k / 100))
        top_idx = sorted_idx[:n_top]
        metrics[f"precision_top_{k}pct"] = float(y_true[top_idx].mean())
    return metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             criterion: nn.Module) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    all_scores: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(loss.item())
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
            all_y.append(yb.cpu().numpy())
    return float(np.mean(losses)), np.concatenate(all_scores), np.concatenate(all_y)


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    pos_weight: float,
    patience: int,
    output_dir: Path,
    use_amp: bool,
) -> dict:
    pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=DEFAULT_LR_REDUCE_FACTOR,
        patience=DEFAULT_LR_REDUCE_PATIENCE,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    best_val_auc = -1.0
    best_epoch = -1
    no_improve = 0
    history: list[dict] = []
    best_path = output_dir / "best_model.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))

        val_loss, val_scores, val_y = evaluate(model, val_loader, device, criterion)
        val_metrics = compute_metrics(val_y, val_scores)
        scheduler.step(val_metrics["roc_auc"])

        epoch_time = time.time() - t0
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": epoch_time,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)

        improved = val_metrics["roc_auc"] > best_val_auc
        if improved:
            best_val_auc = val_metrics["roc_auc"]
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_auc": best_val_auc,
                "metrics": val_metrics,
            }, best_path)
            no_improve = 0
        else:
            no_improve += 1

        flag = "  ⭐ best" if improved else ""
        logger.info(
            "Epoch %3d | train_loss %.4f val_loss %.4f val_acc %.3f val_auc %.4f "
            "p@1%% %.3f p@10%% %.3f lr %.1e t=%.1fs%s",
            epoch, train_loss, val_loss,
            val_metrics["accuracy"], val_metrics["roc_auc"],
            val_metrics["precision_top_1pct"], val_metrics["precision_top_10pct"],
            optimizer.param_groups[0]["lr"], epoch_time, flag,
        )

        if no_improve >= patience:
            logger.info("Early stopping at epoch %d (best epoch=%d, val_auc=%.4f)",
                        epoch, best_epoch, best_val_auc)
            break

    return {"history": history, "best_epoch": best_epoch, "best_val_auc": best_val_auc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=Path, default=Path("data/patchtst_v5/train.npz"))
    p.add_argument("--val", type=Path, default=Path("data/patchtst_v5/val.npz"))
    p.add_argument("--test", type=Path, default=Path("data/patchtst_v5/test.npz"))
    p.add_argument("--output-dir", type=Path, default=Path("models/patchtst_v5/"))
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--patch-len", type=int, default=12)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision (default: enabled on CUDA)")
    p.add_argument("--no-revin", action="store_true", help="Disable RevIN normalization")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data ...")
    train_data = load_split(args.train)
    val_data = load_split(args.val)
    test_data = load_split(args.test)

    n_train, seq_len, n_channels = train_data.X.shape
    n_val, n_test = len(val_data.X), len(test_data.X)
    logger.info("Train: %d events  Val: %d  Test: %d", n_train, n_val, n_test)
    logger.info("Window: %d bars × %d channels", seq_len, n_channels)

    # Class balance
    pos_train = float(train_data.y.mean())
    pos_weight = (1 - pos_train) / max(pos_train, 1e-6)
    logger.info("Train Label=1 ratio: %.3f → BCE pos_weight = %.3f", pos_train, pos_weight)

    train_loader = DataLoader(TensorDataset(train_data.X, train_data.y),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(TensorDataset(val_data.X, val_data.y),
                            batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(TensorDataset(test_data.X, test_data.y),
                             batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=args.num_workers, pin_memory=device.type == "cuda")

    # Build model
    model = PatchTSTClassifier(
        n_channels=n_channels,
        seq_len=seq_len,
        patch_len=args.patch_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        use_revin=not args.no_revin,
    ).to(device)

    n_params = count_parameters(model)
    logger.info("Model: PatchTST CI  patches=%d × patch_len=%d  d_model=%d  heads=%d  layers=%d",
                model.n_patches, args.patch_len, args.d_model, args.n_heads, args.n_layers)
    logger.info("Trainable parameters: %s", f"{n_params:,}")

    # Train
    use_amp = not args.no_amp
    result = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=pos_weight,
        patience=args.patience,
        output_dir=args.output_dir,
        use_amp=use_amp,
    )

    # Save history
    history_path = args.output_dir / "training_history.json"
    history_path.write_text(json.dumps(result["history"], indent=2))
    logger.info("History saved: %s", history_path)

    # Test evaluation with best model
    logger.info("Loading best model (epoch %d, val_auc=%.4f)",
                result["best_epoch"], result["best_val_auc"])
    ckpt = torch.load(args.output_dir / "best_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    test_loss, test_scores, test_y = evaluate(model, test_loader, device, criterion)
    test_metrics = compute_metrics(test_y, test_scores)

    logger.info("=" * 70)
    logger.info("TEST SET RESULTS")
    logger.info("=" * 70)
    logger.info("Test loss          : %.4f", test_loss)
    logger.info("Test accuracy      : %.4f", test_metrics["accuracy"])
    logger.info("Test ROC AUC       : %.4f", test_metrics["roc_auc"])
    logger.info("Test PR AUC        : %.4f", test_metrics["pr_auc"])
    for k in TOP_K_PERCENTS:
        logger.info("Precision @ top %2d%% : %.4f", k, test_metrics[f"precision_top_{k}pct"])
    logger.info("=" * 70)

    # Save predictions for backtest_realistic.py
    pred_path = args.output_dir / "predictions_test.npz"
    np.savez_compressed(
        pred_path,
        scores=test_scores.astype("float32"),
        y_true=test_y.astype("int8"),
        direction=test_data.direction,
        timestamp=test_data.timestamp,
        pnl_after_fees_pct=test_data.pnl_after_fees_pct,
    )
    logger.info("Test predictions saved: %s", pred_path)

    # Save final report
    report = {
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "best_epoch": result["best_epoch"],
        "best_val_auc": result["best_val_auc"],
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_channels": n_channels,
        "seq_len": seq_len,
        "n_parameters": n_params,
        "config": {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
    }
    report_path = args.output_dir / "test_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Test report saved: %s", report_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
