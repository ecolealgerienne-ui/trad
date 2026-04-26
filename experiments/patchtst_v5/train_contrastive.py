"""
train_contrastive.py — PatchTST v5.1 avec Batch-Hard Triplet Loss + BCE multi-task.

Hypothèse v5.1 (suggérée par avis expert externe 2026-04-26):
  Les hard negatives (Label=0 visuellement similaires aux Label=1) ne sont pas
  séparables par BCE seul. La Triplet Loss avec Hard Negative Mining force le
  modèle à éloigner ces look-alikes dans l'espace latent — réutilise la même
  architecture PatchTST CI mais avec un objectif d'entraînement différent.

Architecture loss:
  - Triplet term (Batch-Hard, Schroff et al. CVPR 2015):
      L_triplet = max(0, margin + d(anchor, hardest_positive) - d(anchor, hardest_negative))
    où hardest_pos = max distance same-label, hardest_neg = min distance different-label
  - Classification term: BCEWithLogitsLoss(pos_weight) standard
  - Total: L = α · L_triplet + β · L_bce  (default α=β=1.0)

Réutilise tout le pipeline existant (NPZ datasets, model.py PatchTST CI).
Exigence: seq_len 96 et n_channels (auto-détecté).

Usage:
    python -m experiments.patchtst_v5.train_contrastive \\
        --train data/patchtst_v5/train.npz \\
        --val data/patchtst_v5/val.npz \\
        --test data/patchtst_v5/test.npz \\
        --output-dir models/patchtst_v5_contrastive/

Voir STATUS_v5.0.md (sections v5.1) et experiments/patchtst_v5/README.md.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score

from .model import PatchTSTClassifier, count_parameters

logger = logging.getLogger("patchtst_v5.train_contrastive")


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 256          # plus large pour plus de paires intra-batch
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 15
DEFAULT_LR_REDUCE_FACTOR = 0.5
DEFAULT_LR_REDUCE_PATIENCE = 5
DEFAULT_TRIPLET_MARGIN = 0.5
DEFAULT_TRIPLET_WEIGHT = 1.0
DEFAULT_BCE_WEIGHT = 1.0
DEFAULT_PROJECT_DIM = 128         # projection MLP dim avant triplet
TOP_K_PERCENTS = [1, 5, 10, 25]


# ---------------------------------------------------------------------------
# Data
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
# Contrastive head + loss
# ---------------------------------------------------------------------------

class ContrastiveProjector(nn.Module):
    """Petit MLP qui projette l'embedding PatchTST vers un espace de dim plus
    réduit (par défaut 128) pour stabiliser la triplet loss + L2-normalisation.

    Référence: SimCLR (Chen et al., ICML 2020) + FaceNet (Schroff CVPR 2015).
    """

    def __init__(self, in_dim: int, out_dim: int = DEFAULT_PROJECT_DIM,
                 hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)  # L2-norm sur l'hyper-sphère unité


def batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                            margin: float = DEFAULT_TRIPLET_MARGIN) -> torch.Tensor:
    """Batch-Hard Triplet Loss (Hermans et al., 2017).

    Pour chaque ancre i:
      - hardest positive  = max distance L2 vers samples de même label
      - hardest negative  = min distance L2 vers samples de label opposé
      - loss_i = max(0, margin + d_pos - d_neg)

    Inclut tous les échantillons comme ancres (symétrique).
    """
    if labels.numel() == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # Distances pairwise
    dist = torch.cdist(embeddings, embeddings, p=2)  # (B, B)
    n = dist.size(0)

    same = labels.unsqueeze(0) == labels.unsqueeze(1)         # (B, B) booléen
    different = ~same
    eye = torch.eye(n, dtype=torch.bool, device=dist.device)

    # Hardest positive = max(dist) parmi same_label, exclut self
    pos_mask = same & ~eye
    pos_dist = dist.masked_fill(~pos_mask, -1.0)
    hardest_pos, _ = pos_dist.max(dim=1)

    # Hardest negative = min(dist) parmi different_label
    neg_dist = dist.masked_fill(~different, float("inf"))
    hardest_neg, _ = neg_dist.min(dim=1)

    # Filtrer les ancres sans pair valide (pos ou neg manquant)
    valid = (hardest_pos >= 0) & torch.isfinite(hardest_neg)
    if not valid.any():
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    triplet = F.relu(margin + hardest_pos[valid] - hardest_neg[valid])
    return triplet.mean()


# ---------------------------------------------------------------------------
# Metrics (réutilise la logique de train.py)
# ---------------------------------------------------------------------------

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
        metrics[f"precision_top_{k}pct"] = float(y_true[sorted_idx[:n_top]].mean())
    return metrics


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def evaluate(
    model: PatchTSTClassifier,
    projector: ContrastiveProjector,
    loader: DataLoader,
    device: torch.device,
    bce_criterion: nn.Module,
    triplet_margin: float,
    triplet_weight: float,
    bce_weight: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    projector.eval()
    losses = []
    all_scores: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            emb = model.encode(xb)
            logits = model.classify(emb)
            proj = projector(emb)
            l_bce = bce_criterion(logits, yb)
            l_trip = batch_hard_triplet_loss(proj, yb.to(torch.long), triplet_margin)
            loss = bce_weight * l_bce + triplet_weight * l_trip
            losses.append(loss.item())
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
            all_y.append(yb.cpu().numpy())
    return float(np.mean(losses)), np.concatenate(all_scores), np.concatenate(all_y)


def train_loop(
    model: PatchTSTClassifier,
    projector: ContrastiveProjector,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    pos_weight: float,
    patience: int,
    triplet_margin: float,
    triplet_weight: float,
    bce_weight: float,
    output_dir: Path,
    use_amp: bool,
) -> dict:
    pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    params = list(model.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
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
        projector.train()
        t0 = time.time()
        train_losses = []
        train_bce_losses = []
        train_trip_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                emb = model.encode(xb)
                logits = model.classify(emb)
                proj = projector(emb)
                l_bce = bce_criterion(logits, yb)
                l_trip = batch_hard_triplet_loss(proj, yb.to(torch.long), triplet_margin)
                loss = bce_weight * l_bce + triplet_weight * l_trip
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
            train_losses.append(loss.item())
            train_bce_losses.append(l_bce.item())
            train_trip_losses.append(l_trip.item())

        train_loss = float(np.mean(train_losses))
        train_bce = float(np.mean(train_bce_losses))
        train_trip = float(np.mean(train_trip_losses))

        val_loss, val_scores, val_y = evaluate(
            model, projector, val_loader, device,
            bce_criterion, triplet_margin, triplet_weight, bce_weight,
        )
        val_metrics = compute_metrics(val_y, val_scores)
        scheduler.step(val_metrics["roc_auc"])

        epoch_time = time.time() - t0
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_bce": train_bce,
            "train_triplet": train_trip,
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
                "projector_state_dict": projector.state_dict(),
                "epoch": epoch,
                "val_auc": best_val_auc,
                "metrics": val_metrics,
            }, best_path)
            no_improve = 0
        else:
            no_improve += 1

        flag = "  ⭐ best" if improved else ""
        logger.info(
            "Epoch %3d | train %.4f (bce=%.3f trip=%.3f) val %.4f val_auc %.4f "
            "p@1%% %.3f p@10%% %.3f lr %.1e t=%.1fs%s",
            epoch, train_loss, train_bce, train_trip, val_loss,
            val_metrics["roc_auc"],
            val_metrics["precision_top_1pct"], val_metrics["precision_top_10pct"],
            optimizer.param_groups[0]["lr"], epoch_time, flag,
        )

        if no_improve >= patience:
            logger.info("Early stopping at epoch %d (best epoch=%d, val_auc=%.4f)",
                        epoch, best_epoch, best_val_auc)
            break

    return {"history": history, "best_epoch": best_epoch, "best_val_auc": best_val_auc}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=Path, default=Path("data/patchtst_v5/train.npz"))
    p.add_argument("--val", type=Path, default=Path("data/patchtst_v5/val.npz"))
    p.add_argument("--test", type=Path, default=Path("data/patchtst_v5/test.npz"))
    p.add_argument("--output-dir", type=Path, default=Path("models/patchtst_v5_contrastive/"))
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help=f"Default {DEFAULT_BATCH_SIZE} (plus large pour plus de paires intra-batch)")
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--triplet-margin", type=float, default=DEFAULT_TRIPLET_MARGIN)
    p.add_argument("--triplet-weight", type=float, default=DEFAULT_TRIPLET_WEIGHT,
                   help="Coefficient α du terme triplet dans la loss combinée")
    p.add_argument("--bce-weight", type=float, default=DEFAULT_BCE_WEIGHT,
                   help="Coefficient β du terme BCE dans la loss combinée")
    p.add_argument("--project-dim", type=int, default=DEFAULT_PROJECT_DIM,
                   help="Dimension de projection avant triplet (L2-normalisée)")
    p.add_argument("--patch-len", type=int, default=12)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-revin", action="store_true")
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

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else (args.device if args.device != "auto" else "cpu"))
    logger.info("Device: %s", device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data ...")
    train_data = load_split(args.train)
    val_data = load_split(args.val)
    test_data = load_split(args.test)

    n_train, seq_len, n_channels = train_data.X.shape
    logger.info("Train: %d  Val: %d  Test: %d", n_train, len(val_data.X), len(test_data.X))
    logger.info("Window: %d × %d", seq_len, n_channels)

    pos_train = float(train_data.y.mean())
    pos_weight = (1 - pos_train) / max(pos_train, 1e-6)
    logger.info("Train Label=1 ratio: %.3f → pos_weight = %.3f", pos_train, pos_weight)

    train_loader = DataLoader(
        TensorDataset(train_data.X, train_data.y),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_data.X, val_data.y),
        batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        TensorDataset(test_data.X, test_data.y),
        batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    model = PatchTSTClassifier(
        n_channels=n_channels, seq_len=seq_len,
        patch_len=args.patch_len, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout, use_revin=not args.no_revin,
    ).to(device)
    embedding_dim = n_channels * args.d_model
    projector = ContrastiveProjector(embedding_dim, args.project_dim).to(device)

    n_model_params = count_parameters(model)
    n_proj_params = sum(p.numel() for p in projector.parameters() if p.requires_grad)
    logger.info("PatchTST: %s params  +  Projector(%d→%d): %s params",
                f"{n_model_params:,}", embedding_dim, args.project_dim, f"{n_proj_params:,}")
    logger.info("Loss = %.2f × BCE + %.2f × Triplet (margin=%.2f)",
                args.bce_weight, args.triplet_weight, args.triplet_margin)

    use_amp = not args.no_amp
    result = train_loop(
        model=model, projector=projector,
        train_loader=train_loader, val_loader=val_loader,
        device=device, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, pos_weight=pos_weight,
        patience=args.patience,
        triplet_margin=args.triplet_margin,
        triplet_weight=args.triplet_weight,
        bce_weight=args.bce_weight,
        output_dir=args.output_dir, use_amp=use_amp,
    )

    history_path = args.output_dir / "training_history.json"
    history_path.write_text(json.dumps(result["history"], indent=2))
    logger.info("History saved: %s", history_path)

    logger.info("Loading best model (epoch %d, val_auc=%.4f)",
                result["best_epoch"], result["best_val_auc"])
    ckpt = torch.load(args.output_dir / "best_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    projector.load_state_dict(ckpt["projector_state_dict"])
    pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    test_loss, test_scores, test_y = evaluate(
        model, projector, test_loader, device,
        bce_criterion, args.triplet_margin, args.triplet_weight, args.bce_weight,
    )
    test_metrics = compute_metrics(test_y, test_scores)

    logger.info("=" * 70)
    logger.info("TEST SET RESULTS (v5.1 contrastive)")
    logger.info("=" * 70)
    logger.info("Test loss          : %.4f", test_loss)
    logger.info("Test accuracy      : %.4f", test_metrics["accuracy"])
    logger.info("Test ROC AUC       : %.4f", test_metrics["roc_auc"])
    logger.info("Test PR AUC        : %.4f", test_metrics["pr_auc"])
    for k in TOP_K_PERCENTS:
        logger.info("Precision @ top %2d%% : %.4f", k, test_metrics[f"precision_top_{k}pct"])
    logger.info("=" * 70)

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

    report = {
        "variant": "contrastive_v5.1",
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "best_epoch": result["best_epoch"],
        "best_val_auc": result["best_val_auc"],
        "n_train": n_train,
        "n_val": len(val_data.X),
        "n_test": len(test_data.X),
        "n_channels": n_channels,
        "seq_len": seq_len,
        "n_parameters_model": n_model_params,
        "n_parameters_projector": n_proj_params,
        "config": {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
    }
    report_path = args.output_dir / "test_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Test report saved: %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
