"""Train ChronosRegressor on RSI[t-95:t] -> slope_oracle[t].

Modes:
    --mode probing : freeze T5, train MLP head only       (~16k trainable, lr=1e-3)
    --mode lora    : LoRA r=8 on T5 q/v + train head      (~115k trainable, lr=1e-4)
    --mode full    : full fine-tune                       (~8.4M trainable, lr=1e-5)

Loss : MSE on slope. Early stopping on val MSE.
Saves: best checkpoint + JSON history to models/foundation_finetune/.

Usage:
    # Quick probing (debug, 5k train samples, 2 epochs)
    python experiments/foundation_finetune/train.py --mode probing \
        --max-train 5000 --max-val 1000 --epochs 2

    # Full probing run
    python experiments/foundation_finetune/train.py --mode probing --epochs 5

    # LoRA fine-tuning
    python experiments/foundation_finetune/train.py --mode lora --epochs 5
"""

import argparse
import json
import sys
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ChronosRegressor


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "foundation" / "rsi_btc_5min_slope.npz"
DEFAULT_OUTDIR = ROOT / "models" / "foundation_finetune"

DEFAULT_LRS = {"probing": 1e-3, "lora": 1e-4, "full": 1e-5}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--model", default="amazon/chronos-t5-tiny")
    p.add_argument("--mode", choices=["probing", "lora", "full"], default="probing")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--head-hidden", type=int, default=64)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=None,
                   help="Override default LR (default: 1e-3 probing, 1e-4 lora, 1e-5 full)")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--clip-grad", type=float, default=1.0)

    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-val", type=int, default=None)

    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--output-dir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--log-every", type=int, default=50,
                   help="Log every N batches.")
    return p.parse_args()


class SlopeDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


def evaluate_split(model, loader, device, criterion):
    model.eval()
    yhats, ys, losses = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yhat = model(x)
            losses.append(criterion(yhat, y).item())
            yhats.append(yhat.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
    yhats = np.concatenate(yhats)
    ys = np.concatenate(ys)

    mse = float(np.mean((yhats - ys) ** 2))
    mae = float(np.mean(np.abs(yhats - ys)))
    same = (np.sign(yhats) * np.sign(ys)) > 0
    nonzero = (yhats != 0) & (ys != 0)
    dirmatch = float(same.sum() / max(nonzero.sum(), 1))
    pearson = 0.0 if yhats.std() < 1e-12 else float(np.corrcoef(yhats, ys)[0, 1])

    return {
        "loss_mean": float(np.mean(losses)),
        "mse": mse,
        "mae": mae,
        "dirmatch": dirmatch,
        "pearson": pearson,
    }


def main():
    args = parse_args()
    if args.lr is None:
        args.lr = DEFAULT_LRS[args.mode]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Load dataset
    print(f"Loading {args.data}...")
    data = np.load(args.data, allow_pickle=True)
    meta = json.loads(str(data["meta"]))
    print(f"  meta: window={meta['window']} Q={meta['process_var']} "
          f"R={meta['measure_var']} rsi_period={meta['rsi_period']}")

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    if args.max_train:
        X_train, y_train = X_train[:args.max_train], y_train[:args.max_train]
    if args.max_val:
        X_val, y_val = X_val[:args.max_val], y_val[:args.max_val]
    print(f"  train={len(y_train):,}  val={len(y_val):,}")

    train_loader = DataLoader(
        SlopeDataset(X_train, y_train),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        SlopeDataset(X_val, y_val),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )

    # Model
    print(f"\nLoading {args.model} (mode={args.mode}) ...")
    kwargs = dict(model_name=args.model, head_hidden=args.head_hidden, device=device)
    if args.mode == "probing":
        kwargs.update(freeze_backbone=True, use_lora=False)
    elif args.mode == "lora":
        kwargs.update(freeze_backbone=True, use_lora=True, lora_rank=args.lora_rank)
    else:
        kwargs.update(freeze_backbone=False, use_lora=False)

    model = ChronosRegressor(**kwargs).to(device)
    print(f"  trainable: {model.count_trainable():,}  total: {model.count_total():,}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                  weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    # Output dir + tags
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model.split('/')[-1]}_{args.mode}"
    ckpt_path = out_dir / f"{tag}.pt"
    history_path = out_dir / f"{tag}_history.json"

    # Train loop
    print(f"\nTraining: {args.epochs} epochs  bs={args.batch_size}  lr={args.lr}  "
          f"patience={args.patience}")
    history = []
    best_val_mse = float("inf")
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time()
        model.train()
        train_losses = []
        n_batches = len(train_loader)
        for i, (x, y) in enumerate(train_loader, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yhat = model(x)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.clip_grad)
            optimizer.step()
            train_losses.append(loss.item())
            if i % args.log_every == 0 or i == n_batches:
                running = float(np.mean(train_losses[-args.log_every:]))
                print(f"  ep{epoch} [{i:>5}/{n_batches}]  loss={loss.item():.5f}  "
                      f"running={running:.5f}", flush=True)

        train_loss = float(np.mean(train_losses))
        val = evaluate_split(model, val_loader, device, criterion)
        elapsed = time() - t0

        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss_mean"],
            "val_mse": val["mse"],
            "val_mae": val["mae"],
            "val_dirmatch": val["dirmatch"],
            "val_pearson": val["pearson"],
            "elapsed_s": elapsed,
        }
        history.append(log)
        print(f"\n  >>> ep{epoch}/{args.epochs}  "
              f"train={train_loss:.5f}  val_mse={val['mse']:.5f}  "
              f"val_dir={val['dirmatch']:.4f}  val_pearson={val['pearson']:+.4f}  "
              f"({elapsed:.0f}s)\n", flush=True)

        if val["mse"] < best_val_mse - 1e-6:
            best_val_mse = val["mse"]
            no_improve = 0
            torch.save({
                "state_dict": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "val_metrics": val,
                "data_meta": meta,
            }, ckpt_path)
            print(f"  [saved best -> {ckpt_path}]")
        else:
            no_improve += 1
            print(f"  [no improvement {no_improve}/{args.patience}]")
            if no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

        history_path.write_text(json.dumps({
            "args": vars(args), "history": history, "best_val_mse": best_val_mse
        }, indent=2))

    # Final history dump
    history_path.write_text(json.dumps({
        "args": vars(args), "history": history, "best_val_mse": best_val_mse
    }, indent=2))
    print(f"\nDone. Best val_mse={best_val_mse:.5f}  ckpt={ckpt_path}  history={history_path}")


if __name__ == "__main__":
    main()
