"""Evaluate trained ChronosRegressor checkpoint(s) on val + test sets.

Loads one or more checkpoints saved by train.py, runs inference on val + test,
and prints a comparison table that includes:
  - Trained models (probing / lora / full)
  - Baselines (identity, raw_slope, ma_slope_K)
  - Lag CCF check (does the prediction lead/lag the Oracle?)

Outputs:
  - Markdown-style table to stdout
  - JSON dump to data/foundation/evaluate_summary.json

Usage:
    # Default: evaluate all .pt files in models/foundation_finetune/
    python experiments/foundation_finetune/evaluate.py

    # Specific checkpoint(s)
    python experiments/foundation_finetune/evaluate.py \
        --ckpt models/foundation_finetune/chronos-t5-tiny_probing.pt \
        --ckpt models/foundation_finetune/chronos-t5-tiny_lora.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import predict_identity, predict_ma_slope, predict_raw_slope
from model import ChronosRegressor
from train import SlopeDataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "foundation" / "rsi_btc_5min_slope.npz"
DEFAULT_CKPT_DIR = ROOT / "models" / "foundation_finetune"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--ckpt", action="append", default=None,
                   help="Checkpoint .pt file(s). If omitted, all .pt in models/foundation_finetune/.")
    p.add_argument("--ckpt-dir", default=str(DEFAULT_CKPT_DIR))
    p.add_argument("--ma-windows", nargs="+", type=int, default=[5, 10, 20])
    p.add_argument("--lag-range", type=int, default=5,
                   help="Compute CCF for lags in [-N, +N].")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap samples per split (debug).")
    p.add_argument("--output",
                   default=str(ROOT / "data" / "foundation" / "evaluate_summary.json"))
    return p.parse_args()


def metrics(yhat, y):
    yhat = np.asarray(yhat, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mse = float(np.mean((yhat - y) ** 2))
    mae = float(np.mean(np.abs(yhat - y)))
    same = (np.sign(yhat) * np.sign(y)) > 0
    nz = (yhat != 0) & (y != 0)
    dm = float(same.sum() / max(nz.sum(), 1))
    pearson = 0.0 if (yhat.std() < 1e-12 or y.std() < 1e-12) else \
        float(np.corrcoef(yhat, y)[0, 1])
    return {"MSE": mse, "MAE": mae, "DirMatch": dm, "Pearson": pearson}


def lag_ccf(yhat, y, max_lag=5):
    """Cross-correlation between yhat and y at lags in [-max_lag, +max_lag].

    lag > 0 means y is shifted forward (yhat[t] vs y[t+lag]):
        positive peak at lag=+k => yhat anticipates y by k steps
    lag < 0 means yhat is shifted forward (yhat[t+|lag|] vs y[t]):
        positive peak at lag=-k => yhat lags y by k steps
    Peak at lag=0 => synchronous (best for our task: predict Oracle[t]).
    """
    yhat = np.asarray(yhat, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    yhat = (yhat - yhat.mean()) / max(yhat.std(), 1e-12)
    y = (y - y.mean()) / max(y.std(), 1e-12)
    n = len(y)
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            c = float(np.mean(yhat * y))
        elif lag > 0:
            c = float(np.mean(yhat[:n - lag] * y[lag:]))
        else:
            c = float(np.mean(yhat[-lag:] * y[:n + lag]))
        out[lag] = c
    best_lag = max(out, key=lambda k: out[k])
    return out, best_lag


def load_model(ckpt_path: Path, device: str) -> tuple[ChronosRegressor, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    extra_dim = int(ckpt.get("extra_dim", 0))
    kwargs = dict(model_name=args["model"], head_hidden=args.get("head_hidden", 64),
                  extra_dim=extra_dim, device=device)
    mode = args["mode"]
    if mode == "probing":
        kwargs.update(freeze_backbone=True, use_lora=False)
    elif mode == "lora":
        kwargs.update(freeze_backbone=True, use_lora=True,
                      lora_rank=args.get("lora_rank", 8))
    else:
        kwargs.update(freeze_backbone=False, use_lora=False)

    model = ChronosRegressor(**kwargs).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def predict_model(model, X, extras, device, batch_size, num_workers):
    has_extras = extras is not None and model.extra_dim > 0
    ds = SlopeDataset(X, np.zeros(len(X), dtype=np.float32),
                      extras=extras if has_extras else None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device == "cuda"))
    preds = []
    with torch.no_grad():
        for batch in loader:
            if has_extras:
                x, ex, _ = batch
                x = x.to(device, non_blocking=True)
                ex = ex.to(device, non_blocking=True)
                preds.append(model(x, ex).cpu().numpy())
            else:
                x, _ = batch
                x = x.to(device, non_blocking=True)
                preds.append(model(x).cpu().numpy())
    return np.concatenate(preds)


def collect_predictors(args, models_info, X, extras):
    """Yield (name, yhat) for baselines + each loaded model."""
    yield "identity", predict_identity(X)
    yield "raw_slope", predict_raw_slope(X)
    for K in args.ma_windows:
        if K + 1 <= X.shape[1]:
            yield f"ma_slope_{K}", predict_ma_slope(X, K)
    for name, (model, _) in models_info.items():
        yhat = predict_model(model, X, extras, args._device,
                             args.batch_size, args.num_workers)
        yield name, yhat


def fmt_row(name, m, lag_info=None):
    s = (f"{name:<26} {m['MSE']:>9.5f} {m['MAE']:>9.5f} "
         f"{m['DirMatch']:>9.4f} {m['Pearson']:>+9.4f}")
    if lag_info is not None:
        ccf, best = lag_info
        s += f"   best_lag={best:+d} (ccf={ccf[best]:+.3f})"
    return s


def print_table(split_name, rows):
    print(f"\n=== {split_name.upper()} ===")
    header = (f"{'predictor':<26} {'MSE':>9} {'MAE':>9} "
              f"{'DirMatch':>9} {'Pearson':>9}   lag-CCF")
    print(header)
    print("-" * len(header))
    for line in rows:
        print(line)


def main():
    args = parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    args._device = device
    print(f"Device: {device}")

    # Load data
    print(f"Loading {args.data} ...")
    data = np.load(args.data, allow_pickle=True)
    meta = json.loads(str(data["meta"]))
    summary = " ".join(f"{k}={v}" for k, v in meta.items()
                       if k in ("window", "process_var", "measure_var",
                                "rsi_period", "indicator", "tf_minutes",
                                "adaptive"))
    print(f"  meta: {summary}")

    # Resolve checkpoints
    if args.ckpt:
        ckpt_paths = [Path(p) for p in args.ckpt]
    else:
        ckpt_dir = Path(args.ckpt_dir)
        ckpt_paths = sorted(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
    print(f"\nCheckpoints to evaluate ({len(ckpt_paths)}):")
    for p in ckpt_paths:
        print(f"  - {p}")

    # Load models
    models_info = {}
    for cp in ckpt_paths:
        model, ckpt = load_model(cp, device)
        tag = cp.stem  # e.g. "chronos-t5-tiny_probing"
        models_info[tag] = (model, ckpt)
        print(f"  loaded {tag}: trainable={model.count_trainable():,}  "
              f"epoch={ckpt.get('epoch', '?')}  val_mse={ckpt.get('val_metrics', {}).get('mse', '?')}")

    has_extras_in_data = "extras_train" in data.files
    if has_extras_in_data:
        print(f"  dataset has extras (extras_train.shape={data['extras_train'].shape})")

    # Evaluate each split
    summary = {"meta": meta, "ckpts": [str(p) for p in ckpt_paths], "splits": {}}
    for split in ("val", "test"):
        X = data[f"X_{split}"]
        y = data[f"y_{split}"]
        extras = data[f"extras_{split}"] if has_extras_in_data else None
        if args.max_samples:
            X, y = X[:args.max_samples], y[:args.max_samples]
            if extras is not None:
                extras = extras[:args.max_samples]

        rows_lines = []
        rows_data = {}
        for name, yhat in collect_predictors(args, models_info, X, extras):
            m = metrics(yhat, y)
            ccf, best_lag = lag_ccf(yhat, y, max_lag=args.lag_range)
            m["best_lag"] = best_lag
            m["ccf_at_best"] = ccf[best_lag]
            m["ccf_full"] = ccf
            rows_data[name] = m
            rows_lines.append(fmt_row(name, m, (ccf, best_lag)))

        print_table(split, rows_lines)
        summary["splits"][split] = rows_data

    # Save JSON summary
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
