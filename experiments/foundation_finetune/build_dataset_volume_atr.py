"""Volume + ATR fusion dataset (V1).

Variant of build_dataset_fusion.py with truly orthogonal features:
    - volume_relative[t] = volume[t] / rolling_mean(volume, 20)[t]
    - atr_normalized[t]  = ATR_14[t] / close[t]

Both z-scored on train stats only (anti-leakage).

Hypothesis: unlike MACD/CCI (redundant momentum projections that yielded
no gain in Phase 6), volume and ATR are *independent* signals:
    - volume = activity / conviction (orthogonal to direction)
    - ATR    = volatility (orthogonal to direction)
neither directly encodes the momentum direction captured by RSI.

Same Oracle Kalman 1D RTS (Q=0.01, R=0.1).
Same target: y[t] = Oracle[t] - Oracle[t-1] (PAST slope, for direct
comparison with simple LoRA at Pearson 0.78).
Same window (96), splits (70/15/15), anti-leakage RTS scheme.

Output: data/foundation/rsi_btc_5min_slope_volatr.npz
    Keys per split:
        X_<split>      : (n, 96) RSI causal window
        extras_<split> : (n, 2)  [volume_rel_z, atr_norm_z]
        y_<split>      : (n,)    target oracle slope (past)
        idx_<split>    : (n,) absolute indices
        oracle_<split> : (n,) oracle value at idx
        rsi_<split>    : (n,) RSI value at idx
    Plus: extras_mean, extras_std (train-only).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from constants import (
    BTC_DATA_FILE_5M,
    KALMAN_MEASURE_VAR,
    KALMAN_PROCESS_VAR,
    RSI_PERIOD,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from data_utils import load_crypto_data
from filters import kalman_filter
from indicators import calculate_atr, calculate_rsi


WINDOW = 96
ATR_PERIOD = 14
VOLUME_MA_PERIOD = 20


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--csv", default=BTC_DATA_FILE_5M)
    p.add_argument("--rsi-period", type=int, default=RSI_PERIOD)
    p.add_argument("--atr-period", type=int, default=ATR_PERIOD)
    p.add_argument("--volume-ma-period", type=int, default=VOLUME_MA_PERIOD)
    p.add_argument("--process-var", type=float, default=KALMAN_PROCESS_VAR)
    p.add_argument("--measure-var", type=float, default=KALMAN_MEASURE_VAR)
    p.add_argument("--window", type=int, default=WINDOW)
    p.add_argument("--train-split", type=float, default=TRAIN_SPLIT)
    p.add_argument("--val-split", type=float, default=VAL_SPLIT)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output",
                   default=str(ROOT / "data" / "foundation" / "rsi_btc_5min_slope_volatr.npz"))
    return p.parse_args()


def build_split(rsi, vol_rel, atr_norm,
                split_start, split_end, window,
                process_var, measure_var, name):
    """Compute Oracle on rsi[:split_end], emit (X, extras_raw, y, idx, ...)."""
    print(f"  [{name}] computing Oracle on rsi[:{split_end}] (RTS smoother)...", flush=True)
    oracle_full = kalman_filter(rsi[:split_end],
                                process_variance=process_var,
                                measurement_variance=measure_var)

    start = max(split_start, window - 1, 1)
    end = split_end
    idxs = np.arange(start, end, dtype=np.int64)

    X = np.stack([rsi[i - window + 1: i + 1] for i in idxs]).astype(np.float32)
    extras_raw = np.stack([vol_rel[idxs], atr_norm[idxs]], axis=1).astype(np.float32)
    slope = (oracle_full[idxs] - oracle_full[idxs - 1]).astype(np.float32)
    oracle_at_idx = oracle_full[idxs].astype(np.float32)
    rsi_at_idx = rsi[idxs].astype(np.float32)

    return X, extras_raw, slope, idxs, oracle_at_idx, rsi_at_idx


def main():
    args = parse_args()

    df = load_crypto_data(args.csv, asset_name="BTC")
    print(f"Loaded {len(df):,} rows from {args.csv}")

    # 1. Indicators (causal)
    rsi_full = calculate_rsi(df["close"], period=args.rsi_period)
    atr_full = calculate_atr(df["high"], df["low"], df["close"], period=args.atr_period)
    atr_norm_full = atr_full / df["close"].values   # ATR / close, scale-invariant

    # Volume relative: volume[t] / rolling_mean(volume, K)[t]
    vol_ma = pd.Series(df["volume"].values).rolling(
        window=args.volume_ma_period, min_periods=args.volume_ma_period
    ).mean().values
    vol_rel_full = df["volume"].values / vol_ma  # NaN on first volume_ma_period samples

    # 2. Combined valid mask
    valid_mask = (
        (~np.isnan(rsi_full))
        & (~np.isnan(atr_norm_full))
        & (~np.isnan(vol_rel_full))
        & (np.isfinite(vol_rel_full))   # in case mean=0 -> inf
    )
    rsi = rsi_full[valid_mask].astype(np.float64)
    atr_norm = atr_norm_full[valid_mask].astype(np.float64)
    vol_rel = vol_rel_full[valid_mask].astype(np.float64)
    n = len(rsi)
    print(f"Valid samples (RSI+ATR+Volume all non-NaN): {n:,}  "
          f"(dropped {(~valid_mask).sum()} warmup)")

    if args.max_samples is not None and n > args.max_samples:
        rsi, atr_norm, vol_rel = rsi[:args.max_samples], atr_norm[:args.max_samples], vol_rel[:args.max_samples]
        n = len(rsi)
        print(f"  Capped to {n:,} samples")

    # Quick stats
    print(f"\nFeature stats (full series):")
    print(f"  rsi      : min={rsi.min():.2f}  max={rsi.max():.2f}  mean={rsi.mean():.2f}")
    print(f"  atr_norm : min={atr_norm.min():.6f}  max={atr_norm.max():.6f}  "
          f"median={np.median(atr_norm):.6f}")
    print(f"  vol_rel  : min={vol_rel.min():.4f}  max={vol_rel.max():.4f}  "
          f"median={np.median(vol_rel):.4f}")

    # 3. Splits
    n_train = int(args.train_split * n)
    n_val = int(args.val_split * n)
    splits = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n),
    }
    print(f"\nSplits: train=[0, {n_train}), val=[{n_train}, {n_train + n_val}), "
          f"test=[{n_train + n_val}, {n})")

    # 4. Build per-split
    splits_data = {}
    for name, (a, b) in splits.items():
        X, extras_raw, y, idx, oracle, rsi_at = build_split(
            rsi, vol_rel, atr_norm, a, b, args.window,
            args.process_var, args.measure_var, name,
        )
        splits_data[name] = (X, extras_raw, y, idx, oracle, rsi_at)

    # 5. Z-score extras on train stats only
    train_extras = splits_data["train"][1]
    extras_mean = train_extras.mean(axis=0)
    extras_std = train_extras.std(axis=0).clip(min=1e-8)
    print(f"\nExtras z-score stats (train only):")
    print(f"  vol_rel : mean={extras_mean[0]:+.4f}  std={extras_std[0]:.4f}")
    print(f"  atr_norm: mean={extras_mean[1]:+.6f}  std={extras_std[1]:.6f}")

    out = {}
    for name, (X, extras_raw, y, idx, oracle, rsi_at) in splits_data.items():
        extras_z = ((extras_raw - extras_mean) / extras_std).astype(np.float32)
        out[f"X_{name}"] = X
        out[f"extras_{name}"] = extras_z
        out[f"y_{name}"] = y
        out[f"idx_{name}"] = idx
        out[f"oracle_{name}"] = oracle
        out[f"rsi_{name}"] = rsi_at

        m, s = float(y.mean()), float(y.std())
        em = extras_z.mean(axis=0)
        es = extras_z.std(axis=0)
        print(f"  [{name}] n={len(y):,}  y: mean={m:+.5f} std={s:.5f}  "
              f"extras_z: vol_mean={em[0]:+.4f}/std={es[0]:.4f}  "
              f"atr_mean={em[1]:+.4f}/std={es[1]:.4f}")

    out["extras_mean"] = extras_mean.astype(np.float64)
    out["extras_std"] = extras_std.astype(np.float64)

    # 6. Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "csv": str(args.csv),
        "rsi_period": int(args.rsi_period),
        "atr_period": int(args.atr_period),
        "volume_ma_period": int(args.volume_ma_period),
        "process_var": float(args.process_var),
        "measure_var": float(args.measure_var),
        "window": int(args.window),
        "splits": {k: [int(v[0]), int(v[1])] for k, v in splits.items()},
        "n_total_valid": int(n),
        "anti_leakage_rts": True,
        "extras": ["volume_relative_z", "atr_normalized_z"],
        "extras_normalization": "z-score using train stats",
        "kalman_model": "1D random-walk (pykalman.smooth = RTS)",
    }
    np.savez_compressed(out_path, meta=json.dumps(meta), **out)
    print(f"\nSaved {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
