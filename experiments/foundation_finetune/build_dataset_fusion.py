"""Fusion dataset: RSI window (96) + instantaneous MACD/CCI slopes as extras.

Variant of build_dataset.py with two additional features per sample:
    extras[i] = [
        MACD_slope[i] = MACD[i] - MACD[i-1]   (z-scored using train stats)
        CCI_slope[i]  = CCI[i]  - CCI[i-1]    (z-scored using train stats)
    ]

Same anti-leakage RTS Oracle as build_dataset.py.
Same window (96), splits (70/15/15), Kalman config (Q=0.01, R=0.1).

Output: data/foundation/rsi_btc_5min_slope_fusion.npz
    Keys per split (train/val/test):
        X_<split>      : (n, 96) RSI causal window
        extras_<split> : (n, 2)  [macd_slope_z, cci_slope_z]
        y_<split>      : (n,)    target oracle slope
        idx_<split>    : (n,) absolute indices
        oracle_<split> : (n,) oracle value at idx
        rsi_<split>    : (n,) RSI value at idx
    Plus: extras_mean, extras_std (train-only, used at inference)
    Plus: meta JSON.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from constants import (
    BTC_DATA_FILE_5M,
    CCI_PERIOD,
    KALMAN_MEASURE_VAR,
    KALMAN_PROCESS_VAR,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_PERIOD,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from data_utils import load_crypto_data
from filters import kalman_filter
from indicators import calculate_cci, calculate_macd, calculate_rsi


WINDOW = 96


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--csv", default=BTC_DATA_FILE_5M)
    p.add_argument("--rsi-period", type=int, default=RSI_PERIOD)
    p.add_argument("--macd-fast", type=int, default=MACD_FAST)
    p.add_argument("--macd-slow", type=int, default=MACD_SLOW)
    p.add_argument("--macd-signal", type=int, default=MACD_SIGNAL)
    p.add_argument("--cci-period", type=int, default=CCI_PERIOD)
    p.add_argument("--process-var", type=float, default=KALMAN_PROCESS_VAR)
    p.add_argument("--measure-var", type=float, default=KALMAN_MEASURE_VAR)
    p.add_argument("--window", type=int, default=WINDOW)
    p.add_argument("--train-split", type=float, default=TRAIN_SPLIT)
    p.add_argument("--val-split", type=float, default=VAL_SPLIT)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output",
                   default=str(ROOT / "data" / "foundation" / "rsi_btc_5min_slope_fusion.npz"))
    return p.parse_args()


def build_split(rsi, macd_slope, cci_slope,
                split_start, split_end, window,
                process_var, measure_var, name):
    """Compute Oracle on rsi[:split_end] (anti-leakage RTS), emit X, extras, y, idx."""
    print(f"  [{name}] computing Oracle on rsi[:{split_end}] (RTS smoother)...", flush=True)
    oracle_full = kalman_filter(rsi[:split_end],
                                process_variance=process_var,
                                measurement_variance=measure_var)

    start = max(split_start, window - 1, 1)
    end = split_end
    idxs = np.arange(start, end, dtype=np.int64)

    X = np.stack([rsi[i - window + 1: i + 1] for i in idxs]).astype(np.float32)
    extras_raw = np.stack([macd_slope[idxs], cci_slope[idxs]], axis=1).astype(np.float32)
    slope = (oracle_full[idxs] - oracle_full[idxs - 1]).astype(np.float32)
    oracle_at_idx = oracle_full[idxs].astype(np.float32)
    rsi_at_idx = rsi[idxs].astype(np.float32)

    return X, extras_raw, slope, idxs, oracle_at_idx, rsi_at_idx


def main():
    args = parse_args()

    # 1. Load BTC 5min OHLC
    df = load_crypto_data(args.csv, asset_name="BTC")
    print(f"Loaded {len(df):,} rows from {args.csv}")

    # 2. Compute RSI, MACD, CCI on full series
    rsi_full = calculate_rsi(df["close"], period=args.rsi_period)
    macd_dict = calculate_macd(df["close"], fast_period=args.macd_fast,
                               slow_period=args.macd_slow,
                               signal_period=args.macd_signal)
    macd_full = macd_dict["macd"]
    cci_full = calculate_cci(df["high"], df["low"], df["close"],
                             period=args.cci_period)

    # 3. Combined valid mask (where ALL three indicators are non-NaN)
    valid_mask = (~np.isnan(rsi_full)) & (~np.isnan(macd_full)) & (~np.isnan(cci_full))
    rsi = rsi_full[valid_mask].astype(np.float64)
    macd = macd_full[valid_mask].astype(np.float64)
    cci = cci_full[valid_mask].astype(np.float64)
    n = len(rsi)
    print(f"Valid samples (RSI+MACD+CCI all non-NaN): {n:,}  "
          f"(dropped {(~valid_mask).sum()} warmup)")

    if args.max_samples is not None and n > args.max_samples:
        rsi, macd, cci = rsi[:args.max_samples], macd[:args.max_samples], cci[:args.max_samples]
        n = len(rsi)
        print(f"  Capped to {n:,} samples (--max-samples)")

    # 4. Slopes (causal first differences)
    macd_slope = np.diff(macd, prepend=np.nan)
    cci_slope = np.diff(cci, prepend=np.nan)

    # 5. Temporal splits
    n_train = int(args.train_split * n)
    n_val = int(args.val_split * n)
    splits = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n),
    }
    print(f"Splits: train=[0, {n_train}), val=[{n_train}, {n_train + n_val}), "
          f"test=[{n_train + n_val}, {n})")

    # 6. Build per-split arrays
    out = {}
    splits_data = {}
    for name, (a, b) in splits.items():
        X, extras_raw, y, idx, oracle, rsi_at = build_split(
            rsi, macd_slope, cci_slope, a, b, args.window,
            args.process_var, args.measure_var, name,
        )
        splits_data[name] = (X, extras_raw, y, idx, oracle, rsi_at)

    # 7. Z-score extras using train stats (anti-leakage)
    train_extras_raw = splits_data["train"][1]
    extras_mean = train_extras_raw.mean(axis=0)
    extras_std = train_extras_raw.std(axis=0).clip(min=1e-8)
    print(f"\nExtras z-score stats (from train only):")
    print(f"  macd_slope: mean={extras_mean[0]:+.5f}  std={extras_std[0]:.5f}")
    print(f"  cci_slope:  mean={extras_mean[1]:+.5f}  std={extras_std[1]:.5f}")

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
              f"extras_z: macd_mean={em[0]:+.4f}/std={es[0]:.4f}  "
              f"cci_mean={em[1]:+.4f}/std={es[1]:.4f}")

    out["extras_mean"] = extras_mean.astype(np.float64)
    out["extras_std"] = extras_std.astype(np.float64)

    # 8. Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "csv": str(args.csv),
        "rsi_period": int(args.rsi_period),
        "macd": {"fast": int(args.macd_fast), "slow": int(args.macd_slow),
                 "signal": int(args.macd_signal)},
        "cci_period": int(args.cci_period),
        "process_var": float(args.process_var),
        "measure_var": float(args.measure_var),
        "window": int(args.window),
        "splits": {k: [int(v[0]), int(v[1])] for k, v in splits.items()},
        "n_total_valid": int(n),
        "anti_leakage_rts": True,
        "extras": ["macd_slope_z", "cci_slope_z"],
        "extras_normalization": "z-score using train stats",
        "kalman_model": "1D random-walk (pykalman.smooth = RTS)",
    }
    np.savez_compressed(out_path, meta=json.dumps(meta), **out)
    print(f"\nSaved {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
