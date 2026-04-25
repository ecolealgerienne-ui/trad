"""Build dataset for Lag-Llama fine-tuning: RSI[t-95:t] -> slope_oracle[t].

Reuses project functions from src/:
    - load_crypto_data    (load BTC 5min CSV)
    - calculate_rsi       (RSI computation)
    - kalman_filter       (1D random-walk + RTS smoother via pykalman)

Anti-leakage RTS: Oracle is computed separately per split.
    oracle_train  = kalman_filter(rsi[:n_train])
    oracle_val    = kalman_filter(rsi[:n_train + n_val])
    oracle_test   = kalman_filter(rsi[:n])             # full history

For each split, sliding windows X[i] = rsi[i-95 : i+1] (96 values, causal),
target y[i] = oracle[i] - oracle[i-1].
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
    RSI_PERIOD,
    KALMAN_PROCESS_VAR,
    KALMAN_MEASURE_VAR,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from data_utils import load_crypto_data
from indicators import calculate_rsi
from filters import kalman_filter


WINDOW = 96


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--csv", default=BTC_DATA_FILE_5M)
    p.add_argument("--rsi-period", type=int, default=RSI_PERIOD)
    p.add_argument("--process-var", type=float, default=KALMAN_PROCESS_VAR,
                   help="Kalman process variance Q (default: project constant 0.01)")
    p.add_argument("--measure-var", type=float, default=KALMAN_MEASURE_VAR,
                   help="Kalman measurement variance R (default: project constant 0.1)")
    p.add_argument("--window", type=int, default=WINDOW)
    p.add_argument("--train-split", type=float, default=TRAIN_SPLIT)
    p.add_argument("--val-split", type=float, default=VAL_SPLIT)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap total RSI length (debug). Default: full history.")
    p.add_argument("--output",
                   default=str(ROOT / "data" / "foundation" / "rsi_btc_5min_slope.npz"))
    return p.parse_args()


def build_split(rsi, split_start, split_end, window, process_var, measure_var, name):
    """Compute Oracle on rsi[:split_end], emit (X, y, idx, oracle_val) for [split_start, split_end)."""
    print(f"  [{name}] computing Oracle on rsi[:{split_end}] (RTS smoother)...", flush=True)
    oracle_full = kalman_filter(rsi[:split_end],
                                process_variance=process_var,
                                measurement_variance=measure_var)

    start = max(split_start, window - 1, 1)
    end = split_end
    idxs = np.arange(start, end, dtype=np.int64)

    X = np.stack([rsi[i - window + 1: i + 1] for i in idxs]).astype(np.float32)
    slope = (oracle_full[idxs] - oracle_full[idxs - 1]).astype(np.float32)
    oracle_at_idx = oracle_full[idxs].astype(np.float32)
    rsi_at_idx = rsi[idxs].astype(np.float32)

    return X, slope, idxs, oracle_at_idx, rsi_at_idx


def main():
    args = parse_args()

    # 1. Load BTC 5min OHLC
    df = load_crypto_data(args.csv, asset_name="BTC")
    print(f"Loaded {len(df):,} rows from {args.csv}")

    # 2. RSI on close prices, drop warmup NaN
    rsi_full = calculate_rsi(df["close"], period=args.rsi_period)
    valid_mask = ~np.isnan(rsi_full)
    rsi = rsi_full[valid_mask].astype(np.float64)  # float64 for pykalman stability
    n = len(rsi)
    print(f"RSI computed (period={args.rsi_period}): {n:,} valid samples "
          f"(dropped {(~valid_mask).sum()} warmup NaN)")

    if args.max_samples is not None and n > args.max_samples:
        rsi = rsi[: args.max_samples]
        n = len(rsi)
        print(f"  Capped to {n:,} samples (--max-samples)")

    # 3. Temporal splits
    n_train = int(args.train_split * n)
    n_val = int(args.val_split * n)
    splits = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n),
    }
    print(f"Splits: train=[0, {n_train}), val=[{n_train}, {n_train + n_val}), "
          f"test=[{n_train + n_val}, {n})")

    # 4. Per-split Oracle (anti-leakage) + sliding windows
    out = {}
    for name, (a, b) in splits.items():
        X, y, idx, oracle, rsi_at = build_split(
            rsi, a, b, args.window, args.process_var, args.measure_var, name
        )
        out[f"X_{name}"] = X
        out[f"y_{name}"] = y
        out[f"idx_{name}"] = idx
        out[f"oracle_{name}"] = oracle
        out[f"rsi_{name}"] = rsi_at
        print(f"  [{name}] n={len(y):,}  "
              f"slope: mean={y.mean():+.5f}  std={y.std():.5f}  "
              f"P5={np.percentile(y, 5):+.5f}  P95={np.percentile(y, 95):+.5f}")

    # 5. Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "csv": str(args.csv),
        "rsi_period": int(args.rsi_period),
        "process_var": float(args.process_var),
        "measure_var": float(args.measure_var),
        "window": int(args.window),
        "splits": {k: [int(v[0]), int(v[1])] for k, v in splits.items()},
        "n_total_rsi": int(n),
        "anti_leakage_rts": True,
        "kalman_model": "1D random-walk (pykalman.smooth = RTS)",
    }
    np.savez_compressed(out_path, meta=json.dumps(meta), **out)
    print(f"\nSaved {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
