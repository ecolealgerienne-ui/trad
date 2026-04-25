"""Future-slope dataset: target = Oracle[t+1] - Oracle[t].

Variant of build_dataset.py where the prediction target is shifted forward
by 1 step. The model can no longer "cheat" by reproducing the recent past
proxy (which was the case with target = Oracle[t] - Oracle[t-1] yielding
best_lag=-1).

Hypothesis to test:
    - If model still atteint Pearson > 0.5 with best_lag=0 or +1
      => true forward anticipation, useful signal extracted from RSI
    - If Pearson collapses to ~0
      => previous results were ALL proxy learning (reconstruction du passe)

Same Oracle (Kalman 1D RTS Q=0.01, R=0.1), same window (96), same splits
(70/15/15), same anti-leakage RTS scheme.

Differences vs build_dataset.py:
    - Target  : y[t] = oracle[t+1] - oracle[t]   (instead of oracle[t] - oracle[t-1])
    - Bounds  : valid samples have t in [WINDOW-1, split_end - 2]
                (need oracle[t+1] inside the split's oracle estimate)

Output:
    data/foundation/rsi_btc_5min_slope_future.npz
    Same key structure as rsi_btc_5min_slope.npz
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
    KALMAN_MEASURE_VAR,
    KALMAN_PROCESS_VAR,
    RSI_PERIOD,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from data_utils import load_crypto_data
from filters import kalman_filter
from indicators import calculate_rsi


WINDOW = 96


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--csv", default=BTC_DATA_FILE_5M)
    p.add_argument("--rsi-period", type=int, default=RSI_PERIOD)
    p.add_argument("--process-var", type=float, default=KALMAN_PROCESS_VAR)
    p.add_argument("--measure-var", type=float, default=KALMAN_MEASURE_VAR)
    p.add_argument("--window", type=int, default=WINDOW)
    p.add_argument("--train-split", type=float, default=TRAIN_SPLIT)
    p.add_argument("--val-split", type=float, default=VAL_SPLIT)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output",
                   default=str(ROOT / "data" / "foundation" / "rsi_btc_5min_slope_future.npz"))
    return p.parse_args()


def build_split_future(rsi, split_start, split_end, window,
                       process_var, measure_var, name):
    """Compute Oracle on rsi[:split_end], emit (X, y, idx, oracle, rsi_at).

    Target: y[t] = oracle_full[t+1] - oracle_full[t]   (FUTURE slope)
    Valid samples: t in [max(split_start, window-1), split_end - 1)
                   so that t+1 < split_end and oracle_full[t+1] exists.
    """
    print(f"  [{name}] computing Oracle on rsi[:{split_end}] (RTS smoother)...", flush=True)
    oracle_full = kalman_filter(rsi[:split_end],
                                process_variance=process_var,
                                measurement_variance=measure_var)

    # Target needs oracle_full[t+1], so t+1 must be < split_end => t <= split_end - 2
    start = max(split_start, window - 1)
    end = split_end - 1  # exclusive upper bound, so last t is split_end - 2
    idxs = np.arange(start, end, dtype=np.int64)

    X = np.stack([rsi[i - window + 1: i + 1] for i in idxs]).astype(np.float32)
    slope_future = (oracle_full[idxs + 1] - oracle_full[idxs]).astype(np.float32)
    oracle_at_idx = oracle_full[idxs].astype(np.float32)
    rsi_at_idx = rsi[idxs].astype(np.float32)

    return X, slope_future, idxs, oracle_at_idx, rsi_at_idx


def main():
    args = parse_args()

    df = load_crypto_data(args.csv, asset_name="BTC")
    print(f"Loaded {len(df):,} rows from {args.csv}")

    rsi_full = calculate_rsi(df["close"], period=args.rsi_period)
    valid_mask = ~np.isnan(rsi_full)
    rsi = rsi_full[valid_mask].astype(np.float64)
    n = len(rsi)
    print(f"RSI computed (period={args.rsi_period}): {n:,} valid samples")

    if args.max_samples is not None and n > args.max_samples:
        rsi = rsi[: args.max_samples]
        n = len(rsi)
        print(f"  Capped to {n:,} samples")

    n_train = int(args.train_split * n)
    n_val = int(args.val_split * n)
    splits = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n),
    }
    print(f"Splits: train=[0, {n_train}), val=[{n_train}, {n_train + n_val}), "
          f"test=[{n_train + n_val}, {n})")
    print("Target: y[t] = oracle[t+1] - oracle[t]   (FUTURE slope)\n")

    out = {}
    for name, (a, b) in splits.items():
        X, y, idx, oracle, rsi_at = build_split_future(
            rsi, a, b, args.window, args.process_var, args.measure_var, name
        )
        out[f"X_{name}"] = X
        out[f"y_{name}"] = y
        out[f"idx_{name}"] = idx
        out[f"oracle_{name}"] = oracle
        out[f"rsi_{name}"] = rsi_at
        print(f"  [{name}] n={len(y):,}  "
              f"y_future: mean={y.mean():+.5f}  std={y.std():.5f}  "
              f"P5={np.percentile(y, 5):+.5f}  P95={np.percentile(y, 95):+.5f}")

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
        "target": "future_slope",
        "target_formula": "y[t] = oracle[t+1] - oracle[t]",
        "kalman_model": "1D random-walk (pykalman.smooth = RTS)",
    }
    np.savez_compressed(out_path, meta=json.dumps(meta), **out)
    print(f"\nSaved {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
