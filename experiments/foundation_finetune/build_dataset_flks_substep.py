"""Build dataset: FLKS progressive slope -> Oracle 30min slope (regression).

Adaptation a notre infra Chronos du pipeline FLKS deja existant dans
src/signal_processing/core.py et src/signal_processing/prepare_flks_csv.py.
Indicator parametrable (macd | rsi | cci, default: macd pour aligner avec
les references slope_improvement et prepare_flks_csv.py).

Pipeline (entierement reutilise depuis src.signal_processing.core) :
    1. load_csv                         (5min OHLCV)
    2. resample_ohlcv                   (-> 30min)
    3. prepare_features_and_labels_progressive(indicator, tf=30)
       -> slope_progressive (FLKS @ sous-pas k), step_k, label_continuous, close
    4. split_train_val_test (70/15/15, gap=window)
    5. normalize_features (z-score sur train uniquement)
    6. make_sequences (window=25)

Output : data/foundation/<indicator>_btc_5min_flks_substep.npz
    Format compatible avec experiments/foundation_finetune/train.py (mode lora)
    et evaluate.py (extras detecte automatiquement).

Architecture Chronos cible :
    X (single-channel) = slope_progressive[t-24:t+1]  (25 valeurs)
    y (regression)     = label_continuous[t]          (Oracle 30min smoothed slope)
    extras (optionnel) = step_k[t] / 5.0              (0..1, position dans la barre)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "signal_processing"))

from signal_processing.core import (
    load_csv,
    resample_ohlcv,
    prepare_features_and_labels_progressive,
    split_train_val_test,
    normalize_features,
    make_sequences,
)


WINDOW = 25
TF_MINUTES = 30


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--csv", default=str(ROOT / "data_trad" / "BTCUSD_all_5m.csv"))
    p.add_argument("--indicator", default="macd", choices=["macd", "rsi", "cci"],
                   help="Indicator to use (matches src.signal_processing.core).")
    p.add_argument("--n-candles-30m", type=int, default=0,
                   help="Cap to last N 30min candles (0 = no cap)")
    p.add_argument("--window", type=int, default=WINDOW)
    p.add_argument("--trim", type=int, default=100,
                   help="Trim N 30min bars at start AND end")
    p.add_argument("--adaptive", action="store_true",
                   help="Use AQ-KF (adaptive Q) instead of Standard FLKS")
    p.add_argument("--include-slope-extra", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Skip connection: pass slope_progressive[t] (z-scored global) "
                        "as extras to bypass Chronos tokenizer's local scaling. "
                        "Required for the model to reproduce FLKS. Default: True.")
    p.add_argument("--use-step-k-extra", action="store_true",
                   help="Add step_k as additional extras feature")
    p.add_argument("--output", default=None,
                   help="Output path. If None: data/foundation/<indicator>_btc_5min_flks_substep.npz")
    args = p.parse_args()
    if args.output is None:
        args.output = str(ROOT / "data" / "foundation"
                          / f"{args.indicator}_btc_5min_flks_substep.npz")
    return args


def main():
    args = parse_args()

    # 1. Load 5min CSV
    df_5m = load_csv(args.csv)
    print(f"[1/6] Loaded {len(df_5m):,} 5min candles from {args.csv}")

    # 2. Resample to 30min
    df_30m = resample_ohlcv(df_5m, TF_MINUTES)
    if args.n_candles_30m > 0 and len(df_30m) > args.n_candles_30m:
        df_30m = df_30m.iloc[-args.n_candles_30m:]
        df_5m = df_5m.loc[df_30m.index[0]:df_30m.index[-1] + pd.Timedelta(minutes=29)]
    print(f"[2/6] Resampled: {len(df_30m):,} 30min bars, {len(df_5m):,} 5min bars")

    # 3. Compute progressive features + labels via core.py
    print(f"[3/6] Computing progressive FLKS slopes + Oracle labels "
          f"(indicator={args.indicator}, tf={TF_MINUTES}min, "
          f"adaptive={args.adaptive}, trim={args.trim})...")
    df_full = prepare_features_and_labels_progressive(
        df_30m, df_5m, args.indicator, TF_MINUTES,
        trim=args.trim,
        adaptive=args.adaptive,
    )
    print(f"  Resulting frame: {len(df_full):,} rows  cols={list(df_full.columns)}")
    y_full = df_full["label_continuous"].values
    print(f"  label_continuous: mean={y_full.mean():+.5f}  std={y_full.std():.5f}  "
          f"P5={np.percentile(y_full, 5):+.5f}  P95={np.percentile(y_full, 95):+.5f}")

    # 4. Split chronological with gap = window
    print(f"\n[4/6] Splitting (70/15/15, gap={args.window})...")
    df_train, df_val, df_test = split_train_val_test(
        df_full, train_ratio=0.70, val_ratio=0.15, gap=args.window
    )
    print(f"  train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}")

    # 5. Z-score (slope_progressive only; step_k stays raw and is normalized to 0..1 later)
    feature_cols = ["slope_progressive"]
    df_train_n, df_val_n, df_test_n, stats = normalize_features(
        df_train, df_val, df_test, feature_cols
    )
    m, s = stats["slope_progressive"]
    print(f"\n[5/6] Z-score stats (train only): slope_progressive mean={m:+.5f} std={s:.5f}")

    # 6. Make sequences
    print(f"\n[6/6] Building sequences (window={args.window})...")
    out = {}
    extras_descr = []
    for name, df in [("train", df_train_n), ("val", df_val_n), ("test", df_test_n)]:
        seq = make_sequences(df, feature_cols, "label_continuous", window=args.window)
        # seq["X"] shape: (n, window, 1). Chronos = single-channel -> squeeze to (n, window).
        X = seq["X"].squeeze(-1).astype(np.float32)
        y = seq["y"].astype(np.float32)
        out[f"X_{name}"] = X
        out[f"y_{name}"] = y
        out[f"closes_{name}"] = seq["closes"].astype(np.float64)

        # Always save step_k aligned with end-of-window (for per-substep eval)
        step_k_at_end = df["step_k"].values[args.window - 1:].astype(np.int8)
        out[f"step_k_{name}"] = step_k_at_end

        # SKIP CONNECTION : pass last slope value (z-scored global) as extras.
        # Chronos tokenizer mean-scales each window locally and loses absolute
        # amplitude. Without this, the model cannot reproduce FLKS at all.
        # With this, the head can learn at worst the identity yhat ~= alpha * extras.
        extras_parts = []
        if args.include_slope_extra:
            slope_extra = X[:, -1:].copy()  # (n, 1) z-scored slope_progressive[t]
            extras_parts.append(slope_extra)
            if name == "train":
                extras_descr.append("slope_progressive_z[t]")
        if args.use_step_k_extra:
            step_k_z = ((step_k_at_end.astype(np.float32) - 2.5) /
                        max(np.std(np.arange(6, dtype=np.float32)), 1e-8))
            extras_parts.append(step_k_z.reshape(-1, 1))
            if name == "train":
                extras_descr.append("step_k_z")

        if extras_parts:
            extras = np.concatenate(extras_parts, axis=1).astype(np.float32)
            out[f"extras_{name}"] = extras
            extras_summary = (f"  extras_{name}: shape={extras.shape}  "
                              f"mean={extras.mean(0).round(3).tolist()}  "
                              f"std={extras.std(0).round(3).tolist()}")
        else:
            extras_summary = ""

        print(f"  [{name}] X={X.shape} y={y.shape}  step_k={step_k_at_end.shape}  "
              f"(mean y={y.mean():+.5f} std={y.std():.5f})" + extras_summary)

    if extras_descr:
        out["extras_descr"] = np.array(extras_descr, dtype=object)

    # 7. Save .npz
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "csv": args.csv,
        "indicator": args.indicator,
        "tf_minutes": TF_MINUTES,
        "window": args.window,
        "trim": args.trim,
        "adaptive": args.adaptive,
        "use_step_k_extra": args.use_step_k_extra,
        "feature_in_X": "slope_progressive (FLKS @ sub-step k)",
        "label": "label_continuous (Oracle 30min smoothed slope, ffill to 5min)",
        "norm_stats": {k: [float(v[0]), float(v[1])] for k, v in stats.items()},
        "n_train": int(out["X_train"].shape[0]),
        "n_val": int(out["X_val"].shape[0]),
        "n_test": int(out["X_test"].shape[0]),
        "anti_leakage_rts": True,
        "kalman_model": f"FLKS {'AQ-KF' if args.adaptive else 'Standard'} on {args.indicator.upper()} {TF_MINUTES}min",
    }
    np.savez_compressed(out_path, meta=json.dumps(meta), **out)
    print(f"\nSaved {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
