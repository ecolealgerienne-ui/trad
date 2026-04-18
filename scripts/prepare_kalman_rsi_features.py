#!/usr/bin/env python3
"""
Prépare les features Kalman AQ-KF sur RSI pour clustering non-supervisé.

Pipeline :
  1. Charge CSV BTCUSD 5min → resample 30m
  2. Calcule RSI sur bougies 30m
  3. Applique AQ-KF (forward_filter_30m_adaptive) sur le RSI
  4. Extrait 4 features :
       - position = x_filt[:,0]       (RSI filtré)
       - velocity = x_filt[:,1]       (pente filtrée)
       - P_pos    = P_filt[:,0,0]     (variance posterior position)
       - P_vel    = P_filt[:,1,1]     (variance posterior velocity)
  5. Trim edges (warmup Kalman)
  6. Split chronologique (70/15/15, cohérent avec pipeline progressif)
  7. Sauvegarde NPZ avec features + dates/closes 30m + df_5m pour backtest

⚠️ Note : Q_adaptive NON exposé (core.forward_filter_30m_adaptive inchangé
par décision utilisateur — on utilise uniquement x_filt et P_filt).

Usage :
    python scripts/prepare_kalman_rsi_features.py
    python scripts/prepare_kalman_rsi_features.py --tf 30 --trim 100
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import (
    load_csv, resample_ohlcv,
    calculate_rsi,
    forward_filter_30m_adaptive,
)

SRC_5M = Path('data_trad/BTCUSD_all_5m.csv')
PREP_DIR = Path('data/prepared')


def drop_incomplete_last(df_tf, df_5m, tf_minutes):
    expected = tf_minutes // 5
    drop_count = 0
    for ts in reversed(df_tf.index):
        end = ts + pd.Timedelta(minutes=tf_minutes)
        mask = (df_5m.index >= ts) & (df_5m.index < end)
        if mask.sum() < expected:
            drop_count += 1
        else:
            break
    if drop_count > 0:
        df_tf = df_tf.iloc[:-drop_count]
    return df_tf, drop_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', type=int, default=30, choices=[30, 60])
    parser.add_argument('--trim', type=int, default=100,
                        help='Bougies TF à retirer début ET fin (warmup Kalman)')
    parser.add_argument('--train-ratio', type=float, default=0.70)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    args = parser.parse_args()

    tf_label = f'{args.tf}m' if args.tf < 60 else '1h'
    print("=" * 80)
    print(f"PRÉPARATION FEATURES KALMAN AQ-KF sur RSI × {tf_label}")
    print(f"  trim={args.trim}  train={args.train_ratio}  val={args.val_ratio}")
    print("=" * 80)

    PREP_DIR.mkdir(parents=True, exist_ok=True)
    if not SRC_5M.exists():
        print(f"❌ Source introuvable : {SRC_5M}")
        return

    # ========== [1] Load + resample ==========
    print(f"\n[1/6] Load 5m ({SRC_5M}) + resample → {tf_label} ...")
    df_5m = load_csv(SRC_5M)
    df_tf = resample_ohlcv(df_5m, args.tf)
    df_tf, n_dropped = drop_incomplete_last(df_tf, df_5m, args.tf)
    print(f"   df_5m: {len(df_5m):,} rows  |  df_tf: {len(df_tf):,} bougies "
          f"(dropped {n_dropped} incomplete)")

    # ========== [2] RSI ==========
    print(f"\n[2/6] Calcul RSI sur df_tf ...")
    rsi = calculate_rsi(df_tf)  # np array
    n_nan = int(np.isnan(rsi).sum())
    print(f"   {len(rsi):,} valeurs RSI  ({n_nan} NaN en warmup)")

    # ========== [3] AQ-KF forward filter ==========
    print(f"\n[3/6] forward_filter_30m_adaptive (AQ-KF sur RSI) ...")
    x_filt, P_filt, x_pred, P_pred, C_gains = forward_filter_30m_adaptive(rsi)
    print(f"   x_filt shape: {x_filt.shape}  |  P_filt shape: {P_filt.shape}")

    # ========== [4] Extraction des 4 features ==========
    print(f"\n[4/6] Extraction features : position, velocity, P_pos, P_vel")
    features = np.column_stack([
        x_filt[:, 0],     # position (RSI filtré)
        x_filt[:, 1],     # velocity (pente filtrée)
        P_filt[:, 0, 0],  # P_pos (variance posterior position)
        P_filt[:, 1, 1],  # P_vel (variance posterior velocity)
    ]).astype(np.float64)
    feature_cols = ['position', 'velocity', 'P_pos', 'P_vel']
    print(f"   features shape: {features.shape}")
    # Stats descriptives
    for i, name in enumerate(feature_cols):
        col = features[:, i]
        valid = ~np.isnan(col)
        if valid.sum() > 0:
            print(f"   {name:<10}  mean={col[valid].mean():+.6f}  "
                  f"std={col[valid].std():.6f}  "
                  f"min={col[valid].min():+.4f}  max={col[valid].max():+.4f}")

    # Check NaN
    nan_mask = np.isnan(features).any(axis=1)
    n_nan_rows = int(nan_mask.sum())
    if n_nan_rows > 0:
        print(f"   ⚠️ {n_nan_rows} rows avec NaN (warmup)")

    # ========== [5] Trim edges ==========
    print(f"\n[5/6] Trim {args.trim} au début et à la fin de df_tf")
    n_tf = len(df_tf)
    start = args.trim
    end = n_tf - args.trim
    features_trim = features[start:end]
    dates_tf = df_tf.index.values[start:end]
    closes_tf = df_tf['close'].values[start:end]
    rsi_trim = rsi[start:end]

    # Re-check NaN après trim
    nan_after = int(np.isnan(features_trim).any(axis=1).sum())
    print(f"   Après trim : {len(features_trim):,} bougies  ({nan_after} NaN restants)")

    # ========== [6] Split chronologique ==========
    print(f"\n[6/6] Split chronologique {args.train_ratio}/{args.val_ratio}/{1-args.train_ratio-args.val_ratio:.2f} ...")
    n = len(features_trim)
    train_end = int(n * args.train_ratio)
    val_end = int(n * (args.train_ratio + args.val_ratio))

    def split_arr(arr):
        return arr[:train_end], arr[train_end:val_end], arr[val_end:]

    feat_tr, feat_va, feat_te = split_arr(features_trim)
    dates_tr, dates_va, dates_te = split_arr(dates_tf)
    closes_tr, closes_va, closes_te = split_arr(closes_tf)
    rsi_tr, rsi_va, rsi_te = split_arr(rsi_trim)

    print(f"   Train : {len(feat_tr):,} bougies  "
          f"{pd.Timestamp(dates_tr[0])} → {pd.Timestamp(dates_tr[-1])}")
    print(f"   Val   : {len(feat_va):,} bougies  "
          f"{pd.Timestamp(dates_va[0])} → {pd.Timestamp(dates_va[-1])}")
    print(f"   Test  : {len(feat_te):,} bougies  "
          f"{pd.Timestamp(dates_te[0])} → {pd.Timestamp(dates_te[-1])}")

    # Sauvegarde
    npz_path = PREP_DIR / f'kalman_rsi_features_{tf_label}.npz'
    np.savez(
        npz_path,
        # Features 30m par split
        features_train=feat_tr.astype(np.float64),
        features_val=feat_va.astype(np.float64),
        features_test=feat_te.astype(np.float64),
        # Métadonnées features
        feature_cols=np.array(feature_cols),
        # Dates 30m (pour propagation 5min downstream)
        dates_train_tf=dates_tr,
        dates_val_tf=dates_va,
        dates_test_tf=dates_te,
        # Closes 30m (backtest 30m si besoin)
        closes_train_tf=closes_tr.astype(np.float64),
        closes_val_tf=closes_va.astype(np.float64),
        closes_test_tf=closes_te.astype(np.float64),
        # RSI brut (pour debug/visualisation)
        rsi_train=rsi_tr.astype(np.float64),
        rsi_val=rsi_va.astype(np.float64),
        rsi_test=rsi_te.astype(np.float64),
        # df_tf complet (pour alignement 5min via t_ref)
        df_tf_dates=df_tf.index.values,
        df_tf_closes=df_tf['close'].values.astype(np.float64),
        # df_5m complet (pour backtest 5min cohérent avec pipeline progressif)
        df_5m_dates=df_5m.index.values,
        df_5m_closes=df_5m['close'].values.astype(np.float64),
        # Métadonnées
        tf_minutes=args.tf,
        trim=args.trim,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(f"\n✅ Sauvé : {npz_path}  "
          f"({npz_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"\nPour cluster :")
    print(f"  python scripts/cluster_kalman_rsi.py --npz {npz_path}")


if __name__ == '__main__':
    main()
