#!/usr/bin/env python3
"""
Préparation complète du dataset ML depuis le CSV 5m historique.

Pipeline:
  1. Load 5m (data_trad/BTCUSD_all_5m.csv, 8.5 ans BTC)
  2. Resample 5m → 30m (via core.resample_ohlcv, bit-à-bit prouvé)
  3. Save CSV 30m (data/raw/BTCUSD_full_30m.csv, pour debug/inspection)
  4. prepare_features_and_labels → DataFrame (6 slopes + labels + close)
  5. split_train_val_test → 3 DataFrames chronologiques avec gap
  6. normalize_features → stats fittées SUR TRAIN UNIQUEMENT
  7. make_sequences pour train/val/test → X, y, closes, dates, indices
  8. Sauvegarde un NPZ unique contenant TOUT ce qu'il faut pour le train

Sortie:
  data/raw/BTCUSD_full_30m.csv                           (CSV debug)
  data/prepared/dataset_{indicator}_{tf}m_full.npz       (dataset ML)

Usage:
    python scripts/prepare_full_data.py
    python scripts/prepare_full_data.py --indicator rsi --tf 30 --window 25
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
    prepare_features_and_labels, split_train_val_test,
    normalize_features, make_sequences,
)

SRC_5M = Path('data_trad/BTCUSD_all_5m.csv')
RAW_DIR = Path('data/raw')
PREP_DIR = Path('data/prepared')

FEATURE_COLS = [f'slope_k{k}' for k in range(1, 7)]


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
    parser.add_argument('--indicator', default='macd',
                        choices=['macd', 'rsi', 'cci'])
    parser.add_argument('--tf', type=int, default=30, choices=[30, 60])
    parser.add_argument('--window', type=int, default=25)
    parser.add_argument('--trim', type=int, default=100)
    parser.add_argument('--train-ratio', type=float, default=0.70)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PREP_DIR.mkdir(parents=True, exist_ok=True)

    tf_label = f'{args.tf}m' if args.tf < 60 else '1h'
    print("=" * 80)
    print(f"PRÉPARATION DATASET ML — {args.indicator.upper()} × {tf_label}  "
          f"source=full  (window={args.window}, trim={args.trim})")
    print("=" * 80)

    if not SRC_5M.exists():
        print(f"❌ Source introuvable: {SRC_5M}")
        return

    # ========== [1] Load 5m ==========
    print(f"\n[1/8] Load 5m ({SRC_5M}) ...")
    df_5m = load_csv(SRC_5M)
    print(f"  {len(df_5m):,} rows  |  {df_5m.index[0]} → {df_5m.index[-1]}")
    years = (df_5m.index[-1] - df_5m.index[0]).total_seconds() / (365.25 * 24 * 3600)
    print(f"  ≈ {years:.1f} années")

    # ========== [2] Resample 5m → TF ==========
    print(f"\n[2/8] Resample 5m → {tf_label} ...")
    df_tf = resample_ohlcv(df_5m, args.tf)
    # Drop bougies TF incomplètes en fin (si 5m s'arrête avant la fin d'une bougie)
    df_tf, n_dropped = drop_incomplete_last(df_tf, df_5m, args.tf)
    print(f"  {len(df_tf):,} rows {tf_label}  (dropped {n_dropped} incomplete)")

    # ========== [3] Save CSV TF (debug) ==========
    csv_path = RAW_DIR / f'BTCUSD_full_{tf_label}.csv'
    df_tf.reset_index().to_csv(csv_path, index=False)
    print(f"\n[3/8] Save CSV debug: {csv_path} ({csv_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # ========== [4] prepare_features_and_labels ==========
    print(f"\n[4/8] prepare_features_and_labels ...")
    data = prepare_features_and_labels(df_tf, df_5m, args.indicator,
                                         args.tf, trim=args.trim)
    print(f"  Shape: {data.shape}  |  colonnes: {list(data.columns)}")

    # Oracle slopes + label au TF complet (non-trimé)
    # Utile pour le backtest qui s'aligne sur df_tf.index
    from src.signal_processing.core import compute_oracle_labels
    print(f"  Computing oracle_labels (full tf, pour backtest) ...")
    oracle_full = compute_oracle_labels(df_tf, args.indicator)
    oracle_slopes_full = oracle_full['slope'].values.astype(np.float64)
    print(f"  oracle_slopes_full shape: {oracle_slopes_full.shape}")

    # ========== [5] split ==========
    print(f"\n[5/8] split_train_val_test (gap={args.window}) ...")
    df_train, df_val, df_test = split_train_val_test(
        data, args.train_ratio, args.val_ratio, gap=args.window)
    print(f"  train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}")

    # ========== [6] normalize ==========
    print(f"\n[6/8] normalize_features (stats from train) ...")
    df_tr_n, df_va_n, df_te_n, stats = normalize_features(
        df_train, df_val, df_test, FEATURE_COLS)
    print(f"  Stats: {FEATURE_COLS[0]} mean={stats[FEATURE_COLS[0]][0]:.4f} "
          f"std={stats[FEATURE_COLS[0]][1]:.4f}")

    # ========== [7] sequences ==========
    print(f"\n[7/8] make_sequences (window={args.window}) ...")
    # Multi-label : on garde binary (train) + continuous (pour analyse)
    seq_tr = make_sequences(df_tr_n, FEATURE_COLS,
                              ['label_binary', 'label_continuous'], args.window)
    seq_va = make_sequences(df_va_n, FEATURE_COLS,
                              ['label_binary', 'label_continuous'], args.window)
    seq_te = make_sequences(df_te_n, FEATURE_COLS,
                              ['label_binary', 'label_continuous'], args.window)
    print(f"  X_train={seq_tr['X'].shape}  "
          f"X_val={seq_va['X'].shape}  X_test={seq_te['X'].shape}")

    # Distribution labels
    for name, seq in [('train', seq_tr), ('val', seq_va), ('test', seq_te)]:
        y = seq['y']['label_binary']
        up = int((y == 1).sum())
        down = int((y == 0).sum())
        print(f"  {name}: UP={up:,} ({up/(up+down)*100:.1f}%)  "
              f"DOWN={down:,} ({down/(up+down)*100:.1f}%)")

    # ========== [8] Sauvegarde NPZ unique ==========
    # Indices dans df_tf (pour reconstruire slopes_from_preds dans le backtest)
    def indices_in_df_tf(seq_dates):
        return np.array([df_tf.index.get_loc(pd.Timestamp(d)) for d in seq_dates])

    train_indices = indices_in_df_tf(seq_tr['dates'])
    val_indices = indices_in_df_tf(seq_va['dates'])
    test_indices = indices_in_df_tf(seq_te['dates'])

    npz_path = PREP_DIR / f'dataset_{args.indicator}_{tf_label}_full.npz'
    print(f"\n[8/8] Sauvegarde NPZ: {npz_path}")
    np.savez(
        npz_path,
        # Train
        X_train=seq_tr['X'].astype(np.float32),
        y_train_binary=seq_tr['y']['label_binary'].astype(np.int64),
        y_train_continuous=seq_tr['y']['label_continuous'].astype(np.float64),
        closes_train=seq_tr['closes'].astype(np.float64),
        dates_train=seq_tr['dates'],
        indices_train=train_indices.astype(np.int64),
        # Val
        X_val=seq_va['X'].astype(np.float32),
        y_val_binary=seq_va['y']['label_binary'].astype(np.int64),
        y_val_continuous=seq_va['y']['label_continuous'].astype(np.float64),
        closes_val=seq_va['closes'].astype(np.float64),
        dates_val=seq_va['dates'],
        indices_val=val_indices.astype(np.int64),
        # Test
        X_test=seq_te['X'].astype(np.float32),
        y_test_binary=seq_te['y']['label_binary'].astype(np.int64),
        y_test_continuous=seq_te['y']['label_continuous'].astype(np.float64),
        closes_test=seq_te['closes'].astype(np.float64),
        dates_test=seq_te['dates'],
        indices_test=test_indices.astype(np.int64),
        # Métadonnées
        feature_cols=np.array(FEATURE_COLS),
        window=args.window,
        trim=args.trim,
        tf_minutes=args.tf,
        indicator=args.indicator,
        source='full',
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        # Stats pour dé-normalisation en prod
        norm_means=np.array([stats[c][0] for c in FEATURE_COLS]),
        norm_stds=np.array([stats[c][1] for c in FEATURE_COLS]),
        # Oracle slopes au TF complet (pour backtest, évite recalcul)
        oracle_slopes_full=oracle_slopes_full,
        # Dates TF complètes (pour alignement backtest, évite reload df_tf)
        df_tf_dates=df_tf.index.values,
        df_tf_closes=df_tf['close'].values.astype(np.float64),
    )
    print(f"  ✅ {npz_path.stat().st_size / 1024 / 1024:.1f} MB sauvé")
    print(f"\nPour entraîner:")
    print(f"  python scripts/train_model.py --indicator {args.indicator} "
          f"--tf {args.tf} --source full")


if __name__ == '__main__':
    main()
