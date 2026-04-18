#!/usr/bin/env python3
"""
Préparation dataset ML "progressive" (résolution 5min).

Différence vs prepare_full_data.py :
  - Historique (full) : 1 ligne par bougie TF, features slope_k1..k6 constantes
  - Progressive       : 6 lignes par bougie 30m (1 par sous-pas 5min), features
                        évoluant à chaque 5min selon step_k, labels ffill.

Pipeline :
  1. Load 5m historique (data_trad/BTCUSD_all_5m.csv, 8.4 ans BTC)
  2. Resample 5m → 30m (via core.resample_ohlcv)
  3. prepare_features_and_labels_progressive
     → DataFrame 5min : [slope_progressive, step_k, label_binary, label_continuous, close]
  4. split_train_val_test chronologique (avec gap)
  5. normalize_features sur train uniquement (slope_progressive uniquement)
     step_k laissé brut (catégoriel, 0..5)
  6. Sauvegarde NPZ unique (pas de sequences : format tabulaire pour XGBoost)

Sortie :
  data/prepared/dataset_{indicator}_{tf}_full_progressive.npz

Usage :
  python scripts/prepare_progressive_data.py                     # tout l'historique
  python scripts/prepare_progressive_data.py --days 180          # 6 derniers mois
  python scripts/prepare_progressive_data.py --indicator macd --tf 30 --days 180
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
    prepare_features_and_labels_progressive,
    split_train_val_test, normalize_features,
    compute_oracle_labels,
)

SRC_5M = Path('data_trad/BTCUSD_all_5m.csv')
RAW_DIR = Path('data/raw')
PREP_DIR = Path('data/prepared')

# Features V1 progressive : 1 slope + 1 step_k (catégoriel)
FEATURE_COLS_NORM = ['slope_progressive']       # à z-scorer
FEATURE_COLS_RAW = ['step_k']                    # laissé brut (0..n_sub-1)
FEATURE_COLS = FEATURE_COLS_NORM + FEATURE_COLS_RAW


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
    parser.add_argument('--days', type=int, default=0,
                        help='Derniers N jours à utiliser (0 = tout, default 0)')
    parser.add_argument('--trim', type=int, default=100,
                        help='Bougies TF à retirer début ET fin (warm-up Kalman)')
    parser.add_argument('--train-ratio', type=float, default=0.70)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--gap-5m', type=int, default=0,
                        help='Gap en lignes 5min entre splits (default 0)')
    parser.add_argument('--adaptive', action='store_true',
                        help='Utiliser AQ-KF (Adaptive Q Kalman Filter) au forward pass '
                             '→ meilleure détection des transitions. Oracle RTS inchangé.')
    parser.add_argument('--slope-lag', type=int, default=1, choices=[0, 1],
                        help='Décalage pente oracle. 1 (legacy) = pente t-1 vs t-2. '
                             '0 = pente t vs t-1 (gain de TF en précocité).')
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PREP_DIR.mkdir(parents=True, exist_ok=True)

    tf_label = f'{args.tf}m' if args.tf < 60 else '1h'
    filter_tag_disp = 'AQ-KF adaptive' if args.adaptive else 'standard Kalman'
    slope_tag_disp = 'slope_lag=0 (récente t/t-1)' if args.slope_lag == 0 else 'slope_lag=1 (legacy t-1/t-2)'
    print("=" * 80)
    print(f"PRÉPARATION DATASET ML — PROGRESSIVE — {args.indicator.upper()} × {tf_label}")
    print(f"  filter={filter_tag_disp}  oracle={slope_tag_disp}")
    print(f"  trim={args.trim}  train={args.train_ratio}  "
          f"val={args.val_ratio}  gap_5m={args.gap_5m}")
    print("=" * 80)

    if not SRC_5M.exists():
        print(f"❌ Source introuvable: {SRC_5M}")
        return

    # ========== [1] Load 5m ==========
    print(f"\n[1/7] Load 5m ({SRC_5M}) ...")
    df_5m_full = load_csv(SRC_5M)
    print(f"  Full: {len(df_5m_full):,} rows  |  "
          f"{df_5m_full.index[0]} → {df_5m_full.index[-1]}")

    if args.days > 0:
        end_date = df_5m_full.index[-1]
        start_date = end_date - pd.Timedelta(days=args.days)
        df_5m = df_5m_full.loc[df_5m_full.index >= start_date].copy()
        print(f"  Filter last {args.days} days: {len(df_5m):,} rows  |  "
              f"{df_5m.index[0]} → {df_5m.index[-1]}")
    else:
        df_5m = df_5m_full
        years = (df_5m.index[-1] - df_5m.index[0]).total_seconds() / (365.25 * 24 * 3600)
        print(f"  ≈ {years:.1f} années")

    # ========== [2] Resample 5m → TF ==========
    print(f"\n[2/7] Resample 5m → {tf_label} ...")
    df_tf = resample_ohlcv(df_5m, args.tf)
    df_tf, n_dropped = drop_incomplete_last(df_tf, df_5m, args.tf)
    print(f"  {len(df_tf):,} bougies {tf_label}  (dropped {n_dropped} incomplete)")

    # ========== [3] Progressive features + labels ==========
    print(f"\n[3/7] prepare_features_and_labels_progressive "
          f"(adaptive={args.adaptive}, slope_lag={args.slope_lag}) ...")
    data = prepare_features_and_labels_progressive(
        df_tf, df_5m, args.indicator, args.tf, trim=args.trim,
        adaptive=args.adaptive, slope_lag=args.slope_lag)
    print(f"  Shape: {data.shape}  |  colonnes: {list(data.columns)}")
    print(f"  Plage: {data.index[0]} → {data.index[-1]}")
    print(f"  Distribution step_k: {dict(data['step_k'].value_counts().sort_index())}")
    print(f"  Distribution label_binary: "
          f"UP={(data['label_binary']==1).sum():,} / "
          f"DOWN={(data['label_binary']==0).sum():,}")

    # ========== [4] Oracle slopes full TF (pour backtest) ==========
    print(f"\n[4/7] compute_oracle_labels (full tf, pour backtest, "
          f"slope_lag={args.slope_lag}) ...")
    oracle_full = compute_oracle_labels(df_tf, args.indicator,
                                          slope_lag=args.slope_lag)
    oracle_slopes_full = oracle_full['slope'].values.astype(np.float64)
    print(f"  oracle_slopes_full shape: {oracle_slopes_full.shape}")

    # ========== [5] Split ==========
    print(f"\n[5/7] split_train_val_test (gap_5m={args.gap_5m}) ...")
    df_train, df_val, df_test = split_train_val_test(
        data, args.train_ratio, args.val_ratio, gap=args.gap_5m)
    print(f"  train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}")
    print(f"  train: {df_train.index[0]} → {df_train.index[-1]}")
    print(f"  val:   {df_val.index[0]} → {df_val.index[-1]}")
    print(f"  test:  {df_test.index[0]} → {df_test.index[-1]}")

    # ========== [6] Normalize (slope_progressive uniquement) ==========
    print(f"\n[6/7] normalize_features (z-score sur {FEATURE_COLS_NORM}, "
          f"stats from train) ...")
    df_tr_n, df_va_n, df_te_n, stats = normalize_features(
        df_train, df_val, df_test, FEATURE_COLS_NORM)
    for col in FEATURE_COLS_NORM:
        print(f"  {col}: mean={stats[col][0]:.6f}  std={stats[col][1]:.6f}")

    # ========== [7] Sauvegarde NPZ ==========
    # Indices dans df_5m (pour backtest progressive)
    def indices_in_df_5m(df_split):
        return np.array([df_5m.index.get_loc(ts) for ts in df_split.index],
                        dtype=np.int64)

    print(f"\n[7/7] Build arrays ...")
    # X : features tabulaires (pas de sequences)
    def to_x(df_split):
        return df_split[FEATURE_COLS].values.astype(np.float32)

    def to_y_bin(df_split):
        return df_split['label_binary'].values.astype(np.int64)

    def to_y_cont(df_split):
        return df_split['label_continuous'].values.astype(np.float64)

    def to_closes(df_split):
        return df_split['close'].values.astype(np.float64)

    def to_dates(df_split):
        return df_split.index.values

    # Nom du NPZ : si --days > 0, on tag avec _<N>d, sinon _full
    # Suffixes optionnels (cumulés, ordre fixe : _adaptive puis _lag<N>)
    period_tag = f'{args.days}d' if args.days > 0 else 'full'
    adaptive_suffix = '_adaptive' if args.adaptive else ''
    lag_suffix = f'_lag{args.slope_lag}' if args.slope_lag != 1 else ''
    full_suffix = adaptive_suffix + lag_suffix
    npz_path = PREP_DIR / f'dataset_{args.indicator}_{tf_label}_{period_tag}_progressive{full_suffix}.npz'
    print(f"  Sauvegarde NPZ: {npz_path}")
    np.savez(
        npz_path,
        # Train
        X_train=to_x(df_tr_n),
        y_train_binary=to_y_bin(df_tr_n),
        y_train_continuous=to_y_cont(df_tr_n),
        closes_train=to_closes(df_tr_n),
        dates_train=to_dates(df_tr_n),
        indices_train=indices_in_df_5m(df_tr_n),
        # Val
        X_val=to_x(df_va_n),
        y_val_binary=to_y_bin(df_va_n),
        y_val_continuous=to_y_cont(df_va_n),
        closes_val=to_closes(df_va_n),
        dates_val=to_dates(df_va_n),
        indices_val=indices_in_df_5m(df_va_n),
        # Test
        X_test=to_x(df_te_n),
        y_test_binary=to_y_bin(df_te_n),
        y_test_continuous=to_y_cont(df_te_n),
        closes_test=to_closes(df_te_n),
        dates_test=to_dates(df_te_n),
        indices_test=indices_in_df_5m(df_te_n),
        # Meta
        feature_cols=np.array(FEATURE_COLS),
        feature_cols_norm=np.array(FEATURE_COLS_NORM),
        feature_cols_raw=np.array(FEATURE_COLS_RAW),
        tf_minutes=args.tf,
        indicator=args.indicator,
        source=f'{period_tag}_progressive',
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        gap_5m=args.gap_5m,
        trim=args.trim,
        # Stats de normalisation
        norm_means=np.array([stats[c][0] for c in FEATURE_COLS_NORM]),
        norm_stds=np.array([stats[c][1] for c in FEATURE_COLS_NORM]),
        # Oracle + df_tf (pour backtest)
        oracle_slopes_full=oracle_slopes_full,
        df_tf_dates=df_tf.index.values,
        df_tf_closes=df_tf['close'].values.astype(np.float64),
        # df_5m (pour backtest progressive : exec à close_5m[i+1])
        df_5m_dates=df_5m.index.values,
        df_5m_closes=df_5m['close'].values.astype(np.float64),
    )
    size_mb = npz_path.stat().st_size / 1024 / 1024
    print(f"  ✅ {size_mb:.1f} MB sauvé")
    print(f"\nPour entraîner:")
    print(f"  python scripts/train_progressive.py --indicator {args.indicator} "
          f"--tf {args.tf}")


if __name__ == '__main__':
    main()
