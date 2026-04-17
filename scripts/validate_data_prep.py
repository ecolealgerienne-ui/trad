#!/usr/bin/env python3
"""
Valide les 4 fonctions de préparation de données ML sur MACD × 30m.

Fonctions testées:
  prepare_features_and_labels(df_tf, df_5m, indicator, tf) -> DataFrame
  split_train_val_test(df, train_ratio, val_ratio, gap) -> 3 DataFrames
  normalize_features(df_train, df_val, df_test, feature_cols) -> 3 + stats
  make_sequences(df, feature_cols, label_cols, window) -> dict X/y/closes/dates

Tests:
  [1] prepare_features_and_labels : shape, colonnes, no NaN, labels cohérents
  [2] split : chronologie stricte, no overlap, ratios respectés, gap
  [3] normalize : train mean≈0/std≈1, val/test réversibles, stats OK
  [4] make_sequences : shapes, dtypes, alignement X ↔ y, labels multiples
  [5] Path A vs Path B : X et y identiques entre 5m-resamplé et TF téléchargé
  [6] Reproductibilité : 2 runs consécutifs → sorties identiques

Scope: MACD × 30m.

Usage:
    python scripts/validate_data_prep.py
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

DATA_DIR = Path('data/raw')
TF = 30
TRIM = 100
WINDOW = 25
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TOL_ABS = 1e-10

FEATURE_COLS = [f'slope_k{k}' for k in range(1, 7)]  # V1 : 6 slopes k=1..6


def check(name, passed, detail=""):
    status = "✅" if passed else "❌"
    print(f"  {status} {name}  {detail}")
    return passed


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
    args = parser.parse_args()
    INDICATOR = args.indicator

    print("=" * 80)
    print(f"VALIDATION data preparation — {INDICATOR.upper()} × {TF}m")
    print("=" * 80)

    paths = {
        '5m': DATA_DIR / 'BTCUSD_3months_5m.csv',
        'tf': DATA_DIR / f'BTCUSD_3months_{TF}m.csv',
    }
    df_5m = load_csv(paths['5m'])
    df_tf = load_csv(paths['tf'])
    df_tf, _ = drop_incomplete_last(df_tf, df_5m, TF)
    print(f"\nChargement: 5m={len(df_5m):,}  {TF}m={len(df_tf):,} rows")

    all_ok = True

    # ==========================================================================
    # [1] prepare_features_and_labels
    # ==========================================================================
    print(f"\n[1] prepare_features_and_labels")
    data = prepare_features_and_labels(df_tf, df_5m, INDICATOR, TF, trim=TRIM)

    expected_cols = FEATURE_COLS + ['label_binary', 'label_continuous', 'close']
    all_ok &= check(
        f"  colonnes = {expected_cols}",
        list(data.columns) == expected_cols, "")
    all_ok &= check(
        f"  pas de NaN",
        not data.isna().any().any(), "")
    all_ok &= check(
        f"  taille ≈ len(df_tf) - 2*TRIM",
        len(data) == len(df_tf) - 2 * TRIM,
        f"{len(data)} vs {len(df_tf) - 2 * TRIM}")
    # label_binary doit être cohérent avec label_continuous
    lb_expected = (data['label_continuous'] > 0).astype(int)
    all_ok &= check(
        f"  label_binary == (label_continuous > 0)",
        (data['label_binary'].values == lb_expected.values).all(), "")
    print(f"  Shape: {data.shape}, range label_cont: "
          f"[{data['label_continuous'].min():.4f}, {data['label_continuous'].max():.4f}]")

    # ==========================================================================
    # [2] split_train_val_test
    # ==========================================================================
    print(f"\n[2] split_train_val_test (gap=WINDOW={WINDOW})")
    df_train, df_val, df_test = split_train_val_test(
        data, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, gap=WINDOW)
    print(f"  train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}")

    # Chronologie
    all_ok &= check(
        f"  train.index[-1] < val.index[0]",
        df_train.index[-1] < df_val.index[0],
        f"{df_train.index[-1]} < {df_val.index[0]}")
    all_ok &= check(
        f"  val.index[-1] < test.index[0]",
        df_val.index[-1] < df_test.index[0], "")

    # Gap vérifié
    idx_all = data.index
    n = len(data)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    all_ok &= check(
        f"  taille train = train_end - gap",
        len(df_train) == train_end - WINDOW,
        f"{len(df_train)} vs {train_end - WINDOW}")
    all_ok &= check(
        f"  taille val = val_end - train_end - gap",
        len(df_val) == val_end - train_end - WINDOW,
        f"{len(df_val)} vs {val_end - train_end - WINDOW}")
    all_ok &= check(
        f"  taille test = n - val_end",
        len(df_test) == n - val_end, "")

    # Pas d'overlap d'index
    train_idx = set(df_train.index)
    val_idx = set(df_val.index)
    test_idx = set(df_test.index)
    all_ok &= check(
        f"  no overlap train ∩ val",
        len(train_idx & val_idx) == 0, "")
    all_ok &= check(
        f"  no overlap val ∩ test",
        len(val_idx & test_idx) == 0, "")

    # ==========================================================================
    # [3] normalize_features
    # ==========================================================================
    print(f"\n[3] normalize_features (stats from train only)")
    df_tr_n, df_va_n, df_te_n, stats = normalize_features(
        df_train, df_val, df_test, FEATURE_COLS)

    # Train : mean ≈ 0, std ≈ 1
    for col in FEATURE_COLS:
        m = df_tr_n[col].mean()
        s = df_tr_n[col].std()
        all_ok &= check(
            f"  train[{col}] mean={m:+.2e}, std={s:.4f}",
            abs(m) < 1e-6 and abs(s - 1.0) < 1e-3, "")

    # Réversibilité : val_raw == val_normalized * std_train + mean_train
    for col in FEATURE_COLS:
        mean, std = stats[col]
        reversed_val = df_va_n[col] * std + mean
        max_diff = np.max(np.abs(reversed_val.values - df_val[col].values))
        all_ok &= check(
            f"  réversibilité val[{col}]: max |orig - reversed| = {max_diff:.2e}",
            max_diff < 1e-10, "")

    # stats utilise bien train
    for col in FEATURE_COLS:
        mean_stat, std_stat = stats[col]
        mean_train = df_train[col].mean()
        std_train = df_train[col].std()
        all_ok &= check(
            f"  stats[{col}] == train stats: mean={mean_stat:.4f}, std={std_stat:.4f}",
            abs(mean_stat - mean_train) < 1e-10
            and abs(std_stat - std_train) < 1e-10, "")

    # ==========================================================================
    # [4] make_sequences
    # ==========================================================================
    print(f"\n[4] make_sequences (window={WINDOW})")

    # Test avec label unique (string)
    seq_single = make_sequences(
        df_tr_n, FEATURE_COLS, 'label_binary', window=WINDOW)
    X = seq_single['X']
    y = seq_single['y']
    closes = seq_single['closes']
    dates = seq_single['dates']

    expected_n_seq = len(df_tr_n) - WINDOW + 1
    all_ok &= check(
        f"  X.shape = (n_seq, window, n_feat)",
        X.shape == (expected_n_seq, WINDOW, len(FEATURE_COLS)),
        f"got {X.shape}")
    all_ok &= check(
        f"  X dtype float32",
        X.dtype == np.float32, f"got {X.dtype}")
    all_ok &= check(
        f"  y.shape = (n_seq,)",
        y.shape == (expected_n_seq,), f"got {y.shape}")
    all_ok &= check(
        f"  y dtype int64 (label_binary)",
        y.dtype == np.int64, f"got {y.dtype}")
    all_ok &= check(
        f"  closes.shape = (n_seq,)",
        closes.shape == (expected_n_seq,), "")
    all_ok &= check(
        f"  dates.shape = (n_seq,)",
        dates.shape == (expected_n_seq,), "")
    all_ok &= check(
        f"  pas de NaN dans X",
        not np.any(np.isnan(X)), "")

    # Alignement X ↔ y : pour i arbitraire, y[i] == df.label_binary à i+window-1
    i = min(100, expected_n_seq - 1)
    ts_y = dates[i]  # = df_tr_n.index[i + window - 1]
    label_from_df = df_tr_n['label_binary'].iloc[i + WINDOW - 1]
    all_ok &= check(
        f"  alignement X↔y at i={i}: y[i]={y[i]} vs df[i+w-1]={label_from_df}",
        y[i] == label_from_df, "")

    # Test avec labels multiples (liste)
    seq_multi = make_sequences(
        df_tr_n, FEATURE_COLS, ['label_binary', 'label_continuous'], window=WINDOW)
    all_ok &= check(
        f"  labels multiples: y est un dict",
        isinstance(seq_multi['y'], dict), "")
    all_ok &= check(
        f"  y['label_binary'] dtype int64",
        seq_multi['y']['label_binary'].dtype == np.int64, "")
    all_ok &= check(
        f"  y['label_continuous'] dtype float64",
        seq_multi['y']['label_continuous'].dtype == np.float64, "")

    # Distribution labels
    up_train = int((seq_multi['y']['label_binary'] == 1).sum())
    n_train = len(seq_multi['y']['label_binary'])
    print(f"  distribution train: UP={up_train:,} ({up_train/n_train*100:.1f}%)  "
          f"DOWN={n_train-up_train:,} ({(n_train-up_train)/n_train*100:.1f}%)")

    # ==========================================================================
    # [5] Path A vs Path B
    # ==========================================================================
    print(f"\n[5] Path A (5m resamplé) vs Path B (TF téléchargé)")
    df_tf_R = resample_ohlcv(df_5m, TF)
    df_tf_R, _ = drop_incomplete_last(df_tf_R, df_5m, TF)

    # Chemin A complet
    data_A = prepare_features_and_labels(df_tf_R, df_5m, INDICATOR, TF, trim=TRIM)
    tr_A, va_A, te_A = split_train_val_test(data_A, TRAIN_RATIO, VAL_RATIO, WINDOW)
    tr_A_n, va_A_n, te_A_n, _ = normalize_features(tr_A, va_A, te_A, FEATURE_COLS)
    seq_A = make_sequences(tr_A_n, FEATURE_COLS, 'label_binary', WINDOW)

    # Chemin B (= déjà calculé plus haut = seq_single sur df_tr_n)
    seq_B = seq_single

    all_ok &= check(
        f"  X identiques : shapes",
        seq_A['X'].shape == seq_B['X'].shape, "")
    max_diff_X = np.max(np.abs(seq_A['X'] - seq_B['X']))
    all_ok &= check(
        f"  max |A.X - B.X| = {max_diff_X:.2e}",
        max_diff_X < TOL_ABS, "")
    all_ok &= check(
        f"  y identiques",
        np.array_equal(seq_A['y'], seq_B['y']), "")
    max_diff_c = np.max(np.abs(seq_A['closes'] - seq_B['closes']))
    all_ok &= check(
        f"  max |A.closes - B.closes| = {max_diff_c:.2e}",
        max_diff_c < TOL_ABS, "")

    # ==========================================================================
    # [6] Reproductibilité
    # ==========================================================================
    print(f"\n[6] Reproductibilité (2 runs identiques)")
    data2 = prepare_features_and_labels(df_tf, df_5m, INDICATOR, TF, trim=TRIM)
    tr2, va2, te2 = split_train_val_test(data2, TRAIN_RATIO, VAL_RATIO, WINDOW)
    tr2_n, va2_n, te2_n, _ = normalize_features(tr2, va2, te2, FEATURE_COLS)
    seq2 = make_sequences(tr2_n, FEATURE_COLS, 'label_binary', WINDOW)
    all_ok &= check(
        f"  X identique entre 2 runs",
        np.array_equal(seq_single['X'], seq2['X']), "")
    all_ok &= check(
        f"  y identique entre 2 runs",
        np.array_equal(seq_single['y'], seq2['y']), "")

    # Verdict
    print("\n" + "=" * 80)
    print(f"VERDICT : {'✅ TOUS TESTS PASS' if all_ok else '❌ AU MOINS UN ÉCHEC'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
