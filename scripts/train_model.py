#!/usr/bin/env python3
"""
Entraîne un XGBoost sur les features FLKS V1 (6 slopes k=1..6).

Pipeline:
  1. Charge 5m + TF téléchargés
  2. prepare_features_and_labels → DataFrame (6 slopes + label_binary + close)
  3. split_train_val_test (gap = window)
  4. normalize_features (stats from train only)
  5. make_sequences (X shape (n, window, 6), y = label_binary)
  6. XGBoost.fit sur X_flat (n, window*6) avec early stopping sur val
  7. Prédit probas sur test
  8. Sauvegarde modèle + NPZ (pour backtest_model.py)

Sorties:
  models/xgb_{indicator}_{tf}m.json
  data/prepared/preds_{indicator}_{tf}m.npz

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --indicator rsi --tf 60
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import (
    load_csv,
    prepare_features_and_labels, split_train_val_test,
    normalize_features, make_sequences,
)

DATA_DIR = Path('data/raw')
OUT_DIR = Path('data/prepared')
MODELS_DIR = Path('models')

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
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    tf_label = f'{args.tf}m' if args.tf < 60 else '1h'
    print("=" * 80)
    print(f"TRAIN XGBoost — {args.indicator.upper()} × {tf_label}  "
          f"(window={args.window}, trim={args.trim})")
    print("=" * 80)

    # [1] Load
    print("\n[1/6] Load data ...")
    df_5m = load_csv(DATA_DIR / 'BTCUSD_3months_5m.csv')
    df_tf = load_csv(DATA_DIR / f'BTCUSD_3months_{tf_label}.csv')
    df_tf, _ = drop_incomplete_last(df_tf, df_5m, args.tf)
    print(f"  5m: {len(df_5m):,}  |  {tf_label}: {len(df_tf):,}")

    # [2] Prepare
    print("\n[2/6] prepare_features_and_labels ...")
    data = prepare_features_and_labels(df_tf, df_5m, args.indicator,
                                         args.tf, trim=args.trim)
    print(f"  Shape: {data.shape}  |  colonnes: {list(data.columns)}")

    # [3] Split (gap = window)
    print(f"\n[3/6] split_train_val_test (gap={args.window}) ...")
    df_train, df_val, df_test = split_train_val_test(
        data, args.train_ratio, args.val_ratio, gap=args.window)
    print(f"  train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}")

    # [4] Normalize
    print("\n[4/6] normalize_features (stats from train) ...")
    df_tr_n, df_va_n, df_te_n, stats = normalize_features(
        df_train, df_val, df_test, FEATURE_COLS)

    # [5] Sequences
    print(f"\n[5/6] make_sequences (window={args.window}) ...")
    seq_tr = make_sequences(df_tr_n, FEATURE_COLS, 'label_binary', args.window)
    seq_va = make_sequences(df_va_n, FEATURE_COLS, 'label_binary', args.window)
    seq_te = make_sequences(df_te_n, FEATURE_COLS, 'label_binary', args.window)
    X_tr, y_tr = seq_tr['X'], seq_tr['y']
    X_va, y_va = seq_va['X'], seq_va['y']
    X_te, y_te = seq_te['X'], seq_te['y']
    print(f"  X_train={X_tr.shape}  X_val={X_va.shape}  X_test={X_te.shape}")

    # Distribution labels
    for name, y in [('train', y_tr), ('val', y_va), ('test', y_te)]:
        up = int((y == 1).sum())
        down = int((y == 0).sum())
        print(f"  {name}: UP={up:,} ({up/(up+down)*100:.1f}%) "
              f"DOWN={down:,} ({down/(up+down)*100:.1f}%)")

    # Flatten pour XGBoost (pas de notion de séquence native)
    X_tr_flat = X_tr.reshape(len(X_tr), -1)
    X_va_flat = X_va.reshape(len(X_va), -1)
    X_te_flat = X_te.reshape(len(X_te), -1)
    n_feat = len(FEATURE_COLS)
    print(f"  Flatten: X_train_flat={X_tr_flat.shape} "
          f"(window*n_feat = {args.window}*{n_feat} = {args.window*n_feat})")

    # [6] XGBoost
    print("\n[6/6] Train XGBoost ...")
    try:
        import xgboost as xgb
    except ImportError:
        print("❌ xgboost non installé. pip install xgboost")
        return

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        random_state=args.seed, eval_metric='logloss',
        early_stopping_rounds=20, n_jobs=-1,
    )
    model.fit(X_tr_flat, y_tr, eval_set=[(X_va_flat, y_va)], verbose=50)

    # Metrics
    print("\n" + "=" * 80)
    print("MÉTRIQUES")
    print("=" * 80)
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    for name, Xf, y in [('TRAIN', X_tr_flat, y_tr),
                         ('VAL', X_va_flat, y_va),
                         ('TEST', X_te_flat, y_te)]:
        preds = model.predict(Xf)
        probas = model.predict_proba(Xf)[:, 1]
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds)
        auc = roc_auc_score(y, probas)
        print(f"  {name:<6} acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    # Feature importance (Top 10)
    print("\nTop 10 feature importance:")
    importances = model.feature_importances_
    # Reconstruire names: feature × timestep
    flat_names = [f'{feat}_t-{args.window - 1 - ts}'
                  for ts in range(args.window) for feat in FEATURE_COLS]
    top_idx = np.argsort(importances)[::-1][:10]
    for i in top_idx:
        print(f"  {flat_names[i]:<25} {importances[i]:.4f}")

    # Save model
    model_path = MODELS_DIR / f'xgb_{args.indicator}_{tf_label}.json'
    model.save_model(model_path)
    print(f"\n✅ Modèle sauvé: {model_path}")

    # Save predictions NPZ (for backtest_model.py)
    # Indices dans df_tf pour chaque sample test
    # Dans make_sequences: dates[i] = df_te_n.index[i + window - 1]
    test_preds_proba = model.predict_proba(X_te_flat)[:, 1]
    test_dates = seq_te['dates']  # ndarray datetime64
    test_closes = seq_te['closes']
    # Reconstruction des indices dans df_tf
    test_indices = np.array([df_tf.index.get_loc(pd.Timestamp(d)) for d in test_dates])

    npz_path = OUT_DIR / f'preds_{args.indicator}_{tf_label}.npz'
    np.savez(npz_path,
             test_preds_proba=test_preds_proba.astype(np.float64),
             test_y_true=y_te.astype(np.int64),
             test_dates=test_dates,
             test_closes=test_closes.astype(np.float64),
             test_indices=test_indices.astype(np.int64),
             indicator=args.indicator,
             tf_minutes=args.tf,
             window=args.window,
             trim=args.trim,
             train_ratio=args.train_ratio,
             val_ratio=args.val_ratio,
             )
    print(f"✅ Predictions sauvées: {npz_path}")
    print(f"   test_preds_proba: {test_preds_proba.shape}  "
          f"test_indices: {test_indices.shape}")


if __name__ == '__main__':
    main()
