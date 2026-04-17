#!/usr/bin/env python3
"""
XGBoost training — drop-in replacement for CNN-LSTM.

IDENTICAL pipeline to train_multitf_aqkf.py:
  - Same CSV, same features, same split, same normalization
  - Same sequences (25 steps × N features)
  - Sequences FLATTENED for XGBoost (25×3 = 75 columns)
  - Same NPZ output format for analyze_predictions_aqkf.py

Usage:
    python src/train_xgboost_aqkf.py --indicator macd --timeframe 30m
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
import argparse
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from constants import PREPARED_DATA_DIR

# Same defaults as train_multitf_aqkf.py
WINDOW = 25
SEED = 42

ASSET_CSV_MAP = {
    'BTC': 'BTCUSD', 'ETH': 'ETHUSD', 'BNB': 'BNBUSD',
    'ADA': 'ADAUSD', 'LTC': 'LTCUSD',
}


# =============================================================================
# REUSE EXACT SAME DATA PIPELINE FROM train_multitf_aqkf.py
# =============================================================================

def find_csv(asset_name, indicator):
    base = ASSET_CSV_MAP[asset_name]
    candidates = [
        f'{PREPARED_DATA_DIR}/{base}_multitf_macd_rsi_cci.csv',
        f'{PREPARED_DATA_DIR}/{base}_multitf_{indicator}.csv',
        f'{PREPARED_DATA_DIR}/{base}_multitf.csv',
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    raise FileNotFoundError(f"No CSV found for {asset_name}. Tried: {candidates}")


def load_asset_data(asset_name, indicator, timeframe, crossfeat=False, target_type='binary'):
    csv_path = find_csv(asset_name, indicator)
    df = pd.read_csv(csv_path, parse_dates=['datetime']).set_index('datetime').sort_index()

    if crossfeat:
        all_indicators = ['macd', 'rsi', 'cci']
        feature_cols = []
        for ind in all_indicators:
            feature_cols.append(f'{ind}_30m_live')
            feature_cols.append(f'{ind}_30m_filtered')
        if timeframe == '1h':
            for ind in all_indicators:
                feature_cols.append(f'{ind}_1h_live')
                feature_cols.append(f'{ind}_1h_filtered')
    else:
        feature_cols = [f'{indicator}_{timeframe}_live', f'{indicator}_{timeframe}_filtered']
        vel_col = f'{indicator}_{timeframe}_velocity'
        if vel_col in df.columns:
            feature_cols.append(vel_col)

    if target_type == 'continuous':
        label_col = f'oracle_slope_{indicator}_{timeframe}'
    else:
        label_col = f'oracle_label_{indicator}_{timeframe}'

    missing = [c for c in feature_cols + [label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    n_features = len(feature_cols)
    result = df[feature_cols + [label_col]].copy()
    new_col_names = [f'feature_{i}' for i in range(n_features)] + ['label']
    result.columns = new_col_names

    n_before = len(result)
    result = result.dropna()
    n_dropped = n_before - len(result)
    logger.info(f"  {asset_name}: {len(result):,} rows, {n_features} features (dropped {n_dropped:,} NaN)")
    return result


def split_chronological(df, train_ratio=0.70, val_ratio=0.15, gap=WINDOW):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end - gap], df.iloc[train_end:val_end - gap], df.iloc[val_end:]


def compute_norm_stats(df_train, feature_cols):
    stats = {}
    for col in feature_cols:
        mean = df_train[col].mean()
        std = df_train[col].std()
        if std < 1e-10:
            std = 1.0
        stats[col] = {'mean': float(mean), 'std': float(std)}
    return stats


def apply_norm(df, stats):
    df = df.copy()
    for col, s in stats.items():
        if col in df.columns:
            df[col] = (df[col] - s['mean']) / s['std']
    return df


def create_sequences(df, window=WINDOW, target_type='binary'):
    feat_cols = [c for c in df.columns if c.startswith('feature_')]
    features = df[feat_cols].values.astype(np.float32)
    if target_type == 'continuous':
        labels = df['label'].values.astype(np.float32)
    else:
        labels = df['label'].values.astype(np.int64)

    n = len(df)
    n_feat = features.shape[1]
    if n < window:
        return np.empty((0, window, n_feat), dtype=np.float32), np.empty((0,), dtype=np.int64)

    indices = np.arange(window)[None, :] + np.arange(n - window + 1)[:, None]
    X = features[indices]
    y = labels[window - 1:]
    return X, y


def prepare_all_assets(assets, indicator, timeframe, crossfeat=False,
                       target_type='binary', window=WINDOW):
    all_X = {'train': [], 'val': [], 'test': []}
    all_y = {'train': [], 'val': [], 'test': []}

    for asset in assets:
        df = load_asset_data(asset, indicator, timeframe,
                             crossfeat=crossfeat, target_type=target_type)
        feature_cols = [c for c in df.columns if c.startswith('feature_')]

        df_train, df_val, df_test = split_chronological(df)
        logger.info(f"    Split: train={len(df_train):,}, val={len(df_val):,}, test={len(df_test):,}")

        stats = compute_norm_stats(df_train, feature_cols)
        df_train = apply_norm(df_train, stats)
        df_val = apply_norm(df_val, stats)
        df_test = apply_norm(df_test, stats)

        for split_name, split_df in [('train', df_train), ('val', df_val), ('test', df_test)]:
            X, y = create_sequences(split_df, window=window, target_type=target_type)
            all_X[split_name].append(X)
            all_y[split_name].append(y)

    X_train = np.concatenate(all_X['train'])
    y_train = np.concatenate(all_y['train'])
    X_val = np.concatenate(all_X['val'])
    y_val = np.concatenate(all_y['val'])
    X_test = np.concatenate(all_X['test'])
    y_test = np.concatenate(all_y['test'])

    return X_train, y_train, X_val, y_val, X_test, y_test


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='XGBoost — drop-in replacement for CNN-LSTM')
    parser.add_argument('--indicator', default='macd', choices=['macd', 'rsi', 'cci'])
    parser.add_argument('--timeframe', default='30m', choices=['30m', '1h'])
    parser.add_argument('--assets', nargs='+', default=['BTC'])
    parser.add_argument('--window', type=int, default=WINDOW)
    parser.add_argument('--crossfeat', action='store_true')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    try:
        import xgboost as xgb
    except ImportError:
        logger.error("XGBoost not installed. Run: pip install xgboost")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"TRAINING XGBoost — {args.indicator}_{args.timeframe}")
    logger.info("=" * 60)

    # 1. Prepare data (IDENTICAL to CNN-LSTM pipeline)
    logger.info("\n1. Loading + preparing data (same pipeline as CNN-LSTM)...")
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_all_assets(
        args.assets, args.indicator, args.timeframe,
        crossfeat=args.crossfeat, window=args.window)

    logger.info(f"\n  Sequences: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}")
    logger.info(f"  Shape: {X_train.shape} (samples, window, features)")

    # 2. Flatten sequences for XGBoost: (n, 25, 3) → (n, 75)
    n_feat_original = X_train.shape[2]
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    logger.info(f"  Flattened: {X_train_flat.shape[1]} columns "
                f"({args.window} steps × {n_feat_original} features)")

    # 3. Train XGBoost
    logger.info("\n2. Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=args.seed,
        eval_metric='logloss',
        early_stopping_rounds=20,
        n_jobs=-1,
    )

    model.fit(
        X_train_flat, y_train,
        eval_set=[(X_val_flat, y_val)],
        verbose=50,
    )

    # 4. Evaluate
    logger.info("\n3. Evaluating...")
    train_acc = (model.predict(X_train_flat) == y_train).mean()
    val_acc = (model.predict(X_val_flat) == y_val).mean()
    test_acc = (model.predict(X_test_flat) == y_test).mean()

    logger.info(f"  Train accuracy: {train_acc:.4f}")
    logger.info(f"  Val accuracy:   {val_acc:.4f}")
    logger.info(f"  Test accuracy:  {test_acc:.4f}")

    # Probabilities
    train_probs = model.predict_proba(X_train_flat)[:, 1]
    val_probs = model.predict_proba(X_val_flat)[:, 1]
    test_probs = model.predict_proba(X_test_flat)[:, 1]

    logger.info(f"\n  Train pred: mean={train_probs.mean():.4f}, std={train_probs.std():.4f}")
    logger.info(f"  Val pred:   mean={val_probs.mean():.4f}, std={val_probs.std():.4f}")
    logger.info(f"  Test pred:  mean={test_probs.mean():.4f}, std={test_probs.std():.4f}")

    # 5. Feature importance (top 15)
    logger.info("\n4. Top 15 feature importances:")
    importances = model.feature_importances_
    # Create readable names: feature_0_step_0, feature_0_step_1, ...
    flat_names = []
    for step in range(args.window):
        for feat in range(n_feat_original):
            flat_names.append(f"f{feat}_step{step}")

    indices = np.argsort(importances)[::-1]
    for i in range(min(15, len(flat_names))):
        idx = indices[i]
        logger.info(f"    {flat_names[idx]:<25} {importances[idx]:.4f}")

    # 6. Save NPZ (same format as train_multitf for analyze_predictions)
    logger.info("\n5. Saving...")
    npz_path = f'{PREPARED_DATA_DIR}/{args.indicator}_{args.timeframe}_dataset.npz'
    np.savez(npz_path,
             train_preds=train_probs,
             train_labels=y_train,
             val_preds=val_probs,
             val_labels=y_val,
             test_preds=test_probs,
             test_labels=y_test)
    logger.info(f"  NPZ saved: {npz_path}")

    model_path = f'models/xgboost_{args.indicator}_{args.timeframe}.json'
    Path('models').mkdir(exist_ok=True)
    model.save_model(model_path)
    logger.info(f"  Model saved: {model_path}")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"DONE — XGBoost {args.indicator}_{args.timeframe}")
    logger.info(f"  Train: {train_acc:.4f}  Val: {val_acc:.4f}  Test: {test_acc:.4f}")
    logger.info(f"{'=' * 60}")


if __name__ == '__main__':
    main()
