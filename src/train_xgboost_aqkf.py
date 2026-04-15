#!/usr/bin/env python3
"""
XGBoost training on AQ-KF features — comparison with CNN-LSTM.

Same data pipeline as train_multitf_aqkf.py:
  - Same CSV (from prepare_multitf_csv_aqkf.py)
  - Same chronological split (70/15/15 with gap)
  - Same z-score normalization (stats from train only)
  - Same oracle labels

Differences:
  - XGBoost instead of CNN-LSTM
  - No sequences: each row = 1 sample with lagged features
  - Outputs same NPZ format for analyze_predictions_aqkf.py

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

WINDOW = 25  # Number of lags to add
SEED = 42

ASSET_CSV_MAP = {
    'BTC': 'BTCUSD', 'ETH': 'ETHUSD', 'BNB': 'BNBUSD',
    'ADA': 'ADAUSD', 'LTC': 'LTCUSD',
}


# =============================================================================
# DATA LOADING
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


def load_and_prepare(asset_name, indicator, timeframe, n_lags=WINDOW):
    """Load CSV, extract features + lags, return DataFrame."""
    csv_path = find_csv(asset_name, indicator)
    df = pd.read_csv(csv_path, parse_dates=['datetime']).set_index('datetime').sort_index()

    # Base features
    base_features = [
        f'{indicator}_{timeframe}_live',
        f'{indicator}_{timeframe}_filtered',
    ]
    vel_col = f'{indicator}_{timeframe}_velocity'
    if vel_col in df.columns:
        base_features.append(vel_col)

    label_col = f'oracle_label_{indicator}_{timeframe}'

    missing = [c for c in base_features + [label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Build feature matrix with lags
    feature_cols = []
    for col in base_features:
        # Current value
        feature_cols.append(col)
        # Lagged values
        for lag in range(1, n_lags + 1):
            lag_col = f'{col}_lag{lag}'
            df[lag_col] = df[col].shift(lag)
            feature_cols.append(lag_col)

    # Add derived features
    for col in base_features:
        # Diff (momentum)
        diff_col = f'{col}_diff1'
        df[diff_col] = df[col].diff()
        feature_cols.append(diff_col)

        # Rolling mean/std over window
        mean_col = f'{col}_mean{n_lags}'
        std_col = f'{col}_std{n_lags}'
        df[mean_col] = df[col].rolling(n_lags).mean()
        df[std_col] = df[col].rolling(n_lags).std()
        feature_cols.append(mean_col)
        feature_cols.append(std_col)

    # Extract and drop NaN
    result = df[feature_cols + [label_col]].copy()
    n_before = len(result)
    result = result.dropna()
    n_dropped = n_before - len(result)

    n_features = len(feature_cols)
    logger.info(f"  {asset_name}: {len(result):,} rows, {n_features} features "
                f"({len(base_features)} base + {n_lags} lags + derived, dropped {n_dropped:,} NaN)")

    return result, feature_cols, label_col


# =============================================================================
# SPLIT + NORMALIZATION
# =============================================================================

def split_chronological(df, train_ratio=0.70, val_ratio=0.15, gap=WINDOW):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end - gap], df.iloc[train_end:val_end - gap], df.iloc[val_end:]


def normalize(df_train, df_val, df_test, feature_cols):
    """Z-score normalization with stats from train only."""
    stats = {}
    for col in feature_cols:
        mean = df_train[col].mean()
        std = df_train[col].std()
        if std < 1e-10:
            std = 1.0
        stats[col] = {'mean': float(mean), 'std': float(std)}

    for df in [df_train, df_val, df_test]:
        for col, s in stats.items():
            if col in df.columns:
                df[col] = (df[col] - s['mean']) / s['std']

    return stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='XGBoost on AQ-KF features')
    parser.add_argument('--indicator', default='macd')
    parser.add_argument('--timeframe', default='30m')
    parser.add_argument('--assets', nargs='+', default=['BTC'])
    parser.add_argument('--n-lags', type=int, default=WINDOW)
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

    # Load data
    logger.info("\n1. Loading data...")
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    all_test_preds = []

    for asset in args.assets:
        df, feature_cols, label_col = load_and_prepare(
            asset, args.indicator, args.timeframe, n_lags=args.n_lags)

        # Split
        df_train, df_val, df_test = split_chronological(df)
        logger.info(f"    Split: train={len(df_train):,}, val={len(df_val):,}, test={len(df_test):,}")

        # Normalize (copies to avoid SettingWithCopyWarning)
        df_train = df_train.copy()
        df_val = df_val.copy()
        df_test = df_test.copy()
        norm_stats = normalize(df_train, df_val, df_test, feature_cols)

        all_train_X.append(df_train[feature_cols].values)
        all_train_y.append(df_train[label_col].values)
        all_val_X.append(df_val[feature_cols].values)
        all_val_y.append(df_val[label_col].values)
        all_test_X.append(df_test[feature_cols].values)
        all_test_y.append(df_test[label_col].values)

    X_train = np.concatenate(all_train_X)
    y_train = np.concatenate(all_train_y)
    X_val = np.concatenate(all_val_X)
    y_val = np.concatenate(all_val_y)
    X_test = np.concatenate(all_test_X)
    y_test = np.concatenate(all_test_y)

    logger.info(f"\n  Total: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}")
    logger.info(f"  Features: {X_train.shape[1]}")

    # Train XGBoost
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
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # Evaluate
    logger.info("\n3. Evaluating...")
    train_acc = (model.predict(X_train) == y_train).mean()
    val_acc = (model.predict(X_val) == y_val).mean()
    test_acc = (model.predict(X_test) == y_test).mean()

    logger.info(f"  Train accuracy: {train_acc:.4f}")
    logger.info(f"  Val accuracy:   {val_acc:.4f}")
    logger.info(f"  Test accuracy:  {test_acc:.4f}")

    # Predictions (probabilities)
    train_probs = model.predict_proba(X_train)[:, 1]
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    logger.info(f"\n  Train pred: mean={train_probs.mean():.4f}, std={train_probs.std():.4f}")
    logger.info(f"  Val pred:   mean={val_probs.mean():.4f}, std={val_probs.std():.4f}")
    logger.info(f"  Test pred:  mean={test_probs.mean():.4f}, std={test_probs.std():.4f}")

    # Feature importance (top 10)
    logger.info("\n4. Top 10 feature importances:")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in range(min(10, len(feature_cols))):
        idx = indices[i]
        logger.info(f"    {feature_cols[idx]:<40} {importances[idx]:.4f}")

    # Save NPZ (same format as train_multitf for analyze_predictions)
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

    # Save model
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
