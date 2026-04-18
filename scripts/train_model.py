#!/usr/bin/env python3
"""
Entraîne un XGBoost depuis un dataset ML préparé par prepare_*_data.py.

Ce script fait UNIQUEMENT le train. La préparation (load, resample, features,
labels, split, normalize, sequences) est faite en amont et sauvée dans un NPZ.

Pipeline:
  1. Charge dataset_{indicator}_{tf}_{source}.npz
  2. Flatten X: (n, window, n_feat) → (n, window*n_feat)
  3. XGBoost.fit sur train, eval_set=val, early stopping
  4. Predict proba sur test
  5. Metrics: accuracy, F1, AUC
  6. Feature importance top 10
  7. Save model + NPZ predictions (pour backtest_model.py)

Sorties:
  models/xgb_{indicator}_{tf}m_{source}.json     (modèle)
  data/prepared/preds_{indicator}_{tf}m_{source}.npz (predictions)

Usage:
    python scripts/train_model.py --source full --indicator macd --tf 30
    python scripts/train_model.py --source full --n-estimators 1000
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


PREP_DIR = Path('data/prepared')
MODELS_DIR = Path('models')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indicator', default='macd',
                        choices=['macd', 'rsi', 'cci'])
    parser.add_argument('--tf', type=int, default=30, choices=[30, 60])
    parser.add_argument('--source', default='full',
                        choices=['3months', 'full'])
    # Hyperparams XGBoost
    parser.add_argument('--n-estimators', type=int, default=500)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample-bytree', type=float, default=0.8)
    parser.add_argument('--min-child-weight', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--reg-alpha', type=float, default=0.1)
    parser.add_argument('--reg-lambda', type=float, default=1.0)
    parser.add_argument('--early-stopping-rounds', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    tf_label = f'{args.tf}m' if args.tf < 60 else '1h'
    print("=" * 80)
    print(f"TRAIN XGBoost — {args.indicator.upper()} × {tf_label}  "
          f"source={args.source}")
    print("=" * 80)

    # ========== [1] Charger dataset ==========
    dataset_path = PREP_DIR / f'dataset_{args.indicator}_{tf_label}_{args.source}.npz'
    if not dataset_path.exists():
        print(f"❌ Dataset non trouvé: {dataset_path}")
        if args.source == 'full':
            print(f"   Lance d'abord: python scripts/prepare_full_data.py "
                  f"--indicator {args.indicator} --tf {args.tf}")
        return

    print(f"\n[1/5] Charge dataset: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    X_train = data['X_train']
    y_train = data['y_train_binary']
    X_val = data['X_val']
    y_val = data['y_val_binary']
    X_test = data['X_test']
    y_test = data['y_test_binary']
    window = int(data['window'])
    n_feat = X_train.shape[2]
    print(f"  X_train={X_train.shape}  X_val={X_val.shape}  X_test={X_test.shape}")
    print(f"  window={window}  n_features={n_feat}")

    # ========== [2] Flatten ==========
    print(f"\n[2/5] Flatten X → (n, {window}×{n_feat}={window*n_feat}) ...")
    X_tr_flat = X_train.reshape(len(X_train), -1)
    X_va_flat = X_val.reshape(len(X_val), -1)
    X_te_flat = X_test.reshape(len(X_test), -1)

    # ========== [3] XGBoost fit ==========
    print(f"\n[3/5] XGBoost.fit ...")
    try:
        import xgboost as xgb
    except ImportError:
        print("❌ xgboost non installé. pip install xgboost")
        return

    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=args.seed,
        eval_metric='logloss',
        early_stopping_rounds=args.early_stopping_rounds,
        n_jobs=-1,
    )
    model.fit(X_tr_flat, y_train, eval_set=[(X_va_flat, y_val)], verbose=50)

    # ========== [4] Métriques ==========
    print("\n" + "=" * 80)
    print("MÉTRIQUES")
    print("=" * 80)
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    for name, Xf, y in [('TRAIN', X_tr_flat, y_train),
                         ('VAL', X_va_flat, y_val),
                         ('TEST', X_te_flat, y_test)]:
        preds = model.predict(Xf)
        probas = model.predict_proba(Xf)[:, 1]
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds)
        auc = roc_auc_score(y, probas)
        print(f"  {name:<6} acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    # Feature importance
    print("\nTop 10 feature importance:")
    importances = model.feature_importances_
    feature_cols = [str(c) for c in data['feature_cols']]
    flat_names = [f'{feat}_t-{window - 1 - ts}'
                  for ts in range(window) for feat in feature_cols]
    top_idx = np.argsort(importances)[::-1][:10]
    for i in top_idx:
        print(f"  {flat_names[i]:<25} {importances[i]:.4f}")

    # ========== [5] Sauvegarde ==========
    model_path = MODELS_DIR / f'xgb_{args.indicator}_{tf_label}_{args.source}.json'
    model.save_model(model_path)
    print(f"\n✅ Modèle sauvé: {model_path}")

    # NPZ predictions pour le backtest
    test_preds_proba = model.predict_proba(X_te_flat)[:, 1]
    npz_path = PREP_DIR / f'preds_{args.indicator}_{tf_label}_{args.source}.npz'
    np.savez(
        npz_path,
        test_preds_proba=test_preds_proba.astype(np.float64),
        test_y_true=y_test.astype(np.int64),
        test_dates=data['dates_test'],
        test_closes=data['closes_test'].astype(np.float64),
        test_indices=data['indices_test'].astype(np.int64),
        indicator=args.indicator,
        tf_minutes=args.tf,
        source=args.source,
    )
    print(f"✅ Predictions sauvées: {npz_path}")
    print(f"\nPour backtester:")
    print(f"  python scripts/backtest_model.py --indicator {args.indicator} "
          f"--tf {args.tf} --source {args.source} --threshold 0.5")


if __name__ == '__main__':
    main()
