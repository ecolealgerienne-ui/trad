#!/usr/bin/env python3
"""
Entraîne un XGBoost binaire sur le dataset progressif.

Features tabulaires (2) :
  - slope_progressive : slope Kalman/FLKS à chaque 5min, normalisée (z-score)
      * step_k=0 → slope_t1 (backward pur, pas de sous-pas)
      * step_k=1..5 → slope_k1..k5 (backward + k sous-pas 5min)
  - step_k : indice du sous-pas dans la bougie TF (0..5), brut

Label binaire : sign(oracle.slope[t_ref]) ffill sur les 6 rows 5min.

Sortie :
  - Modèle pickle  : models/xgb_progressive_<ind>_<tf>_<period>.pkl
  - Preds NPZ      : data/prepared/preds_<ind>_<tf>_<period>_progressive.npz
                     (train_preds_proba, val_preds_proba, test_preds_proba)
                     → compatible avec scripts/backtest_progressive.py --mode model

Usage :
    python scripts/train_progressive.py
    python scripts/train_progressive.py --npz data/prepared/dataset_macd_30m_180d_progressive.npz
    python scripts/train_progressive.py --n-estimators 1000 --early-stop 50
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import xgboost as xgb
except ImportError:
    print("❌ xgboost non installé. pip install xgboost")
    sys.exit(1)

from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                confusion_matrix)

PREP_DIR = Path('data/prepared')
MODELS_DIR = Path('models')


def parse_npz_path(npz_path):
    """Extrait (indicator, tf_label, period_tag, filter_tag) du nom du NPZ.

    Attend :
      dataset_<ind>_<tf>_<period>_progressive.npz             → filter_tag=''
      dataset_<ind>_<tf>_<period>_progressive_adaptive.npz    → filter_tag='_adaptive'
    """
    name = npz_path.stem
    if not name.startswith('dataset_'):
        return None, None, None, None
    # Suffixe adaptive optionnel
    if name.endswith('_progressive_adaptive'):
        core = name[len('dataset_'):-len('_progressive_adaptive')]
        filter_tag = '_adaptive'
    elif name.endswith('_progressive'):
        core = name[len('dataset_'):-len('_progressive')]
        filter_tag = ''
    else:
        return None, None, None, None
    parts = core.split('_')
    if len(parts) != 3:
        return None, None, None, None
    return parts[0], parts[1], parts[2], filter_tag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', default=None,
                        help='Path NPZ (default: auto-detect latest full)')
    parser.add_argument('--n-estimators', type=int, default=500)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--early-stop', type=int, default=20,
                        help='Early stopping rounds (0 = désactivé)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Résoudre NPZ
    if args.npz is None:
        # default : full
        npz_path = PREP_DIR / 'dataset_macd_30m_full_progressive.npz'
    else:
        npz_path = Path(args.npz)
    if not npz_path.exists():
        print(f"❌ NPZ introuvable: {npz_path}")
        return

    indicator, tf_label, period_tag, filter_tag = parse_npz_path(npz_path)
    if indicator is None:
        print(f"❌ Impossible de parser le NPZ: {npz_path.name}")
        print(f"   Attendu: dataset_<ind>_<tf>_<period>_progressive.npz")
        return

    print("=" * 80)
    print(f"TRAIN PROGRESSIVE — {indicator.upper()} × {tf_label}  "
          f"period={period_tag}")
    print("=" * 80)

    # Load NPZ
    print(f"\n[1/5] Load NPZ: {npz_path}")
    ds = np.load(npz_path, allow_pickle=True)
    X_train = ds['X_train']
    y_train = ds['y_train_binary']
    X_val = ds['X_val']
    y_val = ds['y_val_binary']
    X_test = ds['X_test']
    y_test = ds['y_test_binary']
    feature_cols = [str(c) for c in ds['feature_cols']]
    print(f"   Features: {feature_cols}  (tabulaire, pas de séquences)")
    print(f"   X_train: {X_train.shape}  |  "
          f"X_val: {X_val.shape}  |  X_test: {X_test.shape}")
    print(f"   y_train UP ratio: {y_train.mean()*100:.2f}%  "
          f"(val: {y_val.mean()*100:.2f}%  test: {y_test.mean()*100:.2f}%)")

    # XGBoost
    print(f"\n[2/5] XGBoost: n_estimators={args.n_estimators}  "
          f"max_depth={args.max_depth}  lr={args.learning_rate}")
    fit_params = dict(eval_set=[(X_val, y_val)], verbose=50)
    model_kwargs = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=args.seed,
        n_jobs=-1,
    )
    if args.early_stop > 0:
        model_kwargs['early_stopping_rounds'] = args.early_stop

    model = xgb.XGBClassifier(**model_kwargs)
    model.fit(X_train, y_train, **fit_params)
    best_iter = getattr(model, 'best_iteration', args.n_estimators)
    print(f"   best_iteration: {best_iter}")

    # Prédictions
    print(f"\n[3/5] Predict sur train / val / test ...")
    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    # Métriques classification
    print(f"\n[4/5] Métriques classification (threshold=0.5) :")
    print(f"{'Split':<10} {'AUC':>8} {'Acc':>8} {'F1':>8}  {'Balance':>10}")
    print("-" * 50)
    for name, y, p in [('train', y_train, train_proba),
                        ('val', y_val, val_proba),
                        ('test', y_test, test_proba)]:
        y_pred = (p > 0.5).astype(int)
        auc = roc_auc_score(y, p)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        # Confusion
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        balance = (tp + fn) / (tn + fp + tp + fn)  # UP ratio réel
        print(f"{name:<10} {auc:>8.4f} {acc:>8.4f} {f1:>8.4f}  "
              f"{balance*100:>9.2f}%")

    # Feature importance
    print(f"\n   Feature importance (gain):")
    imp = model.feature_importances_
    for col, v in sorted(zip(feature_cols, imp), key=lambda x: -x[1]):
        print(f"     {col:<20} {v:.4f}")

    # Sauvegarde
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f'xgb_progressive_{indicator}_{tf_label}_{period_tag}{filter_tag}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_cols': feature_cols,
            'indicator': indicator,
            'tf_label': tf_label,
            'period_tag': period_tag,
            'best_iteration': int(best_iter),
        }, f)
    print(f"\n[5/5] Modèle sauvé: {model_path}  "
          f"({model_path.stat().st_size / 1024:.1f} KB)")

    # Preds NPZ (compatible avec backtest_progressive.py --mode model)
    preds_path = PREP_DIR / f'preds_{indicator}_{tf_label}_{period_tag}_progressive{filter_tag}.npz'
    np.savez(
        preds_path,
        train_preds_proba=train_proba.astype(np.float32),
        val_preds_proba=val_proba.astype(np.float32),
        test_preds_proba=test_proba.astype(np.float32),
        indicator=indicator,
        tf_label=tf_label,
        period_tag=period_tag,
        feature_cols=np.array(feature_cols),
    )
    print(f"   Preds sauvées: {preds_path}  "
          f"({preds_path.stat().st_size / 1024:.1f} KB)")

    print(f"\nPour backtester :")
    print(f"  python scripts/backtest_progressive.py \\")
    print(f"      --npz {npz_path} \\")
    print(f"      --preds {preds_path} \\")
    print(f"      --mode model --split test --fees 0.001")


if __name__ == '__main__':
    main()
