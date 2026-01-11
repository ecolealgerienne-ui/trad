#!/usr/bin/env python3
"""
Regime Classifier Training - Model A (Meta-Regime Phase 1)

Entraîne un classifieur XGBoost multiclass pour prédire le régime de marché (4 classes).

Architecture:
    ~20 features de régime (trend/vol/volume)
    → XGBoost Multiclass
    → Probabilités 4 régimes [0, 1, 2, 3]

Régimes (basés sur Trend Strength × Volatility Cluster):
    0: RANGE LOW VOL  - Consolidation calme
    1: RANGE HIGH VOL - Consolidation agitée
    2: TREND LOW VOL  - Tendance stable
    3: TREND HIGH VOL - Tendance explosive

Features (~20):
    Trend: MA slopes, ADX, regression R², Hurst, MACD histogram
    Volatility: ATR, BB width, realized vol, compression
    Volume: Volume ratio, spikes, VWAP deviation, OBV

Target:
    regime = 0, 1, 2, ou 3 (4 classes)

Performance attendue:
    - Accuracy: 45-55%
    - Macro F1: 0.40-0.50
    - AUC (macro OvR): 0.65-0.75

Référence:
    - Ang & Bekaert (2002) - Regime Switches
    - López de Prado (2018) - Feature Engineering
    - Documentation: docs/META_REGIME_TRADING_SPECS.md
"""

import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import joblib
import json
from typing import Dict

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Error: XGBoost not installed. Install with: pip install xgboost")
    exit(1)


def load_regime_dataset(npz_path: Path) -> Dict:
    """
    Charge le dataset de régimes préparé.

    Structure attendue du NPZ:
        - X: (n, 12, ~22) = [timestamp, asset_id, features...]
        - Y: (n, 5) = [timestamp, asset_id, regime, ts_score, vc_score]
        - OHLCV: (n, 7) = [timestamp, asset_id, O, H, L, C, V]
        - metadata: JSON avec infos

    Args:
        npz_path: Chemin vers le fichier .npz

    Returns:
        Dict avec X, Y, OHLCV, metadata
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    print(f"Loading dataset: {npz_path.name}")
    data = np.load(npz_path, allow_pickle=True)

    # Extraire les données
    X = data['X']  # (n, 12, ~22)
    Y = data['Y']  # (n, 5)
    OHLCV = data['OHLCV']  # (n, 7)
    metadata = json.loads(str(data['metadata'])) if 'metadata' in data else {}

    # Extraire les régimes (colonne 2 de Y)
    regimes = Y[:, 2].astype(int)

    print(f"  Total samples: {len(regimes):,}")
    print(f"  Sequences shape: {X.shape}")
    print(f"  Feature columns: {X.shape[2] - 2}")  # -2 pour timestamp, asset_id

    # Distribution des régimes
    print(f"\n  Regime distribution:")
    for regime_id in range(4):
        count = np.sum(regimes == regime_id)
        pct = 100 * count / len(regimes)
        regime_names = {
            0: 'RANGE LOW VOL',
            1: 'RANGE HIGH VOL',
            2: 'TREND LOW VOL',
            3: 'TREND HIGH VOL'
        }
        print(f"    Regime {regime_id} ({regime_names[regime_id]:15s}): {count:,} ({pct:.1f}%)")

    return {
        'X': X,
        'Y': Y,
        'OHLCV': OHLCV,
        'regimes': regimes,
        'metadata': metadata
    }


def prepare_features_for_xgboost(X: np.ndarray) -> np.ndarray:
    """
    Prépare les features pour XGBoost depuis les séquences.

    XGBoost ne prend pas de séquences directement, donc on doit:
    - Option A: Flatten (12 × features) → grand vecteur
    - Option B: Aggregate (mean, std, min, max sur 12 steps)
    - Option C: Keep last timestep only

    Pour ce baseline, on utilise Option B (aggregate stats).

    Args:
        X: Séquences (n, 12, features+2) avec [timestamp, asset_id, features...]

    Returns:
        Features aggregated (n, 4*n_features) = [mean, std, min, max] × features
    """
    print("\nAggregating sequence features for XGBoost...")

    # Extraire les features (skip timestamp et asset_id)
    features = X[:, :, 2:]  # (n, 12, n_features)

    # Calculer stats sur la dimension temporelle (axis=1)
    feat_mean = np.mean(features, axis=1)  # (n, n_features)
    feat_std = np.std(features, axis=1)    # (n, n_features)
    feat_min = np.min(features, axis=1)    # (n, n_features)
    feat_max = np.max(features, axis=1)    # (n, n_features)

    # Concatener toutes les stats
    X_aggregated = np.hstack([feat_mean, feat_std, feat_min, feat_max])  # (n, 4*n_features)

    print(f"  Input shape: {X.shape}")
    print(f"  Aggregated shape: {X_aggregated.shape}")
    print(f"  Features per sample: {X_aggregated.shape[1]}")

    return X_aggregated


def train_xgboost_regime_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> xgb.XGBClassifier:
    """
    Entraîne le classifieur XGBoost multiclass pour les régimes.

    Args:
        X_train: Features train (n_train, n_features)
        y_train: Régimes train (n_train,) - valeurs [0, 1, 2, 3]
        X_val: Features val (n_val, n_features)
        y_val: Régimes val (n_val,)

    Returns:
        Modèle XGBoost entraîné
    """
    print("\n" + "="*80)
    print("TRAINING XGBOOST REGIME CLASSIFIER (Multiclass)")
    print("="*80)

    print(f"\nTrain samples: {len(X_train):,}")
    print(f"Val samples: {len(X_val):,}")

    # Distribution des régimes
    print(f"\nTrain regime distribution:")
    for regime_id in range(4):
        count = np.sum(y_train == regime_id)
        pct = 100 * count / len(y_train)
        print(f"  Regime {regime_id}: {count:,} ({pct:.1f}%)")

    # Entraîner XGBoost multiclass
    print("\nTraining XGBoost with multiclass objective...")
    model = xgb.XGBClassifier(
        objective='multi:softprob',   # Multiclass avec probabilités
        num_class=4,                   # 4 régimes
        n_estimators=200,              # Plus d'arbres pour multiclass
        max_depth=6,                   # Profondeur augmentée (vs 5 binary)
        learning_rate=0.05,            # LR réduit pour plus de stabilité
        subsample=0.8,                 # Row sampling pour régularisation
        colsample_bytree=0.8,          # Column sampling pour régularisation
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',        # Multiclass log loss
        early_stopping_rounds=20       # Early stopping
    )

    # Fit avec early stopping sur val
    print("Training with early stopping...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10  # Print tous les 10 rounds
    )

    print(f"\nBest iteration: {model.best_iteration}")
    print(f"Best score (val mlogloss): {model.best_score:.4f}")

    # Feature importance (top 20)
    print("\nTop 20 most important features:")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]

    for i, idx in enumerate(indices, 1):
        print(f"  {i:2d}. Feature {idx:3d}: {importances[idx]:.4f}")

    # Évaluation train
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"\nTrain Accuracy: {train_acc:.4f}")

    # Évaluation val
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Val Accuracy: {val_acc:.4f}")

    return model


def evaluate_regime_classifier(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str
) -> Dict[str, float]:
    """
    Évalue le classifieur de régimes sur un split.

    Args:
        model: Modèle XGBoost entraîné
        X: Features (n, n_features)
        y: Régimes (n,) - valeurs [0, 1, 2, 3]
        split_name: Nom du split (train/val/test)

    Returns:
        Métriques: {accuracy, precision_macro, recall_macro, f1_macro, roc_auc_ovr}
    """
    print("\n" + "="*80)
    print(f"EVALUATION - {split_name.upper()} SET")
    print("="*80)

    print(f"Samples: {len(X):,}")

    # Prédictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)  # (n, 4) probabilités

    # Métriques
    acc = accuracy_score(y, y_pred)
    prec_macro = precision_score(y, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y, y_pred, average='macro', zero_division=0)

    # ROC AUC (One-vs-Rest pour multiclass)
    try:
        auc_ovr = roc_auc_score(y, y_pred_proba, multi_class='ovr', average='macro')
    except ValueError:
        auc_ovr = 0.0  # Si une classe manque dans y

    print(f"\nMetrics:")
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  Precision (macro): {prec_macro:.4f}")
    print(f"  Recall (macro):    {rec_macro:.4f}")
    print(f"  F1-Score (macro):  {f1_macro:.4f}")
    print(f"  ROC AUC (OvR):     {auc_ovr:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print("     ", "  ".join([f"R{i}" for i in range(4)]))
    for i, row in enumerate(cm):
        print(f"  R{i}:", "  ".join([f"{val:4d}" for val in row]))

    # Per-class metrics
    print(f"\nPer-class metrics:")
    regime_names = {
        0: 'RANGE LOW VOL',
        1: 'RANGE HIGH VOL',
        2: 'TREND LOW VOL',
        3: 'TREND HIGH VOL'
    }

    prec_per_class = precision_score(y, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y, y_pred, average=None, zero_division=0)

    for i in range(4):
        print(f"  Regime {i} ({regime_names[i]:15s}): "
              f"Prec={prec_per_class[i]:.3f}, "
              f"Rec={rec_per_class[i]:.3f}, "
              f"F1={f1_per_class[i]:.3f}")

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, digits=4, target_names=[
        'R0: RANGE LOW',
        'R1: RANGE HIGH',
        'R2: TREND LOW',
        'R3: TREND HIGH'
    ]))

    return {
        'accuracy': acc,
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'f1_macro': f1_macro,
        'roc_auc_ovr': auc_ovr,
        'confusion_matrix': cm.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description='Train Regime Classifier (Model A - XGBoost)')
    parser.add_argument('--data', type=Path, required=True,
                        help='Path to prepared regime dataset (.npz)')
    parser.add_argument('--output-dir', type=Path, default=Path('models/regime'),
                        help='Output directory for regime classifier')
    args = parser.parse_args()

    print("="*80)
    print("REGIME CLASSIFIER TRAINING - Model A (XGBoost Multiclass)")
    print("="*80)
    print(f"Dataset: {args.data}")
    print(f"Output: {args.output_dir}")

    # Vérifier que XGBoost est disponible
    if not XGBOOST_AVAILABLE:
        print("\n❌ XGBoost not installed!")
        print("Install with: pip install xgboost")
        return

    # Créer répertoire output
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Charger dataset
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)

    full_data = load_regime_dataset(args.data)

    # Extraire metadata pour identifier les splits
    metadata = full_data['metadata']
    split_indices = metadata.get('split_indices', {})

    if not split_indices:
        raise ValueError("No split indices found in metadata. Run prepare_data_regime.py first.")

    # Extraire les splits
    train_end = split_indices['train_end']
    val_end = split_indices['val_end']

    X_full = full_data['X']
    y_full = full_data['regimes']

    X_train_seq = X_full[:train_end]
    y_train = y_full[:train_end]

    X_val_seq = X_full[train_end:val_end]
    y_val = y_full[train_end:val_end]

    X_test_seq = X_full[val_end:]
    y_test = y_full[val_end:]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train_seq):,} samples")
    print(f"  Val:   {len(X_val_seq):,} samples")
    print(f"  Test:  {len(X_test_seq):,} samples")

    # Préparer features pour XGBoost (aggregate séquences)
    print("\n" + "="*80)
    print("PREPARING FEATURES")
    print("="*80)

    X_train = prepare_features_for_xgboost(X_train_seq)
    X_val = prepare_features_for_xgboost(X_val_seq)
    X_test = prepare_features_for_xgboost(X_test_seq)

    # Entraîner le modèle
    regime_classifier = train_xgboost_regime_classifier(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )

    # Évaluer sur les 3 splits
    results = {}
    for split_name, X_split, y_split in [
        ('train', X_train, y_train),
        ('val', X_val, y_val),
        ('test', X_test, y_test)
    ]:
        results[split_name] = evaluate_regime_classifier(
            model=regime_classifier,
            X=X_split,
            y=y_split,
            split_name=split_name
        )

    # Sauvegarder modèle
    model_path = args.output_dir / 'regime_classifier_xgboost.pkl'
    print(f"\nSaving regime classifier to: {model_path}")
    joblib.dump(regime_classifier, model_path)

    # Sauvegarder résultats
    results_path = args.output_dir / 'regime_classifier_results.json'
    print(f"Saving results to: {results_path}")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("✅ REGIME CLASSIFIER TRAINING COMPLETED")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Results: {results_path}")
    print(f"\nTest Metrics:")
    print(f"  Accuracy:    {results['test']['accuracy']:.4f}")
    print(f"  F1 (macro):  {results['test']['f1_macro']:.4f}")
    print(f"  ROC AUC (OvR): {results['test']['roc_auc_ovr']:.4f}")


if __name__ == '__main__':
    main()
