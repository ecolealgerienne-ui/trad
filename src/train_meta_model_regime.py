#!/usr/bin/env python3
"""
Meta-Model Regime Training - XGBoost avec Features Indicateurs

Entraîne un classifieur XGBoost pour prédire le régime de marché (3 classes)
en utilisant UNIQUEMENT les features indicateurs (pas les raw returns).

Architecture:
    20 features indicateurs (excluant h_ret, l_ret, c_ret)
    → Agrégation temporelle (mean, std, min, max, last)
    → XGBoost Classifier
    → Probabilités 3 régimes [0, 1, 2]

Features utilisées (20 indicateurs, colonnes 5-24 du X):
    Trend (7):
        [5] ma20_slope, [6] ma50_slope, [7] regression_slope, [8] regression_r2
        [9] adx, [10] macd_histogram_norm, [11] hurst_exponent

    Volatility (9):
        [12] atr_normalized, [13] bb_upper, [14] bb_middle, [15] bb_lower
        [16] bb_width, [17] percent_b, [18] realized_volatility
        [19] volatility_compression, [20] range_atr_ratio

    Volume (4):
        [21] volume_ratio, [22] volume_spike, [23] vwap_deviation, [24] obv_derivative

Features EXCLUES (raw returns, colonnes 2-4):
    [2] h_ret, [3] l_ret, [4] c_ret

Régimes (3 classes):
    0: RANGE_LOW_VOL  - Consolidation calme (TS < 0.45, vol ≤ P50)
    1: RANGE_HIGH_VOL - Consolidation agitée (TS < 0.45, vol > P50)
    2: TREND          - Tendance (TS ≥ 0.45)

Performance attendue: ~92-93% accuracy (vs 86% CNN-LSTM avec raw returns)

Référence:
    - XGBoost Baseline validé avec 92.67% accuracy (session précédente)
    - López de Prado (2018) - Feature Engineering for ML in Finance
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
from typing import Dict, Tuple, List
import logging

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not installed. Install with: pip install xgboost")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS - INDICES DES COLONNES
# ═══════════════════════════════════════════════════════════════════════════════

# Structure X: (n, 25, 25)
# Colonnes 0-1: metadata
COL_TIMESTAMP = 0
COL_ASSET_ID = 1

# Colonnes 2-4: raw returns (EXCLUES)
COL_H_RET = 2
COL_L_RET = 3
COL_C_RET = 4

# Colonnes 5-24: indicateurs (UTILISEES)
INDICATOR_START_COL = 5
INDICATOR_END_COL = 24  # inclusive

# Noms des features indicateurs (20 features)
INDICATOR_FEATURE_NAMES = [
    # Trend (7) - colonnes 5-11
    'ma20_slope', 'ma50_slope', 'regression_slope', 'regression_r2',
    'adx', 'macd_histogram_norm', 'hurst_exponent',
    # Volatility (9) - colonnes 12-20
    'atr_normalized', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'percent_b',
    'realized_volatility', 'volatility_compression', 'range_atr_ratio',
    # Volume (4) - colonnes 21-24
    'volume_ratio', 'volume_spike', 'vwap_deviation', 'obv_derivative'
]

# Structure Y: (n, 6) ou (n, 10) si enrichi
COL_Y_REGIME = 2  # Régime target


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_regime_dataset(npz_path: Path) -> Dict:
    """
    Charge le dataset de régimes préparé.

    Args:
        npz_path: Chemin vers le fichier .npz

    Returns:
        Dict avec splits et metadata
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    logger.info(f"Loading dataset: {npz_path.name}")
    data = np.load(npz_path, allow_pickle=True)

    # Extraire les splits
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_val = data['X_val']
    Y_val = data['Y_val']
    X_test = data['X_test']
    Y_test = data['Y_test']

    # Metadata
    metadata = {}
    if 'metadata' in data:
        try:
            meta_raw = data['metadata']
            if hasattr(meta_raw, 'item'):
                meta_item = meta_raw.item()
                if isinstance(meta_item, dict):
                    metadata = meta_item
                elif isinstance(meta_item, str):
                    metadata = json.loads(meta_item)
        except Exception:
            pass

    # Extraire les régimes (colonne 2 de Y)
    regimes_train = Y_train[:, COL_Y_REGIME].astype(int)
    regimes_val = Y_val[:, COL_Y_REGIME].astype(int)
    regimes_test = Y_test[:, COL_Y_REGIME].astype(int)

    logger.info(f"\n  Dataset structure:")
    logger.info(f"    X shape: {X_train.shape} (n, seq_len, features)")
    logger.info(f"    Total features: {X_train.shape[2]} columns")
    logger.info(f"    - Metadata: columns 0-1 (timestamp, asset_id)")
    logger.info(f"    - Raw returns: columns 2-4 (h_ret, l_ret, c_ret) [EXCLUDED]")
    logger.info(f"    - Indicators: columns 5-24 (20 features) [USED]")

    logger.info(f"\n  Split sizes:")
    logger.info(f"    Train: {len(regimes_train):,} samples")
    logger.info(f"    Val:   {len(regimes_val):,} samples")
    logger.info(f"    Test:  {len(regimes_test):,} samples")

    # Distribution des régimes (Train)
    logger.info(f"\n  Train regime distribution:")
    regime_names = {0: 'RANGE_LOW_VOL', 1: 'RANGE_HIGH_VOL', 2: 'TREND'}
    for regime_id in range(3):
        count = np.sum(regimes_train == regime_id)
        pct = 100 * count / len(regimes_train)
        logger.info(f"    Regime {regime_id} ({regime_names[regime_id]:15s}): {count:,} ({pct:.1f}%)")

    return {
        'X_train': X_train,
        'Y_train': Y_train,
        'regimes_train': regimes_train,
        'X_val': X_val,
        'Y_val': Y_val,
        'regimes_val': regimes_val,
        'X_test': X_test,
        'Y_test': Y_test,
        'regimes_test': regimes_test,
        'metadata': metadata
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def extract_indicator_features(X: np.ndarray) -> np.ndarray:
    """
    Extrait uniquement les features indicateurs (colonnes 5-24).

    Args:
        X: Séquences (n, 25, 25) avec toutes les colonnes

    Returns:
        Features indicateurs (n, 25, 20)
    """
    # Colonnes 5-24 = indices 5:25 (exclusive)
    indicators = X[:, :, INDICATOR_START_COL:INDICATOR_END_COL + 1]
    return indicators.astype(np.float32)


def aggregate_sequences_for_xgboost(X_indicators: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Agrège les séquences temporelles en features pour XGBoost.

    Pour chaque feature indicateur, calcule:
        - mean: moyenne sur la séquence
        - std: écart-type sur la séquence
        - min: minimum sur la séquence
        - max: maximum sur la séquence
        - last: dernière valeur (état le plus récent)

    Args:
        X_indicators: Features indicateurs (n, 25, 20)

    Returns:
        (X_aggregated, feature_names) - (n, 100), liste de noms
    """
    n_samples, seq_len, n_features = X_indicators.shape
    logger.info(f"\n  Aggregating sequences for XGBoost...")
    logger.info(f"    Input shape: {X_indicators.shape}")

    # Calculer les agrégations
    agg_mean = np.mean(X_indicators, axis=1)  # (n, 20)
    agg_std = np.std(X_indicators, axis=1)    # (n, 20)
    agg_min = np.min(X_indicators, axis=1)    # (n, 20)
    agg_max = np.max(X_indicators, axis=1)    # (n, 20)
    agg_last = X_indicators[:, -1, :]         # (n, 20) - dernier timestep

    # Concaténer toutes les agrégations
    X_aggregated = np.concatenate([
        agg_mean, agg_std, agg_min, agg_max, agg_last
    ], axis=1)  # (n, 100)

    # Générer noms des features
    feature_names = []
    for agg_name in ['mean', 'std', 'min', 'max', 'last']:
        for feat_name in INDICATOR_FEATURE_NAMES:
            feature_names.append(f"{feat_name}_{agg_name}")

    logger.info(f"    Output shape: {X_aggregated.shape}")
    logger.info(f"    Features: {len(feature_names)}")

    # Gérer les NaN/Inf
    n_nan = np.sum(np.isnan(X_aggregated))
    n_inf = np.sum(np.isinf(X_aggregated))
    if n_nan > 0 or n_inf > 0:
        logger.warning(f"    ⚠️ Found {n_nan} NaN and {n_inf} Inf values - replacing with 0")
        X_aggregated = np.nan_to_num(X_aggregated, nan=0.0, posinf=0.0, neginf=0.0)

    return X_aggregated, feature_names


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_xgboost_regime_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1
) -> 'xgb.XGBClassifier':
    """
    Entraîne le classifieur XGBoost pour les régimes.

    Args:
        X_train, y_train: Données train
        X_val, y_val: Données validation
        feature_names: Noms des features
        n_estimators: Nombre d'arbres
        max_depth: Profondeur max
        learning_rate: Learning rate

    Returns:
        Modèle XGBoost entraîné
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")

    print("\n" + "="*80)
    print("TRAINING XGBOOST REGIME CLASSIFIER")
    print("="*80)

    print(f"\nTrain samples: {len(X_train):,}")
    print(f"Val samples: {len(X_val):,}")
    print(f"Features: {X_train.shape[1]}")

    # Distribution des classes
    print(f"\nTrain regime distribution:")
    regime_names = {0: 'RANGE_LOW_VOL', 1: 'RANGE_HIGH_VOL', 2: 'TREND'}
    for regime_id in range(3):
        count = np.sum(y_train == regime_id)
        pct = 100 * count / len(y_train)
        print(f"  Regime {regime_id} ({regime_names[regime_id]:15s}): {count:,} ({pct:.1f}%)")

    # Créer modèle XGBoost
    print(f"\nTraining XGBoost...")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  learning_rate: {learning_rate}")

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        early_stopping_rounds=20
    )

    # Fit avec early stopping sur val
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    # Feature importance (top 20)
    print("\nTop 20 Feature Importance:")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:20]
    for idx in sorted_idx:
        print(f"  {feature_names[idx]:35s}: {importances[idx]:.4f}")

    # Évaluation train
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"\nTrain Accuracy: {train_acc:.4f}")

    # Évaluation val
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Val Accuracy: {val_acc:.4f}")

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_regime_classifier(
    model: 'xgb.XGBClassifier',
    X: np.ndarray,
    y: np.ndarray,
    split_name: str
) -> Dict:
    """
    Évalue le classifieur de régimes sur un split.

    Args:
        model: Modèle XGBoost entraîné
        X: Features agrégées (n, 100)
        y: Régimes (n,)
        split_name: Nom du split (train/val/test)

    Returns:
        Dict de métriques
    """
    print("\n" + "="*80)
    print(f"EVALUATION - {split_name.upper()} SET")
    print("="*80)

    print(f"Samples: {len(X):,}")

    # Prédictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)

    # Métriques
    acc = accuracy_score(y, y_pred)
    prec_macro = precision_score(y, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y, y_pred, average='macro', zero_division=0)

    # ROC AUC (One-vs-Rest)
    try:
        auc_ovr = roc_auc_score(y, y_pred_proba, multi_class='ovr', average='macro')
    except ValueError:
        auc_ovr = 0.0

    print(f"\nMetrics:")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Precision (macro): {prec_macro:.4f}")
    print(f"  Recall (macro):    {rec_macro:.4f}")
    print(f"  F1-Score (macro):  {f1_macro:.4f}")
    print(f"  ROC AUC (OvR):     {auc_ovr:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print("     ", "  ".join([f"R{i}" for i in range(3)]))
    for i, row in enumerate(cm):
        print(f"  R{i}:", "  ".join([f"{val:6d}" for val in row]))

    # Per-class metrics
    print(f"\nPer-class metrics:")
    regime_names = {0: 'RANGE_LOW_VOL', 1: 'RANGE_HIGH_VOL', 2: 'TREND'}

    prec_per_class = precision_score(y, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y, y_pred, average=None, zero_division=0)

    for i in range(3):
        print(f"  Regime {i} ({regime_names[i]:15s}): "
              f"Prec={prec_per_class[i]:.3f}, "
              f"Rec={rec_per_class[i]:.3f}, "
              f"F1={f1_per_class[i]:.3f}")

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, digits=4, target_names=[
        'R0: RANGE_LOW_VOL',
        'R1: RANGE_HIGH_VOL',
        'R2: TREND'
    ]))

    return {
        'accuracy': acc,
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'f1_macro': f1_macro,
        'roc_auc_ovr': auc_ovr,
        'confusion_matrix': cm.tolist()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Train Meta-Model Regime Classifier (XGBoost with Indicator Features)'
    )
    parser.add_argument('--data', type=Path, required=True,
                        help='Path to prepared regime dataset (.npz)')
    parser.add_argument('--output-dir', type=Path, default=Path('models/regime'),
                        help='Output directory (default: models/regime)')
    parser.add_argument('--n-estimators', type=int, default=200,
                        help='Number of XGBoost trees (default: 200)')
    parser.add_argument('--max-depth', type=int, default=6,
                        help='Max tree depth (default: 6)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate (default: 0.1)')
    args = parser.parse_args()

    print("="*80)
    print("META-MODEL REGIME CLASSIFIER - XGBoost with Indicator Features")
    print("="*80)
    print(f"Dataset: {args.data}")
    print(f"Output: {args.output_dir}")
    print(f"\nUsing ONLY indicator features (columns 5-24)")
    print(f"Excluding raw returns (h_ret, l_ret, c_ret)")

    # Créer répertoire output
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Charger dataset
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)

    data = load_regime_dataset(args.data)

    # Extraire features indicateurs (colonnes 5-24)
    print("\n" + "="*80)
    print("EXTRACTING INDICATOR FEATURES")
    print("="*80)

    X_train_indicators = extract_indicator_features(data['X_train'])
    X_val_indicators = extract_indicator_features(data['X_val'])
    X_test_indicators = extract_indicator_features(data['X_test'])

    print(f"\n  Indicator features extracted:")
    print(f"    Train: {X_train_indicators.shape}")
    print(f"    Val:   {X_val_indicators.shape}")
    print(f"    Test:  {X_test_indicators.shape}")

    # Agréger pour XGBoost
    print("\n" + "="*80)
    print("AGGREGATING SEQUENCES FOR XGBOOST")
    print("="*80)

    X_train_agg, feature_names = aggregate_sequences_for_xgboost(X_train_indicators)
    X_val_agg, _ = aggregate_sequences_for_xgboost(X_val_indicators)
    X_test_agg, _ = aggregate_sequences_for_xgboost(X_test_indicators)

    print(f"\n  Aggregated features:")
    print(f"    Train: {X_train_agg.shape}")
    print(f"    Val:   {X_val_agg.shape}")
    print(f"    Test:  {X_test_agg.shape}")
    print(f"    Total features: {len(feature_names)}")

    # Entraîner XGBoost
    model = train_xgboost_regime_classifier(
        X_train=X_train_agg,
        y_train=data['regimes_train'],
        X_val=X_val_agg,
        y_val=data['regimes_val'],
        feature_names=feature_names,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr
    )

    # Évaluer sur les 3 splits
    results = {}
    for split_name, X_split, y_split in [
        ('train', X_train_agg, data['regimes_train']),
        ('val', X_val_agg, data['regimes_val']),
        ('test', X_test_agg, data['regimes_test'])
    ]:
        results[split_name] = evaluate_regime_classifier(
            model=model,
            X=X_split,
            y=y_split,
            split_name=split_name
        )

    # Sauvegarder modèle
    model_path = args.output_dir / 'regime_classifier_xgboost_indicators.pkl'
    print(f"\n💾 Saving model to: {model_path}")
    joblib.dump({
        'model': model,
        'feature_names': feature_names,
        'indicator_cols': list(range(INDICATOR_START_COL, INDICATOR_END_COL + 1)),
        'excluded_cols': [COL_H_RET, COL_L_RET, COL_C_RET]
    }, model_path)

    # Sauvegarder résultats
    results_path = args.output_dir / 'regime_classifier_xgboost_indicators_results.json'
    print(f"💾 Saving results to: {results_path}")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("✅ META-MODEL REGIME CLASSIFIER TRAINING COMPLETED")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Results: {results_path}")
    print(f"\n📊 Test Metrics:")
    print(f"  Accuracy:    {results['test']['accuracy']:.4f}")
    print(f"  F1 (macro):  {results['test']['f1_macro']:.4f}")
    print(f"  ROC AUC:     {results['test']['roc_auc_ovr']:.4f}")


if __name__ == '__main__':
    main()
