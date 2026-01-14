#!/usr/bin/env python3
"""
Regime Classifier Training - Model A (Meta-Regime Phase 1)

Entraîne un classifieur XGBoost multiclass pour prédire le régime de marché (3 classes).

Architecture:
    3 raw returns features (c_ret, h_ret, l_ret)
    → Agrégation temporelle [mean, std, min, max]
    → XGBoost Multiclass
    → Probabilités 3 régimes [0, 1, 2]

Régimes (basés sur Trend Strength × Volatility Cluster):
    0: RANGE_LOW_VOL  - Consolidation calme (TS < 0.45, vol ≤ P50)
    1: RANGE_HIGH_VOL - Consolidation agitée (TS < 0.45, vol > P50)
    2: TREND          - Tendance (TS ≥ 0.45)

Note: En crypto, TREND = VOLATILITÉ (Oxford-Man Institute, BIS 2020).

Features (3 raw returns):
    [0] c_ret - Close return (close[t] - close[t-1]) / close[t-1]
    [1] h_ret - High return (high[t] - close[t-1]) / close[t-1]
    [2] l_ret - Low return (low[t] - close[t-1]) / close[t-1]

Target:
    regime = 0, 1, ou 2 (3 classes)

Performance attendue:
    - Accuracy: 45-55%
    - Macro F1: 0.40-0.50
    - AUC (macro OvR): 0.65-0.75

Référence:
    - Ang & Bekaert (2002) - Regime Switches
    - López de Prado (2018) - Feature Engineering
    - Documentation: docs/META_REGIME_TRADING_SPECS.md

═══════════════════════════════════════════════════════════════════════════════
DONNÉES D'ENTRAÎNEMENT - Structure détaillée
═══════════════════════════════════════════════════════════════════════════════

INPUT: X_train
────────────────
Shape: (n_train, 25, 5)
  - n_train: Nombre d'échantillons train
  - 25: Longueur séquence (25 timesteps × 5min = 2h05 de contexte)
  - 5: Nombre de colonnes (2 metadata + 3 raw returns)

Colonnes X_train[:, :, i]:
  Index 0-1: METADATA
    [0] timestamp    - Unix timestamp (int64)
    [1] asset_id     - ID asset 0-4 (BTC=0, ETH=1, BNB=2, ADA=3, LTC=4)

  Index 2-4: RAW RETURNS FEATURES (3)
    [2] c_ret - Close return
    [3] h_ret - High return
    [4] l_ret - Low return

Source: prepare_data_regime.py

TARGET: Y_train
────────────────
Shape: (n_train, 6)

Colonnes Y_train[:, i]:
  [0] timestamp       - Unix timestamp (int64)
  [1] asset_id        - ID asset 0-4
  [2] regime          - Régime 0-2 (TARGET PRINCIPAL)
  [3] macd_dir        - Direction MACD Kalman 0/1 (0=DOWN, 1=UP)
  [4] rsi_dir         - Direction RSI Kalman 0/1
  [5] cci_dir         - Direction CCI Kalman 0/1

RÉGIMES (3 classes):
  0: RANGE_LOW_VOL  - Consolidation calme (TS < 0.45, vol ≤ P50)
  1: RANGE_HIGH_VOL - Consolidation agitée (TS < 0.45, vol > P50)
  2: TREND          - Tendance (TS ≥ 0.45)

UTILISATION PAR XGBOOST:
  Ce script aggregate les 25 timesteps en 4 statistiques [mean, std, min, max]:
  X_aggregated shape: (n_train, 4 × 3) = (n_train, 12) features
  Target: regimes_train = Y_train[:, 2]

ENRICHISSEMENT POST-TRAINING:
  Après entraînement, Y est enrichi avec les prédictions du modèle:
  Y enrichi shape: (n, 10) = Y original (6) + regime_pred (1) + probs (3)
  Nouvelles colonnes:
    [6] regime_pred - Prédiction du classifieur
    [7] prob_R0     - Probabilité RANGE_LOW_VOL
    [8] prob_R1     - Probabilité RANGE_HIGH_VOL
    [9] prob_R2     - Probabilité TREND

Source dataset: data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz
Généré par: src/prepare_data_regime.py
═══════════════════════════════════════════════════════════════════════════════
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
import shutil
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
        - X_train, Y_train, OHLCV_train
        - X_val, Y_val, OHLCV_val
        - X_test, Y_test, OHLCV_test
        - metadata: JSON avec infos

    Args:
        npz_path: Chemin vers le fichier .npz

    Returns:
        Dict avec splits séparés et metadata
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    print(f"Loading dataset: {npz_path.name}")
    data = np.load(npz_path, allow_pickle=True)

    # Extraire les splits
    X_train = data['X_train']
    Y_train = data['Y_train']
    OHLCV_train = data['OHLCV_train']

    X_val = data['X_val']
    Y_val = data['Y_val']
    OHLCV_val = data['OHLCV_val']

    X_test = data['X_test']
    Y_test = data['Y_test']
    OHLCV_test = data['OHLCV_test']

    metadata = json.loads(str(data['metadata'])) if 'metadata' in data else {}

    # Extraire les régimes (colonne 2 de Y)
    regimes_train = Y_train[:, 2].astype(int)
    regimes_val = Y_val[:, 2].astype(int)
    regimes_test = Y_test[:, 2].astype(int)

    print(f"\n  Split sizes:")
    print(f"    Train: {len(regimes_train):,} samples")
    print(f"    Val:   {len(regimes_val):,} samples")
    print(f"    Test:  {len(regimes_test):,} samples")
    print(f"  Sequences shape: {X_train.shape}")
    print(f"  Feature columns: {X_train.shape[2] - 2}")  # -2 pour timestamp, asset_id

    # Distribution des régimes (Train uniquement)
    print(f"\n  Train regime distribution:")
    regime_names = {
        0: 'RANGE LOW VOL',
        1: 'RANGE HIGH VOL',
        2: 'TREND'
    }
    for regime_id in range(3):
        count = np.sum(regimes_train == regime_id)
        pct = 100 * count / len(regimes_train)
        print(f"    Regime {regime_id} ({regime_names[regime_id]:15s}): {count:,} ({pct:.1f}%)")

    return {
        'X_train': X_train,
        'Y_train': Y_train,
        'OHLCV_train': OHLCV_train,
        'regimes_train': regimes_train,
        'X_val': X_val,
        'Y_val': Y_val,
        'OHLCV_val': OHLCV_val,
        'regimes_val': regimes_val,
        'X_test': X_test,
        'Y_test': Y_test,
        'OHLCV_test': OHLCV_test,
        'regimes_test': regimes_test,
        'metadata': metadata
    }


def prepare_features_for_xgboost(X: np.ndarray) -> np.ndarray:
    """
    Prépare les features pour XGBoost depuis les séquences.

    XGBoost ne prend pas de séquences directement, donc on doit:
    - Option A: Flatten (25 × features) → grand vecteur
    - Option B: Aggregate (mean, std, min, max sur 25 steps)
    - Option C: Keep last timestep only

    Pour ce baseline, on utilise Option B (aggregate stats).

    Args:
        X: Séquences (n, 25, 5) avec [timestamp, asset_id, c_ret, h_ret, l_ret]

    Returns:
        Features aggregated (n, 4*3) = (n, 12) = [mean, std, min, max] × 3 raw returns
    """
    print("\nAggregating sequence features for XGBoost...")

    # Extraire les features (skip timestamp et asset_id)
    features = X[:, :, 2:]  # (n, 25, 3) = [c_ret, h_ret, l_ret]

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
        y_train: Régimes train (n_train,) - valeurs [0, 1, 2]
        X_val: Features val (n_val, n_features)
        y_val: Régimes val (n_val,) - valeurs [0, 1, 2]

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
    for regime_id in range(3):
        count = np.sum(y_train == regime_id)
        pct = 100 * count / len(y_train)
        print(f"  Regime {regime_id}: {count:,} ({pct:.1f}%)")

    # Entraîner XGBoost multiclass
    print("\nTraining XGBoost with multiclass objective...")
    model = xgb.XGBClassifier(
        objective='multi:softprob',   # Multiclass avec probabilités
        num_class=3,                   # 3 régimes (RANGE LOW VOL, RANGE HIGH VOL, TREND)
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
        y: Régimes (n,) - valeurs [0, 1, 2]
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
    y_pred_proba = model.predict_proba(X)  # (n, 3) probabilités pour 3 régimes

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
    print("     ", "  ".join([f"R{i}" for i in range(3)]))
    for i, row in enumerate(cm):
        print(f"  R{i}:", "  ".join([f"{val:4d}" for val in row]))

    # Per-class metrics
    print(f"\nPer-class metrics:")
    regime_names = {
        0: 'RANGE LOW VOL',
        1: 'RANGE HIGH VOL',
        2: 'TREND'
    }

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
        'R0: RANGE LOW VOL',
        'R1: RANGE HIGH VOL',
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

    # Extraire les splits (déjà séparés dans le NPZ)
    X_train_seq = full_data['X_train']
    y_train = full_data['regimes_train']

    X_val_seq = full_data['X_val']
    y_val = full_data['regimes_val']

    X_test_seq = full_data['X_test']
    y_test = full_data['regimes_test']

    # Les tailles ont déjà été affichées par load_regime_dataset()

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

    # Générer prédictions et probabilités pour enrichir le dataset
    print("\n" + "="*80)
    print("ENRICHING DATASET WITH REGIME PREDICTIONS")
    print("="*80)

    # Prédictions (classe) et probabilités (3 colonnes pour 3 régimes)
    regime_preds_train = regime_classifier.predict(X_train)
    regime_probs_train = regime_classifier.predict_proba(X_train)

    regime_preds_val = regime_classifier.predict(X_val)
    regime_probs_val = regime_classifier.predict_proba(X_val)

    regime_preds_test = regime_classifier.predict(X_test)
    regime_probs_test = regime_classifier.predict_proba(X_test)

    # Enrichir Y avec les prédictions
    # Y original: (n, 6) - [timestamp, asset_id, regime, macd_dir, rsi_dir, cci_dir]
    # Y enrichi: (n, 10) - [Y_original (6), regime_pred (1), prob_R0, prob_R1, prob_R2 (3)]
    Y_train_enriched = np.column_stack([
        full_data['Y_train'],
        regime_preds_train.reshape(-1, 1),
        regime_probs_train
    ])

    Y_val_enriched = np.column_stack([
        full_data['Y_val'],
        regime_preds_val.reshape(-1, 1),
        regime_probs_val
    ])

    Y_test_enriched = np.column_stack([
        full_data['Y_test'],
        regime_preds_test.reshape(-1, 1),
        regime_probs_test
    ])

    # Créer backup de l'original (seulement la première fois)
    backup_path = args.data.parent / f"{args.data.stem}_original.npz"
    if not backup_path.exists():
        print(f"\n📦 Creating backup of original dataset...")
        shutil.copy(args.data, backup_path)
        print(f"  ✅ Backup saved: {backup_path.name}")
    else:
        print(f"\n📦 Backup already exists: {backup_path.name}")

    # Remplacer le fichier original avec la version enrichie
    print(f"\n💾 Enriching and saving dataset: {args.data.name}")
    print(f"  Added columns: regime_pred, prob_R0, prob_R1, prob_R2")
    print(f"  Y shape: {full_data['Y_train'].shape} → {Y_train_enriched.shape}")

    np.savez_compressed(
        args.data,  # Remplace l'original
        X_train=full_data['X_train'],
        Y_train=Y_train_enriched,
        OHLCV_train=full_data['OHLCV_train'],
        X_val=full_data['X_val'],
        Y_val=Y_val_enriched,
        OHLCV_val=full_data['OHLCV_val'],
        X_test=full_data['X_test'],
        Y_test=Y_test_enriched,
        OHLCV_test=full_data['OHLCV_test'],
        metadata=full_data['metadata']
    )

    print(f"✅ Dataset enriched and saved!")
    print(f"  Original backup: {backup_path.name}")

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
