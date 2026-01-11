#!/usr/bin/env python3
"""
Meta-Model Training - Phase 2.17

Entraîne un meta-modèle (Logistic Regression baseline) pour prédire
si un trade sera profitable selon les méta-labels Triple Barrier.

Architecture:
    6 features (probs primaires + confidence + volatility)
    → Logistic Regression
    → Probabilité trade profitable

Features (Phase 1 - Kalman only):
    1. macd_prob      - Probabilité MACD direction
    2. rsi_prob       - Probabilité RSI direction
    3. cci_prob       - Probabilité CCI direction
    4. confidence_spread - max(probs) - min(probs)
    5. confidence_mean   - mean(probs)
    6. volatility_atr    - ATR normalisé

Target:
    meta_label = 1 si trade profitable NET ET duration >= 5p
    meta_label = 0 sinon (rejeté)

Approche progressive (López de Prado):
    1. Baseline: Logistic Regression (interprétable)
    2. Si gain > +5%: XGBoost
    3. Si gain > +5%: MLP

Référence:
    López de Prado, "Advances in Financial ML", Chapter 3: Meta-Labeling
"""

import argparse
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import joblib
import json
from typing import Tuple, Dict


def load_meta_dataset(split: str, indicator: str = 'macd', filter_type: str = 'kalman', aligned: bool = False) -> Dict:
    """
    Charge le dataset meta-labels avec prédictions.

    Args:
        split: 'train', 'val', ou 'test'
        indicator: Indicateur utilisé pour meta-labels (default: 'macd')
        filter_type: Type de filtre (default: 'kalman')
        aligned: Si True, charge labels aligned (signal reversal) au lieu de Triple Barrier

    Returns:
        Dict avec predictions, meta_labels, ohlcv, etc.
    """
    suffix = '_aligned' if aligned else ''
    npz_path = Path(f'data/prepared/meta_labels_{indicator}_{filter_type}_{split}{suffix}.npz')

    if not npz_path.exists():
        raise FileNotFoundError(f"Meta-labels file not found: {npz_path}")

    print(f"Loading {split} set: {npz_path.name}")
    data = np.load(npz_path, allow_pickle=True)

    result = {
        'predictions_macd': data['predictions_macd'],
        'predictions_rsi': data['predictions_rsi'],
        'predictions_cci': data['predictions_cci'],
        'meta_labels': data['meta_labels'],
        'ohlcv': data['OHLCV'],
        'metadata': json.loads(str(data['metadata'])) if 'metadata' in data else {}
    }

    print(f"  Samples: {len(result['meta_labels'])}")
    print(f"  Positive: {np.sum(result['meta_labels'] == 1)}")
    print(f"  Negative: {np.sum(result['meta_labels'] == 0)}")
    print(f"  Ignored: {np.sum(result['meta_labels'] == -1)}")

    return result


def calculate_atr(ohlcv: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calcule l'ATR (Average True Range) normalisé.

    Args:
        ohlcv: Array (n, 7) [timestamp, asset_id, O, H, L, C, V]
        period: Période ATR (défaut 14)

    Returns:
        ATR normalisé (n,) - ATR / Close
    """
    highs = ohlcv[:, 3]  # H
    lows = ohlcv[:, 4]   # L
    closes = ohlcv[:, 5] # C

    # True Range = max(H-L, |H-C_prev|, |L-C_prev|)
    tr1 = highs - lows
    tr2 = np.abs(highs - np.roll(closes, 1))
    tr3 = np.abs(lows - np.roll(closes, 1))
    tr2[0] = tr1[0]  # Pas de précédent pour le premier
    tr3[0] = tr1[0]

    true_range = np.maximum.reduce([tr1, tr2, tr3])

    # ATR = EMA du True Range
    atr = np.zeros_like(true_range)
    atr[0] = true_range[0]

    alpha = 1.0 / period
    for i in range(1, len(true_range)):
        atr[i] = alpha * true_range[i] + (1 - alpha) * atr[i-1]

    # Normaliser par le prix
    atr_normalized = atr / closes

    return atr_normalized


def build_meta_features(
    predictions_macd: np.ndarray,
    predictions_rsi: np.ndarray,
    predictions_cci: np.ndarray,
    ohlcv: np.ndarray
) -> np.ndarray:
    """
    Construit les features du meta-modèle depuis les prédictions existantes.

    Args:
        predictions_*: Probabilités des modèles primaires (n,)
        ohlcv: Données OHLCV (n, 7)

    Returns:
        Features (n, 6):
            [macd_prob, rsi_prob, cci_prob,
             confidence_spread, confidence_mean, volatility_atr]
    """
    print("Building meta-features...")

    # 1. Probabilités primaires (déjà disponibles)
    macd_prob = predictions_macd
    rsi_prob = predictions_rsi
    cci_prob = predictions_cci

    # 2. Calculer confidence metrics
    print("  Computing confidence metrics...")
    probs = np.stack([macd_prob, rsi_prob, cci_prob], axis=1)  # (n, 3)
    confidence_spread = np.max(probs, axis=1) - np.min(probs, axis=1)  # (n,)
    confidence_mean = np.mean(probs, axis=1)  # (n,)

    # 3. Calculer volatilité ATR
    print("  Computing ATR volatility...")
    volatility_atr = calculate_atr(ohlcv, period=14)

    # 4. Stack features
    X_meta = np.stack([
        macd_prob,
        rsi_prob,
        cci_prob,
        confidence_spread,
        confidence_mean,
        volatility_atr
    ], axis=1)  # (n, 6)

    print(f"  Meta-features shape: {X_meta.shape}")
    print(f"  Features stats:")
    print(f"    MACD prob: [{macd_prob.min():.3f}, {macd_prob.max():.3f}]")
    print(f"    RSI prob: [{rsi_prob.min():.3f}, {rsi_prob.max():.3f}]")
    print(f"    CCI prob: [{cci_prob.min():.3f}, {cci_prob.max():.3f}]")
    print(f"    Confidence spread: [{confidence_spread.min():.3f}, {confidence_spread.max():.3f}]")
    print(f"    Confidence mean: [{confidence_mean.min():.3f}, {confidence_mean.max():.3f}]")
    print(f"    ATR: [{volatility_atr.min():.6f}, {volatility_atr.max():.6f}]")

    return X_meta


def train_baseline_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> LogisticRegression:
    """
    Entraîne le meta-modèle baseline (Logistic Regression).

    Args:
        X_train: Features train (n_train, 6)
        y_train: Labels train (n_train,)
        X_val: Features val (n_val, 6)
        y_val: Labels val (n_val,)

    Returns:
        Modèle entraîné
    """
    print("\n" + "="*80)
    print("TRAINING BASELINE META-MODEL (Logistic Regression)")
    print("="*80)

    # Filtrer les labels ignored (-1)
    train_mask = y_train != -1
    val_mask = y_val != -1

    X_train_filtered = X_train[train_mask]
    y_train_filtered = y_train[train_mask]
    X_val_filtered = X_val[val_mask]
    y_val_filtered = y_val[val_mask]

    print(f"\nTrain samples: {len(X_train_filtered):,} (after filtering)")
    print(f"Val samples: {len(X_val_filtered):,} (after filtering)")

    # Distribution des classes
    pos_train = np.sum(y_train_filtered == 1)
    neg_train = np.sum(y_train_filtered == 0)
    print(f"\nTrain distribution:")
    print(f"  Positive (1): {pos_train:,} ({100*pos_train/len(y_train_filtered):.1f}%)")
    print(f"  Negative (0): {neg_train:,} ({100*neg_train/len(y_train_filtered):.1f}%)")

    # Entraîner Logistic Regression avec class_weight='balanced'
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',  # Important pour class imbalance
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_filtered, y_train_filtered)

    # Poids des features (interprétabilité)
    print("\nFeature weights (coefficients):")
    feature_names = ['macd_prob', 'rsi_prob', 'cci_prob',
                     'confidence_spread', 'confidence_mean', 'volatility_atr']
    for name, coef in zip(feature_names, model.coef_[0]):
        print(f"  {name:20s}: {coef:+.4f}")
    print(f"  Intercept: {model.intercept_[0]:+.4f}")

    # Évaluation train
    y_train_pred = model.predict(X_train_filtered)
    train_acc = accuracy_score(y_train_filtered, y_train_pred)
    print(f"\nTrain Accuracy: {train_acc:.4f}")

    # Évaluation val
    y_val_pred = model.predict(X_val_filtered)
    val_acc = accuracy_score(y_val_filtered, y_val_pred)
    print(f"Val Accuracy: {val_acc:.4f}")

    return model


def evaluate_model(
    model: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str
) -> Dict[str, float]:
    """
    Évalue le meta-modèle sur un split.

    Args:
        model: Meta-modèle entraîné
        X: Features (n, 6)
        y: Labels (n,)
        split_name: Nom du split (train/val/test)

    Returns:
        Métriques: {accuracy, precision, recall, f1, roc_auc}
    """
    print("\n" + "="*80)
    print(f"EVALUATION - {split_name.upper()} SET")
    print("="*80)

    # Filtrer ignored labels
    mask = y != -1
    X_filtered = X[mask]
    y_filtered = y[mask]

    print(f"Samples: {len(X_filtered):,}")

    # Prédictions
    y_pred = model.predict(X_filtered)
    y_pred_proba = model.predict_proba(X_filtered)[:, 1]  # Probabilité classe 1

    # Métriques
    acc = accuracy_score(y_filtered, y_pred)
    prec = precision_score(y_filtered, y_pred, zero_division=0)
    rec = recall_score(y_filtered, y_pred, zero_division=0)
    f1 = f1_score(y_filtered, y_pred, zero_division=0)
    auc = roc_auc_score(y_filtered, y_pred_proba)

    print(f"\nMetrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC AUC:   {auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_filtered, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    print(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_filtered, y_pred, digits=4))

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': auc
    }


def main():
    parser = argparse.ArgumentParser(description='Train meta-model Phase 2.17/2.18')
    parser.add_argument('--filter', type=str, default='kalman', choices=['kalman', 'octave20'],
                        help='Filter type (default: kalman)')
    parser.add_argument('--aligned', action='store_true',
                        help='Use aligned labels (signal reversal) instead of Triple Barrier')
    parser.add_argument('--output-dir', type=Path, default=Path('models/meta_model'),
                        help='Output directory for meta-model')
    args = parser.parse_args()

    print("="*80)
    print("META-MODEL TRAINING - Phase 2.17")
    print("="*80)
    print(f"Filter: {args.filter}")
    print(f"Output: {args.output_dir}")

    # Créer répertoire output
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Charger meta-labels avec prédictions
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)

    datasets = {}
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        datasets[split] = load_meta_dataset(split, indicator='macd', filter_type=args.filter, aligned=args.aligned)

    # Construire features meta-modèle pour chaque split
    print("\n" + "="*80)
    print("BUILDING META-FEATURES")
    print("="*80)

    X_meta = {}
    y_meta = {}

    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()} split:")
        X_meta[split] = build_meta_features(
            predictions_macd=datasets[split]['predictions_macd'],
            predictions_rsi=datasets[split]['predictions_rsi'],
            predictions_cci=datasets[split]['predictions_cci'],
            ohlcv=datasets[split]['ohlcv']
        )
        y_meta[split] = datasets[split]['meta_labels']

    # Entraîner baseline
    meta_model = train_baseline_model(
        X_train=X_meta['train'],
        y_train=y_meta['train'],
        X_val=X_meta['val'],
        y_val=y_meta['val']
    )

    # Évaluer sur les 3 splits
    results = {}
    for split in ['train', 'val', 'test']:
        results[split] = evaluate_model(
            model=meta_model,
            X=X_meta[split],
            y=y_meta[split],
            split_name=split
        )

    # Sauvegarder modèle
    suffix = '_aligned' if args.aligned else ''
    model_path = args.output_dir / f'meta_model_baseline_{args.filter}{suffix}.pkl'
    print(f"\nSaving meta-model to: {model_path}")
    joblib.dump(meta_model, model_path)

    # Sauvegarder résultats
    results_path = args.output_dir / f'meta_model_results_{args.filter}{suffix}.json'
    print(f"Saving results to: {results_path}")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("✅ META-MODEL TRAINING COMPLETED")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Results: {results_path}")
    print(f"\nTest Accuracy: {results['test']['accuracy']:.4f}")
    print(f"Test F1-Score: {results['test']['f1']:.4f}")
    print(f"Test ROC AUC: {results['test']['roc_auc']:.4f}")


if __name__ == '__main__':
    main()
