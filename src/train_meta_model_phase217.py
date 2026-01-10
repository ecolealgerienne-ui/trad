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
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import joblib
import json
from typing import Tuple, Dict

# Ajouter le répertoire parent au path pour importer les modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.model import CNNLSTMMultiOutput
from src.constants import DEVICE


def load_model(model_path: Path, n_features: int) -> CNNLSTMMultiOutput:
    """
    Charge un modèle primaire entraîné.

    Args:
        model_path: Chemin vers le .pth
        n_features: Nombre de features (1 pour MACD/RSI, 3 pour CCI)

    Returns:
        Modèle chargé en mode eval
    """
    model = CNNLSTMMultiOutput(n_features=n_features, n_outputs=1)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def extract_probabilities(
    model: CNNLSTMMultiOutput,
    sequences: np.ndarray
) -> np.ndarray:
    """
    Extrait les probabilités de direction d'un modèle primaire.

    Args:
        model: Modèle primaire (direction-only)
        sequences: Séquences (n, seq_len, n_features)

    Returns:
        Probabilités (n,) - probabilité de direction UP
    """
    model.eval()
    with torch.no_grad():
        X = torch.tensor(sequences, dtype=torch.float32, device=DEVICE)
        outputs = model(X)  # (n, 1) - déjà après sigmoid
        probs = outputs[:, 0].cpu().numpy()  # (n,)
    return probs


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
    sequences_macd: np.ndarray,
    sequences_rsi: np.ndarray,
    sequences_cci: np.ndarray,
    ohlcv: np.ndarray,
    model_macd: CNNLSTMMultiOutput,
    model_rsi: CNNLSTMMultiOutput,
    model_cci: CNNLSTMMultiOutput
) -> np.ndarray:
    """
    Construit les features du meta-modèle.

    Args:
        sequences_*: Séquences pour chaque indicateur
        ohlcv: Données OHLCV (n, 7)
        model_*: Modèles primaires

    Returns:
        Features (n, 6):
            [macd_prob, rsi_prob, cci_prob,
             confidence_spread, confidence_mean, volatility_atr]
    """
    print("Building meta-features...")

    # 1. Extraire probabilités primaires
    print("  Extracting probabilities from primary models...")
    macd_prob = extract_probabilities(model_macd, sequences_macd)
    rsi_prob = extract_probabilities(model_rsi, sequences_rsi)
    cci_prob = extract_probabilities(model_cci, sequences_cci)

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
    parser = argparse.ArgumentParser(description='Train meta-model Phase 2.17')
    parser.add_argument('--filter', type=str, default='kalman', choices=['kalman', 'octave20'],
                        help='Filter type (default: kalman)')
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

    # Chemins datasets
    data_dir = Path('data/prepared')
    models_dir = Path('models')

    # Charger meta-labels
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)

    splits = ['train', 'val', 'test']
    datasets = {}

    for split in splits:
        print(f"\nLoading {split} split...")
        meta_path = data_dir / f'meta_labels_macd_{args.filter}_{split}.npz'
        meta_data = np.load(meta_path, allow_pickle=True)

        # Charger aussi les datasets originaux pour RSI et CCI
        macd_path = data_dir / f'dataset_btc_eth_bnb_ada_ltc_macd_direction_only_{args.filter}.npz'
        rsi_path = data_dir / f'dataset_btc_eth_bnb_ada_ltc_rsi_direction_only_{args.filter}.npz'
        cci_path = data_dir / f'dataset_btc_eth_bnb_ada_ltc_cci_direction_only_{args.filter}.npz'

        macd_data = np.load(macd_path, allow_pickle=True)
        rsi_data = np.load(rsi_path, allow_pickle=True)
        cci_data = np.load(cci_path, allow_pickle=True)

        # Extraire selon le split
        metadata = macd_data['metadata'].item()
        split_ranges = metadata['split_ranges'][split]
        start, end = split_ranges['start'], split_ranges['end']

        datasets[split] = {
            'sequences_macd': macd_data['X'][start:end],
            'sequences_rsi': rsi_data['X'][start:end],
            'sequences_cci': cci_data['X'][start:end],
            'ohlcv': meta_data['ohlcv'],
            'meta_labels': meta_data['meta_labels']
        }

        print(f"  Sequences: {datasets[split]['sequences_macd'].shape}")
        print(f"  Meta-labels: {datasets[split]['meta_labels'].shape}")

    # Charger modèles primaires
    print("\n" + "="*80)
    print("LOADING PRIMARY MODELS")
    print("="*80)

    model_macd_path = models_dir / f'best_model_macd_{args.filter}_dual_binary.pth'
    model_rsi_path = models_dir / f'best_model_rsi_{args.filter}_dual_binary.pth'
    model_cci_path = models_dir / f'best_model_cci_{args.filter}_dual_binary.pth'

    print(f"Loading MACD model: {model_macd_path}")
    model_macd = load_model(model_macd_path, n_features=1)

    print(f"Loading RSI model: {model_rsi_path}")
    model_rsi = load_model(model_rsi_path, n_features=1)

    print(f"Loading CCI model: {model_cci_path}")
    model_cci = load_model(model_cci_path, n_features=3)

    # Construire features meta-modèle pour chaque split
    print("\n" + "="*80)
    print("BUILDING META-FEATURES")
    print("="*80)

    X_meta = {}
    y_meta = {}

    for split in splits:
        print(f"\n{split.upper()} split:")
        X_meta[split] = build_meta_features(
            sequences_macd=datasets[split]['sequences_macd'],
            sequences_rsi=datasets[split]['sequences_rsi'],
            sequences_cci=datasets[split]['sequences_cci'],
            ohlcv=datasets[split]['ohlcv'],
            model_macd=model_macd,
            model_rsi=model_rsi,
            model_cci=model_cci
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
    for split in splits:
        results[split] = evaluate_model(
            model=meta_model,
            X=X_meta[split],
            y=y_meta[split],
            split_name=split
        )

    # Sauvegarder modèle
    model_path = args.output_dir / f'meta_model_baseline_{args.filter}.pkl'
    print(f"\nSaving meta-model to: {model_path}")
    joblib.dump(meta_model, model_path)

    # Sauvegarder résultats
    results_path = args.output_dir / f'meta_model_results_{args.filter}.json'
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
