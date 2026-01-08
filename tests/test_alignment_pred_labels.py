#!/usr/bin/env python3
"""
Test Alignement Prédictions vs Labels

Objectif:
- Vérifier que les prédictions ML sont bien alignées avec les labels
- Mesurer l'accuracy réelle (pred vs label au même timestamp)
- Détecter décalages temporels potentiels
- Comparer avec les 92.5% attendus (MACD Kalman)

Hypothèse:
Si alignement ~92%, le problème vient juste des sorties prématurées (flickering).
Si alignement < 92%, il y a un problème structurel (décalage temporel, bug).

Usage:
    python tests/test_alignment_pred_labels.py --indicator macd --split test
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class AlignmentResult:
    """Résultats du test d'alignement."""
    n_samples: int
    accuracy: float
    agreement: int
    disagreement: int
    pred_up: int
    pred_down: int
    label_up: int
    label_down: int
    confusion_matrix: np.ndarray  # [[TN, FP], [FN, TP]]
    precision: float
    recall: float
    f1: float


def load_predictions_and_labels(
    indicator: str,
    filter_type: str,
    split: str = 'test'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge prédictions et labels depuis .npz.

    Args:
        indicator: 'macd', 'rsi', 'cci'
        filter_type: 'kalman' ou 'octave20'
        split: 'train', 'val', 'test'

    Returns:
        (predictions, labels)
        - predictions: Prédictions ML (0-1 probabilités)
        - labels: Labels vrais (0-1 binaire)
    """
    dataset_pattern = f"dataset_btc_eth_bnb_ada_ltc_{indicator}_direction_only_{filter_type}.npz"
    dataset_path = Path("data/prepared") / dataset_pattern

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {dataset_path}")

    print(f"📂 Chargement: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)

    # Charger prédictions et labels
    X = data[f'X_{split}']
    Y = data[f'Y_{split}']
    Y_pred = data.get(f'Y_{split}_pred', None)

    if Y_pred is None:
        raise ValueError(f"Y_{split}_pred introuvable dans {dataset_path}")

    # Y et Y_pred shape: (n_samples, 1) pour Direction-Only
    predictions = Y_pred.flatten()  # (n_samples,) - probabilités [0,1]
    labels = Y.flatten()  # (n_samples,) - binaire [0,1]

    print(f"✅ Chargé: {len(predictions):,} samples")
    print(f"   Predictions shape: {Y_pred.shape} → {predictions.shape}")
    print(f"   Labels shape: {Y.shape} → {labels.shape}")
    print()

    return predictions, labels


def compute_alignment(
    predictions: np.ndarray,
    labels: np.ndarray
) -> AlignmentResult:
    """
    Calcule l'alignement entre prédictions et labels.

    Args:
        predictions: Prédictions ML (0-1 probabilités)
        labels: Labels vrais (0-1 binaire)

    Returns:
        AlignmentResult avec métriques complètes
    """
    n_samples = len(predictions)

    # Convertir prédictions en binaire (seuil 0.5)
    pred_binary = (predictions > 0.5).astype(int)

    # Accuracy globale
    agreement = (pred_binary == labels).sum()
    disagreement = (pred_binary != labels).sum()
    accuracy = agreement / n_samples * 100

    # Distribution prédictions et labels
    pred_up = (pred_binary == 1).sum()
    pred_down = (pred_binary == 0).sum()
    label_up = (labels == 1).sum()
    label_down = (labels == 0).sum()

    # Matrice de confusion
    # [[TN, FP],
    #  [FN, TP]]
    tn = ((pred_binary == 0) & (labels == 0)).sum()
    fp = ((pred_binary == 1) & (labels == 0)).sum()
    fn = ((pred_binary == 0) & (labels == 1)).sum()
    tp = ((pred_binary == 1) & (labels == 1)).sum()

    confusion_matrix = np.array([[tn, fp], [fn, tp]])

    # Métriques
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return AlignmentResult(
        n_samples=n_samples,
        accuracy=accuracy,
        agreement=agreement,
        disagreement=disagreement,
        pred_up=pred_up,
        pred_down=pred_down,
        label_up=label_up,
        label_down=label_down,
        confusion_matrix=confusion_matrix,
        precision=precision,
        recall=recall,
        f1=f1
    )


def analyze_disagreements(
    predictions: np.ndarray,
    labels: np.ndarray,
    max_samples: int = 100
) -> pd.DataFrame:
    """
    Analyse les désaccords entre prédictions et labels.

    Args:
        predictions: Prédictions ML (0-1 probabilités)
        labels: Labels vrais (0-1 binaire)
        max_samples: Nombre max de désaccords à afficher

    Returns:
        DataFrame avec les désaccords
    """
    pred_binary = (predictions > 0.5).astype(int)
    disagreements = (pred_binary != labels)

    indices = np.where(disagreements)[0]
    if len(indices) == 0:
        return pd.DataFrame()

    # Limiter au max_samples
    indices = indices[:max_samples]

    df = pd.DataFrame({
        'index': indices,
        'prediction_prob': predictions[indices],
        'prediction_binary': pred_binary[indices],
        'label': labels[indices],
        'error_type': ['FP' if pred_binary[i] == 1 else 'FN' for i in indices],
        'confidence': np.abs(predictions[indices] - 0.5) * 2
    })

    return df


def run_alignment_test(
    indicator: str,
    filter_type: str,
    split: str
):
    """
    Lance le test d'alignement complet.

    Args:
        indicator: 'macd', 'rsi', 'cci'
        filter_type: 'kalman' ou 'octave20'
        split: 'train', 'val', 'test'
    """
    print("=" * 80)
    print(f"TEST ALIGNEMENT PRÉDICTIONS vs LABELS")
    print("=" * 80)
    print(f"Indicateur: {indicator.upper()}")
    print(f"Filtre: {filter_type}")
    print(f"Split: {split}")
    print()

    # Charger données
    predictions, labels = load_predictions_and_labels(indicator, filter_type, split)

    # Calculer alignement
    result = compute_alignment(predictions, labels)

    # Afficher résultats
    print("=" * 80)
    print("RÉSULTATS ALIGNEMENT")
    print("=" * 80)
    print(f"Samples: {result.n_samples:,}")
    print()

    print(f"✅ **ACCURACY: {result.accuracy:.2f}%**")
    print(f"   Accord: {result.agreement:,} ({result.agreement/result.n_samples*100:.2f}%)")
    print(f"   Désaccord: {result.disagreement:,} ({result.disagreement/result.n_samples*100:.2f}%)")
    print()

    print("Distribution Prédictions vs Labels:")
    print(f"   Pred UP:   {result.pred_up:,} ({result.pred_up/result.n_samples*100:.2f}%)")
    print(f"   Pred DOWN: {result.pred_down:,} ({result.pred_down/result.n_samples*100:.2f}%)")
    print(f"   Label UP:   {result.label_up:,} ({result.label_up/result.n_samples*100:.2f}%)")
    print(f"   Label DOWN: {result.label_down:,} ({result.label_down/result.n_samples*100:.2f}%)")
    print()

    print("Matrice de Confusion:")
    print("                  Label DOWN  Label UP")
    print(f"   Pred DOWN (TN/FN)  {result.confusion_matrix[0,0]:8,}  {result.confusion_matrix[1,0]:8,}")
    print(f"   Pred UP   (FP/TP)  {result.confusion_matrix[0,1]:8,}  {result.confusion_matrix[1,1]:8,}")
    print()

    print(f"Precision: {result.precision:.4f}")
    print(f"Recall:    {result.recall:.4f}")
    print(f"F1 Score:  {result.f1:.4f}")
    print()

    # Comparer avec attendu
    expected_accuracy = {
        'macd': 92.5,
        'rsi': 87.6,
        'cci': 90.2
    }

    if indicator in expected_accuracy:
        expected = expected_accuracy[indicator]
        delta = result.accuracy - expected
        print("=" * 80)
        print("COMPARAISON AVEC ATTENDU")
        print("=" * 80)
        print(f"Accuracy attendue: {expected:.2f}%")
        print(f"Accuracy mesurée:  {result.accuracy:.2f}%")
        print(f"Delta: {delta:+.2f}%")
        print()

        if abs(delta) < 0.5:
            print("✅ **ALIGNEMENT PARFAIT** - Prédictions = Labels (±0.5%)")
            print("   → Le problème vient des SORTIES PRÉMATURÉES, pas de l'alignement")
        elif delta > 0.5:
            print("⚠️  **ALIGNEMENT MEILLEUR QUE ATTENDU** (+{:.2f}%)".format(delta))
            print("   → Possible overfit ou labels plus faciles sur ce split")
        else:
            print("❌ **ALIGNEMENT DÉGRADÉ** ({:.2f}%)".format(delta))
            print("   → Problème potentiel: décalage temporel ou bug")
        print()

    # Analyser désaccords
    print("=" * 80)
    print("ANALYSE DÉSACCORDS (premiers 20)")
    print("=" * 80)
    df_disagreements = analyze_disagreements(predictions, labels, max_samples=20)

    if len(df_disagreements) > 0:
        print(df_disagreements.to_string(index=False))
        print()

        # Statistiques erreurs
        n_fp = (df_disagreements['error_type'] == 'FP').sum()
        n_fn = (df_disagreements['error_type'] == 'FN').sum()
        avg_confidence = df_disagreements['confidence'].mean()

        print(f"Erreurs FP (Faux Positifs): {n_fp}")
        print(f"Erreurs FN (Faux Négatifs): {n_fn}")
        print(f"Confiance moyenne des erreurs: {avg_confidence:.3f}")
    else:
        print("Aucun désaccord trouvé!")
    print()


def main():
    parser = argparse.ArgumentParser(description="Test Alignement Prédictions vs Labels")
    parser.add_argument('--indicator', type=str, default='macd', choices=['macd', 'rsi', 'cci'],
                        help="Indicateur à tester (défaut: macd)")
    parser.add_argument('--filter-type', type=str, default='kalman', choices=['kalman', 'octave20'],
                        help="Filtre à utiliser (défaut: kalman)")
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help="Split à tester (défaut: test)")

    args = parser.parse_args()

    run_alignment_test(
        indicator=args.indicator,
        filter_type=args.filter_type,
        split=args.split
    )


if __name__ == '__main__':
    main()
