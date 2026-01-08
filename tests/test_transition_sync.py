#!/usr/bin/env python3
"""
Test Transition Synchronization - Oracle vs Model

Question Critique:
Quand l'Oracle change d'avis (UP→DOWN ou DOWN→UP), est-ce que le modèle
change d'avis AU MÊME MOMENT?

Test:
Si label_oracle(t) != label_oracle(t-1):  # Oracle transition
    Est-ce que pred_model(t) != pred_model(t-1)?  # Modèle transition aussi?

Objectif:
- Mesurer accuracy des transitions (pas accuracy globale)
- Vérifier si modèle entre au bon moment ou en retard
- Comprendre gap 92.5% accuracy → 34% Win Rate

Usage:
    python tests/test_transition_sync.py --indicator macd --split test
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import numpy as np
from typing import Tuple, Dict


def load_predictions_and_labels(indicator: str, filter_type: str, split: str, weighted: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load predictions and labels.

    Args:
        indicator: 'macd', 'rsi', 'cci'
        filter_type: 'kalman' ou 'octave20'
        split: 'train', 'val', 'test'
        weighted: Si True, charge le dataset Phase 2.11 (_wt suffix)

    Returns:
        (predictions, labels)
        - predictions: (n_samples,) - binary [0,1]
        - labels: (n_samples,) - binary [0,1]
    """
    # Phase 2.11: Ajouter suffixe _wt si weighted=True
    suffix = "_wt" if weighted else ""
    dataset_pattern = f"dataset_btc_eth_bnb_ada_ltc_{indicator}_direction_only_{filter_type}{suffix}.npz"
    dataset_path = Path("data/prepared") / dataset_pattern

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {dataset_path}")

    print(f"📂 Chargement: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)

    # Load predictions
    Y_pred = data.get(f'Y_{split}_pred', None)
    if Y_pred is None:
        raise ValueError(f"Y_{split}_pred introuvable dans {dataset_path}")

    predictions = (Y_pred.flatten() > 0.5).astype(int)  # Binary [0,1]

    # Load labels
    Y = data[f'Y_{split}']
    labels = Y.flatten().astype(int)  # Binary [0,1]

    print(f"✅ Chargé: {len(predictions):,} samples")
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Labels shape: {labels.shape}")
    print()

    return predictions, labels


def analyze_transition_sync(predictions: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Analyze transition synchronization between Oracle and Model.

    Args:
        predictions: Binary predictions [0,1]
        labels: Binary labels [0,1]

    Returns:
        Dict with transition analysis metrics
    """
    n_samples = len(predictions)

    # Find Oracle transitions
    oracle_transitions = []
    oracle_transition_types = []
    for i in range(1, n_samples):
        if labels[i] != labels[i-1]:
            oracle_transitions.append(i)
            # Type: 0→1 (DOWN→UP) or 1→0 (UP→DOWN)
            transition_type = f"{labels[i-1]}→{labels[i]}"
            oracle_transition_types.append(transition_type)

    # Find Model transitions
    model_transitions = []
    model_transition_types = []
    for i in range(1, n_samples):
        if predictions[i] != predictions[i-1]:
            model_transitions.append(i)
            transition_type = f"{predictions[i-1]}→{predictions[i]}"
            model_transition_types.append(transition_type)

    print("=" * 80)
    print("ANALYSE DES TRANSITIONS")
    print("=" * 80)
    print(f"Total samples: {n_samples:,}")
    print(f"Oracle transitions: {len(oracle_transitions):,} ({len(oracle_transitions)/n_samples*100:.2f}%)")
    print(f"Model transitions: {len(model_transitions):,} ({len(model_transitions)/n_samples*100:.2f}%)")
    print()

    # Count transition types
    from collections import Counter
    oracle_types_count = Counter(oracle_transition_types)
    model_types_count = Counter(model_transition_types)

    print("Transition types (Oracle):")
    for ttype, count in oracle_types_count.items():
        print(f"  {ttype}: {count:,} ({count/len(oracle_transitions)*100:.1f}%)")
    print()

    print("Transition types (Model):")
    for ttype, count in model_types_count.items():
        print(f"  {ttype}: {count:,} ({count/len(model_transitions)*100:.1f}%)")
    print()

    # Measure synchronization
    # For each Oracle transition, check if Model transitions at same timestep
    model_synced = 0
    model_synced_correct = 0  # Same transition type
    model_synced_wrong = 0     # Opposite transition type
    model_not_synced = 0

    for i, oracle_idx in enumerate(oracle_transitions):
        oracle_type = oracle_transition_types[i]

        # Check if model transitions at same timestep
        if oracle_idx in model_transitions:
            model_synced += 1
            model_idx_in_list = model_transitions.index(oracle_idx)
            model_type = model_transition_types[model_idx_in_list]

            if model_type == oracle_type:
                model_synced_correct += 1
            else:
                model_synced_wrong += 1
        else:
            model_not_synced += 1

    print("=" * 80)
    print("SYNCHRONISATION DES TRANSITIONS")
    print("=" * 80)
    print(f"Oracle transitions: {len(oracle_transitions):,}")
    print()
    print(f"Model synced (transitions au même moment): {model_synced:,} ({model_synced/len(oracle_transitions)*100:.2f}%)")
    print(f"  - Correct (même type): {model_synced_correct:,} ({model_synced_correct/len(oracle_transitions)*100:.2f}%)")
    print(f"  - Wrong (type opposé): {model_synced_wrong:,} ({model_synced_wrong/len(oracle_transitions)*100:.2f}%)")
    print(f"Model NOT synced (pas de transition): {model_not_synced:,} ({model_not_synced/len(oracle_transitions)*100:.2f}%)")
    print()

    # Key metric: Transition Accuracy
    transition_accuracy = model_synced_correct / len(oracle_transitions) * 100

    print("=" * 80)
    print("🎯 MÉTRIQUE CLÉ: TRANSITION ACCURACY")
    print("=" * 80)
    print(f"Transition Accuracy: {transition_accuracy:.2f}%")
    print()

    if transition_accuracy >= 90:
        print("✅ EXCELLENT: Modèle détecte les retournements au bon moment (>90%)")
    elif transition_accuracy >= 70:
        print("⚠️ BON: Modèle détecte la plupart des retournements (70-90%)")
    elif transition_accuracy >= 50:
        print("❌ MOYEN: Modèle rate beaucoup de retournements (50-70%)")
    else:
        print("❌ CRITIQUE: Modèle rate la majorité des retournements (<50%)")
    print()

    # Compare to global accuracy
    global_accuracy = (predictions == labels).sum() / len(predictions) * 100

    print("=" * 80)
    print("COMPARAISON ACCURACY GLOBALE vs TRANSITIONS")
    print("=" * 80)
    print(f"Global Accuracy: {global_accuracy:.2f}%")
    print(f"Transition Accuracy: {transition_accuracy:.2f}%")
    print(f"Gap: {global_accuracy - transition_accuracy:+.2f}%")
    print()

    if transition_accuracy < global_accuracy - 10:
        print("❌ PROBLÈME IDENTIFIÉ:")
        print("   Le modèle est bon en 'continuation' mais MAUVAIS en 'retournement'")
        print("   → Entre en RETARD sur les transitions critiques")
        print("   → Explique le gap 92% accuracy → 34% Win Rate")
    elif transition_accuracy >= global_accuracy - 5:
        print("✅ COHÉRENT:")
        print("   Le modèle est aussi bon sur transitions que sur continuations")
        print("   → Le problème n'est PAS le timing d'entrée")
        print("   → Le problème est ailleurs (micro-sorties, frais, etc.)")
    print()

    # Analyze latency (when model transitions vs oracle)
    print("=" * 80)
    print("ANALYSE DE LATENCE (Model vs Oracle)")
    print("=" * 80)

    # Optimized latency calculation using numpy (O(n log n) instead of O(n²))
    oracle_arr = np.array(oracle_transitions)
    model_arr = np.array(model_transitions)

    latencies = []
    for oracle_idx in oracle_arr:
        # Find nearest model transition using binary search
        idx = np.searchsorted(model_arr, oracle_idx)

        # Check both neighbors (before and after)
        candidates = []
        if idx > 0:
            candidates.append(model_arr[idx - 1] - oracle_idx)  # Before
        if idx < len(model_arr):
            candidates.append(model_arr[idx] - oracle_idx)  # After

        # Pick nearest
        if candidates:
            min_distance = min(candidates, key=abs)
        else:
            min_distance = 0  # Shouldn't happen

        latencies.append(min_distance)

    latencies = np.array(latencies)

    print(f"Latence moyenne: {latencies.mean():.2f} périodes")
    print(f"Latence médiane: {np.median(latencies):.2f} périodes")
    print(f"Latence std: {latencies.std():.2f} périodes")
    print()

    print("Distribution latence:")
    print(f"  Avance (-3 à -1): {(latencies < 0).sum():,} ({(latencies < 0).sum()/len(latencies)*100:.1f}%)")
    print(f"  Synchro (0): {(latencies == 0).sum():,} ({(latencies == 0).sum()/len(latencies)*100:.1f}%)")
    print(f"  Retard (+1 à +3): {(latencies > 0).sum():,} ({(latencies > 0).sum()/len(latencies)*100:.1f}%)")
    print()

    if latencies.mean() > 1:
        print(f"❌ RETARD DÉTECTÉ: Modèle transition en moyenne {latencies.mean():.1f} périodes APRÈS l'Oracle")
        print("   → Entre trop tard → Rate le début du mouvement")
    elif latencies.mean() < -1:
        print(f"⚠️ AVANCE DÉTECTÉE: Modèle transition en moyenne {abs(latencies.mean()):.1f} périodes AVANT l'Oracle")
        print("   → Entre trop tôt → Possibles faux signaux")
    else:
        print("✅ SYNCHRONISÉ: Modèle transition en moyenne au même moment que l'Oracle")
    print()

    return {
        'n_samples': n_samples,
        'oracle_transitions': len(oracle_transitions),
        'model_transitions': len(model_transitions),
        'model_synced': model_synced,
        'model_synced_correct': model_synced_correct,
        'model_synced_wrong': model_synced_wrong,
        'model_not_synced': model_not_synced,
        'transition_accuracy': transition_accuracy,
        'global_accuracy': global_accuracy,
        'latency_mean': latencies.mean(),
        'latency_median': np.median(latencies),
        'latency_std': latencies.std()
    }


def main():
    parser = argparse.ArgumentParser(description="Test Transition Synchronization")
    parser.add_argument('--indicator', type=str, default='macd', choices=['macd', 'rsi', 'cci'],
                        help="Indicateur à tester (défaut: macd)")
    parser.add_argument('--filter-type', type=str, default='kalman', choices=['kalman', 'octave20'],
                        help="Filtre à utiliser (défaut: kalman)")
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help="Split à tester (défaut: test)")
    parser.add_argument('--weighted', action='store_true',
                        help="Phase 2.11: Tester le modèle avec WeightedTransitionBCELoss (_wt suffix)")

    args = parser.parse_args()

    print("=" * 80)
    print("TEST TRANSITION SYNCHRONIZATION")
    print("=" * 80)
    print(f"Indicateur: {args.indicator.upper()}")
    print(f"Filtre: {args.filter_type}")
    print(f"Split: {args.split}")
    if args.weighted:
        print(f"Phase 2.11: WeightedTransitionBCELoss (dataset _wt)")
    print()

    # Load data
    predictions, labels = load_predictions_and_labels(args.indicator, args.filter_type, args.split, args.weighted)

    # Analyze transitions
    results = analyze_transition_sync(predictions, labels)

    # Summary
    print("=" * 80)
    print("RÉSUMÉ")
    print("=" * 80)
    print(f"Global Accuracy: {results['global_accuracy']:.2f}%")
    print(f"Transition Accuracy: {results['transition_accuracy']:.2f}%")
    print(f"Gap: {results['global_accuracy'] - results['transition_accuracy']:+.2f}%")
    print()
    print(f"Latence moyenne: {results['latency_mean']:+.2f} périodes")
    print()

    if results['transition_accuracy'] >= 90:
        print("✅ Modèle excellent sur transitions - Le timing n'est PAS le problème")
    elif results['transition_accuracy'] < 70:
        print("❌ Modèle mauvais sur transitions - PROBLÈME IDENTIFIÉ!")
        print("   → Modèle entre en retard sur les retournements")
        print("   → Explique pourquoi trading échoue malgré 92% accuracy globale")
    else:
        print("⚠️ Modèle correct sur transitions mais perfectible")

    print("=" * 80)


if __name__ == '__main__':
    main()
