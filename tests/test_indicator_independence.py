#!/usr/bin/env python3
"""
Test d'indépendance des indicateurs RSI/CCI/MACD.

Vérifie si les indicateurs capturent des informations différentes ou similaires:
1. Corrélation des labels Oracle (directions)
2. Recouvrement des erreurs ML (prédictions)
3. Complémentarité (quand A se trompe, B a raison?)

Usage:
    python tests/test_indicator_independence.py --split test
    python tests/test_indicator_independence.py --split test --use-predictions
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def load_dataset(indicator: str, filter_type: str = 'kalman') -> Dict:
    """Charge un dataset direction-only."""
    data_dir = Path('data/prepared')
    pattern = f'dataset_btc_eth_bnb_ada_ltc_{indicator}_direction_only_{filter_type}.npz'
    path = data_dir / pattern

    if not path.exists():
        raise FileNotFoundError(f"Dataset non trouvé: {path}")

    data = np.load(path, allow_pickle=True)

    # Les datasets utilisent X_train/X_val/X_test au lieu de X
    result = {}
    for split in ['train', 'val', 'test']:
        x_key = f'X_{split}'
        y_key = f'Y_{split}'
        if x_key in data and y_key in data:
            result[f'X_{split}'] = data[x_key]
            result[f'Y_{split}'] = data[y_key]

    # Aussi charger les prédictions si disponibles
    for split in ['train', 'val', 'test']:
        pred_key = f'Y_{split}_pred'
        if pred_key in data:
            result[pred_key] = data[pred_key]

    return result


def get_split_data(data: Dict, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extrait les données pour un split donné."""
    x_key = f'X_{split}'
    y_key = f'Y_{split}'

    if x_key not in data or y_key not in data:
        raise ValueError(f"Split '{split}' non trouvé. Clés disponibles: {list(data.keys())}")

    return data[x_key], data[y_key]


def get_predictions_from_dataset(data: Dict, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Récupère les prédictions déjà stockées dans le dataset."""
    pred_key = f'Y_{split}_pred'

    if pred_key not in data:
        raise ValueError(f"Prédictions '{pred_key}' non trouvées. Clés disponibles: {list(data.keys())}")

    probs = data[pred_key].flatten()
    binary_preds = (probs > 0.5).astype(int)

    return binary_preds, probs


def analyze_oracle_correlation(labels: Dict[str, np.ndarray]) -> Dict:
    """Analyse la corrélation entre les labels Oracle."""
    indicators = list(labels.keys())
    n = len(indicators)

    results = {
        'correlation_matrix': np.zeros((n, n)),
        'agreement_matrix': np.zeros((n, n)),
        'pairs': {}
    }

    for i, ind1 in enumerate(indicators):
        for j, ind2 in enumerate(indicators):
            if i <= j:
                # Corrélation de Pearson
                corr = np.corrcoef(labels[ind1], labels[ind2])[0, 1]
                results['correlation_matrix'][i, j] = corr
                results['correlation_matrix'][j, i] = corr

                # Taux d'accord
                agreement = np.mean(labels[ind1] == labels[ind2])
                results['agreement_matrix'][i, j] = agreement
                results['agreement_matrix'][j, i] = agreement

                if i < j:
                    results['pairs'][f'{ind1}-{ind2}'] = {
                        'correlation': corr,
                        'agreement': agreement,
                        'both_up': np.mean((labels[ind1] == 1) & (labels[ind2] == 1)),
                        'both_down': np.mean((labels[ind1] == 0) & (labels[ind2] == 0)),
                        'disagree': np.mean(labels[ind1] != labels[ind2])
                    }

    results['indicators'] = indicators
    return results


def analyze_error_overlap(labels: Dict[str, np.ndarray],
                          predictions: Dict[str, np.ndarray]) -> Dict:
    """Analyse le recouvrement des erreurs entre indicateurs."""
    indicators = list(labels.keys())

    # Calculer les erreurs pour chaque indicateur
    errors = {}
    for ind in indicators:
        errors[ind] = (predictions[ind] != labels[ind])

    results = {
        'error_rates': {},
        'overlap_matrix': {},
        'complementarity': {}
    }

    # Taux d'erreur par indicateur
    for ind in indicators:
        results['error_rates'][ind] = np.mean(errors[ind])

    # Analyse par paire
    for i, ind1 in enumerate(indicators):
        for j, ind2 in enumerate(indicators):
            if i < j:
                pair_key = f'{ind1}-{ind2}'

                # Erreurs communes
                both_wrong = errors[ind1] & errors[ind2]
                either_wrong = errors[ind1] | errors[ind2]

                # Recouvrement = erreurs communes / total erreurs
                if np.sum(either_wrong) > 0:
                    overlap = np.sum(both_wrong) / np.sum(either_wrong)
                else:
                    overlap = 0

                results['overlap_matrix'][pair_key] = {
                    'both_wrong': np.mean(both_wrong),
                    'either_wrong': np.mean(either_wrong),
                    'overlap_ratio': overlap,
                    # Jaccard index des erreurs
                    'jaccard': np.sum(both_wrong) / np.sum(either_wrong) if np.sum(either_wrong) > 0 else 0
                }

                # Complémentarité: quand A se trompe, B a raison?
                a_wrong_b_right = errors[ind1] & ~errors[ind2]
                b_wrong_a_right = errors[ind2] & ~errors[ind1]

                results['complementarity'][pair_key] = {
                    f'{ind1}_wrong_{ind2}_right': np.mean(a_wrong_b_right),
                    f'{ind2}_wrong_{ind1}_right': np.mean(b_wrong_a_right),
                    'complementarity_score': np.mean(a_wrong_b_right) + np.mean(b_wrong_a_right)
                }

    return results


def analyze_conditional_accuracy(labels: Dict[str, np.ndarray],
                                  predictions: Dict[str, np.ndarray]) -> Dict:
    """Analyse l'accuracy conditionnelle (quand A prédit X, B a raison?)."""
    indicators = list(labels.keys())
    results = {}

    for i, ind1 in enumerate(indicators):
        for j, ind2 in enumerate(indicators):
            if i != j:
                pair_key = f'{ind1}_predicts_{ind2}'

                # Quand ind1 prédit UP (1)
                mask_up = predictions[ind1] == 1
                # Quand ind1 prédit DOWN (0)
                mask_down = predictions[ind1] == 0

                # Accuracy de ind2 conditionnée sur la prédiction de ind1
                if np.sum(mask_up) > 0:
                    acc_when_up = np.mean(predictions[ind2][mask_up] == labels[ind2][mask_up])
                else:
                    acc_when_up = 0

                if np.sum(mask_down) > 0:
                    acc_when_down = np.mean(predictions[ind2][mask_down] == labels[ind2][mask_down])
                else:
                    acc_when_down = 0

                results[pair_key] = {
                    f'{ind2}_acc_when_{ind1}_up': acc_when_up,
                    f'{ind2}_acc_when_{ind1}_down': acc_when_down,
                    'samples_up': np.sum(mask_up),
                    'samples_down': np.sum(mask_down)
                }

    return results


def analyze_voting_potential(labels: Dict[str, np.ndarray],
                             predictions: Dict[str, np.ndarray]) -> Dict:
    """Analyse le potentiel du vote majoritaire."""
    indicators = list(labels.keys())
    n_samples = len(labels[indicators[0]])

    # Vote majoritaire (2/3 ou 3/3 d'accord)
    pred_matrix = np.column_stack([predictions[ind] for ind in indicators])
    votes = np.sum(pred_matrix, axis=1)  # 0, 1, 2, ou 3

    # Majorité: >= 2 = UP, < 2 = DOWN
    majority_pred = (votes >= 2).astype(int)

    # Unanimité: 0 ou 3
    unanimous = (votes == 0) | (votes == 3)

    # Référence: utiliser un indicateur comme ground truth (ex: MACD qui a le meilleur Oracle)
    # Mais on veut mesurer si le vote améliore vs chaque indicateur individuellement

    results = {
        'vote_distribution': {
            '0_down': np.mean(votes == 0),
            '1_down': np.mean(votes == 1),
            '2_up': np.mean(votes == 2),
            '3_up': np.mean(votes == 3)
        },
        'unanimous_rate': np.mean(unanimous),
        'split_rate': np.mean(~unanimous),
        'individual_accuracies': {},
        'majority_vs_individual': {}
    }

    # Accuracy individuelle vs labels de chaque indicateur
    for ind in indicators:
        ind_acc = np.mean(predictions[ind] == labels[ind])
        majority_acc = np.mean(majority_pred == labels[ind])

        results['individual_accuracies'][ind] = ind_acc
        results['majority_vs_individual'][ind] = {
            'individual_acc': ind_acc,
            'majority_acc': majority_acc,
            'delta': majority_acc - ind_acc
        }

    # Quand unanime, accuracy?
    if np.sum(unanimous) > 0:
        for ind in indicators:
            results[f'unanimous_acc_{ind}'] = np.mean(
                majority_pred[unanimous] == labels[ind][unanimous]
            )

    # Quand split (1 vs 2), qui a raison?
    split_mask = ~unanimous
    if np.sum(split_mask) > 0:
        results['split_analysis'] = {}
        for ind in indicators:
            # L'indicateur minoritaire a-t-il plus souvent raison?
            minority_mask = split_mask & (predictions[ind] != majority_pred)
            if np.sum(minority_mask) > 0:
                minority_correct = np.mean(predictions[ind][minority_mask] == labels[ind][minority_mask])
                results['split_analysis'][f'{ind}_minority_correct'] = minority_correct

    return results


def print_results(oracle_results: Dict, error_results: Dict = None,
                  conditional_results: Dict = None, voting_results: Dict = None):
    """Affiche les résultats de manière formatée."""

    print("\n" + "="*80)
    print("ANALYSE D'INDÉPENDANCE DES INDICATEURS RSI/CCI/MACD")
    print("="*80)

    # 1. Corrélation Oracle
    print("\n" + "-"*40)
    print("1. CORRÉLATION DES LABELS ORACLE")
    print("-"*40)

    indicators = oracle_results['indicators']
    print("\nMatrice de corrélation (Pearson):")
    print(f"{'':>8}", end='')
    for ind in indicators:
        print(f"{ind:>10}", end='')
    print()
    for i, ind1 in enumerate(indicators):
        print(f"{ind1:>8}", end='')
        for j, ind2 in enumerate(indicators):
            print(f"{oracle_results['correlation_matrix'][i,j]:>10.3f}", end='')
        print()

    print("\nMatrice d'accord (% mêmes labels):")
    print(f"{'':>8}", end='')
    for ind in indicators:
        print(f"{ind:>10}", end='')
    print()
    for i, ind1 in enumerate(indicators):
        print(f"{ind1:>8}", end='')
        for j, ind2 in enumerate(indicators):
            print(f"{oracle_results['agreement_matrix'][i,j]*100:>9.1f}%", end='')
        print()

    print("\nDétail par paire:")
    for pair, stats in oracle_results['pairs'].items():
        print(f"\n  {pair}:")
        print(f"    Corrélation: {stats['correlation']:.3f}")
        print(f"    Accord: {stats['agreement']*100:.1f}%")
        print(f"    Désaccord: {stats['disagree']*100:.1f}%")
        print(f"    Both UP: {stats['both_up']*100:.1f}% | Both DOWN: {stats['both_down']*100:.1f}%")

    # 2. Recouvrement des erreurs (si predictions)
    if error_results:
        print("\n" + "-"*40)
        print("2. RECOUVREMENT DES ERREURS ML")
        print("-"*40)

        print("\nTaux d'erreur par indicateur:")
        for ind, rate in error_results['error_rates'].items():
            print(f"  {ind}: {rate*100:.2f}%")

        print("\nRecouvrement des erreurs par paire:")
        for pair, stats in error_results['overlap_matrix'].items():
            print(f"\n  {pair}:")
            print(f"    Erreurs communes: {stats['both_wrong']*100:.2f}%")
            print(f"    Erreurs totales (union): {stats['either_wrong']*100:.2f}%")
            print(f"    Ratio recouvrement: {stats['overlap_ratio']*100:.1f}%")
            print(f"    Jaccard index: {stats['jaccard']:.3f}")

        print("\nComplémentarité (quand A se trompe, B a raison?):")
        for pair, stats in error_results['complementarity'].items():
            print(f"\n  {pair}:")
            for key, value in stats.items():
                if 'score' in key:
                    print(f"    Score complémentarité: {value*100:.2f}%")
                else:
                    print(f"    {key}: {value*100:.2f}%")

    # 3. Accuracy conditionnelle
    if conditional_results:
        print("\n" + "-"*40)
        print("3. ACCURACY CONDITIONNELLE")
        print("-"*40)
        print("(Quand A prédit X, quelle est l'accuracy de B?)")

        for pair, stats in conditional_results.items():
            ind1, _, ind2 = pair.split('_')
            print(f"\n  Quand {ind1.upper()} prédit:")
            for key, value in stats.items():
                if 'acc' in key:
                    if 'up' in key:
                        print(f"    UP → {ind2.upper()} accuracy: {value*100:.1f}%")
                    elif 'down' in key:
                        print(f"    DOWN → {ind2.upper()} accuracy: {value*100:.1f}%")

    # 4. Analyse du vote
    if voting_results:
        print("\n" + "-"*40)
        print("4. POTENTIEL DU VOTE MAJORITAIRE")
        print("-"*40)

        print("\nDistribution des votes:")
        for vote, pct in voting_results['vote_distribution'].items():
            print(f"  {vote}: {pct*100:.1f}%")

        print(f"\nTaux d'unanimité (0 ou 3): {voting_results['unanimous_rate']*100:.1f}%")
        print(f"Taux de split (1 vs 2): {voting_results['split_rate']*100:.1f}%")

        print("\nAccuracy individuelle vs vote majoritaire:")
        for ind, stats in voting_results['majority_vs_individual'].items():
            delta_str = f"+{stats['delta']*100:.2f}%" if stats['delta'] >= 0 else f"{stats['delta']*100:.2f}%"
            print(f"  {ind}: individuel {stats['individual_acc']*100:.1f}% → majorité {stats['majority_acc']*100:.1f}% ({delta_str})")

    # 5. Conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    # Analyser les résultats
    avg_correlation = np.mean([s['correlation'] for s in oracle_results['pairs'].values()])
    avg_agreement = np.mean([s['agreement'] for s in oracle_results['pairs'].values()])

    print(f"\nCorrélation moyenne des labels Oracle: {avg_correlation:.3f}")
    print(f"Accord moyen des labels Oracle: {avg_agreement*100:.1f}%")

    if avg_correlation > 0.8:
        print("\n⚠️ CORRÉLATION ÉLEVÉE (>0.8): Les indicateurs capturent un signal SIMILAIRE")
    elif avg_correlation > 0.5:
        print("\n⚡ CORRÉLATION MODÉRÉE (0.5-0.8): Les indicateurs ont des composantes communes ET différentes")
    else:
        print("\n✅ CORRÉLATION FAIBLE (<0.5): Les indicateurs capturent des signaux DIFFÉRENTS")

    if error_results:
        avg_overlap = np.mean([s['overlap_ratio'] for s in error_results['overlap_matrix'].values()])
        avg_complementarity = np.mean([s['complementarity_score'] for s in error_results['complementarity'].values()])

        print(f"\nRecouvrement moyen des erreurs ML: {avg_overlap*100:.1f}%")
        print(f"Score de complémentarité moyen: {avg_complementarity*100:.1f}%")

        if avg_overlap > 0.7:
            print("\n⚠️ ERREURS TRÈS CORRÉLÉES: Les modèles font les mêmes erreurs")
        elif avg_overlap > 0.4:
            print("\n⚡ ERREURS PARTIELLEMENT CORRÉLÉES: Potentiel de complémentarité limité")
        else:
            print("\n✅ ERREURS DÉCORRÉLÉES: Fort potentiel de complémentarité!")


def main():
    parser = argparse.ArgumentParser(description='Test d\'indépendance des indicateurs')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--use-predictions', action='store_true',
                        help='Analyser aussi les prédictions ML (pas seulement Oracle)')
    parser.add_argument('--filter', type=str, default='kalman', choices=['kalman', 'octave20'])
    args = parser.parse_args()

    print(f"Chargement des datasets ({args.filter})...")

    indicators = ['rsi', 'cci', 'macd']

    # Charger les données
    datasets = {}
    labels = {}
    X_data = {}

    for ind in indicators:
        try:
            datasets[ind] = load_dataset(ind, args.filter)
            X, Y = get_split_data(datasets[ind], args.split)
            X_data[ind] = X
            labels[ind] = Y.flatten()
            print(f"  {ind.upper()}: {len(labels[ind])} samples")
        except FileNotFoundError as e:
            print(f"  ⚠️ {ind.upper()}: {e}")

    if len(labels) < 2:
        print("Erreur: Au moins 2 indicateurs requis")
        return

    # 2. Charger les prédictions ML (si demandé) AVANT d'aligner les tailles
    error_results = None
    conditional_results = None
    voting_results = None
    predictions = {}
    probs = {}

    if args.use_predictions:
        print("\nChargement des prédictions depuis les datasets...")

        for ind in indicators:
            if ind in datasets:
                try:
                    preds, prob = get_predictions_from_dataset(datasets[ind], args.split)
                    predictions[ind] = preds
                    probs[ind] = prob
                    print(f"  {ind.upper()}: {len(preds)} prédictions chargées")
                except Exception as e:
                    print(f"  ⚠️ {ind.upper()}: {e}")

    # Aligner les tailles (prendre le minimum entre labels ET predictions)
    all_sizes = [len(l) for l in labels.values()]
    if predictions:
        all_sizes.extend([len(p) for p in predictions.values()])

    min_size = min(all_sizes)
    print(f"\nTaille alignée: {min_size} samples")

    for ind in labels:
        labels[ind] = labels[ind][:min_size]
        X_data[ind] = X_data[ind][:min_size]

    for ind in predictions:
        predictions[ind] = predictions[ind][:min_size]
        probs[ind] = probs[ind][:min_size]

    # 1. Analyse Oracle (labels)
    oracle_results = analyze_oracle_correlation(labels)

    # 3. Analyse des erreurs ML
    if len(predictions) >= 2:
        error_results = analyze_error_overlap(labels, predictions)
        conditional_results = analyze_conditional_accuracy(labels, predictions)
        voting_results = analyze_voting_potential(labels, predictions)

    # Afficher les résultats
    print_results(oracle_results, error_results, conditional_results, voting_results)


if __name__ == '__main__':
    main()
