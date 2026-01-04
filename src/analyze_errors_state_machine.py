"""
Analyse des erreurs du modèle pour déduire des règles de state machine.

Principe: Les accords sont sans intérêt, les désaccords contiennent toute l'information.

Ce script analyse les erreurs (prédiction != Oracle) et identifie les patterns
pour construire une machine à états qui améliore les décisions.

Usage:
    # Analyse simple (un seul filtre)
    python src/analyze_errors_state_machine.py \
        --data data/prepared/dataset_btc_eth_bnb_ada_ltc_ohlcv2_rsi_octave20.npz \
        --split test

    # Analyse avec comparaison de filtres
    python src/analyze_errors_state_machine.py \
        --data data/prepared/dataset_btc_eth_bnb_ada_ltc_ohlcv2_rsi_octave20.npz \
        --data-kalman data/prepared/dataset_btc_eth_bnb_ada_ltc_ohlcv2_rsi_kalman.npz \
        --split test \
        --output results/error_analysis.csv
"""

import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
from collections import Counter


def load_dataset(path: str, split: str = 'test') -> dict:
    """
    Charge un dataset et retourne les données pour le split demandé.

    Returns:
        dict avec X, Y (labels), Y_pred (prédictions si disponibles), metadata
    """
    data = np.load(path, allow_pickle=True)

    result = {
        'X': data[f'X_{split}'],
        'Y': data[f'Y_{split}'],
        'Y_pred': None,
        'metadata': None
    }

    # Charger les prédictions si disponibles
    pred_key = f'Y_{split}_pred'
    if pred_key in data:
        result['Y_pred'] = data[pred_key]

    # Charger métadonnées
    if 'metadata' in data:
        try:
            result['metadata'] = json.loads(str(data['metadata']))
        except:
            pass

    return result


def analyze_errors(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    Y_kalman: np.ndarray = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Analyse les erreurs et crée un DataFrame avec contexte.

    Args:
        Y_true: Labels Oracle (Octave ou Kalman selon le dataset principal)
        Y_pred: Prédictions du modèle
        Y_kalman: Labels Kalman (optionnel, pour comparer les filtres)
        verbose: Afficher les statistiques

    Returns:
        DataFrame avec colonnes: idx, label, pred, is_error, octave_dir, kalman_dir, filters_agree
    """
    n_samples = len(Y_true)

    # Aplatir si nécessaire (single-output)
    if Y_true.ndim > 1:
        Y_true = Y_true.flatten()
    if Y_pred.ndim > 1:
        Y_pred = Y_pred.flatten()
    if Y_kalman is not None and Y_kalman.ndim > 1:
        Y_kalman = Y_kalman.flatten()

    # Créer le DataFrame
    df = pd.DataFrame({
        'idx': np.arange(n_samples),
        'label': Y_true.astype(int),
        'pred': Y_pred.astype(int),
    })

    # Calculer les erreurs
    df['is_error'] = (df['pred'] != df['label']).astype(int)
    df['error_type'] = 'correct'
    df.loc[(df['pred'] == 1) & (df['label'] == 0), 'error_type'] = 'false_positive'  # Prédit UP, était DOWN
    df.loc[(df['pred'] == 0) & (df['label'] == 1), 'error_type'] = 'false_negative'  # Prédit DOWN, était UP

    # Direction selon le filtre principal (Octave si c'est le dataset principal)
    df['octave_dir'] = df['label'].map({0: 'DOWN', 1: 'UP'})

    # Si on a les labels Kalman, ajouter la comparaison
    if Y_kalman is not None:
        df['kalman_dir'] = pd.Series(Y_kalman.astype(int)).map({0: 'DOWN', 1: 'UP'})
        df['filters_agree'] = (df['label'] == Y_kalman.astype(int)).astype(int)
    else:
        df['kalman_dir'] = None
        df['filters_agree'] = None

    # Statistiques
    if verbose:
        n_errors = df['is_error'].sum()
        accuracy = 1 - (n_errors / n_samples)

        print("\n" + "="*80)
        print("ANALYSE DES ERREURS")
        print("="*80)
        print(f"\n📊 Statistiques globales:")
        print(f"   Total samples: {n_samples:,}")
        print(f"   Erreurs: {n_errors:,} ({n_errors/n_samples*100:.1f}%)")
        print(f"   Accuracy: {accuracy*100:.1f}%")

        # Breakdown par type d'erreur
        error_counts = df['error_type'].value_counts()
        print(f"\n📈 Types d'erreurs:")
        for error_type, count in error_counts.items():
            print(f"   {error_type}: {count:,} ({count/n_samples*100:.1f}%)")

        # Si on a les filtres
        if Y_kalman is not None:
            n_agree = df['filters_agree'].sum()
            n_disagree = n_samples - n_agree

            print(f"\n🔀 Accord des filtres:")
            print(f"   Accord (Octave == Kalman): {n_agree:,} ({n_agree/n_samples*100:.1f}%)")
            print(f"   Désaccord: {n_disagree:,} ({n_disagree/n_samples*100:.1f}%)")

            # Erreurs dans les zones d'accord vs désaccord
            errors_in_agree = df[df['filters_agree'] == 1]['is_error'].sum()
            errors_in_disagree = df[df['filters_agree'] == 0]['is_error'].sum()

            print(f"\n🎯 Erreurs par zone:")
            print(f"   Erreurs quand filtres d'accord: {errors_in_agree:,} ({errors_in_agree/n_agree*100:.1f}% des accords)")
            print(f"   Erreurs quand filtres en désaccord: {errors_in_disagree:,} ({errors_in_disagree/n_disagree*100:.1f}% des désaccords)")

    return df


def analyze_error_patterns(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Analyse les patterns dans les erreurs.

    Returns:
        dict avec les patterns identifiés
    """
    patterns = {}

    # Filtrer seulement les erreurs
    errors_df = df[df['is_error'] == 1].copy()

    if len(errors_df) == 0:
        if verbose:
            print("\n✅ Aucune erreur à analyser!")
        return patterns

    if verbose:
        print("\n" + "="*80)
        print("PATTERNS D'ERREURS")
        print("="*80)

    # Pattern 1: Erreurs consécutives (blocs)
    error_indices = errors_df['idx'].values
    if len(error_indices) > 1:
        gaps = np.diff(error_indices)
        n_consecutive = np.sum(gaps == 1)
        n_isolated = np.sum(gaps > 1) + 1  # +1 pour le premier

        # Calculer la taille des blocs
        block_sizes = []
        current_block = 1
        for gap in gaps:
            if gap == 1:
                current_block += 1
            else:
                block_sizes.append(current_block)
                current_block = 1
        block_sizes.append(current_block)

        patterns['block_sizes'] = block_sizes
        patterns['avg_block_size'] = np.mean(block_sizes)
        patterns['max_block_size'] = max(block_sizes)
        patterns['n_blocks'] = len(block_sizes)
        patterns['n_isolated'] = sum(1 for s in block_sizes if s == 1)

        if verbose:
            print(f"\n📦 Blocs d'erreurs:")
            print(f"   Nombre de blocs: {len(block_sizes)}")
            print(f"   Taille moyenne: {np.mean(block_sizes):.1f}")
            print(f"   Taille max: {max(block_sizes)}")
            print(f"   Erreurs isolées (1 sample): {patterns['n_isolated']} ({patterns['n_isolated']/len(block_sizes)*100:.1f}%)")

    # Pattern 2: Erreurs par zone de filtre (si disponible)
    if 'filters_agree' in df.columns and df['filters_agree'].notna().any():
        errors_by_agree = errors_df.groupby('filters_agree').size()
        total_by_agree = df.groupby('filters_agree').size()

        patterns['error_rate_when_agree'] = errors_by_agree.get(1, 0) / total_by_agree.get(1, 1)
        patterns['error_rate_when_disagree'] = errors_by_agree.get(0, 0) / total_by_agree.get(0, 1)

        if verbose:
            print(f"\n🔀 Taux d'erreur par zone de filtre:")
            print(f"   Quand filtres d'accord: {patterns['error_rate_when_agree']*100:.1f}%")
            print(f"   Quand filtres en désaccord: {patterns['error_rate_when_disagree']*100:.1f}%")

    # Pattern 3: Transitions (erreurs après un changement de direction Oracle)
    df_with_prev = df.copy()
    df_with_prev['prev_label'] = df_with_prev['label'].shift(1)
    df_with_prev['label_changed'] = (df_with_prev['label'] != df_with_prev['prev_label']).astype(int)

    errors_after_change = df_with_prev[(df_with_prev['is_error'] == 1) & (df_with_prev['label_changed'] == 1)]
    errors_no_change = df_with_prev[(df_with_prev['is_error'] == 1) & (df_with_prev['label_changed'] == 0)]

    total_changes = df_with_prev['label_changed'].sum()
    total_no_changes = len(df_with_prev) - total_changes

    patterns['error_rate_after_change'] = len(errors_after_change) / max(total_changes, 1)
    patterns['error_rate_no_change'] = len(errors_no_change) / max(total_no_changes, 1)

    if verbose:
        print(f"\n🔄 Erreurs et transitions Oracle:")
        print(f"   Après changement de direction: {len(errors_after_change):,} ({patterns['error_rate_after_change']*100:.1f}%)")
        print(f"   Sans changement: {len(errors_no_change):,} ({patterns['error_rate_no_change']*100:.1f}%)")

    return patterns


def generate_rules_suggestions(df: pd.DataFrame, patterns: dict, verbose: bool = True) -> list:
    """
    Génère des suggestions de règles pour la state machine basées sur les patterns.

    Returns:
        list de règles suggérées (strings)
    """
    rules = []

    if verbose:
        print("\n" + "="*80)
        print("SUGGESTIONS DE RÈGLES POUR LA STATE MACHINE")
        print("="*80)

    # Règle 1: Si les erreurs sont plus fréquentes en zone de désaccord
    if 'error_rate_when_disagree' in patterns and 'error_rate_when_agree' in patterns:
        ratio = patterns['error_rate_when_disagree'] / max(patterns['error_rate_when_agree'], 0.001)
        if ratio > 1.5:
            rule = f"RÈGLE 1: Quand filtres en désaccord (Octave != Kalman), augmenter la confirmation requise"
            rules.append(rule)
            if verbose:
                print(f"\n✅ {rule}")
                print(f"   Ratio erreur désaccord/accord: {ratio:.1f}x")

    # Règle 2: Si les erreurs sont groupées en blocs
    if 'avg_block_size' in patterns and patterns['avg_block_size'] > 2:
        rule = f"RÈGLE 2: Les erreurs arrivent en blocs (taille moyenne: {patterns['avg_block_size']:.1f}). Implémenter un filtre de lissage sur les prédictions."
        rules.append(rule)
        if verbose:
            print(f"\n✅ {rule}")

    # Règle 3: Si beaucoup d'erreurs isolées
    if 'n_isolated' in patterns and 'n_blocks' in patterns:
        isolated_ratio = patterns['n_isolated'] / patterns['n_blocks']
        if isolated_ratio > 0.5:
            rule = f"RÈGLE 3: {isolated_ratio*100:.0f}% des erreurs sont isolées. Implémenter une confirmation sur 2+ périodes avant d'agir."
            rules.append(rule)
            if verbose:
                print(f"\n✅ {rule}")

    # Règle 4: Si les erreurs sont plus fréquentes après transition
    if 'error_rate_after_change' in patterns and 'error_rate_no_change' in patterns:
        ratio = patterns['error_rate_after_change'] / max(patterns['error_rate_no_change'], 0.001)
        if ratio > 1.5:
            rule = f"RÈGLE 4: Les erreurs sont {ratio:.1f}x plus fréquentes après un changement de direction Oracle. Implémenter un délai de stabilisation."
            rules.append(rule)
            if verbose:
                print(f"\n✅ {rule}")

    if verbose and not rules:
        print("\n⚠️ Pas de patterns clairs détectés. Le modèle semble bien calibré.")

    return rules


def main():
    parser = argparse.ArgumentParser(
        description="Analyse des erreurs du modèle pour déduire des règles de state machine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--data', '-d', type=str, required=True,
                        help='Chemin vers le dataset principal (.npz avec prédictions)')
    parser.add_argument('--data-kalman', '-k', type=str, default=None,
                        help='Chemin vers le dataset Kalman (optionnel, pour comparer les filtres)')
    parser.add_argument('--split', '-s', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Split à analyser')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Chemin de sortie pour le CSV (optionnel)')
    parser.add_argument('--sample', type=int, default=None,
                        help='Nombre de samples à analyser (optionnel, pour test rapide)')

    args = parser.parse_args()

    print("="*80)
    print("ANALYSE DES ERREURS - State Machine Learning")
    print("="*80)

    # Charger le dataset principal
    print(f"\n📂 Chargement du dataset principal...")
    print(f"   {args.data}")
    data_main = load_dataset(args.data, args.split)

    if data_main['Y_pred'] is None:
        print("\n❌ ERREUR: Le dataset ne contient pas de prédictions!")
        print("   Exécutez d'abord train.py pour générer les prédictions.")
        return

    Y_true = data_main['Y']
    Y_pred = data_main['Y_pred']

    print(f"   Split: {args.split}")
    print(f"   Samples: {len(Y_true):,}")

    # Charger le dataset Kalman si fourni
    Y_kalman = None
    if args.data_kalman:
        print(f"\n📂 Chargement du dataset Kalman...")
        print(f"   {args.data_kalman}")
        data_kalman = load_dataset(args.data_kalman, args.split)
        Y_kalman = data_kalman['Y']
        print(f"   Samples: {len(Y_kalman):,}")

    # Échantillonnage si demandé
    if args.sample and args.sample < len(Y_true):
        print(f"\n🎲 Échantillonnage: {args.sample:,} samples")
        indices = np.random.choice(len(Y_true), args.sample, replace=False)
        indices = np.sort(indices)
        Y_true = Y_true[indices]
        Y_pred = Y_pred[indices]
        if Y_kalman is not None:
            Y_kalman = Y_kalman[indices]

    # Analyse des erreurs
    df = analyze_errors(Y_true, Y_pred, Y_kalman)

    # Analyse des patterns
    patterns = analyze_error_patterns(df)

    # Générer des suggestions de règles
    rules = generate_rules_suggestions(df, patterns)

    # Sauvegarder le CSV si demandé
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n💾 Résultats sauvegardés: {output_path}")

    # Résumé final
    print("\n" + "="*80)
    print("RÉSUMÉ")
    print("="*80)
    n_errors = df['is_error'].sum()
    accuracy = 1 - (n_errors / len(df))
    print(f"\n   Accuracy: {accuracy*100:.1f}%")
    print(f"   Erreurs: {n_errors:,}")
    if rules:
        print(f"   Règles suggérées: {len(rules)}")
    print("")


if __name__ == '__main__':
    main()
