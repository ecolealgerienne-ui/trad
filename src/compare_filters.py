"""
Script de comparaison des filtres Octave vs Kalman pour architecture Dual-Binary.

Objectifs:
1. Comparer les labels Direction + Force entre Octave et Kalman
2. Mesurer la concordance (% labels identiques)
3. Mesurer le lag (déphasage temporel)
4. Vérifier l'alignement des index
5. Analyser les patterns de désaccord

Usage:
    # Comparer RSI Octave vs Kalman
    python src/compare_filters.py \
        --indicator rsi \
        --file-octave data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_dual_binary_octave20.npz \
        --file-kalman data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_dual_binary_kalman.npz \
        --split test

    # Comparer MACD avec échantillon
    python src/compare_filters.py \
        --indicator macd \
        --file-octave data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_octave20.npz \
        --file-kalman data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_kalman.npz \
        --split test \
        --sample 20000
"""

import numpy as np
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch


# Device global
DEVICE = torch.device('cpu')


def load_dataset(path: str, split: str = 'test') -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
    """
    Charge un dataset dual-binary et retourne X, Y pour le split demandé.

    Args:
        path: Chemin vers le fichier .npz
        split: 'train', 'val', ou 'test'

    Returns:
        X: Features (n, seq_len, n_features)
        Y: Labels (n, 2) pour dual-binary [Direction, Force]
        metadata: Dictionnaire de métadonnées
    """
    data = np.load(path, allow_pickle=True)

    X = data[f'X_{split}']
    Y = data[f'Y_{split}']

    # Charger métadonnées
    metadata = None
    if 'metadata' in data:
        try:
            metadata = json.loads(str(data['metadata']))
        except:
            pass

    return X, Y, metadata


def check_index_alignment(X1: np.ndarray, X2: np.ndarray) -> Dict:
    """
    Vérifie que les deux datasets ont des index alignés.

    Pour dual-binary, les features (X) devraient être identiques car elles
    ne dépendent pas du filtre (Octave ou Kalman).
    """
    # Vérifier les shapes
    if X1.shape != X2.shape:
        return {
            'aligned': False,
            'error': f'Shapes différentes: {X1.shape} vs {X2.shape}'
        }

    # Comparer les features (devraient être identiques)
    diff = np.abs(X1 - X2)
    is_identical = np.allclose(X1, X2, rtol=1e-10, atol=1e-10)

    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    return {
        'aligned': is_identical,
        'shape': X1.shape,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'n_different': int(np.sum(diff > 1e-10)),
        'pct_different': float(np.sum(diff > 1e-10) / diff.size * 100),
    }


def compute_concordance(y1: np.ndarray, y2: np.ndarray, lag: int = 0) -> float:
    """
    Calcule la concordance entre deux séries de labels avec un lag donné.

    Args:
        y1: Première série (n,)
        y2: Deuxième série (n,)
        lag: Décalage temporel (négatif = y1 en avance sur y2)

    Returns:
        Concordance (% de labels identiques)
    """
    # Appliquer le lag
    if lag < 0:
        # y1 en avance: comparer y1[:-lag] avec y2[lag:]
        y1_shifted = y1[:-lag]
        y2_shifted = y2[-lag:]
    elif lag > 0:
        # y1 en retard: comparer y1[lag:] avec y2[:-lag]
        y1_shifted = y1[lag:]
        y2_shifted = y2[:-lag]
    else:
        # Pas de lag
        y1_shifted = y1
        y2_shifted = y2

    # Calculer concordance
    n_same = np.sum(y1_shifted == y2_shifted)
    concordance = float(n_same / len(y1_shifted) * 100)

    return concordance


def find_optimal_lag(y1: np.ndarray, y2: np.ndarray, lag_range: Tuple[int, int] = (-5, 6)) -> Dict:
    """
    Trouve le lag optimal entre deux séries en maximisant la concordance.

    Args:
        y1: Première série (Octave)
        y2: Deuxième série (Kalman)
        lag_range: Plage de lag à tester (min, max)

    Returns:
        Dictionnaire avec optimal_lag, max_concordance, concordances par lag
    """
    min_lag, max_lag = lag_range
    lags = range(min_lag, max_lag)
    concordances = []

    for lag in lags:
        conc = compute_concordance(y1, y2, lag)
        concordances.append(conc)

    # Trouver le maximum
    best_idx = np.argmax(concordances)
    optimal_lag = lags[best_idx]
    max_concordance = concordances[best_idx]

    # Concordance à lag 0 (référence)
    concordance_lag0 = concordances[abs(min_lag)]  # lag=0 est à l'index abs(min_lag)

    return {
        'optimal_lag': optimal_lag,
        'max_concordance': max_concordance,
        'concordance_lag0': concordance_lag0,
        'all_lags': list(lags),
        'all_concordances': concordances,
    }


def compare_labels_dual_binary(
    Y1: np.ndarray,
    Y2: np.ndarray,
    sample_size: Optional[int] = None
) -> Dict:
    """
    Compare les labels dual-binary entre deux filtres.

    Args:
        Y1: Labels Octave (n, 2) - [Direction, Force]
        Y2: Labels Kalman (n, 2) - [Direction, Force]
        sample_size: Nombre de samples à analyser (None = tous)

    Returns:
        Dictionnaire de statistiques
    """
    if sample_size and sample_size < len(Y1):
        Y1 = Y1[:sample_size]
        Y2 = Y2[:sample_size]

    # Vérifier shapes
    if Y1.shape != Y2.shape:
        return {
            'error': f'Shapes différentes: {Y1.shape} vs {Y2.shape}',
            'valid': False
        }

    if Y1.shape[1] != 2:
        return {
            'error': f'Pas dual-binary: Y.shape[1] = {Y1.shape[1]} (attendu: 2)',
            'valid': False
        }

    # Séparer Direction et Force
    dir1 = Y1[:, 0]  # Direction Octave
    force1 = Y1[:, 1]  # Force Octave
    dir2 = Y2[:, 0]  # Direction Kalman
    force2 = Y2[:, 1]  # Force Kalman

    # Trouver lag optimal pour Direction
    lag_results_dir = find_optimal_lag(dir1, dir2, lag_range=(-5, 6))

    # Trouver lag optimal pour Force
    lag_results_force = find_optimal_lag(force1, force2, lag_range=(-5, 6))

    # Statistiques Direction
    dir_same_lag0 = np.sum(dir1 == dir2)
    dir_concordance_lag0 = float(dir_same_lag0 / len(dir1) * 100)

    # Statistiques Force
    force_same_lag0 = np.sum(force1 == force2)
    force_concordance_lag0 = float(force_same_lag0 / len(force1) * 100)

    # Balance des labels
    dir1_up_pct = float(np.mean(dir1) * 100)
    dir2_up_pct = float(np.mean(dir2) * 100)
    force1_strong_pct = float(np.mean(force1) * 100)
    force2_strong_pct = float(np.mean(force2) * 100)

    return {
        'valid': True,
        'n_samples': len(Y1),
        'direction': {
            'concordance_lag0': dir_concordance_lag0,
            'optimal_lag': lag_results_dir['optimal_lag'],
            'max_concordance': lag_results_dir['max_concordance'],
            'octave_up_pct': dir1_up_pct,
            'kalman_up_pct': dir2_up_pct,
            'all_lags': lag_results_dir['all_lags'],
            'all_concordances': lag_results_dir['all_concordances'],
        },
        'force': {
            'concordance_lag0': force_concordance_lag0,
            'optimal_lag': lag_results_force['optimal_lag'],
            'max_concordance': lag_results_force['max_concordance'],
            'octave_strong_pct': force1_strong_pct,
            'kalman_strong_pct': force2_strong_pct,
            'all_lags': lag_results_force['all_lags'],
            'all_concordances': lag_results_force['all_concordances'],
        }
    }


def analyze_disagreement_patterns(
    Y1: np.ndarray,
    Y2: np.ndarray,
    output_idx: int = 0
) -> Dict:
    """
    Analyse les patterns de désaccord pour un output spécifique.

    Args:
        Y1: Labels Octave (n, 2)
        Y2: Labels Kalman (n, 2)
        output_idx: 0=Direction, 1=Force

    Returns:
        Statistiques sur les blocs de désaccord
    """
    y1 = Y1[:, output_idx]
    y2 = Y2[:, output_idx]

    # Trouver les désaccords
    disagreement = (y1 != y2).astype(int)

    # Compter les blocs de désaccord consécutifs
    blocks = []
    current_block = 0

    for i in range(len(disagreement)):
        if disagreement[i] == 1:
            current_block += 1
        else:
            if current_block > 0:
                blocks.append(current_block)
                current_block = 0

    if current_block > 0:
        blocks.append(current_block)

    if len(blocks) == 0:
        return {
            'n_blocks': 0,
            'avg_block_size': 0,
            'max_block_size': 0,
            'isolated_disagreements': 0,
            'pct_isolated': 0,
        }

    blocks = np.array(blocks)

    return {
        'n_blocks': len(blocks),
        'avg_block_size': float(np.mean(blocks)),
        'max_block_size': int(np.max(blocks)),
        'min_block_size': int(np.min(blocks)),
        'isolated_disagreements': int(np.sum(blocks == 1)),
        'pct_isolated': float(np.sum(blocks == 1) / len(blocks) * 100),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare les filtres Octave vs Kalman (Dual-Binary)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Comparer RSI
  python src/compare_filters.py \\
      --indicator rsi \\
      --file-octave data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_dual_binary_octave20.npz \\
      --file-kalman data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_dual_binary_kalman.npz \\
      --split test

  # Comparer MACD avec échantillon
  python src/compare_filters.py \\
      --indicator macd \\
      --file-octave data/prepared/dataset_..._macd_dual_binary_octave20.npz \\
      --file-kalman data/prepared/dataset_..._macd_dual_binary_kalman.npz \\
      --split test \\
      --sample 20000
        """
    )

    parser.add_argument('--indicator', '-i', type=str, required=True,
                        choices=['rsi', 'cci', 'macd'],
                        help='Indicateur à comparer')
    parser.add_argument('--file-octave', '-fo', type=str, required=True,
                        help='Fichier .npz Octave')
    parser.add_argument('--file-kalman', '-fk', type=str, required=True,
                        help='Fichier .npz Kalman')
    parser.add_argument('--split', '-s', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Split à comparer (défaut: test)')
    parser.add_argument('--sample', '-n', type=int, default=None,
                        help='Nombre de samples à analyser (défaut: tous)')

    args = parser.parse_args()

    # Vérifier les fichiers
    if not Path(args.file_octave).exists():
        print(f"❌ Fichier Octave non trouvé: {args.file_octave}")
        return
    if not Path(args.file_kalman).exists():
        print(f"❌ Fichier Kalman non trouvé: {args.file_kalman}")
        return

    print("=" * 80)
    print(f"COMPARAISON FILTRES: Octave vs Kalman - {args.indicator.upper()}")
    print("=" * 80)

    # Charger les datasets
    print(f"\n📂 Chargement...")
    print(f"   Octave: {args.file_octave}")
    print(f"   Kalman: {args.file_kalman}")
    print(f"   Split: {args.split}")

    X_octave, Y_octave, meta_octave = load_dataset(args.file_octave, args.split)
    X_kalman, Y_kalman, meta_kalman = load_dataset(args.file_kalman, args.split)

    sample_size = args.sample
    if sample_size:
        print(f"   Sample: {sample_size} (sur {len(X_octave)} disponibles)")
        X_octave = X_octave[:sample_size]
        X_kalman = X_kalman[:sample_size]
        Y_octave = Y_octave[:sample_size]
        Y_kalman = Y_kalman[:sample_size]
    else:
        sample_size = len(X_octave)
        print(f"   Sample: {sample_size} (tous)")

    # Afficher métadonnées
    if meta_octave and meta_kalman:
        print(f"\n📋 Métadonnées:")
        print(f"   Octave filter: {meta_octave.get('filter_type', 'N/A')}")
        print(f"   Kalman filter: {meta_kalman.get('filter_type', 'N/A')}")
        print(f"   Indicateur: {meta_octave.get('label_names', ['N/A'])[0]}")

    # 1. Vérifier l'alignement des index
    print(f"\n" + "=" * 80)
    print("1. VÉRIFICATION ALIGNEMENT INDEX")
    print("=" * 80)

    alignment = check_index_alignment(X_octave, X_kalman)

    if alignment.get('error'):
        print(f"   ❌ Erreur: {alignment['error']}")
        return

    print(f"\n   Shape: {alignment['shape']}")
    print(f"   Features identiques: {'✅ OUI' if alignment['aligned'] else '❌ NON'}")
    print(f"   Max diff: {alignment['max_diff']:.2e}")
    print(f"   Mean diff: {alignment['mean_diff']:.2e}")

    if not alignment['aligned']:
        print(f"\n   ⚠️  ATTENTION: Les features ne sont pas identiques!")
        print(f"   Les datasets ne sont peut-être pas synchronisés.")
        print(f"   Différences: {alignment['n_different']} ({alignment['pct_different']:.4f}%)")
        return

    # 2. Comparer les labels dual-binary
    print(f"\n" + "=" * 80)
    print("2. COMPARAISON LABELS DUAL-BINARY")
    print("=" * 80)

    label_results = compare_labels_dual_binary(Y_octave, Y_kalman, sample_size)

    if not label_results.get('valid'):
        print(f"   ❌ Erreur: {label_results.get('error')}")
        return

    # Résultats Direction
    print(f"\n   📊 DIRECTION:")
    dir_stats = label_results['direction']
    print(f"      Concordance (lag=0): {dir_stats['concordance_lag0']:.1f}%")
    print(f"      Lag optimal: {dir_stats['optimal_lag']}")
    print(f"      Concordance max: {dir_stats['max_concordance']:.1f}% (à lag {dir_stats['optimal_lag']})")
    print(f"      Balance UP:")
    print(f"        Octave: {dir_stats['octave_up_pct']:.1f}%")
    print(f"        Kalman: {dir_stats['kalman_up_pct']:.1f}%")

    # Résultats Force
    print(f"\n   📊 FORCE:")
    force_stats = label_results['force']
    print(f"      Concordance (lag=0): {force_stats['concordance_lag0']:.1f}%")
    print(f"      Lag optimal: {force_stats['optimal_lag']}")
    print(f"      Concordance max: {force_stats['max_concordance']:.1f}% (à lag {force_stats['optimal_lag']})")
    print(f"      Balance STRONG:")
    print(f"        Octave: {force_stats['octave_strong_pct']:.1f}%")
    print(f"        Kalman: {force_stats['kalman_strong_pct']:.1f}%")

    # 3. Analyser les patterns de désaccord
    print(f"\n" + "=" * 80)
    print("3. PATTERNS DE DÉSACCORD")
    print("=" * 80)

    dir_patterns = analyze_disagreement_patterns(Y_octave, Y_kalman, output_idx=0)
    force_patterns = analyze_disagreement_patterns(Y_octave, Y_kalman, output_idx=1)

    print(f"\n   Direction:")
    print(f"      Blocs de désaccord: {dir_patterns['n_blocks']}")
    print(f"      Taille moyenne: {dir_patterns['avg_block_size']:.1f} samples")
    print(f"      Taille max: {dir_patterns['max_block_size']}")
    print(f"      Désaccords isolés (1 sample): {dir_patterns['isolated_disagreements']} ({dir_patterns['pct_isolated']:.1f}%)")

    print(f"\n   Force:")
    print(f"      Blocs de désaccord: {force_patterns['n_blocks']}")
    print(f"      Taille moyenne: {force_patterns['avg_block_size']:.1f} samples")
    print(f"      Taille max: {force_patterns['max_block_size']}")
    print(f"      Désaccords isolés (1 sample): {force_patterns['isolated_disagreements']} ({force_patterns['pct_isolated']:.1f}%)")

    # 4. Afficher toutes les concordances par lag
    print(f"\n" + "=" * 80)
    print("4. CONCORDANCES PAR LAG")
    print("=" * 80)

    print(f"\n   Direction:")
    for lag, conc in zip(dir_stats['all_lags'], dir_stats['all_concordances']):
        marker = '🎯' if lag == dir_stats['optimal_lag'] else '  '
        print(f"      {marker} Lag {lag:+2d}: {conc:.2f}%")

    print(f"\n   Force:")
    for lag, conc in zip(force_stats['all_lags'], force_stats['all_concordances']):
        marker = '🎯' if lag == force_stats['optimal_lag'] else '  '
        print(f"      {marker} Lag {lag:+2d}: {conc:.2f}%")

    # Résumé
    print(f"\n" + "=" * 80)
    print("RÉSUMÉ")
    print("=" * 80)

    print(f"\n✅ Index alignés: Les features sont identiques (OK)")

    print(f"\n📊 Concordance Direction:")
    dir_conc = dir_stats['concordance_lag0']
    if dir_conc > 90:
        print(f"   {dir_conc:.1f}% → 🟢 Très similaires (lag=0)")
    elif dir_conc > 70:
        print(f"   {dir_conc:.1f}% → 🟡 Modérément similaires (lag=0)")
    else:
        print(f"   {dir_conc:.1f}% → 🔴 Divergent (lag=0)")

    if dir_stats['optimal_lag'] != 0:
        print(f"   ⚠️  Lag optimal: {dir_stats['optimal_lag']} → Déphasage temporel détecté!")
        print(f"   Concordance max: {dir_stats['max_concordance']:.1f}% (à lag {dir_stats['optimal_lag']})")
    else:
        print(f"   ✅ Lag optimal: 0 → Synchronisés")

    print(f"\n📊 Concordance Force:")
    force_conc = force_stats['concordance_lag0']
    if force_conc > 90:
        print(f"   {force_conc:.1f}% → 🟢 Très similaires (lag=0)")
    elif force_conc > 70:
        print(f"   {force_conc:.1f}% → 🟡 Modérément similaires (lag=0)")
    else:
        print(f"   {force_conc:.1f}% → 🔴 Divergent (lag=0)")

    if force_stats['optimal_lag'] != 0:
        print(f"   ⚠️  Lag optimal: {force_stats['optimal_lag']} → Déphasage temporel détecté!")
        print(f"   Concordance max: {force_stats['max_concordance']:.1f}% (à lag {force_stats['optimal_lag']})")
    else:
        print(f"   ✅ Lag optimal: 0 → Synchronisés")

    print(f"\n💡 Interprétation:")
    print(f"   - Lag 0 = Pas de déphasage temporel")
    print(f"   - Lag négatif = Octave en avance sur Kalman")
    print(f"   - Lag positif = Octave en retard sur Kalman")
    print(f"   - Désaccords isolés ({dir_patterns['pct_isolated']:.1f}% Dir, {force_patterns['pct_isolated']:.1f}% Force) = Bruit transitoire")
    print(f"   - Blocs de désaccord = Zones d'incertitude structurelle")


if __name__ == '__main__':
    main()
