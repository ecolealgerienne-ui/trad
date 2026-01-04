"""
Script de comparaison des datasets Octave20 vs Kalman.

Objectifs:
1. Vérifier que les indices/features sont synchronisés entre les deux filtres
2. Mesurer la différence de labels entre Octave20 et Kalman
3. Analyser sur un échantillon (ex: 20000 samples BTC)

Usage:
    python src/compare_datasets.py --file1 data/prepared/dataset_btc_eth_bnb_ada_ltc_ohlcv2_rsi_octave20.npz \
                                   --file2 data/prepared/dataset_btc_eth_bnb_ada_ltc_ohlcv2_rsi_kalman.npz \
                                   --split train --sample 20000
"""

import numpy as np
import argparse
import json
from pathlib import Path


def load_dataset(path: str, split: str = 'train') -> tuple:
    """Charge un dataset et retourne X, Y pour le split demandé."""
    data = np.load(path, allow_pickle=True)

    X = data[f'X_{split}']
    Y = data[f'Y_{split}']

    # Charger métadonnées si disponibles
    metadata = None
    if 'metadata' in data:
        try:
            metadata = json.loads(str(data['metadata']))
        except:
            pass

    return X, Y, metadata


def compare_features(X1: np.ndarray, X2: np.ndarray, sample_size: int = None) -> dict:
    """
    Compare les features entre deux datasets.

    Les features OHLC (o_ret, h_ret, l_ret, c_ret, range_ret) devraient être
    IDENTIQUES entre Octave et Kalman car elles ne dépendent pas du filtre.
    """
    if sample_size and sample_size < len(X1):
        X1 = X1[:sample_size]
        X2 = X2[:sample_size]

    # Vérifier les shapes
    if X1.shape != X2.shape:
        return {
            'error': f'Shapes différentes: {X1.shape} vs {X2.shape}',
            'identical': False
        }

    # Comparer
    diff = np.abs(X1 - X2)
    is_identical = np.allclose(X1, X2, rtol=1e-10, atol=1e-10)

    results = {
        'shape': X1.shape,
        'identical': is_identical,
        'max_diff': float(np.max(diff)),
        'mean_diff': float(np.mean(diff)),
        'n_different': int(np.sum(diff > 1e-10)),
        'pct_different': float(np.sum(diff > 1e-10) / diff.size * 100),
    }

    # Analyse par feature (5 canaux OHLC)
    feature_names = ['o_ret', 'h_ret', 'l_ret', 'c_ret', 'range_ret']
    results['per_feature'] = {}

    for i, name in enumerate(feature_names):
        feat_diff = np.abs(X1[:, :, i] - X2[:, :, i])
        results['per_feature'][name] = {
            'max_diff': float(np.max(feat_diff)),
            'mean_diff': float(np.mean(feat_diff)),
            'identical': bool(np.allclose(X1[:, :, i], X2[:, :, i], rtol=1e-10, atol=1e-10))
        }

    return results


def compare_labels(Y1: np.ndarray, Y2: np.ndarray, sample_size: int = None) -> dict:
    """
    Compare les labels entre deux datasets.

    Les labels DEVRAIENT être différents car ils dépendent du filtre utilisé.
    """
    if sample_size and sample_size < len(Y1):
        Y1 = Y1[:sample_size]
        Y2 = Y2[:sample_size]

    # Vérifier les shapes
    if Y1.shape != Y2.shape:
        return {
            'error': f'Shapes différentes: {Y1.shape} vs {Y2.shape}',
            'identical': False
        }

    # Flatten pour comparaison
    y1_flat = Y1.flatten()
    y2_flat = Y2.flatten()

    # Statistiques
    n_total = len(y1_flat)
    n_same = int(np.sum(y1_flat == y2_flat))
    n_diff = n_total - n_same

    # Transitions (changements de direction)
    # 0->1 ou 1->0 entre les deux filtres
    transitions_01 = int(np.sum((y1_flat == 0) & (y2_flat == 1)))  # Octave=DOWN, Kalman=UP
    transitions_10 = int(np.sum((y1_flat == 1) & (y2_flat == 0)))  # Octave=UP, Kalman=DOWN

    # Balance des labels
    y1_up_pct = float(np.mean(y1_flat) * 100)
    y2_up_pct = float(np.mean(y2_flat) * 100)

    results = {
        'n_total': n_total,
        'n_same': n_same,
        'n_different': n_diff,
        'pct_same': float(n_same / n_total * 100),
        'pct_different': float(n_diff / n_total * 100),
        'file1_up_pct': y1_up_pct,
        'file2_up_pct': y2_up_pct,
        'transitions': {
            'file1_down_file2_up': transitions_01,
            'file1_up_file2_down': transitions_10,
        }
    }

    return results


def analyze_disagreement_patterns(Y1: np.ndarray, Y2: np.ndarray, sample_size: int = None) -> dict:
    """
    Analyse les patterns de désaccord entre les deux filtres.

    Quand ils sont en désaccord, est-ce que c'est:
    - Isolé (1 sample) ou en bloc (plusieurs consécutifs)?
    - Aléatoire ou structurel?
    """
    if sample_size and sample_size < len(Y1):
        Y1 = Y1[:sample_size]
        Y2 = Y2[:sample_size]

    y1_flat = Y1.flatten()
    y2_flat = Y2.flatten()

    # Trouver les désaccords
    disagreement = (y1_flat != y2_flat).astype(int)

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
        }

    blocks = np.array(blocks)

    return {
        'n_blocks': len(blocks),
        'avg_block_size': float(np.mean(blocks)),
        'max_block_size': int(np.max(blocks)),
        'min_block_size': int(np.min(blocks)),
        'isolated_disagreements': int(np.sum(blocks == 1)),
        'pct_isolated': float(np.sum(blocks == 1) / len(blocks) * 100) if len(blocks) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare deux datasets (Octave20 vs Kalman)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python src/compare_datasets.py --file1 data/prepared/dataset_btc_eth_bnb_ada_ltc_ohlcv2_rsi_octave20.npz \\
                                 --file2 data/prepared/dataset_btc_eth_bnb_ada_ltc_ohlcv2_rsi_kalman.npz \\
                                 --split train --sample 20000

  python src/compare_datasets.py --file1 ...octave20.npz --file2 ...kalman.npz --split val
        """
    )

    parser.add_argument('--file1', '-f1', type=str, required=True,
                        help='Premier fichier .npz (ex: octave20)')
    parser.add_argument('--file2', '-f2', type=str, required=True,
                        help='Deuxième fichier .npz (ex: kalman)')
    parser.add_argument('--split', '-s', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Split à comparer (défaut: train)')
    parser.add_argument('--sample', '-n', type=int, default=None,
                        help='Nombre de samples à analyser (défaut: tous)')

    args = parser.parse_args()

    # Vérifier les fichiers
    if not Path(args.file1).exists():
        print(f"❌ Fichier non trouvé: {args.file1}")
        return
    if not Path(args.file2).exists():
        print(f"❌ Fichier non trouvé: {args.file2}")
        return

    print("="*80)
    print("COMPARAISON DATASETS: Octave20 vs Kalman")
    print("="*80)

    # Charger les datasets
    print(f"\n📂 Chargement...")
    print(f"   File1: {args.file1}")
    print(f"   File2: {args.file2}")
    print(f"   Split: {args.split}")

    X1, Y1, meta1 = load_dataset(args.file1, args.split)
    X2, Y2, meta2 = load_dataset(args.file2, args.split)

    sample_size = args.sample
    if sample_size:
        print(f"   Sample: {sample_size} (sur {len(X1)} disponibles)")
    else:
        sample_size = len(X1)
        print(f"   Sample: {sample_size} (tous)")

    # Afficher métadonnées
    if meta1 and meta2:
        print(f"\n📋 Métadonnées:")
        print(f"   File1 filter: {meta1.get('filter_type', 'N/A')}")
        print(f"   File2 filter: {meta2.get('filter_type', 'N/A')}")
        print(f"   Target: {meta1.get('target', 'N/A')}")

    # Comparer les features
    print(f"\n" + "="*80)
    print("1. COMPARAISON DES FEATURES (devraient être identiques)")
    print("="*80)

    feat_results = compare_features(X1, X2, sample_size)

    if feat_results.get('error'):
        print(f"   ❌ Erreur: {feat_results['error']}")
    else:
        print(f"\n   Shape: {feat_results['shape']}")
        print(f"   Identiques: {'✅ OUI' if feat_results['identical'] else '❌ NON'}")
        print(f"   Max diff: {feat_results['max_diff']:.2e}")
        print(f"   Mean diff: {feat_results['mean_diff']:.2e}")
        print(f"   N différents: {feat_results['n_different']} ({feat_results['pct_different']:.4f}%)")

        print(f"\n   Par feature:")
        for name, stats in feat_results['per_feature'].items():
            status = '✅' if stats['identical'] else '❌'
            print(f"     {name}: {status} (max_diff={stats['max_diff']:.2e})")

    # Comparer les labels
    print(f"\n" + "="*80)
    print("2. COMPARAISON DES LABELS (devraient différer)")
    print("="*80)

    label_results = compare_labels(Y1, Y2, sample_size)

    if label_results.get('error'):
        print(f"   ❌ Erreur: {label_results['error']}")
    else:
        print(f"\n   Total samples: {label_results['n_total']}")
        print(f"   Identiques: {label_results['n_same']} ({label_results['pct_same']:.2f}%)")
        print(f"   Différents: {label_results['n_different']} ({label_results['pct_different']:.2f}%)")
        print(f"\n   Balance labels:")
        print(f"     File1 (Octave20?): {label_results['file1_up_pct']:.1f}% UP")
        print(f"     File2 (Kalman?):   {label_results['file2_up_pct']:.1f}% UP")
        print(f"\n   Transitions (désaccords):")
        print(f"     File1=DOWN, File2=UP: {label_results['transitions']['file1_down_file2_up']}")
        print(f"     File1=UP, File2=DOWN: {label_results['transitions']['file1_up_file2_down']}")

    # Analyser les patterns de désaccord
    print(f"\n" + "="*80)
    print("3. ANALYSE DES PATTERNS DE DÉSACCORD")
    print("="*80)

    pattern_results = analyze_disagreement_patterns(Y1, Y2, sample_size)

    print(f"\n   Nombre de blocs de désaccord: {pattern_results['n_blocks']}")
    print(f"   Taille moyenne des blocs: {pattern_results['avg_block_size']:.1f} samples")
    print(f"   Taille max bloc: {pattern_results['max_block_size']}")
    print(f"   Désaccords isolés (1 sample): {pattern_results['isolated_disagreements']} ({pattern_results['pct_isolated']:.1f}%)")

    # Résumé
    print(f"\n" + "="*80)
    print("RÉSUMÉ")
    print("="*80)

    if feat_results.get('identical'):
        print("✅ Features IDENTIQUES entre les deux filtres (attendu)")
    else:
        print("⚠️  Features DIFFÉRENTES entre les deux filtres (inattendu!)")

    concordance = label_results.get('pct_same', 0)
    print(f"\n📊 Concordance labels: {concordance:.1f}%")

    if concordance > 90:
        print("   → Les deux filtres sont très similaires")
    elif concordance > 70:
        print("   → Les deux filtres ont une concordance modérée")
    else:
        print("   → Les deux filtres divergent significativement")

    print(f"\n💡 Interprétation:")
    print(f"   - Concordance élevée = les filtres voient la même tendance")
    print(f"   - Les {label_results.get('pct_different', 0):.1f}% de désaccord = zones de transition/incertitude")
    print(f"   - Ces zones peuvent être utilisées dans la state machine pour modérer les décisions")


if __name__ == '__main__':
    main()
