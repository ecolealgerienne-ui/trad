"""
Test de stabilité des filtres sur données du .npz (BTC uniquement).

Inspiré de src/state_machine.py pour le chargement des données.

Usage:
    python src/test_filter_stability_from_npz.py \
        --npz-file data/prepared/dataset_btc_eth_bnb_ada_ltc_ohlcv2_macd_kalman.npz \
        --filter kalman \
        --asset BTC \
        --window-size 100 \
        --n-samples 200 \
        --split test
"""

import numpy as np
import argparse
import json
import sys
from pathlib import Path
from typing import Dict

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent))

from filters import signal_filtfilt
from pykalman import KalmanFilter


def load_dataset(path: str, split: str = 'test') -> dict:
    """Charge un dataset .npz."""
    data = np.load(path, allow_pickle=True)

    result = {
        'X': data[f'X_{split}'],
        'Y': data[f'Y_{split}'],
        'metadata': None,
        'assets': None,
        'samples_per_asset': None
    }

    # Charger métadonnées
    if 'metadata' in data:
        try:
            meta = json.loads(str(data['metadata']))
            result['metadata'] = meta
            if 'assets' in meta:
                result['assets'] = meta['assets']
            if f'samples_per_asset_{split}' in meta:
                result['samples_per_asset'] = meta[f'samples_per_asset_{split}']
        except:
            pass

    return result


def extract_asset_data(dataset: dict, asset_name: str) -> Dict:
    """
    Extrait les données d'un seul asset.

    Returns:
        Dict avec X, Y pour l'asset demandé
    """
    assets = dataset['assets']
    samples_per_asset = dataset['samples_per_asset']

    if not assets or not samples_per_asset:
        raise ValueError("Métadonnées assets/samples_per_asset manquantes")

    if asset_name not in assets:
        raise ValueError(f"Asset '{asset_name}' non trouvé. Assets disponibles: {assets}")

    asset_idx = assets.index(asset_name)

    # Calculer offset
    offset = sum(samples_per_asset[:asset_idx])
    count = samples_per_asset[asset_idx]

    print(f"\n📊 Extraction asset '{asset_name}':")
    print(f"   Position: {asset_idx + 1}/{len(assets)}")
    print(f"   Offset: {offset:,}")
    print(f"   Samples: {count:,}")

    # Extraire
    X_asset = dataset['X'][offset:offset + count]
    Y_asset = dataset['Y'][offset:offset + count]

    return {
        'X': X_asset,
        'Y': Y_asset,
        'asset': asset_name,
        'n_samples': count
    }


def apply_kalman_filter(data: np.ndarray) -> np.ndarray:
    """Applique filtre de Kalman."""
    mask = ~np.isnan(data)
    valid_data = data[mask]

    if len(valid_data) == 0:
        return data

    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=valid_data[0],
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.01
    )

    state_means, _ = kf.filter(valid_data)
    filtered_valid = state_means.flatten()

    filtered = np.full_like(data, np.nan, dtype=float)
    filtered[mask] = filtered_valid

    return filtered


def apply_filter(data: np.ndarray, filter_name: str, **kwargs) -> np.ndarray:
    """Applique un filtre."""
    if filter_name == 'kalman':
        return apply_kalman_filter(data)
    elif filter_name == 'octave':
        step = kwargs.get('step', 0.20)
        order = kwargs.get('order', 3)
        return signal_filtfilt(data, step=step, order=order)
    else:
        raise ValueError(f"Filtre inconnu: {filter_name}")


def compute_slope_label(filtered: np.ndarray, index: int) -> int:
    """
    Calcule le label de pente.
    Label = 1 si filtered[index-2] > filtered[index-3], sinon 0
    """
    if index < 3:
        return -1
    return int(filtered[index - 2] > filtered[index - 3])


def reconstruct_signal_from_features(X: np.ndarray, feature_idx: int = 3) -> np.ndarray:
    """
    Reconstruit un signal approximatif depuis les features OHLC.

    X shape: (n_samples, seq_length, n_features)
    feature_idx: 3 = c_ret (rendements close-to-close)

    NOTE: Reconstruction approximative! On perd l'échelle absolue.
    """
    n_samples = len(X)

    # Extraire c_ret (dernière feature de chaque séquence)
    c_ret = X[:, -1, feature_idx]

    # Reconstruire prix cumulatif (commence à 100)
    cumulative = np.zeros(n_samples)
    cumulative[0] = 100.0

    for i in range(1, n_samples):
        cumulative[i] = cumulative[i-1] * (1 + c_ret[i])

    return cumulative


def test_filter_stability(
    signal: np.ndarray,
    Y_global: np.ndarray,
    filter_name: str,
    window_size: int = 100,
    n_samples: int = 200,
    **filter_kwargs
) -> Dict:
    """
    Teste la stabilité du filtre.

    Compare:
    - Y_global: labels du .npz (calculés sur tout le dataset)
    - Y_local: labels recalculés sur fenêtre glissante
    """
    n = len(signal)

    print(f"\n{'='*80}")
    print(f"TEST DE STABILITÉ - {filter_name.upper()}")
    print(f"{'='*80}")

    print(f"\n⚙️ Configuration:")
    print(f"   Filtre: {filter_name}")
    print(f"   Fenêtre: {window_size} samples")
    print(f"   Tests: {n_samples} positions")
    print(f"   Paramètres: {filter_kwargs}")

    # ÉTAPE 1: Appliquer filtre sur TOUT le signal
    print(f"\n🔧 Étape 1: Application sur signal complet...")
    filtered_global = apply_filter(signal, filter_name, **filter_kwargs)

    # ÉTAPE 2: Échantillonner positions
    min_idx = window_size + 3
    max_idx = n - 3

    if max_idx <= min_idx:
        raise ValueError(f"Dataset trop petit: {n}, besoin >= {window_size + 6}")

    sample_indices = np.linspace(min_idx, max_idx, n_samples, dtype=int)

    print(f"\n🔧 Étape 2: Test sur fenêtre glissante...")
    print(f"   Indices: [{sample_indices[0]}, ..., {sample_indices[-1]}]")

    # ÉTAPE 3: Tester chaque position
    concordance = []
    labels_local = []
    labels_global_sampled = []

    for i, t in enumerate(sample_indices):
        # Fenêtre locale
        window_data = signal[t - window_size : t + 1]

        # Appliquer filtre
        filtered_local = apply_filter(window_data, filter_name, **filter_kwargs)

        # Label local
        label_local = int(filtered_local[-2] > filtered_local[-3])

        # Label global à cette position (depuis Y du .npz)
        label_global = int(Y_global[t])

        # Concordance
        agree = (label_local == label_global)
        concordance.append(agree)
        labels_local.append(label_local)
        labels_global_sampled.append(label_global)

        if (i + 1) % 50 == 0:
            curr = np.mean(concordance) * 100
            print(f"   Progression: {i+1}/{n_samples} - Concordance: {curr:.1f}%")

    # ÉTAPE 4: Statistiques
    concordance = np.array(concordance)
    labels_local = np.array(labels_local)
    labels_global_sampled = np.array(labels_global_sampled)

    concordance_pct = concordance.mean() * 100

    # Par classe
    mask_up = (labels_global_sampled == 1)
    mask_down = (labels_global_sampled == 0)

    conc_up = concordance[mask_up].mean() * 100 if mask_up.any() else 0
    conc_down = concordance[mask_down].mean() * 100 if mask_down.any() else 0

    print(f"\n{'='*80}")
    print("RÉSULTATS")
    print(f"{'='*80}")

    print(f"\n📊 Labels (global .npz vs local fenêtre):")
    print(f"   Global: {mask_up.sum()} UP, {mask_down.sum()} DOWN")
    print(f"   Local:  {(labels_local == 1).sum()} UP, {(labels_local == 0).sum()} DOWN")

    print(f"\n✅ Concordance: {concordance_pct:.2f}%")
    print(f"   - Sur UP:   {conc_up:.2f}%")
    print(f"   - Sur DOWN: {conc_down:.2f}%")

    n_disagree = (~concordance).sum()
    print(f"\n❌ Désaccords: {n_disagree}/{n_samples} ({n_disagree/n_samples*100:.1f}%)")

    return {
        'concordance': concordance_pct,
        'concordance_up': conc_up,
        'concordance_down': conc_down,
        'n_samples': n_samples,
        'n_disagree': n_disagree
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test stabilité filtres depuis .npz (single asset)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--npz-file', type=str, required=True,
                        help='Fichier .npz')
    parser.add_argument('--filter', type=str, default='kalman',
                        choices=['kalman', 'octave'],
                        help='Filtre à tester')
    parser.add_argument('--asset', type=str, default='BTC',
                        help='Asset à extraire (BTC, ETH, etc.)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--window-size', type=int, default=100)
    parser.add_argument('--n-samples', type=int, default=200)

    # Paramètres filtre
    parser.add_argument('--step', type=float, default=0.20,
                        help='Step pour Octave')
    parser.add_argument('--order', type=int, default=3,
                        help='Ordre Butterworth')

    args = parser.parse_args()

    print("="*80)
    print("TEST DE STABILITÉ - DEPUIS .NPZ (SINGLE ASSET)")
    print("="*80)

    # Charger dataset complet
    print(f"\n📂 Chargement {args.npz_file} ({args.split})...")
    dataset = load_dataset(args.npz_file, args.split)

    print(f"   Total samples: {len(dataset['X']):,}")
    print(f"   Assets: {dataset['assets']}")
    print(f"   Samples par asset: {dataset['samples_per_asset']}")

    # Extraire asset
    asset_data = extract_asset_data(dataset, args.asset)

    X = asset_data['X']
    Y = asset_data['Y']
    n = asset_data['n_samples']

    print(f"\n📊 Données {args.asset}:")
    print(f"   X shape: {X.shape}")
    print(f"   Y shape: {Y.shape}")
    print(f"   Y mean: {Y.mean():.3f} ({(Y==1).sum()} UP, {(Y==0).sum()} DOWN)")

    # Reconstruire signal approximatif depuis c_ret
    print(f"\n🔧 Reconstruction signal depuis c_ret...")
    signal = reconstruct_signal_from_features(X, feature_idx=3)
    print(f"   Signal min: {signal.min():.2f}, max: {signal.max():.2f}")

    # Paramètres filtre
    filter_kwargs = {}
    if args.filter == 'octave':
        filter_kwargs = {'step': args.step, 'order': args.order}

    # Test de stabilité
    results = test_filter_stability(
        signal=signal,
        Y_global=Y,
        filter_name=args.filter,
        window_size=args.window_size,
        n_samples=args.n_samples,
        **filter_kwargs
    )

    # Résumé
    print(f"\n{'='*80}")
    print("RÉSUMÉ")
    print(f"{'='*80}")

    print(f"\n🎯 Asset: {args.asset}")
    print(f"   Filtre: {args.filter.upper()}")
    print(f"   Fenêtre: {args.window_size} samples")

    print(f"\n📈 Concordance: {results['concordance']:.2f}%")

    if results['concordance'] >= 95:
        print("\n✅ EXCELLENT - Filtre très stable")
    elif results['concordance'] >= 85:
        print("\n✅ BON - Filtre stable")
    elif results['concordance'] >= 70:
        print("\n⚠️ MOYEN - Stabilité acceptable")
    else:
        print("\n❌ FAIBLE - Filtre instable")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
