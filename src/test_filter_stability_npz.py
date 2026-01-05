"""
Test de stabilité des filtres à partir des datasets .npz.

Ce script :
1. Charge un dataset .npz (ex: dataset_..._macd_kalman.npz)
2. Reconstitue le signal MACD brut (ou charge depuis métadonnées)
3. Teste la stabilité du filtre en comparant :
   - Labels globaux (Y du .npz, calculés sur tout le dataset)
   - Labels locaux (filtre appliqué sur fenêtre glissante)

Usage:
    python src/test_filter_stability_npz.py \
        --npz-file data/prepared/dataset_btc_eth_bnb_ada_ltc_ohlcv2_macd_kalman.npz \
        --csv-file data_trad/BTCUSD_all_5m.csv \
        --filter kalman \
        --window-size 100 \
        --n-samples 200 \
        --split test
"""

import numpy as np
import pandas as pd
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent))

from filters import signal_filtfilt
from pykalman import KalmanFilter


def load_npz_dataset(npz_path: str, split: str = 'test') -> Dict:
    """
    Charge un dataset .npz et extrait les données pour le split demandé.

    Returns:
        Dict avec X, Y, metadata
    """
    print(f"📂 Chargement {npz_path} (split={split})...")

    data = np.load(npz_path, allow_pickle=True)

    X_key = f'X_{split}'
    Y_key = f'Y_{split}'

    if X_key not in data or Y_key not in data:
        raise ValueError(f"Split '{split}' non trouvé dans le fichier .npz")

    X = data[X_key]
    Y = data[Y_key].flatten()  # Labels binaires

    # Charger métadonnées
    metadata = {}
    if 'metadata' in data:
        try:
            metadata = json.loads(str(data['metadata']))
        except:
            pass

    print(f"   X shape: {X.shape} (samples, seq_length, features)")
    print(f"   Y shape: {Y.shape} (labels)")
    print(f"   Metadata: {list(metadata.keys()) if metadata else 'N/A'}")

    return {
        'X': X,
        'Y': Y,
        'metadata': metadata
    }


def load_csv_and_compute_macd(csv_path: str, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Charge un CSV et calcule l'indicateur MACD.

    Returns:
        DataFrame avec colonnes : close, macd, macd_signal, macd_hist
    """
    print(f"\n📂 Chargement {csv_path}...")

    df = pd.read_csv(csv_path)

    # Vérifier colonnes requises
    if 'close' not in df.columns:
        raise ValueError("Colonne 'close' manquante")

    print(f"   Samples: {len(df):,}")

    # Calculer MACD avec ta library
    try:
        from indicators_ta import calculate_macd_ta

        macd_line, signal_line, histogram = calculate_macd_ta(
            df['close'],
            fast_period=fast,
            slow_period=slow,
            signal_period=signal
        )

        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = histogram

    except ImportError:
        # Fallback : calcul manuel
        print("   ⚠️ indicators_ta non disponible, calcul manuel MACD")

        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = macd_line - signal_line

    # Normaliser MACD (0-100) comme dans prepare_data
    macd_values = df['macd'].values
    macd_min = np.nanmin(macd_values)
    macd_max = np.nanmax(macd_values)

    if macd_max > macd_min:
        df['macd_normalized'] = (macd_values - macd_min) / (macd_max - macd_min) * 100
    else:
        df['macd_normalized'] = np.full_like(macd_values, 50.0)

    print(f"   MACD calculé : min={macd_min:.4f}, max={macd_max:.4f}")

    return df


def apply_kalman_filter(data: np.ndarray) -> np.ndarray:
    """Applique filtre de Kalman."""
    mask = ~np.isnan(data)
    valid_data = data[mask]

    if len(valid_data) == 0:
        return data

    # Kalman 1D
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

    # Reconstruire
    filtered = np.full_like(data, np.nan, dtype=float)
    filtered[mask] = filtered_valid

    return filtered


def apply_filter(data: np.ndarray, filter_name: str, **kwargs) -> np.ndarray:
    """Applique un filtre sur les données."""
    if filter_name == 'kalman':
        return apply_kalman_filter(data)

    elif filter_name == 'octave':
        step = kwargs.get('step', 0.20)
        order = kwargs.get('order', 3)
        return signal_filtfilt(data, step=step, order=order)

    else:
        raise ValueError(f"Filtre inconnu: {filter_name}")


def compute_slope_label(filtered: np.ndarray, index: int, lag: int = 1) -> int:
    """
    Calcule le label de pente à l'index donné.

    Label = 1 si filtered[index-lag-1] > filtered[index-lag-2], sinon 0
    (décalage de 1 pour être cohérent avec prepare_data_ohlc_v2.py)
    """
    if index < lag + 2:
        return -1

    return int(filtered[index - lag - 1] > filtered[index - lag - 2])


def test_filter_stability_from_npz(
    npz_data: Dict,
    macd_signal: np.ndarray,
    filter_name: str,
    window_size: int = 100,
    n_samples: int = 200,
    **filter_kwargs
) -> Dict:
    """
    Teste la stabilité d'un filtre en comparant labels globaux (Y du .npz)
    avec labels locaux (filtre sur fenêtre glissante).

    Args:
        npz_data: Données du .npz (X, Y, metadata)
        macd_signal: Signal MACD brut (normalisé 0-100)
        filter_name: 'kalman' ou 'octave'
        window_size: Taille de la fenêtre locale
        n_samples: Nombre de positions à échantillonner
        **filter_kwargs: Paramètres du filtre

    Returns:
        Dict avec statistiques de concordance
    """
    Y_global = npz_data['Y']  # Labels globaux du .npz
    n = len(macd_signal)

    print(f"\n{'='*80}")
    print(f"TEST DE STABILITÉ - {filter_name.upper()}")
    print(f"{'='*80}")

    print(f"\n⚙️ Configuration:")
    print(f"   Filtre: {filter_name}")
    print(f"   Taille fenêtre: {window_size}")
    print(f"   Samples testés: {n_samples}")
    print(f"   Paramètres: {filter_kwargs}")

    # ÉTAPE 1: Appliquer filtre sur TOUT le signal MACD
    print(f"\n🔧 Étape 1: Application filtre sur TOUT le signal MACD...")
    filtered_global = apply_filter(macd_signal, filter_name, **filter_kwargs)

    # Calculer labels globaux RECALCULÉS (pour vérifier cohérence avec Y du .npz)
    labels_global_computed = np.zeros(n, dtype=int)
    for i in range(3, n):
        labels_global_computed[i] = compute_slope_label(filtered_global, i, lag=1)

    # Comparer avec Y du .npz
    # NOTE: Y du .npz peut avoir un offset dû au trim/séquences
    # On compare juste sur les indices disponibles
    y_len = len(Y_global)
    concordance_npz = (labels_global_computed[-y_len:] == Y_global).mean() * 100
    print(f"   Concordance avec Y du .npz: {concordance_npz:.1f}%")
    print(f"   (Si <90%, offset possible entre indices .npz et signal brut)")

    # ÉTAPE 2: Échantillonner positions
    min_idx = window_size + 3
    max_idx = n - 3

    if max_idx <= min_idx:
        raise ValueError(f"Dataset trop petit: {n}, besoin >= {window_size + 6}")

    sample_indices = np.linspace(min_idx, max_idx, n_samples, dtype=int)

    print(f"\n🔧 Étape 2: Test sur fenêtre glissante...")
    print(f"   Indices: [{sample_indices[0]}, ..., {sample_indices[-1]}]")

    # ÉTAPE 3: Pour chaque position, filtre local
    concordance = []
    labels_local = []
    labels_global_sampled = []

    for i, t in enumerate(sample_indices):
        # Fenêtre locale
        window_data = macd_signal[t - window_size : t + 1]

        # Appliquer filtre
        filtered_local = apply_filter(window_data, filter_name, **filter_kwargs)

        # Label local (dans la fenêtre, t = index -1)
        label_local = int(filtered_local[-2] > filtered_local[-3])

        # Label global à cette position
        label_global = labels_global_computed[t]

        # Concordance
        agree = (label_local == label_global)
        concordance.append(agree)
        labels_local.append(label_local)
        labels_global_sampled.append(label_global)

        if (i + 1) % 50 == 0:
            curr_conc = np.mean(concordance) * 100
            print(f"   Progression: {i+1}/{n_samples} - Concordance: {curr_conc:.1f}%")

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

    print(f"\n📊 Labels (global échantillonné vs local):")
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
        description="Test stabilité filtres depuis datasets .npz",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Données
    parser.add_argument('--npz-file', type=str, required=True,
                        help='Fichier .npz avec les labels (ex: dataset_..._macd_kalman.npz)')
    parser.add_argument('--csv-file', type=str, required=True,
                        help='Fichier CSV original pour calculer MACD (ex: BTCUSD_all_5m.csv)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Split à utiliser du .npz')

    # Filtre
    parser.add_argument('--filter', type=str, default='kalman',
                        choices=['kalman', 'octave'],
                        help='Filtre à tester')
    parser.add_argument('--step', type=float, default=0.20,
                        help='Paramètre step pour Octave (0.20 = Octave20)')
    parser.add_argument('--order', type=int, default=3,
                        help='Ordre du filtre Butterworth')

    # Test
    parser.add_argument('--window-size', type=int, default=100,
                        help='Taille fenêtre locale')
    parser.add_argument('--n-samples', type=int, default=200,
                        help='Nombre de positions à échantillonner')

    args = parser.parse_args()

    print("="*80)
    print("TEST DE STABILITÉ - DATASETS .NPZ")
    print("="*80)

    # Charger .npz
    npz_data = load_npz_dataset(args.npz_file, args.split)

    # Charger CSV et calculer MACD
    df = load_csv_and_compute_macd(args.csv_file)
    macd_signal = df['macd_normalized'].values

    print(f"\n📊 MACD normalisé:")
    print(f"   Shape: {macd_signal.shape}")
    print(f"   Min: {np.nanmin(macd_signal):.2f}, Max: {np.nanmax(macd_signal):.2f}")
    print(f"   Mean: {np.nanmean(macd_signal):.2f}")

    # Paramètres filtre
    filter_kwargs = {}
    if args.filter == 'octave':
        filter_kwargs = {'step': args.step, 'order': args.order}

    # Test de stabilité
    results = test_filter_stability_from_npz(
        npz_data=npz_data,
        macd_signal=macd_signal,
        filter_name=args.filter,
        window_size=args.window_size,
        n_samples=args.n_samples,
        **filter_kwargs
    )

    # Résumé
    print(f"\n{'='*80}")
    print("RÉSUMÉ")
    print(f"{'='*80}")

    print(f"\n🎯 Filtre: {args.filter.upper()}")
    print(f"   Paramètres: {filter_kwargs}")
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
