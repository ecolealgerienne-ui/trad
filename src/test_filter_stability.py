"""
Test de stabilité des filtres sur fenêtre glissante.

Objectif:
    Tester si un filtre produit le même label (pente) qu'on l'applique sur
    une fenêtre locale (ex: t-100 à t) ou sur tout le dataset.

Protocole:
    1. Appliquer filtre sur TOUT le dataset → labels globaux
    2. Pour 200 positions échantillonnées:
       - Appliquer filtre sur fenêtre [t-100:t] → label local
       - Comparer label local vs label global
    3. Calculer concordance (% d'accord)

Usage:
    python src/test_filter_stability.py \
        --data data_trad/BTCUSD_all_5m.csv \
        --filter octave \
        --window-size 100 \
        --n-samples 200
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import sys
from typing import Tuple, Dict

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent))

from filters import signal_filtfilt


def load_data(file_path: str) -> pd.DataFrame:
    """Charge les données CSV."""
    print(f"📂 Chargement {file_path}...")
    df = pd.read_csv(file_path)

    # Vérifier colonnes requises
    if 'close' not in df.columns:
        raise ValueError("Colonne 'close' manquante dans le CSV")

    print(f"   Samples: {len(df):,}")
    print(f"   Colonnes: {list(df.columns)}")

    return df


def apply_filter(data: np.ndarray, filter_name: str, **kwargs) -> np.ndarray:
    """Applique un filtre sur les données."""
    if filter_name == 'octave':
        step = kwargs.get('step', 0.20)  # Octave20 = step 0.20
        order = kwargs.get('order', 3)
        return signal_filtfilt(data, step=step, order=order)
    else:
        raise ValueError(f"Filtre inconnu: {filter_name}")


def compute_slope_label(filtered: np.ndarray, index: int) -> int:
    """
    Calcule le label de pente à l'index donné.

    Label = 1 si filtered[index-2] > filtered[index-3], sinon 0
    """
    if index < 3:
        return -1  # Pas assez de données

    return int(filtered[index - 2] > filtered[index - 3])


def test_filter_stability(
    data: np.ndarray,
    filter_name: str,
    window_size: int = 100,
    n_samples: int = 200,
    **filter_kwargs
) -> Dict[str, float]:
    """
    Teste la stabilité d'un filtre sur fenêtre glissante.

    Args:
        data: Données brutes (Close)
        filter_name: Nom du filtre ('octave', 'savgol', etc.)
        window_size: Taille de la fenêtre locale
        n_samples: Nombre de positions à échantillonner
        **filter_kwargs: Paramètres du filtre

    Returns:
        Dict avec statistiques de concordance
    """
    n = len(data)

    print(f"\n{'='*80}")
    print(f"TEST DE STABILITÉ - {filter_name.upper()}")
    print(f"{'='*80}")

    print(f"\n⚙️ Configuration:")
    print(f"   Filtre: {filter_name}")
    print(f"   Taille fenêtre: {window_size}")
    print(f"   Samples testés: {n_samples}")
    print(f"   Paramètres filtre: {filter_kwargs}")

    # ÉTAPE 1: Appliquer filtre sur TOUT le dataset (label global)
    print(f"\n🔧 Étape 1: Application du filtre sur TOUT le dataset...")
    filtered_global = apply_filter(data, filter_name, **filter_kwargs)

    # Calculer tous les labels globaux
    labels_global = np.zeros(n, dtype=int)
    for i in range(3, n):
        labels_global[i] = compute_slope_label(filtered_global, i)

    n_up_global = (labels_global == 1).sum()
    n_down_global = (labels_global == 0).sum()
    print(f"   Labels globaux: {n_up_global:,} UP ({n_up_global/n*100:.1f}%), "
          f"{n_down_global:,} DOWN ({n_down_global/n*100:.1f}%)")

    # ÉTAPE 2: Échantillonner les positions à tester
    # Éviter les bords (besoin de window_size avant et 3 après pour le label)
    min_idx = window_size + 3
    max_idx = n - 3

    if max_idx <= min_idx:
        raise ValueError(f"Dataset trop petit: {n} samples, besoin de >= {window_size + 6}")

    sample_indices = np.linspace(min_idx, max_idx, n_samples, dtype=int)

    print(f"\n🔧 Étape 2: Test sur fenêtre glissante ({n_samples} positions)...")
    print(f"   Indices testés: [{sample_indices[0]}, ..., {sample_indices[-1]}]")

    # ÉTAPE 3: Pour chaque position, appliquer filtre sur fenêtre locale
    concordance = []
    labels_local = []
    labels_global_sampled = []

    for i, t in enumerate(sample_indices):
        # Fenêtre locale: [t-window_size : t+1]
        window_start = t - window_size
        window_end = t + 1
        window_data = data[window_start:window_end]

        # Appliquer filtre sur fenêtre
        filtered_local = apply_filter(window_data, filter_name, **filter_kwargs)

        # Calculer label local
        # Dans la fenêtre, t correspond à l'index -1 (dernier élément)
        # Donc on compare filtered_local[-2] vs filtered_local[-3]
        label_local = int(filtered_local[-2] > filtered_local[-3])

        # Label global à cette position
        label_global = labels_global[t]

        # Concordance
        agree = (label_local == label_global)
        concordance.append(agree)
        labels_local.append(label_local)
        labels_global_sampled.append(label_global)

        # Affichage progressif
        if (i + 1) % 50 == 0:
            current_concordance = np.mean(concordance) * 100
            print(f"   Progression: {i+1}/{n_samples} - Concordance actuelle: {current_concordance:.1f}%")

    # ÉTAPE 4: Statistiques finales
    concordance = np.array(concordance)
    labels_local = np.array(labels_local)
    labels_global_sampled = np.array(labels_global_sampled)

    concordance_pct = concordance.mean() * 100

    # Concordance par classe
    mask_up_global = (labels_global_sampled == 1)
    mask_down_global = (labels_global_sampled == 0)

    concordance_up = concordance[mask_up_global].mean() * 100 if mask_up_global.any() else 0
    concordance_down = concordance[mask_down_global].mean() * 100 if mask_down_global.any() else 0

    # Distribution des labels locaux
    n_up_local = (labels_local == 1).sum()
    n_down_local = (labels_local == 0).sum()

    print(f"\n{'='*80}")
    print("RÉSULTATS")
    print(f"{'='*80}")

    print(f"\n📊 Labels locaux vs globaux:")
    print(f"   Global échantillonné: {mask_up_global.sum()} UP, {mask_down_global.sum()} DOWN")
    print(f"   Local (fenêtre):      {n_up_local} UP, {n_down_local} DOWN")

    print(f"\n✅ Concordance globale: {concordance_pct:.2f}%")
    print(f"   - Sur labels UP:   {concordance_up:.2f}%")
    print(f"   - Sur labels DOWN: {concordance_down:.2f}%")

    # Analyse des désaccords
    n_disagree = (~concordance).sum()
    print(f"\n❌ Désaccords: {n_disagree}/{n_samples} ({n_disagree/n_samples*100:.1f}%)")

    if n_disagree > 0:
        # Types de désaccords
        flip_to_up = ((labels_global_sampled == 0) & (labels_local == 1)).sum()
        flip_to_down = ((labels_global_sampled == 1) & (labels_local == 0)).sum()
        print(f"   - Global DOWN → Local UP:   {flip_to_up}")
        print(f"   - Global UP   → Local DOWN: {flip_to_down}")

    return {
        'concordance': concordance_pct,
        'concordance_up': concordance_up,
        'concordance_down': concordance_down,
        'n_samples': n_samples,
        'n_disagree': n_disagree
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test de stabilité des filtres sur fenêtre glissante",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Données
    parser.add_argument('--data', type=str, required=True,
                        help='Fichier CSV avec les données (doit contenir colonne "close")')

    # Filtre
    parser.add_argument('--filter', type=str, default='octave',
                        choices=['octave'],
                        help='Filtre à tester')
    parser.add_argument('--step', type=float, default=0.20,
                        help='Paramètre step pour Octave (0.20 = Octave20)')
    parser.add_argument('--order', type=int, default=3,
                        help='Ordre du filtre Butterworth')

    # Test
    parser.add_argument('--window-size', type=int, default=100,
                        help='Taille de la fenêtre locale')
    parser.add_argument('--n-samples', type=int, default=200,
                        help='Nombre de positions à échantillonner')

    args = parser.parse_args()

    print("="*80)
    print("TEST DE STABILITÉ DES FILTRES")
    print("="*80)

    # Charger données
    df = load_data(args.data)
    close = df['close'].values

    print(f"\n📊 Statistiques des données:")
    print(f"   Close: min={close.min():.2f}, max={close.max():.2f}, "
          f"mean={close.mean():.2f}, std={close.std():.2f}")

    # Paramètres du filtre
    filter_kwargs = {}
    if args.filter == 'octave':
        filter_kwargs = {'step': args.step, 'order': args.order}

    # Test de stabilité
    results = test_filter_stability(
        data=close,
        filter_name=args.filter,
        window_size=args.window_size,
        n_samples=args.n_samples,
        **filter_kwargs
    )

    # Résumé final
    print(f"\n{'='*80}")
    print("RÉSUMÉ")
    print(f"{'='*80}")

    print(f"\n🎯 Filtre testé: {args.filter.upper()}")
    print(f"   Paramètres: {filter_kwargs}")
    print(f"   Fenêtre: {args.window_size} samples")

    print(f"\n📈 Concordance: {results['concordance']:.2f}%")

    if results['concordance'] >= 95:
        print("\n✅ EXCELLENT - Le filtre est très stable sur fenêtre glissante")
    elif results['concordance'] >= 85:
        print("\n✅ BON - Le filtre est stable")
    elif results['concordance'] >= 70:
        print("\n⚠️ MOYEN - Le filtre a une stabilité acceptable")
    else:
        print("\n❌ FAIBLE - Le filtre est instable sur fenêtre glissante")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
