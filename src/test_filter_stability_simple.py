"""
Test de stabilité du filtre Kalman - Version SIMPLIFIÉE.

Scénario :
1. Charger données BTC brutes
2. Extraire 10,000 valeurs
3. Calculer MACD
4. Appliquer Kalman sur MACD → MACD_filtré (GLOBAL)
5. Calculer labels globaux : label[i] = (MACD_filtré[i-2] > MACD_filtré[i-3])
6. Tests de stabilité à partir de l'index 1000 :
   - Pour 200 positions échantillonnées
   - Fenêtre locale [t-100:t+1]
   - Appliquer Kalman sur fenêtre
   - Comparer label local vs global

⚠️ ATTENTION aux indices pour éviter désynchronisation temporelle !

Usage:
    python src/test_filter_stability_simple.py \\
        --csv-file data_trad/BTCUSD_all_5m.csv \\
        --n-samples-total 10000 \\
        --start-test 1000 \\
        --window-size 100 \\
        --n-tests 200
"""

import numpy as np
import pandas as pd
import argparse
import sys
from pathlib import Path
from typing import Dict

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent))

from pykalman import KalmanFilter


def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calcule MACD.

    Returns:
        DataFrame avec colonnes : macd, signal, hist
    """
    try:
        from indicators_ta import calculate_macd_ta

        result = calculate_macd_ta(
            close,
            window_fast=fast,
            window_slow=slow,
            window_sign=signal
        )

        df_macd = pd.DataFrame({
            'macd': result['macd'],
            'signal': result['signal'],
            'hist': result['diff']
        })

    except ImportError:
        # Fallback : calcul manuel
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        df_macd = pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'hist': macd_line - signal_line
        })

    return df_macd


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


def compute_labels_from_filtered(filtered: np.ndarray) -> np.ndarray:
    """
    Calcule les labels de pente depuis un signal filtré.

    ⚠️ FORMULE EXACTE :
        label[i] = 1 si filtered[i-2] > filtered[i-3], sinon 0

    Args:
        filtered: Signal filtré

    Returns:
        labels: Array de labels (NaN pour i < 3)
    """
    n = len(filtered)
    labels = np.full(n, np.nan)

    # On peut calculer le label à partir de i=3
    # (besoin de i-2 et i-3 qui existent pour i >= 3)
    for i in range(3, n):
        labels[i] = 1 if filtered[i-2] > filtered[i-3] else 0

    return labels


def test_filter_stability(
    macd_signal: np.ndarray,
    labels_global: np.ndarray,
    start_index: int = 1000,
    window_size: int = 100,
    n_tests: int = 200
) -> Dict:
    """
    Teste la stabilité du filtre Kalman.

    ⚠️ INDICES CRITIQUES :
        - Position testée : t
        - Fenêtre locale : macd_signal[t-window_size:t+1]  (inclut t)
        - Label local calculé : correspond à l'index t
        - Comparaison : label_local == labels_global[t]

    Args:
        macd_signal: Signal MACD brut
        labels_global: Labels précalculés (pente globale)
        start_index: Commencer les tests à cet index (défaut: 1000)
        window_size: Taille fenêtre locale (défaut: 100)
        n_tests: Nombre de positions à tester (défaut: 200)

    Returns:
        Dict avec statistiques
    """
    n = len(macd_signal)

    print(f"\n{'='*80}")
    print("TEST DE STABILITÉ - KALMAN")
    print(f"{'='*80}")

    print(f"\n⚙️ Configuration:")
    print(f"   Signal total: {n:,} samples")
    print(f"   Début tests: index {start_index:,}")
    print(f"   Fenêtre locale: {window_size} samples")
    print(f"   Nombre tests: {n_tests}")

    # Échantillonner positions
    # ⚠️ ATTENTION : on doit avoir au moins window_size avant, et 3 après pour le label
    min_idx = max(start_index, window_size + 3)
    max_idx = n - 3

    if max_idx <= min_idx:
        raise ValueError(f"Pas assez de données : n={n}, min={min_idx}, max={max_idx}")

    test_indices = np.linspace(min_idx, max_idx, n_tests, dtype=int)

    print(f"   Indices testés: [{test_indices[0]:,}, ..., {test_indices[-1]:,}]")

    # Tests
    print(f"\n🔧 Test de stabilité en cours...")

    concordance = []
    labels_local = []
    labels_global_sampled = []

    for i, t in enumerate(test_indices):
        # ============================================
        # FENÊTRE LOCALE
        # ============================================
        # Extraire fenêtre : [t-window_size, t] inclus
        # Cela donne window_size + 1 samples
        window_start = t - window_size
        window_end = t + 1  # +1 car python range est exclusif

        window_data = macd_signal[window_start:window_end]

        # Vérification
        assert len(window_data) == window_size + 1, f"Fenêtre incorrecte: {len(window_data)} != {window_size + 1}"

        # ============================================
        # APPLIQUER KALMAN LOCAL
        # ============================================
        filtered_local = apply_kalman_filter(window_data)

        # ============================================
        # CALCULER LABEL LOCAL
        # ============================================
        # Dans la fenêtre locale :
        # - index -1 correspond à t (global)
        # - index -2 correspond à t-1 (global)
        # - index -3 correspond à t-2 (global)
        # - index -4 correspond à t-3 (global)
        #
        # Formule : label[t] = filtered[t-2] > filtered[t-3]
        # Dans fenêtre : label = filtered[-3] > filtered[-4]
        #
        # ⚠️ VÉRIFICATION :
        # - filtered[-3] = filtered_local de l'index (window_size + 1) - 3 = window_size - 2
        # - Cela correspond à l'index global t - 2
        # - filtered[-4] correspond à l'index global t - 3
        # ✅ Correct !

        label_local = 1 if filtered_local[-3] > filtered_local[-4] else 0

        # ============================================
        # LABEL GLOBAL
        # ============================================
        label_global = labels_global[t]

        # ============================================
        # COMPARAISON
        # ============================================
        agree = (label_local == label_global)
        concordance.append(agree)
        labels_local.append(label_local)
        labels_global_sampled.append(label_global)

        # Progression
        if (i + 1) % 50 == 0:
            curr = np.mean(concordance) * 100
            print(f"   Progression: {i+1}/{n_tests} - Concordance actuelle: {curr:.1f}%")

    # ============================================
    # STATISTIQUES FINALES
    # ============================================
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

    print(f"\n📊 Distribution des labels:")
    print(f"   Global: {mask_up.sum()} UP ({mask_up.sum()/len(labels_global_sampled)*100:.1f}%), "
          f"{mask_down.sum()} DOWN ({mask_down.sum()/len(labels_global_sampled)*100:.1f}%)")
    print(f"   Local:  {(labels_local == 1).sum()} UP ({(labels_local == 1).sum()/len(labels_local)*100:.1f}%), "
          f"{(labels_local == 0).sum()} DOWN ({(labels_local == 0).sum()/len(labels_local)*100:.1f}%)")

    print(f"\n✅ Concordance globale: {concordance_pct:.2f}%")
    print(f"   - Sur labels UP:   {conc_up:.2f}%")
    print(f"   - Sur labels DOWN: {conc_down:.2f}%")

    n_disagree = (~concordance).sum()
    print(f"\n❌ Désaccords: {n_disagree}/{n_tests} ({n_disagree/n_tests*100:.1f}%)")

    return {
        'concordance': concordance_pct,
        'concordance_up': conc_up,
        'concordance_down': conc_down,
        'n_tests': n_tests,
        'n_disagree': n_disagree
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test de stabilité Kalman - Version simple",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--csv-file', type=str, required=True,
                        help='Fichier CSV BTC')
    parser.add_argument('--n-samples-total', type=int, default=10000,
                        help='Nombre de samples à extraire du CSV')
    parser.add_argument('--start-test', type=int, default=1000,
                        help='Index de début des tests')
    parser.add_argument('--window-size', type=int, default=100,
                        help='Taille fenêtre locale')
    parser.add_argument('--n-tests', type=int, default=200,
                        help='Nombre de positions à tester')

    args = parser.parse_args()

    print("="*80)
    print("TEST DE STABILITÉ KALMAN - VERSION SIMPLIFIÉE")
    print("="*80)

    # ============================================
    # PHASE 1 : PRÉPARATION DONNÉES
    # ============================================
    print(f"\n📂 PHASE 1 : Chargement et préparation données")
    print(f"{'='*80}")

    print(f"\n1. Chargement CSV : {args.csv_file}")
    df = pd.read_csv(args.csv_file)
    print(f"   Total samples CSV: {len(df):,}")

    print(f"\n2. Extraction {args.n_samples_total:,} premières valeurs")
    df = df.head(args.n_samples_total).copy()
    print(f"   Samples retenus: {len(df):,}")

    print(f"\n3. Calcul MACD")
    df_macd = calculate_macd(df['close'])
    df['macd'] = df_macd['macd']
    print(f"   MACD calculé: min={df['macd'].min():.4f}, max={df['macd'].max():.4f}")

    print(f"\n4. Application Kalman sur MACD (GLOBAL)")
    macd_filtered_global = apply_kalman_filter(df['macd'].values)
    df['macd_filtered'] = macd_filtered_global
    print(f"   MACD filtré: min={np.nanmin(macd_filtered_global):.4f}, max={np.nanmax(macd_filtered_global):.4f}")

    print(f"\n5. Calcul labels globaux : label[i] = (filtered[i-2] > filtered[i-3])")
    labels_global = compute_labels_from_filtered(macd_filtered_global)
    df['label'] = labels_global

    # Stats labels
    n_valid_labels = (~np.isnan(labels_global)).sum()
    n_up = (labels_global == 1).sum()
    n_down = (labels_global == 0).sum()
    print(f"   Labels valides: {n_valid_labels:,}")
    print(f"   UP: {n_up:,} ({n_up/n_valid_labels*100:.1f}%)")
    print(f"   DOWN: {n_down:,} ({n_down/n_valid_labels*100:.1f}%)")

    # ============================================
    # PHASE 2 : TEST DE STABILITÉ
    # ============================================
    print(f"\n📊 PHASE 2 : Test de stabilité")
    print(f"{'='*80}")

    results = test_filter_stability(
        macd_signal=df['macd'].values,
        labels_global=labels_global,
        start_index=args.start_test,
        window_size=args.window_size,
        n_tests=args.n_tests
    )

    # ============================================
    # RÉSUMÉ FINAL
    # ============================================
    print(f"\n{'='*80}")
    print("RÉSUMÉ FINAL")
    print(f"{'='*80}")

    print(f"\n🎯 Configuration:")
    print(f"   Dataset: {args.n_samples_total:,} samples")
    print(f"   Tests: {args.n_tests} positions (de {args.start_test:,} à {args.n_samples_total:,})")
    print(f"   Fenêtre: {args.window_size} samples")

    print(f"\n📈 Concordance: {results['concordance']:.2f}%")

    if results['concordance'] >= 95:
        print("\n✅ EXCELLENT - Filtre Kalman très stable")
        print("   → Labels quasi-identiques en fenêtre glissante vs global")
    elif results['concordance'] >= 85:
        print("\n✅ BON - Filtre Kalman stable")
        print("   → Variations mineures acceptables")
    elif results['concordance'] >= 70:
        print("\n⚠️ MOYEN - Stabilité acceptable")
        print("   → Sensible à la taille de fenêtre")
    else:
        print("\n❌ FAIBLE - Filtre instable")
        print("   → Très sensible à la taille de fenêtre")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
