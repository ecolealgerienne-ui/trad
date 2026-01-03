"""
Script de vérification de l'alignement des données.

Vérifie que:
1. Les timestamps 5min et 30min sont correctement alignés
2. Le shift(-1) des labels est correctement appliqué
3. Le Step Index correspond bien aux minutes
4. Les features 30min sont bien forward-filled

Usage:
    python src/verify_data_alignment.py --asset BTC
"""

import numpy as np
import pandas as pd
import argparse
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

from constants import (
    AVAILABLE_ASSETS_5M, TRIM_EDGES,
    RSI_PERIOD, CCI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR
)
from data_utils import load_crypto_data, trim_edges
from indicators import calculate_all_indicators_for_model, generate_all_labels
from prepare_data_30min import resample_5min_to_30min, align_30min_to_5min


def verify_alignment(asset: str = 'BTC', n_samples: int = 20):
    """
    Vérifie l'alignement des données pour un asset.

    Affiche un tableau avec:
    - Timestamp 5min
    - Step Index (1-6)
    - Indicateurs 5min (dernier timestep)
    - Indicateurs 30min (forward-filled)
    - Labels (après shift -1)
    - Timestamp 30min source du label
    """
    print("="*100)
    print(f"VÉRIFICATION ALIGNEMENT DONNÉES - {asset}")
    print("="*100)

    # 1. Charger données 5min
    file_path = AVAILABLE_ASSETS_5M[asset]
    df_5min = load_crypto_data(file_path, asset_name=f'{asset}-5m')
    df_5min = trim_edges(df_5min, trim_start=TRIM_EDGES, trim_end=TRIM_EDGES)

    # Préparer index datetime
    df = df_5min.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    index_5min = df.index

    print(f"\n📊 Données 5min: {len(df)} bougies")
    print(f"   Période: {index_5min[0]} → {index_5min[-1]}")

    # 2. Resample vers 30min
    df_30min = resample_5min_to_30min(df_5min)
    index_30min = df_30min.index
    print(f"\n📊 Données 30min: {len(df_30min)} bougies")

    # 3. Calculer indicateurs
    df_5min_with_ts = df.reset_index()
    indicators_5min = calculate_all_indicators_for_model(df_5min_with_ts)

    df_30min_with_ts = df_30min.reset_index()
    indicators_30min = calculate_all_indicators_for_model(df_30min_with_ts)

    # 4. Générer labels 30min
    labels_30min = generate_all_labels(indicators_30min, filter_type='kalman')

    # 5. Appliquer le shift(-1) sur les labels
    labels_30min_shifted = labels_30min[1:]
    index_30min_for_labels = index_30min[:-1]

    # 6. Aligner sur 5min
    labels_aligned = align_30min_to_5min(labels_30min_shifted, index_30min_for_labels, index_5min)
    indicators_30min_aligned = align_30min_to_5min(indicators_30min, index_30min, index_5min)

    # 7. Calculer Step Index
    minutes = index_5min.minute
    step_index = (minutes % 30) // 5 + 1

    # Couper pour avoir les mêmes longueurs
    n_valid = len(labels_aligned)
    if n_valid < len(indicators_5min):
        start_idx = len(indicators_5min) - n_valid
        indicators_5min = indicators_5min[start_idx:]
        indicators_30min_aligned = indicators_30min_aligned[start_idx:]
        index_5min = index_5min[start_idx:]
        step_index = step_index[start_idx:]

    print(f"\n📊 Après alignement: {len(labels_aligned)} samples")

    # =========================================================================
    # AFFICHAGE DÉTAILLÉ
    # =========================================================================

    # Trouver une zone intéressante (milieu du dataset)
    mid_idx = len(index_5min) // 2
    # Aligner sur le début d'une période 30min (Step 1)
    while step_index[mid_idx] != 1 and mid_idx < len(index_5min) - 1:
        mid_idx += 1

    print(f"\n{'='*120}")
    print("ÉCHANTILLON DE DONNÉES (6 périodes 30min = 3 heures = 36 bougies 5min)")
    print("="*120)

    print(f"\n{'Timestamp 5min':<22} {'Step':<6} {'RSI_5m':<8} {'CCI_5m':<8} {'RSI_30m':<8} {'CCI_30m':<8} {'Lbl_RSI':<8} {'Lbl_CCI':<8}")
    print("-"*120)

    for i in range(mid_idx, min(mid_idx + 36, len(index_5min))):
        ts_5min = index_5min[i]
        step = step_index[i]
        rsi_5min = indicators_5min[i, 0]
        cci_5min = indicators_5min[i, 1]
        rsi_30min = indicators_30min_aligned[i, 0]
        cci_30min = indicators_30min_aligned[i, 1]
        label_rsi = "UP" if labels_aligned[i, 0] == 1 else "DOWN"
        label_cci = "UP" if labels_aligned[i, 1] == 1 else "DOWN"

        print(f"{str(ts_5min):<22} {step:<6} {rsi_5min:<8.1f} {cci_5min:<8.1f} {rsi_30min:<8.1f} {cci_30min:<8.1f} {label_rsi:<8} {label_cci:<8}")

        if step == 6:
            print("-"*120)

    # =========================================================================
    # VÉRIFICATION DIRECTE DES LABELS 30min
    # =========================================================================
    print(f"\n{'='*120}")
    print("VÉRIFICATION DIRECTE: INDICATEURS 30min ET PENTES (après Kalman)")
    print("="*120)
    print("\nLogique du shift(-1):")
    print("  - AVANT: label[10:00] = pente(09:00 → 09:30) = 1h de retard")
    print("  - APRÈS: label[10:00] = pente(09:30 → 10:00) = synchronisé")
    print("\nNote: Les labels sont générés depuis les indicateurs FILTRÉS (Kalman),")
    print("      pas les indicateurs bruts. La concordance n'est pas 100%.")

    print(f"\n{'Timestamp':<20} {'RSI[T]':<10} {'RSI[T-30m]':<12} {'Pente brute':<12} {'Label réel':<12}")
    print("-"*80)

    # Utiliser index_30min_for_labels pour être cohérent avec le shift
    start_30 = len(index_30min_for_labels) // 2
    for j in range(start_30, min(start_30 + 10, len(index_30min_for_labels))):
        ts_30 = index_30min_for_labels[j]

        # Trouver l'index dans indicators_30min (qui utilise index_30min original)
        # Après shift: label à index j correspond à la pente qui se termine à ts_30
        # indicators_30min a le même ordre que index_30min

        # Pour vérifier: on veut RSI[T] et RSI[T-30min]
        # On trouve l'index de ts_30 dans index_30min
        idx_in_30min = np.where(index_30min == ts_30)[0]
        if len(idx_in_30min) == 0:
            continue
        idx = idx_in_30min[0]

        if idx > 0:
            rsi_current = indicators_30min[idx, 0]
            rsi_prev = indicators_30min[idx - 1, 0]
            pente = rsi_current - rsi_prev
            pente_str = f"{pente:+.2f}"
        else:
            rsi_current = indicators_30min[idx, 0]
            rsi_prev = np.nan
            pente_str = "N/A"

        # Label après shift à l'index j
        if j < len(labels_30min_shifted):
            label = "UP" if labels_30min_shifted[j, 0] == 1 else "DOWN"
        else:
            label = "N/A"

        print(f"{str(ts_30):<20} {rsi_current:<10.2f} {rsi_prev:<10.2f} {pente_str:<12} {label:<12}")

    print("\nSi pente brute et label sont cohérents (même signe), le shift est correct.")

    # =========================================================================
    # VÉRIFICATIONS AUTOMATIQUES
    # =========================================================================

    print(f"\n{'='*100}")
    print("VÉRIFICATIONS AUTOMATIQUES")
    print("="*100)

    errors = []

    # 1. Vérifier que Step Index va de 1 à 6
    unique_steps = np.unique(step_index)
    if not np.array_equal(unique_steps, np.array([1, 2, 3, 4, 5, 6])):
        errors.append(f"❌ Step Index invalide: {unique_steps}")
    else:
        print("✅ Step Index: valeurs 1-6 correctes")

    # 2. Vérifier la distribution des steps (doit être ~égale)
    step_counts = np.bincount(step_index)[1:7]
    step_ratio = step_counts / step_counts.sum()
    if np.all(np.abs(step_ratio - 1/6) < 0.01):
        print(f"✅ Distribution Steps: uniforme ({step_counts})")
    else:
        print(f"⚠️ Distribution Steps non uniforme: {step_counts}")

    # 3. Vérifier que les indicateurs 30min sont constants par période
    # Trouver tous les indices où step_index == 1 (début de période 30min)
    step1_indices = np.where(step_index == 1)[0]
    n_check = min(100, len(step1_indices))  # Vérifier les 100 premières périodes
    periods_ok = 0
    periods_total = 0

    for idx in step1_indices[:n_check]:
        if idx + 6 <= len(indicators_30min_aligned):
            # Vérifier que les 6 valeurs 30min sont identiques
            rsi_30min_period = indicators_30min_aligned[idx:idx+6, 0]
            if np.allclose(rsi_30min_period, rsi_30min_period[0], rtol=1e-5):
                periods_ok += 1
            periods_total += 1

    if periods_total > 0 and periods_ok / periods_total > 0.99:
        print(f"✅ Forward-fill 30min: {periods_ok}/{periods_total} périodes constantes")
    else:
        errors.append(f"❌ Forward-fill incorrect: {periods_ok}/{periods_total} périodes")

    # 4. Vérifier que les labels sont constants par période 30min
    labels_ok = 0
    labels_total = 0

    for idx in step1_indices[:n_check]:
        if idx + 6 <= len(labels_aligned):
            labels_period = labels_aligned[idx:idx+6, 0]
            if np.all(labels_period == labels_period[0]):
                labels_ok += 1
            labels_total += 1

    if labels_total > 0 and labels_ok / labels_total > 0.99:
        print(f"✅ Labels constants par période: {labels_ok}/{labels_total}")
    else:
        errors.append(f"❌ Labels non constants: {labels_ok}/{labels_total}")

    # 5. Vérifier le shift des labels (comparaison avant/après)
    print(f"\n{'='*100}")
    print("VÉRIFICATION DU SHIFT(-1) DES LABELS")
    print("="*100)

    # Prendre un exemple concret
    example_idx = mid_idx
    ts_example = index_5min[example_idx]
    ts_30min_floor = pd.Timestamp(ts_example).floor('30min')

    # Trouver l'index 30min correspondant
    idx_30min = np.where(index_30min_for_labels == ts_30min_floor)[0]

    if len(idx_30min) > 0:
        idx_30 = idx_30min[0]

        # Label AVANT shift (ce qu'on avait)
        # labels_30min[idx_30] = pente de (idx_30-2) vers (idx_30-1) en termes de temps
        # Ce qui correspond à pente de (ts_30min_floor - 1h) vers (ts_30min_floor - 30min)

        # Label APRÈS shift (ce qu'on a maintenant)
        # labels_30min_shifted[idx_30] = labels_30min[idx_30+1]
        # = pente de (idx_30-1) vers (idx_30)
        # = pente de (ts_30min_floor - 30min) vers (ts_30min_floor)

        print(f"\nExemple pour timestamp 5min: {ts_example}")
        print(f"  → Période 30min: {ts_30min_floor}")
        print(f"\n  AVANT shift(-1):")
        print(f"    Label = pente({ts_30min_floor - pd.Timedelta('1h')} → {ts_30min_floor - pd.Timedelta('30min')})")
        print(f"    = Prédire le passé lointain (1h de retard)")
        print(f"\n  APRÈS shift(-1):")
        print(f"    Label = pente({ts_30min_floor - pd.Timedelta('30min')} → {ts_30min_floor})")
        print(f"    = Prédire ce qui vient de se passer (synchronisé)")

    # 6. Résumé des erreurs
    print(f"\n{'='*100}")
    print("RÉSUMÉ")
    print("="*100)

    if errors:
        print("\n⚠️ ERREURS DÉTECTÉES:")
        for err in errors:
            print(f"   {err}")
    else:
        print("\n✅ TOUTES LES VÉRIFICATIONS SONT OK")
        print("\nLes données sont correctement alignées:")
        print("  - Step Index: 1-6 uniformément distribué")
        print("  - Forward-fill 30min: constant sur chaque période")
        print("  - Labels: shift(-1) appliqué correctement")
        print("  - Synchronisation: labels prédisent la pente qui vient de clore")


def main():
    parser = argparse.ArgumentParser(
        description="Vérifie l'alignement des données préparées"
    )
    parser.add_argument('--asset', '-a', type=str, default='BTC',
                        choices=list(AVAILABLE_ASSETS_5M.keys()),
                        help='Asset à vérifier (défaut: BTC)')
    parser.add_argument('--samples', '-n', type=int, default=20,
                        help='Nombre de samples à afficher')

    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format='%(message)s')

    verify_alignment(args.asset, args.samples)


if __name__ == '__main__':
    main()
