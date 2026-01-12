#!/usr/bin/env python3
"""
Affiche un échantillon de données X et Y pour vérification visuelle.
Vérifie si les valeurs sont logiques (pas de timestamps dans labels, ranges corrects).
"""

import sys
import numpy as np

def check_sample(npz_path, n_samples=5, sample_idx=None):
    """
    Affiche n_samples échantillons de X et Y pour inspection visuelle.

    Args:
        npz_path: Chemin du fichier NPZ
        n_samples: Nombre d'échantillons à afficher
        sample_idx: Index spécifique à afficher (optionnel)
    """

    print("=" * 100)
    print("🔍 VÉRIFICATION ÉCHANTILLON DE DONNÉES")
    print("=" * 100)
    print(f"\n📂 Fichier: {npz_path}")

    # Charger données
    data = np.load(npz_path, allow_pickle=True)

    # Utiliser train par défaut
    X = data['X_train']
    Y = data['Y_train']

    print(f"\n📊 Shapes:")
    print(f"   X_train: {X.shape}  # (samples, 25 timesteps, ~22 features)")
    print(f"   Y_train: {Y.shape}  # (samples, 8 ou 13 colonnes)")

    # Déterminer les indices à afficher
    if sample_idx is not None:
        indices = [sample_idx]
    else:
        # Prendre échantillons répartis dans le dataset
        total = X.shape[0]
        step = total // (n_samples + 1)
        indices = [step * (i + 1) for i in range(n_samples)]

    print(f"\n📋 Indices affichés: {indices}")

    # ========================================================================
    # VÉRIFICATION RANGES ATTENDUES
    # ========================================================================
    print("\n" + "=" * 100)
    print("⚠️  RANGES ATTENDUES")
    print("=" * 100)
    print("""
Y colonnes:
  [0]  timestamp       : Unix timestamp (>1600000000, <2000000000)
  [1]  asset_id        : 0-4 (BTC=0, ETH=1, BNB=2, ADA=3, LTC=4)
  [2]  regime          : 0-3 (RANGE_LOW=0, RANGE_HIGH=1, TREND_LOW=2, TREND_HIGH=3)
  [3]  trend_strength  : 0.0-1.0
  [4]  volatility      : 0.0-1.0
  [5]  macd_direction  : 0 ou 1 (DOWN=0, UP=1)
  [6]  rsi_direction   : 0 ou 1
  [7]  cci_direction   : 0 ou 1

Si Y a 13 colonnes (enrichi):
  [8]  regime_prob_0   : 0.0-1.0
  [9]  regime_prob_1   : 0.0-1.0
  [10] regime_prob_2   : 0.0-1.0
  [11] regime_prob_3   : 0.0-1.0
  [12] regime_pred     : 0-3
""")

    # ========================================================================
    # AFFICHER ÉCHANTILLONS
    # ========================================================================
    for idx in indices:
        print("\n" + "=" * 100)
        print(f"📌 ÉCHANTILLON #{idx}")
        print("=" * 100)

        x_sample = X[idx]  # Shape: (25, ~22)
        y_sample = Y[idx]  # Shape: (8 ou 13,)

        # ====================================================================
        # Y (LABELS)
        # ====================================================================
        print(f"\n🎯 Y[{idx}] - LABELS:")
        print(f"   Shape: {y_sample.shape}")
        print()

        # Colonnes de base (0-7)
        timestamp = y_sample[0]
        asset_id = int(y_sample[1])
        regime = int(y_sample[2])
        trend_strength = y_sample[3]
        volatility = y_sample[4]
        macd_dir = int(y_sample[5])
        rsi_dir = int(y_sample[6])
        cci_dir = int(y_sample[7])

        # Vérifier timestamp valide
        timestamp_ok = 1600000000 < timestamp < 2000000000
        timestamp_status = "✅" if timestamp_ok else "❌ INVALIDE"
        print(f"   [0] timestamp:       {timestamp:.0f}  {timestamp_status}")

        # Vérifier asset_id
        asset_ok = 0 <= asset_id <= 4
        asset_status = "✅" if asset_ok else "❌ INVALIDE"
        asset_names = {0: "BTC", 1: "ETH", 2: "BNB", 3: "ADA", 4: "LTC"}
        asset_name = asset_names.get(asset_id, "UNKNOWN")
        print(f"   [1] asset_id:        {asset_id} ({asset_name})  {asset_status}")

        # Vérifier regime
        regime_ok = 0 <= regime <= 3
        regime_status = "✅" if regime_ok else "❌ INVALIDE"
        regime_names = {0: "RANGE_LOW", 1: "RANGE_HIGH", 2: "TREND_LOW", 3: "TREND_HIGH"}
        regime_name = regime_names.get(regime, "INVALID")
        print(f"   [2] regime:          {regime} ({regime_name})  {regime_status}")

        # Vérifier trend_strength et volatility
        ts_ok = 0.0 <= trend_strength <= 1.0
        ts_status = "✅" if ts_ok else "❌ INVALIDE"
        print(f"   [3] trend_strength:  {trend_strength:.4f}  {ts_status}")

        vol_ok = 0.0 <= volatility <= 1.0
        vol_status = "✅" if vol_ok else "❌ INVALIDE"
        print(f"   [4] volatility:      {volatility:.4f}  {vol_status}")

        # Vérifier directions
        macd_ok = macd_dir in [0, 1]
        macd_status = "✅" if macd_ok else "❌ INVALIDE"
        macd_label = "UP" if macd_dir == 1 else "DOWN"
        print(f"   [5] macd_direction:  {macd_dir} ({macd_label})  {macd_status}")

        rsi_ok = rsi_dir in [0, 1]
        rsi_status = "✅" if rsi_ok else "❌ INVALIDE"
        rsi_label = "UP" if rsi_dir == 1 else "DOWN"
        print(f"   [6] rsi_direction:   {rsi_dir} ({rsi_label})  {rsi_status}")

        cci_ok = cci_dir in [0, 1]
        cci_status = "✅" if cci_ok else "❌ INVALIDE"
        cci_label = "UP" if cci_dir == 1 else "DOWN"
        print(f"   [7] cci_direction:   {cci_dir} ({cci_label})  {cci_status}")

        # Si enrichi (13 colonnes)
        if len(y_sample) == 13:
            print(f"\n   📊 ENRICHISSEMENT:")
            regime_prob_0 = y_sample[8]
            regime_prob_1 = y_sample[9]
            regime_prob_2 = y_sample[10]
            regime_prob_3 = y_sample[11]
            regime_pred = int(y_sample[12])

            probs_ok = all(0.0 <= p <= 1.0 for p in [regime_prob_0, regime_prob_1, regime_prob_2, regime_prob_3])
            probs_status = "✅" if probs_ok else "❌ INVALIDE"
            print(f"   [8-11] regime_probs: [{regime_prob_0:.3f}, {regime_prob_1:.3f}, {regime_prob_2:.3f}, {regime_prob_3:.3f}]  {probs_status}")

            pred_ok = 0 <= regime_pred <= 3
            pred_status = "✅" if pred_ok else "❌ INVALIDE"
            pred_name = regime_names.get(regime_pred, "INVALID")
            print(f"   [12] regime_pred:    {regime_pred} ({pred_name})  {pred_status}")
        elif len(y_sample) == 8:
            print(f"\n   ⚠️  Dataset PAS ENRICHI (8 colonnes)")

        # Résumé validité Y
        print(f"\n   📋 RÉSUMÉ Y:")
        all_ok = all([timestamp_ok, asset_ok, regime_ok, ts_ok, vol_ok, macd_ok, rsi_ok, cci_ok])
        if all_ok:
            print(f"      ✅ Toutes les valeurs Y sont VALIDES")
        else:
            print(f"      ❌ PROBLÈMES DÉTECTÉS dans Y")

        # ====================================================================
        # X (FEATURES)
        # ====================================================================
        print(f"\n📊 X[{idx}] - FEATURES:")
        print(f"   Shape: {x_sample.shape}  # (25 timesteps, {x_sample.shape[1]} features)")
        print()

        # Afficher premier et dernier timestep
        print(f"   Timestep 0 (début séquence):")
        first = x_sample[0]
        print(f"      timestamp:    {first[0]:.0f}")
        print(f"      asset_id:     {int(first[1])}")
        print(f"      features[2-]: {first[2:5]}... (premiers 3 features)")
        print(f"      min:          {first[2:].min():.4f}")
        print(f"      max:          {first[2:].max():.4f}")
        print(f"      mean:         {first[2:].mean():.4f}")

        print(f"\n   Timestep 24 (fin séquence):")
        last = x_sample[24]
        print(f"      timestamp:    {last[0]:.0f}")
        print(f"      asset_id:     {int(last[1])}")
        print(f"      features[2-]: {last[2:5]}... (premiers 3 features)")
        print(f"      min:          {last[2:].min():.4f}")
        print(f"      max:          {last[2:].max():.4f}")
        print(f"      mean:         {last[2:].mean():.4f}")

        # Vérifications X
        print(f"\n   🔍 VÉRIFICATIONS X:")

        # Timestamps croissants?
        timestamps_x = x_sample[:, 0]
        timestamps_increasing = np.all(np.diff(timestamps_x) > 0)
        ts_increase_status = "✅" if timestamps_increasing else "❌ NON CROISSANTS"
        print(f"      Timestamps croissants: {ts_increase_status}")

        # Asset ID constant?
        asset_ids_x = x_sample[:, 1]
        asset_constant = np.all(asset_ids_x == asset_ids_x[0])
        asset_const_status = "✅" if asset_constant else "❌ ASSET_ID VARIE"
        print(f"      Asset ID constant:     {asset_const_status}")

        # Features (colonnes 2+) ont des ranges raisonnables?
        features = x_sample[:, 2:]
        has_nan = np.isnan(features).any()
        has_inf = np.isinf(features).any()
        nan_status = "❌ NaN DÉTECTÉ" if has_nan else "✅"
        inf_status = "❌ Inf DÉTECTÉ" if has_inf else "✅"
        print(f"      NaN dans features:     {nan_status}")
        print(f"      Inf dans features:     {inf_status}")

        feat_min = features.min()
        feat_max = features.max()
        feat_mean = features.mean()
        print(f"      Range features:        [{feat_min:.4f}, {feat_max:.4f}]")
        print(f"      Mean features:         {feat_mean:.4f}")

        # Les features devraient être normalisées (la plupart entre -3 et +3 si z-score)
        if feat_min < -10 or feat_max > 10:
            print(f"      ⚠️  WARNING: Range très large, features peut-être pas normalisées")
        else:
            print(f"      ✅ Range features raisonnable")

    # ========================================================================
    # DIAGNOSTIC GLOBAL
    # ========================================================================
    print("\n" + "=" * 100)
    print("🔍 DIAGNOSTIC GLOBAL")
    print("=" * 100)

    # Vérifier toutes les colonnes Y
    print(f"\n📊 Vérification globale Y_train:")

    # Colonne 2 (regime) - problème critique détecté par user
    regimes_all = Y[:, 2]
    unique_regimes = np.unique(regimes_all)
    print(f"\n   Colonne [2] regime:")
    print(f"      Valeurs uniques: {unique_regimes}")

    # ❌ PROBLÈME: Si la colonne regime contient des timestamps au lieu de 0-3
    if np.any(unique_regimes > 10):
        print(f"      ❌ PROBLÈME CRITIQUE DÉTECTÉ!")
        print(f"      ❌ La colonne regime contient des valeurs > 10 (probablement des timestamps)")
        print(f"      ❌ Valeurs attendues: 0, 1, 2, 3 (régimes)")
        print(f"      ❌ Valeurs trouvées: {unique_regimes[:10]}...")
        print(f"\n   🔧 CAUSE PROBABLE:")
        print(f"      Les colonnes Y sont peut-être mal indexées ou mal calculées")
        print(f"      Vérifier prepare_data_regime.py ligne ~800-900 (création de Y)")
    else:
        print(f"      ✅ Valeurs correctes (0-3)")
        for r in unique_regimes:
            if 0 <= r <= 3:
                count = np.sum(regimes_all == r)
                pct = count / len(regimes_all) * 100
                regime_name = {0: "RANGE_LOW", 1: "RANGE_HIGH", 2: "TREND_LOW", 3: "TREND_HIGH"}[int(r)]
                print(f"      Régime {int(r)} ({regime_name:15}): {count:7,} ({pct:5.1f}%)")

    # Vérifier colonnes 5-7 (directions)
    print(f"\n   Colonnes [5-7] directions (MACD/RSI/CCI):")
    for i, name in [(5, "MACD"), (6, "RSI"), (7, "CCI")]:
        directions = Y[:, i]
        unique_dirs = np.unique(directions)
        if not np.all(np.isin(unique_dirs, [0, 1])):
            print(f"      ❌ {name}: Valeurs non binaires: {unique_dirs}")
        else:
            count_0 = np.sum(directions == 0)
            count_1 = np.sum(directions == 1)
            pct_1 = count_1 / len(directions) * 100
            print(f"      ✅ {name}: DOWN={count_0:,}, UP={count_1:,} ({pct_1:.1f}% UP)")

    print("\n" + "=" * 100)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Affiche échantillon de données pour vérification visuelle')
    parser.add_argument('--data', type=str,
                       default='data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz',
                       help='Chemin du fichier NPZ')
    parser.add_argument('--n-samples', type=int, default=3,
                       help='Nombre d\'échantillons à afficher (défaut: 3)')
    parser.add_argument('--index', type=int, default=None,
                       help='Index spécifique à afficher (optionnel)')

    args = parser.parse_args()

    check_sample(args.data, args.n_samples, args.index)
