#!/usr/bin/env python3
"""
Script de démonstration: Navigation entre matrices avec métadonnées intégrées.

Montre comment utiliser la nouvelle structure avec (timestamp, asset_id) intégrés
dans X, Y, T, OHLCV.

CLÉ PRIMAIRE COMMUNE: (timestamp, asset_id)
→ Même index i = même sample dans toutes les matrices

Usage:
    python tests/demo_metadata_navigation.py --data data/prepared/dataset_..._macd_direction_only_kalman.npz
"""

import numpy as np
import argparse
from pathlib import Path


def demo_navigation(data_path: str):
    """
    Démontre la navigation entre X, Y, T, OHLCV avec la clé primaire (timestamp, asset_id).
    """
    print("="*80)
    print("DÉMONSTRATION: Navigation entre matrices avec métadonnées intégrées")
    print("="*80)

    # Charger le dataset
    print(f"\n📂 Chargement: {data_path}")
    data = np.load(data_path, allow_pickle=True)

    # Afficher les clés disponibles
    print(f"\n🔑 Clés disponibles dans le .npz:")
    for key in sorted(data.keys()):
        if key != 'metadata':
            print(f"  - {key}: {data[key].shape}")

    # Utiliser le test set pour la démo
    X_test = data['X_test']
    Y_test = data['Y_test']
    T_test = data['T_test']
    OHLCV_test = data['OHLCV_test']

    print(f"\n📊 Shapes Test Set:")
    print(f"  X_test:    {X_test.shape}")
    print(f"  Y_test:    {Y_test.shape}")
    print(f"  T_test:    {T_test.shape}")
    print(f"  OHLCV_test: {OHLCV_test.shape}")

    # Démonstration sur 5 samples
    print(f"\n{'='*80}")
    print("NAVIGATION: 5 exemples")
    print('='*80)

    # Asset ID mapping (inverse)
    asset_names = {0: 'BTC', 1: 'ETH', 2: 'BNB', 3: 'ADA', 4: 'LTC'}

    for i in [0, 100, 500, 1000, 1500]:
        if i >= len(X_test):
            break

        print(f"\n{'─'*80}")
        print(f"Sample {i}:")
        print('─'*80)

        # Extraire la clé primaire depuis chaque matrice
        timestamp_x = X_test[i, -1, 0]  # Dernier timestep de la séquence
        asset_id_x = int(X_test[i, -1, 1])

        timestamp_y = Y_test[i, 0]
        asset_id_y = int(Y_test[i, 1])

        timestamp_t = T_test[i, 0]
        asset_id_t = int(T_test[i, 1])

        timestamp_ohlcv = OHLCV_test[i, 0]
        asset_id_ohlcv = int(OHLCV_test[i, 1])

        # Vérifier l'alignement
        assert timestamp_x == timestamp_y == timestamp_t == timestamp_ohlcv, \
            f"Timestamps désalignés pour sample {i}"
        assert asset_id_x == asset_id_y == asset_id_t == asset_id_ohlcv, \
            f"Asset IDs désalignés pour sample {i}"

        asset_name = asset_names.get(asset_id_x, f"Unknown({asset_id_x})")

        print(f"🔑 CLÉ PRIMAIRE:")
        print(f"  Timestamp: {timestamp_x}")
        print(f"  Asset:     {asset_name} (ID={asset_id_x})")

        # Extraire les données
        print(f"\n📈 FEATURES (X):")
        n_features = X_test.shape[2] - 2  # Exclure timestamp et asset_id
        features_current = X_test[i, -1, 2:]  # Dernier timestep, sans métadonnées
        print(f"  Features (timestep courant): {features_current}")

        print(f"\n🎯 LABEL (Y):")
        direction = int(Y_test[i, 2])
        direction_str = "UP ↑" if direction == 1 else "DOWN ↓"
        print(f"  Direction: {direction_str}")

        print(f"\n🔄 TRANSITION (T):")
        is_transition = int(T_test[i, 2])
        transition_str = "OUI (retournement)" if is_transition == 1 else "NON (continuation)"
        print(f"  Is Transition: {transition_str}")

        print(f"\n💰 PRIX BRUTS (OHLCV):")
        open_price = OHLCV_test[i, 2]
        high_price = OHLCV_test[i, 3]
        low_price = OHLCV_test[i, 4]
        close_price = OHLCV_test[i, 5]
        volume = OHLCV_test[i, 6]
        print(f"  Open:   {open_price}")
        print(f"  High:   {high_price}")
        print(f"  Low:    {low_price}")
        print(f"  Close:  {close_price}")
        print(f"  Volume: {volume}")

    # Démonstration de backtest causal
    print(f"\n{'='*80}")
    print("EXEMPLE: Backtest Causal Correct")
    print('='*80)

    print("\n📝 Code de backtest:")
    print("""
# Pour chaque sample i:
for i in range(len(Y_test)):
    # 1. Clé primaire
    timestamp = Y_test[i, 0]
    asset = Y_test[i, 1]

    # 2. Label
    direction = Y_test[i, 2]

    # 3. Prix bruts (même clé primaire)
    entry_price = OHLCV_test[i, 2]  # Open

    # 4. Trading causal
    if direction == 1:  # UP
        position = 'LONG'
        # Sortie au prochain signal opposé...
        exit_price = OHLCV_test[exit_idx, 2]
        pnl = (exit_price - entry_price) / entry_price  # ✅ Compound returns

    # 5. Statistiques
    print(f"{timestamp} - {asset} - {position} - PnL: {pnl:.2%}")
    """)

    print(f"\n{'='*80}")
    print("✅ Démonstration terminée")
    print("="*80)
    print("\n📌 Points clés:")
    print("  1. Clé primaire (timestamp, asset_id) commune à toutes les matrices")
    print("  2. Même index i = même sample partout")
    print("  3. Navigation simple et garantie sans désalignement")
    print("  4. Prix OHLCV directement disponibles pour backtest causal")
    print("  5. Pour ML: Supprimer colonnes 0 et 1 (timestamp, asset_id)")


def main():
    parser = argparse.ArgumentParser(
        description="Démo de navigation entre matrices avec métadonnées intégrées"
    )
    parser.add_argument(
        '--data', type=str, required=True,
        help='Chemin vers le fichier .npz'
    )
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"❌ Fichier introuvable: {args.data}")
        return

    demo_navigation(args.data)


if __name__ == '__main__':
    main()
