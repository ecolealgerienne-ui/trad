#!/usr/bin/env python3
"""Script pour vérifier la normalisation des features dans le dataset."""

import numpy as np
import sys
from pathlib import Path

def check_normalization(npz_path: str):
    """Vérifie si les features sont normalisées."""

    print("="*80)
    print("VÉRIFICATION NORMALISATION DES FEATURES")
    print("="*80)

    # Charger le dataset
    data = np.load(npz_path)
    X_train = data['X_train']

    print(f"\n📂 Fichier: {npz_path}")
    print(f"📊 Shape X_train: {X_train.shape}")

    # Statistiques globales
    print(f"\n📊 Statistiques globales:")
    print(f"  Min:  {X_train.min():.6f}")
    print(f"  Max:  {X_train.max():.6f}")
    print(f"  Mean: {X_train.mean():.6f}")
    print(f"  Std:  {X_train.std():.6f}")

    # Par feature
    print(f"\n📊 Par feature:")
    n_features = X_train.shape[2]

    for i in range(n_features):
        feat_data = X_train[:, :, i]
        min_val = feat_data.min()
        max_val = feat_data.max()
        mean_val = feat_data.mean()
        std_val = feat_data.std()

        # Déterminer si normalisé
        is_normalized = abs(mean_val) < 0.1 and 0.8 < std_val < 1.2
        status = "✅ OK" if is_normalized else "❌ NON NORMALISÉ"

        print(f"  Feature {i}: min={min_val:8.4f}, max={max_val:8.4f}, "
              f"mean={mean_val:7.4f}, std={std_val:7.4f} {status}")

    # Vérifier NaN/Inf
    print(f"\n🔍 Vérifications:")
    print(f"  NaN: {np.isnan(X_train).any()}")
    print(f"  Inf: {np.isinf(X_train).any()}")

    # Recommandations
    print(f"\n💡 Recommandations:")
    if X_train.max() > 10 or X_train.min() < -10:
        print("  ⚠️  CRITIQUE: Features non normalisées (valeurs > 10)")
        print("  → Appliquer StandardScaler ou normalisation [-1, 1]")
    elif abs(X_train.mean()) > 0.5:
        print("  ⚠️  Features pas centrées (mean éloigné de 0)")
        print("  → Centrer autour de 0")
    else:
        print("  ✅ Features semblent correctement normalisées")

    print("\n" + "="*80)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Chemin du dataset .npz')
    args = parser.parse_args()

    check_normalization(args.data)
