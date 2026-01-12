#!/usr/bin/env python3
"""Script de diagnostic pour vérifier le dataset Direction-Only."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np

def diagnose_dataset(npz_path: str):
    """Diagnostique le dataset et affiche les statistiques."""

    print("="*80)
    print("DIAGNOSTIC DU DATASET")
    print("="*80)

    # Charger le fichier brut (sans extraction)
    data = np.load(npz_path, allow_pickle=True)

    print(f"\n📂 Fichier: {npz_path}")
    print(f"\n🔑 Clés disponibles: {list(data.keys())}")

    # Analyser X_train
    X_train = data['X_train']
    Y_train = data['Y_train']

    print(f"\n📊 Shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  Y_train: {Y_train.shape}")

    # Analyser le contenu de X (première séquence, premier timestep)
    print(f"\n🔍 Contenu de X (première séquence, timestep 0):")
    print(f"  Feature 0: {X_train[0, 0, 0]:.6f}")
    print(f"  Feature 1: {X_train[0, 0, 1]:.6f}")
    print(f"  Feature 2: {X_train[0, 0, 2]:.6f}")

    # Vérifier si features 0 et 1 sont timestamp et asset_id
    print(f"\n🔍 Analyse des features:")
    print(f"  Feature 0 min/max: {X_train[:, 0, 0].min():.2f} / {X_train[:, 0, 0].max():.2f}")
    print(f"  Feature 1 min/max: {X_train[:, 0, 1].min():.2f} / {X_train[:, 0, 1].max():.2f}")
    print(f"  Feature 2 min/max: {X_train[:, 0, 2].min():.6f} / {X_train[:, 0, 2].max():.6f}")

    # Si feature 1 ressemble à asset_id
    if np.all(X_train[:, 0, 1] >= 0) and np.all(X_train[:, 0, 1] <= 10):
        print(f"  ⚠️  Feature 1 ressemble à asset_id (0-4)")

    # Si feature 2 ressemble à un return
    if np.abs(X_train[:, 0, 2]).max() < 1.0:
        print(f"  ✅ Feature 2 ressemble à un return (-1 à +1)")

    # Asset IDs uniques dans X
    asset_ids_x = np.unique(X_train[:, 0, 1])
    print(f"\n🎯 Asset IDs uniques dans X[:, 0, 1]: {asset_ids_x}")
    print(f"  Nombre d'assets: {len(asset_ids_x)}")

    # Si Y a 3 colonnes, vérifier asset_ids dans Y
    if Y_train.ndim == 2 and Y_train.shape[1] == 3:
        asset_ids_y = np.unique(Y_train[:, 1])
        print(f"\n🎯 Asset IDs uniques dans Y[:, 1]: {asset_ids_y}")
        print(f"  Nombre d'assets: {len(asset_ids_y)}")

        # Vérifier si X et Y ont les mêmes asset_ids
        if np.array_equal(asset_ids_x, asset_ids_y):
            print("  ✅ Asset IDs cohérents entre X et Y")
        else:
            print("  ⚠️ INCOHÉRENCE entre X et Y!")

    # Distribution des labels (colonne 2 si Y a 3 colonnes)
    if Y_train.ndim == 2 and Y_train.shape[1] == 3:
        labels = Y_train[:, 2]
    else:
        labels = Y_train.flatten()

    print(f"\n📊 Distribution des labels:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        pct = count / len(labels) * 100
        print(f"  Label {label}: {count:,} ({pct:.1f}%)")

    # Vérifier les valeurs invalides
    print(f"\n🔍 Vérifications:")
    print(f"  NaN dans X: {np.isnan(X_train).any()}")
    print(f"  NaN dans Y: {np.isnan(Y_train).any()}")
    print(f"  Inf dans X: {np.isinf(X_train).any()}")
    print(f"  Inf dans Y: {np.isinf(Y_train).any()}")

    # Compter les séquences par asset
    print(f"\n📊 Séquences par asset (depuis X):")
    for asset_id in sorted(asset_ids_x):
        count = np.sum(X_train[:, 0, 1] == asset_id)
        pct = count / len(X_train) * 100
        print(f"  Asset ID {int(asset_id)}: {count:,} séquences ({pct:.1f}%)")

    # Metadata
    if 'metadata' in data:
        metadata_raw = data['metadata']
        print(f"\n📋 Metadata:")
        print(f"  Type: {type(metadata_raw)}")

        # Essayer de charger selon le type
        try:
            if isinstance(metadata_raw, np.ndarray):
                # Si c'est un array numpy avec item(), l'extraire
                metadata = metadata_raw.item()
            else:
                metadata = metadata_raw

            # Vérifier si c'est un dict
            if isinstance(metadata, dict):
                if 'assets' in metadata:
                    print(f"  Assets: {metadata['assets']}")
                if 'features' in metadata:
                    print(f"  Features: {metadata['features']}")
                if 'labels' in metadata:
                    print(f"  Labels: {metadata['labels']}")
            else:
                print(f"  Metadata n'est pas un dict: {type(metadata)}")
                print(f"  Contenu: {str(metadata)[:200]}...")
        except Exception as e:
            print(f"  ⚠️  Erreur lors du chargement metadata: {e}")

    print("\n" + "="*80)
    print("FIN DU DIAGNOSTIC")
    print("="*80)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Chemin du dataset .npz')
    args = parser.parse_args()

    diagnose_dataset(args.data)
