#!/usr/bin/env python3
"""
Test de chargement du format Direction-Only.

Vérifie que load_prepared_data() charge correctement les datasets Direction-Only.
"""

import sys
from pathlib import Path

# Ajouter src/ au path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from prepare_data import load_prepared_data
import numpy as np


def test_load_direction_only(npz_path: str):
    """
    Teste le chargement d'un dataset Direction-Only.

    Args:
        npz_path: Chemin vers le .npz Direction-Only
    """
    print(f"\n{'='*80}")
    print(f"TEST CHARGEMENT DIRECTION-ONLY")
    print('='*80)
    print(f"Fichier: {npz_path}")

    # Charger les données
    prepared = load_prepared_data(npz_path)

    # Unpacking
    if len(prepared['train']) == 3:
        X_train, Y_train, T_train = prepared['train']
        X_val, Y_val, T_val = prepared['val']
        X_test, Y_test, T_test = prepared['test']
        has_transitions = True
    else:
        X_train, Y_train = prepared['train']
        X_val, Y_val = prepared['val']
        X_test, Y_test = prepared['test']
        T_train = T_val = T_test = None
        has_transitions = False

    metadata = prepared['metadata']

    # Vérifications
    print(f"\n📊 Shapes chargées:")
    print(f"  Train: X={X_train.shape}, Y={Y_train.shape}, T={T_train.shape if has_transitions else 'N/A'}")
    print(f"  Val:   X={X_val.shape}, Y={Y_val.shape}, T={T_val.shape if has_transitions else 'N/A'}")
    print(f"  Test:  X={X_test.shape}, Y={Y_test.shape}, T={T_test.shape if has_transitions else 'N/A'}")

    # Vérifier que Y a bien shape (n, 1) pour l'entraînement
    print(f"\n✅ VÉRIFICATIONS:")

    # 1. Y doit avoir 1 colonne (direction seule)
    assert Y_train.shape[1] == 1, f"❌ Y_train doit avoir 1 colonne, obtenu {Y_train.shape[1]}"
    print(f"  ✅ Y_train shape correct: {Y_train.shape}")

    assert Y_val.shape[1] == 1, f"❌ Y_val doit avoir 1 colonne, obtenu {Y_val.shape[1]}"
    print(f"  ✅ Y_val shape correct: {Y_val.shape}")

    assert Y_test.shape[1] == 1, f"❌ Y_test doit avoir 1 colonne, obtenu {Y_test.shape[1]}"
    print(f"  ✅ Y_test shape correct: {Y_test.shape}")

    # 2. Si transitions présentes, T doit aussi avoir 1 colonne
    if has_transitions:
        assert T_train.shape[1] == 1, f"❌ T_train doit avoir 1 colonne, obtenu {T_train.shape[1]}"
        print(f"  ✅ T_train shape correct: {T_train.shape}")

        assert T_val.shape[1] == 1, f"❌ T_val doit avoir 1 colonne, obtenu {T_val.shape[1]}"
        print(f"  ✅ T_val shape correct: {T_val.shape}")

        assert T_test.shape[1] == 1, f"❌ T_test doit avoir 1 colonne, obtenu {T_test.shape[1]}"
        print(f"  ✅ T_test shape correct: {T_test.shape}")

    # 3. X doit avoir shape (n, seq_length, n_features)
    seq_length = X_train.shape[1]
    n_features = X_train.shape[2]
    print(f"  ✅ X_train: seq_length={seq_length}, n_features={n_features}")

    # 4. Vérifier cohérence tailles
    assert len(X_train) == len(Y_train), "❌ X_train et Y_train de tailles différentes"
    assert len(X_val) == len(Y_val), "❌ X_val et Y_val de tailles différentes"
    assert len(X_test) == len(Y_test), "❌ X_test et Y_test de tailles différentes"
    print(f"  ✅ Cohérence tailles X/Y/T")

    # 5. Vérifier valeurs Y (doivent être 0 ou 1)
    assert np.all((Y_train == 0) | (Y_train == 1)), "❌ Y_train contient des valeurs != 0/1"
    assert np.all((Y_val == 0) | (Y_val == 1)), "❌ Y_val contient des valeurs != 0/1"
    assert np.all((Y_test == 0) | (Y_test == 1)), "❌ Y_test contient des valeurs != 0/1"
    print(f"  ✅ Y contient uniquement 0/1")

    # 6. Si transitions, vérifier valeurs T (doivent être 0 ou 1)
    if has_transitions:
        assert np.all((T_train == 0) | (T_train == 1)), "❌ T_train contient des valeurs != 0/1"
        assert np.all((T_val == 0) | (T_val == 1)), "❌ T_val contient des valeurs != 0/1"
        assert np.all((T_test == 0) | (T_test == 1)), "❌ T_test contient des valeurs != 0/1"
        print(f"  ✅ T contient uniquement 0/1")

    # 7. Vérifier distributions
    dir_train_pct = Y_train.mean() * 100
    dir_val_pct = Y_val.mean() * 100
    dir_test_pct = Y_test.mean() * 100

    print(f"\n📊 Distributions Direction (% UP):")
    print(f"  Train: {dir_train_pct:.1f}%")
    print(f"  Val:   {dir_val_pct:.1f}%")
    print(f"  Test:  {dir_test_pct:.1f}%")

    if has_transitions:
        trans_train_pct = T_train.mean() * 100
        trans_val_pct = T_val.mean() * 100
        trans_test_pct = T_test.mean() * 100

        print(f"\n📊 Distributions Transitions (% retournements):")
        print(f"  Train: {trans_train_pct:.1f}%")
        print(f"  Val:   {trans_val_pct:.1f}%")
        print(f"  Test:  {trans_test_pct:.1f}%")

    # 8. Afficher métadonnées
    print(f"\n📋 Métadonnées:")
    for key, value in sorted(metadata.items()):
        if isinstance(value, (list, dict)):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")

    print(f"\n{'='*80}")
    print("✅ TOUS LES TESTS PASSÉS - Dataset Direction-Only valide!")
    print('='*80)

    return True


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Tester le chargement Direction-Only')
    parser.add_argument('--data', type=str, required=True,
                        help='Chemin vers le .npz Direction-Only')

    args = parser.parse_args()

    test_load_direction_only(args.data)
