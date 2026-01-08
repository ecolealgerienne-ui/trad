#!/usr/bin/env python3
"""
Vérification complète du pipeline train/eval en 4 passes.

Vérifie que:
- X contient seulement les features utiles (c_ret), pas timestamp/asset_id
- Y contient seulement les labels, pas timestamp/asset_id
- Les shapes sont correctes à chaque étape
- Les valeurs sont dans les plages attendues
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from prepare_data import load_prepared_data, filter_by_assets

def verify_pipeline(npz_path: str):
    """Vérifie le pipeline complet en 4 passes."""

    print("="*80)
    print("VÉRIFICATION PIPELINE EN 4 PASSES")
    print("="*80)
    print(f"\nDataset: {npz_path}")

    # =========================================================================
    # PASSE 1: CHARGEMENT BRUT (avant extraction)
    # =========================================================================
    print("\n" + "="*80)
    print("PASSE 1: CHARGEMENT BRUT (avant extraction Direction-Only)")
    print("="*80)

    data_raw = np.load(npz_path, allow_pickle=True)
    X_raw = data_raw['X_train']
    Y_raw = data_raw['Y_train']

    print(f"\n📊 Shapes BRUTES:")
    print(f"  X_train: {X_raw.shape}")
    print(f"  Y_train: {Y_raw.shape}")

    print(f"\n🔍 Contenu X BRUT (première séquence, timestep 0):")
    print(f"  Feature 0 (timestamp): {X_raw[0, 0, 0]:.0f}")
    print(f"  Feature 1 (asset_id):  {X_raw[0, 0, 1]:.0f}")
    print(f"  Feature 2 (c_ret):     {X_raw[0, 0, 2]:.6f}")

    print(f"\n🔍 Contenu Y BRUT (première séquence):")
    print(f"  Colonne 0 (timestamp): {Y_raw[0, 0]:.0f}")
    print(f"  Colonne 1 (asset_id):  {Y_raw[0, 1]:.0f}")
    print(f"  Colonne 2 (label):     {Y_raw[0, 2]:.0f}")

    # Vérifications
    assert X_raw.shape[2] == 3, f"❌ X devrait avoir 3 colonnes, a {X_raw.shape[2]}"
    assert Y_raw.shape[1] == 3, f"❌ Y devrait avoir 3 colonnes, a {Y_raw.shape[1]}"
    print("\n✅ PASSE 1 OK: Format brut Direction-Only détecté")

    # =========================================================================
    # PASSE 2: EXTRACTION DIRECTION-ONLY (via load_prepared_data)
    # =========================================================================
    print("\n" + "="*80)
    print("PASSE 2: EXTRACTION DIRECTION-ONLY (load_prepared_data)")
    print("="*80)

    prepared = load_prepared_data(npz_path)
    X_train, Y_train, T_train = prepared['train']
    metadata = prepared['metadata']

    print(f"\n📊 Shapes APRÈS EXTRACTION:")
    print(f"  X_train: {X_train.shape}")
    print(f"  Y_train: {Y_train.shape}")
    print(f"  T_train: {T_train.shape}")

    print(f"\n🔍 Contenu X EXTRAIT (première séquence, timestep 0):")
    print(f"  Feature 0 (devrait être c_ret): {X_train[0, 0, 0]:.6f}")
    if X_train.shape[2] > 1:
        print(f"  ⚠️  ATTENTION: X a {X_train.shape[2]} features au lieu de 1!")
        for i in range(1, X_train.shape[2]):
            print(f"  Feature {i}: {X_train[0, 0, i]:.6f}")

    print(f"\n🔍 Contenu Y EXTRAIT (première séquence):")
    print(f"  Label (devrait être 0 ou 1): {Y_train[0, 0]:.0f}")
    if Y_train.shape[1] > 1:
        print(f"  ⚠️  ATTENTION: Y a {Y_train.shape[1]} colonnes au lieu de 1!")
        for i in range(1, Y_train.shape[1]):
            print(f"  Colonne {i}: {Y_train[0, i]:.6f}")

    # Vérifications
    assert X_train.shape[2] == 1, f"❌ X devrait avoir 1 feature (c_ret), a {X_train.shape[2]}"
    assert Y_train.shape[1] == 1, f"❌ Y devrait avoir 1 colonne (label), a {Y_train.shape[1]}"

    # Vérifier plages de valeurs
    print(f"\n🔍 Plages de valeurs:")
    print(f"  X (c_ret) min/max: {X_train.min():.6f} / {X_train.max():.6f}")
    print(f"  Y (label) unique: {np.unique(Y_train)}")

    assert X_train.min() >= -1.0 and X_train.max() <= 1.0, "❌ X hors plage [-1, 1]"
    assert set(np.unique(Y_train)) == {0.0, 1.0} or set(np.unique(Y_train)) == {0, 1}, "❌ Y devrait contenir seulement 0 et 1"

    print("\n✅ PASSE 2 OK: Extraction correcte (X=c_ret, Y=label)")

    # =========================================================================
    # PASSE 3: FILTRAGE PAR ASSETS (filter_by_assets)
    # =========================================================================
    print("\n" + "="*80)
    print("PASSE 3: FILTRAGE PAR ASSETS")
    print("="*80)

    # Tester le filtrage sur BTC uniquement
    OHLCV_train = data_raw['OHLCV_train']
    X_filtered, Y_filtered, T_filtered, OHLCV_filtered = filter_by_assets(
        X_train, Y_train, T_train, OHLCV_train,
        ['BTC'], metadata
    )

    print(f"\n📊 Shapes APRÈS FILTRAGE (BTC seul):")
    print(f"  X_filtered: {X_filtered.shape}")
    print(f"  Y_filtered: {Y_filtered.shape}")
    print(f"  T_filtered: {T_filtered.shape}")

    # Vérifier que le filtrage a gardé les bonnes shapes
    assert X_filtered.shape[1] == 25, f"❌ Sequence length devrait être 25, est {X_filtered.shape[1]}"
    assert X_filtered.shape[2] == 1, f"❌ X devrait avoir 1 feature après filtrage, a {X_filtered.shape[2]}"
    assert Y_filtered.shape[1] == 1, f"❌ Y devrait avoir 1 colonne après filtrage, a {Y_filtered.shape[1]}"

    # Vérifier que les valeurs sont toujours bonnes
    print(f"\n🔍 Valeurs après filtrage:")
    print(f"  X (c_ret) min/max: {X_filtered.min():.6f} / {X_filtered.max():.6f}")
    print(f"  Y (label) unique: {np.unique(Y_filtered)}")

    assert X_filtered.min() >= -1.0 and X_filtered.max() <= 1.0, "❌ X hors plage après filtrage"
    assert set(np.unique(Y_filtered)) <= {0.0, 1.0, 0, 1}, "❌ Y contient d'autres valeurs que 0 et 1"

    # Vérifier que OHLCV a bien asset_id=0 (BTC)
    print(f"\n🔍 Vérification OHLCV (asset_id de BTC):")
    ohlcv_asset_ids = np.unique(OHLCV_filtered[:, 1])
    print(f"  Asset IDs dans OHLCV filtré: {ohlcv_asset_ids}")
    assert ohlcv_asset_ids.tolist() == [0.0], f"❌ Devrait contenir seulement asset_id 0 (BTC), a {ohlcv_asset_ids}"

    print("\n✅ PASSE 3 OK: Filtrage préserve les shapes et valeurs correctes")

    # =========================================================================
    # PASSE 4: VÉRIFICATION FINALE (résumé)
    # =========================================================================
    print("\n" + "="*80)
    print("PASSE 4: VÉRIFICATION FINALE (RÉSUMÉ)")
    print("="*80)

    print(f"\n🎯 PIPELINE COMPLET VÉRIFIÉ:")
    print(f"  1. ✅ Format brut détecté: X(n,25,3), Y(n,3)")
    print(f"  2. ✅ Extraction correcte: X(n,25,1), Y(n,1)")
    print(f"  3. ✅ X contient seulement c_ret (plage: [{X_train.min():.4f}, {X_train.max():.4f}])")
    print(f"  4. ✅ Y contient seulement labels 0/1")
    print(f"  5. ✅ Filtrage préserve les shapes correctes")
    print(f"  6. ✅ Filtrage utilise OHLCV[:, 1] (asset_id)")

    # Distribution des labels
    label_0 = (Y_train == 0).sum()
    label_1 = (Y_train == 1).sum()
    total = len(Y_train)
    print(f"\n📊 Distribution des labels (train):")
    print(f"  Label 0: {label_0:,} ({label_0/total*100:.1f}%)")
    print(f"  Label 1: {label_1:,} ({label_1/total*100:.1f}%)")

    balance = min(label_0, label_1) / max(label_0, label_1)
    assert balance >= 0.45, f"❌ Labels déséquilibrés: {balance:.1%}"
    print(f"  ✅ Balance: {balance:.1%} (bien équilibré)")

    print("\n" + "="*80)
    print("🎉 TOUTES LES VÉRIFICATIONS PASSÉES!")
    print("="*80)
    print("\n✅ Le pipeline train/eval utilise les BONNES données:")
    print("   - X = c_ret uniquement (pas de timestamp/asset_id)")
    print("   - Y = labels 0/1 uniquement (pas de timestamp/asset_id)")
    print("   - Shapes correctes à chaque étape")
    print("   - Valeurs dans les plages attendues")
    print("   - Filtrage fonctionne correctement")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Chemin du dataset .npz')
    args = parser.parse_args()

    verify_pipeline(args.data)
