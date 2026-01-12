#!/usr/bin/env python3
"""
Vérification minimale du dataset regime.npz
Nécessite seulement numpy (installation: pip install numpy)
"""

import sys
import os

try:
    import numpy as np
except ImportError:
    print("❌ ERREUR: numpy n'est pas installé")
    print("\n📦 Installation requise:")
    print("   pip install numpy")
    print("\n   OU installer toutes les dépendances:")
    print("   pip install -r requirements.txt")
    sys.exit(1)

def minimal_verify(npz_path):
    """Vérifications minimales critiques"""

    print("=" * 80)
    print("🔍 VÉRIFICATION MINIMALE DU DATASET")
    print("=" * 80)
    print(f"\n📂 Fichier: {npz_path}")

    # Vérifier existence
    if not os.path.exists(npz_path):
        print(f"\n❌ ERREUR: Fichier introuvable!")
        return False

    file_size_mb = os.path.getsize(npz_path) / (1024 * 1024)
    print(f"📏 Taille: {file_size_mb:.1f} MB")

    # Charger NPZ
    print("\n⏳ Chargement du fichier NPZ...")
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"\n❌ ERREUR lors du chargement: {e}")
        return False

    print("✅ Fichier NPZ chargé avec succès")

    # 1. VÉRIFIER LES CLÉS
    print("\n" + "=" * 80)
    print("1️⃣  VÉRIFICATION DES CLÉS")
    print("=" * 80)

    keys = list(data.keys())
    print(f"\n🔑 Clés disponibles ({len(keys)} total):")
    for key in sorted(keys):
        print(f"   - {key}")

    expected_keys = {
        'X_train', 'X_val', 'X_test',
        'Y_train', 'Y_val', 'Y_test',
        'OHLCV_train', 'OHLCV_val', 'OHLCV_test',
        'metadata'
    }

    missing_keys = expected_keys - set(keys)
    if missing_keys:
        print(f"\n❌ CLÉS MANQUANTES: {missing_keys}")
        return False

    print("\n✅ Toutes les clés requises présentes")

    # 2. VÉRIFIER LES SHAPES
    print("\n" + "=" * 80)
    print("2️⃣  VÉRIFICATION DES SHAPES")
    print("=" * 80)

    shapes_ok = True

    for split in ['train', 'val', 'test']:
        X = data[f'X_{split}']
        Y = data[f'Y_{split}']
        OHLCV = data[f'OHLCV_{split}']

        print(f"\n📊 Split {split.upper()}:")
        print(f"   X_{split}:     {X.shape}")
        print(f"   Y_{split}:     {Y.shape}")
        print(f"   OHLCV_{split}: {OHLCV.shape}")

        # Vérifier cohérence tailles
        if X.shape[0] != Y.shape[0] or X.shape[0] != OHLCV.shape[0]:
            print(f"   ❌ ERREUR: Tailles incohérentes!")
            print(f"      X samples: {X.shape[0]}")
            print(f"      Y samples: {Y.shape[0]}")
            print(f"      OHLCV samples: {OHLCV.shape[0]}")
            shapes_ok = False
            continue

        # Vérifier dimensions
        if X.ndim != 3:
            print(f"   ❌ ERREUR: X devrait être 3D, trouvé {X.ndim}D")
            shapes_ok = False
        elif X.shape[1] != 25:
            print(f"   ⚠️  WARNING: X séquence length = {X.shape[1]} (attendu 25)")
        elif X.shape[2] < 20 or X.shape[2] > 25:
            print(f"   ⚠️  WARNING: X features = {X.shape[2]} (attendu ~22)")

        if Y.ndim != 2:
            print(f"   ❌ ERREUR: Y devrait être 2D, trouvé {Y.ndim}D")
            shapes_ok = False
        elif Y.shape[1] == 8:
            print(f"   ⚠️  WARNING: Y a {Y.shape[1]} colonnes - enrichissement NOT done (attendu 13)")
        elif Y.shape[1] != 13:
            print(f"   ⚠️  WARNING: Y a {Y.shape[1]} colonnes (attendu 13 après enrichissement)")

        if OHLCV.shape[1] != 7:
            print(f"   ❌ ERREUR: OHLCV devrait avoir 7 colonnes, trouvé {OHLCV.shape[1]}")
            shapes_ok = False

        if shapes_ok:
            print(f"   ✅ Shapes cohérentes")

    if not shapes_ok:
        print("\n❌ ERREURS DE SHAPE DÉTECTÉES")
        return False

    print("\n✅ Toutes les shapes cohérentes")

    # 3. VÉRIFIER Y COLONNES (CRITIQUE)
    print("\n" + "=" * 80)
    print("3️⃣  VÉRIFICATION COLONNES Y (CRITIQUE)")
    print("=" * 80)

    Y_train = data['Y_train']
    n_cols = Y_train.shape[1]

    print(f"\n📋 Y_train a {n_cols} colonnes")

    if n_cols == 8:
        print("\n⚠️  ATTENTION: Dataset PAS ENRICHI")
        print("   Colonnes Y (8):")
        print("     [0] timestamp")
        print("     [1] asset_id")
        print("     [2] regime (TARGET)")
        print("     [3] trend_strength")
        print("     [4] volatility")
        print("     [5] macd_direction")
        print("     [6] rsi_direction")
        print("     [7] cci_direction")
        print("\n   ⚠️  Il manque les 5 colonnes d'enrichissement:")
        print("     [8] regime_prob_0")
        print("     [9] regime_prob_1")
        print("     [10] regime_prob_2")
        print("     [11] regime_prob_3")
        print("     [12] regime_pred")
        print("\n   🔧 Action requise:")
        print("      python src/train_regime_classifier.py \\")
        print("          --data data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz \\")
        print("          --epochs 100")
        enriched = False
    elif n_cols == 13:
        print("\n✅ Dataset ENRICHI (13 colonnes)")
        print("   Colonnes Y complètes:")
        print("     [0-7]  Base (timestamp, asset_id, regime, TS, VC, dirs)")
        print("     [8-12] Enrichissement (regime_prob_0-3, regime_pred)")
        enriched = True
    else:
        print(f"\n❌ ERREUR: Nombre de colonnes Y inattendu: {n_cols}")
        print("   Attendu: 8 (base) ou 13 (enrichi)")
        return False

    # 4. VÉRIFIER NaN/Inf
    print("\n" + "=" * 80)
    print("4️⃣  VÉRIFICATION NaN/Inf")
    print("=" * 80)

    nan_found = False

    for split in ['train', 'val', 'test']:
        X = data[f'X_{split}']
        Y = data[f'Y_{split}']
        OHLCV = data[f'OHLCV_{split}']

        x_nan = np.isnan(X).any()
        x_inf = np.isinf(X).any()
        y_nan = np.isnan(Y).any()
        y_inf = np.isinf(Y).any()
        ohlcv_nan = np.isnan(OHLCV).any()
        ohlcv_inf = np.isinf(OHLCV).any()

        print(f"\n📊 Split {split.upper()}:")
        print(f"   X_{split}:     NaN={x_nan}, Inf={x_inf}")
        print(f"   Y_{split}:     NaN={y_nan}, Inf={y_inf}")
        print(f"   OHLCV_{split}: NaN={ohlcv_nan}, Inf={ohlcv_inf}")

        if any([x_nan, x_inf, y_nan, y_inf, ohlcv_nan, ohlcv_inf]):
            print(f"   ❌ PROBLÈME DÉTECTÉ!")
            nan_found = True
        else:
            print(f"   ✅ Aucun NaN/Inf")

    if nan_found:
        print("\n❌ NaN/Inf DÉTECTÉS - données corrompues")
        return False

    print("\n✅ Aucun NaN/Inf détecté")

    # 5. VÉRIFIER RANGES DES LABELS
    print("\n" + "=" * 80)
    print("5️⃣  VÉRIFICATION RANGES DES LABELS")
    print("=" * 80)

    Y_train = data['Y_train']

    # Asset IDs (colonne 1)
    asset_ids = Y_train[:, 1]
    unique_assets = np.unique(asset_ids)
    print(f"\n🎯 Asset IDs uniques: {unique_assets}")
    if not np.all((unique_assets >= 0) & (unique_assets <= 4)):
        print(f"   ❌ ERREUR: Asset IDs hors range 0-4")
        return False
    print(f"   ✅ Asset IDs valides (0-4)")

    # Régimes (colonne 2)
    regimes = Y_train[:, 2]
    unique_regimes = np.unique(regimes)
    print(f"\n🎯 Régimes uniques: {unique_regimes}")
    if not np.all((unique_regimes >= 0) & (unique_regimes <= 3)):
        print(f"   ❌ ERREUR: Régimes hors range 0-3")
        return False
    print(f"   ✅ Régimes valides (0-3)")

    # Distribution des régimes
    print(f"\n📊 Distribution des régimes (train):")
    for regime in [0, 1, 2, 3]:
        count = np.sum(regimes == regime)
        pct = count / len(regimes) * 100
        label = ["RANGE LOW VOL", "RANGE HIGH VOL", "TREND LOW VOL", "TREND HIGH VOL"][regime]
        print(f"   Régime {regime} ({label:15}): {count:7,} ({pct:5.1f}%)")
        if pct < 5 or pct > 50:
            print(f"      ⚠️  Distribution déséquilibrée!")

    # Trend strength et volatility (colonnes 3-4)
    ts = Y_train[:, 3]
    vc = Y_train[:, 4]
    print(f"\n📊 Trend Strength: min={ts.min():.3f}, max={ts.max():.3f}")
    print(f"📊 Volatility:     min={vc.min():.3f}, max={vc.max():.3f}")
    if ts.min() < 0 or ts.max() > 1 or vc.min() < 0 or vc.max() > 1:
        print(f"   ⚠️  WARNING: Valeurs hors range [0, 1]")
    else:
        print(f"   ✅ Valeurs dans range [0, 1]")

    # Directions (colonnes 5-7)
    dirs = Y_train[:, 5:8]
    if not np.all(np.isin(dirs, [0, 1])):
        print(f"\n❌ ERREUR: Directions non binaires (0/1)")
        return False
    print(f"\n✅ Directions MACD/RSI/CCI binaires (0/1)")

    # RÉSUMÉ FINAL
    print("\n" + "=" * 80)
    print("📋 RÉSUMÉ FINAL")
    print("=" * 80)

    print(f"\n✅ Fichier NPZ valide: {os.path.basename(npz_path)}")
    print(f"✅ Taille: {file_size_mb:.1f} MB")
    print(f"✅ Clés: {len(keys)} présentes")
    print(f"✅ Shapes cohérentes")
    print(f"✅ Aucun NaN/Inf")
    print(f"✅ Ranges des labels valides")

    if not enriched:
        print(f"\n⚠️  ATTENTION: Dataset PAS ENRICHI (Y a 8 colonnes)")
        print(f"   Exécuter train_regime_classifier.py pour enrichir")
    else:
        print(f"\n✅ Dataset ENRICHI (Y a 13 colonnes)")
        print(f"   Prêt pour entraînement des modèles direction")

    print("\n" + "=" * 80)

    return True

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Vérification minimale dataset regime.npz')
    parser.add_argument('--data', type=str,
                       default='data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz',
                       help='Chemin du fichier NPZ')

    args = parser.parse_args()

    success = minimal_verify(args.data)
    sys.exit(0 if success else 1)
