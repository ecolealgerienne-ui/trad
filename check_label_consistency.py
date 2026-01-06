#!/usr/bin/env python3
"""
Vérification de la Cohérence des Labels entre MACD/RSI/CCI

Objectif: Diagnostiquer pourquoi Y_test diffère entre les 3 datasets.

Hypothèses:
1. Nombre de samples différent (trim edges différent?)
2. Labels calculés différemment (bug dans prepare_data?)
3. Ordre des samples différent (shuffle accidentel?)
"""

import numpy as np
from pathlib import Path

# Chemins datasets
DATASET_PATHS = {
    'macd': 'data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_kalman.npz',
    'rsi': 'data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_dual_binary_kalman.npz',
    'cci': 'data/prepared/dataset_btc_eth_bnb_ada_ltc_cci_dual_binary_kalman.npz',
}

print("="*80)
print("🔍 VÉRIFICATION COHÉRENCE DES LABELS")
print("="*80)

# Charger les 3 datasets
datasets = {}
for name, path in DATASET_PATHS.items():
    if not Path(path).exists():
        print(f"❌ Dataset introuvable: {path}")
        exit(1)
    datasets[name] = np.load(path, allow_pickle=True)
    print(f"✅ {name.upper()}: {path}")

# Vérifier shapes
print("\n" + "="*80)
print("📊 SHAPES")
print("="*80)

for split in ['train', 'val', 'test']:
    print(f"\n{split.upper()}:")
    for name in ['macd', 'rsi', 'cci']:
        Y = datasets[name][f'Y_{split}']
        X = datasets[name][f'X_{split}']
        print(f"  {name.upper():4s}: X={X.shape}, Y={Y.shape}")

# Vérifier différences Y_test
print("\n" + "="*80)
print("🔬 DIFFÉRENCES Y_TEST (Direction - Colonne 0)")
print("="*80)

macd_y = datasets['macd']['Y_test'][:, 0]
rsi_y = datasets['rsi']['Y_test'][:, 0]
cci_y = datasets['cci']['Y_test'][:, 0]

n_samples = len(macd_y)

diff_macd_rsi = (macd_y != rsi_y).sum()
diff_macd_cci = (macd_y != cci_y).sum()
diff_rsi_cci = (rsi_y != cci_y).sum()

print(f"\n  Total samples: {n_samples:,}")
print(f"\n  MACD vs RSI:  {diff_macd_rsi:,} différences ({diff_macd_rsi/n_samples*100:.2f}%)")
print(f"  MACD vs CCI:  {diff_macd_cci:,} différences ({diff_macd_cci/n_samples*100:.2f}%)")
print(f"  RSI vs CCI:   {diff_rsi_cci:,} différences ({diff_rsi_cci/n_samples*100:.2f}%)")

# Vérifier Force
print("\n" + "="*80)
print("🔬 DIFFÉRENCES Y_TEST (Force - Colonne 1)")
print("="*80)

macd_force = datasets['macd']['Y_test'][:, 1]
rsi_force = datasets['rsi']['Y_test'][:, 1]
cci_force = datasets['cci']['Y_test'][:, 1]

diff_force_macd_rsi = (macd_force != rsi_force).sum()
diff_force_macd_cci = (macd_force != cci_force).sum()
diff_force_rsi_cci = (rsi_force != cci_force).sum()

print(f"\n  MACD vs RSI:  {diff_force_macd_rsi:,} différences ({diff_force_macd_rsi/n_samples*100:.2f}%)")
print(f"  MACD vs CCI:  {diff_force_macd_cci:,} différences ({diff_force_macd_cci/n_samples*100:.2f}%)")
print(f"  RSI vs CCI:   {diff_force_rsi_cci:,} différences ({diff_force_rsi_cci/n_samples*100:.2f}%)")

# Analyser POURQUOI Direction diffère
print("\n" + "="*80)
print("🔍 ANALYSE RACINE - Pourquoi Direction diffère?")
print("="*80)

# Hypothèse 1: Les 3 indicateurs sont calculés différemment
print("\nHypothèse 1: Indicateurs calculés sur inputs différents")
print("  MACD: Kalman(Close)")
print("  RSI:  Kalman(Close)")
print("  CCI:  Kalman(Typical Price = (H+L+C)/3)")
print("  → Normal que CCI diffère de MACD/RSI")

# Vérifier si MACD == RSI (devraient être identiques car même input Close)
if diff_macd_rsi == 0:
    print(f"\n✅ MACD == RSI (identiques, comme attendu)")
else:
    print(f"\n❌ MACD ≠ RSI ({diff_macd_rsi:,} différences)")
    print(f"   → BUG POTENTIEL dans prepare_data_purified_dual_binary.py")
    print(f"   → Les deux devraient utiliser Kalman(Close)")

# Comparer premiers samples pour debug
if diff_macd_rsi > 0:
    print("\n🔍 Premiers échantillons divergents (MACD vs RSI):")
    diff_idx = np.where(macd_y != rsi_y)[0][:5]
    for idx in diff_idx:
        print(f"  Sample {idx:6d}: MACD={macd_y[idx]}, RSI={rsi_y[idx]}")

# Distribution Direction
print("\n" + "="*80)
print("📊 DISTRIBUTION DIRECTION (Y_test)")
print("="*80)

for name in ['macd', 'rsi', 'cci']:
    y = datasets[name]['Y_test'][:, 0]
    pct_up = (y == 1).sum() / len(y) * 100
    print(f"  {name.upper():4s}: UP={pct_up:.2f}%, DOWN={100-pct_up:.2f}%")

# Distribution Force
print("\n" + "="*80)
print("📊 DISTRIBUTION FORCE (Y_test)")
print("="*80)

for name in ['macd', 'rsi', 'cci']:
    force = datasets[name]['Y_test'][:, 1]
    pct_strong = (force == 1).sum() / len(force) * 100
    print(f"  {name.upper():4s}: STRONG={pct_strong:.2f}%, WEAK={100-pct_strong:.2f}%")

# Conclusion
print("\n" + "="*80)
print("🎯 CONCLUSION")
print("="*80)

if diff_macd_rsi == 0:
    print("\n✅ MACD et RSI ont des labels IDENTIQUES (attendu)")
    print("✅ CCI diffère car calculé sur Typical Price au lieu de Close (normal)")
    print("\n💡 Pour le Stacking:")
    print("   - Utiliser Y_test de MACD comme référence unique")
    print("   - Recalculer les métriques avec cette référence")
elif diff_macd_rsi < n_samples * 0.01:  # < 1% différences
    print(f"\n⚠️  MACD et RSI ont {diff_macd_rsi:,} différences ({diff_macd_rsi/n_samples*100:.4f}%)")
    print("   → Probablement lié à trim edges ou arrondis")
    print("   → Impact négligeable, peut utiliser MACD comme référence")
else:
    print(f"\n❌ MACD et RSI ont {diff_macd_rsi:,} différences ({diff_macd_rsi/n_samples*100:.2f}%)")
    print("   → BUG dans prepare_data_purified_dual_binary.py!")
    print("   → Les deux devraient calculer Kalman(Close) identiquement")
    print("\n🔧 Actions à prendre:")
    print("   1. Vérifier prepare_data_purified_dual_binary.py")
    print("   2. S'assurer que MACD et RSI utilisent le même calcul Kalman")
    print("   3. Régénérer les datasets")

print("\n" + "="*80)
