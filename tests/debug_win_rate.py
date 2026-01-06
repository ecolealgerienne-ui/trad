#!/usr/bin/env python3
"""
Debug script to verify Win Rate calculation logic.
"""

# Simuler un trade
fees_percent = 0.1  # 0.1%
fees_decimal = fees_percent / 100.0  # 0.001

# Cas 1: Trade gagne 0.5% brut
print("=" * 60)
print("CAS 1: Trade gagne 0.5% brut")
print("=" * 60)

# AVANT (bug - unités mixtes)
current_pnl_old = 0.5  # En pourcentage (0.5%)
trade_fees_old = 2 * fees_decimal  # 0.002 (en décimal)
pnl_after_fees_old = current_pnl_old - trade_fees_old  # 0.5 - 0.002 = 0.498
is_win_old = pnl_after_fees_old > 0
print(f"AVANT (bug):")
print(f"  current_pnl = {current_pnl_old} (en %)")
print(f"  trade_fees = {trade_fees_old} (en décimal)")
print(f"  pnl_after_fees = {pnl_after_fees_old}")
print(f"  Win? {is_win_old} ✅ (FAUX positif - unités différentes)")

# APRÈS (correct - tout en décimal)
current_pnl_new = 0.005  # En décimal (0.5%)
trade_fees_new = 2 * fees_decimal  # 0.002 (en décimal)
pnl_after_fees_new = current_pnl_new - trade_fees_new  # 0.005 - 0.002 = 0.003
is_win_new = pnl_after_fees_new > 0
print(f"\nAPRÈS (correct):")
print(f"  current_pnl = {current_pnl_new} (en décimal = {current_pnl_new*100}%)")
print(f"  trade_fees = {trade_fees_new} (en décimal = {trade_fees_new*100}%)")
print(f"  pnl_after_fees = {pnl_after_fees_new} (en décimal = {pnl_after_fees_new*100}%)")
print(f"  Win? {is_win_new} ✅ (correct)")

# Cas 2: Trade gagne 0.1% brut (< fees)
print("\n" + "=" * 60)
print("CAS 2: Trade gagne 0.1% brut (gain < fees)")
print("=" * 60)

# AVANT (bug)
current_pnl_old = 0.1  # En pourcentage (0.1%)
pnl_after_fees_old = current_pnl_old - trade_fees_old  # 0.1 - 0.002 = 0.098
is_win_old = pnl_after_fees_old > 0
print(f"AVANT (bug):")
print(f"  current_pnl = {current_pnl_old} (en %)")
print(f"  trade_fees = {trade_fees_old} (en décimal)")
print(f"  pnl_after_fees = {pnl_after_fees_old}")
print(f"  Win? {is_win_old} ✅ (FAUX positif - devrait être loss)")

# APRÈS (correct)
current_pnl_new = 0.001  # En décimal (0.1%)
pnl_after_fees_new = current_pnl_new - trade_fees_new  # 0.001 - 0.002 = -0.001
is_win_new = pnl_after_fees_new > 0
print(f"\nAPRÈS (correct):")
print(f"  current_pnl = {current_pnl_new} (en décimal = {current_pnl_new*100}%)")
print(f"  trade_fees = {trade_fees_new} (en décimal = {trade_fees_new*100}%)")
print(f"  pnl_after_fees = {pnl_after_fees_new} (en décimal = {pnl_after_fees_new*100}%)")
print(f"  Win? {is_win_new} ❌ (correct - loss car gain < fees)")

# Statistiques globales
print("\n" + "=" * 60)
print("IMPACT SUR WIN RATE")
print("=" * 60)
print(f"\nSi 50% des trades gagnent entre 0% et 0.2% brut:")
print(f"  AVANT: Tous comptés WIN (faux positifs)")
print(f"  APRÈS: Tous comptés LOSS (correct - mangés par fees)")
print(f"\nRésultat:")
print(f"  AVANT: Win Rate artificiellement élevé (~42%)")
print(f"  APRÈS: Win Rate réel (~26%) ← CORRECT mais catastrophique")
print(f"\nConclusion:")
print(f"  ❌ L'ancien calcul était FAUX (comparait % et décimal)")
print(f"  ✅ Le nouveau calcul est CORRECT (tout en décimal)")
print(f"  ⚠️  La stratégie est vraiment mauvaise (edge < fees)")
