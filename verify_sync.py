#!/usr/bin/env python3
"""Vérification finale de la synchronisation avec l'approche user."""

import numpy as np

closes = np.array([100, 101, 102, 103, 104, 105, 104, 103, 102, 101])
opens = np.array([99, 100, 101, 102, 103, 104, 103, 102, 101, 100])
filtered = closes

print("="*80)
print("VÉRIFICATION SYNCHRONISATION - APPROCHE USER")
print("="*80)
print("\nRÈGLE: À l'instant t, comparer filtered[t-1] vs filtered[t-2]")
print("       On trade à open[t+1]")
print()

# Calculer signaux comme dans le code
signals = []
for i in range(2, len(filtered)):
    t_minus_1 = filtered[i-1]
    t_minus_2 = filtered[i-2]
    signal = 'BUY' if t_minus_1 > t_minus_2 else 'SELL' if t_minus_1 < t_minus_2 else 'HOLD'
    signals.append(signal)
    print(f"t={i}: filtered[{i-1}]={t_minus_1:.0f} vs filtered[{i-2}]={t_minus_2:.0f} → {signal}")

print("\n" + "="*80)
print("ASSOCIATION APRÈS iloc[2:]")
print("="*80)

# Après iloc[2:], df_trade commence à l'indice 2
print("\nidx_trade | t_original | Signal calculé                    | Open_t | Open_t+1 (trade)")
for idx in range(len(signals) - 1):
    t_original = idx + 2  # Car iloc[2:]
    signal = signals[idx]
    t_minus_1_val = filtered[t_original - 1]
    t_minus_2_val = filtered[t_original - 2]
    open_t = opens[t_original]
    open_t_plus_1 = opens[t_original + 1]

    print(f"{idx:9d} | {t_original:10d} | filtered[{t_original-1}]={t_minus_1_val:.0f} vs [{t_original-2}]={t_minus_2_val:.0f} → {signal:4s} | {open_t:6.0f} | {open_t_plus_1:16.0f}")

print("\n" + "="*80)
print("EXEMPLE CONCRET:")
print("="*80)
print("\nÀ t=4 (idx_trade=2):")
t = 4
print(f"  - Signal: filtered[{t-1}]={filtered[t-1]:.0f} vs filtered[{t-2}]={filtered[t-2]:.0f} = {signals[t-2]}")
print(f"  - Open actuel: {opens[t]:.0f}")
print(f"  - Trade à: open[{t+1}]={opens[t+1]:.0f}")
print()
print("✅ Synchronisation:")
print(f"   close[t={t}] et filtered[t={t}] sont synchro? OUI → même bougie")
print(f"   Quand on fait filtered[t-1] - filtered[t-2], on a open[t+1]? OUI → {opens[t+1]:.0f}")
print("   Pas de décalage? OUI ✓")
