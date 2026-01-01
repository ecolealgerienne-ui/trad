#!/usr/bin/env python3
"""Comparaison des deux approches de calcul de signal."""

import numpy as np

# Données test
closes = np.array([100, 101, 102, 103, 104, 105, 104, 103, 102, 101])
opens = np.array([99, 100, 101, 102, 103, 104, 103, 102, 101, 100])
filtered = closes

print("="*80)
print("COMPARAISON DES DEUX APPROCHES")
print("="*80)

print("\n" + "="*80)
print("APPROCHE 1: filtered[t] vs filtered[t-1] (CODE ACTUEL)")
print("="*80)
print("\nÀ l'instant t, on compare filtered[t] vs filtered[t-1]")
print("On trade à open[t+1]")

for i in range(1, len(filtered) - 1):
    current = filtered[i]
    previous = filtered[i-1]
    next_open = opens[i+1]

    signal = 'BUY' if current > previous else 'SELL' if current < previous else 'HOLD'
    print(f"t={i}: filtered[{i}]={current:.0f} vs filtered[{i-1}]={previous:.0f} → {signal} → trade à open[{i+1}]={next_open:.0f}")

print("\n" + "="*80)
print("APPROCHE 2: filtered[t-1] vs filtered[t-2] (USER?)")
print("="*80)
print("\nÀ l'instant t, on compare filtered[t-1] vs filtered[t-2]")
print("On trade à open[t+1]")

for i in range(2, len(filtered) - 1):
    t_minus_1 = filtered[i-1]
    t_minus_2 = filtered[i-2]
    next_open = opens[i+1]

    signal = 'BUY' if t_minus_1 > t_minus_2 else 'SELL' if t_minus_1 < t_minus_2 else 'HOLD'
    print(f"t={i}: filtered[{i-1}]={t_minus_1:.0f} vs filtered[{i-2}]={t_minus_2:.0f} → {signal} → trade à open[{i+1}]={next_open:.0f}")

print("\n" + "="*80)
print("DIFFÉRENCE CLEF:")
print("="*80)
print("\nApproche 1: On utilise l'information de la bougie ACTUELLE (t)")
print("  → Peut introduire un look-ahead bias si on trade trop tôt")
print("\nApproche 2: On utilise SEULEMENT l'information du PASSÉ (t-1, t-2)")
print("  → Plus sûr, pas de look-ahead bias")
print("\n❓ Laquelle est la bonne?")
