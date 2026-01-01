#!/usr/bin/env python3
"""Debug de la synchronisation signal/prix."""

import numpy as np

# Simuler des données simplifiées
n = 10
closes = np.array([100, 101, 102, 103, 104, 105, 104, 103, 102, 101])
opens = np.array([99, 100, 101, 102, 103, 104, 103, 102, 101, 100])
filtered = closes  # Pour simplifier, filtre = close

print("="*80)
print("VÉRIFICATION DE LA SYNCHRONISATION")
print("="*80)
print("\nDonnées initiales:")
print("Indice | Close | Filtered | Open")
for i in range(n):
    print(f"{i:6d} | {closes[i]:5.0f} | {filtered[i]:8.0f} | {opens[i]:4.0f}")

print("\n" + "="*80)
print("CALCUL DES SIGNAUX")
print("="*80)

# Logique actuelle (APRÈS ma correction)
signals = []
for i in range(1, n):
    current = filtered[i]
    previous = filtered[i-1]

    if current > previous:
        signal = 'BUY'
    elif current < previous:
        signal = 'SELL'
    else:
        signal = 'HOLD'

    signals.append(signal)
    print(f"i={i}: filtered[{i}]={current:.0f} vs filtered[{i-1}]={previous:.0f} → {signal}")

print("\n" + "="*80)
print("ASSOCIATION SIGNAL → BOUGIE")
print("="*80)

# On enlève la première bougie (pas de signal à i=0)
# signals[0] est pour i=1, signals[1] est pour i=2, etc.

print("\nAPRÈS iloc[1:]:")
print("idx_trade | idx_original | Signal | Open_actuel | Open_suivant (trade)")
for idx in range(len(signals) - 1):
    idx_original = idx + 1  # Car on a fait iloc[1:]
    signal = signals[idx]
    open_current = opens[idx_original]
    open_next = opens[idx_original + 1]
    print(f"{idx:9d} | {idx_original:12d} | {signal:6s} | {open_current:11.0f} | {open_next:20.0f}")

print("\n" + "="*80)
print("EXEMPLE CONCRET:")
print("="*80)
print("\nÀ l'instant t=2:")
print(f"  - filtered[2]={filtered[2]:.0f} vs filtered[1]={filtered[1]:.0f}")
print(f"  - Signal: {signals[1]}")  # signals[1] car i=2 → signals[i-1]
print(f"  - Ce signal est à idx_trade=1 (car iloc[1:])")
print(f"  - On trade à open[3]={opens[3]:.0f}")

print("\n✅ Vérification:")
print(f"  - À t=2, on compare filtered[2] vs filtered[1] ✓")
print(f"  - On trade à open[t+1] = open[3] ✓")
print(f"  - Synchronisation correcte? OUI")
