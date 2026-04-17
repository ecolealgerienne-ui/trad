#!/usr/bin/env python3
"""Debug: verify oracle PnL step by step."""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'signal_processing'))
from core import load_test_data

y_test, _, _, closes, n_test, csv_path = load_test_data('macd', '30m')
print(f"CSV: {csv_path}")
print(f"Test samples: {n_test:,}")
print(f"Closes range: {closes[0]:.2f} → {closes[-1]:.2f}")
print(f"B&H: {(closes[-1]-closes[0])/closes[0]*100:+.1f}%")

# Oracle labels stats
n_up = (y_test == 1).sum()
n_down = (y_test == 0).sum()
transitions = sum(1 for i in range(1, n_test) if y_test[i] != y_test[i-1])
print(f"Labels: {n_up:,} UP, {n_down:,} DOWN")
print(f"Transitions: {transitions:,}")

# Sample: first 30 labels and closes
print(f"\nFirst 30 samples:")
print(f"{'i':>5} {'label':>6} {'close':>12} {'change':>8}")
for i in range(min(30, n_test)):
    label = y_test[i]
    close = closes[i]
    change = "" if i == 0 else ("FLIP" if y_test[i] != y_test[i-1] else "")
    print(f"{i:>5} {label:>6} {close:>12.2f} {change:>8}")

# Manual PnL calculation on first 1000 transitions
pnl = 0.0
trades = 0
wins = 0
position = 0
entry = 0.0
fees = 0.001

for i in range(n_test):
    target = 1 if y_test[i] == 1 else -1
    if position == target:
        continue
    price = closes[i]
    if np.isnan(price):
        continue
    if position != 0:
        if position == 1:
            tp = (price - entry) / entry
        else:
            tp = (entry - price) / entry
        tp -= fees
        pnl += tp
        if tp > 0:
            wins += 1
        trades += 1
    entry = price
    position = target
    pnl -= fees

# Close last
if position != 0:
    price = closes[-1]
    if position == 1:
        tp = (price - entry) / entry
    else:
        tp = (entry - price) / entry
    tp -= fees
    pnl += tp
    if tp > 0:
        wins += 1
    trades += 1

wr = wins / trades * 100 if trades > 0 else 0
print(f"\nOracle PnL: {pnl*100:+.1f}%")
print(f"Trades: {trades:,}")
print(f"WR: {wr:.1f}%")
print(f"Fees total: {trades * 0.2:.1f}%")

# Avg trade duration (in 5min steps)
durations = []
last_switch = 0
for i in range(1, n_test):
    if y_test[i] != y_test[i-1]:
        durations.append(i - last_switch)
        last_switch = i
if durations:
    print(f"Avg trade duration: {np.mean(durations):.1f} steps = {np.mean(durations)*5:.0f} min")
    print(f"Min duration: {min(durations)} steps = {min(durations)*5} min")
