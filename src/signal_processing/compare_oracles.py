#!/usr/bin/env python3
"""
Compare Oracle labels: raw MACD vs normalized MACD
===================================================

Vérifie si les labels oracle changent avec/sans normalisation MACD.

Usage:
    python src/signal_processing/compare_oracles.py --csv data_trad/BTCUSD_all_5m.csv
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import load_csv, resample_ohlcv, calculate_macd, compute_oracle


def main():
    csv_path = sys.argv[-1] if len(sys.argv) > 1 else 'data_trad/BTCUSD_all_5m.csv'

    print(f"Loading {csv_path} ...")
    df_5m = load_csv(csv_path)
    df_30m = resample_ohlcv(df_5m, 30)
    # Last 5000 for comparison
    df_30m = df_30m.iloc[-5000:]
    n = len(df_30m)
    print(f"  {n:,} bougies 30min")

    # MACD brut
    macd_raw = calculate_macd(df_30m)

    # MACD normalisé (comme l'ancien pipeline)
    close_30m = df_30m['close'].values
    macd_norm = macd_raw / close_30m * 10000

    # Oracle sur les deux
    _, slopes_raw = compute_oracle(macd_raw)
    _, slopes_norm = compute_oracle(macd_norm)

    labels_raw = (slopes_raw > 0).astype(int)
    labels_norm = (slopes_norm > 0).astype(int)

    # Comparaison
    TRIM = 100
    s, e = TRIM, n - TRIM
    lr = labels_raw[s:e]
    ln = labels_norm[s:e]

    accord = np.mean(lr == ln) * 100
    desaccord = 100 - accord

    print(f"\n{'=' * 60}")
    print(f"  COMPARAISON ORACLE : MACD brut vs MACD normalisé")
    print(f"  Eval [{s}:{e}] = {e-s:,} bougies")
    print(f"{'=' * 60}")
    print(f"  Accord labels:    {accord:.2f}%")
    print(f"  Désaccord labels: {desaccord:.2f}%")
    print(f"  Labels UP (raw):  {(lr == 1).sum():,} ({(lr == 1).mean()*100:.1f}%)")
    print(f"  Labels UP (norm): {(ln == 1).sum():,} ({(ln == 1).mean()*100:.1f}%)")

    # Transitions
    trans_raw = sum(1 for i in range(1, len(lr)) if lr[i] != lr[i-1])
    trans_norm = sum(1 for i in range(1, len(ln)) if ln[i] != ln[i-1])
    print(f"  Transitions (raw):  {trans_raw:,}")
    print(f"  Transitions (norm): {trans_norm:,}")

    # PnL oracle sur les deux
    closes = df_30m['close'].values
    for name, slopes in [('Oracle RAW', slopes_raw), ('Oracle NORM', slopes_norm)]:
        labels = (slopes > 0).astype(int)
        pnl = 0.0
        trades = 0
        wins = 0
        position = 0
        entry = 0.0
        for t in range(s, e):
            target = 1 if labels[t] == 1 else -1
            if position == target:
                continue
            price = closes[t]
            if np.isnan(price):
                continue
            if position != 0:
                if position == 1:
                    tp = (price - entry) / entry
                else:
                    tp = (entry - price) / entry
                tp -= 0.001
                pnl += tp
                if tp > 0:
                    wins += 1
            entry = price
            position = target
            trades += 1
            pnl -= 0.001
        # Close last
        if position != 0:
            price = closes[e-1]
            if position == 1:
                tp = (price - entry) / entry
            else:
                tp = (entry - price) / entry
            tp -= 0.001
            pnl += tp
            if tp > 0:
                wins += 1
        wr = wins / trades * 100 if trades > 0 else 0
        print(f"\n  {name}: PnL={pnl*100:+.1f}%, trades={trades:,}, WR={wr:.1f}%")

    bh = (closes[e-1] - closes[s]) / closes[s] * 100
    print(f"  Buy & Hold: {bh:+.1f}%")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
