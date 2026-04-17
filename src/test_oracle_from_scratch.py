#!/usr/bin/env python3
"""
Test Oracle PnL from scratch — zero dependency on existing scripts
===================================================================

1. Charger CSV brut BTC 5min
2. Resample 30min
3. Calculer MACD brut sur 30min
4. Appliquer pykalman.smooth()
5. Calculer pente = smoothed[t-1] - smoothed[t-2]
6. Label = sign(pente) → LONG si >0, SHORT si <0
7. PnL : entrée/sortie au close de la bougie 30min

Usage:
    python src/test_oracle_from_scratch.py data_trad/BTCUSD_all_5m.csv
"""

import numpy as np
import pandas as pd
import sys


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'data_trad/BTCUSD_all_5m.csv'
    FEES = 0.001

    # 1. Charger CSV brut
    print(f"[1] Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    for col in ['date', 'datetime', 'time', 'timestamp', 'Date', 'Datetime']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col)
            break
    df.columns = df.columns.str.lower()
    df = df.sort_index()
    print(f"   {len(df):,} 5min candles")

    # 2. Resample 30min
    print(f"[2] Resample 30min ...")
    df_30m = df.resample('30min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()
    n30 = len(df_30m)
    print(f"   {n30:,} bougies 30min")

    # 3. MACD brut
    print(f"[3] MACD brut ...")
    close = df_30m['close']
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    macd = (macd_line - signal).values.astype(np.float64)
    print(f"   MACD range: [{np.nanmin(macd):.1f}, {np.nanmax(macd):.1f}]")

    # 4. pykalman.smooth()
    print(f"[4] pykalman.smooth() ...")
    from pykalman import KalmanFilter as KF
    kf = KF(
        transition_matrices=[[1, 1], [0, 1]],
        observation_matrices=[[1, 0]],
        initial_state_mean=[macd[0], 0.0],
        initial_state_covariance=np.eye(2),
        observation_covariance=0.1,
        transition_covariance=np.eye(2) * 0.01,
    )
    smoothed, _ = kf.smooth(macd)
    positions = smoothed[:, 0]

    # 5. Pente et labels
    print(f"[5] Pentes ...")
    slopes = np.full(n30, np.nan)
    for t in range(2, n30):
        slopes[t] = positions[t-1] - positions[t-2]
    labels = np.where(slopes > 0, 1, 0)

    closes_30m = df_30m['close'].values

    def backtest_oracle(start, end, name):
        pnl = 0.0
        trades = 0
        wins = 0
        position = 0
        entry = 0.0

        for t in range(start, end):
            if np.isnan(slopes[t]):
                continue
            target = 1 if labels[t] == 1 else -1
            if position == target:
                continue
            price = closes_30m[t]
            if np.isnan(price):
                continue
            if position != 0:
                if position == 1:
                    tp = (price - entry) / entry
                else:
                    tp = (entry - price) / entry
                tp -= FEES
                pnl += tp
                if tp > 0:
                    wins += 1
            entry = price
            position = target
            trades += 1
            pnl -= FEES

        if position != 0:
            price = closes_30m[min(end - 1, n30 - 1)]
            if position == 1:
                tp = (price - entry) / entry
            else:
                tp = (entry - price) / entry
            tp -= FEES
            pnl += tp
            if tp > 0:
                wins += 1
            trades += 1

        wr = wins / trades * 100 if trades > 0 else 0
        bh = (closes_30m[min(end-1, n30-1)] - closes_30m[start]) / closes_30m[start] * 100
        frais = trades * FEES * 2 * 100
        trans = sum(1 for t in range(start+1, end) if labels[t] != labels[t-1])

        print(f"\n  {name}:")
        print(f"    Bougies: {end-start:,} | Prix: {closes_30m[start]:.0f} → {closes_30m[min(end-1,n30-1)]:.0f}")
        print(f"    Transitions: {trans:,} | Trades: {trades:,}")
        print(f"    PnL brut: {pnl*100 + frais:+.1f}%")
        print(f"    Frais: {frais:.1f}%")
        print(f"    PnL net: {pnl*100:+.1f}%")
        print(f"    WR: {wr:.1f}% | B&H: {bh:+.1f}%")
        print(f"    Durée moy: {(end-start)/max(trades,1):.0f} bougies = {(end-start)/max(trades,1)*30/60:.1f}h")

    print(f"\n{'=' * 60}")
    print(f"  ORACLE PnL FROM SCRATCH — BTC MACD 30min")
    print(f"  Fees: {FEES*100:.1f}%/trade | Q=0.01, R=0.1")
    print(f"{'=' * 60}")

    backtest_oracle(n30 - 5000, n30, "5000 dernières (83j)")
    backtest_oracle(n30 - 10000, n30, "10000 dernières (166j)")
    test_start = int(n30 * 0.85)
    backtest_oracle(test_start, n30, f"Test split 15% [{test_start}:{n30}]")
    backtest_oracle(100, n30 - 100, "Toute la série (trim 100)")

    print(f"\n{'=' * 60}")


if __name__ == '__main__':
    main()
