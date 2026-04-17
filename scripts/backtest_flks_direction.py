#!/usr/bin/env python3
"""
Backtest direction-based sur les slopes FLKS (pas de ML).

Stratégie:
  À chaque bougie TF, regarder slope_k6:
    - slope > 0 → LONG
    - slope < 0 → SHORT
  Exécution: closes_5m_per_candle[t+1][k_substep-1] (cohérent avec
  la disponibilité de slope_k6 à la fin de la bougie TF suivante).

Comparaisons:
  1. Modèle:  backtest sur slope_k6 (feature causale FLKS)
  2. Oracle:  backtest sur slope_oracle (plafond théorique, non-causal)
  3. B&H:     buy & hold passif

Scope par défaut: MACD × 30m.

Usage:
    python scripts/backtest_flks_direction.py
    python scripts/backtest_flks_direction.py --indicator rsi --tf 60
    python scripts/backtest_flks_direction.py --fees 0.0005  # 5bps
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import (
    load_csv, resample_ohlcv, group_per_candle,
    compute_flks_slopes, compute_oracle_labels,
    backtest_5m, buy_and_hold,
)


DATA_DIR = Path('data/raw')


def drop_incomplete_last(df_tf, df_5m, tf_minutes):
    expected = tf_minutes // 5
    drop_count = 0
    for ts in reversed(df_tf.index):
        end = ts + pd.Timedelta(minutes=tf_minutes)
        mask = (df_5m.index >= ts) & (df_5m.index < end)
        if mask.sum() < expected:
            drop_count += 1
        else:
            break
    if drop_count > 0:
        df_tf = df_tf.iloc[:-drop_count]
    return df_tf, drop_count


def compute_extra_stats(slopes, closes_5m_per_candle, k_substep, start, end, fees):
    """
    Backtest manuel pour stats supplémentaires: profit factor, sharpe, avg dur.
    Utilise la même logique que backtest_5m mais log chaque trade.
    """
    n = len(slopes)
    trades = []  # liste de (entry_t, exit_t, pnl, position)
    position = 0
    entry_price = 0.0
    entry_t = -1
    for t in range(start, end):
        if np.isnan(slopes[t]) or slopes[t] == 0:
            continue
        target = 1 if slopes[t] > 0 else -1
        if position == target:
            continue
        if t + 1 >= len(closes_5m_per_candle):
            continue
        closes_5m = closes_5m_per_candle[t + 1]
        step_idx = k_substep - 1
        if step_idx >= len(closes_5m):
            continue
        exec_price = closes_5m[step_idx]
        if np.isnan(exec_price):
            continue
        # Close existing position
        if position != 0:
            pnl = (exec_price - entry_price) / entry_price if position == 1 \
                  else (entry_price - exec_price) / entry_price
            pnl -= 2 * fees
            trades.append({
                'entry_t': entry_t, 'exit_t': t, 'duration': t - entry_t,
                'pnl': pnl, 'position': position,
            })
        # Open new
        entry_price = exec_price
        position = target
        entry_t = t
    # Close last open position
    if position != 0:
        last_candle = min(end, len(closes_5m_per_candle) - 1)
        closes_last = closes_5m_per_candle[last_candle]
        if len(closes_last) > 0 and not np.isnan(closes_last[-1]):
            exit_price = closes_last[-1]
            pnl = (exit_price - entry_price) / entry_price if position == 1 \
                  else (entry_price - exit_price) / entry_price
            pnl -= 2 * fees
            trades.append({
                'entry_t': entry_t, 'exit_t': last_candle,
                'duration': last_candle - entry_t,
                'pnl': pnl, 'position': position,
            })
    if not trades:
        return {'n_trades': 0, 'pnl_pct': 0.0, 'win_rate': 0.0,
                'profit_factor': 0.0, 'sharpe': 0.0,
                'avg_duration': 0.0, 'avg_pnl': 0.0,
                'n_long': 0, 'n_short': 0}
    pnls = np.array([t['pnl'] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    pnl_total = pnls.sum() * 100
    n_wins = len(wins)
    win_rate = n_wins / len(pnls) * 100
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    pf = gross_profit / gross_loss if gross_loss > 1e-10 else np.inf
    # Sharpe: mean / std per trade, annualized rough (assume ~1 trade/TF)
    mu = pnls.mean()
    sd = pnls.std()
    sharpe = mu / sd if sd > 1e-10 else 0.0
    avg_dur = np.mean([t['duration'] for t in trades])
    n_long = sum(1 for t in trades if t['position'] == 1)
    n_short = sum(1 for t in trades if t['position'] == -1)
    return {
        'n_trades': len(trades),
        'pnl_pct': pnl_total,
        'win_rate': win_rate,
        'profit_factor': pf,
        'sharpe': sharpe,
        'avg_duration': avg_dur,
        'avg_pnl': pnls.mean() * 100,
        'n_long': n_long,
        'n_short': n_short,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indicator', default='macd',
                        choices=['macd', 'rsi', 'cci'])
    parser.add_argument('--tf', type=int, default=30, choices=[30, 60])
    parser.add_argument('--k', type=int, default=6,
                        help='k substeps pour slope_k (défaut 6)')
    parser.add_argument('--fees', type=float, default=0.001,
                        help='Fees par trade (défaut 0.001 = 0.1%%)')
    parser.add_argument('--trim', type=int, default=50,
                        help='Bougies à écarter début/fin (warm-up)')
    args = parser.parse_args()

    INDICATOR = args.indicator
    TF = args.tf
    K = args.k
    FEES = args.fees
    TRIM = args.trim

    print("=" * 80)
    print(f"BACKTEST direction-based — {INDICATOR.upper()} × {TF}m  "
          f"(k={K}, fees={FEES*100:.2f}%)")
    print("=" * 80)

    # Load data
    df_5m = load_csv(DATA_DIR / 'BTCUSD_3months_5m.csv')
    tf_label = f'{TF}m' if TF < 60 else '1h'
    df_tf = load_csv(DATA_DIR / f'BTCUSD_3months_{tf_label}.csv')
    df_tf, _ = drop_incomplete_last(df_tf, df_5m, TF)
    print(f"\n5m: {len(df_5m):,} rows  |  {tf_label}: {len(df_tf):,} rows")

    # Compute slopes FLKS + oracle
    print(f"Computing slopes FLKS (k=1..6) ...")
    slopes_df = compute_flks_slopes(df_tf, df_5m, INDICATOR, TF)
    print(f"Computing oracle labels ...")
    oracle_df = compute_oracle_labels(df_tf, INDICATOR)

    # Build closes_5m_per_candle (same grouping as used in FLKS)
    closes_5m_per_candle = group_per_candle(df_5m, df_tf, df_5m['close'].values)

    # Slopes pour le backtest
    slopes_k6 = slopes_df[f'slope_k{K}'].values
    slopes_oracle = oracle_df['slope'].values

    n = len(df_tf)
    start = TRIM
    end = n - 1  # -1 pour que t+1 soit dispo

    # Backtest modèle (slope_k6)
    print(f"\nBacktest range: [{start}, {end}) = {end - start:,} bougies TF")
    print("-" * 80)

    res_model = compute_extra_stats(slopes_k6, closes_5m_per_candle, K,
                                      start, end, FEES)
    res_oracle = compute_extra_stats(slopes_oracle, closes_5m_per_candle, K,
                                       start, end, FEES)

    # Buy & Hold sur df_tf['close']
    bh_pnl = buy_and_hold(df_tf['close'].values, start, end)

    # Affichage
    print(f"\n{'Stratégie':<25} {'PnL %':>10} {'Trades':>8} {'WR %':>8} "
          f"{'PF':>7} {'Sharpe':>8} {'AvgDur':>8} {'L/S':>10}")
    print("-" * 95)

    def print_row(name, r):
        print(f"{name:<25} {r['pnl_pct']:>+10.2f} {r['n_trades']:>8} "
              f"{r['win_rate']:>7.1f}% {r['profit_factor']:>7.2f} "
              f"{r['sharpe']:>8.3f} {r['avg_duration']:>8.1f} "
              f"{r['n_long']}/{r['n_short']:<6}")

    print_row("Oracle (slope)",       res_oracle)
    print_row(f"Model (slope_k{K})",   res_model)
    print(f"{'Buy & Hold':<25} {bh_pnl:>+10.2f}")
    print("-" * 95)

    # Ratios
    if abs(res_oracle['pnl_pct']) > 1e-6:
        capture_ratio = res_model['pnl_pct'] / res_oracle['pnl_pct'] * 100
        print(f"\nCapture ratio (Model / Oracle): {capture_ratio:.1f}%")
    if abs(bh_pnl) > 1e-6:
        alpha = res_model['pnl_pct'] - bh_pnl
        print(f"Alpha vs Buy & Hold: {alpha:+.2f}% (Model - B&H)")

    print(f"\nFees totaux: {res_model['n_trades'] * 2 * FEES * 100:.2f}% "
          f"({res_model['n_trades']} trades × 2 × {FEES*100:.2f}%)")


if __name__ == '__main__':
    main()
