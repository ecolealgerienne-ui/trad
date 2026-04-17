#!/usr/bin/env python3
"""
Backtest PnL de l'oracle (smoother RTS non-causal) sur une période récente.

Mesure la BORNE SUPÉRIEURE du signal 'sign(slope_smoother)' :
  - Utilise tout l'historique + tout le futur (non-causal)
  - Trim 100 début + 100 fin pour éviter les effets de bord du smoother
  - Backtest en 2 modes d'exécution pour comparaison :
      * 30m : exec à close_30m[t] (disponible 30 min avant signal suivant)
      * 5m k=6 : exec à close_5m_per_candle[t+1][5] (exec 30 min plus tard)

Usage:
    python scripts/backtest_oracle_pnl.py
    python scripts/backtest_oracle_pnl.py --days 450 --fees 0.001
    python scripts/backtest_oracle_pnl.py --days 180 --fees 0.0002
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
    compute_oracle_labels,
    backtest_5m, backtest_30m, buy_and_hold,
)

SRC_5M = Path('data_trad/BTCUSD_all_5m.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indicator', default='macd',
                        choices=['macd', 'rsi', 'cci'])
    parser.add_argument('--tf', type=int, default=30)
    parser.add_argument('--days', type=int, default=450,
                        help='Derniers N jours à utiliser (default 450 = 15 mois)')
    parser.add_argument('--trim', type=int, default=100,
                        help='Trim au début ET à la fin (default 100 bougies TF)')
    parser.add_argument('--fees', type=float, default=0.001,
                        help='Fees par trade (default 0.001 = 0.1%%)')
    parser.add_argument('--k', type=int, default=6,
                        help='k substep pour backtest_5m (default 6)')
    parser.add_argument('--holding-min', type=int, default=0)
    args = parser.parse_args()

    print("=" * 80)
    print(f"BACKTEST ORACLE — {args.indicator.upper()} × {args.tf}m  "
          f"(derniers {args.days} jours, trim={args.trim}, fees={args.fees*100:.2f}%)")
    print("=" * 80)

    if not SRC_5M.exists():
        print(f"❌ Source introuvable: {SRC_5M}")
        return

    # ========== [1] Load + filter last N days ==========
    print(f"\n[1/6] Load {SRC_5M} ...")
    df_5m_full = load_csv(SRC_5M)
    print(f"  Full: {len(df_5m_full):,} rows  |  "
          f"{df_5m_full.index[0]} → {df_5m_full.index[-1]}")

    end_date = df_5m_full.index[-1]
    start_date = end_date - pd.Timedelta(days=args.days)
    df_5m = df_5m_full.loc[df_5m_full.index >= start_date].copy()
    print(f"  Filter last {args.days} days: {len(df_5m):,} rows  |  "
          f"{df_5m.index[0]} → {df_5m.index[-1]}")

    # ========== [2] Resample 30m ==========
    print(f"\n[2/6] Resample 5m → {args.tf}m ...")
    df_tf = resample_ohlcv(df_5m, args.tf)
    print(f"  {len(df_tf):,} bougies {args.tf}m")

    # ========== [3] Oracle (RTS smoother non-causal) ==========
    print(f"\n[3/6] compute_oracle_labels (pykalman smoother) ...")
    oracle_df = compute_oracle_labels(df_tf, args.indicator)
    slopes_oracle = oracle_df['slope'].values
    print(f"  positions/slope/label: {oracle_df.shape}")
    # Info smoother
    n_positive = int((slopes_oracle > 0).sum())
    n_negative = int((slopes_oracle < 0).sum())
    n_zero = int((slopes_oracle == 0).sum())
    print(f"  Distribution slopes: +{n_positive:,} UP / -{n_negative:,} DOWN "
          f"/ 0={n_zero:,}")

    # ========== [4] Trim ==========
    n_tf = len(df_tf)
    start = args.trim
    end = n_tf - args.trim
    n_kept = end - start
    print(f"\n[4/6] Trim {args.trim} début + {args.trim} fin → "
          f"range [{start}, {end}) = {n_kept:,} bougies {args.tf}m")

    # ========== [5] closes_5m_per_candle (pour backtest_5m) ==========
    print(f"\n[5/6] group_per_candle (5min → per {args.tf}m) ...")
    closes_5m_per_candle = group_per_candle(df_5m, df_tf, df_5m['close'].values)
    print(f"  {len(closes_5m_per_candle):,} buckets")

    # ========== [6] Backtests ==========
    print(f"\n[6/6] Backtests (fees={args.fees*100:.2f}%, holding_min={args.holding_min}, "
          f"lag=+1 tick 5min)")

    closes_30m = df_tf['close'].values

    # Mode "30m honnête" : signal à close bougie t, exec à xx:05 ou xx:35
    # (équivalent à backtest_5m avec k_substep=0 + lag_5min=1)
    # Signal slope_oracle[t] dispo à t+30min, exec au close de la 1ère 5min de la
    # bougie t+1 (prix à t+35min). Évite xx:00 et xx:30.
    res_30m_honest = backtest_5m(slopes_oracle, closes_5m_per_candle, 0,
                                   start, end, args.fees,
                                   threshold=0.0,
                                   holding_min=args.holding_min,
                                   lag_5min=1)

    # Mode "30m instantané" (legacy) : exec à close 30m exact (= xx:00 / xx:30)
    res_30m_instant = backtest_30m(slopes_oracle, closes_30m, start, end,
                                     args.fees, threshold=0.0,
                                     holding_min=args.holding_min)

    # Mode "5m k=6 honnête" : signal à t+30min (fin 6ème sous-pas), exec à xx:35 ou xx:05
    res_5m_k6_honest = backtest_5m(slopes_oracle, closes_5m_per_candle, args.k,
                                     start, end, args.fees,
                                     threshold=0.0,
                                     holding_min=args.holding_min,
                                     lag_5min=1)

    # Mode "5m k=6 instantané" (legacy) : exec au dernier sous-pas (xx:30 / xx:00)
    res_5m_k6_instant = backtest_5m(slopes_oracle, closes_5m_per_candle, args.k,
                                      start, end, args.fees,
                                      threshold=0.0,
                                      holding_min=args.holding_min,
                                      lag_5min=0)

    # Buy & Hold
    bh_pnl = buy_and_hold(closes_30m, start, end)

    # Période
    period_start = df_tf.index[start]
    period_end = df_tf.index[end - 1]
    period_days = (period_end - period_start).total_seconds() / (24 * 3600)

    # Affichage
    print("\n" + "=" * 95)
    print(f"RÉSULTATS  ({period_start} → {period_end}, {period_days:.0f} jours)")
    print("=" * 95)
    print(f"{'Stratégie':<40} {'PnL %':>12} {'Trades':>8} {'WR %':>8}")
    print("-" * 70)
    print(f"{'Oracle 30m instantané (legacy, xx:00/30)':<40} "
          f"{res_30m_instant['pnl_pct']:>+12.2f} {res_30m_instant['trades']:>8} "
          f"{res_30m_instant['win_rate']:>7.1f}%")
    print(f"{'Oracle 30m honnête (exec xx:05/35)':<40} "
          f"{res_30m_honest['pnl_pct']:>+12.2f} {res_30m_honest['trades']:>8} "
          f"{res_30m_honest['win_rate']:>7.1f}%")
    print(f"{'Oracle 5m k=6 instantané (legacy)':<40} "
          f"{res_5m_k6_instant['pnl_pct']:>+12.2f} {res_5m_k6_instant['trades']:>8} "
          f"{res_5m_k6_instant['win_rate']:>7.1f}%")
    print(f"{'Oracle 5m k=6 honnête (exec xx:35/05)':<40} "
          f"{res_5m_k6_honest['pnl_pct']:>+12.2f} {res_5m_k6_honest['trades']:>8} "
          f"{res_5m_k6_honest['win_rate']:>7.1f}%")
    print(f"{'Buy & Hold':<40} {bh_pnl:>+12.2f}")
    print("-" * 70)

    # Fees et PnL brut
    n_tr = res_30m_honest['trades']
    fees_tot = n_tr * 2 * args.fees * 100
    print(f"\nFees 30m honnête : {n_tr} trades × 2 × {args.fees*100:.2f}% = "
          f"{fees_tot:.2f}%")
    print(f"  PnL brut  : {res_30m_honest['pnl_pct'] + fees_tot:+.2f}%")
    print(f"  PnL net   : {res_30m_honest['pnl_pct']:+.2f}%")
    print(f"  vs B&H    : alpha = {res_30m_honest['pnl_pct'] - bh_pnl:+.2f}%")


if __name__ == '__main__':
    main()
