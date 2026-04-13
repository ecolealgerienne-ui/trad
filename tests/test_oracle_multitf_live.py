#!/usr/bin/env python3
"""
Oracle Test on Live Multi-Timeframe CSV.

PURPOSE:
    Backtest using the direction labels from the live-style multitf CSV.
    These labels are computed from Kalman-filtered indicators that update
    every 5min (partial candle estimation), not just at candle closure.

PIPELINE:
    1. Load pre-computed CSV (BTCUSD_multitf_macd.csv, etc.)
    2. Extract direction labels ({ind}_{tf}_label)
    3. Split train/val/test (70/15/15 chronological)
    4. Backtest on 5min prices using these labels

CAUSALITY:
    Labels are based on Kalman filter_update (forward-only, causal).
    close_live[i] = close_5min[i], known at time i.
    Backtest executes at open[i+1]. No look-ahead.

Usage:
    python tests/test_oracle_multitf_live.py --indicator macd --timeframe 30m --fees 0.001
    python tests/test_oracle_multitf_live.py --indicator macd --timeframe 1h --fees 0.001
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import argparse
from dataclasses import dataclass
from typing import List
from enum import IntEnum
import logging
from collections import defaultdict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ASSET_CSV_MAP = {
    'BTC': 'BTCUSD', 'ETH': 'ETHUSD', 'BNB': 'BNBUSD',
    'ADA': 'ADAUSD', 'LTC': 'LTCUSD',
}

class Position(IntEnum):
    FLAT = 0; LONG = 1; SHORT = -1

@dataclass
class Trade:
    entry_idx: int; exit_idx: int; duration: int; position: str
    entry_price: float; exit_price: float; pnl: float; pnl_after_fees: float
    asset_name: str = ''; entry_timestamp: float = 0.0


# =============================================================================
# LOAD CSV
# =============================================================================

def load_multitf_csv(asset_name, indicators):
    """Load pre-computed multitf CSV for an asset."""
    ind_tag = '_'.join(indicators)
    filename = f'data/prepared/{ASSET_CSV_MAP[asset_name]}_multitf_{ind_tag}.csv'
    if not Path(filename).exists():
        raise FileNotFoundError(
            f"{filename} not found. Run: python src/prepare_multitf_csv.py "
            f"--assets {asset_name} --indicators {' '.join(indicators)}")
    df = pd.read_csv(filename, parse_dates=['datetime'])
    df = df.set_index('datetime').sort_index()
    logger.info(f"  {asset_name}: {len(df):,} rows, {df.index[0]} -> {df.index[-1]}")
    return df


# =============================================================================
# BACKTEST
# =============================================================================

def backtest_single_asset(labels, opens, timestamps, asset_name, fees):
    """Backtest for one asset. Signal at i, execute at open[i+1]."""
    n = len(labels)
    trades = []
    pos = Position.FLAT
    entry_idx = 0; entry_price = 0.0; entry_ts = 0.0

    for i in range(n - 1):
        target = Position.LONG if int(labels[i]) == 1 else Position.SHORT

        if pos == Position.FLAT:
            pos = target; entry_idx = i
            entry_price = opens[i + 1]; entry_ts = timestamps[i + 1]
            continue

        if pos != target:
            exit_price = opens[i + 1]
            pnl = (exit_price - entry_price) / entry_price if pos == Position.LONG \
                else (entry_price - exit_price) / entry_price
            trades.append(Trade(
                entry_idx=entry_idx, exit_idx=i, duration=i - entry_idx,
                position='LONG' if pos == Position.LONG else 'SHORT',
                entry_price=entry_price, exit_price=exit_price,
                pnl=pnl, pnl_after_fees=pnl - 2 * fees,
                asset_name=asset_name, entry_timestamp=entry_ts))
            pos = target; entry_idx = i
            entry_price = opens[i + 1]; entry_ts = timestamps[i + 1]

    # Close final position
    if pos != Position.FLAT:
        exit_price = opens[-1]
        pnl = (exit_price - entry_price) / entry_price if pos == Position.LONG \
            else (entry_price - exit_price) / entry_price
        trades.append(Trade(
            entry_idx=entry_idx, exit_idx=n-1, duration=n-1-entry_idx,
            position='LONG' if pos == Position.LONG else 'SHORT',
            entry_price=entry_price, exit_price=exit_price,
            pnl=pnl, pnl_after_fees=pnl - 2 * fees,
            asset_name=asset_name, entry_timestamp=entry_ts))
    return trades


# =============================================================================
# STATS
# =============================================================================

def compute_stats(trades):
    if not trades:
        return {}
    pnls = np.array([t.pnl for t in trades])
    pnls_net = np.array([t.pnl_after_fees for t in trades])
    durs = np.array([t.duration for t in trades])
    wins = pnls_net > 0; losses = pnls_net < 0
    sw = pnls_net[wins].sum() if wins.any() else 0
    sl = abs(pnls_net[losses].sum()) if losses.any() else 0
    n_long = sum(1 for t in trades if t.position == 'LONG')
    sharpe = (pnls_net.mean() / pnls_net.std() * np.sqrt(288*365)) if len(pnls_net)>1 and pnls_net.std()>0 else 0
    cum = np.cumsum(pnls_net); rm = np.maximum.accumulate(cum)
    return {
        'n_trades': len(trades), 'n_long': n_long, 'n_short': len(trades)-n_long,
        'pnl_brut': pnls.sum(), 'pnl_net': pnls_net.sum(),
        'fees': pnls.sum() - pnls_net.sum(),
        'win_rate': wins.mean(), 'profit_factor': sw/sl if sl>0 else 0,
        'avg_win': pnls_net[wins].mean() if wins.any() else 0,
        'avg_loss': pnls_net[losses].mean() if losses.any() else 0,
        'avg_duration': durs.mean(), 'sharpe': sharpe,
        'max_drawdown': (rm - cum).max() if len(cum)>0 else 0,
    }

def compute_monthly(trades):
    monthly = defaultdict(list)
    for t in trades:
        ts = t.entry_timestamp
        if ts > 1e18: ts /= 1e9
        elif ts > 1e15: ts /= 1e6
        elif ts > 1e12: ts /= 1e3
        try:
            monthly[datetime.fromtimestamp(ts).strftime('%Y-%m')].append(t)
        except: pass
    results = []
    for ym in sorted(monthly):
        mt = monthly[ym]; n = len(mt)
        results.append({
            'month': ym, 'trades': n,
            'pnl_brut': sum(t.pnl for t in mt),
            'pnl_net': sum(t.pnl_after_fees for t in mt),
            'win_rate': sum(1 for t in mt if t.pnl_after_fees > 0) / n if n else 0
        })
    return results


# =============================================================================
# DISPLAY
# =============================================================================

def print_results(s, indicator, tf, split):
    print(f"\n{'='*70}")
    print(f"ORACLE LIVE — {indicator.upper()} {tf} (split={split})")
    print(f"{'='*70}")
    print(f"\nTrades: {s['n_trades']:,} (L:{s['n_long']:,} S:{s['n_short']:,})")
    print(f"Duration: {s['avg_duration']:.1f}p (~{s['avg_duration']*5:.0f} min)")
    print(f"\nPnL Brut: {s['pnl_brut']*100:+.2f}%")
    print(f"Fees:     {s['fees']*100:.2f}%")
    print(f"PnL Net:  {s['pnl_net']*100:+.2f}%")
    print(f"\nWin Rate: {s['win_rate']*100:.1f}%")
    print(f"PF:       {s['profit_factor']:.2f}")
    print(f"Avg Win:  {s['avg_win']*100:+.3f}%")
    print(f"Avg Loss: {s['avg_loss']*100:+.3f}%")
    print(f"Sharpe:   {s['sharpe']:.2f}")
    print(f"Max DD:   {s['max_drawdown']*100:.2f}%")
    print(f"\n{'✅ PnL Net POSITIF' if s['pnl_net']>0 else '❌ PnL Net NÉGATIF'}")

def print_assets(asset_stats):
    print(f"\n{'='*70}\nPAR ASSET\n{'='*70}")
    print(f"{'Asset':<6} {'Trades':>8} {'PnL Brut':>11} {'PnL Net':>11} {'WR':>7} {'Dur':>7}")
    print('-'*55)
    for a in asset_stats:
        print(f"{a['name']:<6} {a['trades']:>8,} {a['pnl_brut']*100:>+10.2f}% "
              f"{a['pnl_net']*100:>+10.2f}% {a['wr']*100:>6.1f}% {a['dur']:>6.1f}p")

def print_monthly(trades):
    mr = compute_monthly(trades)
    print(f"\n{'='*70}\nPAR MOIS\n{'='*70}")
    print(f"{'Mois':<8} {'Trades':>8} {'PnL Net':>11} {'WR':>7}")
    print('-'*38)
    for m in mr:
        print(f"{m['month']:<8} {m['trades']:>8,} {m['pnl_net']*100:>+10.2f}% {m['win_rate']*100:>6.1f}%")
    if mr:
        avg = sum(m['pnl_net'] for m in mr)/len(mr)
        print(f"{'MOYEN':<8} {'':>8} {avg*100:>+10.2f}%")
        neg = sum(1 for m in mr if m['pnl_net'] < 0)
        print(f"\n{len(mr)} mois, {neg} négatifs")

def print_comparison(s, indicator, tf):
    """Compare with all known Oracle results."""
    refs = {
        'macd': [
            ('5min Oracle (Phase 2.15)',        68924, 53.4, 14359, 2.79, 85.44),
            ('30min pur (Kalman 30min)',         11158, 64.2,  8316, 4.76, 133.62),
            ('1h pur (Kalman 1h)',                5422, 70.8,  7083, 6.72, 161.38),
            ('30min (Kalman 5min, escalier)',    30528, 33.0,   421, 1.05, 4.02),
        ]
    }
    ref_list = refs.get(indicator, [])
    print(f"\n{'='*82}\nCOMPARAISON TOUTES APPROCHES\n{'='*82}")
    print(f"{'Approche':<35} {'Trades':>8} {'WR':>7} {'PnL Net':>10} {'PF':>6} {'Sharpe':>8}")
    print('-'*82)
    for name, tr, wr, pnl, pf, sh in ref_list:
        print(f"{name:<35} {tr:>8,} {wr:>6.1f}% {pnl:>+9.0f}% {pf:>6.2f} {sh:>8.2f}")
    # Current
    label = f"{tf} live (Kalman causal)"
    print(f"{label:<35} {s['n_trades']:>8,} {s['win_rate']*100:>6.1f}% "
          f"{s['pnl_net']*100:>+9.0f}% {s['profit_factor']:>6.2f} {s['sharpe']:>8.2f}")
    print('-'*82)
    print(f"  ← NOUVEAU")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Oracle test on live multitf CSV')
    parser.add_argument('--indicator', default='macd', choices=['macd','rsi','cci'])
    parser.add_argument('--timeframe', default='30m', choices=['30m','1h'])
    parser.add_argument('--split', default='test', choices=['train','val','test'])
    parser.add_argument('--fees', type=float, default=0.001)
    parser.add_argument('--assets', nargs='+', default=['BTC','ETH','BNB','ADA','LTC'])
    args = parser.parse_args()

    label_col = f'{args.indicator}_{args.timeframe}_label'

    logger.info(f"ORACLE LIVE — {args.indicator.upper()} {args.timeframe}")
    logger.info(f"Label column: {label_col}")
    logger.info(f"Split: {args.split}, Fees: {args.fees*100:.2f}%/side")

    all_trades = []
    asset_stats = []

    for asset in args.assets:
        if asset not in ASSET_CSV_MAP:
            continue
        try:
            df = load_multitf_csv(asset, [args.indicator])
        except FileNotFoundError as e:
            logger.warning(str(e)); continue

        if label_col not in df.columns:
            logger.error(f"  Column {label_col} not found in CSV. Available: {list(df.columns)}")
            continue

        # Split
        n = len(df)
        n_train = int(n * 0.70); n_val = int(n * 0.15)
        if args.split == 'train': sl = slice(0, n_train)
        elif args.split == 'val': sl = slice(n_train, n_train + n_val)
        else: sl = slice(n_train + n_val, n)

        labels = df[label_col].values[sl]
        opens = df['open'].values[sl]
        timestamps = df.index[sl].astype(np.int64) / 1e9

        logger.info(f"  Backtest {asset} ({args.split}): {len(labels):,} samples")

        trades = backtest_single_asset(labels, opens, timestamps, asset, args.fees)

        # Per-asset stats
        if trades:
            pnl_b = sum(t.pnl for t in trades)
            pnl_n = sum(t.pnl_after_fees for t in trades)
            wr = sum(1 for t in trades if t.pnl_after_fees > 0) / len(trades)
            dur = sum(t.duration for t in trades) / len(trades)
            asset_stats.append({'name': asset, 'trades': len(trades),
                               'pnl_brut': pnl_b, 'pnl_net': pnl_n, 'wr': wr, 'dur': dur})

        all_trades.extend(trades)

    if not all_trades:
        logger.error("No trades generated!"); return

    s = compute_stats(all_trades)
    print_results(s, args.indicator, args.timeframe, args.split)
    print_assets(asset_stats)
    print_monthly(all_trades)
    print_comparison(s, args.indicator, args.timeframe)


if __name__ == '__main__':
    main()
