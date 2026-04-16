#!/usr/bin/env python3
"""
Backtest PnL — trade uniquement quand modèle et oracle sont d'accord
=====================================================================

Règle simple:
  modèle = UP  ET oracle = UP  → LONG
  modèle = DOWN ET oracle = DOWN → SHORT
  désaccord → FLAT (pas de position)

Mesure le PnL du signal quand il est correct.

Usage:
    python src/backtest_consensus_direction.py --indicator macd --timeframe 30m
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from constants import PREPARED_DATA_DIR

ASSET_CSV_MAP = {'BTC': 'BTCUSD'}
FEES = 0.001


def backtest_consensus(y_pred_binary, y_oracle, closes, fees):
    """
    Follow oracle direction, but only act on transitions detected by the model.

    At each oracle transition:
      - If the model also switches (within ±6 steps) in the same direction → trade
      - If the model doesn't switch → skip this transition, keep previous position

    Number of trades ≤ oracle transitions.
    """
    n = len(y_pred_binary)
    NEAR = 6

    # Find oracle transitions
    oracle_transitions = []
    for i in range(1, n):
        if y_oracle[i] != y_oracle[i - 1]:
            oracle_transitions.append(i)

    # Find model transitions
    model_transitions = set()
    for i in range(1, n):
        if y_pred_binary[i] != y_pred_binary[i - 1]:
            model_transitions.add(i)

    pnl_total = 0.0
    n_trades = 0
    n_wins = 0
    n_confirmed = 0
    n_skipped = 0
    position = 0  # 0=flat, +1=long, -1=short
    entry_price = 0.0

    for o_idx in oracle_transitions:
        oracle_dir = 1 if y_oracle[o_idx] == 1 else -1

        # Check if model also switches nearby in the same direction
        model_confirms = False
        for delta in range(-NEAR, NEAR + 1):
            m_idx = o_idx + delta
            if m_idx in model_transitions:
                model_dir = 1 if y_pred_binary[m_idx] == 1 else -1
                if model_dir == oracle_dir:
                    model_confirms = True
                    break

        if not model_confirms:
            n_skipped += 1
            continue

        n_confirmed += 1
        target = oracle_dir

        if position == target:
            continue

        exec_price = closes[min(o_idx, n - 1)]
        if np.isnan(exec_price):
            continue

        # Close existing position
        if position != 0:
            if position == 1:
                trade_pnl = (exec_price - entry_price) / entry_price
            else:
                trade_pnl = (entry_price - exec_price) / entry_price
            trade_pnl -= fees
            pnl_total += trade_pnl
            if trade_pnl > 0:
                n_wins += 1

        # Open new position
        entry_price = exec_price
        position = target
        n_trades += 1
        pnl_total -= fees

    # Close last position
    if position != 0 and not np.isnan(closes[-1]):
        if position == 1:
            trade_pnl = (closes[-1] - entry_price) / entry_price
        else:
            trade_pnl = (entry_price - closes[-1]) / entry_price
        trade_pnl -= fees
        pnl_total += trade_pnl
        if trade_pnl > 0:
            n_wins += 1

    wr = (n_wins / n_trades * 100.0) if n_trades > 0 else 0.0

    return {
        'pnl': pnl_total * 100,
        'trades': n_trades,
        'wr': wr,
        'confirmed': n_confirmed,
        'skipped': n_skipped,
    }


def backtest_model_only(y_pred_binary, closes, fees):
    """Baseline: always in position based on model."""
    n = len(y_pred_binary)
    pnl_total = 0.0
    n_trades = 0
    n_wins = 0
    position = 0
    entry_price = 0.0

    for i in range(n):
        target = 1 if y_pred_binary[i] == 1 else -1
        if position == target:
            continue
        exec_price = closes[i]
        if np.isnan(exec_price):
            continue
        if position != 0:
            if position == 1:
                trade_pnl = (exec_price - entry_price) / entry_price
            else:
                trade_pnl = (entry_price - exec_price) / entry_price
            trade_pnl -= fees
            pnl_total += trade_pnl
            if trade_pnl > 0:
                n_wins += 1
            n_trades += 1
        entry_price = exec_price
        position = target
        pnl_total -= fees

    if position != 0 and not np.isnan(closes[-1]):
        if position == 1:
            trade_pnl = (closes[-1] - entry_price) / entry_price
        else:
            trade_pnl = (entry_price - closes[-1]) / entry_price
        trade_pnl -= fees
        pnl_total += trade_pnl
        if trade_pnl > 0:
            n_wins += 1
        n_trades += 1

    wr = (n_wins / n_trades * 100.0) if n_trades > 0 else 0.0
    return {'pnl': pnl_total * 100, 'trades': n_trades, 'wr': wr}


def backtest_oracle_only(y_oracle, closes, fees):
    """Oracle: always in position based on oracle labels."""
    return backtest_model_only(y_oracle, closes, fees)


def main():
    parser = argparse.ArgumentParser(description='Backtest consensus direction')
    parser.add_argument('--indicator', default='macd')
    parser.add_argument('--timeframe', default='30m')
    parser.add_argument('--fees', type=float, default=FEES)
    args = parser.parse_args()

    fees = args.fees
    ind = args.indicator
    tf = args.timeframe

    # Load predictions
    npz_path = f'{PREPARED_DATA_DIR}/{ind}_{tf}_dataset.npz'
    data = np.load(npz_path, allow_pickle=True)
    if 'y_test' in data:
        y_test = data['y_test']
        y_pred_proba = data['y_test_pred']
    else:
        y_test = data['test_labels']
        y_pred_proba = data['test_preds']

    n_test = len(y_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Load closes
    base = ASSET_CSV_MAP['BTC']
    csv_path = f'{PREPARED_DATA_DIR}/{base}_multitf_macd_rsi_cci.csv'
    df = pd.read_csv(csv_path, parse_dates=['datetime']).set_index('datetime').sort_index()
    closes_all = df['close'].dropna().values
    closes_test = closes_all[-n_test:]

    bh = (closes_test[-1] - closes_test[0]) / closes_test[0] * 100

    # Agreement stats
    agree = (y_pred == y_test).sum()
    disagree = (y_pred != y_test).sum()

    print(f"\n{'=' * 70}")
    print(f"  BACKTEST CONSENSUS DIRECTION — {ind.upper()}_{tf}")
    print(f"  Test: {n_test:,} samples | Fees: {fees*100:.1f}%/trade")
    print(f"  Agreement: {agree:,} ({agree/n_test*100:.1f}%)")
    print(f"  Disagreement: {disagree:,} ({disagree/n_test*100:.1f}%)")
    print(f"{'=' * 70}")

    # Oracle
    r_oracle = backtest_oracle_only(y_test, closes_test, fees)

    # Model only (baseline)
    r_model = backtest_model_only(y_pred, closes_test, fees)

    # Consensus
    r_consensus = backtest_consensus(y_pred, y_test, closes_test, fees)

    print(f"\n  {'Method':<45} {'PnL':>8} {'Trades':>7} {'WR':>6}")
    print(f"  {'-' * 68}")
    print(f"  {'Oracle (toutes transitions)':<45} {r_oracle['pnl']:>+7.1f}% {r_oracle['trades']:>7} {r_oracle['wr']:>5.1f}%")
    print(f"  {'Modèle seul (tous switches)':<45} {r_model['pnl']:>+7.1f}% {r_model['trades']:>7} {r_model['wr']:>5.1f}%")
    print(f"  {'Consensus (oracle + modèle confirme)':<45} {r_consensus['pnl']:>+7.1f}% {r_consensus['trades']:>7} {r_consensus['wr']:>5.1f}%")
    print(f"  {'Buy & Hold':<45} {bh:>+7.1f}%")
    print(f"  {'-' * 68}")
    print(f"\n  Consensus details:")
    oracle_trans = sum(1 for i in range(1, len(y_test)) if y_test[i] != y_test[i-1])
    print(f"    Oracle transitions: {oracle_trans:,}")
    print(f"    Modèle confirme (±6 steps, même dir): {r_consensus['confirmed']:,}")
    print(f"    Modèle ne confirme pas (skipped): {r_consensus['skipped']:,}")
    print(f"    Trades effectifs: {r_consensus['trades']:,} (≤ oracle)")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
