#!/usr/bin/env python3
"""
Backtest PnL — trade uniquement sur les vrais switches
=======================================================

Le modèle switch → on vérifie si c'est un vrai switch (oracle switch ±6 steps).
  Si OUI → on trade (reversal)
  Si NON → on ignore (on garde la position actuelle)

Objectif : mesurer le PnL des vrais switches isolés.
Si c'est rentable → le signal est bon, le problème est le filtre.
Si c'est pas rentable → le signal lui-même est mauvais.

Usage:
    python src/backtest_vrai_switches_only.py --indicator macd --timeframe 30m
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
NEAR_THRESHOLD = 6
FEES = 0.001


def find_switches(labels):
    switches = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            switches.append(i)
    return switches


def is_near_oracle(model_idx, oracle_switches, near_thr):
    """Check if model switch is near any oracle switch."""
    if len(oracle_switches) == 0:
        return False
    oracle_arr = np.array(oracle_switches)
    return np.min(np.abs(oracle_arr - model_idx)) <= near_thr


def backtest_vrai_only(y_pred_proba, y_test, closes, pred_threshold, fees, near_thr):
    """
    Trade only when model switch is a vrai switch (near oracle transition).
    """
    n = len(y_pred_proba)
    y_pred = (y_pred_proba > pred_threshold).astype(int)
    oracle_switches = find_switches(y_test)

    pnl_total = 0.0
    n_trades = 0
    n_wins = 0
    n_skipped = 0
    position = 0
    entry_price = 0.0
    current_label = y_pred[0]

    for i in range(1, n):
        if y_pred[i] == current_label:
            continue

        # Model wants to switch
        if is_near_oracle(i, oracle_switches, near_thr):
            # VRAI switch — execute trade
            exec_price = closes[i]
            if np.isnan(exec_price):
                continue

            target = 1 if y_pred[i] == 1 else -1

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
            current_label = y_pred[i]
        else:
            # FAUX switch — skip, keep current position and label
            n_skipped += 1
            # Don't update current_label — ignore this switch

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
    return pnl_total * 100, n_trades, wr, n_skipped


def backtest_all_switches(y_pred_proba, closes, pred_threshold, fees):
    """Baseline: trade on ALL model switches."""
    n = len(y_pred_proba)
    y_pred = (y_pred_proba > pred_threshold).astype(int)

    pnl_total = 0.0
    n_trades = 0
    n_wins = 0
    position = 0
    entry_price = 0.0

    for i in range(1, n):
        target = 1 if y_pred[i] == 1 else -1
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

        entry_price = exec_price
        position = target
        n_trades += 1
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

    wr = (n_wins / n_trades * 100.0) if n_trades > 0 else 0.0
    return pnl_total * 100, n_trades, wr


def backtest_oracle(y_test, closes, fees):
    """Oracle: trade on oracle labels directly."""
    n = len(y_test)
    pnl_total = 0.0
    n_trades = 0
    n_wins = 0
    position = 0
    entry_price = 0.0

    for i in range(1, n):
        target = 1 if y_test[i] == 1 else -1
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

        entry_price = exec_price
        position = target
        n_trades += 1
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

    wr = (n_wins / n_trades * 100.0) if n_trades > 0 else 0.0
    return pnl_total * 100, n_trades, wr


def main():
    parser = argparse.ArgumentParser(description='Backtest vrai switches only')
    parser.add_argument('--indicator', default='macd')
    parser.add_argument('--timeframe', default='30m')
    parser.add_argument('--fees', type=float, default=FEES)
    args = parser.parse_args()

    fees = args.fees
    ind = args.indicator
    tf = args.timeframe

    # Load predictions + closes (aligned)
    sys.path.insert(0, str(Path(__file__).parent / 'signal_processing'))
    from core import load_test_data
    y_test, y_pred_proba, _, closes_test, n_test, csv_used = load_test_data(ind, tf)
    print(f"  CSV: {csv_used}")

    bh = (closes_test[-1] - closes_test[0]) / closes_test[0] * 100

    oracle_switches = find_switches(y_test)
    model_switches = find_switches((y_pred_proba > 0.5).astype(int))

    print(f"\n{'=' * 70}")
    print(f"  BACKTEST VRAI SWITCHES ONLY — {ind.upper()}_{tf}")
    print(f"  Test: {n_test:,} samples | Fees: {fees*100:.1f}%/trade")
    print(f"  Oracle switches: {len(oracle_switches):,}")
    print(f"  Model switches: {len(model_switches):,}")
    print(f"{'=' * 70}")

    # Oracle backtest
    o_pnl, o_trades, o_wr = backtest_oracle(y_test, closes_test, fees)

    # All switches backtest
    a_pnl, a_trades, a_wr = backtest_all_switches(y_pred_proba, closes_test, 0.5, fees)

    # Vrai switches only (different near thresholds)
    print(f"\n  {'Method':<35} {'PnL':>8} {'Trades':>7} {'WR':>6} {'Skipped':>8}")
    print(f"  {'-' * 68}")
    print(f"  {'Oracle (labels parfaits)':<35} {o_pnl:>+7.1f}% {o_trades:>7} {o_wr:>5.1f}%")
    print(f"  {'Modèle (tous switches)':<35} {a_pnl:>+7.1f}% {a_trades:>7} {a_wr:>5.1f}%")

    for near in [3, 6, 10, 15]:
        v_pnl, v_trades, v_wr, v_skip = backtest_vrai_only(
            y_pred_proba, y_test, closes_test, 0.5, fees, near)
        print(f"  {'Vrai only (±' + str(near) + ' steps)':<35} "
              f"{v_pnl:>+7.1f}% {v_trades:>7} {v_wr:>5.1f}% {v_skip:>8}")

    print(f"  {'-' * 68}")
    print(f"  {'Buy & Hold':<35} {bh:>+7.1f}%")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
