#!/usr/bin/env python3
"""
Viterbi + CUSUM Post-Processing — reduce false switches without retraining
==========================================================================

Applies post-processing to existing model predictions to reduce spurious
switches. No retraining needed.

Method 1: Viterbi decoding with transition penalty
Method 2: CUSUM filter (López de Prado)
Method 3: Combined Viterbi + CUSUM

Compares: switches count, ratio, PnL, WR vs baseline and oracle.

Usage:
    python src/postprocess_viterbi_cusum.py --indicator macd --timeframe 30m
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


# ============================================================================
# VITERBI DECODING
# ============================================================================

def viterbi_decode(probs, self_trans=0.95):
    """
    Viterbi decoding on binary probability sequence.

    probs: array of P(UP) at each timestep
    self_trans: probability of staying in current state (0.9 to 0.99)

    Transition matrix:
      [[self_trans, 1-self_trans],
       [1-self_trans, self_trans]]

    Returns: binary labels (0/1)
    """
    n = len(probs)
    trans = self_trans

    # Log probabilities for numerical stability
    log_trans_same = np.log(trans)
    log_trans_switch = np.log(1 - trans)

    # Emission log-probs
    log_emit = np.zeros((n, 2))
    log_emit[:, 1] = np.log(np.clip(probs, 1e-10, 1 - 1e-10))
    log_emit[:, 0] = np.log(np.clip(1 - probs, 1e-10, 1 - 1e-10))

    # Viterbi forward
    V = np.zeros((n, 2))
    backptr = np.zeros((n, 2), dtype=int)

    # Initialize
    V[0] = log_emit[0] + np.log(0.5)  # uniform prior

    for t in range(1, n):
        for s in range(2):
            # Stay in same state
            score_same = V[t-1, s] + log_trans_same
            # Switch from other state
            other = 1 - s
            score_switch = V[t-1, other] + log_trans_switch

            if score_same >= score_switch:
                V[t, s] = score_same + log_emit[t, s]
                backptr[t, s] = s
            else:
                V[t, s] = score_switch + log_emit[t, s]
                backptr[t, s] = other

    # Backtrack
    labels = np.zeros(n, dtype=int)
    labels[-1] = np.argmax(V[-1])
    for t in range(n-2, -1, -1):
        labels[t] = backptr[t+1, labels[t+1]]

    return labels


# ============================================================================
# CUSUM FILTER
# ============================================================================

def cusum_filter(probs, threshold=2.0):
    """
    CUSUM filter on probability sequence.

    Accumulates evidence for direction change. Only switches when
    cumulative evidence exceeds threshold.

    probs: array of P(UP)
    threshold: cumulative sum threshold to trigger switch

    Returns: binary labels (0/1)
    """
    n = len(probs)
    labels = np.zeros(n, dtype=int)

    # Initialize
    current_state = 1 if probs[0] > 0.5 else 0
    labels[0] = current_state

    s_up = 0.0   # accumulator for UP evidence
    s_down = 0.0  # accumulator for DOWN evidence

    for t in range(1, n):
        # Centered signal: deviation from 0.5
        x = probs[t] - 0.5

        # Accumulate
        s_up = max(0, s_up + x)
        s_down = min(0, s_down + x)

        if current_state == 0 and s_up > threshold:
            # Switch to UP
            current_state = 1
            s_up = 0.0
            s_down = 0.0
        elif current_state == 1 and -s_down > threshold:
            # Switch to DOWN
            current_state = 0
            s_up = 0.0
            s_down = 0.0

        labels[t] = current_state

    return labels


# ============================================================================
# BACKTEST
# ============================================================================

def backtest(labels, closes, fees):
    n = len(labels)
    pnl_total = 0.0
    n_trades = 0
    n_wins = 0
    position = 0
    entry_price = 0.0

    for i in range(n):
        target = 1 if labels[i] == 1 else -1
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
    switches = sum(1 for i in range(1, n) if labels[i] != labels[i-1])
    return pnl_total * 100, n_trades, wr, switches


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Viterbi + CUSUM post-processing')
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

    # Load closes
    base = ASSET_CSV_MAP['BTC']
    csv_path = f'{PREPARED_DATA_DIR}/{base}_multitf_macd_rsi_cci.csv'
    df = pd.read_csv(csv_path, parse_dates=['datetime']).set_index('datetime').sort_index()
    closes_test = df['close'].dropna().values[-n_test:]

    bh = (closes_test[-1] - closes_test[0]) / closes_test[0] * 100
    oracle_switches = sum(1 for i in range(1, n_test) if y_test[i] != y_test[i-1])

    print(f"\n{'=' * 80}")
    print(f"  VITERBI + CUSUM POST-PROCESSING — {ind.upper()}_{tf}")
    print(f"  Test: {n_test:,} samples | Oracle switches: {oracle_switches:,} | B&H: {bh:+.1f}%")
    print(f"{'=' * 80}")

    # Baseline
    y_baseline = (y_pred_proba > 0.5).astype(int)
    b_pnl, b_tr, b_wr, b_sw = backtest(y_baseline, closes_test, fees)

    # Oracle
    o_pnl, o_tr, o_wr, o_sw = backtest(y_test, closes_test, fees)

    print(f"\n  {'Method':<40} {'PnL':>8} {'Trades':>7} {'WR':>6} {'Switches':>9} {'Ratio':>6}")
    print(f"  {'-' * 78}")
    print(f"  {'Oracle':<40} {o_pnl:>+7.1f}% {o_tr:>7} {o_wr:>5.1f}% {o_sw:>9,} {1.0:>5.1f}×")
    print(f"  {'Baseline (threshold 0.5)':<40} {b_pnl:>+7.1f}% {b_tr:>7} {b_wr:>5.1f}% {b_sw:>9,} {b_sw/oracle_switches:>5.1f}×")
    print(f"  {'-' * 78}")

    # Viterbi with different self-transition probabilities
    print(f"\n  --- VITERBI ---")
    for self_trans in [0.90, 0.93, 0.95, 0.97, 0.99]:
        y_vit = viterbi_decode(y_pred_proba, self_trans=self_trans)
        v_pnl, v_tr, v_wr, v_sw = backtest(y_vit, closes_test, fees)
        ratio = v_sw / oracle_switches
        marker = " ***" if v_pnl > 0 else ""
        print(f"  {'Viterbi p=' + str(self_trans):<40} {v_pnl:>+7.1f}% {v_tr:>7} {v_wr:>5.1f}% {v_sw:>9,} {ratio:>5.1f}×{marker}")

    # CUSUM with different thresholds
    print(f"\n  --- CUSUM ---")
    for thr in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
        y_cusum = cusum_filter(y_pred_proba, threshold=thr)
        c_pnl, c_tr, c_wr, c_sw = backtest(y_cusum, closes_test, fees)
        ratio = c_sw / oracle_switches
        marker = " ***" if c_pnl > 0 else ""
        print(f"  {'CUSUM h=' + str(thr):<40} {c_pnl:>+7.1f}% {c_tr:>7} {c_wr:>5.1f}% {c_sw:>9,} {ratio:>5.1f}×{marker}")

    # Combined: Viterbi then CUSUM
    print(f"\n  --- VITERBI + CUSUM ---")
    for self_trans in [0.95, 0.97]:
        for thr in [1.0, 2.0, 3.0]:
            y_vit = viterbi_decode(y_pred_proba, self_trans=self_trans)
            # Convert Viterbi labels back to "soft" probabilities for CUSUM
            # Use original probs but masked by Viterbi state
            y_combined = cusum_filter(y_pred_proba, threshold=thr)
            # Actually: apply CUSUM on Viterbi output probabilities
            vit_probs = np.where(y_vit == 1,
                                  np.maximum(y_pred_proba, 0.5),
                                  np.minimum(y_pred_proba, 0.5))
            y_comb = cusum_filter(vit_probs, threshold=thr)
            co_pnl, co_tr, co_wr, co_sw = backtest(y_comb, closes_test, fees)
            ratio = co_sw / oracle_switches
            marker = " ***" if co_pnl > 0 else ""
            label = f"Vit({self_trans})+CUSUM({thr})"
            print(f"  {label:<40} {co_pnl:>+7.1f}% {co_tr:>7} {co_wr:>5.1f}% {co_sw:>9,} {ratio:>5.1f}×{marker}")

    print(f"  {'-' * 78}")
    print(f"  {'Buy & Hold':<40} {bh:>+7.1f}%")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
