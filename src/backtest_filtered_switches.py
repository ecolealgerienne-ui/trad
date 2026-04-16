#!/usr/bin/env python3
"""
Backtest PnL with filtered switches — velocity + macd_live thresholds
=====================================================================

Uses XGBoost/LSTM predictions + feature-based switch filter.
Model only reverses if |velocity| > vel_thr AND |macd_live| > macd_thr.

Grid search: vel_thr × macd_thr × holding_min.
Compares with baseline (no filter) and Buy & Hold.

Usage:
    python src/backtest_filtered_switches.py --indicator macd --timeframe 30m
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
FEES = 0.001  # 0.1% per trade


def apply_filter_and_trade(y_pred_proba, vel_values, macd_values, closes,
                            pred_threshold, vel_thr, macd_thr, holding_min,
                            fees):
    """
    Apply switch filter + backtest in one pass.
    Entry/exit at close of current candle (≈ open next).
    """
    n = len(y_pred_proba)
    y_raw = (y_pred_proba > pred_threshold).astype(int)

    pnl_total = 0.0
    n_trades = 0
    n_wins = 0
    position = 0  # 0=flat, +1=long, -1=short
    entry_price = 0.0
    entry_t = -holding_min
    prev_label = y_raw[0]

    for i in range(1, n):
        # Determine filtered label
        target_raw = 1 if y_raw[i] == 1 else -1

        if y_raw[i] != prev_label:
            # Model wants to switch — check filter conditions
            vel_ok = abs(vel_values[i]) > vel_thr if vel_thr > 0 else True
            macd_ok = abs(macd_values[i]) > macd_thr if macd_thr > 0 else True

            if vel_ok and macd_ok:
                prev_label = y_raw[i]  # accept switch
            # else: keep prev_label (block switch)

        target = 1 if prev_label == 1 else -1

        if position == target:
            continue
        if position != 0 and (i - entry_t) < holding_min:
            continue

        exec_price = closes[i]
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
        entry_t = i
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
    return pnl_total * 100, n_trades, wr


def main():
    parser = argparse.ArgumentParser(description='Backtest with filtered switches')
    parser.add_argument('--indicator', default='macd')
    parser.add_argument('--timeframe', default='30m')
    parser.add_argument('--pred-threshold', type=float, default=0.5)
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

    # Load CSV for features + prices
    base = ASSET_CSV_MAP['BTC']
    csv_path = f'{PREPARED_DATA_DIR}/{base}_multitf_macd_rsi_cci.csv'
    df = pd.read_csv(csv_path, parse_dates=['datetime']).set_index('datetime').sort_index()

    vel_col = f'{ind}_{tf}_velocity'
    live_col = f'{ind}_{tf}_live'
    close_col = 'close'

    df_clean = df.dropna(subset=[vel_col, live_col, close_col])
    vel_test = df_clean[vel_col].values[-n_test:]
    macd_test = df_clean[live_col].values[-n_test:]
    closes_test = df_clean[close_col].values[-n_test:]

    # Buy & Hold
    bh = (closes_test[-1] - closes_test[0]) / closes_test[0] * 100

    # Percentiles for threshold calibration
    vel_abs = np.abs(vel_test[~np.isnan(vel_test)])
    macd_abs = np.abs(macd_test[~np.isnan(macd_test)])
    vel_p50 = round(np.percentile(vel_abs, 50), 2)
    macd_p25 = round(np.percentile(macd_abs, 25), 2)

    # Grid
    vel_thresholds = [0, round(vel_p50 * 0.5, 2), vel_p50]
    macd_thresholds = [0, round(macd_p25 * 0.5, 2), macd_p25]
    holding_values = [0, 4, 8]

    logger.info(f"\n{'=' * 90}")
    logger.info(f"  BACKTEST FILTERED SWITCHES — {ind.upper()}_{tf}")
    logger.info(f"  Test: {n_test:,} samples | Fees: {fees*100:.1f}%/trade | B&H: {bh:+.1f}%")
    logger.info(f"  vel_thr: {vel_thresholds} | macd_thr: {macd_thresholds} | hold: {holding_values}")
    logger.info(f"{'=' * 90}")

    print(f"\n  {'vel':>6} {'macd':>7} {'hold':>5} │ {'PnL':>8} {'Trades':>7} {'WR':>6}")
    print(f"  {'-' * 50}")

    best_pnl = -1e9
    best_cfg = ""

    for v_thr in vel_thresholds:
        for m_thr in macd_thresholds:
            for hold in holding_values:
                pnl, trades, wr = apply_filter_and_trade(
                    y_pred_proba, vel_test, macd_test, closes_test,
                    args.pred_threshold, v_thr, m_thr, hold, fees)

                marker = ""
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_cfg = f"vel={v_thr} macd={m_thr} hold={hold}"
                    marker = " ***"

                print(f"  {v_thr:>6.1f} {m_thr:>7.1f} {hold:>5} │ "
                      f"{pnl:>+7.1f}% {trades:>7} {wr:>5.1f}%{marker}")

    print(f"  {'-' * 50}")
    print(f"  B&H: {bh:+.1f}%")
    print(f"\n  BEST: {best_cfg} → PnL={best_pnl:+.1f}%")
    print(f"{'=' * 90}")


if __name__ == '__main__':
    main()
