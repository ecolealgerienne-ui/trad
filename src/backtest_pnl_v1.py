#!/usr/bin/env python3
"""
PnL Backtest v1 — 4 signal configurations + buy & hold baseline on BTC.

Configs:
  1. Ultra-conservative: Unanimous vote (LSTM + GRU + TCN) → flat if no agreement
  2. Conservative: R_strong_agree (binary ∩ regression magnitude) → hold if no agreement
  3. Moderate: Majority 2/3 vote → flat if no majority
  4. Aggressive: Crossfeat simple CNN-LSTM → always in position

Baseline: Buy & Hold BTC over the test period.

Usage:
    python src/backtest_pnl_v1.py
"""

import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from constants import PREPARED_DATA_DIR

# Fees
FEE_PER_SIDE = 0.0007   # 0.07% per side (0.05% fee + 0.02% slippage)
FEE_ROUND_TRIP = 2 * FEE_PER_SIDE  # 0.14%

# Positions
LONG = 1
SHORT = -1
FLAT = 0


def load_predictions(model_name):
    """Load test predictions from NPZ."""
    path = f'{PREPARED_DATA_DIR}/{model_name}_dataset.npz'
    if not Path(path).exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        'y_test': data['y_test'],
        'y_test_pred': data['y_test_pred'],
    }


def load_price_data():
    """Load BTC 5min OHLCV from the multitf CSV for the test period."""
    csv_path = f'{PREPARED_DATA_DIR}/BTCUSD_multitf_macd_rsi_cci.csv'
    df = pd.read_csv(csv_path, parse_dates=['datetime']).set_index('datetime')

    # Split: test = last 15%
    n = len(df)
    test_start = int(n * 0.85)
    df_test = df.iloc[test_start:].copy()

    return df_test


def build_signals():
    """Build signal arrays for all 4 configs."""
    # Load predictions
    lstm = load_predictions('macd_30m_crossfeat')
    gru = load_predictions('macd_30m_crossfeat_cnngru')
    tcn = load_predictions('macd_30m_crossfeat_tcn')
    regression = load_predictions('macd_30m_crossfeat_regression')

    if not all([lstm, gru, tcn, regression]):
        missing = []
        if not lstm: missing.append('LSTM')
        if not gru: missing.append('GRU')
        if not tcn: missing.append('TCN')
        if not regression: missing.append('Regression')
        raise FileNotFoundError(f"Missing models: {missing}")

    # Align lengths
    min_n = min(len(lstm['y_test_pred']), len(gru['y_test_pred']),
                len(tcn['y_test_pred']), len(regression['y_test_pred']))

    lstm_bin = (lstm['y_test_pred'][:min_n] > 0.5).astype(int)
    gru_bin = (gru['y_test_pred'][:min_n] > 0.5).astype(int)
    tcn_bin = (tcn['y_test_pred'][:min_n] > 0.5).astype(int)
    reg_raw = regression['y_test_pred'][:min_n]
    oracle = lstm['y_test'][:min_n]

    # Median true magnitude for R_strong_agree threshold
    oracle_trans = np.where(np.diff(oracle) != 0)[0] + 1
    true_mags = [abs(reg_raw[t]) for t in oracle_trans if t < min_n]
    median_mag = np.median(true_mags) if true_mags else 0.1

    signals = {}

    # Config 1: Ultra-conservative (unanimous 3 archi, flat if disagreement)
    sig1 = np.full(min_n, FLAT)
    for i in range(min_n):
        if lstm_bin[i] == gru_bin[i] == tcn_bin[i]:
            sig1[i] = LONG if lstm_bin[i] == 1 else SHORT
        # else: FLAT
    signals['Ultra-conservative'] = sig1

    # Config 2: Conservative (R_strong_agree, hold if no agreement)
    sig2 = np.full(min_n, FLAT)
    current = FLAT
    for i in range(min_n):
        bin_dir = LONG if lstm_bin[i] == 1 else SHORT
        reg_dir = LONG if reg_raw[i] > 0 else SHORT
        if bin_dir == reg_dir and abs(reg_raw[i]) > median_mag:
            current = bin_dir
        sig2[i] = current
    signals['Conservative'] = sig2

    # Config 3: Moderate (majority 2/3, flat if no majority)
    sig3 = np.full(min_n, FLAT)
    for i in range(min_n):
        votes = lstm_bin[i] + gru_bin[i] + tcn_bin[i]
        if votes >= 2:
            sig3[i] = LONG
        elif votes <= 0:
            sig3[i] = SHORT
        # votes == 1: no clear majority → FLAT
    signals['Moderate'] = sig3

    # Config 4: Aggressive (crossfeat simple, always in position)
    sig4 = np.full(min_n, FLAT)
    for i in range(min_n):
        sig4[i] = LONG if lstm_bin[i] == 1 else SHORT
    signals['Aggressive'] = sig4

    return signals, oracle, min_n, median_mag


def backtest_signal(signal, opens, closes, timestamps):
    """
    Run backtest on a signal array.

    Signal: LONG (1), SHORT (-1), FLAT (0) at each step.
    Execution: when signal changes, trade at open of next bar.
    """
    n = len(signal)
    trades = []
    position = FLAT
    entry_price = 0.0
    entry_idx = 0

    pnl_curve = np.zeros(n)
    cumulative_pnl = 0.0

    for i in range(1, n):
        # Mark-to-market (unrealized PnL for equity curve)
        if position != FLAT:
            if position == LONG:
                step_return = (closes[i] - closes[i-1]) / closes[i-1]
            else:
                step_return = (closes[i-1] - closes[i]) / closes[i-1]
            cumulative_pnl += step_return
        pnl_curve[i] = cumulative_pnl

        # Check for signal change
        if signal[i] != signal[i-1]:
            # Close existing position
            if position != FLAT:
                exit_price = opens[i]
                if position == LONG:
                    trade_pnl = (exit_price - entry_price) / entry_price
                else:
                    trade_pnl = (entry_price - exit_price) / entry_price

                trade_pnl_net = trade_pnl - FEE_ROUND_TRIP
                trades.append({
                    'entry_idx': entry_idx, 'exit_idx': i,
                    'duration': i - entry_idx,
                    'position': 'LONG' if position == LONG else 'SHORT',
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'pnl_gross': trade_pnl, 'pnl_net': trade_pnl_net,
                    'entry_ts': float(timestamps[entry_idx]),
                    'exit_ts': float(timestamps[i]),
                })

            # Open new position
            if signal[i] != FLAT:
                position = signal[i]
                entry_price = opens[i]
                entry_idx = i
            else:
                position = FLAT

    # Close final position
    if position != FLAT:
        exit_price = closes[-1]
        if position == LONG:
            trade_pnl = (exit_price - entry_price) / entry_price
        else:
            trade_pnl = (entry_price - exit_price) / entry_price
        trade_pnl_net = trade_pnl - FEE_ROUND_TRIP
        trades.append({
            'entry_idx': entry_idx, 'exit_idx': n-1,
            'duration': n-1 - entry_idx,
            'position': 'LONG' if position == LONG else 'SHORT',
            'entry_price': entry_price, 'exit_price': closes[-1],
            'pnl_gross': trade_pnl, 'pnl_net': trade_pnl_net,
            'entry_ts': float(timestamps[entry_idx]),
            'exit_ts': float(timestamps[n-1]),
        })

    return trades, pnl_curve


def compute_metrics(trades, pnl_curve):
    """Compute backtest metrics from trades and equity curve."""
    if not trades:
        return {k: 0 for k in ['pnl_gross', 'pnl_net', 'fees', 'n_trades',
                                'win_rate', 'avg_pnl', 'median_pnl',
                                'sharpe', 'max_dd', 'avg_duration', 'fee_ratio']}

    pnls_gross = np.array([t['pnl_gross'] for t in trades])
    pnls_net = np.array([t['pnl_net'] for t in trades])
    durations = np.array([t['duration'] for t in trades])

    total_gross = pnls_gross.sum()
    total_fees = len(trades) * FEE_ROUND_TRIP
    total_net = pnls_net.sum()

    wins = (pnls_net > 0).sum()
    win_rate = wins / len(trades) if trades else 0

    # Sharpe: annualized (252 trading days × 288 5min bars per day)
    if len(pnls_net) > 1 and pnls_net.std() > 0:
        sharpe = (pnls_net.mean() / pnls_net.std()) * np.sqrt(252 * 288)
    else:
        sharpe = 0

    # Max drawdown from equity curve
    running_max = np.maximum.accumulate(pnl_curve)
    drawdowns = running_max - pnl_curve
    max_dd = drawdowns.max()

    fee_ratio = total_fees / total_gross if total_gross > 0 else float('inf')

    return {
        'pnl_gross': float(total_gross),
        'pnl_net': float(total_net),
        'fees': float(total_fees),
        'n_trades': len(trades),
        'win_rate': float(win_rate),
        'avg_pnl': float(pnls_net.mean()),
        'median_pnl': float(np.median(pnls_net)),
        'sharpe': float(sharpe),
        'max_dd': float(max_dd),
        'avg_duration': float(durations.mean()),
        'fee_ratio': float(fee_ratio),
    }


def compute_yearly(trades):
    """Split PnL by year."""
    from datetime import datetime
    yearly = defaultdict(lambda: {'pnl_net': 0, 'n_trades': 0, 'wins': 0})
    for t in trades:
        ts = t['entry_ts']
        if ts > 1e18: ts /= 1e9
        elif ts > 1e15: ts /= 1e6
        elif ts > 1e12: ts /= 1e3
        try:
            year = str(datetime.fromtimestamp(ts).year)
        except:
            continue
        yearly[year]['pnl_net'] += t['pnl_net']
        yearly[year]['n_trades'] += 1
        if t['pnl_net'] > 0:
            yearly[year]['wins'] += 1
    return dict(yearly)


def main():
    print("=" * 90)
    print("PNL BACKTEST v1 — BTC, MACD 30m, 4 configurations")
    print("=" * 90)
    print(f"Fees: {FEE_PER_SIDE*100:.2f}% per side ({FEE_ROUND_TRIP*100:.2f}% round-trip)")

    # Build signals
    signals, oracle, n_signals, median_mag = build_signals()
    print(f"Signal length: {n_signals:,} steps")
    print(f"R_strong_agree median magnitude threshold: {median_mag:.4f}")

    # Load price data
    df_test = load_price_data()
    opens = df_test['open'].values
    closes = df_test['close'].values
    timestamps = df_test.index.astype(np.int64) / 1e9

    # Align price data with signal length
    min_len = min(n_signals, len(opens))
    opens = opens[:min_len]
    closes = closes[:min_len]
    timestamps = timestamps[:min_len]

    # Trim signals
    for k in signals:
        signals[k] = signals[k][:min_len]

    print(f"Price data aligned: {min_len:,} bars")
    print(f"Period: {df_test.index[0]} → {df_test.index[min_len-1]}")

    # Buy & Hold baseline
    bh_return = (closes[-1] - opens[0]) / opens[0]

    # Run backtests
    all_results = {}

    print(f"\n{'='*90}")
    print(f"{'Config':<22} {'PnL Gross':>10} {'Fees':>8} {'PnL Net':>10} {'Trades':>7} "
          f"{'WR':>6} {'Sharpe':>7} {'MaxDD':>7} {'AvgDur':>7} {'Fee%':>6}")
    print("-" * 90)

    for config_name, signal in signals.items():
        trades, pnl_curve = backtest_signal(signal, opens, closes, timestamps)
        metrics = compute_metrics(trades, pnl_curve)
        yearly = compute_yearly(trades)

        all_results[config_name] = {
            'metrics': metrics,
            'yearly': yearly,
            'n_flat_bars': int((signal == FLAT).sum()),
            'pct_flat': float((signal == FLAT).mean() * 100),
        }

        print(f"{config_name:<22} {metrics['pnl_gross']*100:>+9.2f}% "
              f"{metrics['fees']*100:>7.2f}% {metrics['pnl_net']*100:>+9.2f}% "
              f"{metrics['n_trades']:>7,} {metrics['win_rate']*100:>5.1f}% "
              f"{metrics['sharpe']:>7.2f} {metrics['max_dd']*100:>6.2f}% "
              f"{metrics['avg_duration']:>6.1f}p {metrics['fee_ratio']*100:>5.1f}%")

    # Buy & Hold line
    print(f"{'Buy & Hold':<22} {bh_return*100:>+9.2f}% {'0.14':>7}% {(bh_return-FEE_ROUND_TRIP)*100:>+9.2f}% "
          f"{'1':>7} {'N/A':>6} {'N/A':>7} {'N/A':>7} {min_len:>6.0f}p {'N/A':>6}")

    # Yearly breakdown
    print(f"\n{'='*90}")
    print(f"YEARLY BREAKDOWN (PnL Net %)")
    print(f"{'='*90}")

    years = sorted(set(y for r in all_results.values() for y in r['yearly']))
    header = f"{'Config':<22}" + "".join(f"{y:>10}" for y in years) + f"{'Total':>10}"
    print(header)
    print("-" * (22 + 10 * (len(years) + 1)))

    for config_name, result in all_results.items():
        row = f"{config_name:<22}"
        for y in years:
            if y in result['yearly']:
                row += f"{result['yearly'][y]['pnl_net']*100:>+9.2f}%"
            else:
                row += f"{'N/A':>10}"
        row += f"{result['metrics']['pnl_net']*100:>+9.2f}%"
        print(row)

    # Flat time analysis
    print(f"\n{'='*90}")
    print(f"POSITION ANALYSIS")
    print(f"{'='*90}")
    print(f"{'Config':<22} {'Flat %':>8} {'Flat bars':>10} {'In position':>12}")
    print("-" * 55)
    for config_name, result in all_results.items():
        in_pos = 100 - result['pct_flat']
        print(f"{config_name:<22} {result['pct_flat']:>7.1f}% {result['n_flat_bars']:>10,} {in_pos:>11.1f}%")

    # Verdict
    print(f"\n{'='*90}")
    print(f"VERDICT")
    print(f"{'='*90}")

    profitable = {k: v for k, v in all_results.items() if v['metrics']['pnl_net'] > 0}
    beats_bh = {k: v for k, v in all_results.items() if v['metrics']['pnl_net'] > bh_return}

    if profitable:
        best = max(profitable.items(), key=lambda x: x[1]['metrics']['sharpe'])
        print(f"  ✅ {len(profitable)}/4 configs are profitable after fees")
        print(f"  Best Sharpe: {best[0]} (Sharpe={best[1]['metrics']['sharpe']:.2f}, "
              f"PnL={best[1]['metrics']['pnl_net']*100:+.2f}%)")
    else:
        print(f"  ❌ 0/4 configs are profitable after fees")

    if beats_bh:
        print(f"  ✅ {len(beats_bh)}/4 configs beat Buy & Hold ({bh_return*100:+.2f}%)")
    else:
        print(f"  ❌ 0/4 configs beat Buy & Hold ({bh_return*100:+.2f}%)")

    # Save
    output = {
        'configs': all_results,
        'buy_hold_return': float(bh_return),
        'period': f"{df_test.index[0]} to {df_test.index[min_len-1]}",
        'n_bars': min_len,
        'fees': {'per_side': FEE_PER_SIDE, 'round_trip': FEE_ROUND_TRIP},
        'median_mag_threshold': float(median_mag),
    }

    json_path = 'models/pnl_backtest_v1.json'
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {json_path}")


if __name__ == '__main__':
    main()
