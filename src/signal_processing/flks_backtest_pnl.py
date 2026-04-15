#!/usr/bin/env python3
"""
FLKS Backtest PnL — Oracle, Test 1, Test 2 k=1..6 × MACD/RSI/CCI
==================================================================

Backtest sur les 4000 dernières bougies 30min.
Signal = signe pente FLKS → LONG ou SHORT (toujours en position).
Exécution au open de la bougie suivante après signal disponible.
Frais 0.1% par trade (reversal).

Usage:
    python src/signal_processing/flks_backtest_pnl.py \
        --csv data_trad/BTCUSD_all_5m.csv

Requires: numpy, pandas, matplotlib, pykalman
"""

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# PARAMETERS (from pipeline)
# ============================================================================

KALMAN_PROCESS_VAR = 0.01
KALMAN_MEASURE_VAR = 0.1

A = np.array([[1.0, 1.0],
              [0.0, 1.0]])
H = np.array([[1.0, 0.0]])
Q = np.eye(2) * KALMAN_PROCESS_VAR
R = np.array([[KALMAN_MEASURE_VAR]])

DT_SUB = 1.0 / 6.0
A_SUB = np.array([[1.0, DT_SUB],
                   [0.0, 1.0]])
Q_SUB = Q * DT_SUB

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
CCI_PERIOD = 20

FEES = 0.001  # 0.1% per trade


# ============================================================================
# DATA LOADING (from pipeline)
# ============================================================================

def load_csv(path):
    df = pd.read_csv(path)
    date_col = None
    for col in ['date', 'datetime', 'time', 'timestamp', 'Date', 'Datetime']:
        if col in df.columns:
            date_col = col
            break
    if date_col is None:
        raise ValueError(f"No date column found in {path}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = 'datetime'
    df.columns = df.columns.str.lower()
    return df.sort_index()


def resample_ohlcv(df_5min, tf_minutes):
    return df_5min.resample(f'{tf_minutes}min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()


def compute_bucket_close_mask(index_5min, tf_minutes):
    bucket = index_5min.floor(f'{tf_minutes}min').values
    next_bucket = np.append(bucket[1:], np.datetime64('NaT'))
    return (bucket != next_bucket) | pd.isna(next_bucket)


def compute_live_ohlcv(df_5min, tf_minutes):
    group = df_5min.index.floor(f'{tf_minutes}min')
    r = pd.DataFrame(index=df_5min.index)
    r['open'] = df_5min.groupby(group)['open'].transform('first')
    r['high'] = df_5min.groupby(group)['high'].cummax()
    r['low'] = df_5min.groupby(group)['low'].cummin()
    r['close'] = df_5min['close']
    return r


# ============================================================================
# INDICATORS — Standard 30min
# ============================================================================

def calculate_macd(df):
    ema_f = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_s = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return (line - sig).values.astype(np.float64)


def calculate_rsi(df):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    ag = gain.ewm(span=RSI_PERIOD, adjust=False).mean()
    al = loss.ewm(span=RSI_PERIOD, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).values.astype(np.float64)


def calculate_cci(df):
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(CCI_PERIOD).mean()
    mad = tp.rolling(CCI_PERIOD).apply(lambda x: np.abs(x - x.mean()).mean())
    return ((tp - sma) / (0.015 * mad)).values.astype(np.float64)


# ============================================================================
# INDICATORS — Live frozen/provisional
# ============================================================================

def compute_macd_live(close_5min, is_close):
    n = len(close_5min)
    alpha_f = 2.0 / (MACD_FAST + 1)
    alpha_s = 2.0 / (MACD_SLOW + 1)
    alpha_sig = 2.0 / (MACD_SIGNAL + 1)
    out = np.full(n, np.nan)
    ema_f_cl = ema_s_cl = ema_sig_cl = np.nan
    init = False
    for i in range(n):
        c = close_5min[i]
        if np.isnan(c):
            continue
        if not init:
            if is_close[i]:
                ema_f_cl = c
                ema_s_cl = c
                ema_sig_cl = 0.0
                out[i] = 0.0
                init = True
            continue
        ef = alpha_f * c + (1.0 - alpha_f) * ema_f_cl
        es = alpha_s * c + (1.0 - alpha_s) * ema_s_cl
        ml = ef - es
        esg = alpha_sig * ml + (1.0 - alpha_sig) * ema_sig_cl
        out[i] = ml - esg
        if is_close[i]:
            ema_f_cl = ef
            ema_s_cl = es
            ema_sig_cl = esg
    return out


def compute_rsi_live(close_5min, is_close):
    n = len(close_5min)
    alpha = 2.0 / (RSI_PERIOD + 1)
    out = np.full(n, np.nan)
    closure_indices = []
    closure_closes = []
    for i in range(n):
        if not np.isnan(close_5min[i]) and is_close[i]:
            closure_indices.append(i)
            closure_closes.append(close_5min[i])
    if len(closure_closes) < 2:
        return out
    closes_arr = np.array(closure_closes)
    deltas = np.diff(closes_arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    gains_padded = np.concatenate([[0.0], gains])
    losses_padded = np.concatenate([[0.0], losses])
    ag = gains_padded[0]
    al = losses_padded[0]
    closure_states = [(ag, al, closure_closes[0])]
    for k in range(1, len(gains_padded)):
        ag = alpha * gains_padded[k] + (1.0 - alpha) * ag
        al = alpha * losses_padded[k] + (1.0 - alpha) * al
        closure_states.append((ag, al, closure_closes[k]))
    for k, ci in enumerate(closure_indices):
        ag_k, al_k, _ = closure_states[k]
        if al_k > 1e-15:
            out[ci] = 100.0 - 100.0 / (1.0 + ag_k / al_k)
    closure_set = set(closure_indices)
    current_k = -1
    ag_cl = 0.0
    al_cl = 0.0
    prev_cl = np.nan
    for i in range(n):
        c = close_5min[i]
        if np.isnan(c):
            continue
        if i in closure_set:
            current_k += 1
            ag_cl, al_cl, prev_cl = closure_states[current_k]
            continue
        if current_k >= 0 and not np.isnan(prev_cl):
            delta = c - prev_cl
            gn = max(delta, 0.0)
            ls = max(-delta, 0.0)
            ag_p = alpha * gn + (1.0 - alpha) * ag_cl
            al_p = alpha * ls + (1.0 - alpha) * al_cl
            if al_p > 1e-15:
                out[i] = 100.0 - 100.0 / (1.0 + ag_p / al_p)
    return out


def compute_cci_live(high_live, low_live, close_5min, is_close):
    n = len(close_5min)
    out = np.full(n, np.nan)
    tp_buf = deque(maxlen=CCI_PERIOD - 1)
    for i in range(n):
        c = close_5min[i]
        h = high_live[i]
        lo = low_live[i]
        if np.isnan(c) or np.isnan(h) or np.isnan(lo):
            continue
        tp = (h + lo + c) / 3.0
        if len(tp_buf) >= CCI_PERIOD - 1:
            all_tp = np.array(list(tp_buf) + [tp])
            sma = all_tp.mean()
            mad = np.abs(all_tp - sma).mean()
            out[i] = (tp - sma) / (0.015 * mad) if mad > 1e-15 else 0.0
        if is_close[i]:
            tp_buf.append(tp)
    return out


# ============================================================================
# ORACLE
# ============================================================================

def compute_oracle(indicator_30m):
    from pykalman import KalmanFilter as KF
    n = len(indicator_30m)
    valid = ~np.isnan(indicator_30m)
    if valid.sum() < 3:
        return np.full(n, np.nan)
    vd = indicator_30m[valid]
    kf = KF(
        transition_matrices=[[1, 1], [0, 1]],
        observation_matrices=[[1, 0]],
        initial_state_mean=[vd[0], 0.0],
        initial_state_covariance=np.eye(2),
        observation_covariance=KALMAN_MEASURE_VAR,
        transition_covariance=np.eye(2) * KALMAN_PROCESS_VAR,
    )
    smooth_means, _ = kf.smooth(vd)
    positions = np.full(n, np.nan)
    positions[valid] = smooth_means[:, 0]
    slopes = np.full(n, np.nan)
    for t in range(2, n):
        if not np.isnan(positions[t - 1]) and not np.isnan(positions[t - 2]):
            slopes[t] = positions[t - 1] - positions[t - 2]
    return slopes


# ============================================================================
# KALMAN PRIMITIVES
# ============================================================================

def kf_update(x_p, P_p, z_obs):
    y = z_obs - H @ x_p
    S = H @ P_p @ H.T + R
    K = P_p @ H.T / S[0, 0]
    return x_p + (K @ y).ravel(), (np.eye(2) - K @ H) @ P_p


def kf_predict_sub(x, P):
    return A_SUB @ x, A_SUB @ P @ A_SUB.T + Q_SUB


# ============================================================================
# FORWARD FILTER + RTS GAINS
# ============================================================================

def forward_filter_30m(indicator_30m):
    n = len(indicator_30m)
    first_valid_val = indicator_30m[~np.isnan(indicator_30m)][0]
    x_filt = np.zeros((n, 2))
    P_filt = np.zeros((n, 2, 2))
    x_pred = np.zeros((n, 2))
    P_pred = np.zeros((n, 2, 2))

    for t in range(n):
        if t == 0:
            x_p = np.array([first_valid_val, 0.0])
            P_p = np.eye(2)
        else:
            x_p = A @ x_filt[t - 1]
            P_p = A @ P_filt[t - 1] @ A.T + Q
        x_pred[t] = x_p
        P_pred[t] = P_p
        if np.isnan(indicator_30m[t]):
            x_filt[t] = x_p
            P_filt[t] = P_p
        else:
            x_filt[t], P_filt[t] = kf_update(x_p, P_p, indicator_30m[t])

    C = np.zeros((n, 2, 2))
    for t in range(n - 1):
        P_pk1 = P_pred[t + 1]
        det = P_pk1[0, 0] * P_pk1[1, 1] - P_pk1[0, 1] * P_pk1[1, 0]
        if abs(det) > 1e-15:
            inv_P = np.array([[P_pk1[1, 1], -P_pk1[0, 1]],
                              [-P_pk1[1, 0], P_pk1[0, 0]]]) / det
        else:
            inv_P = np.linalg.pinv(P_pk1)
        C[t] = P_filt[t] @ A.T @ inv_P

    return x_filt, P_filt, x_pred, P_pred, C


# ============================================================================
# SLOPE COMPUTATIONS
# ============================================================================

def compute_slopes_test1(x_filt, x_pred, C):
    n = len(x_filt)
    slopes = np.full(n, np.nan)
    for t in range(2, n):
        sm_t1 = x_filt[t - 1] + C[t - 1] @ (x_filt[t] - x_pred[t])
        sm_t2 = x_filt[t - 2] + C[t - 2] @ (sm_t1 - x_pred[t - 1])
        slopes[t] = sm_t1[0] - sm_t2[0]
    return slopes


def compute_slopes_test2(x_filt, P_filt, x_pred, C,
                          live_per_candle, n_substeps):
    n = len(x_filt)
    slopes = np.full(n, np.nan)
    for t in range(2, n - 1):
        x_cur = x_filt[t].copy()
        P_cur = P_filt[t].copy()

        live_vals = live_per_candle[t + 1]
        valid_vals = [v for v in live_vals if not np.isnan(v)]
        use = valid_vals[:n_substeps]

        if len(use) > 0:
            for m5 in use:
                x_cur, P_cur = kf_predict_sub(x_cur, P_cur)
                x_cur, P_cur = kf_update(x_cur, P_cur, m5)

        x_prov = x_cur
        k_actual = len(use) if len(use) > 0 else 1

        # Pas 1 : lisser t avec x_prov
        A_k = np.linalg.matrix_power(A_SUB, k_actual)
        Q_k = Q_SUB * k_actual
        x_pred_partial = A_k @ x_filt[t]
        P_pred_partial = A_k @ P_filt[t] @ A_k.T + Q_k
        det = P_pred_partial[0, 0] * P_pred_partial[1, 1] - P_pred_partial[0, 1] * P_pred_partial[1, 0]
        if abs(det) > 1e-15:
            inv_Pp = np.array([[P_pred_partial[1, 1], -P_pred_partial[0, 1]],
                               [-P_pred_partial[1, 0], P_pred_partial[0, 0]]]) / det
        else:
            inv_Pp = np.linalg.pinv(P_pred_partial)
        C_partial = P_filt[t] @ A_k.T @ inv_Pp
        sm_t = x_filt[t] + C_partial @ (x_prov - x_pred_partial)

        # Pas 2 : lisser t-1
        sm_t1 = x_filt[t - 1] + C[t - 1] @ (sm_t - x_pred[t])

        # Pas 3 : lisser t-2
        sm_t2 = x_filt[t - 2] + C[t - 2] @ (sm_t1 - x_pred[t - 1])

        slopes[t] = sm_t1[0] - sm_t2[0]
    return slopes


# ============================================================================
# BACKTEST
# ============================================================================

def _exec_trade(position, entry_price, exec_price, fees):
    """Close existing position at exec_price, return trade PnL."""
    if position == 1:
        pnl = (exec_price - entry_price) / entry_price
    else:
        pnl = (entry_price - exec_price) / entry_price
    return pnl - fees


def backtest_30m(slopes, closes_30m, start, end, fees, label=""):
    """
    Backtest pour Oracle et Test 1.
    Signal slope[t] disponible à close de t.
    Exécution au close[t] ≈ open[t+1].
    """
    pnl_total = 0.0
    n_trades = 0
    n_wins = 0
    position = 0
    entry_price = 0.0

    for t in range(start, end):
        if np.isnan(slopes[t]):
            continue
        target = 1 if slopes[t] > 0 else -1
        if position == target:
            continue

        # Prix d'exécution = close[t] ≈ open[t+1]
        if t + 1 >= len(closes_30m):
            continue
        exec_price = closes_30m[t]
        if np.isnan(exec_price):
            continue

        # Clôturer position existante
        if position != 0:
            trade_pnl = _exec_trade(position, entry_price, exec_price, fees)
            pnl_total += trade_pnl
            if trade_pnl > 0:
                n_wins += 1

        # Ouvrir nouvelle position
        entry_price = exec_price
        position = target
        n_trades += 1
        pnl_total -= fees

    # Clôturer dernière position
    if position != 0 and end < len(closes_30m):
        exec_price = closes_30m[min(end, len(closes_30m) - 1)]
        if not np.isnan(exec_price):
            trade_pnl = _exec_trade(position, entry_price, exec_price, fees)
            pnl_total += trade_pnl
            if trade_pnl > 0:
                n_wins += 1

    wr = (n_wins / n_trades * 100.0) if n_trades > 0 else 0.0
    return {'label': label, 'pnl_pct': pnl_total * 100,
            'trades': n_trades, 'win_rate': wr}


def backtest_5m(slopes, closes_5m_per_candle, k_substep, start, end, fees, label=""):
    """
    Backtest pour Test 2 k=1..6.
    Signal slope[t] disponible au step k de la bougie t+1.
    Exécution au close du step k de la bougie t+1.

    k=1: signal à 5min dans t+1 → exécution close step 1 (= close 5min)
    k=6: signal à 30min dans t+1 → exécution close step 6 (= close 30min de t+1)
    """
    pnl_total = 0.0
    n_trades = 0
    n_wins = 0
    position = 0
    entry_price = 0.0

    for t in range(start, end):
        if np.isnan(slopes[t]):
            continue
        target = 1 if slopes[t] > 0 else -1
        if position == target:
            continue

        # Prix d'exécution = close du step k dans la bougie t+1
        candle_idx = t + 1
        if candle_idx >= len(closes_5m_per_candle):
            continue
        closes_5m = closes_5m_per_candle[candle_idx]
        step_idx = k_substep - 1  # k=1 → index 0, k=6 → index 5
        if step_idx >= len(closes_5m):
            continue
        exec_price = closes_5m[step_idx]
        if np.isnan(exec_price):
            continue

        # Clôturer position existante
        if position != 0:
            trade_pnl = _exec_trade(position, entry_price, exec_price, fees)
            pnl_total += trade_pnl
            if trade_pnl > 0:
                n_wins += 1

        # Ouvrir nouvelle position
        entry_price = exec_price
        position = target
        n_trades += 1
        pnl_total -= fees

    # Clôturer dernière position
    if position != 0:
        last_candle = min(end, len(closes_5m_per_candle) - 1)
        closes_last = closes_5m_per_candle[last_candle]
        if len(closes_last) > 0 and not np.isnan(closes_last[-1]):
            trade_pnl = _exec_trade(position, entry_price, closes_last[-1], fees)
            pnl_total += trade_pnl
            if trade_pnl > 0:
                n_wins += 1

    wr = (n_wins / n_trades * 100.0) if n_trades > 0 else 0.0
    return {'label': label, 'pnl_pct': pnl_total * 100,
            'trades': n_trades, 'win_rate': wr}


def buy_and_hold(closes_30m, start, end):
    c = closes_30m[start:end + 1]
    valid = c[~np.isnan(c)]
    if len(valid) < 2:
        return 0.0
    return (valid[-1] - valid[0]) / valid[0] * 100


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='FLKS Backtest PnL — Oracle, T1, T2 k=1..6')
    parser.add_argument('--csv', type=str, default='data_trad/BTCUSD_all_5m.csv')
    parser.add_argument('--n-candles-30m', type=int, default=5000)
    parser.add_argument('--eval-start', type=int, default=1000,
                        help='Start of eval window (last 4000 candles)')
    parser.add_argument('--fees', type=float, default=0.001,
                        help='Fees per trade (default 0.1%%)')
    parser.add_argument('--output-dir', type=str, default='plots')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n30 = args.n_candles_30m
    fees = args.fees

    # ==================================================================
    print(f"[1/8] Loading {args.csv} ...")
    df_5m = load_csv(args.csv)
    print(f"       {len(df_5m):,} 5min candles")

    # ==================================================================
    print("[2/8] Resampling to 30min ...")
    df_30m = resample_ohlcv(df_5m, 30)
    if len(df_30m) > n30:
        df_30m = df_30m.iloc[-n30:]
    df_5m = df_5m.loc[df_30m.index[0]:df_30m.index[-1] + pd.Timedelta(minutes=29)]
    closes_30m = df_30m['close'].values.astype(np.float64)
    print(f"       {len(df_30m):,} bougies 30min, {len(df_5m):,} bougies 5min")
    print(f"       Eval: [{args.eval_start}:{n30}] = {n30 - args.eval_start} bougies")

    # ==================================================================
    print("[3/8] Computing indicators ...")
    is_close = compute_bucket_close_mask(df_5m.index, 30)
    close_5m = df_5m['close'].values.astype(np.float64)

    indicators = {}

    # MACD
    indicators['MACD'] = {
        'std': calculate_macd(df_30m),
        'live': compute_macd_live(close_5m, is_close),
    }
    # RSI
    indicators['RSI'] = {
        'std': calculate_rsi(df_30m),
        'live': compute_rsi_live(close_5m, is_close),
    }
    # CCI
    live_ohlcv = compute_live_ohlcv(df_5m, 30)
    indicators['CCI'] = {
        'std': calculate_cci(df_30m),
        'live': compute_cci_live(
            live_ohlcv['high'].values.astype(np.float64),
            live_ohlcv['low'].values.astype(np.float64),
            close_5m, is_close),
    }
    print("       MACD, RSI, CCI computed.")

    # ==================================================================
    print("[4/8] Grouping live values + closes 5min per 30min candle ...")
    closes_5m_per_candle = []
    for ts_30m in df_30m.index:
        bucket_end = ts_30m + pd.Timedelta(minutes=29, seconds=59)
        mask = (df_5m.index >= ts_30m) & (df_5m.index <= bucket_end)
        closes_5m_per_candle.append(close_5m[mask])
    for name in indicators:
        live_arr = indicators[name]['live']
        per_candle = []
        for ts_30m in df_30m.index:
            bucket_end = ts_30m + pd.Timedelta(minutes=29, seconds=59)
            mask = (df_5m.index >= ts_30m) & (df_5m.index <= bucket_end)
            per_candle.append(live_arr[mask])
        indicators[name]['live_pc'] = per_candle
    print(f"       Done. Closes 5min per candle: {len(closes_5m_per_candle)}")

    # ==================================================================
    print("[5/8] Buy & Hold ...")
    bh = buy_and_hold(closes_30m, args.eval_start, n30 - 1)
    print(f"       Buy & Hold: {bh:+.2f}%")

    # ==================================================================
    print("[6/8] Computing slopes + backtest for each indicator ...")

    all_results = {}

    for name in ['MACD', 'RSI', 'CCI']:
        print(f"\n  --- {name} ---")
        ind_30m = indicators[name]['std']
        live_pc = indicators[name]['live_pc']

        # Forward filter
        print(f"  Forward filter ...", end=" ", flush=True)
        x_filt, P_filt, x_pred, P_pred, C = forward_filter_30m(ind_30m)
        print("done.")

        results = []

        # Oracle
        print(f"  Oracle ...", end=" ", flush=True)
        slopes_oracle = compute_oracle(ind_30m)
        r = backtest_30m(slopes_oracle, closes_30m, args.eval_start, n30 - 1, fees, "Oracle")
        results.append(r)
        print(f"PnL={r['pnl_pct']:+.1f}%  trades={r['trades']}  WR={r['win_rate']:.1f}%")

        # Test 1
        print(f"  Test 1 (30m pur) ...", end=" ", flush=True)
        slopes_t1 = compute_slopes_test1(x_filt, x_pred, C)
        r = backtest_30m(slopes_t1, closes_30m, args.eval_start, n30 - 1, fees, "T1: 30m pur")
        results.append(r)
        print(f"PnL={r['pnl_pct']:+.1f}%  trades={r['trades']}  WR={r['win_rate']:.1f}%")

        # Test 2 k=1..6
        for k in range(1, 7):
            print(f"  Test 2 k={k} ({k*5}min) ...", end=" ", flush=True)
            slopes_k = compute_slopes_test2(x_filt, P_filt, x_pred, C, live_pc, k)
            r = backtest_5m(slopes_k, closes_5m_per_candle, k,
                            args.eval_start, n30 - 1, fees, f"T2: k={k} ({k*5}min)")
            results.append(r)
            print(f"PnL={r['pnl_pct']:+.1f}%  trades={r['trades']}  WR={r['win_rate']:.1f}%")

        all_results[name] = results

    # ==================================================================
    print(f"\n[7/8] Tableau comparatif")
    print(f"{'=' * 95}")
    print(f"  FLKS Backtest PnL — BTC 30min — [{args.eval_start}:{n30}]"
          f" = {n30 - args.eval_start} bougies — Fees {fees*100:.1f}%/trade")
    print(f"  Buy & Hold: {bh:+.2f}%")
    print(f"{'=' * 95}")

    header = f"  {'Méthode':<20}"
    for name in ['MACD', 'RSI', 'CCI']:
        header += f" │ {'PnL':>8} {'Trades':>7} {'WR':>6}"
    print(header)
    print(f"  {'-' * 90}")

    n_rows = len(all_results['MACD'])
    for i in range(n_rows):
        label = all_results['MACD'][i]['label']
        row = f"  {label:<20}"
        for name in ['MACD', 'RSI', 'CCI']:
            r = all_results[name][i]
            row += f" │ {r['pnl_pct']:>+7.1f}% {r['trades']:>7} {r['win_rate']:>5.1f}%"
        print(row)

    print(f"  {'-' * 90}")
    row = f"  {'Buy & Hold':<20}"
    for _ in ['MACD', 'RSI', 'CCI']:
        row += f" │ {bh:>+7.1f}% {'1':>7} {'—':>6}"
    print(row)
    print(f"{'=' * 95}")

    # ==================================================================
    print(f"\n[8/8] Plot ...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for idx, name in enumerate(['MACD', 'RSI', 'CCI']):
        ax = axes[idx]
        results = all_results[name]
        labels = [r['label'] for r in results]
        pnls = [r['pnl_pct'] for r in results]

        colors = ['gold'] + ['gray'] + [plt.cm.Blues(0.3 + 0.1 * k) for k in range(6)]
        bars = ax.bar(range(len(labels)), pnls, color=colors, edgecolor='white')
        ax.axhline(y=bh, color='green', linestyle='--', linewidth=1, label=f'B&H {bh:+.1f}%')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace('T2: ', '').replace('T1: ', '')
                            for l in labels], rotation=45, ha='right', fontsize=7)
        ax.set_title(name, fontsize=12)
        ax.grid(True, axis='y', alpha=0.3)
        if idx == 0:
            ax.set_ylabel('PnL net (%)')
            ax.legend(fontsize=8)

        for bar, pnl in zip(bars, pnls):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{pnl:+.0f}%', ha='center',
                    va='bottom' if pnl >= 0 else 'top', fontsize=7)

    plt.suptitle(f'FLKS Backtest PnL — BTC 30min — Fees {fees*100:.1f}%/trade', fontsize=13)
    plt.tight_layout()
    out_path = output_dir / 'flks_backtest_pnl.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
