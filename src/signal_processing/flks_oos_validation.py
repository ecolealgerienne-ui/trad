#!/usr/bin/env python3
"""
FLKS Out-of-Sample Validation — 2 configs gagnantes, paramètres fixés
=====================================================================

Teste les 2 meilleures configs de la session sur des périodes séparées :
  Config 1: AQ-KF k=6, hold=8, thr=22.0 (P75)
  Config 2: Standard T2 k=1, hold=8, thr=36.8 (P90)

Périodes :
  - In-sample  [1000:5000] (là où les paramètres ont été optimisés)
  - OOS-early  [0:1000]    (avant la période d'optimisation)
  - OOS-next   [5000:10000] (5000 bougies suivantes dans le CSV)

Paramètres fixés, pas de re-sweep.

Usage:
    python src/signal_processing/flks_oos_validation.py --csv data_trad/BTCUSD_all_5m.csv
"""

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# PARAMETERS
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

# Fixed winning configs
CONFIG_AQ = {'name': 'AQ:k=6 hold=8 thr=22.0', 'k': 6, 'hold': 8, 'thr': 22.0, 'adaptive': True}
CONFIG_STD = {'name': 'Std:k=1 hold=8 thr=36.8', 'k': 1, 'hold': 8, 'thr': 36.8, 'adaptive': False}


# ============================================================================
# DATA LOADING
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


def calculate_macd(df):
    ema_f = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_s = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return (line - sig).values.astype(np.float64)


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


# ============================================================================
# ORACLE
# ============================================================================

def compute_oracle_slopes(indicator_30m):
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


def _inv2x2(M):
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if abs(det) > 1e-15:
        return np.array([[M[1, 1], -M[0, 1]],
                         [-M[1, 0], M[0, 0]]]) / det
    return np.linalg.pinv(M)


def _is_pos_semidef(M):
    return M[0, 0] >= 0 and M[1, 1] >= 0 and (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]) >= -1e-12


# ============================================================================
# FORWARD FILTERS
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
        C[t] = P_filt[t] @ A.T @ _inv2x2(P_pred[t + 1])
    return x_filt, P_filt, x_pred, P_pred, C


def forward_filter_30m_adaptive(indicator_30m, window=30, Q_max_factor=10.0):
    n = len(indicator_30m)
    first_valid_val = indicator_30m[~np.isnan(indicator_30m)][0]
    x_filt = np.zeros((n, 2))
    P_filt = np.zeros((n, 2, 2))
    x_pred = np.zeros((n, 2))
    P_pred = np.zeros((n, 2, 2))
    Q_current = Q.copy()
    innovation_buffer = []
    Q_FLOOR = Q * 0.1
    Q_CEIL = Q * Q_max_factor
    for t in range(n):
        if t == 0:
            x_p = np.array([first_valid_val, 0.0])
            P_p = np.eye(2)
        else:
            x_p = A @ x_filt[t - 1]
            P_p = A @ P_filt[t - 1] @ A.T + Q_current
        x_pred[t] = x_p
        P_pred[t] = P_p
        if np.isnan(indicator_30m[t]):
            x_filt[t] = x_p
            P_filt[t] = P_p
            continue
        S_t = (H @ P_p @ H.T + R)[0, 0]
        x_filt[t], P_filt[t] = kf_update(x_p, P_p, indicator_30m[t])
        v_t = indicator_30m[t] - (H @ x_p)[0]
        innovation_buffer.append(v_t)
        if len(innovation_buffer) > window:
            innovation_buffer.pop(0)
        if len(innovation_buffer) >= window and t > 0:
            C_vv = np.mean(np.array(innovation_buffer) ** 2)
            delta = C_vv - S_t
            if delta > 0:
                P_pred_next = A @ P_filt[t] @ A.T + Q_current
                C_rts = P_filt[t] @ A.T @ _inv2x2(P_pred_next)
                Q_candidate = delta * (C_rts @ C_rts.T)
                if _is_pos_semidef(Q_candidate):
                    Q_current = np.clip(Q_candidate, Q_FLOOR, Q_CEIL)
    C_gains = np.zeros((n, 2, 2))
    for t in range(n - 1):
        C_gains[t] = P_filt[t] @ A.T @ _inv2x2(P_pred[t + 1])
    return x_filt, P_filt, x_pred, P_pred, C_gains


# ============================================================================
# SLOPES
# ============================================================================

def compute_slopes_test1(x_filt, x_pred, C):
    n = len(x_filt)
    slopes = np.full(n, np.nan)
    for t in range(2, n):
        sm_t1 = x_filt[t - 1] + C[t - 1] @ (x_filt[t] - x_pred[t])
        sm_t2 = x_filt[t - 2] + C[t - 2] @ (sm_t1 - x_pred[t - 1])
        slopes[t] = sm_t1[0] - sm_t2[0]
    return slopes


def compute_slopes_test2(x_filt, P_filt, x_pred, C, live_per_candle, n_substeps):
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
        A_k = np.linalg.matrix_power(A_SUB, k_actual)
        Q_k = Q_SUB * k_actual
        x_pred_partial = A_k @ x_filt[t]
        P_pred_partial = A_k @ P_filt[t] @ A_k.T + Q_k
        C_partial = P_filt[t] @ A_k.T @ _inv2x2(P_pred_partial)
        sm_t = x_filt[t] + C_partial @ (x_prov - x_pred_partial)
        sm_t1 = x_filt[t - 1] + C[t - 1] @ (sm_t - x_pred[t])
        sm_t2 = x_filt[t - 2] + C[t - 2] @ (sm_t1 - x_pred[t - 1])
        slopes[t] = sm_t1[0] - sm_t2[0]
    return slopes


# ============================================================================
# BACKTEST
# ============================================================================

def _exec_trade(position, entry_price, exec_price, fees):
    if position == 1:
        pnl = (exec_price - entry_price) / entry_price
    else:
        pnl = (entry_price - exec_price) / entry_price
    return pnl - fees


def backtest_30m(slopes, closes_30m, start, end, fees, threshold=0.0, holding_min=0):
    pnl_total = 0.0
    n_trades = 0
    n_wins = 0
    position = 0
    entry_price = 0.0
    entry_t = -holding_min
    for t in range(start, end):
        if np.isnan(slopes[t]) or abs(slopes[t]) < threshold:
            continue
        target = 1 if slopes[t] > 0 else -1
        if position == target:
            continue
        if position != 0 and (t - entry_t) < holding_min:
            continue
        if t + 1 >= len(closes_30m):
            continue
        exec_price = closes_30m[t]
        if np.isnan(exec_price):
            continue
        if position != 0:
            trade_pnl = _exec_trade(position, entry_price, exec_price, fees)
            pnl_total += trade_pnl
            if trade_pnl > 0:
                n_wins += 1
        entry_price = exec_price
        position = target
        n_trades += 1
        entry_t = t
        pnl_total -= fees
    if position != 0 and end < len(closes_30m):
        exec_price = closes_30m[min(end, len(closes_30m) - 1)]
        if not np.isnan(exec_price):
            trade_pnl = _exec_trade(position, entry_price, exec_price, fees)
            pnl_total += trade_pnl
            if trade_pnl > 0:
                n_wins += 1
    wr = (n_wins / n_trades * 100.0) if n_trades > 0 else 0.0
    return {'pnl_pct': pnl_total * 100, 'trades': n_trades, 'win_rate': wr}


def backtest_5m(slopes, closes_5m_pc, k_substep, start, end, fees,
                threshold=0.0, holding_min=0):
    pnl_total = 0.0
    n_trades = 0
    n_wins = 0
    position = 0
    entry_price = 0.0
    entry_t = -holding_min
    for t in range(start, end):
        if np.isnan(slopes[t]) or abs(slopes[t]) < threshold:
            continue
        target = 1 if slopes[t] > 0 else -1
        if position == target:
            continue
        if position != 0 and (t - entry_t) < holding_min:
            continue
        candle_idx = t + 1
        if candle_idx >= len(closes_5m_pc):
            continue
        closes_5m = closes_5m_pc[candle_idx]
        step_idx = k_substep - 1
        if step_idx >= len(closes_5m):
            continue
        exec_price = closes_5m[step_idx]
        if np.isnan(exec_price):
            continue
        if position != 0:
            trade_pnl = _exec_trade(position, entry_price, exec_price, fees)
            pnl_total += trade_pnl
            if trade_pnl > 0:
                n_wins += 1
        entry_price = exec_price
        position = target
        n_trades += 1
        entry_t = t
        pnl_total -= fees
    if position != 0:
        last_candle = min(end, len(closes_5m_pc) - 1)
        closes_last = closes_5m_pc[last_candle]
        if len(closes_last) > 0 and not np.isnan(closes_last[-1]):
            trade_pnl = _exec_trade(position, entry_price, closes_last[-1], fees)
            pnl_total += trade_pnl
            if trade_pnl > 0:
                n_wins += 1
    wr = (n_wins / n_trades * 100.0) if n_trades > 0 else 0.0
    return {'pnl_pct': pnl_total * 100, 'trades': n_trades, 'win_rate': wr}


def buy_and_hold(closes_30m, start, end):
    c_s = closes_30m[start]
    c_e = closes_30m[min(end, len(closes_30m) - 1)]
    if np.isnan(c_s) or np.isnan(c_e):
        return 0.0
    return (c_e - c_s) / c_s * 100


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='FLKS OOS Validation — fixed configs on separate periods')
    parser.add_argument('--csv', type=str, default='data_trad/BTCUSD_all_5m.csv')
    parser.add_argument('--n-candles-30m', type=int, default=10000,
                        help='Total 30min candles to load (need >5000 for OOS-next)')
    parser.add_argument('--fees', type=float, default=0.001)
    args = parser.parse_args()

    fees = args.fees
    n_total = args.n_candles_30m

    # ==================================================================
    print(f"[1/6] Loading {args.csv} ...")
    df_5m = load_csv(args.csv)
    print(f"       {len(df_5m):,} 5min candles")

    # ==================================================================
    print("[2/6] Resampling to 30min ...")
    df_30m = resample_ohlcv(df_5m, 30)
    if len(df_30m) > n_total:
        df_30m = df_30m.iloc[-n_total:]
    df_5m = df_5m.loc[df_30m.index[0]:df_30m.index[-1] + pd.Timedelta(minutes=29)]
    closes_30m = df_30m['close'].values.astype(np.float64)
    n30 = len(df_30m)
    print(f"       {n30:,} bougies 30min")

    # ==================================================================
    print("[3/6] Computing MACD 30min + live 5min ...")
    macd_30m = calculate_macd(df_30m)
    is_close = compute_bucket_close_mask(df_5m.index, 30)
    close_5m = df_5m['close'].values.astype(np.float64)
    macd_live = compute_macd_live(close_5m, is_close)

    macd_live_pc = []
    closes_5m_pc = []
    for ts_30m in df_30m.index:
        bucket_end = ts_30m + pd.Timedelta(minutes=29, seconds=59)
        mask = (df_5m.index >= ts_30m) & (df_5m.index <= bucket_end)
        macd_live_pc.append(macd_live[mask])
        closes_5m_pc.append(close_5m[mask])
    print("       Done.")

    # ==================================================================
    print("[4/6] Computing slopes (standard + AQ-KF) ...")

    # Standard forward filter
    print("  Standard forward filter ...", end=" ", flush=True)
    x_f, P_f, x_p, P_p, C = forward_filter_30m(macd_30m)
    slopes_std = {}
    slopes_std['t1'] = compute_slopes_test1(x_f, x_p, C)
    for k in [1, 6]:
        slopes_std[f'k{k}'] = compute_slopes_test2(x_f, P_f, x_p, C, macd_live_pc, k)
    print("done.")

    # AQ-KF forward filter
    print("  AQ-KF forward filter ...", end=" ", flush=True)
    ax_f, aP_f, ax_p, aP_p, aC = forward_filter_30m_adaptive(macd_30m, window=30, Q_max_factor=10.0)
    slopes_aq = {}
    slopes_aq['t1'] = compute_slopes_test1(ax_f, ax_p, aC)
    for k in [1, 6]:
        slopes_aq[f'k{k}'] = compute_slopes_test2(ax_f, aP_f, ax_p, aC, macd_live_pc, k)
    print("done.")

    # Oracle
    print("  Oracle ...", end=" ", flush=True)
    slopes_oracle = compute_oracle_slopes(macd_30m)
    print("done.")

    # ==================================================================
    print("[5/6] Defining periods ...")

    # Periods
    periods = []

    # In-sample: [1000:5000] (if we have enough candles)
    if n30 >= 5000:
        periods.append(('In-sample [1000:5000]', 1000, 5000))

    # OOS-early: [0:1000]
    periods.append(('OOS-early [0:1000]', 0, 1000))

    # OOS-next: [5000:10000] or [5000:n30]
    if n30 > 5000:
        oos_end = min(10000, n30)
        periods.append((f'OOS-next [5000:{oos_end}]', 5000, oos_end))

    for pname, pstart, pend in periods:
        n_candles = pend - pstart
        days = n_candles * 30 / 60 / 24
        ts_start = df_30m.index[pstart] if pstart < n30 else "?"
        ts_end = df_30m.index[min(pend - 1, n30 - 1)] if pend <= n30 else "?"
        print(f"  {pname}: {n_candles} candles ({days:.0f} days) [{ts_start} → {ts_end}]")

    # ==================================================================
    print(f"\n[6/6] Backtesting 2 configs × {len(periods)} periods ...")

    configs = [CONFIG_STD, CONFIG_AQ]

    print(f"\n{'=' * 90}")
    print(f"  OOS Validation — MACD FLKS — Fees {fees*100:.1f}%/trade")
    print(f"  Config 1: {CONFIG_STD['name']}")
    print(f"  Config 2: {CONFIG_AQ['name']}")
    print(f"{'=' * 90}")
    print(f"  {'Period':<28} {'Config':<28} {'PnL':>8} {'Trades':>7} {'WR':>6} {'B&H':>8}")
    print(f"  {'-' * 87}")

    for pname, pstart, pend in periods:
        if pend > n30:
            print(f"  {pname:<28} — not enough data —")
            continue

        bh = buy_and_hold(closes_30m, pstart, pend - 1)

        for cfg in configs:
            if cfg['adaptive']:
                sl = slopes_aq[f"k{cfg['k']}"]
            else:
                sl = slopes_std[f"k{cfg['k']}"]

            if cfg['k'] == 0 or (not cfg['adaptive'] and cfg['k'] == 0):
                r = backtest_30m(sl, closes_30m, pstart, pend - 1, fees,
                                 threshold=cfg['thr'], holding_min=cfg['hold'])
            else:
                r = backtest_5m(sl, closes_5m_pc, cfg['k'], pstart, pend - 1, fees,
                                threshold=cfg['thr'], holding_min=cfg['hold'])

            print(f"  {pname:<28} {cfg['name']:<28} {r['pnl_pct']:>+7.1f}% "
                  f"{r['trades']:>7} {r['win_rate']:>5.1f}% {bh:>+7.1f}%")

        # Oracle for reference
        r_oracle = backtest_30m(slopes_oracle, closes_30m, pstart, pend - 1, fees)
        print(f"  {pname:<28} {'Oracle':<28} {r_oracle['pnl_pct']:>+7.1f}% "
              f"{r_oracle['trades']:>7} {r_oracle['win_rate']:>5.1f}% {bh:>+7.1f}%")
        print(f"  {'-' * 87}")

    print(f"{'=' * 90}")
    print("Done.")


if __name__ == '__main__':
    main()
