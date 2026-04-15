#!/usr/bin/env python3
"""
FLKS convergence par sous-pas 5min — MACD, RSI, CCI comparés
=============================================================

Pour chaque indicateur (MACD, RSI, CCI) :
  Oracle : pykalman.smooth() sur 5000 bougies 30min
  Test 1 : FLKS 30min pur (backward depuis x_filt[t])
  Test 2 : FLKS + k=1..6 sous-pas MACD/RSI/CCI live de bougie t+1
  Métrique : % concordance de signe vs oracle sur [1000:5000]

Usage:
    python src/signal_processing/flks_substep_convergence.py \
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
# PARAMETERS (from pipeline: prepare_multitf_csv.py)
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
    """Live-style OHLCV: partial candle updated every 5min (from pipeline)."""
    group = df_5min.index.floor(f'{tf_minutes}min')
    r = pd.DataFrame(index=df_5min.index)
    r['open'] = df_5min.groupby(group)['open'].transform('first')
    r['high'] = df_5min.groupby(group)['high'].cummax()
    r['low'] = df_5min.groupby(group)['low'].cummin()
    r['close'] = df_5min['close']
    return r


# ============================================================================
# INDICATORS — Standard 30min (from pipeline)
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
# INDICATORS — Live frozen/provisional (from pipeline)
# ============================================================================

def compute_macd_live(close_5min, is_close):
    """MACD live frozen/provisional (from pipeline)."""
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
    """RSI live frozen/provisional (from pipeline)."""
    n = len(close_5min)
    alpha = 2.0 / (RSI_PERIOD + 1)
    out = np.full(n, np.nan)

    # Step 1: collect closure closes
    closure_indices = []
    closure_closes = []
    for i in range(n):
        if not np.isnan(close_5min[i]) and is_close[i]:
            closure_indices.append(i)
            closure_closes.append(close_5min[i])
    if len(closure_closes) < 2:
        return out

    # Step 2: compute EWM avg_gain/avg_loss on closure sequence
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

    # Step 3: assign RSI at closures
    for k, ci in enumerate(closure_indices):
        ag_k, al_k, _ = closure_states[k]
        if al_k > 1e-15:
            out[ci] = 100.0 - 100.0 / (1.0 + ag_k / al_k)

    # Step 4: provisional inter-closure points
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
    """CCI live with rolling TP buffer, freeze at closure (from pipeline)."""
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
# ORACLE: pykalman.smooth() on 30min
# ============================================================================

def compute_oracle(indicator_30m):
    from pykalman import KalmanFilter as KF
    n = len(indicator_30m)
    valid = ~np.isnan(indicator_30m)
    if valid.sum() < 3:
        return np.full(n, np.nan), np.full(n, np.nan)
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
    return positions, slopes


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
# FORWARD FILTER 30min (shared by Test 1 and Test 2)
# ============================================================================

def forward_filter_30m(indicator_30m):
    """Standard Kalman forward filter on 30min indicator. Returns states + RTS gains."""
    n = len(indicator_30m)
    # Find first valid observation for initialization
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
            # No observation: filtered = predicted
            x_filt[t] = x_p
            P_filt[t] = P_p
        else:
            x_filt[t], P_filt[t] = kf_update(x_p, P_p, indicator_30m[t])

    # Precompute RTS gains C[t] = P_filt[t] @ A.T @ inv(P_pred[t+1])
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
# FORWARD FILTER 30min — ADAPTIVE Q (Myers-Tapley)
# ============================================================================

def _inv2x2(M):
    """Fast 2x2 matrix inverse."""
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if abs(det) > 1e-15:
        return np.array([[M[1, 1], -M[0, 1]],
                         [-M[1, 0], M[0, 0]]]) / det
    return np.linalg.pinv(M)


def _is_pos_semidef(M):
    """Check if 2x2 matrix is positive semi-definite."""
    return M[0, 0] >= 0 and M[1, 1] >= 0 and (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]) >= -1e-12


def forward_filter_30m_adaptive(indicator_30m, window=30, Q_min=1e-6,
                                 Q_max_factor=None, Q_min_factor=None):
    """
    Kalman forward filter with adaptive Q (Myers-Tapley).

    Returns same format as forward_filter_30m: (x_filt, P_filt, x_pred, P_pred, C)
    Plus diagnostics dict with Q_history, K_history, innovation_history.
    """
    n = len(indicator_30m)
    first_valid_val = indicator_30m[~np.isnan(indicator_30m)][0]

    x_filt = np.zeros((n, 2))
    P_filt = np.zeros((n, 2, 2))
    x_pred = np.zeros((n, 2))
    P_pred = np.zeros((n, 2, 2))

    Q_current = Q.copy()
    innovation_buffer = []
    # Clipping symétrique : Q reste dans [Q*0.1, Q*10]
    Q_FLOOR = Q * 0.1    # 0.001 * I
    Q_CEIL = Q * 10.0    # 0.1 * I
    # Allow override via parameter
    if Q_max_factor is not None:
        Q_CEIL = Q * Q_max_factor
    if Q_min_factor is not None:
        Q_FLOOR = Q * Q_min_factor

    # Diagnostics
    Q_history = np.full((n, 2, 2), np.nan)
    K_history = np.full((n, 2), np.nan)
    innov_history = np.full(n, np.nan)
    delta_history = np.full(n, np.nan)  # C_vv - S before clipping

    for t in range(n):
        # 1. Predict
        if t == 0:
            x_p = np.array([first_valid_val, 0.0])
            P_p = np.eye(2)
        else:
            x_p = A @ x_filt[t - 1]
            P_p = A @ P_filt[t - 1] @ A.T + Q_current

        x_pred[t] = x_p
        P_pred[t] = P_p
        Q_history[t] = Q_current

        # 2. Update
        if np.isnan(indicator_30m[t]):
            x_filt[t] = x_p
            P_filt[t] = P_p
            continue

        # Compute Kalman gain before update (for diagnostics)
        S_t = (H @ P_p @ H.T + R)[0, 0]
        K_t = P_p @ H.T / S_t  # (2,1)
        K_history[t] = K_t.ravel()

        x_filt[t], P_filt[t] = kf_update(x_p, P_p, indicator_30m[t])

        # 3. Innovation
        v_t = indicator_30m[t] - (H @ x_p)[0]
        innov_history[t] = v_t
        innovation_buffer.append(v_t)
        if len(innovation_buffer) > window:
            innovation_buffer.pop(0)

        # 4-6. Adaptive Q update (only when window full and t > 0)
        if len(innovation_buffer) >= window and t > 0:
            C_vv = np.mean(np.array(innovation_buffer) ** 2)
            delta = C_vv - S_t
            delta_history[t] = delta

            if delta > 0:
                P_pred_next = A @ P_filt[t] @ A.T + Q_current
                C_rts = P_filt[t] @ A.T @ _inv2x2(P_pred_next)
                Q_candidate = delta * (C_rts @ C_rts.T)

                if _is_pos_semidef(Q_candidate):
                    Q_current = np.clip(Q_candidate, Q_FLOOR, Q_CEIL)

    # Precompute RTS gains
    C_gains = np.zeros((n, 2, 2))
    for t in range(n - 1):
        C_gains[t] = P_filt[t] @ A.T @ _inv2x2(P_pred[t + 1])

    diagnostics = {
        'Q_history': Q_history,
        'K_history': K_history,
        'innov_history': innov_history,
        'delta_history': delta_history,
    }

    return x_filt, P_filt, x_pred, P_pred, C_gains, diagnostics


# ============================================================================
# TEST 1: FLKS 30min pur
# ============================================================================

def compute_slopes_test1(x_filt, x_pred, C):
    n = len(x_filt)
    slopes = np.full(n, np.nan)
    for t in range(2, n):
        sm_t1 = x_filt[t - 1] + C[t - 1] @ (x_filt[t] - x_pred[t])
        sm_t2 = x_filt[t - 2] + C[t - 2] @ (sm_t1 - x_pred[t - 1])
        slopes[t] = sm_t1[0] - sm_t2[0]
    return slopes


# ============================================================================
# TEST 2: FLKS 30min + sous-pas 5min
# ============================================================================

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

        # Pas 1 : lisser t avec x_prov (backward depuis t+k/6 vers t)
        # Transition partielle = A_SUB^k (k sous-pas entre t et x_prov)
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

        # Pas 2 : lisser t-1 avec smoothed[t]
        sm_t1 = x_filt[t - 1] + C[t - 1] @ (sm_t - x_pred[t])

        # Pas 3 : lisser t-2 avec smoothed[t-1]
        sm_t2 = x_filt[t - 2] + C[t - 2] @ (sm_t1 - x_pred[t - 1])

        slopes[t] = sm_t1[0] - sm_t2[0]
    return slopes


# ============================================================================
# METRICS
# ============================================================================

def sign_concordance(slopes_test, slopes_oracle, start, end):
    EPSILON = 1e-8
    s_t = slopes_test[start:end]
    s_o = slopes_oracle[start:end]
    mask = ~np.isnan(s_t) & ~np.isnan(s_o) & (np.abs(s_o) > EPSILON)
    n_valid = mask.sum()
    if n_valid == 0:
        return np.nan, 0
    return np.mean(np.sign(s_t[mask]) == np.sign(s_o[mask])) * 100.0, n_valid


def find_oracle_transitions(slopes_oracle, start, end):
    EPSILON = 1e-8
    s_o = slopes_oracle[start:end]
    sign_o = np.where(np.abs(s_o) < EPSILON, 0, np.sign(s_o))
    trans = np.zeros(len(s_o), dtype=bool)
    for i in range(1, len(s_o)):
        if sign_o[i] != 0 and sign_o[i - 1] != 0 and sign_o[i] != sign_o[i - 1]:
            trans[i] = True
    return trans


def sign_concordance_at_transitions(slopes_test, slopes_oracle, start, end, trans):
    s_t = slopes_test[start:end]
    s_o = slopes_oracle[start:end]
    mask = trans & ~np.isnan(s_t) & ~np.isnan(s_o)
    n_valid = mask.sum()
    if n_valid == 0:
        return np.nan, 0
    return np.mean(np.sign(s_t[mask]) == np.sign(s_o[mask])) * 100.0, n_valid


# ============================================================================
# RUN ONE INDICATOR
# ============================================================================

def run_indicator_adaptive(name, indicator_30m, live_per_candle, eval_start, n30,
                           slopes_oracle, trans_mask, window=30, output_dir=None,
                           Q_max_factor=10.0, Q_min_factor=0.1):
    """Run AQ-KF Test 1 + Test 2 for one indicator with adaptive Q."""
    print(f"\n  --- AQ-KF {name} (window={window}, Q_max={Q_max_factor}×Q) ---")

    # Adaptive forward filter
    print(f"  Forward filter adaptive ...", end=" ", flush=True)
    x_filt, P_filt, x_pred, P_pred, C, diag = forward_filter_30m_adaptive(
        indicator_30m, window=window,
        Q_max_factor=Q_max_factor, Q_min_factor=Q_min_factor)
    print("done.")

    # --- Diagnostics ---
    Q_00 = diag['Q_history'][:, 0, 0]
    Q_11 = diag['Q_history'][:, 1, 1]
    K_0 = diag['K_history'][:, 0]
    K_1 = diag['K_history'][:, 1]
    valid_q = ~np.isnan(Q_00)
    valid_k = ~np.isnan(K_0)

    print(f"\n  Q[0,0] (position process noise):")
    print(f"    Fixed Q = {KALMAN_PROCESS_VAR}")
    q_vals = Q_00[valid_q]
    print(f"    Adaptive: min={q_vals.min():.2e}  median={np.median(q_vals):.2e}"
          f"  P95={np.percentile(q_vals, 95):.2e}  max={q_vals.max():.2e}")

    print(f"  Q[1,1] (velocity process noise):")
    q1_vals = Q_11[valid_q]
    print(f"    Adaptive: min={q1_vals.min():.2e}  median={np.median(q1_vals):.2e}"
          f"  P95={np.percentile(q1_vals, 95):.2e}  max={q1_vals.max():.2e}")

    print(f"  Kalman gain K[0] (position):")
    k_vals = K_0[valid_k]
    print(f"    min={k_vals.min():.4f}  median={np.median(k_vals):.4f}"
          f"  P95={np.percentile(k_vals, 95):.4f}  max={k_vals.max():.4f}")

    print(f"  Kalman gain K[1] (velocity):")
    k1_vals = K_1[valid_k]
    print(f"    min={k1_vals.min():.4f}  median={np.median(k1_vals):.4f}"
          f"  P95={np.percentile(k1_vals, 95):.4f}  max={k1_vals.max():.4f}")

    # Delta diagnostics
    delta = diag['delta_history']
    valid_d = ~np.isnan(delta)
    if valid_d.sum() > 0:
        d_vals = delta[valid_d]
        pct_pos = np.mean(d_vals > 0) * 100
        pct_neg = np.mean(d_vals < 0) * 100
        print(f"  Delta (C_vv - S) before clipping:")
        print(f"    min={d_vals.min():.2f}  median={np.median(d_vals):.2f}"
              f"  P95={np.percentile(d_vals, 95):.2f}  max={d_vals.max():.2f}")
        print(f"    delta > 0: {pct_pos:.1f}% (Q wants to increase)")
        print(f"    delta < 0: {pct_neg:.1f}% (Q wants to decrease)")

    # Plot diagnostics
    if output_dir is not None:
        fig, axes = plt.subplots(4, 1, figsize=(16, 13), sharex=True)
        t_range = np.arange(len(Q_00))

        ax = axes[0]
        ax.semilogy(t_range[valid_q], Q_00[valid_q], linewidth=0.5, color='tab:blue',
                    label='Q[0,0] adaptive')
        ax.axhline(y=KALMAN_PROCESS_VAR, color='red', linestyle='--',
                   label=f'Q fixed = {KALMAN_PROCESS_VAR}')
        ax.set_ylabel('Q[0,0]')
        ax.set_title(f'AQ-KF Diagnostics — {name} (window={window})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(t_range[valid_k], K_0[valid_k], linewidth=0.5, color='tab:green',
                label='K[0] (position gain)')
        ax.plot(t_range[valid_k], K_1[valid_k], linewidth=0.5, color='tab:orange',
                label='K[1] (velocity gain)')
        ax.set_ylabel('Kalman Gain')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        innov = diag['innov_history']
        valid_i = ~np.isnan(innov)
        ax.plot(t_range[valid_i], innov[valid_i], linewidth=0.3, color='gray',
                alpha=0.5, label='Innovation v[t]')
        # Rolling variance
        w = window
        if valid_i.sum() > w:
            innov_sq = innov[valid_i] ** 2
            roll_var = np.convolve(innov_sq, np.ones(w) / w, mode='same')
            ax.plot(t_range[valid_i], np.sqrt(roll_var), linewidth=1.0,
                    color='tab:red', label=f'sqrt(C_vv) MA({w})')
        ax.set_ylabel('Innovation')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[3]
        if valid_d.sum() > 0:
            ax.plot(t_range[valid_d], delta[valid_d], linewidth=0.5,
                    color='tab:purple', alpha=0.7, label='delta = C_vv - S')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.fill_between(t_range[valid_d], 0, delta[valid_d],
                            where=delta[valid_d] > 0, color='red', alpha=0.15,
                            label='Q wants to increase')
            ax.fill_between(t_range[valid_d], 0, delta[valid_d],
                            where=delta[valid_d] < 0, color='blue', alpha=0.15,
                            label='Q wants to decrease')
        ax.set_ylabel('Delta (C_vv - S)')
        ax.set_xlabel('Candle index')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = output_dir / f'aq_diagnostics_{name.lower()}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Diagnostics plot saved: {out_path}")

    # --- Tests ---
    # AQ-KF Test 1
    slopes_t1 = compute_slopes_test1(x_filt, x_pred, C)
    c_t1_all, _ = sign_concordance(slopes_t1, slopes_oracle, eval_start, n30)
    c_t1_tr, _ = sign_concordance_at_transitions(
        slopes_t1, slopes_oracle, eval_start, n30, trans_mask)
    print(f"\n  AQ T1 (30m pur): all={c_t1_all:.2f}%  trans={c_t1_tr:.2f}%")

    # AQ-KF Test 2 k=1..6
    results_k = []
    for k in range(1, 7):
        slopes_k = compute_slopes_test2(
            x_filt, P_filt, x_pred, C, live_per_candle, k)
        ck_all, _ = sign_concordance(slopes_k, slopes_oracle, eval_start, n30)
        ck_tr, _ = sign_concordance_at_transitions(
            slopes_k, slopes_oracle, eval_start, n30, trans_mask)
        results_k.append((k, ck_all, ck_tr))
        print(f"  AQ T2 k={k} ({k*5}min): all={ck_all:.2f}%  trans={ck_tr:.2f}%")

    return {
        'name': f'AQ-{name}',
        't1_all': c_t1_all,
        't1_tr': c_t1_tr,
        'results_k': results_k,
        'diagnostics': diag,
    }


def run_indicator(name, indicator_30m, live_per_candle, eval_start, n30):
    """Run Test 1 + Test 2 for one indicator. Returns dict of results."""
    print(f"\n  --- {name} ---")

    # Oracle
    _, slopes_oracle = compute_oracle(indicator_30m)
    trans_mask = find_oracle_transitions(slopes_oracle, eval_start, n30)
    n_trans = trans_mask.sum()

    EPSILON = 1e-8
    s_o = slopes_oracle[eval_start:n30]
    sign_o = np.where(np.abs(s_o) < EPSILON, 0, np.sign(s_o))
    valid_signs = sign_o[sign_o != 0]
    persistence = np.mean(valid_signs[1:] == valid_signs[:-1]) * 100.0 if len(valid_signs) > 1 else 0.0
    print(f"  Transitions: {n_trans} ({n_trans/(n30-eval_start)*100:.1f}%)  "
          f"Persistence: {persistence:.1f}%")

    # Forward filter
    x_filt, P_filt, x_pred, P_pred, C = forward_filter_30m(indicator_30m)

    # Test 1
    slopes_t1 = compute_slopes_test1(x_filt, x_pred, C)
    c_t1_all, _ = sign_concordance(slopes_t1, slopes_oracle, eval_start, n30)
    c_t1_tr, _ = sign_concordance_at_transitions(
        slopes_t1, slopes_oracle, eval_start, n30, trans_mask)
    print(f"  Test 1 (30m pur): all={c_t1_all:.2f}%  trans={c_t1_tr:.2f}%")

    # Test 2 k=1..6
    results_k = []
    for k in range(1, 7):
        slopes_k = compute_slopes_test2(
            x_filt, P_filt, x_pred, C, live_per_candle, k)
        ck_all, _ = sign_concordance(slopes_k, slopes_oracle, eval_start, n30)
        ck_tr, _ = sign_concordance_at_transitions(
            slopes_k, slopes_oracle, eval_start, n30, trans_mask)
        results_k.append((k, ck_all, ck_tr))
        print(f"  Test 2 k={k} ({k*5}min): all={ck_all:.2f}%  trans={ck_tr:.2f}%")

    return {
        'name': name,
        'n_trans': n_trans,
        'persistence': persistence,
        't1_all': c_t1_all,
        't1_tr': c_t1_tr,
        'results_k': results_k,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='FLKS substep convergence — MACD, RSI, CCI compared')
    parser.add_argument('--csv', type=str, default='data_trad/BTCUSD_all_5m.csv')
    parser.add_argument('--n-candles-30m', type=int, default=5000)
    parser.add_argument('--eval-start', type=int, default=1000)
    parser.add_argument('--output-dir', type=str, default='plots')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n30 = args.n_candles_30m

    # ------------------------------------------------------------------
    print(f"[1/5] Loading {args.csv} ...")
    df_5m = load_csv(args.csv)
    print(f"       {len(df_5m):,} 5min candles")

    # ------------------------------------------------------------------
    print("[2/5] Resampling to 30min ...")
    df_30m = resample_ohlcv(df_5m, 30)
    if len(df_30m) > n30:
        df_30m = df_30m.iloc[-n30:]
    df_5m = df_5m.loc[df_30m.index[0]:df_30m.index[-1] + pd.Timedelta(minutes=29)]
    print(f"       {len(df_30m):,} bougies 30min, {len(df_5m):,} bougies 5min")

    # ------------------------------------------------------------------
    print("[3/5] Computing indicators (30min standard + 5min live) ...")

    is_close = compute_bucket_close_mask(df_5m.index, 30)
    close_5m = df_5m['close'].values.astype(np.float64)

    # MACD
    macd_30m = calculate_macd(df_30m)
    macd_live = compute_macd_live(close_5m, is_close)

    # RSI
    rsi_30m = calculate_rsi(df_30m)
    rsi_live = compute_rsi_live(close_5m, is_close)

    # CCI (needs live OHLCV)
    cci_30m = calculate_cci(df_30m)
    live_ohlcv = compute_live_ohlcv(df_5m, 30)
    cci_live = compute_cci_live(
        live_ohlcv['high'].values.astype(np.float64),
        live_ohlcv['low'].values.astype(np.float64),
        close_5m, is_close)

    # Group live values per 30min candle
    def group_per_candle(live_array):
        per_candle = []
        for ts_30m in df_30m.index:
            bucket_end = ts_30m + pd.Timedelta(minutes=29, seconds=59)
            mask = (df_5m.index >= ts_30m) & (df_5m.index <= bucket_end)
            per_candle.append(live_array[mask])
        return per_candle

    macd_live_pc = group_per_candle(macd_live)
    rsi_live_pc = group_per_candle(rsi_live)
    cci_live_pc = group_per_candle(cci_live)

    # Coherence check
    for name, ind_30m, live_pc in [('MACD', macd_30m, macd_live_pc),
                                    ('RSI', rsi_30m, rsi_live_pc),
                                    ('CCI', cci_30m, cci_live_pc)]:
        max_err = 0.0
        n_checked = 0
        for t in range(n30):
            vals = [v for v in live_pc[t] if not np.isnan(v)]
            if len(vals) > 0 and not np.isnan(ind_30m[t]):
                max_err = max(max_err, abs(vals[-1] - ind_30m[t]))
                n_checked += 1
        print(f"       {name}: coherence max err = {max_err:.2e} ({n_checked} candles)")

    # ------------------------------------------------------------------
    print("[4/6] Running FLKS for each indicator ...")

    all_results = []
    oracle_data = {}  # store oracle slopes + trans_mask for adaptive reuse

    for name, ind_30m, live_pc in [('MACD', macd_30m, macd_live_pc),
                                    ('RSI', rsi_30m, rsi_live_pc),
                                    ('CCI', cci_30m, cci_live_pc)]:
        res = run_indicator(name, ind_30m, live_pc, args.eval_start, n30)
        all_results.append(res)
        # Store oracle data for adaptive reuse
        _, slopes_oracle = compute_oracle(ind_30m)
        trans_mask = find_oracle_transitions(slopes_oracle, args.eval_start, n30)
        oracle_data[name] = (slopes_oracle, trans_mask)

    # ------------------------------------------------------------------
    print("\n[5/6] Running AQ-KF (adaptive Q) for MACD — Q_max sweep ...")

    slopes_oracle_macd, trans_mask_macd = oracle_data['MACD']
    q_max_factors = [10, 50, 100, 500]
    aq_sweep = {}
    for qmf in q_max_factors:
        res = run_indicator_adaptive(
            'MACD', macd_30m, macd_live_pc, args.eval_start, n30,
            slopes_oracle_macd, trans_mask_macd, window=30,
            output_dir=output_dir, Q_max_factor=qmf)
        aq_sweep[qmf] = res
    aq_result = aq_sweep[q_max_factors[0]]  # default for backward compat

    # ------------------------------------------------------------------
    print(f"\n[6/6] Résultats comparatifs")
    print(f"{'=' * 90}")
    print(f"  Concordance de signe vs Oracle (pykalman.smooth 30min)")
    print(f"  Éval: [{args.eval_start}:{n30}]  |  Pente: smoothed[t-1] - smoothed[t-2]")
    print(f"{'=' * 90}")

    # Header
    ind_names = [r['name'] for r in all_results]
    header = f"  {'Méthode':<22}"
    for r in all_results:
        header += f" │ {r['name']:^17}"
    print(header)

    sub_header = f"  {'':22}"
    for r in all_results:
        sub_header += f" │ {'All':>7}  {'Trans':>7}"
    print(sub_header)
    print(f"  {'-' * (22 + 20 * len(all_results))}")

    # Persistence row
    row = f"  {'Persistence':<22}"
    for r in all_results:
        row += f" │ {r['persistence']:>6.1f}%  {r['n_trans']:>5} tr"
    print(row)
    print(f"  {'-' * (22 + 20 * len(all_results))}")

    # Test 1
    row = f"  {'Test 1: 30m pur':<22}"
    for r in all_results:
        row += f" │ {r['t1_all']:>6.2f}% {r['t1_tr']:>7.2f}%"
    print(row)

    # Test 2 k=1..6
    for ki in range(6):
        k = ki + 1
        row = f"  {'Test 2: k=' + str(k) + ' (' + str(k*5) + 'min)':<22}"
        for r in all_results:
            ck_all = r['results_k'][ki][1]
            ck_tr = r['results_k'][ki][2]
            row += f" │ {ck_all:>6.2f}% {ck_tr:>7.2f}%"
        print(row)

    print(f"  {'-' * (22 + 20 * len(all_results))}")

    # Delta row (k=6 vs T1)
    row = f"  {'Gain k=6 vs T1':<22}"
    for r in all_results:
        d_all = r['results_k'][5][1] - r['t1_all']
        d_tr = r['results_k'][5][2] - r['t1_tr']
        row += f" │ {d_all:>+6.2f}p {d_tr:>+7.2f}p"
    print(row)

    print(f"  {'-' * (22 + 20 * len(all_results))}")

    # AQ-KF section — Q_max sweep
    print(f"\n  --- Adaptive Q (MACD) — Q_max sweep (Trans only) ---")
    macd_std = all_results[0]

    hdr = f"  {'Method':<22}"
    for qmf in q_max_factors:
        hdr += f" │ {'Q×' + str(qmf):>8}"
    hdr += f" │ {'Standard':>10}"
    print(hdr)
    print(f"  {'-' * (24 + 11 * len(q_max_factors) + 13)}")

    # T1 row
    row = f"  {'AQ T1: 30m pur':<22}"
    for qmf in q_max_factors:
        row += f" │ {aq_sweep[qmf]['t1_tr']:>7.2f}%"
    row += f" │ {macd_std['t1_tr']:>9.2f}%"
    print(row)

    # T2 k=1..6
    for ki in range(6):
        k = ki + 1
        label = f"  AQ T2: k={k} ({k*5}min)"
        row = f"{label:<24}"
        for qmf in q_max_factors:
            row += f" │ {aq_sweep[qmf]['results_k'][ki][2]:>7.2f}%"
        row += f" │ {macd_std['results_k'][ki][2]:>9.2f}%"
        print(row)

    print(f"  {'-' * (24 + 11 * len(q_max_factors) + 13)}")

    print(f"{'=' * 90}")

    # ------------------------------------------------------------------
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    colors_all = 'steelblue'
    colors_tr = 'tomato'

    for idx, r in enumerate(all_results):
        ax = axes[idx]
        labels = ['T1'] + [f'k={k}' for k in range(1, 7)]
        vals_all = [r['t1_all']] + [rk[1] for rk in r['results_k']]
        vals_tr = [r['t1_tr']] + [rk[2] for rk in r['results_k']]
        x_pos = np.arange(len(labels))
        w = 0.35

        ax.bar(x_pos - w / 2, vals_all, w, color=colors_all, alpha=0.8, label='All')
        ax.bar(x_pos + w / 2, vals_tr, w, color=colors_tr, alpha=0.8, label='Transitions')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"{r['name']}\nPersist={r['persistence']:.0f}% | "
                     f"Trans={r['n_trans']}", fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        if idx == 0:
            ax.set_ylabel('Sign concordance vs oracle (%)')
            ax.legend(fontsize=8)

        for i, (va, vt) in enumerate(zip(vals_all, vals_tr)):
            ax.text(i - w / 2, va + 0.5, f'{va:.0f}', ha='center', fontsize=7)
            ax.text(i + w / 2, vt + 0.5, f'{vt:.0f}', ha='center', fontsize=7)

    axes[0].set_ylim(0, 100)
    plt.suptitle('FLKS substep convergence — MACD vs RSI vs CCI\n'
                 'Oracle = pykalman.smooth 30min BTC', fontsize=12)
    plt.tight_layout()
    out_path = output_dir / 'flks_substep_convergence.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
