"""
Signal Processing Core — shared functions for all FLKS/AQ-KF scripts
=====================================================================

All data loading, indicator calculation, Kalman filters, slope computation,
metrics, and backtest functions in one place. No duplication.

Import with:
    from signal_processing.core import *
"""

import numpy as np
import pandas as pd
from collections import deque


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

FEES = 0.001


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
# KALMAN LIVE (5min resolution, frozen/provisional)
# ============================================================================

def compute_kalman_live_standard(indicator_live, is_close):
    """Standard Kalman (Q fixe) with frozen/provisional. Returns (n, 2) = [pos, vel]."""
    from pykalman import KalmanFilter as KF
    n = len(indicator_live)
    out = np.full((n, 2), np.nan)
    closure_indices = []
    closure_values = []
    for i in range(n):
        if not np.isnan(indicator_live[i]) and is_close[i]:
            closure_indices.append(i)
            closure_values.append(indicator_live[i])
    if len(closure_values) < 2:
        return out
    cv = np.array(closure_values)
    kf = KF(transition_matrices=A, observation_matrices=np.array([[1, 0]]),
            initial_state_mean=[cv[0], 0.0], initial_state_covariance=np.eye(2),
            observation_covariance=KALMAN_MEASURE_VAR,
            transition_covariance=np.eye(2) * KALMAN_PROCESS_VAR)
    state_means, state_covs = kf.filter(cv)
    for k, ci in enumerate(closure_indices):
        out[ci, 0] = state_means[k, 0]
        out[ci, 1] = state_means[k, 1]
    closure_set = set(closure_indices)
    current_k = -1
    sm_cl = np.array([cv[0], 0.0])
    sc_cl = np.eye(2)
    for i in range(n):
        obs = indicator_live[i]
        if np.isnan(obs):
            continue
        if i in closure_set:
            current_k += 1
            sm_cl = state_means[current_k]
            sc_cl = state_covs[current_k]
            continue
        if current_k >= 0:
            sm_p, _ = kf.filter_update(sm_cl, sc_cl, observation=obs)
            out[i, 0] = sm_p[0]
            out[i, 1] = sm_p[1]
    return out


def compute_kalman_live_aqkf(indicator_live, is_close, aq_window=30, Q_max_factor=10.0):
    """AQ-KF (adaptive Q) with frozen/provisional. Returns (n, 2) = [pos, vel]."""
    _Q_fixed = np.eye(2) * KALMAN_PROCESS_VAR
    _R = np.array([[KALMAN_MEASURE_VAR]])
    Q_FLOOR = _Q_fixed * 0.1
    Q_CEIL = _Q_fixed * Q_max_factor
    n = len(indicator_live)
    out = np.full((n, 2), np.nan)
    closure_indices = []
    closure_values = []
    for i in range(n):
        if not np.isnan(indicator_live[i]) and is_close[i]:
            closure_indices.append(i)
            closure_values.append(indicator_live[i])
    if len(closure_values) < 2:
        return out
    cv = np.array(closure_values)
    nc = len(cv)
    x_filt_cl = np.zeros((nc, 2))
    P_filt_cl = np.zeros((nc, 2, 2))
    Q_current = _Q_fixed.copy()
    innovation_buffer = []
    for k in range(nc):
        if k == 0:
            x_p = np.array([cv[0], 0.0])
            P_p = np.eye(2)
        else:
            x_p = A @ x_filt_cl[k - 1]
            P_p = A @ P_filt_cl[k - 1] @ A.T + Q_current
        y = cv[k] - H @ x_p
        S = (H @ P_p @ H.T + _R)[0, 0]
        K = P_p @ H.T / S
        x_filt_cl[k] = x_p + (K @ y).ravel()
        P_filt_cl[k] = (np.eye(2) - K @ H) @ P_p
        v_t = cv[k] - (H @ x_p)[0]
        innovation_buffer.append(v_t)
        if len(innovation_buffer) > aq_window:
            innovation_buffer.pop(0)
        if len(innovation_buffer) >= aq_window and k > 0:
            C_vv = np.mean(np.array(innovation_buffer) ** 2)
            delta = C_vv - S
            if delta > 0:
                P_pred_next = A @ P_filt_cl[k] @ A.T + Q_current
                C_rts = P_filt_cl[k] @ A.T @ inv2x2(P_pred_next)
                Q_candidate = delta * (C_rts @ C_rts.T)
                if is_pos_semidef(Q_candidate):
                    Q_current = np.clip(Q_candidate, Q_FLOOR, Q_CEIL)
    for k, ci in enumerate(closure_indices):
        out[ci, 0] = x_filt_cl[k, 0]
        out[ci, 1] = x_filt_cl[k, 1]
    closure_set = set(closure_indices)
    current_k = -1
    sm_cl = np.array([cv[0], 0.0])
    sc_cl = np.eye(2)
    for i in range(n):
        obs = indicator_live[i]
        if np.isnan(obs):
            continue
        if i in closure_set:
            current_k += 1
            sm_cl = x_filt_cl[current_k]
            sc_cl = P_filt_cl[current_k]
            continue
        if current_k >= 0:
            x_p = A @ sm_cl
            P_p = A @ sc_cl @ A.T + Q_current
            y_val = obs - (H @ x_p)[0]
            S_val = (H @ P_p @ H.T + _R)[0, 0]
            K_val = P_p @ H.T / S_val
            sm_p = x_p + (K_val * y_val).ravel()
            out[i, 0] = sm_p[0]
            out[i, 1] = sm_p[1]
    return out


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


def inv2x2(M):
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if abs(det) > 1e-15:
        return np.array([[M[1, 1], -M[0, 1]],
                         [-M[1, 0], M[0, 0]]]) / det
    return np.linalg.pinv(M)


def is_pos_semidef(M):
    return M[0, 0] >= 0 and M[1, 1] >= 0 and (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]) >= -1e-12


# ============================================================================
# FORWARD FILTERS
# ============================================================================

def forward_filter_30m(indicator_30m):
    """Standard Kalman forward filter. Returns (x_filt, P_filt, x_pred, P_pred, C)."""
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
        C[t] = P_filt[t] @ A.T @ inv2x2(P_pred[t + 1])
    return x_filt, P_filt, x_pred, P_pred, C


def forward_filter_30m_adaptive(indicator_30m, window=30, Q_max_factor=10.0,
                                  Q_min_factor=0.1):
    """AQ-KF forward filter (Myers-Tapley). Same output format as forward_filter_30m."""
    n = len(indicator_30m)
    first_valid_val = indicator_30m[~np.isnan(indicator_30m)][0]
    x_filt = np.zeros((n, 2))
    P_filt = np.zeros((n, 2, 2))
    x_pred = np.zeros((n, 2))
    P_pred = np.zeros((n, 2, 2))
    Q_current = Q.copy()
    innovation_buffer = []
    Q_FLOOR = Q * Q_min_factor
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
                C_rts = P_filt[t] @ A.T @ inv2x2(P_pred_next)
                Q_candidate = delta * (C_rts @ C_rts.T)
                if is_pos_semidef(Q_candidate):
                    Q_current = np.clip(Q_candidate, Q_FLOOR, Q_CEIL)

    C_gains = np.zeros((n, 2, 2))
    for t in range(n - 1):
        C_gains[t] = P_filt[t] @ A.T @ inv2x2(P_pred[t + 1])
    return x_filt, P_filt, x_pred, P_pred, C_gains


# ============================================================================
# SLOPES (FLKS backward)
# ============================================================================

def compute_slopes_test1(x_filt, x_pred, C):
    """Backward 2 steps from x_filt[t]. slope[t] = smoothed[t-1] - smoothed[t-2]."""
    n = len(x_filt)
    slopes = np.full(n, np.nan)
    for t in range(2, n):
        sm_t1 = x_filt[t - 1] + C[t - 1] @ (x_filt[t] - x_pred[t])
        sm_t2 = x_filt[t - 2] + C[t - 2] @ (sm_t1 - x_pred[t - 1])
        slopes[t] = sm_t1[0] - sm_t2[0]
    return slopes


def compute_slopes_test2(x_filt, P_filt, x_pred, C, live_per_candle, n_substeps):
    """Backward 3 steps: x_prov (from sub-steps of candle t+1) → t → t-1 → t-2."""
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
        C_partial = P_filt[t] @ A_k.T @ inv2x2(P_pred_partial)
        sm_t = x_filt[t] + C_partial @ (x_prov - x_pred_partial)
        sm_t1 = x_filt[t - 1] + C[t - 1] @ (sm_t - x_pred[t])
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
# BACKTEST
# ============================================================================

def _exec_trade(position, entry_price, exec_price, fees):
    if position == 1:
        pnl = (exec_price - entry_price) / entry_price
    else:
        pnl = (entry_price - exec_price) / entry_price
    return pnl - fees


def backtest_30m(slopes, closes_30m, start, end, fees,
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


def backtest_5m(slopes, closes_5m_per_candle, k_substep, start, end, fees,
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
        if candle_idx >= len(closes_5m_per_candle):
            continue
        closes_5m = closes_5m_per_candle[candle_idx]
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
        last_candle = min(end, len(closes_5m_per_candle) - 1)
        closes_last = closes_5m_per_candle[last_candle]
        if len(closes_last) > 0 and not np.isnan(closes_last[-1]):
            trade_pnl = _exec_trade(position, entry_price, closes_last[-1], fees)
            pnl_total += trade_pnl
            if trade_pnl > 0:
                n_wins += 1
    wr = (n_wins / n_trades * 100.0) if n_trades > 0 else 0.0
    return {'pnl_pct': pnl_total * 100, 'trades': n_trades, 'win_rate': wr}


def buy_and_hold(closes, start, end):
    c = closes[start:end + 1]
    valid = c[~np.isnan(c)]
    if len(valid) < 2:
        return 0.0
    return (valid[-1] - valid[0]) / valid[0] * 100


# ============================================================================
# POST-PROCESSING
# ============================================================================

def viterbi_decode(probs, self_trans=0.95):
    """Viterbi decoding on binary probability sequence."""
    n = len(probs)
    log_trans_same = np.log(self_trans)
    log_trans_switch = np.log(1 - self_trans)
    log_emit = np.zeros((n, 2))
    log_emit[:, 1] = np.log(np.clip(probs, 1e-10, 1 - 1e-10))
    log_emit[:, 0] = np.log(np.clip(1 - probs, 1e-10, 1 - 1e-10))
    V = np.zeros((n, 2))
    backptr = np.zeros((n, 2), dtype=int)
    V[0] = log_emit[0] + np.log(0.5)
    for t in range(1, n):
        for s in range(2):
            score_same = V[t-1, s] + log_trans_same
            other = 1 - s
            score_switch = V[t-1, other] + log_trans_switch
            if score_same >= score_switch:
                V[t, s] = score_same + log_emit[t, s]
                backptr[t, s] = s
            else:
                V[t, s] = score_switch + log_emit[t, s]
                backptr[t, s] = other
    labels = np.zeros(n, dtype=int)
    labels[-1] = np.argmax(V[-1])
    for t in range(n-2, -1, -1):
        labels[t] = backptr[t+1, labels[t+1]]
    return labels


def cusum_filter(probs, threshold=2.0):
    """CUSUM filter on probability sequence."""
    n = len(probs)
    labels = np.zeros(n, dtype=int)
    current_state = 1 if probs[0] > 0.5 else 0
    labels[0] = current_state
    s_up = 0.0
    s_down = 0.0
    for t in range(1, n):
        x = probs[t] - 0.5
        s_up = max(0, s_up + x)
        s_down = min(0, s_down + x)
        if current_state == 0 and s_up > threshold:
            current_state = 1
            s_up = 0.0
            s_down = 0.0
        elif current_state == 1 and -s_down > threshold:
            current_state = 0
            s_up = 0.0
            s_down = 0.0
        labels[t] = current_state
    return labels


# ============================================================================
# HELPERS
# ============================================================================

def group_per_candle(df_5m, df_30m, array_5m):
    """Group 5min values by 30min candle."""
    per_candle = []
    for ts_30m in df_30m.index:
        bucket_end = ts_30m + pd.Timedelta(minutes=29, seconds=59)
        mask = (df_5m.index >= ts_30m) & (df_5m.index <= bucket_end)
        per_candle.append(array_5m[mask])
    return per_candle


# ============================================================================
# DATA LOADING — NPZ + CSV aligned for backtests
# ============================================================================

PREPARED_DATA_DIR = 'data/prepared'

ASSET_CSV_MAP = {'BTC': 'BTCUSD'}


def find_features_csv():
    """Find the features CSV. Prefer FLKS features, fall back to old pipeline."""
    candidates = [
        f'{PREPARED_DATA_DIR}/BTCUSD_flks_features.csv',
        f'{PREPARED_DATA_DIR}/BTCUSD_multitf_macd_rsi_cci.csv',
    ]
    for c in candidates:
        from pathlib import Path
        if Path(c).exists():
            return c
    raise FileNotFoundError(f"No features CSV found. Tried: {candidates}")


def load_test_data(indicator='macd', timeframe='30m', threshold=0.5):
    """
    Load NPZ predictions + aligned closes at 30min resolution.

    The NPZ has predictions at 5min resolution (forward-filled labels).
    This function sub-samples to 30min closures so that backtests
    trade at 30min candle closes, not at every 5min step.

    Returns:
        y_test_30m: oracle labels at 30min closures
        y_pred_proba_30m: model probabilities at 30min closures
        y_pred_binary_30m: thresholded predictions at 30min closures
        closes_30m: close prices at 30min closures
        n_test_30m: number of 30min test samples
        csv_path: path to CSV used
    """
    npz_path = f'{PREPARED_DATA_DIR}/{indicator}_{timeframe}_dataset.npz'
    from pathlib import Path
    if not Path(npz_path).exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    if 'y_test' in data:
        y_test = data['y_test']
        y_pred_proba = data['y_test_pred']
    else:
        y_test = data['test_labels']
        y_pred_proba = data['test_preds']

    n_test = len(y_test)

    csv_path = find_features_csv()
    df = pd.read_csv(csv_path, parse_dates=['datetime']).set_index('datetime').sort_index()
    closes_all = df['close'].values

    # Test portion = last n_test rows of the CSV
    df_test = df.iloc[-n_test:]

    # Sub-sample to 30min closures (last row of each 30min bucket)
    bucket = df_test.index.floor('30min')
    is_closure = bucket != np.append(bucket[1:], pd.NaT)

    y_test_30m = y_test[is_closure]
    y_pred_proba_30m = y_pred_proba[is_closure]
    y_pred_binary_30m = (y_pred_proba_30m > threshold).astype(int)
    closes_30m = df_test['close'].values[is_closure]
    n_test_30m = len(y_test_30m)

    return y_test_30m, y_pred_proba_30m, y_pred_binary_30m, closes_30m, n_test_30m, csv_path
