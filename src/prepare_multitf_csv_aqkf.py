#!/usr/bin/env python3
"""
Multi-Timeframe CSV Preparation — AQ-KF ADAPTIVE LIVE FEATURES
===============================================================

FORK of prepare_multitf_csv.py. ONLY CHANGE: compute_kalman_live()
uses AQ-KF (adaptive Q, Myers-Tapley) instead of fixed-Q pykalman.
Labels UNCHANGED (oracle pykalman.smooth). Everything else identical.

PURPOSE:
    Generate one enriched CSV per asset reproducing what Binance API returns
    when querying 30min/1h klines every 5min: the last candle is the one
    currently forming, updated progressively.

CLOSURE DETECTION:
    Uses bucket-change detection (not step_index == max_step) to handle
    data gaps correctly. If a 30min bucket has only 4 out of 6 bars,
    the 4th bar is the closure point. This matches resample(close='last').

CAUSALITY:
    close_live[i] = close_5min[i], known at time i. Backtest at open[i+1].
    No shift(1) needed. Kalman uses filter_update (forward-only), never smooth().

VALIDATION:
    At bucket closure, all live values must match standard resample values
    exactly (atol=1e-10), aligned by timestamp.

Usage:
    python src/prepare_multitf_csv.py --assets BTC --indicators macd
    python src/prepare_multitf_csv.py --assets BTC ETH BNB ADA LTC
"""

import numpy as np
import pandas as pd
import argparse
import logging
import os
import sys
from pathlib import Path
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from constants import AVAILABLE_ASSETS_5M, PREPARED_DATA_DIR

# Indicator periods (same as prepare_data_direction_only.py)
RSI_PERIOD = 14
CCI_PERIOD = 20
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Kalman parameters for LIVE features (causal filter_update)
KALMAN_PROCESS_VAR = 0.01
KALMAN_MEASURE_VAR = 0.1

# Kalman parameters for ORACLE LABELS (non-causal smooth, tunable separately)
KALMAN_LABEL_PROCESS_VAR = 0.01
KALMAN_LABEL_MEASURE_VAR = 0.1


# =============================================================================
# DATA LOADING
# =============================================================================

def load_csv_5min(file_path: str, asset_name: str) -> pd.DataFrame:
    """Load raw 5min OHLCV from CSV. Returns DataFrame with DatetimeIndex."""
    df = pd.read_csv(file_path)
    date_col = None
    for col in ['date', 'datetime', 'time', 'timestamp', 'Date', 'Datetime']:
        if col in df.columns:
            date_col = col
            break
    if date_col is None:
        raise ValueError(f"Date column not found in {file_path}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = 'datetime'
    df.columns = df.columns.str.lower()
    df = df.sort_index()
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    logger.info(f"  {asset_name}: {len(df):,} 5min candles, {df.index[0]} -> {df.index[-1]}")
    return df


# =============================================================================
# RESAMPLING (for validation reference only)
# =============================================================================

def resample_ohlcv(df_5min: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
    """Resample 5min to higher tf with standard OHLCV aggregation."""
    return df_5min.resample(f'{tf_minutes}min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()


# =============================================================================
# STANDARD INDICATORS (for validation reference only)
# =============================================================================

def calculate_macd_standard(df):
    ema_f = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_s = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return line - sig

def calculate_rsi_standard(df):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    ag = gain.ewm(span=RSI_PERIOD, adjust=False).mean()
    al = loss.ewm(span=RSI_PERIOD, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_cci_standard(df):
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(CCI_PERIOD).mean()
    mad = tp.rolling(CCI_PERIOD).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma) / (0.015 * mad)

def kalman_filter_standard(data):
    """Forward-only Kalman on 1D series. Returns (position, velocity) arrays."""
    from pykalman import KalmanFilter as KF
    valid = ~np.isnan(data)
    if valid.sum() < 2:
        return np.full(len(data), np.nan), np.full(len(data), np.nan)
    vd = data[valid]
    kf = KF(transition_matrices=[[1,1],[0,1]], observation_matrices=[[1,0]],
            initial_state_mean=[vd[0], 0.0], initial_state_covariance=np.eye(2),
            observation_covariance=KALMAN_MEASURE_VAR,
            transition_covariance=np.eye(2) * KALMAN_PROCESS_VAR)
    sm, _ = kf.filter(vd)
    pos = np.full(len(data), np.nan)
    vel = np.full(len(data), np.nan)
    pos[valid] = sm[:, 0]
    vel[valid] = sm[:, 1]
    return pos, vel


# =============================================================================
# ORACLE LABEL (non-causal smooth — ML training target)
# =============================================================================

def compute_oracle_label(indicator_tf: np.ndarray):
    """
    Compute non-causal oracle labels AND continuous slope from a resampled indicator series.

    Uses kf.smooth() — NON-CAUSAL by design (RTS smoother, uses future data).

    Returns:
        labels_binary: int array (0 or 1), label[t] = 1 if smoothed[t-1] > smoothed[t-2]
        slope_continuous: float32 array, slope[t] = smoothed[t-1] - smoothed[t-2]
    """
    from pykalman import KalmanFilter as KF

    n = len(indicator_tf)
    labels_binary = np.zeros(n, dtype=int)
    slope_continuous = np.full(n, np.nan, dtype=np.float32)

    valid = ~np.isnan(indicator_tf)
    if valid.sum() < 3:
        return labels_binary, slope_continuous

    vd = indicator_tf[valid]

    kf = KF(
        transition_matrices=[[1, 1], [0, 1]],
        observation_matrices=[[1, 0]],
        initial_state_mean=[vd[0], 0.0],
        initial_state_covariance=np.eye(2),
        observation_covariance=KALMAN_LABEL_MEASURE_VAR,
        transition_covariance=np.eye(2) * KALMAN_LABEL_PROCESS_VAR,
    )

    # SMOOTH (non-causal, RTS smoother) — INTENTIONAL for labels
    smoothed_means, _ = kf.smooth(vd)
    smoothed = np.full(n, np.nan)
    smoothed[valid] = smoothed_means[:, 0]

    for t in range(2, n):
        if not np.isnan(smoothed[t - 1]) and not np.isnan(smoothed[t - 2]):
            delta = smoothed[t - 1] - smoothed[t - 2]
            slope_continuous[t] = delta
            labels_binary[t] = 1 if delta > 0 else 0

    return labels_binary, slope_continuous


# =============================================================================
# LIVE OHLCV
# =============================================================================

def compute_live_ohlcv(df_5min, tf_minutes):
    """Live-style OHLCV: partial candle updated every 5min. No shift."""
    group = df_5min.index.floor(f'{tf_minutes}min')
    r = pd.DataFrame(index=df_5min.index)
    r['open'] = df_5min.groupby(group)['open'].transform('first')
    r['high'] = df_5min.groupby(group)['high'].cummax()
    r['low'] = df_5min.groupby(group)['low'].cummin()
    r['close'] = df_5min['close']
    r['volume'] = df_5min.groupby(group)['volume'].cumsum()
    return r


# =============================================================================
# BUCKET CLOSURE MASK (replaces step_index == max_step)
# =============================================================================

def compute_bucket_close_mask(index_5min, tf_minutes):
    """
    Detect last bar of each tf bucket. Handles gaps correctly.
    Unlike step==max_step, works even when bars are missing.
    """
    bucket = index_5min.floor(f'{tf_minutes}min').values
    next_bucket = np.append(bucket[1:], np.datetime64('NaT'))
    return (bucket != next_bucket) | pd.isna(next_bucket)


def compute_step_index(index_5min, tf_minutes):
    """Position (1-based) within tf candle. For CSV output only, NOT closure logic."""
    minutes = index_5min.minute + index_5min.hour * 60
    return pd.Series((minutes % tf_minutes) // 5 + 1, index=index_5min, dtype=int)


# =============================================================================
# LIVE MACD
# =============================================================================

def compute_macd_live(close_5min, is_close):
    """
    MACD histogram with frozen/provisional EMA. Freeze at bucket closure.
    is_close: boolean array from compute_bucket_close_mask.
    """
    n = len(close_5min)
    alpha_f = 2.0 / (MACD_FAST + 1)
    alpha_s = 2.0 / (MACD_SLOW + 1)
    alpha_sig = 2.0 / (MACD_SIGNAL + 1)

    out = np.full(n, np.nan)
    ema_f_cl = np.nan; ema_s_cl = np.nan; ema_sig_cl = np.nan
    init = False

    for i in range(n):
        c = close_5min[i]
        if np.isnan(c):
            continue
        if not init:
            if is_close[i]:
                ema_f_cl = c; ema_s_cl = c; ema_sig_cl = 0.0
                out[i] = 0.0; init = True
            continue
        ef = alpha_f * c + (1.0 - alpha_f) * ema_f_cl
        es = alpha_s * c + (1.0 - alpha_s) * ema_s_cl
        ml = ef - es
        esg = alpha_sig * ml + (1.0 - alpha_sig) * ema_sig_cl
        out[i] = ml - esg
        if is_close[i]:
            ema_f_cl = ef; ema_s_cl = es; ema_sig_cl = esg
    return out


# =============================================================================
# LIVE RSI
# =============================================================================

def compute_rsi_live(close_5min, is_close):
    """
    RSI with frozen/provisional EWM avg_gain/avg_loss. Freeze at closure.

    Same approach as Kalman: compute exact state on ALL closures first,
    then replay for provisional (inter-closure) points.
    This guarantees exact match with standard RSI at closures (atol=1e-10).
    """
    n = len(close_5min)
    alpha = 2.0 / (RSI_PERIOD + 1)

    out = np.full(n, np.nan)

    # Step 1: Collect ALL closure closes
    closure_indices = []
    closure_closes = []
    for i in range(n):
        if not np.isnan(close_5min[i]) and is_close[i]:
            closure_indices.append(i)
            closure_closes.append(close_5min[i])

    if len(closure_closes) < 2:
        return out

    # Step 2: Compute standard RSI states on closure closes
    # This matches ewm(span=RSI_PERIOD, adjust=False) exactly
    closes_arr = np.array(closure_closes)
    deltas = np.diff(closes_arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # ewm(adjust=False) starts: EMA[0] = x[0]
    # But delta[0] = close[1]-close[0], gain[0] is for candle 1 (not 0)
    # Standard RSI: gain = delta.where(delta>0, 0), first delta is NaN → gain[0]=0
    # Then ewm starts with EMA[0] = 0 (the NaN-derived zero)
    # Our gains start from the first real diff. Prepend 0 to match.
    gains_padded = np.concatenate([[0.0], gains])
    losses_padded = np.concatenate([[0.0], losses])

    # Run EWM to get avg_gain/avg_loss at each closure
    # closure_states[k] = (avg_gain, avg_loss, close) after processing closure k
    ag = gains_padded[0]  # = 0, matches ewm init
    al = losses_padded[0]  # = 0
    closure_states = [(ag, al, closure_closes[0])]  # state after first closure

    for k in range(1, len(gains_padded)):
        ag = alpha * gains_padded[k] + (1.0 - alpha) * ag
        al = alpha * losses_padded[k] + (1.0 - alpha) * al
        closure_states.append((ag, al, closure_closes[k]))

    # Step 3: Assign RSI at closure points
    # Match standard behavior: when avg_loss=0, RSI=NaN (not 100.0)
    # Standard: rs = avg_gain / avg_loss.replace(0, np.nan) → NaN when avg_loss=0
    for k, ci in enumerate(closure_indices):
        ag_k, al_k, _ = closure_states[k]
        if al_k > 1e-15:
            out[ci] = 100.0 - 100.0 / (1.0 + ag_k / al_k)
        # else: stay NaN (matches standard RSI behavior)

    # Step 4: Replay for provisional (inter-closure) points
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

        # Provisional from frozen state
        if current_k >= 0 and not np.isnan(prev_cl):
            delta = c - prev_cl
            gn = max(delta, 0.0)
            ls = max(-delta, 0.0)
            ag_p = alpha * gn + (1.0 - alpha) * ag_cl
            al_p = alpha * ls + (1.0 - alpha) * al_cl
            if al_p > 1e-15:
                out[i] = 100.0 - 100.0 / (1.0 + ag_p / al_p)
            # else: stay NaN (matches standard RSI behavior)

    return out


# =============================================================================
# LIVE CCI
# =============================================================================

def compute_cci_live(high_live, low_live, close_5min, is_close):
    """CCI with rolling TP buffer. Freeze (push TP) at closure."""
    n = len(close_5min)
    out = np.full(n, np.nan)
    tp_buf = deque(maxlen=CCI_PERIOD - 1)

    for i in range(n):
        c = close_5min[i]; h = high_live[i]; lo = low_live[i]
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


# =============================================================================
# LIVE KALMAN
# =============================================================================

def compute_kalman_live(indicator_live, is_close, aq_window=30, Q_max_factor=10.0):
    """
    AQ-KF (adaptive Q, Myers-Tapley) with frozen/provisional state.
    Freeze at closure. Same interface as original compute_kalman_live.

    DIFFERENCE vs original: Q adapts online via innovation statistics
    instead of fixed Q. Clipped to [Q*0.1, Q*10].

    Returns (n, 2) array: [position, velocity] at each 5min step.
    """
    _A = np.array([[1.0, 1.0], [0.0, 1.0]])
    _H = np.array([[1.0, 0.0]])
    _Q_fixed = np.eye(2) * KALMAN_PROCESS_VAR
    _R = np.array([[KALMAN_MEASURE_VAR]])
    Q_FLOOR = _Q_fixed * 0.1
    Q_CEIL = _Q_fixed * Q_max_factor

    n = len(indicator_live)
    out = np.full((n, 2), np.nan)

    # --- Step 1: AQ-KF forward on closure values ---
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

    # Forward filter with adaptive Q on closures
    x_filt_cl = np.zeros((nc, 2))
    P_filt_cl = np.zeros((nc, 2, 2))
    Q_current = _Q_fixed.copy()
    innovation_buffer = []

    for k in range(nc):
        if k == 0:
            x_p = np.array([cv[0], 0.0])
            P_p = np.eye(2)
        else:
            x_p = _A @ x_filt_cl[k - 1]
            P_p = _A @ P_filt_cl[k - 1] @ _A.T + Q_current

        # Update
        y = cv[k] - _H @ x_p
        S = (_H @ P_p @ _H.T + _R)[0, 0]
        K = P_p @ _H.T / S
        x_filt_cl[k] = x_p + (K @ y).ravel()
        P_filt_cl[k] = (np.eye(2) - K @ _H) @ P_p

        # Innovation for adaptive Q
        v_t = cv[k] - (_H @ x_p)[0]
        innovation_buffer.append(v_t)
        if len(innovation_buffer) > aq_window:
            innovation_buffer.pop(0)

        if len(innovation_buffer) >= aq_window and k > 0:
            C_vv = np.mean(np.array(innovation_buffer) ** 2)
            delta = C_vv - S
            if delta > 0:
                P_pred_next = _A @ P_filt_cl[k] @ _A.T + Q_current
                det = P_pred_next[0, 0] * P_pred_next[1, 1] - P_pred_next[0, 1] * P_pred_next[1, 0]
                if abs(det) > 1e-15:
                    inv_P = np.array([[P_pred_next[1, 1], -P_pred_next[0, 1]],
                                      [-P_pred_next[1, 0], P_pred_next[0, 0]]]) / det
                else:
                    inv_P = np.linalg.pinv(P_pred_next)
                C_rts = P_filt_cl[k] @ _A.T @ inv_P
                Q_candidate = delta * (C_rts @ C_rts.T)
                if Q_candidate[0, 0] >= 0 and Q_candidate[1, 1] >= 0:
                    Q_current = np.clip(Q_candidate, Q_FLOOR, Q_CEIL)

    # --- Step 2: Assign output at closure points ---
    for k, ci in enumerate(closure_indices):
        out[ci, 0] = x_filt_cl[k, 0]
        out[ci, 1] = x_filt_cl[k, 1]

    # --- Step 3: Provisional (non-closure) from frozen AQ-KF state ---
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
        # Provisional: predict + update from frozen state
        if current_k >= 0:
            x_p = _A @ sm_cl
            P_p = _A @ sc_cl @ _A.T + Q_current
            y = obs - (_H @ x_p)[0]
            S = (_H @ P_p @ _H.T + _R)[0, 0]
            K = P_p @ _H.T / S
            sm_p = x_p + (K * y).ravel()
            out[i, 0] = sm_p[0]
            out[i, 1] = sm_p[1]

    return out


# =============================================================================
# VALIDATION
# =============================================================================

def validate_at_closure(name, live_vals, std_vals, suffix):
    """Compare aligned live vs standard values at closure. atol=1e-10."""
    lv = ~np.isnan(live_vals); sv = ~np.isnan(std_vals)
    if not lv.any() or not sv.any():
        logger.warning(f"    SKIP {name}_{suffix}: no valid values"); return True
    s = max(np.argmax(lv), np.argmax(sv))
    e = min(len(live_vals), len(std_vals))
    if s >= e:
        logger.warning(f"    SKIP {name}_{suffix}: no overlap"); return True
    l = live_vals[s:e]; v = std_vals[s:e]
    b = ~np.isnan(l) & ~np.isnan(v)
    if not b.any():
        logger.warning(f"    SKIP {name}_{suffix}: all NaN"); return True
    lc = l[b]; vc = v[b]
    md = np.max(np.abs(lc - vc))
    ok = np.allclose(lc, vc, atol=1e-10)
    if ok:
        logger.info(f"    PASS {name}_{suffix}: max_diff={md:.2e}, n={len(lc):,}")
    else:
        logger.error(f"    FAIL {name}_{suffix}: max_diff={md:.2e}, n={len(lc):,}")
        d = np.abs(lc - vc)
        for idx in np.argsort(d)[-3:]:
            logger.error(f"         [{idx}] live={lc[idx]:.15f} std={vc[idx]:.15f}")
    return ok


def run_validation(result, df_5min, tf_minutes, suffix, indicators):
    """
    Validate live values at bucket closure vs standard resample.
    Uses timestamp alignment to handle gaps.
    """
    # Get closure mask and extract live values at closure points
    is_close = compute_bucket_close_mask(df_5min.index, tf_minutes)
    live_at_cl = result.loc[is_close].copy()
    live_at_cl['tf_ts'] = live_at_cl.index.floor(f'{tf_minutes}min')

    # Standard reference
    df_tf = resample_ohlcv(df_5min, tf_minutes)

    # Align by timestamp
    common = set(live_at_cl['tf_ts'].values) & set(df_tf.index.values)
    logger.info(f"\n  --- Validation {suffix} ---")
    logger.info(f"    Alignment: {len(common):,} common candles "
                f"(live closures: {len(live_at_cl):,}, standard: {len(df_tf):,})")

    # Build aligned index maps
    live_ts_map = {ts: idx for idx, ts in enumerate(live_at_cl['tf_ts'].values)}
    std_mask = np.array([ts in common for ts in df_tf.index.values])
    live_order = [live_ts_map[ts] for ts in df_tf.index.values[std_mask]]

    def compare(name, live_col, std_series):
        la = live_at_cl[live_col].values[live_order]
        sa = std_series.values[std_mask]
        return validate_at_closure(name, la, sa, suffix)

    ok = True
    if 'macd' in indicators:
        ms_raw = calculate_macd_standard(df_tf)
        # AQ-KF: raw MACD (no normalization)
        ok &= compare("MACD", f'macd_{suffix}_live', ms_raw)
        # NOTE: Kalman validation skipped — AQ-KF produces different values
        # than fixed-Q pykalman by design. Only indicator values are validated.
    if 'rsi' in indicators:
        ok &= compare("RSI", f'rsi_{suffix}_live', calculate_rsi_standard(df_tf))
    if 'cci' in indicators:
        ok &= compare("CCI", f'cci_{suffix}_live', calculate_cci_standard(df_tf))

    if ok:
        logger.info(f"  ALL {suffix} VALIDATIONS PASSED")
    else:
        logger.error(f"  SOME {suffix} VALIDATIONS FAILED")
    return ok


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def generate_multitf_csv(asset_name, output_dir, indicators=None):
    if indicators is None:
        indicators = ['macd', 'rsi', 'cci']

    file_path = AVAILABLE_ASSETS_5M[asset_name]
    logger.info(f"\n{'='*60}")
    logger.info(f"  ASSET: {asset_name} | INDICATORS: {[i.upper() for i in indicators]}")
    logger.info(f"{'='*60}")

    df_5min = load_csv_5min(file_path, asset_name)
    close_5min = df_5min['close'].values

    result = pd.DataFrame(index=df_5min.index)
    result['open'] = df_5min['open']
    result['high'] = df_5min['high']
    result['low'] = df_5min['low']
    result['close'] = df_5min['close']
    result['volume'] = df_5min['volume']

    for tf_minutes, suffix in [(30, '30m'), (60, '1h')]:
        logger.info(f"\n  --- Timeframe {suffix} ---")

        # Live OHLCV
        ohlcv = compute_live_ohlcv(df_5min, tf_minutes)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            result[f'{col}_{suffix}_live'] = ohlcv[col]

        # Step index (CSV output only) + closure mask (logic)
        result[f'step_{suffix}'] = compute_step_index(df_5min.index, tf_minutes)
        is_close = compute_bucket_close_mask(df_5min.index, tf_minutes)
        n_buckets = is_close.sum()
        logger.info(f"    Live OHLCV computed, {n_buckets:,} bucket closures detected")

        # Live indicators
        high_live = ohlcv['high'].values
        low_live = ohlcv['low'].values
        ind_results = {}

        if 'macd' in indicators:
            logger.info(f"    Computing MACD live...")
            macd_raw = compute_macd_live(close_5min, is_close)
            # AQ-KF: use RAW MACD (no normalization) to match FLKS tests
            ind_results['macd'] = macd_raw
            logger.info(f"      MACD RAW (no normalization, aligned with FLKS tests)")
        if 'rsi' in indicators:
            logger.info(f"    Computing RSI live...")
            ind_results['rsi'] = compute_rsi_live(close_5min, is_close)
        if 'cci' in indicators:
            logger.info(f"    Computing CCI live...")
            ind_results['cci'] = compute_cci_live(high_live, low_live, close_5min, is_close)

        for ind, vals in ind_results.items():
            result[f'{ind}_{suffix}_live'] = vals

        # Kalman on each indicator — extract both position and velocity
        for ind, vals in ind_results.items():
            logger.info(f"    Computing Kalman on {ind}_{suffix}...")
            kalman_out = compute_kalman_live(vals, is_close)  # (n, 2) = [position, velocity]
            filt = kalman_out[:, 0]  # position
            vel = kalman_out[:, 1]   # velocity (slope estimate)
            result[f'{ind}_{suffix}_filtered'] = filt
            result[f'{ind}_{suffix}_velocity'] = vel

            # Direction label (from position)
            fs = pd.Series(filt, index=df_5min.index)
            lab = (fs > fs.shift(1)).astype(float)
            lab.iloc[0] = 0
            result[f'{ind}_{suffix}_label'] = lab.fillna(0).astype(int)

        # Stats
        for ind in ind_results:
            nc = (result[f'{ind}_{suffix}_label'].diff().abs() > 0).sum()
            logger.info(f"    {ind.upper()} label changes: {nc:,}")

        # Validation (live features only)
        run_validation(result, df_5min, tf_minutes, suffix, indicators)

        # =================================================================
        # ORACLE LABELS (non-causal smooth — ML training targets)
        # =================================================================
        logger.info(f"    Computing oracle labels ({suffix}, Kalman SMOOTH)...")
        df_tf = resample_ohlcv(df_5min, tf_minutes)

        for ind_name in indicators:
            if ind_name == 'macd':
                macd_std = calculate_macd_standard(df_tf).values
                # AQ-KF: use RAW MACD (no normalization, aligned with FLKS tests)
                ind_tf_values = macd_std
            elif ind_name == 'rsi':
                ind_tf_values = calculate_rsi_standard(df_tf).values
            elif ind_name == 'cci':
                ind_tf_values = calculate_cci_standard(df_tf).values
            else:
                continue

            # Compute labels + slope at tf resolution (non-causal)
            labels_tf, slope_tf = compute_oracle_label(ind_tf_values)

            # Forward-fill to 5min resolution
            # No shift — label is non-causal by construction
            labels_series = pd.Series(labels_tf, index=df_tf.index)
            labels_5min = labels_series.reindex(df_5min.index, method='ffill').fillna(0).astype(int)
            result[f'oracle_label_{ind_name}_{suffix}'] = labels_5min.values

            # Forward-fill slope to 5min resolution
            slope_series = pd.Series(slope_tf, index=df_tf.index)
            slope_5min = slope_series.reindex(df_5min.index, method='ffill')
            result[f'oracle_slope_{ind_name}_{suffix}'] = slope_5min.values

            n_up = (labels_5min == 1).sum()
            n_down = (labels_5min == 0).sum()
            n_changes = (labels_5min.diff().abs() > 0).sum()
            logger.info(f"      oracle_label_{ind_name}_{suffix}: {n_up:,} UP, {n_down:,} DOWN "
                        f"({n_up/(n_up+n_down)*100:.1f}% UP), {n_changes:,} direction changes")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    asset_fn = file_path.split('/')[-1].replace('_all_5m.csv', '')
    ind_tag = '_'.join(indicators)
    out_path = os.path.join(output_dir, f'{asset_fn}_multitf_{ind_tag}.csv')
    result.reset_index().to_csv(out_path, index=False)
    mb = os.path.getsize(out_path) / (1024**2)
    logger.info(f"\n  Saved: {out_path} ({mb:.1f} MB, {len(result):,} rows, {len(result.columns)} cols)")
    return out_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Prepare multi-timeframe CSV (live-style)')
    parser.add_argument('--assets', nargs='+', default=['BTC'])
    parser.add_argument('--indicators', nargs='+', default=['macd', 'rsi', 'cci'],
                        choices=['macd', 'rsi', 'cci'],
                        help='Indicators to compute (default: all)')
    parser.add_argument('--output-dir', type=str, default=PREPARED_DATA_DIR)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MULTI-TIMEFRAME CSV PREPARATION (LIVE-STYLE)")
    logger.info("=" * 60)
    logger.info(f"Assets: {args.assets}")
    logger.info(f"Indicators: {[i.upper() for i in args.indicators]}")
    logger.info(f"Kalman: PROCESS_VAR={KALMAN_PROCESS_VAR}, MEASURE_VAR={KALMAN_MEASURE_VAR}")

    for asset in args.assets:
        if asset not in AVAILABLE_ASSETS_5M:
            logger.warning(f"Asset {asset} not found, skipping")
            continue
        generate_multitf_csv(asset, args.output_dir, args.indicators)


if __name__ == '__main__':
    main()
