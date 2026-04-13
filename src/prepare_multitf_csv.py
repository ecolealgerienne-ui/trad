#!/usr/bin/env python3
"""
Multi-Timeframe CSV Preparation: live-style 30min and 1h data at 5min resolution.

PURPOSE:
    Generate one enriched CSV per asset reproducing what Binance API returns
    when querying 30min/1h klines every 5min: the last candle is the one
    currently forming, updated progressively.

LIVE-STYLE APPROACH:
    At each 5min step, the current tf candle is partially built:
        open_live  = open of first 5min bar in current tf candle (fixed)
        high_live  = cummax of highs within current tf candle
        low_live   = cummin of lows within current tf candle
        close_live = close of current 5min bar (latest price)
        volume_live = cumsum of volumes within current tf candle

    Indicators (MACD, RSI, CCI) use incremental EMA with frozen/provisional
    states. The EMA advances ONLY at candle closures (step==6 for 30min,
    step==12 for 1h). Between closures, provisional values are computed from
    the frozen state + current live close. They do NOT accumulate.

    Kalman filter follows the same freeze logic: filter_update from frozen
    state at each 5min step, freeze at candle closure. Forward-only (causal).

CAUSALITY:
    close_live[i] = close_5min[i], known at time i. Backtest at open[i+1].
    No shift(1) needed on live columns.
    Kalman uses filter_update (forward-only), never smooth().

VALIDATION:
    At candle closure (step==max_step), all live values must match standard
    resample-then-compute values exactly (atol=1e-10). This proves the EMA
    and Kalman only advance at closures.

OUTPUT COLUMNS:
    5min raw: open, high, low, close, volume
    Per indicator (macd/rsi/cci) x per timeframe (30m/1h):
        {ind}_{tf}_live         — indicator value (live EMA)
        {ind}_{tf}_filtered     — Kalman filtered indicator (live)
        {ind}_{tf}_label        — direction: filtered[i] > filtered[i-1]
    Per timeframe:
        open_{tf}_live, high_{tf}_live, low_{tf}_live, close_{tf}_live, volume_{tf}_live
        step_{tf}

Usage:
    python src/prepare_multitf_csv.py --assets BTC ETH BNB ADA LTC
    python src/prepare_multitf_csv.py --assets BTC
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

# Kalman parameters (same as project-wide defaults)
KALMAN_PROCESS_VAR = 0.01
KALMAN_MEASURE_VAR = 0.1


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
# RESAMPLING (for standard reference computation in validation)
# =============================================================================

def resample_ohlcv(df_5min: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
    """Resample 5min to higher timeframe with standard OHLCV aggregation."""
    df_tf = df_5min.resample(f'{tf_minutes}min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()
    return df_tf


# =============================================================================
# STANDARD INDICATORS (for validation reference)
# =============================================================================

def calculate_macd_standard(df: pd.DataFrame) -> pd.Series:
    """Standard MACD on resampled DataFrame. Used for validation only."""
    ema_fast = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return macd_line - signal_line


def calculate_rsi_standard(df: pd.DataFrame) -> pd.Series:
    """Standard RSI on resampled DataFrame. Used for validation only."""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(span=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(span=RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_cci_standard(df: pd.DataFrame) -> pd.Series:
    """Standard CCI on resampled DataFrame. Used for validation only."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(CCI_PERIOD).mean()
    mad = tp.rolling(CCI_PERIOD).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma_tp) / (0.015 * mad)


def kalman_filter_standard(data: np.ndarray) -> np.ndarray:
    """Standard forward-only Kalman filter on a 1D series. For validation."""
    from pykalman import KalmanFilter as KF

    valid_mask = ~np.isnan(data)
    if valid_mask.sum() < 2:
        return np.full(len(data), np.nan)

    valid_data = data[valid_mask]

    kf = KF(
        transition_matrices=[[1, 1], [0, 1]],
        observation_matrices=[[1, 0]],
        initial_state_mean=[valid_data[0], 0.0],
        initial_state_covariance=np.eye(2),
        observation_covariance=KALMAN_MEASURE_VAR,
        transition_covariance=np.eye(2) * KALMAN_PROCESS_VAR,
    )

    # Forward-only filter (not smooth!)
    state_means, _ = kf.filter(valid_data)

    result = np.full(len(data), np.nan)
    result[valid_mask] = state_means[:, 0]
    return result


# =============================================================================
# LIVE OHLCV
# =============================================================================

def compute_live_ohlcv(df_5min: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
    """
    Compute live-style OHLCV at 5min resolution for a higher timeframe.
    Reproduces Binance API behavior: last candle is partially formed.
    No shift(1) — close_live = close_5min, known at time i.
    """
    group = df_5min.index.floor(f'{tf_minutes}min')
    result = pd.DataFrame(index=df_5min.index)
    result['open'] = df_5min.groupby(group)['open'].transform('first')
    result['high'] = df_5min.groupby(group)['high'].cummax()
    result['low'] = df_5min.groupby(group)['low'].cummin()
    result['close'] = df_5min['close']
    result['volume'] = df_5min.groupby(group)['volume'].cumsum()
    return result


# =============================================================================
# STEP INDEX
# =============================================================================

def compute_step_index(index_5min: pd.DatetimeIndex, tf_minutes: int) -> pd.Series:
    """Position (1-based) of each 5min bar within its parent tf candle."""
    minutes = index_5min.minute + index_5min.hour * 60
    return pd.Series((minutes % tf_minutes) // 5 + 1, index=index_5min, dtype=int)


# =============================================================================
# LIVE MACD (incremental EMA, freeze at candle closure)
# =============================================================================

def compute_macd_live(close_5min: np.ndarray, step_index: np.ndarray,
                      max_step: int) -> np.ndarray:
    """
    MACD histogram on live tf candles at 5min resolution.

    EMA state advances ONLY at step==max_step (candle closure). Between
    closures, provisional values start from frozen state + current close.
    Provisionals do NOT accumulate across 5min steps.

    At step==max_step, output matches standard MACD on resampled data.
    """
    n = len(close_5min)
    alpha_f = 2.0 / (MACD_FAST + 1)
    alpha_s = 2.0 / (MACD_SLOW + 1)
    alpha_sig = 2.0 / (MACD_SIGNAL + 1)

    out = np.full(n, np.nan)
    ema_f_closed = np.nan
    ema_s_closed = np.nan
    ema_sig_closed = np.nan
    init = False

    for i in range(n):
        c = close_5min[i]
        if np.isnan(c):
            continue

        # Init on first completed candle (matches ewm(adjust=False): EMA[0]=x[0])
        if not init:
            if step_index[i] == max_step:
                ema_f_closed = c
                ema_s_closed = c
                ema_sig_closed = 0.0
                out[i] = 0.0
                init = True
            continue

        # Provisional from frozen state
        ema_f_prov = alpha_f * c + (1.0 - alpha_f) * ema_f_closed
        ema_s_prov = alpha_s * c + (1.0 - alpha_s) * ema_s_closed
        macd_line = ema_f_prov - ema_s_prov
        ema_sig_prov = alpha_sig * macd_line + (1.0 - alpha_sig) * ema_sig_closed
        out[i] = macd_line - ema_sig_prov

        # Freeze at closure
        if step_index[i] == max_step:
            ema_f_closed = ema_f_prov
            ema_s_closed = ema_s_prov
            ema_sig_closed = ema_sig_prov

    return out


# =============================================================================
# LIVE RSI (incremental EWM avg_gain/avg_loss, freeze at candle closure)
# =============================================================================

def compute_rsi_live(close_5min: np.ndarray, step_index: np.ndarray,
                     max_step: int) -> np.ndarray:
    """
    RSI on live tf candles at 5min resolution.

    Uses EWM alpha=2/(period+1) for consistency with existing pipeline.
    avg_gain/avg_loss frozen between closures. delta = close_live - prev_candle_close.
    """
    n = len(close_5min)
    alpha = 2.0 / (RSI_PERIOD + 1)

    out = np.full(n, np.nan)
    avg_gain_closed = np.nan
    avg_loss_closed = np.nan
    prev_candle_close = np.nan
    candle_closes = []
    warmed_up = False

    for i in range(n):
        c = close_5min[i]
        if np.isnan(c):
            continue

        # Warm-up: collect first RSI_PERIOD+1 completed candle closes
        if not warmed_up:
            if step_index[i] == max_step:
                candle_closes.append(c)
                if len(candle_closes) >= RSI_PERIOD + 1:
                    # Init EWM from collected closes (same as ewm(adjust=False))
                    closes_arr = np.array(candle_closes)
                    deltas = np.diff(closes_arr)
                    gains = np.where(deltas > 0, deltas, 0.0)
                    losses = np.where(deltas < 0, -deltas, 0.0)
                    ag = gains[0]
                    al = losses[0]
                    for k in range(1, len(gains)):
                        ag = alpha * gains[k] + (1.0 - alpha) * ag
                        al = alpha * losses[k] + (1.0 - alpha) * al
                    avg_gain_closed = ag
                    avg_loss_closed = al
                    prev_candle_close = c
                    if al > 1e-15:
                        out[i] = 100.0 - 100.0 / (1.0 + ag / al)
                    else:
                        out[i] = 100.0
                    warmed_up = True
            continue

        # Provisional from frozen state
        delta = c - prev_candle_close
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        ag_prov = alpha * gain + (1.0 - alpha) * avg_gain_closed
        al_prov = alpha * loss + (1.0 - alpha) * avg_loss_closed

        if al_prov > 1e-15:
            out[i] = 100.0 - 100.0 / (1.0 + ag_prov / al_prov)
        else:
            out[i] = 100.0

        # Freeze at closure
        if step_index[i] == max_step:
            avg_gain_closed = ag_prov
            avg_loss_closed = al_prov
            prev_candle_close = c

    return out


# =============================================================================
# LIVE CCI (rolling buffer of TPs, freeze at candle closure)
# =============================================================================

def compute_cci_live(high_live: np.ndarray, low_live: np.ndarray,
                     close_5min: np.ndarray, step_index: np.ndarray,
                     max_step: int) -> np.ndarray:
    """
    CCI on live tf candles at 5min resolution.

    Buffer of CCI_PERIOD-1 frozen TPs + 1 live TP. TP uses tf-level
    high/low (cummax/cummin), not raw 5min. NaN until buffer full.
    """
    n = len(close_5min)
    out = np.full(n, np.nan)
    tp_buffer = deque(maxlen=CCI_PERIOD - 1)

    for i in range(n):
        c = close_5min[i]
        h = high_live[i]
        lo = low_live[i]

        if np.isnan(c) or np.isnan(h) or np.isnan(lo):
            continue

        tp_live = (h + lo + c) / 3.0

        if len(tp_buffer) >= CCI_PERIOD - 1:
            all_tps = np.array(list(tp_buffer) + [tp_live])
            sma = all_tps.mean()
            mad = np.abs(all_tps - sma).mean()
            if mad > 1e-15:
                out[i] = (tp_live - sma) / (0.015 * mad)
            else:
                out[i] = 0.0

        # Freeze: push completed candle TP into buffer
        if step_index[i] == max_step:
            tp_buffer.append(tp_live)

    return out


# =============================================================================
# LIVE KALMAN (filter_update with freeze at candle closure)
# =============================================================================

def compute_kalman_live(indicator_live: np.ndarray, step_index: np.ndarray,
                        max_step: int) -> np.ndarray:
    """
    Kalman filter on live tf indicator at 5min resolution.

    Same freeze logic as indicators: filter_update from frozen state at each
    5min step, freeze at candle closure (step==max_step). Forward-only (causal).

    The Kalman processes ~147k candle closures (for 30min), not 880k 5min steps.
    Between closures, provisional filtered values start from frozen Kalman state
    but consume different indicator_live observations.

    At step==max_step, matches standard kf.filter() on the resampled series.
    """
    from pykalman import KalmanFilter as KF

    n = len(indicator_live)
    out = np.full(n, np.nan)

    # Find first valid value to initialize
    first_valid_idx = -1
    for i in range(n):
        if not np.isnan(indicator_live[i]):
            first_valid_idx = i
            break
    if first_valid_idx < 0:
        return out

    # Build KalmanFilter object (needed for filter_update method)
    init_val = indicator_live[first_valid_idx]
    # observation_covariance as scalar to match kalman_filter_standard exactly
    kf = KF(
        transition_matrices=np.array([[1, 1], [0, 1]]),
        observation_matrices=np.array([[1, 0]]),
        initial_state_mean=np.array([init_val, 0.0]),
        initial_state_covariance=np.eye(2),
        observation_covariance=KALMAN_MEASURE_VAR,
        transition_covariance=np.eye(2) * KALMAN_PROCESS_VAR,
    )

    # Frozen state (after last completed candle)
    state_mean_closed = np.array([init_val, 0.0])
    state_cov_closed = np.eye(2)
    init = False

    for i in range(n):
        obs = indicator_live[i]
        if np.isnan(obs):
            continue

        # Wait for first completed candle to initialize
        if not init:
            if step_index[i] == max_step:
                # First candle: run filter_update to get proper initial state
                state_mean_closed, state_cov_closed = kf.filter_update(
                    filtered_state_mean=state_mean_closed,
                    filtered_state_covariance=state_cov_closed,
                    observation=obs
                )
                out[i] = state_mean_closed[0]
                init = True
            continue

        # Provisional: filter_update from frozen state (does NOT accumulate)
        state_mean_prov, state_cov_prov = kf.filter_update(
            filtered_state_mean=state_mean_closed,
            filtered_state_covariance=state_cov_closed,
            observation=obs
        )
        out[i] = state_mean_prov[0]

        # Freeze at candle closure
        if step_index[i] == max_step:
            state_mean_closed = state_mean_prov
            state_cov_closed = state_cov_prov

    return out


# =============================================================================
# VALIDATION
# =============================================================================

def validate_at_closure(name: str, live_vals: np.ndarray, std_vals: np.ndarray,
                        suffix: str) -> bool:
    """
    Compare live values at candle closure vs standard reference.
    Returns True if all match within atol=1e-10.
    """
    live_valid = ~np.isnan(live_vals)
    std_valid = ~np.isnan(std_vals)

    if not live_valid.any() or not std_valid.any():
        logger.warning(f"    SKIP {name}_{suffix}: no valid values")
        return True

    # Align from where both are valid
    live_first = np.argmax(live_valid)
    std_first = np.argmax(std_valid)
    start = max(live_first, std_first)
    end = min(len(live_vals), len(std_vals))

    if start >= end:
        logger.warning(f"    SKIP {name}_{suffix}: no overlap")
        return True

    lv = live_vals[start:end]
    sv = std_vals[start:end]
    both = ~np.isnan(lv) & ~np.isnan(sv)

    if not both.any():
        logger.warning(f"    SKIP {name}_{suffix}: all NaN in range")
        return True

    lv_cmp = lv[both]
    sv_cmp = sv[both]
    max_diff = np.max(np.abs(lv_cmp - sv_cmp))
    ok = np.allclose(lv_cmp, sv_cmp, atol=1e-10)

    if ok:
        logger.info(f"    PASS {name}_{suffix}: max_diff={max_diff:.2e}, n={len(lv_cmp):,}")
    else:
        logger.error(f"    FAIL {name}_{suffix}: max_diff={max_diff:.2e}, n={len(lv_cmp):,}")
        diffs = np.abs(lv_cmp - sv_cmp)
        for idx in np.argsort(diffs)[-3:]:
            logger.error(f"         [{idx}] live={lv_cmp[idx]:.15f} std={sv_cmp[idx]:.15f}")
    return ok


def run_validation(result: pd.DataFrame, df_5min: pd.DataFrame,
                   tf_minutes: int, suffix: str, indicators: list = None):
    """
    Run validation checks for computed indicators on one timeframe.
    Only validates indicators that were actually computed.
    Compares live values at candle closure vs standard resample approach.
    """
    if indicators is None:
        indicators = ['macd', 'rsi', 'cci']

    max_step = tf_minutes // 5
    mask = result[f'step_{suffix}'] == max_step

    # Standard reference (resample then compute)
    df_tf = resample_ohlcv(df_5min, tf_minutes)

    logger.info(f"\n  --- Validation {suffix} (at step=={max_step}) ---")

    all_ok = True

    if 'macd' in indicators:
        macd_std = calculate_macd_standard(df_tf).values
        all_ok &= validate_at_closure("MACD", result.loc[mask, f'macd_{suffix}_live'].values, macd_std, suffix)
        # Kalman validation on MACD
        kalman_macd_std = kalman_filter_standard(macd_std)
        all_ok &= validate_at_closure("Kalman_MACD", result.loc[mask, f'macd_{suffix}_filtered'].values, kalman_macd_std, suffix)

    if 'rsi' in indicators:
        rsi_std = calculate_rsi_standard(df_tf).values
        all_ok &= validate_at_closure("RSI", result.loc[mask, f'rsi_{suffix}_live'].values, rsi_std, suffix)

    if 'cci' in indicators:
        cci_std = calculate_cci_standard(df_tf).values
        all_ok &= validate_at_closure("CCI", result.loc[mask, f'cci_{suffix}_live'].values, cci_std, suffix)

    if all_ok:
        logger.info(f"  ALL {suffix} VALIDATIONS PASSED")
    else:
        logger.error(f"  SOME {suffix} VALIDATIONS FAILED")

    return all_ok


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def generate_multitf_csv(asset_name: str, output_dir: str,
                         indicators: list = None) -> str:
    """
    Generate multi-timeframe CSV for one asset with live indicators,
    Kalman filtered values, and direction labels.

    Args:
        asset_name: 'BTC', 'ETH', etc.
        output_dir: Output directory
        indicators: List of indicators to compute, e.g. ['macd'], ['macd','rsi','cci']
                    Default: all three
    """
    if indicators is None:
        indicators = ['macd', 'rsi', 'cci']

    file_path = AVAILABLE_ASSETS_5M[asset_name]

    logger.info(f"\n{'='*60}")
    logger.info(f"  ASSET: {asset_name}")
    logger.info(f"  INDICATORS: {[i.upper() for i in indicators]}")
    logger.info(f"{'='*60}")

    # --- Step 1: Load 5min data ---
    df_5min = load_csv_5min(file_path, asset_name)
    close_5min = df_5min['close'].values

    result = pd.DataFrame(index=df_5min.index)
    result['open'] = df_5min['open']
    result['high'] = df_5min['high']
    result['low'] = df_5min['low']
    result['close'] = df_5min['close']
    result['volume'] = df_5min['volume']

    # --- Step 2-6: For each timeframe ---
    for tf_minutes, suffix in [(30, '30m'), (60, '1h')]:
        max_step = tf_minutes // 5
        logger.info(f"\n  --- Timeframe {suffix} (max_step={max_step}) ---")

        # Step 2: Live OHLCV (vectorized)
        live_ohlcv = compute_live_ohlcv(df_5min, tf_minutes)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            result[f'{col}_{suffix}_live'] = live_ohlcv[col]

        # Step 3: Step index
        step_idx = compute_step_index(df_5min.index, tf_minutes)
        result[f'step_{suffix}'] = step_idx
        step_arr = step_idx.values

        logger.info(f"    Live OHLCV + step index computed")

        # Step 4: Live indicators (only those requested)
        high_live = live_ohlcv['high'].values
        low_live = live_ohlcv['low'].values

        ind_results = {}  # name -> array, for Kalman step

        if 'macd' in indicators:
            logger.info(f"    Computing MACD live...")
            macd_live = compute_macd_live(close_5min, step_arr, max_step)
            result[f'macd_{suffix}_live'] = macd_live
            ind_results['macd'] = macd_live

        if 'rsi' in indicators:
            logger.info(f"    Computing RSI live...")
            rsi_live = compute_rsi_live(close_5min, step_arr, max_step)
            result[f'rsi_{suffix}_live'] = rsi_live
            ind_results['rsi'] = rsi_live

        if 'cci' in indicators:
            logger.info(f"    Computing CCI live...")
            cci_live = compute_cci_live(high_live, low_live, close_5min, step_arr, max_step)
            result[f'cci_{suffix}_live'] = cci_live
            ind_results['cci'] = cci_live

        # Step 5: Kalman on each computed indicator (freeze at closure)
        for ind_name, ind_vals in ind_results.items():
            logger.info(f"    Computing Kalman on {ind_name}_{suffix}...")
            filtered = compute_kalman_live(ind_vals, step_arr, max_step)
            result[f'{ind_name}_{suffix}_filtered'] = filtered

            # Step 6: Direction labels from filtered values
            filt_series = pd.Series(filtered, index=df_5min.index)
            label = (filt_series > filt_series.shift(1)).astype(float)
            label.iloc[0] = 0
            label = label.fillna(0).astype(int)
            result[f'{ind_name}_{suffix}_label'] = label

        # Stats
        if 'macd' in indicators:
            n_changes = (result[f'macd_{suffix}_label'].diff().abs() > 0).sum()
            logger.info(f"    MACD label changes: {n_changes:,}")

        # Step 7: Validation (only for computed indicators)
        run_validation(result, df_5min, tf_minutes, suffix, indicators)

    # --- Save CSV ---
    os.makedirs(output_dir, exist_ok=True)
    asset_filename = file_path.split('/')[-1].replace('_all_5m.csv', '')
    ind_tag = '_'.join(indicators)
    output_path = os.path.join(output_dir, f'{asset_filename}_multitf_{ind_tag}.csv')

    result_save = result.reset_index()
    result_save.to_csv(output_path, index=False)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"\n  Saved: {output_path} ({file_size_mb:.1f} MB)")
    logger.info(f"     Rows: {len(result):,}, Columns: {len(result.columns)}")
    logger.info(f"     Columns: {list(result.columns)}")

    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Prepare multi-timeframe CSV with live indicators + Kalman + labels'
    )
    parser.add_argument('--assets', nargs='+',
                        default=['BTC', 'ETH', 'BNB', 'ADA', 'LTC'],
                        help='Assets to process (default: all)')
    parser.add_argument('--indicators', nargs='+',
                        default=['macd', 'rsi', 'cci'],
                        choices=['macd', 'rsi', 'cci'],
                        help='Indicators to compute (default: all). Use --indicators macd for fast test.')
    parser.add_argument('--output-dir', type=str, default=PREPARED_DATA_DIR,
                        help=f'Output directory (default: {PREPARED_DATA_DIR})')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MULTI-TIMEFRAME CSV PREPARATION (LIVE-STYLE)")
    logger.info("=" * 60)
    logger.info(f"Assets: {args.assets}")
    logger.info(f"Indicators: {[i.upper() for i in args.indicators]}")
    logger.info(f"Timeframes: 30m (step 1-6), 1h (step 1-12)")
    logger.info(f"Kalman: PROCESS_VAR={KALMAN_PROCESS_VAR}, MEASURE_VAR={KALMAN_MEASURE_VAR}")
    logger.info(f"Output: {args.output_dir}/")

    for asset_name in args.assets:
        if asset_name not in AVAILABLE_ASSETS_5M:
            logger.warning(f"Asset {asset_name} not found, skipping")
            continue
        generate_multitf_csv(asset_name, args.output_dir, args.indicators)


if __name__ == '__main__':
    main()
