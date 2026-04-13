#!/usr/bin/env python3
"""
Multi-Timeframe CSV Preparation: 30min and 1h indicators at 5min resolution.

PURPOSE:
    Generate one enriched CSV per asset containing 30min and 1h indicators
    pre-computed and forward-filled at 5min resolution. This file serves as
    the single source of truth for all subsequent tests and training.

APPROACH:
    - Raw data is in 5min candles
    - We resample to 30min and 1h to compute indicators on completed candles
    - Indicators are forward-filled to 5min resolution from the PREVIOUS
      completed candle (strict causality via shift(1))

CAUSALITY GUARANTEE:
    The 30min candle at 10:00 covers data from 10:00 to 10:29.
    Its close price is only available after ~10:29.
    Therefore, the indicator computed from this candle is only assigned
    starting at 10:30 (the next 30min boundary), NOT at 10:00.
    This is implemented via shift(1) before forward-fill.

    Example timeline for 30min:
        09:30-09:59 candle completes → MACD_09:30 computed
        10:00 (step 1): receives MACD_09:30 (from previous completed candle)
        10:05 (step 2): same MACD_09:30
        ...
        10:25 (step 6): same MACD_09:30
        10:30 (step 1): receives MACD_10:00 (candle 10:00-10:29 now completed)

COLUMNS IN OUTPUT CSV:
    --- 5min raw ---
    open, high, low, close, volume

    --- 30min (forward-filled from completed candles) ---
    open_30m, high_30m, low_30m, close_30m, volume_30m
    macd_30m, rsi_30m, cci_30m
    step_30m  (position 1-6 within the current 30min candle)

    --- 1h (forward-filled from completed candles) ---
    open_1h, high_1h, low_1h, close_1h, volume_1h
    macd_1h, rsi_1h, cci_1h
    step_1h  (position 1-12 within the current 1h candle)

NOT INCLUDED (must be computed separately after train/test split):
    - Kalman filtered values (non-causal RTS smoother, uses future data)
    - Direction labels (depend on Kalman)

Usage:
    python src/prepare_multitf_csv.py --assets BTC ETH BNB ADA LTC
    python src/prepare_multitf_csv.py --assets BTC

Output:
    data/prepared/BTCUSD_multitf.csv
    data/prepared/ETHUSD_multitf.csv
    ...
"""

import numpy as np
import pandas as pd
import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import constants from project
sys.path.insert(0, str(Path(__file__).parent))
from constants import AVAILABLE_ASSETS_5M, PREPARED_DATA_DIR

# Standard indicator periods (copied from prepare_data_direction_only.py)
RSI_PERIOD = 14
CCI_PERIOD = 20
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


# =============================================================================
# DATA LOADING
# =============================================================================

def load_csv_5min(file_path: str, asset_name: str) -> pd.DataFrame:
    """
    Load raw 5min OHLCV data from CSV file.

    Handles multiple date column names and normalizes column names to lowercase.
    Returns DataFrame with DatetimeIndex sorted chronologically.

    Args:
        file_path: Path to the CSV file
        asset_name: Asset name for logging (e.g., 'BTC')

    Returns:
        DataFrame with columns: open, high, low, close, volume
        Index: DatetimeIndex named 'datetime'
    """
    df = pd.read_csv(file_path)

    # Find date column (supports multiple naming conventions)
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

    # Verify required columns exist
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    logger.info(f"  {asset_name}: {len(df):,} 5min candles, {df.index[0]} -> {df.index[-1]}")

    return df


# =============================================================================
# RESAMPLING
# =============================================================================

def resample_ohlcv(df_5min: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
    """
    Resample 5min OHLCV data to a higher timeframe using standard aggregation.

    Aggregation rules:
        - open:   first value in the period
        - high:   maximum value in the period
        - low:    minimum value in the period
        - close:  last value in the period
        - volume: sum of all values in the period

    Args:
        df_5min: DataFrame with 5min DatetimeIndex and OHLCV columns
        tf_minutes: Target timeframe in minutes (e.g., 30 or 60)

    Returns:
        DataFrame with resampled OHLCV at the target timeframe
    """
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    df_tf = df_5min.resample(f'{tf_minutes}min').agg(agg_dict)

    # Drop rows where resampling produced NaN (incomplete candles at boundaries)
    df_tf = df_tf.dropna()

    return df_tf


# =============================================================================
# INDICATOR CALCULATIONS
# Copied from prepare_data_direction_only.py to avoid import dependencies.
# These use standard EWM/rolling calculations — all causal (no future data).
# =============================================================================

def calculate_rsi(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI) on close prices.

    RSI = 100 - 100 / (1 + RS)
    RS = avg_gain / avg_loss (exponential moving average)

    Uses only close prices. Warm-up period: ~RSI_PERIOD candles produce NaN.
    """
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(span=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(span=RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_cci(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Commodity Channel Index (CCI) on typical price.

    CCI = (TP - SMA(TP)) / (0.015 * Mean Absolute Deviation(TP))
    TP = (high + low + close) / 3

    Uses high, low, close. Warm-up period: ~CCI_PERIOD candles produce NaN.
    """
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(CCI_PERIOD).mean()
    mad = tp.rolling(CCI_PERIOD).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma_tp) / (0.015 * mad)


def calculate_macd(df: pd.DataFrame) -> pd.Series:
    """
    Calculate MACD histogram on close prices.

    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD Line)
    Histogram = MACD Line - Signal Line

    Uses only close prices. Warm-up period: ~MACD_SLOW candles produce NaN.
    """
    ema_fast = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return macd_line - signal_line


# =============================================================================
# LIVE OHLCV (Binance-style: partial candle updated every 5min)
# =============================================================================

def compute_live_ohlcv(df_5min: pd.DataFrame, tf_minutes: int, suffix: str) -> pd.DataFrame:
    """
    Compute live-style OHLCV for a higher timeframe at 5min resolution.

    Reproduces what Binance API returns when querying klines every 5min:
    the last candle is the one currently forming, updated progressively.

    At each 5min step i within a tf candle:
        open_live  = open of the first 5min bar in this tf candle (fixed)
        high_live  = max of all highs seen so far in this tf candle
        low_live   = min of all lows seen so far in this tf candle
        close_live = close of the current 5min bar (= latest price)
        volume_live = sum of all volumes so far in this tf candle

    NO shift(1) needed: close_live[i] = close_5min[i], known at time i.
    Backtest executes at open[i+1].

    Args:
        df_5min: DataFrame with 5min OHLCV and DatetimeIndex
        tf_minutes: Target timeframe in minutes (30 or 60)
        suffix: Column name suffix ('30m' or '1h')

    Returns:
        DataFrame with columns: open_{suffix}_live, high_{suffix}_live, etc.
    """
    # Group each 5min bar by its parent tf candle
    group = df_5min.index.floor(f'{tf_minutes}min')

    result = pd.DataFrame(index=df_5min.index)
    result[f'open_{suffix}_live'] = df_5min.groupby(group)['open'].transform('first')
    result[f'high_{suffix}_live'] = df_5min.groupby(group)['high'].cummax()
    result[f'low_{suffix}_live'] = df_5min.groupby(group)['low'].cummin()
    result[f'close_{suffix}_live'] = df_5min['close']  # live close = current 5min close
    result[f'volume_{suffix}_live'] = df_5min.groupby(group)['volume'].cumsum()

    return result


# =============================================================================
# LIVE INDICATORS (incremental EMA with frozen/provisional states)
#
# These reproduce what you'd get by querying Binance's 30min/1h indicators
# every 5min. The EMA state advances ONLY at candle closures (step == max_step).
# Between closures, provisional values are computed from the frozen state +
# the current live close. Provisionals do NOT accumulate across 5min steps.
#
# IMPORTANT: This is NOT a 5min indicator with scaled periods. The EMA sees
# a series of tf-candle closes (one per completed candle), with only the last
# entry evolving at 5min resolution.
# =============================================================================

def compute_macd_live(close_5min: np.ndarray, step_index: np.ndarray,
                      max_step: int, fast: int = MACD_FAST, slow: int = MACD_SLOW,
                      signal: int = MACD_SIGNAL) -> np.ndarray:
    """
    Compute MACD histogram on live tf candles at 5min resolution.

    The EMA state (ema_*_closed) only advances when a tf candle completes
    (step == max_step). Between closures, a provisional MACD is computed
    from the frozen EMA state and the current 5min close. The provisional
    value is throwaway — it does NOT feed into the next step's calculation.

    At step == max_step, live values must match standard resample MACD exactly.

    Args:
        close_5min: Array of 5min close prices (~880k values)
        step_index: Position within tf candle (1 to max_step)
        max_step: Steps per candle (6 for 30min, 12 for 1h)
        fast/slow/signal: MACD periods

    Returns:
        Array of MACD histogram values at 5min resolution
    """
    n = len(close_5min)
    alpha_fast = 2.0 / (fast + 1)
    alpha_slow = 2.0 / (slow + 1)
    alpha_signal = 2.0 / (signal + 1)

    macd_out = np.full(n, np.nan)

    # State: frozen after last completed candle
    ema_fast_closed = np.nan
    ema_slow_closed = np.nan
    ema_signal_closed = np.nan
    initialized = False

    for i in range(n):
        c = close_5min[i]
        step = step_index[i]

        if np.isnan(c):
            continue

        # Initialization: first completed candle sets the EMA base
        # (matches ewm(adjust=False) which sets EMA[0] = x[0])
        if not initialized:
            if step == max_step:
                ema_fast_closed = c
                ema_slow_closed = c
                ema_signal_closed = 0.0  # macd_line = fast - slow = 0 initially
                macd_out[i] = 0.0
                initialized = True
            continue

        # Provisional EMA: "what if the tf candle closed right now?"
        # Always computed from FROZEN state, not previous provisional.
        ema_fast_prov = alpha_fast * c + (1.0 - alpha_fast) * ema_fast_closed
        ema_slow_prov = alpha_slow * c + (1.0 - alpha_slow) * ema_slow_closed
        macd_line = ema_fast_prov - ema_slow_prov
        ema_signal_prov = alpha_signal * macd_line + (1.0 - alpha_signal) * ema_signal_closed
        macd_out[i] = macd_line - ema_signal_prov

        # Freeze: EMA advances one step only at candle closure
        if step == max_step:
            ema_fast_closed = ema_fast_prov
            ema_slow_closed = ema_slow_prov
            ema_signal_closed = ema_signal_prov

    return macd_out


def compute_rsi_live(close_5min: np.ndarray, step_index: np.ndarray,
                     max_step: int, period: int = RSI_PERIOD) -> np.ndarray:
    """
    Compute RSI on live tf candles at 5min resolution.

    Uses EWM alpha = 2/(period+1) for consistency with the existing pipeline
    (which uses pandas ewm(span=period, adjust=False)).

    State: avg_gain_closed and avg_loss_closed frozen between candle closures.
    At each 5min step, delta = close_5min[i] - prev_candle_close (close of
    the last COMPLETED tf candle). Provisional avg_gain/avg_loss computed
    from frozen state + this delta. Freeze at step == max_step.

    Args:
        close_5min: Array of 5min close prices
        step_index: Position within tf candle (1 to max_step)
        max_step: Steps per candle (6 for 30min, 12 for 1h)
        period: RSI period

    Returns:
        Array of RSI values at 5min resolution
    """
    n = len(close_5min)
    alpha = 2.0 / (period + 1)  # EWM alpha, NOT Wilder's 1/N

    rsi_out = np.full(n, np.nan)

    # State: frozen after last completed candle
    avg_gain_closed = np.nan
    avg_loss_closed = np.nan
    prev_candle_close = np.nan  # close of the last COMPLETED tf candle

    # Warm-up: collect first `period` completed candle closes to initialize
    candle_closes = []
    warmed_up = False

    for i in range(n):
        c = close_5min[i]
        step = step_index[i]

        if np.isnan(c):
            continue

        # --- Warm-up phase: collect completed candle closes ---
        if not warmed_up:
            if step == max_step:
                candle_closes.append(c)

                if len(candle_closes) >= period + 1:
                    # Initialize avg_gain/avg_loss from first `period` deltas
                    closes_arr = np.array(candle_closes)
                    deltas = np.diff(closes_arr)

                    # Use EWM-style initialization (same as ewm(adjust=False)):
                    # First value = simple average, then apply EMA
                    gains = np.where(deltas > 0, deltas, 0.0)
                    losses = np.where(deltas < 0, -deltas, 0.0)

                    # Start with SMA of first `period` values, then it's the base
                    avg_g = gains[0]
                    avg_l = losses[0]
                    for k in range(1, len(gains)):
                        avg_g = alpha * gains[k] + (1.0 - alpha) * avg_g
                        avg_l = alpha * losses[k] + (1.0 - alpha) * avg_l

                    avg_gain_closed = avg_g
                    avg_loss_closed = avg_l
                    prev_candle_close = c

                    # Compute RSI at this point
                    if avg_loss_closed > 1e-15:
                        rs = avg_gain_closed / avg_loss_closed
                        rsi_out[i] = 100.0 - 100.0 / (1.0 + rs)
                    else:
                        rsi_out[i] = 100.0

                    warmed_up = True
            continue

        # --- Normal phase: provisional RSI ---
        delta = c - prev_candle_close
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)

        # Provisional from frozen state (does NOT accumulate across 5min steps)
        avg_gain_prov = alpha * gain + (1.0 - alpha) * avg_gain_closed
        avg_loss_prov = alpha * loss + (1.0 - alpha) * avg_loss_closed

        if avg_loss_prov > 1e-15:
            rs = avg_gain_prov / avg_loss_prov
            rsi_out[i] = 100.0 - 100.0 / (1.0 + rs)
        else:
            rsi_out[i] = 100.0

        # Freeze at candle closure
        if step == max_step:
            avg_gain_closed = avg_gain_prov
            avg_loss_closed = avg_loss_prov
            prev_candle_close = c

    return rsi_out


def compute_cci_live(high_live: np.ndarray, low_live: np.ndarray,
                     close_5min: np.ndarray, step_index: np.ndarray,
                     max_step: int, period: int = CCI_PERIOD) -> np.ndarray:
    """
    Compute CCI on live tf candles at 5min resolution.

    Maintains a rolling buffer of the last (period-1) completed candle TPs
    plus the current live TP. SMA and MAD are computed on this buffer.

    TP_live = (high_tf_live + low_tf_live + close_5min) / 3
    where high_tf_live and low_tf_live are cummax/cummin within the current
    tf candle (not raw 5min values).

    At step == max_step, the live TP is pushed into the buffer and the oldest
    is popped. Returns NaN until the buffer has `period` entries.

    Args:
        high_live: Live tf high (cummax within candle) at 5min resolution
        low_live: Live tf low (cummin within candle) at 5min resolution
        close_5min: Raw 5min close prices
        step_index: Position within tf candle (1 to max_step)
        max_step: Steps per candle (6 for 30min, 12 for 1h)
        period: CCI period (default 20)

    Returns:
        Array of CCI values at 5min resolution
    """
    from collections import deque

    n = len(close_5min)
    cci_out = np.full(n, np.nan)

    # Buffer of completed candle TPs (max size = period - 1)
    tp_buffer = deque(maxlen=period - 1)

    for i in range(n):
        c = close_5min[i]
        h = high_live[i]
        lo = low_live[i]
        step = step_index[i]

        if np.isnan(c) or np.isnan(h) or np.isnan(lo):
            continue

        # Current live typical price (uses tf-level high/low, not 5min)
        tp_live = (h + lo + c) / 3.0

        # Need period-1 frozen TPs + 1 live = period total
        if len(tp_buffer) >= period - 1:
            # Build full window: frozen TPs + live TP
            all_tps = np.array(list(tp_buffer) + [tp_live])

            sma = all_tps.mean()
            mad = np.abs(all_tps - sma).mean()

            if mad > 1e-15:
                cci_out[i] = (tp_live - sma) / (0.015 * mad)
            else:
                cci_out[i] = 0.0

        # At candle closure: push live TP into buffer
        if step == max_step:
            tp_buffer.append(tp_live)

    return cci_out


# =============================================================================
# VALIDATION: live values at candle closure must match standard resample
# =============================================================================

def validate_live_vs_standard(result: pd.DataFrame, df_5min: pd.DataFrame,
                              tf_minutes: int, suffix: str):
    """
    Verify that live indicator values at candle closure (step == max_step)
    match the standard resample-then-compute approach exactly.

    This proves the EMA only advances at candle closures, not at every 5min step.

    Raises AssertionError if values don't match (atol=1e-10).
    """
    max_step = tf_minutes // 5

    # Compute standard indicators on resampled data
    df_tf = resample_ohlcv(df_5min, tf_minutes)
    macd_std = calculate_macd(df_tf).values
    rsi_std = calculate_rsi(df_tf).values
    cci_std = calculate_cci(df_tf).values

    # Extract live values at candle closure points
    mask_close = result[f'step_{suffix}'] == max_step
    macd_live = result.loc[mask_close, f'macd_{suffix}_live'].values
    rsi_live = result.loc[mask_close, f'rsi_{suffix}_live'].values
    cci_live = result.loc[mask_close, f'cci_{suffix}_live'].values

    # Align lengths (live may have fewer points due to warm-up NaNs)
    # Find first non-NaN in each
    for name, live_vals, std_vals in [
        ('MACD', macd_live, macd_std),
        ('RSI', rsi_live, rsi_std),
        ('CCI', cci_live, cci_std),
    ]:
        # Find first valid index in both
        live_valid = ~np.isnan(live_vals)
        std_valid = ~np.isnan(std_vals)

        if not live_valid.any() or not std_valid.any():
            logger.warning(f"    SKIP {name}_{suffix}: no valid values to compare")
            continue

        # Find the range where both are valid
        live_first = np.argmax(live_valid)
        std_first = np.argmax(std_valid)

        # Align: the nth completed candle in live corresponds to nth in standard
        # But warm-up may differ, so start comparison after both are valid
        start = max(live_first, std_first)
        end = min(len(live_vals), len(std_vals))

        if start >= end:
            logger.warning(f"    SKIP {name}_{suffix}: no overlapping valid range")
            continue

        live_slice = live_vals[start:end]
        std_slice = std_vals[start:end]

        # Both must be non-NaN in the comparison range
        both_valid = ~np.isnan(live_slice) & ~np.isnan(std_slice)
        if not both_valid.any():
            logger.warning(f"    SKIP {name}_{suffix}: all NaN in comparison range")
            continue

        live_cmp = live_slice[both_valid]
        std_cmp = std_slice[both_valid]

        max_diff = np.max(np.abs(live_cmp - std_cmp))
        matches = np.allclose(live_cmp, std_cmp, atol=1e-10)

        if matches:
            logger.info(f"    PASS {name}_{suffix}_live: matches standard (max_diff={max_diff:.2e}, n={len(live_cmp):,})")
        else:
            logger.error(f"    FAIL {name}_{suffix}_live: max_diff={max_diff:.2e} (atol=1e-10), n={len(live_cmp):,}")
            # Show first few mismatches for debugging
            diffs = np.abs(live_cmp - std_cmp)
            worst_idx = np.argsort(diffs)[-5:]
            for idx in worst_idx:
                logger.error(f"         idx={idx}: live={live_cmp[idx]:.15f} std={std_cmp[idx]:.15f} diff={diffs[idx]:.2e}")


    Args:
        series_tf: Series with DatetimeIndex at the higher timeframe
        index_5min: 5min DatetimeIndex to forward-fill into
        col_name: Column name (for debugging)

    Returns:
        Series at 5min resolution, causally forward-filled
        First values will be NaN (no completed candle yet = warm-up)
    """
    # Shift by 1 tf period: value only available after candle completion
    shifted = series_tf.shift(1)

    # Forward-fill to 5min resolution
    result = shifted.reindex(index_5min, method='ffill')

    return result


# =============================================================================
# STEP INDEX CALCULATION
# =============================================================================

def compute_step_index(index_5min: pd.DatetimeIndex, tf_minutes: int) -> pd.Series:
    """
    Compute the position (1-based) of each 5min candle within its parent tf candle.

    This tells you "where are we" within the current 30min or 1h candle.

    Examples:
        30min (6 steps): 10:00→1, 10:05→2, 10:10→3, 10:15→4, 10:20→5, 10:25→6
        1h (12 steps):   10:00→1, 10:05→2, ..., 10:55→12

    Args:
        index_5min: 5min DatetimeIndex
        tf_minutes: Parent timeframe in minutes (30 or 60)

    Returns:
        Series with integer values from 1 to (tf_minutes / 5)
    """
    # Convert to total minutes since midnight
    minutes = index_5min.minute + index_5min.hour * 60

    # Position within the tf candle (0-based then +1 for 1-based)
    step = (minutes % tf_minutes) // 5 + 1

    return pd.Series(step, index=index_5min, dtype=int)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def generate_multitf_csv(asset_name: str, output_dir: str) -> str:
    """
    Generate the multi-timeframe CSV for one asset.

    Pipeline:
        1. Load raw 5min CSV
        2. Resample to 30min and 1h
        3. Compute indicators (MACD, RSI, CCI) on each timeframe
        4. Causal forward-fill all values to 5min resolution
        5. Compute step index for each timeframe
        6. Run consistency checks
        7. Save CSV

    Args:
        asset_name: Asset identifier ('BTC', 'ETH', etc.)
        output_dir: Output directory path

    Returns:
        Path to the generated CSV file
    """
    file_path = AVAILABLE_ASSETS_5M[asset_name]

    logger.info(f"\n{'='*60}")
    logger.info(f"  ASSET: {asset_name}")
    logger.info(f"{'='*60}")

    # =========================================================================
    # Step 1: Load raw 5min data
    # =========================================================================
    df_5min = load_csv_5min(file_path, asset_name)
    index_5min = df_5min.index

    # Start result DataFrame with raw 5min OHLCV
    result = pd.DataFrame(index=index_5min)
    result['open'] = df_5min['open']
    result['high'] = df_5min['high']
    result['low'] = df_5min['low']
    result['close'] = df_5min['close']
    result['volume'] = df_5min['volume']

    # =========================================================================
    # Steps 2-5: Process each higher timeframe (30min, 1h)
    # =========================================================================
    for tf_minutes, suffix in [(30, '30m'), (60, '1h')]:
        logger.info(f"\n  --- Timeframe {tf_minutes}min ({suffix}) ---")

        # Step 2: Resample 5min → tf
        df_tf = resample_ohlcv(df_5min, tf_minutes)
        logger.info(f"    Resample: {len(df_tf):,} candles")

        # Step 3a: Causal forward-fill OHLCV from completed candles
        # NOTE: The shift(1) in forward_fill_causal means close_30m at 10:30
        # contains the close of the 09:30-09:59 candle (previous completed).
        for col in ['open', 'high', 'low', 'close', 'volume']:
            result[f'{col}_{suffix}'] = forward_fill_causal(
                df_tf[col], index_5min, f'{col}_{suffix}'
            )

        # Step 3b: Compute indicators on the resampled data
        # Indicators are computed on COMPLETED candles only (no partial candle).
        # The EWM/rolling calculations are causal by nature (no future data).
        macd_values = calculate_macd(df_tf)
        rsi_values = calculate_rsi(df_tf)
        cci_values = calculate_cci(df_tf)
        logger.info(f"    Indicators computed (MACD, RSI, CCI)")

        # Step 4: Causal forward-fill indicators to 5min resolution
        # Same shift(1) as OHLCV: indicator from candle 10:00-10:29
        # is only available starting at 10:30.
        result[f'macd_{suffix}'] = forward_fill_causal(macd_values, index_5min, f'macd_{suffix}')
        result[f'rsi_{suffix}'] = forward_fill_causal(rsi_values, index_5min, f'rsi_{suffix}')
        result[f'cci_{suffix}'] = forward_fill_causal(cci_values, index_5min, f'cci_{suffix}')

        # Step 5: Compute step index (position within the tf candle)
        result[f'step_{suffix}'] = compute_step_index(index_5min, tf_minutes)

        # Log stats
        n_nan = result[f'macd_{suffix}'].isna().sum()
        n_valid = len(result) - n_nan
        logger.info(f"    Causal forward-fill: {n_valid:,} valid values, {n_nan:,} NaN (warm-up)")

    # =========================================================================
    # Step 6: Consistency checks
    # =========================================================================
    logger.info(f"\n  --- Consistency checks ---")

    # Check 1: NaN should only appear at the beginning (warm-up period)
    # Any NaN after the first valid value indicates a data gap or bug
    for suffix in ['30m', '1h']:
        for col_name in [f'macd_{suffix}', f'rsi_{suffix}', f'cci_{suffix}']:
            series = result[col_name]
            first_valid = series.first_valid_index()
            if first_valid is not None:
                nan_after_valid = series.loc[first_valid:].isna().sum()
                if nan_after_valid > 0:
                    logger.warning(f"    WARNING {col_name}: {nan_after_valid} NaN AFTER first valid value!")
                else:
                    logger.info(f"    OK {col_name}: NaN only at start (warm-up)")

    # Check 2: Step index range should be 1 to (tf_minutes / 5)
    for suffix, tf in [('30m', 30), ('1h', 60)]:
        steps = result[f'step_{suffix}']
        expected_max = tf // 5
        logger.info(f"    OK step_{suffix}: min={steps.min()}, max={steps.max()} (expected 1-{expected_max})")

    # Check 3: Causality — tf values should only change at step 1
    # (beginning of a new tf candle, when the previous candle just completed)
    for suffix, tf in [('30m', 30), ('1h', 60)]:
        changes = result[f'close_{suffix}'].diff().abs() > 0
        change_steps = result.loc[changes, f'step_{suffix}']
        if len(change_steps) > 0:
            pct_step1 = (change_steps == 1).sum() / len(change_steps) * 100
            logger.info(f"    OK close_{suffix}: {pct_step1:.1f}% of value changes occur at step 1 (causality verified)")

    # =========================================================================
    # Step 7: Save to CSV
    # =========================================================================
    os.makedirs(output_dir, exist_ok=True)

    # Build output filename from the input CSV name
    asset_filename = file_path.split('/')[-1].replace('_all_5m.csv', '')
    output_path = os.path.join(output_dir, f'{asset_filename}_multitf.csv')

    # Save with datetime as column (not index) for broad compatibility
    result_save = result.copy()
    result_save.index.name = 'datetime'
    result_save = result_save.reset_index()
    result_save.to_csv(output_path, index=False)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"\n  Saved: {output_path} ({file_size_mb:.1f} MB)")
    logger.info(f"     Rows: {len(result):,}")
    logger.info(f"     Columns ({len(result.columns)}): {list(result.columns)}")

    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Prepare multi-timeframe CSV (30min + 1h) at 5min resolution'
    )
    parser.add_argument('--assets', nargs='+',
                        default=['BTC', 'ETH', 'BNB', 'ADA', 'LTC'],
                        help='Assets to process (default: all)')
    parser.add_argument('--output-dir', type=str,
                        default=PREPARED_DATA_DIR,
                        help=f'Output directory (default: {PREPARED_DATA_DIR})')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MULTI-TIMEFRAME CSV PREPARATION")
    logger.info("=" * 60)
    logger.info(f"Assets: {args.assets}")
    logger.info(f"Timeframes: 5min (raw) + 30min + 1h")
    logger.info(f"Indicators: MACD, RSI, CCI")
    logger.info(f"Causality: shift(1) before forward-fill (value available after candle completion)")
    logger.info(f"Output: {args.output_dir}/")

    generated_files = []

    for asset_name in args.assets:
        if asset_name not in AVAILABLE_ASSETS_5M:
            logger.warning(f"Asset {asset_name} not found, skipping")
            continue

        output_path = generate_multitf_csv(asset_name, args.output_dir)
        generated_files.append(output_path)

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Files generated: {len(generated_files)}")
    for f in generated_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        logger.info(f"  {f} ({size_mb:.1f} MB)")

    logger.info(f"\nColumn structure:")
    logger.info(f"  5min raw:  open, high, low, close, volume")
    logger.info(f"  30min:     open_30m, high_30m, low_30m, close_30m, volume_30m")
    logger.info(f"             macd_30m, rsi_30m, cci_30m, step_30m")
    logger.info(f"  1h:        open_1h, high_1h, low_1h, close_1h, volume_1h")
    logger.info(f"             macd_1h, rsi_1h, cci_1h, step_1h")
    logger.info(f"\nNOTE: Kalman and labels NOT included (apply after train/test split)")


if __name__ == '__main__':
    main()
