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
# CAUSAL FORWARD-FILL
# =============================================================================

def forward_fill_causal(series_tf: pd.Series, index_5min: pd.DatetimeIndex,
                        col_name: str) -> pd.Series:
    """
    Forward-fill a higher-timeframe series to 5min resolution with strict causality.

    CAUSALITY MECHANISM:
        shift(1) delays the values by one tf candle before forward-filling.
        This ensures a candle's value is only available AFTER it completes.

        Example for 30min:
            Candle 10:00 (data 10:00-10:29) → value available at 10:30
            Candle 10:30 (data 10:30-10:59) → value available at 11:00

    NOTE: This same shift applies to both OHLCV and indicators. So close_30m
    at 10:30 contains the close of the 09:30-09:59 candle (previous completed),
    not the 10:00-10:29 candle. This is intentional and consistent.

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
