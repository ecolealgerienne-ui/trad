#!/usr/bin/env python3
"""
Multi-Timeframe CSV Preparation: live-style 30min and 1h data at 5min resolution.

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

# Kalman parameters (project-wide defaults)
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
    """Forward-only Kalman on 1D series. For validation."""
    from pykalman import KalmanFilter as KF
    valid = ~np.isnan(data)
    if valid.sum() < 2:
        return np.full(len(data), np.nan)
    vd = data[valid]
    kf = KF(transition_matrices=[[1,1],[0,1]], observation_matrices=[[1,0]],
            initial_state_mean=[vd[0], 0.0], initial_state_covariance=np.eye(2),
            observation_covariance=KALMAN_MEASURE_VAR,
            transition_covariance=np.eye(2) * KALMAN_PROCESS_VAR)
    sm, _ = kf.filter(vd)
    out = np.full(len(data), np.nan)
    out[valid] = sm[:, 0]
    return out


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
    """RSI with frozen/provisional EWM avg_gain/avg_loss. Freeze at closure."""
    n = len(close_5min)
    alpha = 2.0 / (RSI_PERIOD + 1)

    out = np.full(n, np.nan)
    ag_cl = np.nan; al_cl = np.nan; prev_cl = np.nan
    candle_closes = []; warmed = False

    for i in range(n):
        c = close_5min[i]
        if np.isnan(c):
            continue
        if not warmed:
            if is_close[i]:
                candle_closes.append(c)
                if len(candle_closes) >= RSI_PERIOD + 1:
                    arr = np.array(candle_closes)
                    d = np.diff(arr)
                    g = np.where(d > 0, d, 0.0)
                    l = np.where(d < 0, -d, 0.0)
                    ag = g[0]; al = l[0]
                    for k in range(1, len(g)):
                        ag = alpha * g[k] + (1.0 - alpha) * ag
                        al = alpha * l[k] + (1.0 - alpha) * al
                    ag_cl = ag; al_cl = al; prev_cl = c
                    out[i] = 100.0 - 100.0 / (1.0 + ag / al) if al > 1e-15 else 100.0
                    warmed = True
            continue
        delta = c - prev_cl
        gn = max(delta, 0.0); ls = max(-delta, 0.0)
        ag_p = alpha * gn + (1.0 - alpha) * ag_cl
        al_p = alpha * ls + (1.0 - alpha) * al_cl
        out[i] = 100.0 - 100.0 / (1.0 + ag_p / al_p) if al_p > 1e-15 else 100.0
        if is_close[i]:
            ag_cl = ag_p; al_cl = al_p; prev_cl = c
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

def compute_kalman_live(indicator_live, is_close):
    """
    Kalman filter_update with frozen/provisional state. Freeze at closure.
    Same logic as indicators: state advances only at bucket closure.
    """
    from pykalman import KalmanFilter as KF

    n = len(indicator_live)
    out = np.full(n, np.nan)

    # Find first valid value
    fvi = -1
    for i in range(n):
        if not np.isnan(indicator_live[i]):
            fvi = i; break
    if fvi < 0:
        return out

    iv = indicator_live[fvi]
    kf = KF(transition_matrices=np.array([[1,1],[0,1]]),
            observation_matrices=np.array([[1,0]]),
            initial_state_mean=np.array([iv, 0.0]),
            initial_state_covariance=np.eye(2),
            observation_covariance=KALMAN_MEASURE_VAR,
            transition_covariance=np.eye(2) * KALMAN_PROCESS_VAR)

    sm_cl = np.array([iv, 0.0])
    sc_cl = np.eye(2)
    init = False

    for i in range(n):
        obs = indicator_live[i]
        if np.isnan(obs):
            continue
        if not init:
            if is_close[i]:
                sm_cl, sc_cl = kf.filter_update(sm_cl, sc_cl, observation=obs)
                out[i] = sm_cl[0]; init = True
            continue
        sm_p, sc_p = kf.filter_update(sm_cl, sc_cl, observation=obs)
        out[i] = sm_p[0]
        if is_close[i]:
            sm_cl = sm_p; sc_cl = sc_p
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
        ms = calculate_macd_standard(df_tf)
        ok &= compare("MACD", f'macd_{suffix}_live', ms)
        ks = pd.Series(kalman_filter_standard(ms.values), index=df_tf.index)
        ok &= compare("Kalman_MACD", f'macd_{suffix}_filtered', ks)
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
            ind_results['macd'] = compute_macd_live(close_5min, is_close)
        if 'rsi' in indicators:
            logger.info(f"    Computing RSI live...")
            ind_results['rsi'] = compute_rsi_live(close_5min, is_close)
        if 'cci' in indicators:
            logger.info(f"    Computing CCI live...")
            ind_results['cci'] = compute_cci_live(high_live, low_live, close_5min, is_close)

        for ind, vals in ind_results.items():
            result[f'{ind}_{suffix}_live'] = vals

        # Kalman on each indicator
        for ind, vals in ind_results.items():
            logger.info(f"    Computing Kalman on {ind}_{suffix}...")
            filt = compute_kalman_live(vals, is_close)
            result[f'{ind}_{suffix}_filtered'] = filt

            # Direction label
            fs = pd.Series(filt, index=df_5min.index)
            lab = (fs > fs.shift(1)).astype(float)
            lab.iloc[0] = 0
            result[f'{ind}_{suffix}_label'] = lab.fillna(0).astype(int)

        # Stats
        for ind in ind_results:
            nc = (result[f'{ind}_{suffix}_label'].diff().abs() > 0).sum()
            logger.info(f"    {ind.upper()} label changes: {nc:,}")

        # Validation
        run_validation(result, df_5min, tf_minutes, suffix, indicators)

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
    parser.add_argument('--assets', nargs='+', default=['BTC', 'ETH', 'BNB', 'ADA', 'LTC'])
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
