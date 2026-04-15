#!/usr/bin/env python3
"""
FLKS(N=2) vs Kalman Filter vs Oracle Smoother — Slope Estimation Comparison
===========================================================================

Compares 3 slope estimation methods on BTC 5min MACD data:
  1. pente_filtre : Kalman forward-only (causal)
  2. pente_FLKS   : Fixed-Lag Kalman Smoother N=2 (causal, 2-step lag)
  3. pente_oracle : Kalman RTS smoother (non-causal reference)

Metrics:
  - MSE vs oracle
  - Sign correlation (%) vs oracle
  - Pearson correlation vs oracle

Output:
  - Console table
  - plots/ : 3-series overlay + residuals

Usage:
  python src/signal_processing/flks_vs_filter.py \
      --csv data_trad/BTCUSD_all_5m.csv \
      --indicator macd \
      --n-samples 50000 \
      --plot-window 200

Requires: numpy, pandas, scipy, matplotlib (pykalman NOT required)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# PARAMETERS (from existing pipeline: prepare_multitf_csv.py)
# ============================================================================

KALMAN_PROCESS_VAR = 0.01   # Q diagonal
KALMAN_MEASURE_VAR = 0.1    # R scalar

# 2D state: [position, velocity]
# Transition: x_{t+1} = A x_t + w,  w ~ N(0, Q)
# Observation: z_t = H x_t + v,     v ~ N(0, R)
A = np.array([[1.0, 1.0],
              [0.0, 1.0]])

H = np.array([[1.0, 0.0]])

Q = np.eye(2) * KALMAN_PROCESS_VAR
R = np.array([[KALMAN_MEASURE_VAR]])

# Indicator periods (same as pipeline)
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
CCI_PERIOD = 20


# ============================================================================
# DATA LOADING
# ============================================================================

def load_csv(path: str) -> pd.DataFrame:
    """Load 5min OHLCV CSV. Returns DataFrame with DatetimeIndex."""
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
    df = df.sort_index()
    return df


def compute_indicator(df: pd.DataFrame, indicator: str) -> np.ndarray:
    """Compute a technical indicator from OHLCV DataFrame."""
    close = df['close'].values.astype(np.float64)

    if indicator == 'macd':
        ema_fast = _ema(close, MACD_FAST)
        ema_slow = _ema(close, MACD_SLOW)
        macd_line = ema_fast - ema_slow
        signal_line = _ema(macd_line, MACD_SIGNAL)
        return macd_line - signal_line  # MACD histogram

    elif indicator == 'rsi':
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = _ema(gain, RSI_PERIOD)
        avg_loss = _ema(loss, RSI_PERIOD)
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
        return 100.0 - 100.0 / (1.0 + rs)

    elif indicator == 'cci':
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        tp = (high + low + close) / 3.0
        tp_ma = _sma(tp, CCI_PERIOD)
        mean_dev = _mean_deviation(tp, CCI_PERIOD)
        mean_dev = np.where(mean_dev == 0, 1e-10, mean_dev)
        return (tp - tp_ma) / (0.015 * mean_dev)

    elif indicator == 'close':
        return close

    else:
        raise ValueError(f"Unknown indicator: {indicator}")


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average (forward-only)."""
    alpha = 2.0 / (period + 1)
    out = np.empty_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out


def _sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    out = np.full_like(data, np.nan)
    cs = np.cumsum(data)
    out[period - 1:] = (cs[period - 1:] - np.concatenate([[0], cs[:-period]])) / period
    # Fill initial NaN with expanding mean
    for i in range(period - 1):
        out[i] = np.mean(data[:i + 1])
    return out


def _mean_deviation(data: np.ndarray, period: int) -> np.ndarray:
    """Mean deviation over rolling window."""
    out = np.full_like(data, np.nan)
    sma = _sma(data, period)
    for i in range(period - 1, len(data)):
        out[i] = np.mean(np.abs(data[i - period + 1:i + 1] - sma[i]))
    for i in range(period - 1):
        out[i] = np.mean(np.abs(data[:i + 1] - sma[i]))
    return out


# ============================================================================
# KALMAN IMPLEMENTATIONS (no external dependency)
# ============================================================================

def kalman_filter_forward(z: np.ndarray):
    """
    Standard Kalman filter (forward-only, causal).
    Matches pykalman.KalmanFilter.filter() behavior exactly:
    at t=0, the initial prior is updated with z[0] (predict+update).

    Returns:
        x_filt: (T, 2) filtered state means [position, velocity]
        P_filt: (T, 2, 2) filtered state covariances
        x_pred: (T, 2) predicted state means (prior)
        P_pred: (T, 2, 2) predicted state covariances (prior)
    """
    T = len(z)
    x_filt = np.zeros((T, 2))
    P_filt = np.zeros((T, 2, 2))
    x_pred = np.zeros((T, 2))
    P_pred = np.zeros((T, 2, 2))

    # Initial prior (before seeing any data)
    x0 = np.array([z[0], 0.0])
    P0 = np.eye(2)

    for t in range(T):
        if t == 0:
            # At t=0: prior = initial state, then update with z[0]
            x_p = x0
            P_p = P0
        else:
            # Predict from previous posterior
            x_p = A @ x_filt[t - 1]
            P_p = A @ P_filt[t - 1] @ A.T + Q

        x_pred[t] = x_p
        P_pred[t] = P_p

        # Update with observation z[t]
        y = z[t] - H @ x_p                         # innovation
        S = H @ P_p @ H.T + R                       # innovation covariance
        K = P_p @ H.T @ np.linalg.inv(S)            # Kalman gain

        x_filt[t] = x_p + (K @ y).ravel()
        P_filt[t] = (np.eye(2) - K @ H) @ P_p

    return x_filt, P_filt, x_pred, P_pred


def pykalman_filter_and_smooth(z: np.ndarray):
    """
    Run pykalman's filter() and smooth() — the pipeline reference implementation.

    Returns:
        filter_means: (T, 2) forward-filtered state means [position, velocity]
        smooth_means: (T, 2) RTS-smoothed state means (oracle, non-causal)
    """
    from pykalman import KalmanFilter as KF

    kf = KF(
        transition_matrices=[[1, 1], [0, 1]],
        observation_matrices=[[1, 0]],
        initial_state_mean=[z[0], 0.0],
        initial_state_covariance=np.eye(2),
        observation_covariance=KALMAN_MEASURE_VAR,
        transition_covariance=np.eye(2) * KALMAN_PROCESS_VAR,
    )

    filter_means, _ = kf.filter(z)
    smooth_means, _ = kf.smooth(z)

    return filter_means, smooth_means


def kalman_flks(z: np.ndarray, lag: int = 2):
    """
    Fixed-Lag Kalman Smoother (FLKS) with lag N.

    At time t, produces a smoothed estimate of x[t] using z[0..t+lag].
    This is CAUSAL with a fixed delay of N steps (no future beyond t+N).

    x_flks[t] = E[x_t | z_0, ..., z_{t+N}]

    Implementation: for each t, run a local RTS backward pass of N steps
    over the pre-computed forward filter states. O(T * N), acceptable for
    small N.

    Returns:
        x_flks: (T, 2) smoothed state estimates.
                x_flks[t] uses z[0..t+lag]. For t > T-1-lag, partial smoothing.
    """
    x_filt, P_filt, x_pred, P_pred = kalman_filter_forward(z)
    T = len(z)

    x_flks = np.copy(x_filt)

    for t in range(T):
        end = min(t + lag, T - 1)
        if end <= t:
            continue

        # Local backward RTS from end to t
        x_s = np.copy(x_filt[end])
        P_s = np.copy(P_filt[end])

        for k in range(end - 1, t - 1, -1):
            P_pred_k1 = P_pred[k + 1]
            try:
                C = P_filt[k] @ A.T @ np.linalg.inv(P_pred_k1)
            except np.linalg.LinAlgError:
                C = P_filt[k] @ A.T @ np.linalg.pinv(P_pred_k1)

            x_s = x_filt[k] + C @ (x_s - x_pred[k + 1])
            P_s = P_filt[k] + C @ (P_s - P_pred_k1) @ C.T

        x_flks[t] = x_s

    return x_flks


# ============================================================================
# SLOPE COMPUTATION
# ============================================================================

def compute_slopes(positions: np.ndarray) -> np.ndarray:
    """
    Compute slope as pente[t] = position[t-1] - position[t-2].
    Matches the existing pipeline convention.
    """
    T = len(positions)
    slopes = np.full(T, np.nan)
    for t in range(2, T):
        slopes[t] = positions[t - 1] - positions[t - 2]
    return slopes


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(pente: np.ndarray, oracle: np.ndarray) -> dict:
    """Compute MSE, sign correlation, Pearson correlation vs oracle."""
    mask = ~np.isnan(pente) & ~np.isnan(oracle)
    p = pente[mask]
    o = oracle[mask]

    if len(p) == 0:
        return {'mse': np.nan, 'sign_corr_pct': np.nan, 'pearson': np.nan, 'n': 0}

    mse = np.mean((p - o) ** 2)

    sign_p = np.sign(p)
    sign_o = np.sign(o)
    sign_corr = np.mean(sign_p == sign_o) * 100.0

    # Pearson
    if np.std(p) > 0 and np.std(o) > 0:
        pearson = np.corrcoef(p, o)[0, 1]
    else:
        pearson = np.nan

    return {'mse': mse, 'sign_corr_pct': sign_corr, 'pearson': pearson, 'n': len(p)}


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_slopes(pente_filtre, pente_flks, pente_oracle, window_start, window_size,
                indicator, output_dir):
    """Plot 3 slope series on a window + residuals."""
    ws = window_start
    we = ws + window_size
    t = np.arange(ws, we)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # --- Panel 1: 3 slopes overlay ---
    ax = axes[0]
    ax.plot(t, pente_oracle[ws:we], color='black', linewidth=1.5, alpha=0.8, label='Oracle (pykalman.smooth)')
    ax.plot(t, pente_flks[ws:we], color='tab:blue', linewidth=1.0, alpha=0.8, label='FLKS (N=2, numpy)')
    ax.plot(t, pente_filtre[ws:we], color='tab:red', linewidth=0.8, alpha=0.6, label='Filter (pykalman.filter)')
    ax.set_ylabel(f'Slope ({indicator.upper()})')
    ax.set_title(f'Slope Estimation — {indicator.upper()} 5min BTC (index {ws}:{we})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # --- Panel 2: Residuals vs oracle ---
    ax = axes[1]
    resid_filter = pente_filtre[ws:we] - pente_oracle[ws:we]
    resid_flks = pente_flks[ws:we] - pente_oracle[ws:we]
    ax.plot(t, resid_filter, color='tab:red', linewidth=0.8, alpha=0.7, label='Filter - Oracle')
    ax.plot(t, resid_flks, color='tab:blue', linewidth=0.8, alpha=0.7, label='FLKS - Oracle')
    ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Oracle')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # --- Panel 3: Absolute residuals (smoothed) ---
    ax = axes[2]
    kernel = 20
    abs_resid_f = np.abs(resid_filter)
    abs_resid_flks = np.abs(resid_flks)
    if len(abs_resid_f) >= kernel:
        smooth_f = np.convolve(abs_resid_f, np.ones(kernel) / kernel, mode='same')
        smooth_flks = np.convolve(abs_resid_flks, np.ones(kernel) / kernel, mode='same')
        ax.plot(t, smooth_f, color='tab:red', linewidth=1.0, label=f'|Filter - Oracle| MA({kernel})')
        ax.plot(t, smooth_flks, color='tab:blue', linewidth=1.0, label=f'|FLKS - Oracle| MA({kernel})')
    else:
        ax.plot(t, abs_resid_f, color='tab:red', linewidth=0.8, label='|Filter - Oracle|')
        ax.plot(t, abs_resid_flks, color='tab:blue', linewidth=0.8, label='|FLKS - Oracle|')
    ax.set_ylabel('Absolute Error')
    ax.set_xlabel('Sample Index')
    ax.set_title('Smoothed Absolute Error')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f'flks_slopes_{indicator}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {out_path}")


def plot_scatter(pente_filtre, pente_flks, pente_oracle, indicator, output_dir):
    """Scatter plot: filter vs oracle and FLKS vs oracle."""
    mask = ~np.isnan(pente_filtre) & ~np.isnan(pente_flks) & ~np.isnan(pente_oracle)
    pf = pente_filtre[mask]
    pk = pente_flks[mask]
    po = pente_oracle[mask]

    # Subsample if too many points
    max_pts = 5000
    if len(pf) > max_pts:
        idx = np.random.default_rng(42).choice(len(pf), max_pts, replace=False)
        pf, pk, po = pf[idx], pk[idx], po[idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(po, pf, alpha=0.15, s=4, color='tab:red')
    lims = [min(po.min(), pf.min()), max(po.max(), pf.max())]
    ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Oracle slope')
    ax.set_ylabel('Filter slope')
    ax.set_title('Forward Filter vs Oracle')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(po, pk, alpha=0.15, s=4, color='tab:blue')
    lims = [min(po.min(), pk.min()), max(po.max(), pk.max())]
    ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Oracle slope')
    ax.set_ylabel('FLKS slope')
    ax.set_title('FLKS(N=2) vs Oracle')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'{indicator.upper()} — Slope Scatter vs Oracle', fontsize=13)
    plt.tight_layout()
    out_path = output_dir / f'flks_scatter_{indicator}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {out_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='FLKS(N=2) vs Kalman Filter vs Oracle — Slope Comparison')
    parser.add_argument('--csv', type=str, default='data_trad/BTCUSD_all_5m.csv',
                        help='Path to BTC 5min CSV')
    parser.add_argument('--indicator', type=str, default='macd',
                        choices=['macd', 'rsi', 'cci', 'close'],
                        help='Indicator to filter (default: macd)')
    parser.add_argument('--n-samples', type=int, default=50000,
                        help='Number of samples to use (0=all)')
    parser.add_argument('--plot-window', type=int, default=200,
                        help='Window size for time-series plot')
    parser.add_argument('--plot-start', type=int, default=-1,
                        help='Start index for plot window (-1=auto middle)')
    parser.add_argument('--flks-lag', type=int, default=2,
                        help='FLKS lag N (default: 2)')
    parser.add_argument('--output-dir', type=str, default='plots',
                        help='Directory for output plots')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"Loading {args.csv} ...")
    df = load_csv(args.csv)
    print(f"  {len(df):,} candles loaded ({df.index[0]} → {df.index[-1]})")

    # ------------------------------------------------------------------
    # 2. Compute indicator
    # ------------------------------------------------------------------
    print(f"Computing {args.indicator.upper()} indicator ...")
    indicator_raw = compute_indicator(df, args.indicator)

    # Trim to n_samples (from end, most recent data)
    if args.n_samples > 0 and len(indicator_raw) > args.n_samples:
        indicator_raw = indicator_raw[-args.n_samples:]
        print(f"  Trimmed to last {args.n_samples:,} samples")

    # Remove NaN at start
    first_valid = 0
    for i in range(len(indicator_raw)):
        if not np.isnan(indicator_raw[i]):
            first_valid = i
            break
    z = indicator_raw[first_valid:].astype(np.float64)
    T = len(z)
    print(f"  {T:,} valid samples for Kalman processing")

    # ------------------------------------------------------------------
    # 3. Run the 3 methods
    # ------------------------------------------------------------------
    print("\n--- Running pykalman filter + smooth (pipeline reference) ---")
    filter_means, smooth_means = pykalman_filter_and_smooth(z)
    pos_filter = filter_means[:, 0]
    pos_oracle = smooth_means[:, 0]

    print(f"--- Running FLKS (N={args.flks_lag}, numpy) ---")
    x_flks = kalman_flks(z, lag=args.flks_lag)
    pos_flks = x_flks[:, 0]

    # ------------------------------------------------------------------
    # 4. Compute slopes: pente[t] = position[t-1] - position[t-2]
    # ------------------------------------------------------------------
    print("\nComputing slopes (pente[t] = pos[t-1] - pos[t-2]) ...")
    pente_filtre = compute_slopes(pos_filter)
    pente_flks = compute_slopes(pos_flks)
    pente_oracle = compute_slopes(pos_oracle)

    # ------------------------------------------------------------------
    # 5. Metrics
    # ------------------------------------------------------------------
    m_filter = compute_metrics(pente_filtre, pente_oracle)
    m_flks = compute_metrics(pente_flks, pente_oracle)

    print("\n" + "=" * 72)
    print(f"  FLKS(N={args.flks_lag}) vs Forward Filter — {args.indicator.upper()} BTC 5min")
    print(f"  Samples: {T:,} | Slope formula: pos[t-1] - pos[t-2]")
    print(f"  Kalman params: Q={KALMAN_PROCESS_VAR}, R={KALMAN_MEASURE_VAR}")
    print("=" * 72)
    print(f"{'Method':<22} {'MSE vs Oracle':>15} {'Sign Corr %':>13} {'Pearson':>10} {'N':>8}")
    print("-" * 72)
    print(f"{'Filter (forward)' :<22} {m_filter['mse']:>15.6e} {m_filter['sign_corr_pct']:>12.2f}% {m_filter['pearson']:>10.6f} {m_filter['n']:>8,}")
    print(f"{'FLKS (N=' + str(args.flks_lag) + ')':<22} {m_flks['mse']:>15.6e} {m_flks['sign_corr_pct']:>12.2f}% {m_flks['pearson']:>10.6f} {m_flks['n']:>8,}")
    print("-" * 72)

    # Improvement
    if m_filter['mse'] > 0:
        mse_gain = (1 - m_flks['mse'] / m_filter['mse']) * 100
        sign_gain = m_flks['sign_corr_pct'] - m_filter['sign_corr_pct']
        pearson_gain = m_flks['pearson'] - m_filter['pearson']
        print(f"{'FLKS improvement':<22} {mse_gain:>14.2f}% {sign_gain:>+12.2f}pp {pearson_gain:>+10.6f}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------------
    if args.plot_start < 0:
        # Auto: middle of the series
        plot_start = max(2, T // 2 - args.plot_window // 2)
    else:
        plot_start = args.plot_start
    plot_start = min(plot_start, T - args.plot_window)
    plot_start = max(2, plot_start)

    print(f"\nGenerating plots (window {plot_start}:{plot_start + args.plot_window}) ...")
    plot_slopes(pente_filtre, pente_flks, pente_oracle,
                plot_start, args.plot_window, args.indicator, output_dir)
    plot_scatter(pente_filtre, pente_flks, pente_oracle, args.indicator, output_dir)

    # ------------------------------------------------------------------
    # 7. Bonus: transition-only metrics
    # ------------------------------------------------------------------
    print("\n--- Transition-Only Analysis ---")
    # Transition = oracle slope changes sign (positive <-> negative)
    # Slopes near zero (|slope| < EPSILON) are ignored to avoid noise transitions
    EPSILON = 1e-8
    mask_valid = ~np.isnan(pente_oracle)
    sign_oracle = np.where(np.abs(pente_oracle) < EPSILON, 0, np.sign(pente_oracle))
    transitions = np.zeros(T, dtype=bool)
    for t in range(3, T):
        if mask_valid[t] and mask_valid[t - 1]:
            if sign_oracle[t] != 0 and sign_oracle[t - 1] != 0 and sign_oracle[t] != sign_oracle[t - 1]:
                transitions[t] = True

    n_trans = transitions.sum()
    if n_trans > 0:
        m_filter_trans = compute_metrics(pente_filtre[transitions], pente_oracle[transitions])
        m_flks_trans = compute_metrics(pente_flks[transitions], pente_oracle[transitions])

        print(f"  Transitions detected: {n_trans:,} ({n_trans / mask_valid.sum() * 100:.1f}% of samples)")
        print(f"  {'Method':<22} {'MSE':>15} {'Sign Corr %':>13} {'Pearson':>10}")
        print(f"  {'-' * 62}")
        print(f"  {'Filter (forward)' :<22} {m_filter_trans['mse']:>15.6e} {m_filter_trans['sign_corr_pct']:>12.2f}% {m_filter_trans['pearson']:>10.6f}")
        print(f"  {'FLKS (N=' + str(args.flks_lag) + ')':<22} {m_flks_trans['mse']:>15.6e} {m_flks_trans['sign_corr_pct']:>12.2f}% {m_flks_trans['pearson']:>10.6f}")

        if m_filter_trans['mse'] > 0:
            mse_gain_t = (1 - m_flks_trans['mse'] / m_filter_trans['mse']) * 100
            sign_gain_t = m_flks_trans['sign_corr_pct'] - m_filter_trans['sign_corr_pct']
            print(f"  {'FLKS improvement':<22} {mse_gain_t:>14.2f}% {sign_gain_t:>+12.2f}pp")
    else:
        print("  No transitions detected.")

    print(f"\nDone. Plots in {output_dir}/")


if __name__ == '__main__':
    main()
