#!/usr/bin/env python3
"""
FLKS 30min vs FLKS 30min+5min micro — Sign concordance vs pykalman oracle
=========================================================================

Scénario:
  1. Charger CSV BTC 5min existant
  2. Rééchantillonner en 30min (OHLCV, méthode pipeline)
  3. Calculer MACD 30min
  4. Oracle : pykalman.smooth() sur les bougies 30min
  5. Test 1 : FLKS(N=2) incrémental, bougie 30min par bougie
  6. Test 2 : même FLKS mais entre chaque bougie 30min, injecter
              les 6 closes 5min dans le filtre (filter_update incrémental)
              avant de fixer l'état à t
  7. Métrique : % concordance de signe de pente vs oracle sur [1000:5000]

Pente dans les 3 cas : pos_30m[t-1] - pos_30m[t-2] (échelle 30min)

Usage:
    python src/signal_processing/flks_30m_vs_5m_micro.py \
        --csv data_trad/BTCUSD_all_5m.csv \
        --n-candles-30m 5000 \
        --eval-start 1000

Requires: numpy, pandas, matplotlib, pykalman
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
# PARAMETERS (from pipeline: prepare_multitf_csv.py)
# ============================================================================

KALMAN_PROCESS_VAR = 0.01
KALMAN_MEASURE_VAR = 0.1

A = np.array([[1.0, 1.0],
              [0.0, 1.0]])
H = np.array([[1.0, 0.0]])
Q = np.eye(2) * KALMAN_PROCESS_VAR
R = np.array([[KALMAN_MEASURE_VAR]])

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


# ============================================================================
# DATA LOADING (from pipeline: prepare_multitf_csv.py)
# ============================================================================

def load_csv(path: str) -> pd.DataFrame:
    """Load 5min OHLCV CSV with DatetimeIndex."""
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


def resample_ohlcv(df_5min: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
    """Resample 5min to higher tf (from pipeline)."""
    return df_5min.resample(f'{tf_minutes}min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()


def calculate_macd(df: pd.DataFrame) -> pd.Series:
    """MACD histogram (from pipeline)."""
    ema_f = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_s = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return line - sig


def compute_macd_live(close_5min, is_close):
    """
    MACD histogram with frozen/provisional EMA. Freeze at bucket closure.
    Copied from pipeline (prepare_multitf_csv.py:compute_macd_live).

    Between closures, the EMA state is frozen and the current close is used
    as a provisional update. At closure, the EMA state advances.

    Returns MACD value at each 5min step (same scale as MACD 30min at closures).
    """
    n = len(close_5min)
    alpha_f = 2.0 / (MACD_FAST + 1)
    alpha_s = 2.0 / (MACD_SLOW + 1)
    alpha_sig = 2.0 / (MACD_SIGNAL + 1)

    out = np.full(n, np.nan)
    ema_f_cl = np.nan
    ema_s_cl = np.nan
    ema_sig_cl = np.nan
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


def compute_bucket_close_mask(index_5min, tf_minutes):
    """Detect last bar of each tf bucket (from pipeline)."""
    bucket = index_5min.floor(f'{tf_minutes}min').values
    next_bucket = np.append(bucket[1:], np.datetime64('NaT'))
    return (bucket != next_bucket) | pd.isna(next_bucket)


# ============================================================================
# ORACLE: pykalman.smooth() on 30min candles (from pipeline)
# ============================================================================

def compute_oracle(indicator_30m: np.ndarray):
    """
    pykalman RTS smoother on full 30min series — non-causal reference.
    Copied from pipeline compute_oracle_label().

    Returns:
        positions: (N,) smoothed position at each 30min candle
        slopes:    (N,) pente[t] = pos[t-1] - pos[t-2]
    """
    from pykalman import KalmanFilter as KF

    n = len(indicator_30m)
    valid = ~np.isnan(indicator_30m)
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
# KALMAN FILTER PRIMITIVES (numpy, for FLKS)
# ============================================================================

def kf_predict(x, P):
    """Predict step. Returns (x_pred, P_pred)."""
    x_p = A @ x
    P_p = A @ P @ A.T + Q
    return x_p, P_p


def kf_update(x_p, P_p, z_obs):
    """Update step with scalar observation. Returns (x_filt, P_filt)."""
    y = z_obs - H @ x_p
    S = H @ P_p @ H.T + R
    K = P_p @ H.T / S[0, 0]
    x_f = x_p + (K @ y).ravel()
    P_f = (np.eye(2) - K @ H) @ P_p
    return x_f, P_f


# ============================================================================
# TEST 1: FLKS(N=2) on 30min candles only
# ============================================================================

def run_flks_30m(indicator_30m: np.ndarray, lag: int = 2):
    """
    FLKS(N=2) incrémental, bougie 30min par bougie.

    At each step t: predict + update with indicator_30m[t].
    Then extract smoothed position for t-lag via local RTS backward.

    Returns:
        positions: (N,) smoothed position at each 30min candle
        slopes:    (N,) pente[t] = pos[t-1] - pos[t-2]
    """
    n = len(indicator_30m)

    # Forward filter pass (store all states for backward smoothing)
    x_filt = np.zeros((n, 2))
    P_filt = np.zeros((n, 2, 2))
    x_pred = np.zeros((n, 2))
    P_pred = np.zeros((n, 2, 2))

    for t in range(n):
        if t == 0:
            x_p = np.array([indicator_30m[0], 0.0])
            P_p = np.eye(2)
        else:
            x_p, P_p = kf_predict(x_filt[t - 1], P_filt[t - 1])

        x_pred[t] = x_p
        P_pred[t] = P_p
        x_filt[t], P_filt[t] = kf_update(x_p, P_p, indicator_30m[t])

    # FLKS: for each t, smooth back lag steps
    positions = np.copy(x_filt[:, 0])

    for t in range(n):
        end = min(t + lag, n - 1)
        if end <= t:
            continue

        x_s = np.copy(x_filt[end])
        P_s = np.copy(P_filt[end])

        for k in range(end - 1, t - 1, -1):
            P_pk1 = P_pred[k + 1]
            try:
                C = P_filt[k] @ A.T @ np.linalg.inv(P_pk1)
            except np.linalg.LinAlgError:
                C = P_filt[k] @ A.T @ np.linalg.pinv(P_pk1)
            x_s = x_filt[k] + C @ (x_s - x_pred[k + 1])
            P_s = P_filt[k] + C @ (P_s - P_pk1) @ C.T

        positions[t] = x_s[0]

    slopes = np.full(n, np.nan)
    for t in range(2, n):
        slopes[t] = positions[t - 1] - positions[t - 2]

    return positions, slopes


# ============================================================================
# TEST 2: FLKS(N=2) with 5min micro-updates between 30min candles
# ============================================================================

def run_flks_30m_with_5m_micro(indicator_30m: np.ndarray,
                                macd_live_per_candle: list,
                                lag: int = 2):
    """
    FLKS(N=2) with MACD live 5min micro-injections between 30min candles.

    For each 30min candle t:
      1. Predict from state at candle t-1
      2. For each of the 6 MACD live values (provisional) within candle t:
         - update with macd_live[k], then predict to next sub-step
      3. Fix the state at candle t = state after last valid update
      4. Store for FLKS backward smoothing

    The observations injected are MACD live values (frozen/provisional EMA),
    which are on the same scale as MACD 30min — no space mismatch.

    Returns:
        positions: (N,) smoothed position at each 30min candle
        slopes:    (N,) pente[t] = pos[t-1] - pos[t-2]
    """
    n = len(indicator_30m)

    # Forward filter with 5min micro-updates
    x_filt = np.zeros((n, 2))
    P_filt = np.zeros((n, 2, 2))
    x_pred = np.zeros((n, 2))
    P_pred = np.zeros((n, 2, 2))

    for t in range(n):
        if t == 0:
            # Initialize with first 30min value
            x_cur = np.array([indicator_30m[0], 0.0])
            P_cur = np.eye(2)
            x_pred[0] = x_cur
            P_pred[0] = P_cur
            x_filt[0], P_filt[0] = kf_update(x_cur, P_cur, indicator_30m[0])
        else:
            # Predict from previous 30min state
            x_p, P_p = kf_predict(x_filt[t - 1], P_filt[t - 1])
            x_pred[t] = x_p
            P_pred[t] = P_p

            # Inject MACD live 5min values incrementally
            macd_vals = macd_live_per_candle[t]
            x_cur = x_p
            P_cur = P_p

            # Filter valid (non-NaN) MACD live values
            valid_vals = [v for v in macd_vals if not np.isnan(v)]

            if len(valid_vals) > 0:
                for k, m5 in enumerate(valid_vals):
                    x_cur, P_cur = kf_update(x_cur, P_cur, m5)
                    if k < len(valid_vals) - 1:
                        x_cur, P_cur = kf_predict(x_cur, P_cur)
            else:
                # Fallback: just update with 30min MACD
                x_cur, P_cur = kf_update(x_cur, P_cur, indicator_30m[t])

            # Fix state at candle t
            x_filt[t] = x_cur
            P_filt[t] = P_cur

    # FLKS backward: identical to Test 1
    positions = np.copy(x_filt[:, 0])

    for t in range(n):
        end = min(t + lag, n - 1)
        if end <= t:
            continue

        x_s = np.copy(x_filt[end])
        P_s = np.copy(P_filt[end])

        for k in range(end - 1, t - 1, -1):
            P_pk1 = P_pred[k + 1]
            try:
                C = P_filt[k] @ A.T @ np.linalg.inv(P_pk1)
            except np.linalg.LinAlgError:
                C = P_filt[k] @ A.T @ np.linalg.pinv(P_pk1)
            x_s = x_filt[k] + C @ (x_s - x_pred[k + 1])
            P_s = P_filt[k] + C @ (P_s - P_pk1) @ C.T

        positions[t] = x_s[0]

    slopes = np.full(n, np.nan)
    for t in range(2, n):
        slopes[t] = positions[t - 1] - positions[t - 2]

    return positions, slopes


# ============================================================================
# METRICS
# ============================================================================

def sign_concordance(slopes_test, slopes_oracle, start, end):
    """% of samples where sign(test) == sign(oracle), ignoring NaN and zero."""
    EPSILON = 1e-8
    mask = (~np.isnan(slopes_test[start:end])
            & ~np.isnan(slopes_oracle[start:end])
            & (np.abs(slopes_oracle[start:end]) > EPSILON))
    st = np.sign(slopes_test[start:end][mask])
    so = np.sign(slopes_oracle[start:end][mask])
    n_valid = len(st)
    if n_valid == 0:
        return np.nan, 0
    concordance = np.mean(st == so) * 100.0
    return concordance, n_valid


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(slopes_oracle, slopes_t1, slopes_t2, eval_start, eval_end,
                 output_dir):
    """Plot slopes and concordance comparison."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    t = np.arange(eval_start, eval_end)

    # Panel 1: slopes overlay
    ax = axes[0]
    ax.plot(t, slopes_oracle[eval_start:eval_end], color='black',
            linewidth=1.2, alpha=0.8, label='Oracle (pykalman.smooth)')
    ax.plot(t, slopes_t1[eval_start:eval_end], color='tab:red',
            linewidth=0.8, alpha=0.6, label='Test 1: FLKS 30m only')
    ax.plot(t, slopes_t2[eval_start:eval_end], color='tab:blue',
            linewidth=0.8, alpha=0.6, label='Test 2: FLKS 30m + 5m micro')
    ax.set_ylabel('Slope (MACD 30m)')
    ax.set_title('Slope Estimation — MACD 30min BTC')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.4)

    # Panel 2: sign match (rolling)
    ax = axes[1]
    window = 50
    sign_o = np.sign(slopes_oracle[eval_start:eval_end])
    sign_1 = np.sign(slopes_t1[eval_start:eval_end])
    sign_2 = np.sign(slopes_t2[eval_start:eval_end])
    match_1 = (sign_1 == sign_o).astype(float)
    match_2 = (sign_2 == sign_o).astype(float)
    if len(match_1) >= window:
        roll_1 = np.convolve(match_1, np.ones(window) / window, mode='same') * 100
        roll_2 = np.convolve(match_2, np.ones(window) / window, mode='same') * 100
        ax.plot(t, roll_1, color='tab:red', linewidth=1.0,
                label=f'Test 1 sign match MA({window})')
        ax.plot(t, roll_2, color='tab:blue', linewidth=1.0,
                label=f'Test 2 sign match MA({window})')
    ax.set_ylabel('Sign concordance %')
    ax.set_title(f'Rolling Sign Concordance vs Oracle (window={window})')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(40, 100)

    # Panel 3: residuals
    ax = axes[2]
    resid_1 = slopes_t1[eval_start:eval_end] - slopes_oracle[eval_start:eval_end]
    resid_2 = slopes_t2[eval_start:eval_end] - slopes_oracle[eval_start:eval_end]
    ax.plot(t, resid_1, color='tab:red', linewidth=0.6, alpha=0.5,
            label='Test 1 - Oracle')
    ax.plot(t, resid_2, color='tab:blue', linewidth=0.6, alpha=0.5,
            label='Test 2 - Oracle')
    ax.set_ylabel('Residual')
    ax.set_xlabel('30min candle index')
    ax.set_title('Residuals vs Oracle')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / 'flks_30m_vs_5m_micro.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {out_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='FLKS 30m vs FLKS 30m+5m micro — concordance vs oracle')
    parser.add_argument('--csv', type=str, default='data_trad/BTCUSD_all_5m.csv',
                        help='Path to BTC 5min CSV')
    parser.add_argument('--n-candles-30m', type=int, default=5000,
                        help='Number of 30min candles to use')
    parser.add_argument('--eval-start', type=int, default=1000,
                        help='Start index for evaluation (skip warmup)')
    parser.add_argument('--flks-lag', type=int, default=2,
                        help='FLKS lag N (default: 2)')
    parser.add_argument('--output-dir', type=str, default='plots',
                        help='Directory for output plots')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load 5min CSV
    # ------------------------------------------------------------------
    print(f"[1/7] Loading {args.csv} ...")
    df_5m = load_csv(args.csv)
    print(f"       {len(df_5m):,} 5min candles ({df_5m.index[0]} → {df_5m.index[-1]})")

    # ------------------------------------------------------------------
    # 2. Resample to 30min
    # ------------------------------------------------------------------
    print("[2/7] Resampling to 30min ...")
    df_30m = resample_ohlcv(df_5m, 30)
    print(f"       {len(df_30m):,} 30min candles available")

    # Take last n_candles_30m
    n30 = args.n_candles_30m
    if len(df_30m) > n30:
        df_30m = df_30m.iloc[-n30:]
        print(f"       Using last {n30:,} ({df_30m.index[0]} → {df_30m.index[-1]})")

    # Also trim 5min to match the 30min window
    df_5m = df_5m.loc[df_30m.index[0]:df_30m.index[-1] + pd.Timedelta(minutes=29)]
    print(f"       Corresponding 5min: {len(df_5m):,} candles")

    # ------------------------------------------------------------------
    # 3. Compute MACD on 30min
    # ------------------------------------------------------------------
    print("[3/7] Computing MACD on 30min candles ...")
    macd_30m = calculate_macd(df_30m).values.astype(np.float64)
    print(f"       MACD range: [{np.nanmin(macd_30m):.2f}, {np.nanmax(macd_30m):.2f}]")

    # ------------------------------------------------------------------
    # 4. Compute MACD live at 5min resolution (for Test 2)
    # ------------------------------------------------------------------
    print("[4/7] Computing MACD live (frozen/provisional) at 5min resolution ...")
    is_close_30m = compute_bucket_close_mask(df_5m.index, 30)
    macd_live_5m = compute_macd_live(
        df_5m['close'].values.astype(np.float64),
        is_close_30m
    )
    n_valid_live = np.sum(~np.isnan(macd_live_5m))
    print(f"       {n_valid_live:,} valid MACD live values out of {len(macd_live_5m):,} 5min steps")

    # Group MACD live values per 30min candle
    macd_live_per_candle = []
    for i, ts_30m in enumerate(df_30m.index):
        bucket_start = ts_30m
        bucket_end = ts_30m + pd.Timedelta(minutes=29, seconds=59)
        mask = (df_5m.index >= bucket_start) & (df_5m.index <= bucket_end)
        vals = macd_live_5m[mask.values] if hasattr(mask, 'values') else macd_live_5m[mask]
        macd_live_per_candle.append(vals)

    # Stats
    lengths = [len(c) for c in macd_live_per_candle]
    n_with_data = sum(1 for c in macd_live_per_candle if np.any(~np.isnan(c)))
    print(f"       5min per 30min candle: min={min(lengths)}, "
          f"max={max(lengths)}, median={int(np.median(lengths))}, "
          f"candles with valid MACD live: {n_with_data}/{len(lengths)}")

    # ------------------------------------------------------------------
    # 5. Oracle: pykalman.smooth() on 30min
    # ------------------------------------------------------------------
    print("[5/7] Computing oracle (pykalman.smooth on 30min) ...")
    pos_oracle, slopes_oracle = compute_oracle(macd_30m)
    n_valid_oracle = np.sum(~np.isnan(slopes_oracle[args.eval_start:n30]))
    print(f"       Valid slopes in [{args.eval_start}:{n30}]: {n_valid_oracle:,}")

    # ------------------------------------------------------------------
    # 6. Test 1: FLKS 30min only
    # ------------------------------------------------------------------
    print(f"[6/7] Test 1: FLKS(N={args.flks_lag}) on 30min candles only ...")
    pos_t1, slopes_t1 = run_flks_30m(macd_30m, lag=args.flks_lag)

    conc_t1, n_t1 = sign_concordance(slopes_t1, slopes_oracle,
                                      args.eval_start, n30)
    print(f"       Sign concordance vs oracle: {conc_t1:.2f}% ({n_t1:,} samples)")

    # ------------------------------------------------------------------
    # 7. Test 2: FLKS 30min + 5min micro
    # ------------------------------------------------------------------
    print(f"[7/7] Test 2: FLKS(N={args.flks_lag}) with MACD live 5min micro-updates ...")
    pos_t2, slopes_t2 = run_flks_30m_with_5m_micro(
        macd_30m, macd_live_per_candle, lag=args.flks_lag)

    conc_t2, n_t2 = sign_concordance(slopes_t2, slopes_oracle,
                                      args.eval_start, n30)
    print(f"       Sign concordance vs oracle: {conc_t2:.2f}% ({n_t2:,} samples)")

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print(f"  RÉSULTATS — Concordance de signe vs Oracle (pykalman.smooth)")
    print(f"  Évaluation: [{args.eval_start}:{n30}] = {n30 - args.eval_start} bougies 30min")
    print(f"  Kalman: Q={KALMAN_PROCESS_VAR}, R={KALMAN_MEASURE_VAR}")
    print(f"  FLKS lag: N={args.flks_lag}")
    print(f"{'=' * 65}")
    print(f"  {'Méthode':<35} {'Concordance':>12} {'N samples':>10}")
    print(f"  {'-' * 59}")
    print(f"  {'Test 1: FLKS 30min only':<35} {conc_t1:>11.2f}% {n_t1:>10,}")
    print(f"  {'Test 2: FLKS 30min + 5min micro':<35} {conc_t2:>11.2f}% {n_t2:>10,}")
    print(f"  {'-' * 59}")
    delta = conc_t2 - conc_t1
    print(f"  {'Gain micro-updates':<35} {delta:>+11.2f}pp")
    print(f"{'=' * 65}")

    # MSE bonus
    mask = (~np.isnan(slopes_t1[args.eval_start:n30])
            & ~np.isnan(slopes_t2[args.eval_start:n30])
            & ~np.isnan(slopes_oracle[args.eval_start:n30]))
    if mask.sum() > 0:
        s1 = slopes_t1[args.eval_start:n30][mask]
        s2 = slopes_t2[args.eval_start:n30][mask]
        so = slopes_oracle[args.eval_start:n30][mask]
        mse_t1 = np.mean((s1 - so) ** 2)
        mse_t2 = np.mean((s2 - so) ** 2)
        print(f"\n  MSE vs oracle:")
        print(f"    Test 1: {mse_t1:.6e}")
        print(f"    Test 2: {mse_t2:.6e}")
        if mse_t1 > 0:
            print(f"    Gain:   {(1 - mse_t2 / mse_t1) * 100:+.2f}%")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    eval_end = min(n30, len(slopes_oracle))
    print(f"\nGenerating plots ...")
    plot_results(slopes_oracle, slopes_t1, slopes_t2,
                 args.eval_start, eval_end, output_dir)

    print(f"\nDone.")


if __name__ == '__main__':
    main()
