#!/usr/bin/env python3
"""
FLKS convergence par sous-pas 5min — à quel moment le signe converge ?
======================================================================

Oracle : pykalman.smooth() sur 5000 bougies MACD 30min (référence fixe).

Pour chaque variante k=1..6 :
  - Injecter les k premiers MACD live 5min par bougie dans le FLKS
  - Calculer pente[t] = pos[t-1] - pos[t-2]
  - Mesurer % concordance de signe vs oracle sur [1000:5000]

Baseline : FLKS 30min pur (1 observation par bougie, pas de sous-pas).

Question : à partir de quel sous-pas le signe converge vers l'oracle ?

Usage:
    python src/signal_processing/flks_substep_convergence.py \
        --csv data_trad/BTCUSD_all_5m.csv

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
# PARAMETERS (from pipeline)
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
# DATA LOADING (from pipeline)
# ============================================================================

def load_csv(path: str) -> pd.DataFrame:
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


def resample_ohlcv(df_5min, tf_minutes):
    return df_5min.resample(f'{tf_minutes}min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()


def calculate_macd(df):
    ema_f = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_s = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return line - sig


def compute_macd_live(close_5min, is_close):
    """MACD live frozen/provisional (from pipeline)."""
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
    bucket = index_5min.floor(f'{tf_minutes}min').values
    next_bucket = np.append(bucket[1:], np.datetime64('NaT'))
    return (bucket != next_bucket) | pd.isna(next_bucket)


# ============================================================================
# ORACLE: pykalman.smooth() on 30min (from pipeline)
# ============================================================================

def compute_oracle(indicator_30m):
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
# KALMAN PRIMITIVES
# ============================================================================

def kf_predict(x, P):
    return A @ x, A @ P @ A.T + Q


def kf_update(x_p, P_p, z_obs):
    y = z_obs - H @ x_p
    S = H @ P_p @ H.T + R
    K = P_p @ H.T / S[0, 0]
    x_f = x_p + (K @ y).ravel()
    P_f = (np.eye(2) - K @ H) @ P_p
    return x_f, P_f


# ============================================================================
# FLKS with configurable number of sub-steps per candle
# ============================================================================

def run_flks_with_substeps(indicator_30m, macd_live_per_candle,
                           n_substeps, lag=2):
    """
    FLKS(N=lag) where each 30min candle gets n_substeps micro-updates.

    n_substeps=0 : pure 30min (1 update per candle with indicator_30m)
    n_substeps=k : inject the first k MACD live values, then fix state

    Returns:
        slopes: (N,) pente[t] = pos[t-1] - pos[t-2]
    """
    n = len(indicator_30m)

    x_filt = np.zeros((n, 2))
    P_filt = np.zeros((n, 2, 2))
    x_pred = np.zeros((n, 2))
    P_pred = np.zeros((n, 2, 2))

    for t in range(n):
        if t == 0:
            x_p = np.array([indicator_30m[0], 0.0])
            P_p = np.eye(2)
            x_pred[0] = x_p
            P_pred[0] = P_p
            x_filt[0], P_filt[0] = kf_update(x_p, P_p, indicator_30m[0])
        else:
            x_p, P_p = kf_predict(x_filt[t - 1], P_filt[t - 1])
            x_pred[t] = x_p
            P_pred[t] = P_p

            if n_substeps == 0:
                # Pure 30min: single update
                x_filt[t], P_filt[t] = kf_update(x_p, P_p, indicator_30m[t])
            else:
                # Inject first n_substeps MACD live values
                macd_vals = macd_live_per_candle[t]
                valid_vals = [v for v in macd_vals if not np.isnan(v)]
                use = valid_vals[:n_substeps]

                x_cur, P_cur = x_p, P_p
                if len(use) > 0:
                    for k, m5 in enumerate(use):
                        x_cur, P_cur = kf_update(x_cur, P_cur, m5)
                        if k < len(use) - 1:
                            x_cur, P_cur = kf_predict(x_cur, P_cur)
                else:
                    x_cur, P_cur = kf_update(x_cur, P_cur, indicator_30m[t])

                x_filt[t] = x_cur
                P_filt[t] = P_cur

    # FLKS backward
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

    return slopes


# ============================================================================
# METRICS
# ============================================================================

def sign_concordance(slopes_test, slopes_oracle, start, end):
    EPSILON = 1e-8
    s_t = slopes_test[start:end]
    s_o = slopes_oracle[start:end]
    mask = (~np.isnan(s_t) & ~np.isnan(s_o)
            & (np.abs(s_o) > EPSILON))
    st = np.sign(s_t[mask])
    so = np.sign(s_o[mask])
    n_valid = len(st)
    if n_valid == 0:
        return np.nan, 0
    return np.mean(st == so) * 100.0, n_valid


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='FLKS substep convergence — sign concordance vs oracle')
    parser.add_argument('--csv', type=str, default='data_trad/BTCUSD_all_5m.csv')
    parser.add_argument('--n-candles-30m', type=int, default=5000)
    parser.add_argument('--eval-start', type=int, default=1000)
    parser.add_argument('--flks-lag', type=int, default=2)
    parser.add_argument('--output-dir', type=str, default='plots')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n30 = args.n_candles_30m

    # ------------------------------------------------------------------
    print(f"[1/6] Loading {args.csv} ...")
    df_5m = load_csv(args.csv)
    print(f"       {len(df_5m):,} 5min candles")

    # ------------------------------------------------------------------
    print("[2/6] Resampling to 30min ...")
    df_30m = resample_ohlcv(df_5m, 30)
    if len(df_30m) > n30:
        df_30m = df_30m.iloc[-n30:]
    df_5m = df_5m.loc[df_30m.index[0]:df_30m.index[-1] + pd.Timedelta(minutes=29)]
    print(f"       {len(df_30m):,} bougies 30min, {len(df_5m):,} bougies 5min")

    # ------------------------------------------------------------------
    print("[3/6] Computing MACD 30min + MACD live 5min ...")
    macd_30m = calculate_macd(df_30m).values.astype(np.float64)

    is_close_30m = compute_bucket_close_mask(df_5m.index, 30)
    macd_live_5m = compute_macd_live(
        df_5m['close'].values.astype(np.float64), is_close_30m)

    # Group MACD live per 30min candle
    macd_live_per_candle = []
    for ts_30m in df_30m.index:
        bucket_end = ts_30m + pd.Timedelta(minutes=29, seconds=59)
        mask = (df_5m.index >= ts_30m) & (df_5m.index <= bucket_end)
        macd_live_per_candle.append(macd_live_5m[mask])

    print(f"       MACD 30m range: [{np.nanmin(macd_30m):.1f}, {np.nanmax(macd_30m):.1f}]")

    # ------------------------------------------------------------------
    print("[4/6] Oracle: pykalman.smooth() on 30min ...")
    _, slopes_oracle = compute_oracle(macd_30m)

    # ------------------------------------------------------------------
    print("[5/6] Running FLKS for each substep count (0=baseline, 1..6) ...")

    results = []

    # k=0 : baseline FLKS 30min pur
    print("       k=0 (30min pur) ...", end=" ", flush=True)
    slopes_0 = run_flks_with_substeps(macd_30m, macd_live_per_candle,
                                       n_substeps=0, lag=args.flks_lag)
    conc_0, n_0 = sign_concordance(slopes_0, slopes_oracle, args.eval_start, n30)
    results.append((0, conc_0, n_0))
    print(f"{conc_0:.2f}%")

    # k=1..6
    for k in range(1, 7):
        label = f"{k*5}min"
        print(f"       k={k} ({label}) ...", end=" ", flush=True)
        slopes_k = run_flks_with_substeps(macd_30m, macd_live_per_candle,
                                           n_substeps=k, lag=args.flks_lag)
        conc_k, n_k = sign_concordance(slopes_k, slopes_oracle,
                                        args.eval_start, n30)
        results.append((k, conc_k, n_k))
        print(f"{conc_k:.2f}%")

    # ------------------------------------------------------------------
    print(f"\n[6/6] Résultats")
    print(f"{'=' * 65}")
    print(f"  Concordance de signe vs Oracle (pykalman.smooth 30min)")
    print(f"  Évaluation: [{args.eval_start}:{n30}]  |  FLKS lag N={args.flks_lag}")
    print(f"  Pente: pos[t-1] - pos[t-2]")
    print(f"{'=' * 65}")
    print(f"  {'Sous-pas':<12} {'Temps':<10} {'Concordance':>12} {'Delta vs k=0':>14}")
    print(f"  {'-' * 50}")

    conc_baseline = results[0][1]
    for k, conc, n_valid in results:
        if k == 0:
            label = "30m pur"
            delta_str = "(baseline)"
        else:
            label = f"{k*5}min"
            delta = conc - conc_baseline
            delta_str = f"{delta:+.2f}pp"
        print(f"  k={k:<8} {label:<10} {conc:>11.2f}% {delta_str:>14}")

    print(f"  {'-' * 50}")
    print(f"{'=' * 65}")

    # ------------------------------------------------------------------
    # Plot
    ks = [r[0] for r in results]
    concs = [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['black'] + ['tab:blue'] * 6
    ax.bar(range(len(ks)), concs, color=colors, alpha=0.8, edgecolor='white')
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(['30m\npur'] + [f'k={k}\n{k*5}min' for k in range(1, 7)])
    ax.set_ylabel('Sign concordance vs oracle (%)')
    ax.set_title(f'FLKS(N={args.flks_lag}) — Convergence par sous-pas 5min\n'
                 f'Oracle = pykalman.smooth sur MACD 30min BTC')
    ax.set_ylim(max(0, min(concs) - 5), 100)
    ax.grid(True, axis='y', alpha=0.3)

    # Annotate values
    for i, (k, c, _) in enumerate(results):
        ax.text(i, c + 0.5, f'{c:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out_path = output_dir / 'flks_substep_convergence.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
