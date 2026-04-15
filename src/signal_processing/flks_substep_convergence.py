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
    """Predict with 30min transition."""
    return A @ x, A @ P @ A.T + Q


# Sub-step transition (5min = 1/6 of 30min)
DT_SUB = 1.0 / 6.0
A_SUB = np.array([[1.0, DT_SUB],
                   [0.0, 1.0]])
Q_SUB = Q * DT_SUB


def kf_predict_sub(x, P):
    """Predict with 5min sub-step transition."""
    return A_SUB @ x, A_SUB @ P @ A_SUB.T + Q_SUB


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

    # Forward pass
    # x_pred[t] / P_pred[t] must satisfy the RTS invariant:
    #   x_pred[t] = A @ x_filt[t-1]
    #   P_pred[t] = A @ P_filt[t-1] @ A.T + Q
    # For sub-step variants, x_filt[t] comes from micro-updates,
    # so x_pred[t+1] must be recomputed AFTER x_filt[t] is finalized.

    for t in range(n):
        if t == 0:
            x_p = np.array([indicator_30m[0], 0.0])
            P_p = np.eye(2)
            x_pred[0] = x_p
            P_pred[0] = P_p
            x_filt[0], P_filt[0] = kf_update(x_p, P_p, indicator_30m[0])
        else:
            # x_pred[t] / P_pred[t] already set at end of previous iteration
            x_p = x_pred[t]
            P_p = P_pred[t]

            if n_substeps == 0:
                # Pure 30min: single update
                x_filt[t], P_filt[t] = kf_update(x_p, P_p, indicator_30m[t])
            else:
                # Sub-step forward: n_substeps cycles of (predict_5min + update)
                # Start from previous candle state
                macd_vals = macd_live_per_candle[t]
                valid_vals = [v for v in macd_vals if not np.isnan(v)]
                use = valid_vals[:n_substeps]

                x_cur = x_filt[t - 1].copy()
                P_cur = P_filt[t - 1].copy()

                if len(use) > 0:
                    for k, m5 in enumerate(use):
                        x_cur, P_cur = kf_predict_sub(x_cur, P_cur)
                        x_cur, P_cur = kf_update(x_cur, P_cur, m5)
                else:
                    x_cur, P_cur = kf_update(x_p, P_p, indicator_30m[t])

                x_filt[t] = x_cur
                P_filt[t] = P_cur

        # Compute x_pred[t+1] from finalized x_filt[t] — ensures RTS consistency
        if t < n - 1:
            x_pred[t + 1], P_pred[t + 1] = kf_predict(x_filt[t], P_filt[t])

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


def find_oracle_transitions(slopes_oracle, start, end):
    """
    Find indices where oracle slope changes sign (positive <-> negative).
    These are the only samples that matter for trading.
    Returns boolean mask over [start:end].
    """
    EPSILON = 1e-8
    s_o = slopes_oracle[start:end]
    sign_o = np.where(np.abs(s_o) < EPSILON, 0, np.sign(s_o))
    transitions = np.zeros(len(s_o), dtype=bool)
    for i in range(1, len(s_o)):
        if sign_o[i] != 0 and sign_o[i - 1] != 0 and sign_o[i] != sign_o[i - 1]:
            transitions[i] = True
    return transitions


def sign_concordance_at_transitions(slopes_test, slopes_oracle, start, end,
                                     transition_mask):
    """Concordance only at oracle transition points."""
    s_t = slopes_test[start:end]
    s_o = slopes_oracle[start:end]
    mask = (transition_mask
            & ~np.isnan(s_t) & ~np.isnan(s_o))
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

    # Vérification cohérence : dernier MACD live du bucket == MACD 30min
    print("       Vérification cohérence MACD live vs MACD 30min ...")
    n_checked = 0
    max_err = 0.0
    for t in range(n30):
        vals = macd_live_per_candle[t]
        valid = [v for v in vals if not np.isnan(v)]
        if len(valid) > 0 and not np.isnan(macd_30m[t]):
            err = abs(valid[-1] - macd_30m[t])
            max_err = max(max_err, err)
            n_checked += 1
    print(f"       Checked {n_checked} candles, max |last_5m - macd_30m| = {max_err:.6f}")
    if max_err > 1e-6:
        print(f"       WARNING: écart > 1e-6 entre MACD live closure et MACD 30min!")

    # ------------------------------------------------------------------
    print("[4/7] Oracle: pykalman.smooth() on 30min ...")
    _, slopes_oracle = compute_oracle(macd_30m)

    # Detect oracle transitions for transition-only metric
    trans_mask = find_oracle_transitions(slopes_oracle, args.eval_start, n30)
    n_trans = trans_mask.sum()
    n_eval = n30 - args.eval_start
    print(f"       Transitions oracle dans [{args.eval_start}:{n30}]: "
          f"{n_trans} ({n_trans / n_eval * 100:.1f}% des samples)")

    # Persistence baseline: % of time sign stays the same
    EPSILON = 1e-8
    s_o = slopes_oracle[args.eval_start:n30]
    sign_o = np.where(np.abs(s_o) < EPSILON, 0, np.sign(s_o))
    valid_signs = sign_o[sign_o != 0]
    if len(valid_signs) > 1:
        persistence = np.mean(valid_signs[1:] == valid_signs[:-1]) * 100.0
        print(f"       Persistence oracle (sign[t]==sign[t-1]): {persistence:.1f}%")

    # ------------------------------------------------------------------
    print("[5/7] Running FLKS for each substep count (0=baseline, 1..6) ...")

    all_slopes = {}  # k -> slopes array

    # k=0 : baseline FLKS 30min pur
    print("       k=0 (30min pur) ...", end=" ", flush=True)
    slopes_0 = run_flks_with_substeps(macd_30m, macd_live_per_candle,
                                       n_substeps=0, lag=args.flks_lag)
    all_slopes[0] = slopes_0
    conc_0, _ = sign_concordance(slopes_0, slopes_oracle, args.eval_start, n30)
    conc_0_t, _ = sign_concordance_at_transitions(
        slopes_0, slopes_oracle, args.eval_start, n30, trans_mask)
    print(f"all={conc_0:.2f}%  trans={conc_0_t:.2f}%")

    # k=1..6
    for k in range(1, 7):
        label = f"{k*5}min"
        print(f"       k={k} ({label}) ...", end=" ", flush=True)
        slopes_k = run_flks_with_substeps(macd_30m, macd_live_per_candle,
                                           n_substeps=k, lag=args.flks_lag)
        all_slopes[k] = slopes_k
        conc_k, _ = sign_concordance(slopes_k, slopes_oracle,
                                      args.eval_start, n30)
        conc_k_t, _ = sign_concordance_at_transitions(
            slopes_k, slopes_oracle, args.eval_start, n30, trans_mask)
        print(f"all={conc_k:.2f}%  trans={conc_k_t:.2f}%")

    # ------------------------------------------------------------------
    print(f"\n[6/7] Résultats")

    # Build results table
    results = []
    for k in range(7):
        conc_all, n_all = sign_concordance(
            all_slopes[k], slopes_oracle, args.eval_start, n30)
        conc_trans, n_t = sign_concordance_at_transitions(
            all_slopes[k], slopes_oracle, args.eval_start, n30, trans_mask)
        results.append((k, conc_all, n_all, conc_trans, n_t))

    conc_all_base = results[0][1]
    conc_trans_base = results[0][3]

    print(f"{'=' * 80}")
    print(f"  Concordance de signe vs Oracle (pykalman.smooth 30min)")
    print(f"  Évaluation: [{args.eval_start}:{n30}]  |  FLKS lag N={args.flks_lag}")
    print(f"  Transitions: {n_trans}  |  Persistence: {persistence:.1f}%")
    print(f"  Pente: pos[t-1] - pos[t-2]")
    print(f"{'=' * 80}")
    print(f"  {'Sous-pas':<10} {'Temps':<8} {'All':>8} {'d/k=0':>8}"
          f"  {'Trans':>8} {'d/k=0':>8}  {'N_trans':>7}")
    print(f"  {'-' * 68}")

    for k, conc_a, n_a, conc_t, n_t in results:
        if k == 0:
            label = "30m pur"
            d_a = "  base"
            d_t = "  base"
        else:
            label = f"{k*5}min"
            d_a = f"{conc_a - conc_all_base:+6.2f}"
            d_t = f"{conc_t - conc_trans_base:+6.2f}"
        print(f"  k={k:<6} {label:<8} {conc_a:>7.2f}% {d_a:>8}"
              f"  {conc_t:>7.2f}% {d_t:>8}  {n_t:>7,}")

    print(f"  {'-' * 68}")
    print(f"{'=' * 80}")

    # ------------------------------------------------------------------
    print(f"\n[7/7] Plots ...")

    concs_all = [r[1] for r in results]
    concs_trans = [r[3] for r in results]
    ks = [r[0] for r in results]
    x_pos = np.arange(len(ks))
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x_pos - bar_w / 2, concs_all, bar_w,
                    color='steelblue', alpha=0.8, label='All samples')
    bars2 = ax.bar(x_pos + bar_w / 2, concs_trans, bar_w,
                    color='tomato', alpha=0.8, label='Transitions only')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['30m\npur'] + [f'k={k}\n{k*5}min' for k in range(1, 7)])
    ax.set_ylabel('Sign concordance vs oracle (%)')
    ax.set_title(f'FLKS(N={args.flks_lag}) — Convergence par sous-pas 5min\n'
                 f'Oracle = pykalman.smooth MACD 30min BTC  |  '
                 f'Transitions: {n_trans}  |  Persistence: {persistence:.0f}%')
    ax.set_ylim(max(0, min(min(concs_all), min(concs_trans)) - 10), 100)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out_path = output_dir / 'flks_substep_convergence.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
