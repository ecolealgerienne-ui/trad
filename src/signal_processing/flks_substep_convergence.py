#!/usr/bin/env python3
"""
FLKS convergence par sous-pas 5min — à quel moment le signe converge ?
======================================================================

Oracle : pykalman.smooth() sur 5000 bougies MACD 30min (référence fixe).

Test 1 : FLKS 30min pur
  - Forward filter 30min, puis backward 2 pas depuis x_filt[t]
  - slope[t] = smoothed[t-1] - smoothed[t-2]

Test 2 : FLKS 30min + sous-pas 5min (k=1..6)
  - Même forward filter 30min, mêmes gains C
  - Mais backward 2 pas depuis x_provisoire (x_filt[t] + k micro-updates 5min)
  - slope[t,k] = smoothed[t-1] - smoothed[t-2]

Métrique : % concordance de signe vs oracle sur [eval_start:n30]

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

DT_SUB = 1.0 / 6.0
A_SUB = np.array([[1.0, DT_SUB],
                   [0.0, 1.0]])
Q_SUB = Q * DT_SUB

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


# ============================================================================
# DATA LOADING (from pipeline)
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
    vd = indicator_30m[~np.isnan(indicator_30m)]
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
    positions[~np.isnan(indicator_30m)] = smooth_means[:, 0]

    slopes = np.full(n, np.nan)
    for t in range(2, n):
        if not np.isnan(positions[t - 1]) and not np.isnan(positions[t - 2]):
            slopes[t] = positions[t - 1] - positions[t - 2]
    return positions, slopes


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


# ============================================================================
# FORWARD FILTER 30min (shared by Test 1 and Test 2)
# ============================================================================

def forward_filter_30m(indicator_30m):
    """
    Standard Kalman forward filter on 30min MACD.
    Returns all states needed for backward smoothing.
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
        else:
            x_p = A @ x_filt[t - 1]
            P_p = A @ P_filt[t - 1] @ A.T + Q

        x_pred[t] = x_p
        P_pred[t] = P_p
        x_filt[t], P_filt[t] = kf_update(x_p, P_p, indicator_30m[t])

    # Precompute RTS smoother gains C[t] for t = 0..n-2
    # C[t] = P_filt[t] @ A.T @ inv(P_pred[t+1])
    C = np.zeros((n, 2, 2))
    for t in range(n - 1):
        P_pk1 = P_pred[t + 1]
        det = P_pk1[0, 0] * P_pk1[1, 1] - P_pk1[0, 1] * P_pk1[1, 0]
        if abs(det) > 1e-15:
            inv_P = np.array([[P_pk1[1, 1], -P_pk1[0, 1]],
                              [-P_pk1[1, 0], P_pk1[0, 0]]]) / det
        else:
            inv_P = np.linalg.pinv(P_pk1)
        C[t] = P_filt[t] @ A.T @ inv_P

    return x_filt, P_filt, x_pred, P_pred, C


# ============================================================================
# TEST 1: FLKS 30min pur
# ============================================================================

def compute_slopes_test1(x_filt, x_pred, C):
    """
    Backward 2 pas depuis x_filt[t] pour chaque t >= 2.
    slope[t] = smoothed[t-1][0] - smoothed[t-2][0]
    """
    n = len(x_filt)
    slopes = np.full(n, np.nan)

    for t in range(2, n):
        # Backward pas 1 : smooth t-1 using x_filt[t]
        sm_t1 = x_filt[t - 1] + C[t - 1] @ (x_filt[t] - x_pred[t])
        # Backward pas 2 : smooth t-2 using smoothed[t-1]
        sm_t2 = x_filt[t - 2] + C[t - 2] @ (sm_t1 - x_pred[t - 1])
        slopes[t] = sm_t1[0] - sm_t2[0]

    return slopes


# ============================================================================
# TEST 2: FLKS 30min + sous-pas 5min
# ============================================================================

def compute_slopes_test2(x_filt, P_filt, x_pred, C,
                          macd_live_per_candle, n_substeps):
    """
    Backward 2 pas depuis x_provisoire pour chaque t >= 2.
    x_provisoire = x_filt[t] + n_substeps micro-updates 5min de la bougie t+1.
    slope[t] = smoothed[t-1][0] - smoothed[t-2][0]
    """
    n = len(x_filt)
    slopes = np.full(n, np.nan)

    for t in range(2, n - 1):  # n-1 : besoin de bougie t+1
        # Construire x_provisoire : partir de x_filt[t], injecter k sous-pas
        x_cur = x_filt[t].copy()
        P_cur = P_filt[t].copy()

        macd_vals = macd_live_per_candle[t + 1]
        valid_vals = [v for v in macd_vals if not np.isnan(v)]
        use = valid_vals[:n_substeps]

        if len(use) > 0:
            for m5 in use:
                x_cur, P_cur = kf_predict_sub(x_cur, P_cur)
                x_cur, P_cur = kf_update(x_cur, P_cur, m5)

        x_prov = x_cur

        # Backward pas 1 : smooth t-1 using x_provisoire
        sm_t1 = x_filt[t - 1] + C[t - 1] @ (x_prov - x_pred[t])
        # Backward pas 2 : smooth t-2 using smoothed[t-1]
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
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='FLKS substep convergence — sign concordance vs oracle')
    parser.add_argument('--csv', type=str, default='data_trad/BTCUSD_all_5m.csv')
    parser.add_argument('--n-candles-30m', type=int, default=5000)
    parser.add_argument('--eval-start', type=int, default=1000)
    parser.add_argument('--output-dir', type=str, default='plots')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n30 = args.n_candles_30m

    # ------------------------------------------------------------------
    print(f"[1/7] Loading {args.csv} ...")
    df_5m = load_csv(args.csv)
    print(f"       {len(df_5m):,} 5min candles")

    # ------------------------------------------------------------------
    print("[2/7] Resampling to 30min ...")
    df_30m = resample_ohlcv(df_5m, 30)
    if len(df_30m) > n30:
        df_30m = df_30m.iloc[-n30:]
    df_5m = df_5m.loc[df_30m.index[0]:df_30m.index[-1] + pd.Timedelta(minutes=29)]
    print(f"       {len(df_30m):,} bougies 30min, {len(df_5m):,} bougies 5min")

    # ------------------------------------------------------------------
    print("[3/7] Computing MACD 30min + MACD live 5min ...")
    macd_30m = calculate_macd(df_30m).values.astype(np.float64)

    is_close_30m = compute_bucket_close_mask(df_5m.index, 30)
    macd_live_5m = compute_macd_live(
        df_5m['close'].values.astype(np.float64), is_close_30m)

    macd_live_per_candle = []
    for ts_30m in df_30m.index:
        bucket_end = ts_30m + pd.Timedelta(minutes=29, seconds=59)
        mask = (df_5m.index >= ts_30m) & (df_5m.index <= bucket_end)
        macd_live_per_candle.append(macd_live_5m[mask])

    # Vérification cohérence
    max_err = 0.0
    n_checked = 0
    for t in range(n30):
        vals = [v for v in macd_live_per_candle[t] if not np.isnan(v)]
        if len(vals) > 0 and not np.isnan(macd_30m[t]):
            max_err = max(max_err, abs(vals[-1] - macd_30m[t]))
            n_checked += 1
    print(f"       MACD coherence: {n_checked} candles, max err = {max_err:.2e}")

    # ------------------------------------------------------------------
    print("[4/7] Oracle: pykalman.smooth() on 30min ...")
    _, slopes_oracle = compute_oracle(macd_30m)

    trans_mask = find_oracle_transitions(slopes_oracle, args.eval_start, n30)
    n_trans = trans_mask.sum()
    n_eval = n30 - args.eval_start

    EPSILON = 1e-8
    s_o = slopes_oracle[args.eval_start:n30]
    sign_o = np.where(np.abs(s_o) < EPSILON, 0, np.sign(s_o))
    valid_signs = sign_o[sign_o != 0]
    persistence = np.mean(valid_signs[1:] == valid_signs[:-1]) * 100.0

    print(f"       Transitions: {n_trans} ({n_trans/n_eval*100:.1f}%)  "
          f"Persistence: {persistence:.1f}%")

    # ------------------------------------------------------------------
    print("[5/7] Forward filter 30min (shared) ...")
    x_filt, P_filt, x_pred, P_pred, C = forward_filter_30m(macd_30m)
    print("       Done.")

    # ------------------------------------------------------------------
    print("[6/7] Computing slopes ...")

    # Test 1
    print("       Test 1 (30min pur) ...", end=" ", flush=True)
    slopes_t1 = compute_slopes_test1(x_filt, x_pred, C)
    c_all, _ = sign_concordance(slopes_t1, slopes_oracle, args.eval_start, n30)
    c_tr, _ = sign_concordance_at_transitions(
        slopes_t1, slopes_oracle, args.eval_start, n30, trans_mask)
    print(f"all={c_all:.2f}%  trans={c_tr:.2f}%")

    # Test 2 : k=1..6
    results_t2 = []
    for k in range(1, 7):
        print(f"       Test 2 k={k} ({k*5}min) ...", end=" ", flush=True)
        slopes_k = compute_slopes_test2(
            x_filt, P_filt, x_pred, C, macd_live_per_candle, k)
        ck_all, nk_all = sign_concordance(
            slopes_k, slopes_oracle, args.eval_start, n30)
        ck_tr, nk_tr = sign_concordance_at_transitions(
            slopes_k, slopes_oracle, args.eval_start, n30, trans_mask)
        results_t2.append((k, ck_all, nk_all, ck_tr, nk_tr))
        print(f"all={ck_all:.2f}%  trans={ck_tr:.2f}%")

    # ------------------------------------------------------------------
    print(f"\n[7/7] Résultats")
    print(f"{'=' * 75}")
    print(f"  Concordance de signe vs Oracle (pykalman.smooth 30min)")
    print(f"  Éval: [{args.eval_start}:{n30}]  |  Transitions: {n_trans}  |"
          f"  Persistence: {persistence:.1f}%")
    print(f"  Pente: smoothed[t-1] - smoothed[t-2]")
    print(f"{'=' * 75}")
    print(f"  {'Méthode':<30} {'All':>8} {'d/T1':>8}  {'Trans':>8} {'d/T1':>8}")
    print(f"  {'-' * 65}")

    c_t1_all, _ = sign_concordance(slopes_t1, slopes_oracle, args.eval_start, n30)
    c_t1_tr, _ = sign_concordance_at_transitions(
        slopes_t1, slopes_oracle, args.eval_start, n30, trans_mask)

    print(f"  {'Test 1: FLKS 30min pur':<30} {c_t1_all:>7.2f}% {'  base':>8}"
          f"  {c_t1_tr:>7.2f}% {'  base':>8}")

    for k, ck_a, _, ck_t, _ in results_t2:
        da = ck_a - c_t1_all
        dt = ck_t - c_t1_tr
        print(f"  {'Test 2: k=' + str(k) + ' (' + str(k*5) + 'min)':<30}"
              f" {ck_a:>7.2f}% {da:>+7.2f}p"
              f"  {ck_t:>7.2f}% {dt:>+7.2f}p")

    print(f"  {'-' * 65}")
    print(f"{'=' * 75}")

    # ------------------------------------------------------------------
    # Plot
    labels = ['T1\n30m'] + [f'k={k}\n{k*5}m' for k in range(1, 7)]
    all_concs_all = [c_t1_all] + [r[1] for r in results_t2]
    all_concs_tr = [c_t1_tr] + [r[3] for r in results_t2]
    x_pos = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x_pos - w / 2, all_concs_all, w,
                color='steelblue', alpha=0.8, label='All samples')
    b2 = ax.bar(x_pos + w / 2, all_concs_tr, w,
                color='tomato', alpha=0.8, label='Transitions only')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Sign concordance vs oracle (%)')
    ax.set_title(f'FLKS — Convergence par sous-pas 5min de la bougie suivante\n'
                 f'Oracle=pykalman.smooth MACD 30min BTC | '
                 f'Trans={n_trans} | Persist={persistence:.0f}%')
    ax.set_ylim(max(0, min(min(all_concs_all), min(all_concs_tr)) - 10), 100)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out_path = output_dir / 'flks_substep_convergence.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
