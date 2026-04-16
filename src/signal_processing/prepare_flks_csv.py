#!/usr/bin/env python3
"""
Prepare CSV for LSTM training with FLKS backward slopes as features
====================================================================

Generates a 5min resolution CSV with:
  - MACD live features (macd_30m_live, macd_30m_filtered, macd_30m_velocity)
  - FLKS backward slopes (aq_t1_slope, aq_k1_slope, ..., aq_k6_slope)
  - Oracle labels (oracle_label_macd_30m from pykalman.smooth)

All FLKS slopes are at 30min resolution, forward-filled to 5min.
Also computes concordance table to verify consistency.

Usage:
    python src/signal_processing/prepare_flks_csv.py \
        --csv data_trad/BTCUSD_all_5m.csv --n-candles-30m 5000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    load_csv, resample_ohlcv, compute_bucket_close_mask,
    calculate_macd, compute_macd_live,
    forward_filter_30m, forward_filter_30m_adaptive,
    compute_slopes_test1, compute_slopes_test2,
    compute_oracle, sign_concordance, find_oracle_transitions,
    sign_concordance_at_transitions, group_per_candle,
    kf_update, inv2x2, is_pos_semidef,
    A, H, Q, R, KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR,
)

TRIM = 100


def compute_kalman_live_aqkf(indicator_live, is_close, aq_window=30, Q_max_factor=10.0):
    """
    AQ-KF live Kalman: forward filter with adaptive Q on closure values,
    provisional updates between closures. Returns (n, 2) = [position, velocity].
    Same as prepare_multitf_csv_aqkf.py.
    """
    _Q_fixed = np.eye(2) * KALMAN_PROCESS_VAR
    _R = np.array([[KALMAN_MEASURE_VAR]])
    Q_FLOOR = _Q_fixed * 0.1
    Q_CEIL = _Q_fixed * Q_max_factor

    n = len(indicator_live)
    out = np.full((n, 2), np.nan)

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

    x_filt_cl = np.zeros((nc, 2))
    P_filt_cl = np.zeros((nc, 2, 2))
    Q_current = _Q_fixed.copy()
    innovation_buffer = []

    for k in range(nc):
        if k == 0:
            x_p = np.array([cv[0], 0.0])
            P_p = np.eye(2)
        else:
            x_p = A @ x_filt_cl[k - 1]
            P_p = A @ P_filt_cl[k - 1] @ A.T + Q_current

        y = cv[k] - H @ x_p
        S = (H @ P_p @ H.T + _R)[0, 0]
        K = P_p @ H.T / S
        x_filt_cl[k] = x_p + (K @ y).ravel()
        P_filt_cl[k] = (np.eye(2) - K @ H) @ P_p

        v_t = cv[k] - (H @ x_p)[0]
        innovation_buffer.append(v_t)
        if len(innovation_buffer) > aq_window:
            innovation_buffer.pop(0)

        if len(innovation_buffer) >= aq_window and k > 0:
            C_vv = np.mean(np.array(innovation_buffer) ** 2)
            delta = C_vv - S
            if delta > 0:
                P_pred_next = A @ P_filt_cl[k] @ A.T + Q_current
                C_rts = P_filt_cl[k] @ A.T @ inv2x2(P_pred_next)
                Q_candidate = delta * (C_rts @ C_rts.T)
                if is_pos_semidef(Q_candidate):
                    Q_current = np.clip(Q_candidate, Q_FLOOR, Q_CEIL)

    for k, ci in enumerate(closure_indices):
        out[ci, 0] = x_filt_cl[k, 0]
        out[ci, 1] = x_filt_cl[k, 1]

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
        if current_k >= 0:
            x_p = A @ sm_cl
            P_p = A @ sc_cl @ A.T + Q_current
            y_val = obs - (H @ x_p)[0]
            S_val = (H @ P_p @ H.T + _R)[0, 0]
            K_val = P_p @ H.T / S_val
            sm_p = x_p + (K_val * y_val).ravel()
            out[i, 0] = sm_p[0]
            out[i, 1] = sm_p[1]

    return out


def main():
    parser = argparse.ArgumentParser(
        description='Prepare FLKS features CSV for LSTM training')
    parser.add_argument('--csv', type=str, default='data_trad/BTCUSD_all_5m.csv')
    parser.add_argument('--n-candles-30m', type=int, default=5000)
    parser.add_argument('--output-dir', type=str, default='data/prepared')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    print(f"[1/8] Loading {args.csv} ...")
    df_5m = load_csv(args.csv)
    print(f"       {len(df_5m):,} 5min candles")

    # ==================================================================
    print("[2/8] Resampling to 30min ...")
    df_30m = resample_ohlcv(df_5m, 30)
    if args.n_candles_30m > 0 and len(df_30m) > args.n_candles_30m:
        df_30m = df_30m.iloc[-args.n_candles_30m:]
    df_5m = df_5m.loc[df_30m.index[0]:df_30m.index[-1] + pd.Timedelta(minutes=29)]
    n30 = len(df_30m)
    print(f"       {n30:,} bougies 30min, {len(df_5m):,} bougies 5min")

    # ==================================================================
    print("[3/8] Computing MACD 30min + live 5min ...")
    macd_30m = calculate_macd(df_30m)
    is_close = compute_bucket_close_mask(df_5m.index, 30)
    close_5m = df_5m['close'].values.astype(np.float64)
    macd_live = compute_macd_live(close_5m, is_close)
    macd_live_pc = group_per_candle(df_5m, df_30m, macd_live)
    print(f"       MACD range: [{np.nanmin(macd_30m):.1f}, {np.nanmax(macd_30m):.1f}]")

    # ==================================================================
    print("[4/8] AQ-KF live features (5min resolution) ...")
    kalman_out = compute_kalman_live_aqkf(macd_live, is_close)
    aq_filtered = kalman_out[:, 0]
    aq_velocity = kalman_out[:, 1]
    n_valid = np.sum(~np.isnan(aq_filtered))
    print(f"       {n_valid:,} valid AQ-KF values")

    # ==================================================================
    print("[5/8] Oracle labels (pykalman.smooth on 30min) ...")
    _, slopes_oracle = compute_oracle(macd_30m)
    oracle_labels = np.where(slopes_oracle > 0, 1, 0)
    oracle_labels_30m = pd.Series(oracle_labels, index=df_30m.index)
    oracle_labels_5m = oracle_labels_30m.reindex(df_5m.index, method='ffill').fillna(0).astype(int)
    oracle_slopes_30m = pd.Series(slopes_oracle, index=df_30m.index)
    oracle_slopes_5m = oracle_slopes_30m.reindex(df_5m.index, method='ffill')
    print(f"       Labels: {(oracle_labels_5m == 1).sum():,} UP, {(oracle_labels_5m == 0).sum():,} DOWN")

    # ==================================================================
    print("[6/8] FLKS backward slopes (AQ-KF, T1 + k=1..6) ...")
    x_aq, P_aq, xp_aq, Pp_aq, C_aq = forward_filter_30m_adaptive(
        macd_30m, window=30, Q_max_factor=10.0)

    # T1 slope
    slopes_t1 = compute_slopes_test1(x_aq, xp_aq, C_aq)
    t1_30m = pd.Series(slopes_t1, index=df_30m.index)
    t1_5m = t1_30m.reindex(df_5m.index, method='ffill')

    # k=1..6 slopes
    slopes_k = {}
    for k in range(1, 7):
        sk = compute_slopes_test2(x_aq, P_aq, xp_aq, C_aq, macd_live_pc, k)
        sk_30m = pd.Series(sk, index=df_30m.index)
        slopes_k[k] = sk_30m.reindex(df_5m.index, method='ffill')

    print("       Done.")

    # ==================================================================
    print("[7/8] Building CSV ...")
    result = pd.DataFrame(index=df_5m.index)
    result['close'] = df_5m['close'].values
    result['macd_30m_live'] = macd_live
    result['macd_30m_filtered'] = aq_filtered
    result['macd_30m_velocity'] = aq_velocity
    result['aq_t1_slope'] = t1_5m.values
    for k in range(1, 7):
        result[f'aq_k{k}_slope'] = slopes_k[k].values
    result['oracle_label_macd_30m'] = oracle_labels_5m.values
    result['oracle_slope_macd_30m'] = oracle_slopes_5m.values

    out_path = output_dir / 'BTCUSD_flks_features.csv'
    result.to_csv(out_path)
    n_rows = len(result)
    n_cols = len(result.columns)
    print(f"       Saved: {out_path} ({n_rows:,} rows × {n_cols} columns)")

    # ==================================================================
    print(f"\n[8/8] Concordance verification from CSV ...")

    # Read back and verify
    df_check = result.dropna()
    n_check = len(df_check)

    # Compare at 30min closures only (where slopes are computed)
    df_closures = df_check.loc[df_30m.index].dropna()
    n_cl = len(df_closures)

    eval_start = TRIM
    eval_end = n_cl - TRIM
    n_eval = eval_end - eval_start

    oracle_sl = df_closures['oracle_slope_macd_30m'].values
    trans_mask = find_oracle_transitions(oracle_sl, eval_start, eval_end)
    n_trans = trans_mask.sum()

    s_o = oracle_sl[eval_start:eval_end]
    sign_o = np.where(np.abs(s_o) < 1e-8, 0, np.sign(s_o))
    valid_signs = sign_o[sign_o != 0]
    persistence = np.mean(valid_signs[1:] == valid_signs[:-1]) * 100.0

    print(f"\n{'=' * 70}")
    print(f"  CONCORDANCE VERIFICATION (from saved CSV, closures only)")
    print(f"  Closures: {n_cl:,} | Eval [{eval_start}:{eval_end}] = {n_eval:,}")
    print(f"  Transitions: {n_trans:,} | Persistence: {persistence:.1f}%")
    print(f"{'=' * 70}")

    print(f"\n  {'Méthode':<20} {'All':>9} {'Trans':>10}")
    print(f"  {'-' * 41}")

    for col_name, label in [('aq_t1_slope', 'AQ T1 (0 pas)'),
                             ('aq_k1_slope', 'AQ k=1 (5min)'),
                             ('aq_k2_slope', 'AQ k=2 (10min)'),
                             ('aq_k3_slope', 'AQ k=3 (15min)'),
                             ('aq_k4_slope', 'AQ k=4 (20min)'),
                             ('aq_k5_slope', 'AQ k=5 (25min)'),
                             ('aq_k6_slope', 'AQ k=6 (30min)')]:
        slopes_test = df_closures[col_name].values
        c_all, _ = sign_concordance(slopes_test, oracle_sl, eval_start, eval_end)
        c_tr, _ = sign_concordance_at_transitions(
            slopes_test, oracle_sl, eval_start, eval_end, trans_mask)
        print(f"  {label:<20} {c_all:>8.2f}% {c_tr:>9.2f}%")

    print(f"  {'-' * 41}")
    print(f"{'=' * 70}")
    print("Done.")


if __name__ == '__main__':
    main()
