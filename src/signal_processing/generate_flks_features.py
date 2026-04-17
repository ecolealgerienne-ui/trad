#!/usr/bin/env python3
"""
Generate FLKS backward slopes as features + concordance table
==============================================================

1. Load BTC 5min CSV, resample 30min
2. Compute MACD 30min + MACD live 5min
3. Run Standard + AQ-KF forward filters on 30min
4. Compute FLKS backward slopes (T1 + T2 k=1..6) for both
5. Compute oracle (pykalman.smooth)
6. Output concordance table (Standard vs AQ-KF, All vs Trans)
7. Save slopes as CSV for ML training

Trimming: skip 100 at start and end of eval window.

Usage:
    python src/signal_processing/generate_flks_features.py \
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
)


TRIM = 100  # skip 100 at start and end


def main():
    parser = argparse.ArgumentParser(
        description='Generate FLKS features + concordance table')
    parser.add_argument('--csv', type=str, default='data_trad/BTCUSD_all_5m.csv')
    parser.add_argument('--n-candles-30m', type=int, default=5000)
    parser.add_argument('--output-dir', type=str, default='data/prepared')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    print(f"[1/7] Loading {args.csv} ...")
    df_5m = load_csv(args.csv)
    print(f"       {len(df_5m):,} 5min candles")

    # ==================================================================
    print("[2/7] Resampling to 30min ...")
    df_30m = resample_ohlcv(df_5m, 30)
    n_requested = args.n_candles_30m
    if n_requested > 0 and len(df_30m) > n_requested:
        df_30m = df_30m.iloc[-n_requested:]
    df_5m = df_5m.loc[df_30m.index[0]:df_30m.index[-1] + pd.Timedelta(minutes=29)]
    n30 = len(df_30m)
    print(f"       {n30:,} bougies 30min, {len(df_5m):,} bougies 5min")

    # Eval window with trim
    eval_start = TRIM
    eval_end = n30 - TRIM
    n_eval = eval_end - eval_start
    print(f"       Eval: [{eval_start}:{eval_end}] = {n_eval:,} (trim {TRIM} each side)")

    # ==================================================================
    print("[3/7] Computing MACD 30min + live 5min ...")
    macd_30m = calculate_macd(df_30m)
    is_close = compute_bucket_close_mask(df_5m.index, 30)
    close_5m = df_5m['close'].values.astype(np.float64)
    macd_live = compute_macd_live(close_5m, is_close)
    macd_live_pc = group_per_candle(df_5m, df_30m, macd_live)

    # Coherence check
    max_err = 0.0
    n_checked = 0
    for t in range(n30):
        vals = [v for v in macd_live_pc[t] if not np.isnan(v)]
        if len(vals) > 0 and not np.isnan(macd_30m[t]):
            max_err = max(max_err, abs(vals[-1] - macd_30m[t]))
            n_checked += 1
    print(f"       Coherence: max err = {max_err:.2e} ({n_checked} candles)")

    # ==================================================================
    print("[4/7] Oracle (pykalman.smooth on 30min) ...")
    _, slopes_oracle = compute_oracle(macd_30m)
    trans_mask = find_oracle_transitions(slopes_oracle, eval_start, eval_end)
    n_trans = trans_mask.sum()

    # Persistence
    s_o = slopes_oracle[eval_start:eval_end]
    sign_o = np.where(np.abs(s_o) < 1e-8, 0, np.sign(s_o))
    valid_signs = sign_o[sign_o != 0]
    persistence = np.mean(valid_signs[1:] == valid_signs[:-1]) * 100.0

    print(f"       Transitions: {n_trans:,} ({n_trans/n_eval*100:.1f}%)")
    print(f"       Persistence: {persistence:.1f}%")

    # ==================================================================
    print("[5/7] Standard forward filter + slopes ...")
    x_std, P_std, xp_std, Pp_std, C_std = forward_filter_30m(macd_30m)

    slopes_std = {}
    slopes_std['t1'] = compute_slopes_test1(x_std, xp_std, C_std)
    for k in range(1, 7):
        slopes_std[f'k{k}'] = compute_slopes_test2(
            x_std, P_std, xp_std, C_std, macd_live_pc, k)
    print("       Done.")

    # ==================================================================
    print("[6/7] AQ-KF forward filter + slopes ...")
    x_aq, P_aq, xp_aq, Pp_aq, C_aq = forward_filter_30m_adaptive(
        macd_30m, window=30, Q_max_factor=10.0)

    slopes_aq = {}
    slopes_aq['t1'] = compute_slopes_test1(x_aq, xp_aq, C_aq)
    for k in range(1, 7):
        slopes_aq[f'k{k}'] = compute_slopes_test2(
            x_aq, P_aq, xp_aq, C_aq, macd_live_pc, k)
    print("       Done.")

    # ==================================================================
    print(f"\n[7/7] Concordance table")
    print(f"{'=' * 80}")
    print(f"  CONCORDANCE — MACD 30min BTC — [{eval_start}:{eval_end}] = {n_eval:,} bougies")
    print(f"  Transitions: {n_trans:,} ({n_trans/n_eval*100:.1f}%) | Persistence: {persistence:.1f}%")
    print(f"  Trim: {TRIM} each side")
    print(f"{'=' * 80}")

    print(f"\n  {'Méthode':<20} {'Std All':>9} {'Std Trans':>10} "
          f"{'AQ All':>9} {'AQ Trans':>10}")
    print(f"  {'-' * 60}")

    # T1
    std_all, _ = sign_concordance(slopes_std['t1'], slopes_oracle, eval_start, eval_end)
    std_tr, _ = sign_concordance_at_transitions(
        slopes_std['t1'], slopes_oracle, eval_start, eval_end, trans_mask)
    aq_all, _ = sign_concordance(slopes_aq['t1'], slopes_oracle, eval_start, eval_end)
    aq_tr, _ = sign_concordance_at_transitions(
        slopes_aq['t1'], slopes_oracle, eval_start, eval_end, trans_mask)
    print(f"  {'T1 (0 pas)':<20} {std_all:>8.2f}% {std_tr:>9.2f}% "
          f"{aq_all:>8.2f}% {aq_tr:>9.2f}%")

    # k=1..6
    for k in range(1, 7):
        key = f'k{k}'
        std_all, _ = sign_concordance(slopes_std[key], slopes_oracle, eval_start, eval_end)
        std_tr, _ = sign_concordance_at_transitions(
            slopes_std[key], slopes_oracle, eval_start, eval_end, trans_mask)
        aq_all, _ = sign_concordance(slopes_aq[key], slopes_oracle, eval_start, eval_end)
        aq_tr, _ = sign_concordance_at_transitions(
            slopes_aq[key], slopes_oracle, eval_start, eval_end, trans_mask)
        print(f"  {'k=' + str(k) + ' (' + str(k*5) + 'min)':<20} {std_all:>8.2f}% {std_tr:>9.2f}% "
              f"{aq_all:>8.2f}% {aq_tr:>9.2f}%")

    print(f"  {'-' * 60}")
    print(f"{'=' * 80}")

    # Save slopes as CSV
    out_df = pd.DataFrame(index=df_30m.index)
    out_df['macd_30m'] = macd_30m
    out_df['oracle_slope'] = slopes_oracle
    out_df['oracle_label'] = (slopes_oracle > 0).astype(int)
    out_df['std_t1_slope'] = slopes_std['t1']
    out_df['aq_t1_slope'] = slopes_aq['t1']
    for k in range(1, 7):
        out_df[f'std_k{k}_slope'] = slopes_std[f'k{k}']
        out_df[f'aq_k{k}_slope'] = slopes_aq[f'k{k}']

    out_path = output_dir / 'flks_slopes_macd_30m.csv'
    out_df.to_csv(out_path)
    print(f"\n  Slopes CSV saved: {out_path} ({len(out_df):,} rows)")
    print("Done.")


if __name__ == '__main__':
    main()
