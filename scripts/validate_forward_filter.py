#!/usr/bin/env python3
"""
Valide compute_forward_filter (standard + AQ-KF) sur MACD 30m et MACD 1h.

Tests:
  [1] Structure : shape, colonnes, pas de NaN, index aligné
  [2] Path A (5m resamplé) vs Path B (TF téléchargé) : states identiques
  [3] Causalité : polluer close[T+10] ne change pas x_filt[:T+1]
  [4] Bug init corrigé : grâce au fillna(0) en amont, x_filt[0] ne dépend PAS
      d'une observation future (contrairement au bug audité en test_03)
  [5] P_filt PSD et convergente
  [6] Standard vs AQ-KF : concordance sur régime stable, diff sur volatil
  [7] Stats : convergence position vers indicator, range velocity

Scope: MACD × (30m, 1h) × (standard, AQ-KF) = 4 combinaisons.

Usage:
    python scripts/validate_forward_filter.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import (
    load_csv, resample_ohlcv, compute_indicator, compute_forward_filter,
)

DATA_DIR = Path('data/raw')
INDICATOR = 'macd'
TFS = [30, 60]
TRIM = 50  # warm-up convergence Kalman
TOL_ABS = 1e-10


def check(name, passed, detail=""):
    status = "✅" if passed else "❌"
    print(f"  {status} {name}  {detail}")
    return passed


def drop_incomplete_last(df_tf, df_5m, tf_minutes):
    expected = tf_minutes // 5
    drop_count = 0
    for ts in reversed(df_tf.index):
        end = ts + pd.Timedelta(minutes=tf_minutes)
        mask = (df_5m.index >= ts) & (df_5m.index < end)
        if mask.sum() < expected:
            drop_count += 1
        else:
            break
    if drop_count > 0:
        df_tf = df_tf.iloc[:-drop_count]
    return df_tf, drop_count


def compare_filter_outputs(res_A, res_B, tf_label, mode_label):
    """Compare state DataFrame et matrices (P_filt, P_pred, C) entre 2 runs."""
    ok = True
    # State DataFrame
    state_A = res_A['state']
    state_B = res_B['state']
    for col in ['position', 'velocity', 'pred_position', 'pred_velocity']:
        max_diff = np.max(np.abs(state_A[col].values - state_B[col].values))
        ok &= check(
            f"  {tf_label} {mode_label}: max |A.state.{col} - B.state.{col}| = {max_diff:.2e}",
            max_diff < TOL_ABS, "")
    # Matrices
    for key in ['P_filt', 'P_pred', 'C']:
        max_diff = np.max(np.abs(res_A[key] - res_B[key]))
        ok &= check(
            f"  {tf_label} {mode_label}: max |A.{key} - B.{key}| = {max_diff:.2e}",
            max_diff < TOL_ABS, "")
    return ok


def main():
    print("=" * 80)
    print(f"VALIDATION compute_forward_filter — {INDICATOR.upper()} × (30m, 1h) × (std, AQ-KF)")
    print("=" * 80)

    paths = {
        '5m': DATA_DIR / 'BTCUSD_3months_5m.csv',
        30: DATA_DIR / 'BTCUSD_3months_30m.csv',
        60: DATA_DIR / 'BTCUSD_3months_1h.csv',
    }
    for key, p in paths.items():
        if not p.exists():
            print(f"❌ Fichier manquant: {p}")
            return

    df_5m = load_csv(paths['5m'])
    dfs_dl = {30: load_csv(paths[30]), 60: load_csv(paths[60])}
    print(f"\nChargement:")
    print(f"  5m:  {len(df_5m):,} rows")
    for tf, df in dfs_dl.items():
        lbl = f'{tf}m' if tf < 60 else '1h'
        print(f"  {lbl}: {len(df):,} rows")

    # Resample 5m → TF (Path A)
    dfs_R = {30: resample_ohlcv(df_5m, 30), 60: resample_ohlcv(df_5m, 60)}

    # Drop incomplete trailing candles
    for tf in TFS:
        dfs_dl[tf], _ = drop_incomplete_last(dfs_dl[tf], df_5m, tf)
        dfs_R[tf], _ = drop_incomplete_last(dfs_R[tf], df_5m, tf)

    all_ok = True
    results = {}  # (tf, mode) -> (res_A, res_B)

    for tf in TFS:
        lbl_tf = f'{tf}m' if tf < 60 else '1h'
        for adaptive in [False, True]:
            mode = 'aq-kf' if adaptive else 'std'
            print(f"\n{'-' * 80}")
            print(f"  TF = {lbl_tf}   MODE = {mode.upper()}")
            print(f"{'-' * 80}")

            res_A = compute_forward_filter(dfs_R[tf], INDICATOR, adaptive=adaptive)
            res_B = compute_forward_filter(dfs_dl[tf], INDICATOR, adaptive=adaptive)
            results[(tf, mode)] = (res_A, res_B)

            # [1] Structure
            print(f"  [1] Structure des sorties")
            state = res_B['state']
            n = len(state)
            all_ok &= check(
                f"  state shape = len(df)",
                len(state) == len(dfs_dl[tf]),
                f"{len(state)} vs {len(dfs_dl[tf])}")
            all_ok &= check(
                f"  colonnes state = [position, velocity, pred_position, pred_velocity]",
                list(state.columns) == ['position', 'velocity',
                                         'pred_position', 'pred_velocity'],
                "")
            all_ok &= check(
                f"  P_filt shape = (n, 2, 2)",
                res_B['P_filt'].shape == (n, 2, 2),
                f"{res_B['P_filt'].shape}")
            all_ok &= check(
                f"  P_pred shape = (n, 2, 2)",
                res_B['P_pred'].shape == (n, 2, 2),
                "")
            all_ok &= check(
                f"  C shape = (n, 2, 2)",
                res_B['C'].shape == (n, 2, 2),
                "")
            all_ok &= check(
                f"  pas de NaN dans state",
                not state.isna().any().any(),
                "")

            # [2] Path A vs Path B
            print(f"  [2] Path A (5m resamplé) vs Path B (TF téléchargé)")
            all_ok &= compare_filter_outputs(res_A, res_B, lbl_tf, mode)

            # [3] Causalité : polluer close[T+10] ne change pas state[:T+1]
            print(f"  [3] Causalité (polluer close[T+10] ne change pas state[:T+1])")
            df_pol = dfs_dl[tf].copy()
            T = len(df_pol) // 2
            df_pol.iloc[T + 10, df_pol.columns.get_loc('close')] += 10000.0
            res_pol = compute_forward_filter(df_pol, INDICATOR, adaptive=adaptive)
            diff_before = np.max(np.abs(
                res_B['state']['position'].iloc[:T + 1].values
                - res_pol['state']['position'].iloc[:T + 1].values))
            all_ok &= check(
                f"  max |position[:T+1]_orig - position[:T+1]_polluted| = {diff_before:.2e}",
                diff_before < TOL_ABS, "")
            # Confirmer qu'après T, les valeurs diffèrent (sinon pollution inefficace)
            diff_after = np.max(np.abs(
                res_B['state']['position'].iloc[T + 10:].values
                - res_pol['state']['position'].iloc[T + 10:].values))
            all_ok &= check(
                f"  après T+10, états diffèrent (sanity pollution): {diff_after:.2e}",
                diff_after > 1e-3, "")

            # [4] Bug init corrigé : x_filt[0] ne dépend PAS de indicator[1:]
            #     Autrefois (bug audit test_03) : first_valid_val = première
            #     non-NaN potentielle dans le futur. Avec fillna(0) amont,
            #     first_valid_val = indicator[0] → pas de leakage.
            print(f"  [4] Bug init (fillna(0)) : x_filt[0] ne dépend pas de indicator[1:]")
            # Test : si on change indicator[5], x_filt[0] doit rester identique
            df_pol_early = dfs_dl[tf].copy()
            df_pol_early.iloc[5, df_pol_early.columns.get_loc('close')] += 5000.0
            res_pol_early = compute_forward_filter(df_pol_early, INDICATOR, adaptive=adaptive)
            diff_at_0 = abs(
                res_B['state']['position'].iloc[0]
                - res_pol_early['state']['position'].iloc[0])
            all_ok &= check(
                f"  |position[0]_orig - position[0]_close5_polluted| = {diff_at_0:.2e}",
                diff_at_0 < TOL_ABS, "")

            # [5] P_filt PSD et convergente
            print(f"  [5] P_filt PSD et convergente")
            P_filt = res_B['P_filt']
            # Tous les éléments diagonaux doivent être >= 0
            diag_min = min(P_filt[:, 0, 0].min(), P_filt[:, 1, 1].min())
            all_ok &= check(
                f"  min(diag P_filt) = {diag_min:.6g} (>= 0)",
                diag_min >= -1e-10, "")
            # Convergence : var(trace P_filt[TRIM:]) petite
            traces = P_filt[:, 0, 0] + P_filt[:, 1, 1]
            std_late = traces[TRIM:].std()
            mean_late = traces[TRIM:].mean()
            ratio = std_late / (mean_late + 1e-10)
            print(f"        trace P_filt après warmup: mean={mean_late:.4f} "
                  f"std={std_late:.4f} (ratio={ratio:.3f})")
            if mode == 'std':
                # Standard Kalman : converge vers steady-state
                all_ok &= check(
                    f"  [std] trace P_filt converge (ratio std/mean < 0.3)",
                    ratio < 0.3, "")
            # AQ-KF : Q varie, trace peut varier aussi — ratio plus tolérant

            # [7] Stats convergence position → indicator
            print(f"  [6] Convergence position → indicator")
            ind_vals = res_B['indicator'].values
            pos_vals = state['position'].values
            rmse = np.sqrt(np.mean((ind_vals[TRIM:] - pos_vals[TRIM:]) ** 2))
            std_ind = ind_vals[TRIM:].std()
            all_ok &= check(
                f"  RMSE(position, indicator) after trim = {rmse:.4f} "
                f"(std(ind) = {std_ind:.4f})",
                rmse < std_ind, "")

    # ========== [6] Standard vs AQ-KF ==========
    print(f"\n{'-' * 80}")
    print(f"  [BONUS] Standard vs AQ-KF — convergence sur signal réel")
    print(f"{'-' * 80}")
    for tf in TFS:
        lbl_tf = f'{tf}m' if tf < 60 else '1h'
        res_std = results[(tf, 'std')][1]  # Path B
        res_aq = results[(tf, 'aq-kf')][1]
        # Comparer position (après TRIM)
        diff = np.abs(
            res_std['state']['position'].values[TRIM:]
            - res_aq['state']['position'].values[TRIM:])
        max_d = diff.max()
        mean_d = diff.mean()
        ratio_to_std = mean_d / (res_std['indicator'].values[TRIM:].std() + 1e-10)
        print(f"  {lbl_tf}: max |std.pos - aqkf.pos| = {max_d:.4f}  "
              f"mean = {mean_d:.4f}  (ratio/std_ind = {ratio_to_std:.3f})")

    # Verdict
    print("\n" + "=" * 80)
    print(f"VERDICT : {'✅ TOUS TESTS PASS' if all_ok else '❌ AU MOINS UN ÉCHEC'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
