#!/usr/bin/env python3
"""
Valide compute_flks_slopes (FLKS-2 = backward slopes) sur MACD 30m et 1h.

FLKS-2 = backward slopes (rétrospectifs):
  - slope_t1 : backward 2 pas sans sous-pas
  - slope_k1..k6 : backward 3 pas avec k updates Kalman depuis le 5min live

Tests:
  [1] Structure : DataFrame indexé, 7 colonnes, pas de NaN
  [2] Path A vs Path B : slopes identiques entre 5m-resamplé et TF téléchargé
  [3] Causalité slope_k6 : polluer close[T + (k+1)*5min] ne doit PAS changer
      slope_kN[T] pour N ≤ k. (slope_k6[T] utilise jusqu'à close[T+1][5])
  [4] Stats : distribution, range, corrélation t1 vs k6
  [5] Concordance avec oracle : sign match slope_k6 vs oracle_slope aux
      closes TF (métrique FLKS clé, ~95% attendu)

Scope: MACD × (30m, 1h) × (slope_t1 + slope_k1..k6) = 14 slopes.

Usage:
    python scripts/validate_flks_slopes.py --indicator macd
    python scripts/validate_flks_slopes.py --indicator rsi
    python scripts/validate_flks_slopes.py --indicator cci
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import (
    load_csv, resample_ohlcv,
    compute_flks_slopes, compute_oracle_labels,
)

DATA_DIR = Path('data/raw')
TFS = [30, 60]
TRIM = 50
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indicator', default='macd',
                        choices=['macd', 'rsi', 'cci'])
    args = parser.parse_args()
    INDICATOR = args.indicator

    print("=" * 80)
    print(f"VALIDATION compute_flks_slopes — {INDICATOR.upper()} × (30m, 1h)")
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

    for tf in TFS:
        lbl_tf = f'{tf}m' if tf < 60 else '1h'
        print(f"\n{'-' * 80}")
        print(f"  TF = {lbl_tf}")
        print(f"{'-' * 80}")

        # Calculer slopes via les deux chemins
        slopes_A = compute_flks_slopes(dfs_R[tf], df_5m, INDICATOR, tf)
        slopes_B = compute_flks_slopes(dfs_dl[tf], df_5m, INDICATOR, tf)

        # [1] Structure
        print(f"  [1] Structure des sorties")
        expected_cols = ['slope_t1'] + [f'slope_k{k}' for k in range(1, 7)]
        all_ok &= check(
            f"  shape = len(df_tf)",
            len(slopes_B) == len(dfs_dl[tf]),
            f"{len(slopes_B)} vs {len(dfs_dl[tf])}")
        all_ok &= check(
            f"  colonnes = {expected_cols}",
            list(slopes_B.columns) == expected_cols,
            f"got {list(slopes_B.columns)}")
        all_ok &= check(
            f"  pas de NaN (fillna 0 effectif)",
            not slopes_B.isna().any().any(),
            "")
        all_ok &= check(
            f"  index identique à df_tf",
            slopes_B.index.equals(dfs_dl[tf].index),
            "")

        # [2] Path A vs Path B
        print(f"  [2] Path A (5m resamplé) vs Path B (TF téléchargé)")
        all_ok &= check(
            f"  shapes identiques",
            slopes_A.shape == slopes_B.shape,
            f"A={slopes_A.shape} B={slopes_B.shape}")
        for col in expected_cols:
            max_diff = np.max(np.abs(slopes_A[col].values - slopes_B[col].values))
            all_ok &= check(
                f"  max |A.{col} - B.{col}| = {max_diff:.2e}",
                max_diff < TOL_ABS, "")

        # [3] Causalité slope_k6
        # slope_k6[T] utilise les 6 premières 5min de bougie T+1 (i.e. TF complète)
        # → polluer 5m dans la bougie T+2 ne doit pas changer slope_k6[T]
        print(f"  [3] Causalité : polluer 5m dans bougie TF T+2 ne change pas slope_k6[T]")
        df_5m_pol = df_5m.copy()
        T = len(dfs_dl[tf]) // 2
        ts_T_plus_2 = dfs_dl[tf].index[T + 2]
        bucket_end = ts_T_plus_2 + pd.Timedelta(minutes=tf)
        mask_pol = (df_5m_pol.index >= ts_T_plus_2) & (df_5m_pol.index < bucket_end)
        df_5m_pol.loc[mask_pol, 'close'] += 10000.0
        slopes_pol = compute_flks_slopes(dfs_dl[tf], df_5m_pol, INDICATOR, tf)
        diff_k6_at_T = abs(slopes_B['slope_k6'].iloc[T] - slopes_pol['slope_k6'].iloc[T])
        all_ok &= check(
            f"  |slope_k6[T]_orig - slope_k6[T]_pollutedT+2| = {diff_k6_at_T:.2e}",
            diff_k6_at_T < TOL_ABS, "")

        # Sanity: pollution doit avoir un effet sur slope_k6[T+2] ou T+3
        diff_k6_at_T2 = abs(
            slopes_B['slope_k6'].iloc[T + 3] - slopes_pol['slope_k6'].iloc[T + 3])
        all_ok &= check(
            f"  sanity: slope_k6[T+3] différent: {diff_k6_at_T2:.2e}",
            diff_k6_at_T2 > 1e-6, "")

        # [4] Stats
        print(f"  [4] Statistiques (après TRIM={TRIM})")
        trimmed = slopes_B.iloc[TRIM:-TRIM]
        for col in ['slope_t1', 'slope_k6']:
            s = trimmed[col]
            n_trans = int((np.sign(s).diff().abs() > 0).sum())
            print(f"    {col}: mean={s.mean():+.4f}  std={s.std():.4f}  "
                  f"min={s.min():+.4f}  max={s.max():+.4f}  "
                  f"transitions={n_trans}")
        corr = trimmed['slope_t1'].corr(trimmed['slope_k6'])
        print(f"    Corrélation slope_t1 vs slope_k6: {corr:.4f}")
        all_ok &= check(
            f"  corrélation slope_t1 vs slope_k6 > 0.5",
            corr > 0.5, f"got {corr:.3f}")

        # [5] Concordance avec oracle (métrique FLKS clé)
        print(f"  [5] Concordance signe slope_k6 vs oracle_slope (aux closes TF)")
        oracle = compute_oracle_labels(dfs_dl[tf], INDICATOR)
        # Les deux ont le même index (df_tf)
        # Prendre uniquement l'intersection après TRIM
        common_idx = slopes_B.index.intersection(oracle.index)
        slopes_trim = slopes_B.loc[common_idx].iloc[TRIM:-TRIM]
        oracle_trim = oracle.loc[common_idx].iloc[TRIM:-TRIM]
        # Ignorer les positions où oracle.slope = 0 (zone d'incertitude)
        sign_model = np.sign(slopes_trim['slope_k6'].values)
        sign_oracle = np.sign(oracle_trim['slope'].values)
        mask = (sign_oracle != 0) & (sign_model != 0)
        if mask.sum() > 0:
            concord = (sign_model[mask] == sign_oracle[mask]).mean() * 100
            print(f"    Concordance slope_k6 vs oracle_slope: {concord:.2f}% "
                  f"(sur {mask.sum():,} samples)")
            all_ok &= check(
                f"  concordance > 80%",
                concord > 80, f"got {concord:.2f}%")
        # Pour info, slope_t1 aussi
        sign_t1 = np.sign(slopes_trim['slope_t1'].values)
        mask_t1 = (sign_oracle != 0) & (sign_t1 != 0)
        if mask_t1.sum() > 0:
            concord_t1 = (sign_t1[mask_t1] == sign_oracle[mask_t1]).mean() * 100
            print(f"    Concordance slope_t1 vs oracle_slope: {concord_t1:.2f}%")

    # Verdict
    print("\n" + "=" * 80)
    print(f"VERDICT : {'✅ TOUS TESTS PASS' if all_ok else '❌ AU MOINS UN ÉCHEC'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
