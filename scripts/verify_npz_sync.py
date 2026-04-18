#!/usr/bin/env python3
"""
Vérifie la synchronisation des données dans le NPZ dataset préparé.

5 checks + 1 bonus:
  [1] y_test_binary ↔ oracle_slopes_full[test_indices]
  [2] test_dates    ↔ df_tf_dates[test_indices]
  [3] closes_test   ↔ df_tf_closes[test_indices]
  [4] slope[t] = positions[t-1] - positions[t-2] (formule math oracle)
  [5] Reproductibilité : recalculer l'oracle depuis le CSV et comparer
  [Bonus] X_test reverse-denormalize vs slopes reconstruites

Objectif: détecter toute désynchronisation AVANT de suspecter le backtest.

Usage:
    python scripts/verify_npz_sync.py
    python scripts/verify_npz_sync.py --indicator macd --tf 30 --source full
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
    compute_oracle_labels, compute_flks_slopes,
)

PREP_DIR = Path('data/prepared')
RAW_DIR = Path('data/raw')

SOURCE_PATHS = {
    '3months': {
        5: RAW_DIR / 'BTCUSD_3months_5m.csv',
        30: RAW_DIR / 'BTCUSD_3months_30m.csv',
    },
    'full': {
        5: Path('data_trad/BTCUSD_all_5m.csv'),
        30: RAW_DIR / 'BTCUSD_full_30m.csv',
    },
}


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
    parser.add_argument('--tf', type=int, default=30, choices=[30, 60])
    parser.add_argument('--source', default='full',
                        choices=['3months', 'full'])
    args = parser.parse_args()

    tf_label = f'{args.tf}m' if args.tf < 60 else '1h'
    print("=" * 80)
    print(f"VÉRIFICATION SYNCHRONISATION NPZ — {args.indicator.upper()} × "
          f"{tf_label} × {args.source}")
    print("=" * 80)

    # Charger le NPZ dataset
    npz_path = PREP_DIR / f'dataset_{args.indicator}_{tf_label}_{args.source}.npz'
    if not npz_path.exists():
        print(f"❌ NPZ non trouvé: {npz_path}")
        return
    print(f"\n✅ NPZ chargé: {npz_path}")
    ds = np.load(npz_path, allow_pickle=True)

    # Extraire les arrays (noms tels que sauvés par prepare_full_data.py)
    y_test_binary = ds['y_test_binary']
    y_test_continuous = ds['y_test_continuous']
    test_indices = ds['indices_test']
    test_dates = pd.to_datetime(ds['dates_test'])
    closes_test = ds['closes_test']
    X_test = ds['X_test']
    oracle_slopes_full = ds['oracle_slopes_full']
    df_tf_dates = pd.to_datetime(ds['df_tf_dates'])
    df_tf_closes = ds['df_tf_closes']
    feature_cols = [str(c) for c in ds['feature_cols']]
    norm_means = ds['norm_means']
    norm_stds = ds['norm_stds']
    window = int(ds['window'])

    n_test = len(test_indices)
    n_tf = len(df_tf_dates)
    print(f"   n_test = {n_test:,}  |  n_tf = {n_tf:,}  |  window = {window}")

    all_ok = True

    # ========================================================================
    # [1] y_test_binary ↔ oracle_slopes_full[test_indices]
    # ========================================================================
    print(f"\n[1] y_test_binary ↔ oracle_slopes_full[test_indices]")
    oracle_slopes_at_test = oracle_slopes_full[test_indices]
    y_expected = (oracle_slopes_at_test > 0).astype(int)
    mismatch_1 = (y_test_binary != y_expected).sum()
    all_ok &= check(
        f"  y_test_binary == (oracle_slopes_full[indices] > 0)",
        mismatch_1 == 0,
        f"{mismatch_1}/{n_test} mismatches")
    if mismatch_1 > 0:
        bad = np.where(y_test_binary != y_expected)[0][:5]
        for i in bad:
            print(f"    i={i}  idx_tf={test_indices[i]}  "
                  f"y_test={y_test_binary[i]}  "
                  f"oracle_slope={oracle_slopes_at_test[i]:.6f}  "
                  f"expected={y_expected[i]}")

    # ========================================================================
    # [2] test_dates ↔ df_tf_dates[test_indices]
    # ========================================================================
    print(f"\n[2] test_dates ↔ df_tf_dates[test_indices]")
    df_tf_dates_at_test = df_tf_dates[test_indices]
    mismatch_2 = (test_dates != df_tf_dates_at_test).sum()
    all_ok &= check(
        f"  test_dates == df_tf_dates[indices]",
        mismatch_2 == 0,
        f"{mismatch_2}/{n_test} mismatches")
    if mismatch_2 > 0:
        bad = np.where(test_dates != df_tf_dates_at_test)[0][:5]
        for i in bad:
            print(f"    i={i}  idx_tf={test_indices[i]}  "
                  f"test_date={test_dates[i]}  "
                  f"df_tf_date={df_tf_dates_at_test[i]}")

    # ========================================================================
    # [3] closes_test ↔ df_tf_closes[test_indices]
    # ========================================================================
    print(f"\n[3] closes_test ↔ df_tf_closes[test_indices]")
    df_tf_closes_at_test = df_tf_closes[test_indices]
    max_diff_3 = np.max(np.abs(closes_test - df_tf_closes_at_test))
    mismatch_3 = int((np.abs(closes_test - df_tf_closes_at_test) > 1e-10).sum())
    all_ok &= check(
        f"  closes_test == df_tf_closes[indices]  (max diff = {max_diff_3:.2e})",
        mismatch_3 == 0,
        f"{mismatch_3}/{n_test} mismatches > 1e-10")

    # ========================================================================
    # [4] Formule mathématique oracle: slope[t] = pos[t-1] - pos[t-2]
    # ========================================================================
    print(f"\n[4] Formule oracle : slope[t] = positions[t-1] - positions[t-2]")
    # On doit recalculer positions depuis compute_oracle_labels (pas direct dans NPZ)
    paths = SOURCE_PATHS[args.source]
    df_5m_src = load_csv(paths[5])
    if args.source == 'full':
        df_tf_src = resample_ohlcv(df_5m_src, args.tf)
    else:
        df_tf_src = load_csv(paths[args.tf])
    df_tf_src, _ = drop_incomplete_last(df_tf_src, df_5m_src, args.tf)

    print(f"  Recalcul oracle depuis CSV source ...")
    oracle_recalc = compute_oracle_labels(df_tf_src, args.indicator)
    pos_recalc = oracle_recalc['position'].values
    slope_recalc = oracle_recalc['slope'].values

    # Vérifier la formule sur t >= 2
    expected_slope = np.zeros_like(slope_recalc)
    expected_slope[2:] = pos_recalc[1:-1] - pos_recalc[:-2]
    formula_diff = np.max(np.abs(slope_recalc - expected_slope))
    all_ok &= check(
        f"  max |slope - (pos[t-1]-pos[t-2])| = {formula_diff:.2e}",
        formula_diff < 1e-10,
        "")

    # ========================================================================
    # [5] Reproductibilité : oracle_slopes_full (NPZ) vs oracle recalculé
    # ========================================================================
    print(f"\n[5] Reproductibilité : oracle_slopes_full (NPZ) vs oracle recalculé")
    # Les deux doivent avoir la même longueur
    ok_len = len(oracle_slopes_full) == len(slope_recalc)
    all_ok &= check(
        f"  len match: NPZ={len(oracle_slopes_full)}  recalc={len(slope_recalc)}",
        ok_len, "")
    if ok_len:
        max_diff_5 = np.max(np.abs(oracle_slopes_full - slope_recalc))
        all_ok &= check(
            f"  max |NPZ - recalc| = {max_diff_5:.2e}",
            max_diff_5 < 1e-10, "")

    # ========================================================================
    # [BONUS] X_test reverse check
    # ========================================================================
    print(f"\n[BONUS] X_test : reverse-denormalize et compare aux slopes reconstruites")
    # Dernière timestep de chaque sample : X_test[i, -1, :] = slopes à test_indices[i] normalisées
    # Dé-normaliser: raw = normalized * std + mean
    X_last_norm = X_test[:, -1, :]  # (n_test, 6)
    X_last_denorm = X_last_norm * norm_stds + norm_means  # (n_test, 6)

    # Recalculer les slopes FLKS depuis df_5m/df_tf
    print(f"  Recalcul des slopes FLKS (prend ~30s sur 8.5 ans) ...")
    slopes_recalc_df = compute_flks_slopes(df_tf_src, df_5m_src, args.indicator, args.tf)
    slopes_at_indices = slopes_recalc_df[feature_cols].values[test_indices]

    # Comparaison
    max_diff_bonus = np.max(np.abs(X_last_denorm - slopes_at_indices))
    mean_diff_bonus = np.mean(np.abs(X_last_denorm - slopes_at_indices))
    all_ok &= check(
        f"  max |X_last_denorm - slopes_recalc[indices]| = {max_diff_bonus:.2e}",
        max_diff_bonus < 1e-4,  # tolérance plus large (float32 + norm)
        f"mean diff = {mean_diff_bonus:.2e}")

    if max_diff_bonus > 1e-4:
        bad = np.argsort(np.abs(X_last_denorm - slopes_at_indices).max(axis=1))[::-1][:3]
        print(f"  Top 3 divergences:")
        for i in bad:
            print(f"    i={i}  idx_tf={test_indices[i]}  date={test_dates[i]}")
            print(f"      X_last_denorm: {X_last_denorm[i]}")
            print(f"      slopes_recalc: {slopes_at_indices[i]}")

    # ========================================================================
    # Verdict
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"VERDICT: {'✅ NPZ SYNCHRONIZED' if all_ok else '❌ DÉSYNCHRONISATION DÉTECTÉE'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
