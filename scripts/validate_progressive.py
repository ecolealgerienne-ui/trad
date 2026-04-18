#!/usr/bin/env python3
"""
Smoke test de l'architecture progressive.

Vérifie :
  [1] compute_progressive_slopes : step_k ∈ {0..5} uniformément, cohérence feature
  [2] prepare_features_and_labels_progressive : ffill labels OK, alignement close
  [3] Cohérence avec compute_flks_slopes (historique) au step_k=k

Travaille sur un échantillon (derniers 30 jours par défaut) pour rapidité.

Usage:
    python scripts/validate_progressive.py
    python scripts/validate_progressive.py --days 90 --tf 30
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
    compute_flks_slopes,
    compute_progressive_slopes,
    prepare_features_and_labels_progressive,
    compute_oracle_labels,
)

SRC_5M = Path('data_trad/BTCUSD_all_5m.csv')


def check(name, passed, detail=""):
    status = "✅" if passed else "❌"
    print(f"  {status} {name}  {detail}")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indicator', default='macd',
                        choices=['macd', 'rsi', 'cci'])
    parser.add_argument('--tf', type=int, default=30, choices=[30, 60])
    parser.add_argument('--days', type=int, default=30,
                        help='Derniers N jours à utiliser (default 30)')
    args = parser.parse_args()

    print("=" * 80)
    print(f"VALIDATION PROGRESSIVE — {args.indicator.upper()} × {args.tf}m  "
          f"(derniers {args.days} jours)")
    print("=" * 80)

    if not SRC_5M.exists():
        print(f"❌ Source introuvable: {SRC_5M}")
        return

    # Load + filter
    print(f"\n[Load 5m] {SRC_5M}")
    df_5m_full = load_csv(SRC_5M)
    end = df_5m_full.index[-1]
    start = end - pd.Timedelta(days=args.days)
    df_5m = df_5m_full.loc[df_5m_full.index >= start].copy()
    print(f"  {len(df_5m):,} rows 5min  |  {df_5m.index[0]} → {df_5m.index[-1]}")

    df_tf = resample_ohlcv(df_5m, args.tf)
    print(f"  {len(df_tf):,} bougies {args.tf}m")

    all_ok = True

    # ====================================================================
    # [1] compute_progressive_slopes
    # ====================================================================
    print(f"\n[1] compute_progressive_slopes ...")
    slopes_prog = compute_progressive_slopes(df_tf, df_5m, args.indicator, args.tf)
    n_sub = args.tf // 5
    print(f"  Shape: {slopes_prog.shape}  (attendu: ~{len(df_5m)} rows, 2 cols)")

    # Vérifier step_k
    step_counts = slopes_prog['step_k'].value_counts().sort_index()
    print(f"  Distribution step_k: {dict(step_counts)}")
    all_ok &= check(
        f"step_k ∈ [0, {n_sub-1}]",
        (slopes_prog['step_k'].min() >= 0) and (slopes_prog['step_k'].max() <= n_sub - 1),
        f"min={slopes_prog['step_k'].min()}, max={slopes_prog['step_k'].max()}")

    # Chaque step_k doit être présent ~uniformément
    expected_per_step = len(slopes_prog) / n_sub
    max_dev = max(abs(c - expected_per_step) for c in step_counts.values)
    all_ok &= check(
        f"step_k uniformément réparti",
        max_dev < expected_per_step * 0.05,
        f"max deviation = {max_dev:.0f} (<5% de {expected_per_step:.0f})")

    # ====================================================================
    # [2] Cohérence progressive vs historique (slope_k<k> au step_k=k)
    # ====================================================================
    print(f"\n[2] Cohérence progressive vs historique (compute_flks_slopes)")
    slopes_tf = compute_flks_slopes(df_tf, df_5m, args.indicator, args.tf,
                                      k_range=(1, n_sub - 1))

    # Pour chaque step_k ∈ [0, n_sub-1], vérifier que slope_progressive[step_k==k]
    # correspond à slope_t1 (si k=0) ou slope_k<k> (si k>=1) du t_ref correspondant.
    tf_delta = pd.Timedelta(minutes=args.tf)
    t_ref_5m = slopes_prog.index.floor(tf_delta) - tf_delta

    # On teste avec un échantillon de 500 lignes par step_k
    rng = np.random.default_rng(42)
    for k in range(n_sub):
        mask_k = slopes_prog['step_k'].values == k
        if mask_k.sum() == 0:
            continue
        idx_k = np.where(mask_k)[0]
        sample = rng.choice(idx_k, size=min(500, len(idx_k)), replace=False)

        col = 'slope_t1' if k == 0 else f'slope_k{k}'
        max_diff = 0.0
        for i in sample:
            t_ref = t_ref_5m[i]
            if t_ref not in slopes_tf.index:
                continue
            expected = slopes_tf.loc[t_ref, col]
            got = slopes_prog['slope_progressive'].iloc[i]
            diff = abs(expected - got)
            if diff > max_diff:
                max_diff = diff
        all_ok &= check(
            f"step_k={k}: slope_progressive == {col}  (max diff={max_diff:.2e})",
            max_diff < 1e-10, "")

    # ====================================================================
    # [3] prepare_features_and_labels_progressive
    # ====================================================================
    print(f"\n[3] prepare_features_and_labels_progressive ...")
    trim = min(100, len(df_tf) // 4)  # trim adaptatif pour petit échantillon
    data = prepare_features_and_labels_progressive(
        df_tf, df_5m, args.indicator, args.tf, trim=trim)
    print(f"  Shape: {data.shape}  |  colonnes: {list(data.columns)}")
    print(f"  Plage: {data.index[0]} → {data.index[-1]}")

    # Vérifier que labels sont constants sur chaque bougie TF (ffill OK)
    # Groupby par t_ref et vérifier unicité label_binary par groupe
    t_ref_data = data.index.floor(tf_delta) - tf_delta
    groups = data.groupby(t_ref_data)
    non_constant = 0
    for t, grp in groups:
        if grp['label_binary'].nunique() > 1:
            non_constant += 1
    all_ok &= check(
        f"Labels ffill constants dans chaque bougie TF",
        non_constant == 0,
        f"{non_constant} bougies avec labels non-constants sur {len(groups)}")

    # Vérifier cohérence close avec df_5m
    close_diff = np.max(np.abs(data['close'].values - df_5m['close'].loc[data.index].values))
    all_ok &= check(
        f"close == df_5m.close aligné",
        close_diff < 1e-10,
        f"max diff = {close_diff:.2e}")

    # Vérifier cohérence labels avec oracle[t_ref]
    oracle = compute_oracle_labels(df_tf, args.indicator)
    oracle_aligned = oracle.reindex(t_ref_data).fillna(0)
    label_bin_diff = (data['label_binary'].values !=
                      oracle_aligned['label'].astype(int).values).sum()
    all_ok &= check(
        f"label_binary == oracle.label[t_ref] ffill",
        label_bin_diff == 0,
        f"{label_bin_diff} mismatches sur {len(data)}")

    # ====================================================================
    # Verdict
    # ====================================================================
    print("\n" + "=" * 80)
    print(f"VERDICT: {'✅ OK' if all_ok else '❌ PROBLÈMES DÉTECTÉS'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
