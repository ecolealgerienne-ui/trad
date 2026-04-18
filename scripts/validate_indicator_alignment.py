#!/usr/bin/env python3
"""
Valide l'alignement des datasets progressifs entre MACD / RSI / CCI.

Règle CRITIQUE pour la cross-validation future : les 3 NPZ doivent avoir
EXACTEMENT les mêmes index temporels (dates_5min) sur train/val/test.
Cela garantit qu'on peut comparer / fusionner les prédictions des 3
modèles signal par signal.

Vérifications (bit-exactes sauf indication) :
  [1] Même métadonnées (trim, tf_minutes, train_ratio, val_ratio, gap_5m)
  [2] Mêmes dates_train/val/test (ns precision)
  [3] Mêmes indices_train/val/test (int64)
  [4] Mêmes closes_train/val/test (float64 exact)
  [5] Mêmes df_5m_dates / df_5m_closes (source identique)
  [6] Mêmes df_tf_dates / df_tf_closes (resample identique)

Alerte sur ce qui DOIT différer (par design) :
  - X_{split} (features slope_progressive diffèrent selon indicateur)
  - y_{split}_binary/continuous (labels oracle spécifiques à l'indicateur)
  - oracle_slopes_full (pente oracle spécifique à l'indicateur)
  - norm_means / norm_stds (z-score fitté sur données différentes)

Usage :
    python scripts/validate_indicator_alignment.py
    python scripts/validate_indicator_alignment.py --tf 30 --period full
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PREP_DIR = Path('data/prepared')


def check(label, condition, detail=''):
    status = '✅' if condition else '❌'
    print(f"  {status} {label}  {detail}")
    return condition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', type=int, default=30, choices=[30, 60])
    parser.add_argument('--period', default='full',
                        help='full, 180d, 30d, ...')
    parser.add_argument('--indicators', nargs='+',
                        default=['macd', 'rsi', 'cci'])
    args = parser.parse_args()

    tf_label = f'{args.tf}m' if args.tf < 60 else '1h'

    print("=" * 80)
    print(f"VALIDATE INDICATOR ALIGNMENT — TF={tf_label}  period={args.period}")
    print(f"  indicators : {args.indicators}")
    print("=" * 80)

    # Chargement des 3 NPZ
    dsets = {}
    for ind in args.indicators:
        npz_path = PREP_DIR / f'dataset_{ind}_{tf_label}_{args.period}_progressive.npz'
        if not npz_path.exists():
            print(f"❌ Manquant: {npz_path}")
            print(f"   Génère d'abord avec : "
                  f"python scripts/prepare_progressive_data.py --indicator {ind} --tf {args.tf}"
                  + (f" --days {args.period.rstrip('d')}" if args.period != 'full' else ''))
            return
        dsets[ind] = np.load(npz_path, allow_pickle=True)
        print(f"✅ Chargé: {npz_path.name}  "
              f"({npz_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Indicateur référence = premier de la liste
    ref = args.indicators[0]
    others = args.indicators[1:]
    ds_ref = dsets[ref]
    print(f"\n→ Référence = {ref.upper()}, comparaison avec {[o.upper() for o in others]}")

    all_ok = True

    # ========================================================================
    # [1] Métadonnées
    # ========================================================================
    print(f"\n[1] Métadonnées")
    for key in ['tf_minutes', 'train_ratio', 'val_ratio', 'gap_5m', 'trim']:
        ref_val = ds_ref[key].item() if hasattr(ds_ref[key], 'item') else ds_ref[key]
        for ind in others:
            v = dsets[ind][key].item() if hasattr(dsets[ind][key], 'item') else dsets[ind][key]
            all_ok &= check(f"{ref}.{key} == {ind}.{key}", v == ref_val,
                             f"(ref={ref_val}, {ind}={v})")

    # ========================================================================
    # [2] Dates train/val/test
    # ========================================================================
    print(f"\n[2] dates_train / dates_val / dates_test (identiques aux ns)")
    for split in ['train', 'val', 'test']:
        key = f'dates_{split}'
        ref_arr = ds_ref[key]
        for ind in others:
            arr = dsets[ind][key]
            same_len = len(arr) == len(ref_arr)
            same_content = same_len and np.array_equal(arr, ref_arr)
            all_ok &= check(
                f"{split}: len({ind}) == len({ref}) et contenu identique",
                same_content,
                f"(ref n={len(ref_arr):,}, {ind} n={len(arr):,})")

    # ========================================================================
    # [3] Indices 5min
    # ========================================================================
    print(f"\n[3] indices_train / indices_val / indices_test")
    for split in ['train', 'val', 'test']:
        key = f'indices_{split}'
        ref_arr = ds_ref[key]
        for ind in others:
            arr = dsets[ind][key]
            same = np.array_equal(arr, ref_arr)
            all_ok &= check(f"{split}: {ind} == {ref}", same,
                             f"(n={len(ref_arr):,})")

    # ========================================================================
    # [4] Closes train/val/test
    # ========================================================================
    print(f"\n[4] closes_train / closes_val / closes_test (float64 exact)")
    for split in ['train', 'val', 'test']:
        key = f'closes_{split}'
        ref_arr = ds_ref[key]
        for ind in others:
            arr = dsets[ind][key]
            if len(arr) != len(ref_arr):
                all_ok &= check(f"{split}: len({ind}) == len({ref})", False,
                                 f"(ref n={len(ref_arr):,}, {ind} n={len(arr):,})")
                continue
            max_diff = float(np.max(np.abs(arr - ref_arr)))
            all_ok &= check(f"{split}: max|{ind} - {ref}| = {max_diff:.2e}",
                             max_diff == 0.0, "")

    # ========================================================================
    # [5] df_5m (source globale)
    # ========================================================================
    print(f"\n[5] df_5m_dates / df_5m_closes (source globale)")
    ref_dates_5m = ds_ref['df_5m_dates']
    ref_closes_5m = ds_ref['df_5m_closes']
    for ind in others:
        dates_ok = np.array_equal(dsets[ind]['df_5m_dates'], ref_dates_5m)
        closes_ok = np.array_equal(dsets[ind]['df_5m_closes'], ref_closes_5m)
        all_ok &= check(f"df_5m_dates {ind} == {ref}", dates_ok,
                         f"(n={len(ref_dates_5m):,})")
        all_ok &= check(f"df_5m_closes {ind} == {ref}", closes_ok, "")

    # ========================================================================
    # [6] df_tf (resample TF)
    # ========================================================================
    print(f"\n[6] df_tf_dates / df_tf_closes (resample TF)")
    ref_dates_tf = ds_ref['df_tf_dates']
    ref_closes_tf = ds_ref['df_tf_closes']
    for ind in others:
        dates_ok = np.array_equal(dsets[ind]['df_tf_dates'], ref_dates_tf)
        closes_ok = np.array_equal(dsets[ind]['df_tf_closes'], ref_closes_tf)
        all_ok &= check(f"df_tf_dates {ind} == {ref}", dates_ok,
                         f"(n={len(ref_dates_tf):,})")
        all_ok &= check(f"df_tf_closes {ind} == {ref}", closes_ok, "")

    # ========================================================================
    # Champs qui DOIVENT différer (par design)
    # ========================================================================
    print(f"\n[Rappel] Champs qui DOIVENT différer (spécifiques à chaque indicateur):")
    print("  - X_train/val/test (slope_progressive : valeurs différentes)")
    print("  - y_train/val/test_binary / _continuous (oracle différent)")
    print("  - oracle_slopes_full (pente oracle différente)")
    print("  - norm_means / norm_stds (z-score fitté sur données différentes)")

    # Rapide sanity check : les X doivent bien différer
    print(f"\n[Sanity] X_test et y_test doivent DIFFÉRER entre indicateurs")
    for ind in others:
        X_same = np.array_equal(dsets[ind]['X_test'], ds_ref['X_test'])
        y_same = np.array_equal(dsets[ind]['y_test_binary'], ds_ref['y_test_binary'])
        # Le check inversé : OK si différent
        check(f"X_test {ind} ≠ {ref}", not X_same,
               "(si identique → problème de calcul d'indicateur)")
        check(f"y_test_binary {ind} ≠ {ref}", not y_same,
               "(si identique → problème de calcul d'oracle)")

    # ========================================================================
    # Verdict
    # ========================================================================
    print(f"\n{'=' * 80}")
    if all_ok:
        print(f"VERDICT: ✅ ALIGNEMENT PARFAIT — cross-validation entre indicateurs OK")
    else:
        print(f"VERDICT: ❌ DÉSALIGNEMENT DÉTECTÉ — NE PAS faire de cross-validation")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
