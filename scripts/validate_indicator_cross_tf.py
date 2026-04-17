#!/usr/bin/env python3
"""
Valide que les indicateurs standards calculés via deux chemins coïncident :

  Path A : 5m → resample_ohlcv(5m, tf) → compute_indicator(df_tf_R, indic)
  Path B : df_tf_downloaded → compute_indicator(df_tf_dl, indic)

Puisque resample_ohlcv(5m, tf) a déjà été prouvé strictement identique à
df_tf_downloaded en OHLC (validate_resample.py), les indicateurs A et B
devraient coïncider à la tolérance près (erreurs d'arrondi float64 sur le
EWM).

Testé pour toutes les combinaisons (tf, indicateur).

Usage:
    python scripts/validate_indicator_cross_tf.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import (
    resample_ohlcv, load_csv, compute_indicator,
)


DATA_DIR = Path('data/raw')

COMBINATIONS = [
    (30, 'macd'),
    (60, 'macd'),
    (30, 'rsi'),
    (60, 'rsi'),
    (30, 'cci'),
    (60, 'cci'),
]

# Tolérance sur indicateurs (en valeur absolue).
# - MACD ≈ valeurs brutes du prix (milliers) → tolerance absolue modeste.
# - RSI ∈ [0, 100] → tolerance relative au pourcent.
# - CCI ∈ [-300..+300] typique → tolerance modeste.
# On utilise relative pour robustesse, avec floor absolu.
TOL_ABS = 1e-6     # seuil minimal pour ignorer les ≈0
TOL_REL = 1e-10    # tolérance relative sur la valeur


def compare_indicator(arr_A, arr_B, tf_minutes, indicator, tol_abs=TOL_ABS, tol_rel=TOL_REL):
    """Compare deux séries d'indicateurs (aligned index). Retourne (ok, stats)."""
    # Masque : ne compare que les positions où LES DEUX valeurs sont finies
    mask = np.isfinite(arr_A) & np.isfinite(arr_B)
    n_valid = mask.sum()
    if n_valid == 0:
        return False, {'n_valid': 0, 'max_abs': np.nan, 'max_rel': np.nan, 'n_mismatch': 0}

    a = arr_A[mask]
    b = arr_B[mask]

    abs_diff = np.abs(a - b)
    # Relative diff : diviser par max(|a|, |b|, floor)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), tol_abs)
    rel_diff = abs_diff / denom

    max_abs = abs_diff.max()
    max_rel = rel_diff.max()
    n_mismatch = (rel_diff > tol_rel).sum()

    # OK si la tolérance relative est respectée partout
    ok = n_mismatch == 0

    return ok, {
        'n_valid': int(n_valid),
        'max_abs': float(max_abs),
        'max_rel': float(max_rel),
        'n_mismatch': int(n_mismatch),
    }


def main():
    print("=" * 80)
    print("VALIDATION INDICATEURS : resample(5m→TF) vs TF téléchargé")
    print("=" * 80)

    paths = {
        '5m': DATA_DIR / 'BTCUSD_3months_5m.csv',
        '30m': DATA_DIR / 'BTCUSD_3months_30m.csv',
        '1h': DATA_DIR / 'BTCUSD_3months_1h.csv',
    }
    for tf, p in paths.items():
        if not p.exists():
            print(f"❌ Fichier manquant: {p}")
            return

    print("\nChargement ...")
    df_5m = load_csv(paths['5m'])
    df_tf = {
        30: load_csv(paths['30m']),
        60: load_csv(paths['1h']),
    }
    print(f"  5m:  {len(df_5m):,} rows")
    print(f"  30m: {len(df_tf[30]):,} rows")
    print(f"  1h:  {len(df_tf[60]):,} rows")

    # Précalcul : resample 5m → 30m et 5m → 1h
    print("\nResample ...")
    df_tf_R = {
        30: resample_ohlcv(df_5m, 30),
        60: resample_ohlcv(df_5m, 60),
    }
    print(f"  Resample 30m: {len(df_tf_R[30]):,} rows")
    print(f"  Resample 1h:  {len(df_tf_R[60]):,} rows")

    # Table de résultats
    print(f"\n{'TF':<5} {'Indic':<6} {'N valid':>10} {'Max |diff|':>15} {'Max rel':>15} {'N mismatch':>12} {'Status':<8}")
    print("-" * 80)

    all_ok = True
    for tf_minutes, indicator in COMBINATIONS:
        tf_label = f'{tf_minutes}m' if tf_minutes < 60 else '1h'

        # Restreindre le resample aux timestamps communs avec le téléchargé
        common = df_tf_R[tf_minutes].index.intersection(df_tf[tf_minutes].index)
        if len(common) == 0:
            print(f"{tf_label:<5} {indicator:<6}  [aucun timestamp commun]")
            continue

        df_R = df_tf_R[tf_minutes].loc[common]
        df_D = df_tf[tf_minutes].loc[common]

        ind_A = compute_indicator(df_R, indicator)
        ind_B = compute_indicator(df_D, indicator)

        ok, stats = compare_indicator(ind_A, ind_B, tf_minutes, indicator)
        status = '✅ PASS' if ok else '❌ FAIL'
        print(f"{tf_label:<5} {indicator:<6} {stats['n_valid']:>10} "
              f"{stats['max_abs']:>15.6g} {stats['max_rel']:>15.6g} "
              f"{stats['n_mismatch']:>12} {status:<8}")
        if not ok:
            all_ok = False
            # Afficher les 3 premières divergences
            mask = np.isfinite(ind_A) & np.isfinite(ind_B)
            abs_diff = np.abs(ind_A - ind_B)
            denom = np.maximum(np.maximum(np.abs(ind_A), np.abs(ind_B)), TOL_ABS)
            rel_diff = np.where(mask, abs_diff / denom, 0)
            bad_idx = np.where(rel_diff > TOL_REL)[0]
            print(f"  → Premières 3 divergences :")
            for i in bad_idx[:3]:
                ts = df_R.index[i]
                print(f"    {ts}  A={ind_A[i]:.8f}  B={ind_B[i]:.8f}  "
                      f"|diff|={abs_diff[i]:.2e}  rel={rel_diff[i]:.2e}")

    print("-" * 80)
    print("=" * 80)
    if all_ok:
        print("✅ TOUTES les combinaisons (TF × indicateur) PASS.")
    else:
        print("❌ Au moins une combinaison a échoué.")
    print("=" * 80)


if __name__ == '__main__':
    main()
