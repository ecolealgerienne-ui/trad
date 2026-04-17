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

    # Exclure les bougies incomplètes (la dernière typiquement, car 5m s'arrête
    # avant la fin de la bougie TF). Sinon OHLC diffère → MACD/RSI/CCI diffèrent.
    # Les indicateurs (EWM) sont CUMULATIFS donc on doit tronquer AVANT le calcul,
    # pas juste au moment de la comparaison.
    def drop_incomplete_last(df_tf, tf_minutes):
        expected = tf_minutes // 5
        # On itère en arrière pour retirer les bougies incomplètes à la fin
        # (typiquement la toute dernière).
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

    print("\nExclusion des bougies de bord incomplètes (5m insuffisantes) :")
    for tf_minutes in (30, 60):
        tf_label = f'{tf_minutes}m' if tf_minutes < 60 else '1h'
        df_tf_R[tf_minutes], n_dropped_R = drop_incomplete_last(df_tf_R[tf_minutes], tf_minutes)
        df_tf[tf_minutes], n_dropped_D = drop_incomplete_last(df_tf[tf_minutes], tf_minutes)
        print(f"  {tf_label}: droppé {n_dropped_R} (resample) / {n_dropped_D} (téléchargé)")

    # -----------------------------------------------------------------------
    # Diagnostic CRITIQUE : vérifier que les closes sont bit-à-bit identiques
    # -----------------------------------------------------------------------
    import hashlib
    print("\n" + "=" * 80)
    print("DIAGNOSTIC — closes 5m-resamplé vs téléchargé (bit-à-bit)")
    print("=" * 80)
    for tf_minutes in (30, 60):
        tf_label = f'{tf_minutes}m' if tf_minutes < 60 else '1h'
        common = df_tf_R[tf_minutes].index.intersection(df_tf[tf_minutes].index)
        df_R = df_tf_R[tf_minutes].loc[common]
        df_D = df_tf[tf_minutes].loc[common]

        for col in ['open', 'high', 'low', 'close']:
            arr_R = df_R[col].values
            arr_D = df_D[col].values
            bit_eq = np.array_equal(arr_R, arr_D)  # strict, pas de tolérance
            max_abs = np.max(np.abs(arr_R - arr_D))
            hash_R = hashlib.md5(arr_R.tobytes()).hexdigest()[:16]
            hash_D = hashlib.md5(arr_D.tobytes()).hexdigest()[:16]
            status = '✅ identiques' if bit_eq else '❌ différents'
            print(f"  {tf_label:<5} {col:<6} bit-eq={bit_eq}  max|diff|={max_abs:.6g}  "
                  f"md5_R={hash_R}  md5_D={hash_D}  {status}")

    # -----------------------------------------------------------------------
    # Contrôle positif : perturber 1 close et vérifier qu'on détecte un diff
    # (pour prouver que le comparator n'est pas cassé)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CONTRÔLE POSITIF — perturbation volontaire pour valider le test")
    print("=" * 80)
    common_30 = df_tf_R[30].index.intersection(df_tf[30].index)
    df_R_30 = df_tf_R[30].loc[common_30].copy()
    df_D_30 = df_tf[30].loc[common_30].copy()
    # Perturbation de 1e-6 (bien plus grand que float64 epsilon) sur le MILIEU
    mid = len(df_R_30) // 2
    close_orig = df_R_30['close'].iloc[mid]
    df_R_30.iloc[mid, df_R_30.columns.get_loc('close')] = close_orig + 1e-6
    ind_A_pert = compute_indicator(df_R_30, 'macd')
    ind_B_pert = compute_indicator(df_D_30, 'macd')
    ok_pert, stats_pert = compare_indicator(ind_A_pert, ind_B_pert, 30, 'macd')
    print(f"  Perturbation close[{mid}] += 1e-6")
    print(f"  Max |diff| MACD: {stats_pert['max_abs']:.6e}")
    print(f"  Max rel:         {stats_pert['max_rel']:.6e}")
    print(f"  N mismatch:      {stats_pert['n_mismatch']} / {stats_pert['n_valid']}")
    if stats_pert['n_mismatch'] > 0:
        print("  ✅ Comparator détecte bien les perturbations.")
    else:
        print("  ❌ ALARM: comparator ne détecte pas la perturbation !")

    # -----------------------------------------------------------------------
    # Table de résultats principale
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RÉSULTATS PRINCIPAUX")
    print("=" * 80)
    print(f"{'TF':<5} {'Indic':<6} {'N valid':>10} {'Max |diff|':>15} {'Max rel':>15} {'N mismatch':>12} {'Status':<8}")
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
