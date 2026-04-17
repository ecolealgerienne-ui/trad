#!/usr/bin/env python3
"""
Valide compute_oracle_labels(df, indicator) sur MACD 30m et MACD 1h.

Tests:
  1. Structure: shape, colonnes, pas de NaN (fillna 0 effectif)
  2. Relation mathématique: slope[t] == position[t-1] - position[t-2]
  3. Relation mathématique: label[t] == (1 if slope > 0 else 0)
  4. Propriété du smoother: std(position) < std(indicator) — lissage
  5. Non-causalité: perturber close[T+k] change position[T]
  6. Statistiques: distribution slopes, transitions, ratio UP/DOWN
  7. Path A vs Path B: oracle calculé via resample(5m→TF) == oracle calculé
     directement sur TF téléchargé (cohérence pipeline)
  Bonus: Cross-TF 30m vs 1h (concordance de signe)

Usage:
    python scripts/validate_oracle.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import (
    load_csv, resample_ohlcv, compute_indicator, compute_oracle_labels,
)


DATA_DIR = Path('data/raw')
INDICATOR = 'macd'
TFS = [30, 60]
TRIM = 100


def check(name, passed, detail=""):
    status = "✅" if passed else "❌"
    print(f"  {status} {name}  {detail}")
    return passed


def drop_incomplete_last(df_tf, df_5m, tf_minutes):
    """Supprime les bougies de fin où les 5m ne sont pas toutes présentes."""
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
    print("=" * 80)
    print(f"VALIDATION compute_oracle_labels — {INDICATOR.upper()} sur 30m et 1h")
    print("=" * 80)

    # Charger les 3 fichiers
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
    print(f"  5m: {len(df_5m):,} rows | {df_5m.index[0]} → {df_5m.index[-1]}")
    for tf, df in dfs_dl.items():
        lbl = f'{tf}m' if tf < 60 else '1h'
        print(f"  {lbl} (téléchargé): {len(df):,} rows | {df.index[0]} → {df.index[-1]}")

    # Path A : 5m → resample → df_tf_R
    print("\nResample 5m → 30m et 5m → 1h ...")
    dfs_R = {30: resample_ohlcv(df_5m, 30), 60: resample_ohlcv(df_5m, 60)}

    # Tronquer les bougies incomplètes des DEUX côtés (resample ET téléchargé)
    # pour que les closes alimentant l'oracle soient bit-à-bit identiques.
    for tf in TFS:
        dfs_R[tf], n_dR = drop_incomplete_last(dfs_R[tf], df_5m, tf)
        dfs_dl[tf], n_dD = drop_incomplete_last(dfs_dl[tf], df_5m, tf)
        lbl = f'{tf}m' if tf < 60 else '1h'
        print(f"  {lbl}: droppé {n_dR} (resample) / {n_dD} (téléchargé)  "
              f"→ tailles: {len(dfs_R[tf]):,} et {len(dfs_dl[tf]):,}")

    # ========== Calculer les oracles SUR LES DEUX CHEMINS ==========
    print(f"\n[1/7] Calcul compute_oracle_labels via Path A (5m resamplé) "
          f"ET Path B (téléchargé) ...")
    oracles_A = {}  # Path A: depuis resample 5m
    oracles_B = {}  # Path B: depuis téléchargé direct
    for tf in TFS:
        oracles_A[tf] = compute_oracle_labels(dfs_R[tf], INDICATOR)
        oracles_B[tf] = compute_oracle_labels(dfs_dl[tf], INDICATOR)
    for tf in TFS:
        lbl = f'{tf}m' if tf < 60 else '1h'
        print(f"  {lbl}: Path A shape={oracles_A[tf].shape}  "
              f"Path B shape={oracles_B[tf].shape}")

    # ========== Test 1: Structure (sur Path B) ==========
    print(f"\n[2/7] Structure (vérifiée sur Path B = oracle téléchargé) ...")
    all_ok = True
    for tf in TFS:
        lbl = f'{tf}m' if tf < 60 else '1h'
        orc = oracles_B[tf]
        df = dfs_dl[tf]
        all_ok &= check(
            f"{lbl}: len(oracle) == len(df)",
            len(orc) == len(df),
            f"{len(orc)} vs {len(df)}")
        all_ok &= check(
            f"{lbl}: colonnes = [position, slope, label]",
            list(orc.columns) == ['position', 'slope', 'label'],
            f"got {list(orc.columns)}")
        all_ok &= check(
            f"{lbl}: pas de NaN (fillna 0 effectif)",
            not orc.isna().any().any(),
            "")
        all_ok &= check(
            f"{lbl}: index identique au df",
            orc.index.equals(df.index),
            "")

    # ========== Test 2: Relation mathématique slope = diff position ==========
    print(f"\n[3/7] Relation slope[t] == position[t-1] - position[t-2] (Path B) ...")
    for tf in TFS:
        lbl = f'{tf}m' if tf < 60 else '1h'
        orc = oracles_B[tf]
        pos = orc['position'].values
        slope = orc['slope'].values
        expected = np.zeros_like(slope)
        expected[2:] = pos[1:-1] - pos[:-2]
        max_diff = np.max(np.abs(slope - expected))
        all_ok &= check(
            f"{lbl}: max |slope - (pos[t-1]-pos[t-2])| = {max_diff:.2e}",
            max_diff < 1e-12,
            "")

    # ========== Test 3: Relation label = (slope > 0) ==========
    print(f"\n[4/7] Relation label[t] == (1 if slope[t] > 0 else 0) (Path B) ...")
    for tf in TFS:
        lbl = f'{tf}m' if tf < 60 else '1h'
        orc = oracles_B[tf]
        expected = (orc['slope'].values > 0).astype(int)
        match = np.array_equal(orc['label'].values, expected)
        all_ok &= check(
            f"{lbl}: label cohérent avec slope",
            match,
            f"mismatches: {int((orc['label'].values != expected).sum())}")

    # ========== Test 4: Propriété de lissage ==========
    print(f"\n[5/7] Propriété du smoother : std(position) < std(indicator) (Path B) ...")
    for tf in TFS:
        lbl = f'{tf}m' if tf < 60 else '1h'
        orc = oracles_B[tf]
        ind = compute_indicator(dfs_dl[tf], INDICATOR)
        std_ind = np.std(ind.values[TRIM:-TRIM])
        std_pos = np.std(orc['position'].values[TRIM:-TRIM])
        ratio = std_pos / std_ind if std_ind > 0 else float('inf')
        all_ok &= check(
            f"{lbl}: std(pos)={std_pos:.4f} vs std(ind)={std_ind:.4f} — "
            f"ratio={ratio:.3f}",
            std_pos < std_ind,
            f"(lissage {'effectif' if std_pos < std_ind else 'absent'})")

    # ========== Test 5: Non-causalité (Path B) ==========
    print(f"\n[6/7] Non-causalité : perturber close[T+10] change position[T] (Path B) ...")
    for tf in TFS:
        lbl = f'{tf}m' if tf < 60 else '1h'
        df = dfs_dl[tf]
        T = len(df) // 2
        df_B = df.copy()
        df_B.iloc[T + 10, df_B.columns.get_loc('close')] += 1000.0
        orc_A = compute_oracle_labels(df, INDICATOR)
        orc_Bp = compute_oracle_labels(df_B, INDICATOR)
        diff_at_T = abs(orc_A['position'].iloc[T] - orc_Bp['position'].iloc[T])
        all_ok &= check(
            f"{lbl}: |position_A[{T}] - position_B[{T}]| = {diff_at_T:.6e} "
            f"(doit être > 0)",
            diff_at_T > 1e-6,
            "")

    # ========== Test 6: Statistiques (Path B) ==========
    print(f"\n[7/7] Statistiques globales (Path B, après TRIM={TRIM}) ...")
    for tf in TFS:
        lbl = f'{tf}m' if tf < 60 else '1h'
        orc = oracles_B[tf]
        trimmed = orc.iloc[TRIM:-TRIM]
        n = len(trimmed)
        n_up = int((trimmed['label'] == 1).sum())
        n_down = int((trimmed['label'] == 0).sum())
        n_trans = int((trimmed['label'].diff().abs() > 0).sum())
        slope_stats = trimmed['slope'].describe()
        print(f"  [{lbl}] N={n:,}  UP={n_up:,} ({n_up/n*100:.1f}%)  "
              f"DOWN={n_down:,} ({n_down/n*100:.1f}%)  Transitions={n_trans:,} "
              f"({n_trans/n*100:.2f}%)")
        print(f"         slope: mean={slope_stats['mean']:+.4f}  std={slope_stats['std']:.4f}  "
              f"min={slope_stats['min']:+.4f}  max={slope_stats['max']:+.4f}")

    # ========== TEST CRITIQUE : Path A vs Path B ==========
    print(f"\n[CRITIQUE] Path A (resample 5m→TF) vs Path B (TF téléchargé) ...")
    print("  Attendu: DataFrames oracle IDENTIQUES (closes bit-à-bit identiques prouvés)")
    for tf in TFS:
        lbl = f'{tf}m' if tf < 60 else '1h'
        orc_A = oracles_A[tf]
        orc_B = oracles_B[tf]
        # Taille identique?
        same_shape = orc_A.shape == orc_B.shape
        all_ok &= check(f"{lbl}: shapes identiques", same_shape,
                        f"A={orc_A.shape} vs B={orc_B.shape}")
        if not same_shape:
            continue
        # Index identique?
        idx_eq = orc_A.index.equals(orc_B.index)
        all_ok &= check(f"{lbl}: index identiques", idx_eq, "")
        # Valeurs identiques par colonne
        for col in ['position', 'slope', 'label']:
            max_diff = np.max(np.abs(orc_A[col].values - orc_B[col].values))
            ok_col = max_diff < 1e-10
            all_ok &= check(
                f"{lbl}: max |A.{col} - B.{col}| = {max_diff:.2e}",
                ok_col, "")

    # ========== Cohérence Cross-TF ==========
    print(f"\n[BONUS] Cohérence Cross-TF : signe slope 30m vs 1h sur timestamps communs ...")
    orc_30 = oracles_B[30].iloc[TRIM:-TRIM]
    orc_1h = oracles_B[60].iloc[TRIM:-TRIM]
    common = orc_30.index.intersection(orc_1h.index)
    if len(common) > 10:
        s30 = np.sign(orc_30.loc[common, 'slope'].values)
        s1h = np.sign(orc_1h.loc[common, 'slope'].values)
        mask = (s30 != 0) & (s1h != 0)
        if mask.sum() > 0:
            concord = (s30[mask] == s1h[mask]).mean() * 100
            print(f"  Concordance de signe sur {mask.sum():,} timestamps communs: {concord:.1f}%")

    print("\n" + "=" * 80)
    print(f"VERDICT : {'✅ TOUS TESTS PASS' if all_ok else '❌ AU MOINS UN ÉCHEC'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
