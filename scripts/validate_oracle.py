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
  7. Cross-TF: concordance de signe des slopes entre 30m et 1h

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
    load_csv, compute_indicator, compute_oracle_labels,
)


DATA_DIR = Path('data/raw')
INDICATOR = 'macd'
TFS = [30, 60]  # minutes
TRIM = 100


def check(name, passed, detail=""):
    status = "✅" if passed else "❌"
    print(f"  {status} {name}  {detail}")
    return passed


def main():
    print("=" * 80)
    print(f"VALIDATION compute_oracle_labels — {INDICATOR.upper()} sur 30m et 1h")
    print("=" * 80)

    # Charger les données
    paths = {30: DATA_DIR / 'BTCUSD_3months_30m.csv',
             60: DATA_DIR / 'BTCUSD_3months_1h.csv'}
    dfs = {}
    for tf, p in paths.items():
        if not p.exists():
            print(f"❌ Fichier manquant: {p}")
            return
        dfs[tf] = load_csv(p)

    print(f"\nChargement:")
    for tf, df in dfs.items():
        label = f'{tf}m' if tf < 60 else '1h'
        print(f"  {label}: {len(df):,} rows | {df.index[0]} → {df.index[-1]}")

    # ========== Calculer les oracles ==========
    print(f"\n[1/7] Calcul compute_oracle_labels(df, '{INDICATOR}') ...")
    oracles = {}
    for tf in TFS:
        oracles[tf] = compute_oracle_labels(dfs[tf], INDICATOR)
    for tf in TFS:
        label = f'{tf}m' if tf < 60 else '1h'
        print(f"  {label}: {oracles[tf].shape} | colonnes = {list(oracles[tf].columns)}")

    # ========== Test 1: Structure ==========
    print(f"\n[2/7] Structure ...")
    all_ok = True
    for tf in TFS:
        label = f'{tf}m' if tf < 60 else '1h'
        orc = oracles[tf]
        df = dfs[tf]
        all_ok &= check(
            f"{label}: len(oracle) == len(df)",
            len(orc) == len(df),
            f"{len(orc)} vs {len(df)}")
        all_ok &= check(
            f"{label}: colonnes = [position, slope, label]",
            list(orc.columns) == ['position', 'slope', 'label'],
            f"got {list(orc.columns)}")
        all_ok &= check(
            f"{label}: pas de NaN (fillna 0 effectif)",
            not orc.isna().any().any(),
            "")
        all_ok &= check(
            f"{label}: index identique au df",
            orc.index.equals(df.index),
            "")

    # ========== Test 2: Relation mathématique slope = diff position ==========
    print(f"\n[3/7] Relation mathématique slope[t] == position[t-1] - position[t-2] ...")
    for tf in TFS:
        label = f'{tf}m' if tf < 60 else '1h'
        orc = oracles[tf]
        pos = orc['position'].values
        slope = orc['slope'].values
        # Pour t >= 2 : slope[t] doit être position[t-1] - position[t-2]
        # Avant fillna(0), slope[0] et slope[1] étaient NaN. Après fillna → 0.
        expected = np.zeros_like(slope)
        expected[2:] = pos[1:-1] - pos[:-2]
        max_diff = np.max(np.abs(slope - expected))
        all_ok &= check(
            f"{label}: max |slope - (pos[t-1]-pos[t-2])| = {max_diff:.2e}",
            max_diff < 1e-12,
            "")

    # ========== Test 3: Relation mathématique label = (slope > 0) ==========
    print(f"\n[4/7] Relation mathématique label[t] == (1 if slope[t] > 0 else 0) ...")
    for tf in TFS:
        label_ = f'{tf}m' if tf < 60 else '1h'
        orc = oracles[tf]
        expected = (orc['slope'].values > 0).astype(int)
        match = np.array_equal(orc['label'].values, expected)
        all_ok &= check(
            f"{label_}: label cohérent avec slope",
            match,
            f"mismatches: {int((orc['label'].values != expected).sum())}")

    # ========== Test 4: Propriété de lissage ==========
    print(f"\n[5/7] Propriété du smoother : std(position) < std(indicator) ...")
    for tf in TFS:
        label_ = f'{tf}m' if tf < 60 else '1h'
        orc = oracles[tf]
        ind = compute_indicator(dfs[tf], INDICATOR)
        # Comparer après warmup (TRIM)
        std_ind = np.std(ind.values[TRIM:-TRIM])
        std_pos = np.std(orc['position'].values[TRIM:-TRIM])
        ratio = std_pos / std_ind if std_ind > 0 else float('inf')
        all_ok &= check(
            f"{label_}: std(pos)={std_pos:.4f} vs std(ind)={std_ind:.4f} — "
            f"ratio={ratio:.3f}",
            std_pos < std_ind,
            f"(lissage {'effectif' if std_pos < std_ind else 'absent'})")

    # ========== Test 5: Non-causalité (comportement voulu) ==========
    print(f"\n[6/7] Non-causalité : perturber close[T+10] change position[T] ...")
    for tf in TFS:
        label_ = f'{tf}m' if tf < 60 else '1h'
        df = dfs[tf]
        T = len(df) // 2  # milieu
        df_B = df.copy()
        df_B.iloc[T + 10, df_B.columns.get_loc('close')] += 1000.0  # pollution forte
        orc_A = compute_oracle_labels(df, INDICATOR)
        orc_B = compute_oracle_labels(df_B, INDICATOR)
        diff_at_T = abs(orc_A['position'].iloc[T] - orc_B['position'].iloc[T])
        all_ok &= check(
            f"{label_}: |position_A[{T}] - position_B[{T}]| = {diff_at_T:.6e} "
            f"(doit être > 0 — smoother regarde le futur)",
            diff_at_T > 1e-6,
            "")

    # ========== Test 6: Statistiques ==========
    print(f"\n[7/7] Statistiques globales (après TRIM={TRIM}) ...")
    for tf in TFS:
        label_ = f'{tf}m' if tf < 60 else '1h'
        orc = oracles[tf]
        trimmed = orc.iloc[TRIM:-TRIM]
        n = len(trimmed)
        n_up = int((trimmed['label'] == 1).sum())
        n_down = int((trimmed['label'] == 0).sum())
        # Transitions = changements de label
        n_trans = int((trimmed['label'].diff().abs() > 0).sum())
        slope_stats = trimmed['slope'].describe()
        print(f"  [{label_}] N={n:,}  UP={n_up:,} ({n_up/n*100:.1f}%)  "
              f"DOWN={n_down:,} ({n_down/n*100:.1f}%)  Transitions={n_trans:,} "
              f"({n_trans/n*100:.2f}%)")
        print(f"         slope: mean={slope_stats['mean']:+.4f}  std={slope_stats['std']:.4f}  "
              f"min={slope_stats['min']:+.4f}  max={slope_stats['max']:+.4f}")

    # ========== Cohérence Cross-TF ==========
    print(f"\n[BONUS] Cohérence Cross-TF : signe slope 30m vs 1h sur timestamps communs ...")
    orc_30 = oracles[30].iloc[TRIM:-TRIM]
    orc_1h = oracles[60].iloc[TRIM:-TRIM]
    # Timestamps communs (les 1h tombent aussi sur les 30m aux xx:00)
    common = orc_30.index.intersection(orc_1h.index)
    if len(common) > 10:
        s30 = np.sign(orc_30.loc[common, 'slope'].values)
        s1h = np.sign(orc_1h.loc[common, 'slope'].values)
        # Ignorer les positions où un des deux est 0 (zone d'incertitude)
        mask = (s30 != 0) & (s1h != 0)
        if mask.sum() > 0:
            concord = (s30[mask] == s1h[mask]).mean() * 100
            print(f"  Concordance de signe sur {mask.sum():,} timestamps communs: {concord:.1f}%")
            print(f"  (≥70% attendu : MACD 30m et 1h capturent les mêmes tendances)")

    print("\n" + "=" * 80)
    print(f"VERDICT : {'✅ TOUS TESTS PASS' if all_ok else '❌ AU MOINS UN ÉCHEC'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
