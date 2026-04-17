#!/usr/bin/env python3
"""
Valide que MACD live 5min converge vers MACD standard TF aux closes.

Comparaison STRICTEMENT aux instants `is_close=True` :
  live[ts_5m_close]  vs  standard[ts_tf_bucket]
où ts_tf_bucket = ts_5m_close.floor(f'{tf}min').

Entre deux closes TF, le live n'est PAS comparé au standard (pas de standard
intermédiaire). Oracle (smoother non-causal) n'est PAS comparé non plus
(par construction différent).

Tests:
  [1] compute_indicator_live → Series 5min indexée, pas de NaN
  [2] Extraction aux closes TF → sous-Series de taille len(df_tf) - incomplete
  [3] Alignement timestamps : floor('TFmin') du 5m_close == index TF
  [4] Valeurs identiques (après warm-up EMA slow=26 bougies TF)
  [5] Diff = 0 à la tolérance float64

Scope actuel: MACD sur 30m et 1h.

Usage:
    python scripts/validate_live_vs_standard.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import (
    load_csv, resample_ohlcv, compute_bucket_close_mask,
    compute_indicator, compute_indicator_live,
)

DATA_DIR = Path('data/raw')
INDICATOR = 'macd'
TFS = [30, 60]
WARMUP_TF = 30  # nombre de bougies TF à écarter (EMA slow 26 + marge)
TOL_ABS = 1e-10


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
    print(f"VALIDATION {INDICATOR.upper()} live (5min) vs standard (TF aux closes)")
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
    print(f"  5m:  {len(df_5m):,} rows | {df_5m.index[0]} → {df_5m.index[-1]}")
    for tf, df in dfs_dl.items():
        lbl = f'{tf}m' if tf < 60 else '1h'
        print(f"  {lbl}: {len(df):,} rows | {df.index[0]} → {df.index[-1]}")

    # Supprimer les bougies incomplètes de bord sur les téléchargés
    for tf in TFS:
        dfs_dl[tf], n_d = drop_incomplete_last(dfs_dl[tf], df_5m, tf)
        lbl = f'{tf}m' if tf < 60 else '1h'
        print(f"  Drop incomplete {lbl}: -{n_d} → {len(dfs_dl[tf]):,} rows")

    all_ok = True
    for tf in TFS:
        lbl = f'{tf}m' if tf < 60 else '1h'
        df_tf = dfs_dl[tf]
        print(f"\n{'-' * 80}")
        print(f"  TF = {lbl}  ({tf} minutes)")
        print(f"{'-' * 80}")

        # [1] Calculer live 5min
        is_close = compute_bucket_close_mask(df_5m.index, tf)
        ind_live = compute_indicator_live(df_5m, is_close, INDICATOR, tf)
        print(f"  [1] compute_indicator_live: Series shape={ind_live.shape}, "
              f"name={ind_live.name}, no NaN: {not ind_live.isna().any()}")
        all_ok &= check(
            f"  live 5min shape = len(df_5m)",
            len(ind_live) == len(df_5m),
            f"{len(ind_live)} vs {len(df_5m)}")

        # [2] Extraire live aux closes TF (is_close=True)
        live_at_closes = ind_live[is_close]
        print(f"  [2] Extraction aux closes (is_close=True): "
              f"{len(live_at_closes):,} valeurs")
        all_ok &= check(
            f"  nb de closes live ≥ nb de bougies TF downloaded (au +/- bord)",
            abs(len(live_at_closes) - len(df_tf)) <= 2,
            f"{len(live_at_closes)} vs {len(df_tf)}")

        # [3] Alignement timestamps : floor('TFmin') du 5m_close == index TF
        #     Ex: 10:25.floor('30min') = 10:00 = index TF
        tf_str = f'{tf}min' if tf < 60 else f'{tf // 60}h'
        # Forcer aussi les minutes à xx:00 pour 1h (1h ≡ 60min)
        if tf == 60:
            # pandas floor('1h') nécessite l'alias 'h' (ou '60min')
            aligned_tf_idx = live_at_closes.index.floor('h')
        else:
            aligned_tf_idx = live_at_closes.index.floor(f'{tf}min')
        # Comparer aligned_tf_idx (dérivé de 5m) à df_tf.index
        common_tf = pd.Index(aligned_tf_idx).intersection(df_tf.index)
        print(f"  [3] Alignement: {len(common_tf):,} timestamps TF communs")
        all_ok &= check(
            f"  alignement ≥ 99% des bougies TF téléchargées",
            len(common_tf) >= int(len(df_tf) * 0.99),
            f"{len(common_tf)} / {len(df_tf)}")

        # [4] Calculer standard TF
        ind_std = compute_indicator(df_tf, INDICATOR)
        print(f"  [4] compute_indicator (standard): Series shape={ind_std.shape}, "
              f"no NaN: {not ind_std.isna().any()}")

        # [5] Aligner et comparer (après WARMUP_TF bougies)
        # Construire mapping : pour chaque ts_5m close, trouver son ts_tf correspondant
        # live_at_closes est indexé par ts_5m_close (ex: 10:25)
        # On veut le mapper à ts_tf = ts_5m_close.floor('TFmin') (ex: 10:00)
        live_df = pd.DataFrame({
            'ts_5m': live_at_closes.index,
            'ts_tf': aligned_tf_idx,
            'live_val': live_at_closes.values,
        })
        # Joindre avec ind_std sur ts_tf
        std_df = pd.DataFrame({
            'ts_tf': ind_std.index,
            'std_val': ind_std.values,
        })
        merged = live_df.merge(std_df, on='ts_tf', how='inner')

        # Tronquer le warm-up
        merged = merged.iloc[WARMUP_TF:]
        n_compared = len(merged)
        abs_diff = np.abs(merged['live_val'].values - merged['std_val'].values)
        max_diff = abs_diff.max() if n_compared > 0 else np.nan
        n_mismatch = int((abs_diff > TOL_ABS).sum())

        print(f"\n  [5] Comparaison live vs standard (après warm-up={WARMUP_TF} bougies TF)")
        print(f"      N comparées  : {n_compared:,}")
        print(f"      Max |diff|   : {max_diff:.6g}")
        print(f"      N mismatch   : {n_mismatch} (tol {TOL_ABS:.0e})")
        all_ok &= check(
            f"  live == standard aux closes TF (tolérance {TOL_ABS:.0e})",
            n_mismatch == 0,
            f"max |diff| = {max_diff:.2e}")

        if n_mismatch > 0:
            bad = merged[abs_diff > TOL_ABS].head(5)
            print(f"\n      Échantillon mismatches (premières 5):")
            for _, row in bad.iterrows():
                d = abs(row['live_val'] - row['std_val'])
                print(f"        {row['ts_5m']} (5m) → {row['ts_tf']} (TF) "
                      f"live={row['live_val']:.8f}  std={row['std_val']:.8f}  "
                      f"|diff|={d:.2e}")

    # ========== Verdict ==========
    print("\n" + "=" * 80)
    print(f"VERDICT : {'✅ TOUS TESTS PASS' if all_ok else '❌ AU MOINS UN ÉCHEC'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
