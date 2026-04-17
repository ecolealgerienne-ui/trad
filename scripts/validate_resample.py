#!/usr/bin/env python3
"""
Valide le resample 5m -> 30m et 5m -> 1h contre les fichiers téléchargés.

Pour chaque bougie 30m (resp. 1h) du fichier téléchargé:
  - Ne comparer QUE si les 6 bougies 5m (resp. 12) qui la composent sont toutes
    présentes dans le fichier 5m.
  - Open = first 5m open, High = max, Low = min, Close = last, Volume = sum.

OHLC doit être strictement égal. Volume peut différer très légèrement
(arrondi API Binance). On reporte les écarts.

Usage:
    python scripts/validate_resample.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import resample_ohlcv, load_csv


DATA_DIR = Path('data/raw')


def compare_candles(df_resampled, df_downloaded, tf_label, tol_ohlc=1e-8, tol_vol_rel=1e-4):
    """
    Compare deux DataFrames OHLCV par timestamp commun.
    Retourne un rapport des écarts.
    """
    # Intersection des timestamps
    common = df_resampled.index.intersection(df_downloaded.index)
    n_common = len(common)

    if n_common == 0:
        print(f"  ⚠️  Aucun timestamp commun.")
        return

    rs = df_resampled.loc[common]
    dl = df_downloaded.loc[common]

    # Différences par colonne
    diffs = {}
    for col in ['open', 'high', 'low', 'close', 'volume']:
        d = (rs[col] - dl[col]).abs()
        if col == 'volume':
            # Relative diff
            rel = d / dl[col].abs().replace(0, np.nan)
            diffs[col] = {
                'max_abs': d.max(),
                'max_rel': rel.max() if not rel.isna().all() else 0.0,
                'n_nonzero': (d > tol_vol_rel * dl[col].abs()).sum(),
            }
        else:
            diffs[col] = {
                'max_abs': d.max(),
                'max_rel': 0.0,
                'n_nonzero': (d > tol_ohlc).sum(),
            }

    # Rapport
    print(f"\n  [{tf_label}] Comparaison sur {n_common:,} bougies communes")
    print(f"  {'Col':<10} {'Max |diff|':>15} {'Max rel':>12} {'N > tol':>10}")
    print(f"  {'-' * 52}")
    all_ohlc_ok = True
    for col, d in diffs.items():
        status = ""
        if col == 'volume':
            if d['n_nonzero'] > 0:
                status = f"({d['n_nonzero']} > {tol_vol_rel:.0e} rel)"
        else:
            if d['n_nonzero'] > 0:
                status = f"❌ {d['n_nonzero']} mismatches"
                all_ohlc_ok = False
            else:
                status = "✅ exact"
        print(f"  {col:<10} {d['max_abs']:>15.6g} {d['max_rel']:>12.2e}  {status}")

    # Si mismatch OHLC, afficher les 5 premières divergences
    if not all_ohlc_ok:
        print(f"\n  [{tf_label}] ÉCHANTILLON des mismatches OHLC:")
        for col in ['open', 'high', 'low', 'close']:
            d_col = (rs[col] - dl[col]).abs()
            bad = d_col[d_col > tol_ohlc]
            if len(bad) > 0:
                print(f"    {col}: {len(bad)} lignes. Premières 3:")
                for ts in bad.index[:3]:
                    print(f"      {ts}  rs={rs.loc[ts, col]:.4f}  dl={dl.loc[ts, col]:.4f}  diff={d_col.loc[ts]:.6g}")

    return all_ohlc_ok, n_common


def filter_complete_buckets(df_5m, df_tf, tf_minutes):
    """
    Ne garde dans df_tf que les bougies dont TOUTES les 5min qui les composent
    sont présentes dans df_5m.

    Pour tf_minutes=30 (resp. 60), on attend 6 (resp. 12) bougies 5min par
    bougie de plus haut timeframe.
    """
    expected = tf_minutes // 5
    # Pour chaque timestamp de df_tf, compter les 5min qui tombent dedans
    complete = []
    for ts in df_tf.index:
        start = ts
        end = ts + pd.Timedelta(minutes=tf_minutes)
        mask = (df_5m.index >= start) & (df_5m.index < end)
        n = mask.sum()
        if n == expected:
            complete.append(ts)
    return df_tf.loc[complete]


def main():
    print("=" * 80)
    print("VALIDATION RESAMPLE 5m -> 30m et 5m -> 1h")
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
    print("✅ Les 3 fichiers sont présents.")

    print("\nChargement ...")
    df_5m = load_csv(paths['5m'])
    df_30m_dl = load_csv(paths['30m'])
    df_1h_dl = load_csv(paths['1h'])
    print(f"  5m:  {len(df_5m):,} rows | {df_5m.index[0]} → {df_5m.index[-1]}")
    print(f"  30m: {len(df_30m_dl):,} rows | {df_30m_dl.index[0]} → {df_30m_dl.index[-1]}")
    print(f"  1h:  {len(df_1h_dl):,} rows | {df_1h_dl.index[0]} → {df_1h_dl.index[-1]}")

    # ------- Resample 5m -> 30m -------
    print("\n[1/2] Resample 5m -> 30m ...")
    df_30m_rs = resample_ohlcv(df_5m, 30)
    # Filtrer : ne comparer que les bougies 30m dont les 6 bougies 5m sont toutes présentes
    df_30m_dl_complete = filter_complete_buckets(df_5m, df_30m_dl, 30)
    print(f"  Téléchargées:       {len(df_30m_dl):,} bougies 30m")
    print(f"  Complètes (6×5m):   {len(df_30m_dl_complete):,}")
    print(f"  Resample 5m->30m:   {len(df_30m_rs):,}")
    ok_30m, n_30m = compare_candles(df_30m_rs, df_30m_dl_complete, '30m')

    # ------- Resample 5m -> 1h -------
    print("\n[2/2] Resample 5m -> 1h ...")
    df_1h_rs = resample_ohlcv(df_5m, 60)
    df_1h_dl_complete = filter_complete_buckets(df_5m, df_1h_dl, 60)
    print(f"  Téléchargées:       {len(df_1h_dl):,} bougies 1h")
    print(f"  Complètes (12×5m):  {len(df_1h_dl_complete):,}")
    print(f"  Resample 5m->1h:    {len(df_1h_rs):,}")
    ok_1h, n_1h = compare_candles(df_1h_rs, df_1h_dl_complete, '1h')

    # ------- Verdict -------
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    print(f"30m: {'✅ PASS' if ok_30m else '❌ FAIL'} ({n_30m:,} bougies comparées)")
    print(f"1h:  {'✅ PASS' if ok_1h else '❌ FAIL'} ({n_1h:,} bougies comparées)")

    if ok_30m and ok_1h:
        print("\n✅ Resample validé : resample_ohlcv(5m, N) == fichier Nm téléchargé (OHLC exact)")
    else:
        print("\n❌ Au moins une divergence. Voir détails ci-dessus.")


if __name__ == '__main__':
    main()
