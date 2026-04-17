#!/usr/bin/env python3
"""
Prépare le fichier 30m depuis le CSV historique 5min complet.

Input:  data_trad/BTCUSD_all_5m.csv (8.5 ans BTC, ~879k lignes)
Output: data/raw/BTCUSD_full_30m.csv

Aucun alignement xx:00 (on ne compare avec aucun 30m téléchargé).
Le resample utilise resample_ohlcv du core (validé bit-à-bit).

Usage:
    python scripts/prepare_full_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import load_csv, resample_ohlcv


SRC_5M = Path('data_trad/BTCUSD_all_5m.csv')
DST_DIR = Path('data/raw')
DST_30M = DST_DIR / 'BTCUSD_full_30m.csv'


def main():
    print("=" * 80)
    print("PRÉPARATION CSV HISTORIQUE FULL — 5m → 30m")
    print("=" * 80)

    if not SRC_5M.exists():
        print(f"❌ Fichier 5m introuvable: {SRC_5M}")
        return

    DST_DIR.mkdir(parents=True, exist_ok=True)

    # [1] Load 5m
    print(f"\n[1/3] Chargement 5m ...")
    df_5m = load_csv(SRC_5M)
    print(f"  {len(df_5m):,} rows  |  {df_5m.index[0]} → {df_5m.index[-1]}")
    years = (df_5m.index[-1] - df_5m.index[0]).total_seconds() / (365.25 * 24 * 3600)
    print(f"  ≈ {years:.1f} années de données")

    # [2] Resample → 30m
    print(f"\n[2/3] Resample 5m → 30m ...")
    df_30m = resample_ohlcv(df_5m, 30)
    print(f"  {len(df_30m):,} rows 30m  |  {df_30m.index[0]} → {df_30m.index[-1]}")

    # Sanity check : ratio ~6×
    ratio = len(df_5m) / len(df_30m)
    print(f"  Ratio 5m/30m = {ratio:.2f} (attendu ~6.0)")

    # [3] Save
    print(f"\n[3/3] Sauvegarde ...")
    df_30m_save = df_30m.reset_index()
    df_30m_save.to_csv(DST_30M, index=False)
    print(f"  ✅ Sauvé: {DST_30M}")
    print(f"     ({DST_30M.stat().st_size / 1024 / 1024:.1f} MB)")

    # Info sur le fichier 5m (pas copié, on pointe directement vers data_trad/)
    print(f"\nNote: le 5m reste dans {SRC_5M} (pas de copie).")
    print(f"      Les scripts train/backtest le liront à cet emplacement.")


if __name__ == '__main__':
    main()
