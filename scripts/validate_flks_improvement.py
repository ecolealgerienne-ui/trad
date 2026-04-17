#!/usr/bin/env python3
"""
Mesure l'amélioration FLKS vs Kalman forward pour chaque step_k.

Compare 3 niveaux d'estimation de la pente à la bougie t :
  - Forward naïf : pente = position_filt[t] - position_filt[t-1]
                   (forward Kalman seul, pas de backward ni sous-pas)
  - slope_t1    : FLKS backward sur 2 pas (sans sous-pas)
                   → Amélioration due au backward smoothing SEUL (k=0)
  - slope_k<k>  : FLKS backward + k sous-pas 5min de la bougie t+1 (k=1..5)
                   → Amélioration due aux sous-pas progressifs

Référence : oracle RTS smoother (non-causal, labels parfaits).

Métrique : concordance de signe (% de sign(slope) == sign(oracle)).

Usage :
    python scripts/validate_flks_improvement.py
    python scripts/validate_flks_improvement.py --indicator macd --tf 30 --days 180
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
    compute_forward_filter,
    compute_flks_slopes,
    compute_oracle_labels,
)

SRC_5M = Path('data_trad/BTCUSD_all_5m.csv')


def concordance(slope, oracle):
    """% de sign(slope) == sign(oracle), en excluant les zéros."""
    mask = (slope != 0) & (oracle != 0) & ~np.isnan(slope) & ~np.isnan(oracle)
    if mask.sum() == 0:
        return 0.0, 0
    matches = (np.sign(slope[mask]) == np.sign(oracle[mask]))
    return matches.mean() * 100, mask.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indicator', default='macd',
                        choices=['macd', 'rsi', 'cci'])
    parser.add_argument('--tf', type=int, default=30, choices=[30, 60])
    parser.add_argument('--days', type=int, default=0,
                        help='Derniers N jours (0 = tout)')
    parser.add_argument('--trim', type=int, default=100,
                        help='Bougies TF à retirer début ET fin')
    args = parser.parse_args()

    print("=" * 80)
    print(f"AMÉLIORATION FLKS vs FORWARD — {args.indicator.upper()} × {args.tf}m")
    print("=" * 80)

    # Load data
    print(f"\n[Load 5m] {SRC_5M}")
    df_5m_full = load_csv(SRC_5M)
    if args.days > 0:
        end = df_5m_full.index[-1]
        start = end - pd.Timedelta(days=args.days)
        df_5m = df_5m_full.loc[df_5m_full.index >= start].copy()
        print(f"  Filter {args.days}j: {len(df_5m):,} rows  |  "
              f"{df_5m.index[0]} → {df_5m.index[-1]}")
    else:
        df_5m = df_5m_full
        print(f"  Full: {len(df_5m):,} rows  |  "
              f"{df_5m.index[0]} → {df_5m.index[-1]}")

    df_tf = resample_ohlcv(df_5m, args.tf)
    n_sub = args.tf // 5
    print(f"  {len(df_tf):,} bougies {args.tf}m  "
          f"(n_substeps_per_candle={n_sub})")

    # 1) Forward filter (Kalman standard, sans backward)
    print(f"\n[1] Forward filter Kalman standard ...")
    fwd = compute_forward_filter(df_tf, args.indicator, adaptive=False)
    pos_forward = fwd['state']['position'].values
    # Pente forward naïve = diff de positions filtrées (2 pas)
    slope_forward = np.zeros_like(pos_forward)
    slope_forward[1:] = pos_forward[1:] - pos_forward[:-1]

    # 2) FLKS (slope_t1 + slope_k1..slope_k<n_sub-1>)
    print(f"[2] FLKS (slope_t1 + slope_k1..slope_k{n_sub-1}) ...")
    flks = compute_flks_slopes(df_tf, df_5m, args.indicator, args.tf,
                                 k_range=(1, n_sub - 1))
    # Colonnes : slope_t1, slope_k1..slope_k<n_sub-1>

    # 3) Oracle (référence)
    print(f"[3] Oracle RTS smoother ...")
    oracle = compute_oracle_labels(df_tf, args.indicator)
    oracle_slope = oracle['slope'].values

    # Appliquer trim
    start_i = args.trim
    end_i = len(df_tf) - args.trim
    sl = slice(start_i, end_i)
    slope_forward_t = slope_forward[sl]
    oracle_t = oracle_slope[sl]
    flks_t = flks.iloc[sl]

    # 4) Concordance
    print(f"\n[4] Concordance (% sign match vs oracle) sur "
          f"[{start_i}, {end_i}) = {end_i - start_i:,} bougies {args.tf}m")
    print(f"    (échantillons non-nuls où signe défini)")

    print("\n" + "=" * 80)
    print(f"{'Variante':<30} {'Concordance':>14} {'N':>10} {'Δ vs Forward':>16}")
    print("-" * 80)

    # Baseline
    conc_fwd, n_fwd = concordance(slope_forward_t, oracle_t)
    print(f"{'Forward naïf (Kalman seul)':<30} {conc_fwd:>13.2f}% {n_fwd:>10,} "
          f"{'(baseline)':>16}")

    # slope_t1 (backward sans sous-pas)
    conc_t1, n_t1 = concordance(flks_t['slope_t1'].values, oracle_t)
    delta_t1 = conc_t1 - conc_fwd
    print(f"{'slope_t1 (backward, k=0)':<30} {conc_t1:>13.2f}% {n_t1:>10,} "
          f"{delta_t1:>+15.2f}%")

    # slope_k1..slope_k<n_sub-1>
    for k in range(1, n_sub):
        col = f'slope_k{k}'
        conc_k, n_k = concordance(flks_t[col].values, oracle_t)
        delta_k = conc_k - conc_fwd
        delta_vs_t1 = conc_k - conc_t1
        print(f"{col + ' (backward + ' + str(k) + ' sub)':<30} "
              f"{conc_k:>13.2f}% {n_k:>10,} "
              f"{delta_k:>+15.2f}%  "
              f"(vs t1: {delta_vs_t1:+.2f}%)")

    print("=" * 80)

    # 5) Résumé amélioration relative
    print(f"\nRÉSUMÉ — Amélioration relative vs Forward naïf :")
    print(f"  step_k=0 (slope_t1) : {conc_t1 - conc_fwd:+.2f}%  "
          f"→ gain backward smoothing pur")
    for k in range(1, n_sub):
        col = f'slope_k{k}'
        conc_k, _ = concordance(flks_t[col].values, oracle_t)
        print(f"  step_k={k} ({col}) : {conc_k - conc_fwd:+.2f}%  "
              f"→ backward + {k} sous-pas 5min")


if __name__ == '__main__':
    main()
