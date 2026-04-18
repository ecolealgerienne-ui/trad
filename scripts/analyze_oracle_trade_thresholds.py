#!/usr/bin/env python3
"""
Analyse la distribution des PnL des trades Oracle pour aider à choisir le
seuil d'Option A (sample weighting).

Idée : certains trades Oracle sont très profitables, d'autres marginaux
(PnL proche de 0) ou légèrement perdants à cause des fees. En pondérant
à 0 les trades neutres dans le training, le modèle apprend mieux.

Ce script explore plusieurs seuils ABSOLUS sur |pnl_net| et mesure :
  - Combien de trades Oracle on garde (>= seuil)
  - Combien de rows 5min sont couvertes (pour sample_weight)
  - Impact PnL moyen, WR, distribution
  - Base rate pour le training corrigé

Traite les 3 splits (train, val, test) pour vue d'ensemble.

Usage :
    python scripts/analyze_oracle_trade_thresholds.py \\
        --npz data/prepared/dataset_rsi_30m_full_progressive_lag0.npz

    # Avec grid personnalisé
    python scripts/analyze_oracle_trade_thresholds.py --npz ... \\
        --thresholds 0.0 0.001 0.002 0.003 0.005 0.01 0.02
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import extract_trades_5min_progressive


def analyze_split(split_name, y_cont, closes, dates, fees, thresholds):
    """
    Analyse les trades Oracle d'un split et l'impact des seuils.

    Returns :
        dict avec 'trades', 'pnl_per_row', 'results_per_threshold'
    """
    print(f"\n{'=' * 100}")
    print(f"SPLIT : {split_name.upper()}  ({len(closes):,} rows  "
          f"{dates[0]} → {dates[-1]})")
    print(f"{'=' * 100}")

    # 1. Extraire les trades Oracle
    trades = extract_trades_5min_progressive(y_cont, closes, fees=fees)
    n_trades = len(trades)
    print(f"\n[1] Trades Oracle extraits : {n_trades:,}")

    if n_trades == 0:
        print("   Aucun trade, skip")
        return None

    df_trades = pd.DataFrame(trades)

    # 2. Stats globales
    n_winners = int((df_trades['pnl_net'] > 0).sum())
    n_losers = int((df_trades['pnl_net'] < 0).sum())
    wr = n_winners / n_trades * 100
    pnl_total = df_trades['pnl_net'].sum() * 100
    pnl_mean = df_trades['pnl_net'].mean() * 100
    print(f"   Winners : {n_winners:,} ({wr:.2f}%)")
    print(f"   Losers  : {n_losers:,} ({100-wr:.2f}%)")
    print(f"   PnL Net total : {pnl_total:+.2f}%")
    print(f"   PnL Net mean  : {pnl_mean:+.4f}%/trade")

    # 3. Distribution PnL par trade
    print(f"\n[2] Distribution PnL Net par trade (%) :")
    pnl_pct = df_trades['pnl_net'].values * 100
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(pnl_pct, p)
        print(f"     P{p:>2}  : {val:+.4f}%")
    print(f"     Min  : {pnl_pct.min():+.4f}%")
    print(f"     Max  : {pnl_pct.max():+.4f}%")

    # Distribution absolue |pnl_net|
    print(f"\n[3] Distribution |PnL Net| par trade (%) :")
    abs_pnl_pct = np.abs(pnl_pct)
    for p in [50, 75, 90, 95, 99]:
        val = np.percentile(abs_pnl_pct, p)
        print(f"     P{p} : {val:.4f}%")

    # 4. Mapper chaque row 5min à son trade
    n_rows = len(closes)
    row_to_trade_idx = np.full(n_rows, -1, dtype=np.int64)
    for idx, t in enumerate(trades):
        # Entry = entry_i (row 5min de l'exécution d'entrée)
        # Exit = exit_i (row 5min de l'exécution de sortie)
        # La position est ouverte entre entry_i et exit_i (inclusif)
        row_to_trade_idx[t['entry_i']:t['exit_i'] + 1] = idx

    # PnL par row (du trade qui la contient)
    pnl_per_row = np.zeros(n_rows)
    for i in range(n_rows):
        tid = row_to_trade_idx[i]
        if tid >= 0:
            pnl_per_row[i] = trades[tid]['pnl_net']

    # Rows hors trade (pas de position active) : pnl_per_row = 0
    n_rows_in_trade = int((row_to_trade_idx >= 0).sum())
    n_rows_no_trade = n_rows - n_rows_in_trade
    print(f"\n[4] Rows 5min dans un trade : {n_rows_in_trade:,} / {n_rows:,}  "
          f"({n_rows_in_trade/n_rows*100:.2f}%)")
    print(f"   Rows hors trade : {n_rows_no_trade:,}  "
          f"({n_rows_no_trade/n_rows*100:.2f}%)")

    # 5. Grid de seuils
    print(f"\n[5] Impact de seuils ABSOLU |pnl_net| >= seuil :")
    print(f"   {'Seuil %':>8}  "
          f"{'Trades gardés':>15}  {'% trades':>10}  "
          f"{'Rows gardées':>14}  {'% rows':>10}  "
          f"{'PnL mean keep':>15}  {'WR keep':>10}  "
          f"{'PnL mean drop':>15}")
    print(f"   {'-' * 110}")

    results_per_threshold = []
    for thr in thresholds:
        kept_mask = abs_pnl_pct >= thr * 100  # thr en fraction, pnl_pct en %
        n_kept = int(kept_mask.sum())
        n_drop = n_trades - n_kept
        pct_trades = n_kept / n_trades * 100

        # Rows couvertes
        kept_row_mask = np.zeros(n_rows, dtype=bool)
        for idx, t in enumerate(trades):
            if kept_mask[idx]:
                kept_row_mask[t['entry_i']:t['exit_i'] + 1] = True
        n_rows_kept = int(kept_row_mask.sum())
        pct_rows = n_rows_kept / n_rows * 100

        if n_kept > 0:
            pnl_mean_kept = df_trades.loc[kept_mask, 'pnl_net'].mean() * 100
            wr_kept = ((df_trades.loc[kept_mask, 'pnl_net'] > 0).sum()
                       / n_kept * 100)
        else:
            pnl_mean_kept = 0
            wr_kept = 0

        if n_drop > 0:
            pnl_mean_drop = df_trades.loc[~kept_mask, 'pnl_net'].mean() * 100
        else:
            pnl_mean_drop = 0

        print(f"   {thr*100:>7.3f}%  "
              f"{n_kept:>15,}  {pct_trades:>9.2f}%  "
              f"{n_rows_kept:>14,}  {pct_rows:>9.2f}%  "
              f"{pnl_mean_kept:>+14.4f}%  {wr_kept:>9.2f}%  "
              f"{pnl_mean_drop:>+14.4f}%")

        results_per_threshold.append({
            'threshold': thr,
            'n_trades_kept': n_kept,
            'pct_trades_kept': pct_trades,
            'n_rows_kept': n_rows_kept,
            'pct_rows_kept': pct_rows,
            'pnl_mean_kept': pnl_mean_kept,
            'wr_kept': wr_kept,
            'pnl_mean_drop': pnl_mean_drop,
        })

    return {
        'n_trades': n_trades,
        'wr_global': wr,
        'pnl_total': pnl_total,
        'pnl_mean': pnl_mean,
        'trades_df': df_trades,
        'pnl_per_row': pnl_per_row,
        'row_to_trade_idx': row_to_trade_idx,
        'results': results_per_threshold,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', required=True,
                        help='Path NPZ progressif (tout suffixe)')
    parser.add_argument('--splits', nargs='+',
                        default=['train', 'val', 'test'],
                        help='Splits à analyser (default: all)')
    parser.add_argument('--fees', type=float, default=0.001)
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=[0.0, 0.0005, 0.001, 0.002, 0.003, 0.005,
                                 0.0075, 0.01, 0.015, 0.02],
                        help='Seuils (fraction) |pnl_net| à tester')
    parser.add_argument('--save-pnl-per-row', action='store_true',
                        help='Sauver pnl_per_row pour usage downstream (Option A)')
    parser.add_argument('--output-dir', default='results/oracle_analysis')
    args = parser.parse_args()

    print("=" * 100)
    print(f"ANALYSE SEUILS ORACLE — {args.npz}")
    print(f"  fees={args.fees*100:.2f}%  splits={args.splits}")
    print(f"  thresholds: {args.thresholds}")
    print("=" * 100)

    npz_path = Path(args.npz)
    if not npz_path.exists():
        print(f"❌ NPZ introuvable: {npz_path}")
        return

    ds = np.load(npz_path, allow_pickle=True)

    per_split_results = {}
    per_split_pnl_per_row = {}
    for split in args.splits:
        closes = ds[f'closes_{split}']
        dates = pd.to_datetime(ds[f'dates_{split}'])
        y_cont = ds[f'y_{split}_continuous']
        res = analyze_split(split, y_cont, closes, dates, args.fees,
                             args.thresholds)
        if res is not None:
            per_split_results[split] = res
            per_split_pnl_per_row[split] = res['pnl_per_row']

    # Récap synthétique tableau
    if len(per_split_results) >= 2:
        print(f"\n{'=' * 100}")
        print(f"SYNTHÈSE INTER-SPLITS — % trades gardés par seuil")
        print(f"{'=' * 100}")
        header = f"   {'Seuil %':>8}  "
        for split in per_split_results:
            header += f"{split:>13}  "
        print(header)
        print(f"   {'-' * (8 + 16 * len(per_split_results))}")
        for i, thr in enumerate(args.thresholds):
            line = f"   {thr*100:>7.3f}%  "
            for split in per_split_results:
                pct = per_split_results[split]['results'][i]['pct_trades_kept']
                line += f"{pct:>12.2f}%  "
            print(line)

    # Sauvegarde pnl_per_row pour Option A
    if args.save_pnl_per_row and per_split_pnl_per_row:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = npz_path.stem.replace('dataset_', '')
        out_path = out_dir / f'pnl_per_row_{tag}.npz'
        save_dict = {}
        for split, pnl in per_split_pnl_per_row.items():
            save_dict[f'pnl_per_row_{split}'] = pnl.astype(np.float64)
        np.savez(out_path, **save_dict)
        print(f"\n✅ pnl_per_row sauvé : {out_path}  "
              f"({out_path.stat().st_size / 1024:.1f} KB)")
        print(f"   Utilisable pour Option A (sample weighting) en training")

    # Recommandations
    print(f"\n{'=' * 100}")
    print(f"RECOMMANDATION POUR OPTION A (sample weighting)")
    print(f"{'=' * 100}")
    print("""
  Pour implémenter Option A :
    1. Choisir un seuil où :
       - % rows gardées entre 50-80% du total train (garde assez de samples)
       - PnL mean_kept significativement > PnL mean_drop
       - WR_kept > WR_global (signe que le seuil filtre les mauvais)

    2. En training :
       - sample_weight[i] = 1 si |pnl_per_row[i]| >= seuil
       - sample_weight[i] = 0 sinon
       - XGBoost : fit(X, y, sample_weight=w)
       - CNN-LSTM : pondérer la loss row par row

  Ensuite re-train + re-backtest pour voir si PnL s'améliore.
""")


if __name__ == '__main__':
    main()
