#!/usr/bin/env python3
"""
Backtest à résolution 5min d'un dataset "progressive".

Signal = slope_progressive à chaque ligne 5min → sign(slope) = UP/DOWN/FLAT
Execution à close_5m[i+1] (lag 1 tick 5min, réaliste : on ne trade pas à xx:00/xx:30)

Calcule TOUJOURS l'Oracle (borne supérieure) et, si --preds est fourni, ajoute
le Model en comparaison côte à côte. Cela permet de vérifier que l'algorithme
de backtest est correct (Oracle reproductible) en même temps qu'on évalue le
modèle.

Usage :
    # Oracle seul
    python scripts/backtest_progressive.py --npz data/prepared/dataset_macd_30m_full_progressive.npz --split test

    # Oracle + Model (comparaison)
    python scripts/backtest_progressive.py --npz ... --preds data/prepared/preds_..._progressive.npz --split test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import backtest_5min_progressive


def buy_and_hold_5m(closes):
    """B&H simple : rendement entre premier et dernier close."""
    first = closes[0]
    last = closes[-1]
    if np.isnan(first) or np.isnan(last) or first == 0:
        return 0.0
    return (last - first) / first * 100


def run_backtest(label, slopes, closes, fees):
    """Backtest une série de slopes et retourne le dict résultat."""
    res = backtest_5min_progressive(slopes, closes, fees=fees)
    res['label'] = label
    res['fees_total_pct'] = res['n_trades'] * 2 * fees * 100
    res['pnl_gross_pct'] = res['pnl_pct'] + res['fees_total_pct']
    return res


def print_row(name, r, bh, col_w=14):
    """Affiche une ligne de stats."""
    alpha = r['pnl_pct'] - bh
    print(f"  {name:<20}"
          f"{r['n_trades']:>{col_w},}"
          f"{r['win_rate']:>{col_w-2}.1f}% "
          f"{r['profit_factor']:>{col_w-1}.2f}"
          f"{r['sharpe']:>{col_w-1}.3f}"
          f"{r['pnl_gross_pct']:>+{col_w-1}.2f}%"
          f"{r['fees_total_pct']:>{col_w-1}.2f}%"
          f"{r['pnl_pct']:>+{col_w-1}.2f}%"
          f"{alpha:>+{col_w-1}.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', required=True,
                        help='Path vers dataset_..._progressive.npz')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--preds', default=None,
                        help='Path vers preds NPZ (si fourni → compare Oracle vs Model)')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--fees', type=float, default=0.001)
    args = parser.parse_args()

    print("=" * 100)
    print(f"BACKTEST PROGRESSIVE — split={args.split}  fees={args.fees*100:.2f}%")
    print("=" * 100)

    # Load NPZ
    npz_path = Path(args.npz)
    if not npz_path.exists():
        print(f"❌ NPZ introuvable: {npz_path}")
        return
    ds = np.load(npz_path, allow_pickle=True)
    print(f"\n✅ NPZ chargé: {npz_path}")

    closes = ds[f'closes_{args.split}']
    dates = pd.to_datetime(ds[f'dates_{args.split}'])
    y_cont = ds[f'y_{args.split}_continuous']
    period_days = (dates[-1] - dates[0]).total_seconds() / 86400
    print(f"   Split {args.split}: {len(closes):,} rows  |  "
          f"{dates[0]} → {dates[-1]}  ({period_days:.0f} jours)")

    # Backtest Oracle (toujours calculé)
    print(f"\n→ Backtest ORACLE (signal = y_{args.split}_continuous)")
    slopes_oracle = y_cont
    res_oracle = run_backtest('ORACLE', slopes_oracle, closes, args.fees)

    # Backtest Model (optionnel)
    res_model = None
    if args.preds:
        preds_path = Path(args.preds)
        if not preds_path.exists():
            print(f"❌ Preds NPZ introuvable: {preds_path}")
            return
        preds = np.load(preds_path, allow_pickle=True)
        p = preds[f'{args.split}_preds_proba']
        slopes_model = np.where(p > args.threshold, 1.0, -1.0)
        print(f"→ Backtest MODEL (threshold={args.threshold}, "
              f"UP={(slopes_model>0).sum():,} DOWN={(slopes_model<0).sum():,})")
        res_model = run_backtest(f'MODEL t={args.threshold}',
                                   slopes_model, closes, args.fees)

    # B&H
    bh = buy_and_hold_5m(closes)

    # Tableau comparatif
    print(f"\n{'=' * 100}")
    print(f"RÉSULTATS — {args.split}  ({period_days:.0f} jours)")
    print(f"{'=' * 100}")
    print(f"  {'Stratégie':<20}{'Trades':>14}{'WinRate':>14}{'PF':>13}"
          f"{'Sharpe':>13}{'PnL Brut':>14}{'Fees':>13}{'PnL Net':>14}{'Alpha B&H':>14}")
    print(f"  {'-' * 96}")
    print_row('Oracle', res_oracle, bh)
    if res_model is not None:
        print_row(f'Model (t={args.threshold})', res_model, bh)
    print(f"  {'-' * 96}")
    print(f"  {'Buy & Hold':<20}{'—':>14}{'—':>14}{'—':>13}{'—':>13}{bh:>+13.2f}%")

    # Comparaison Model vs Oracle
    if res_model is not None:
        print(f"\n{'=' * 100}")
        print("COMPARAISON MODEL vs ORACLE")
        print(f"{'=' * 100}")
        capture = (res_model['pnl_pct'] / res_oracle['pnl_pct'] * 100
                   if res_oracle['pnl_pct'] != 0 else 0)
        excess_trades = res_model['n_trades'] - res_oracle['n_trades']
        print(f"  Capture PnL Net    : {capture:+.1f}%  "
              f"(model {res_model['pnl_pct']:+.2f}% vs oracle {res_oracle['pnl_pct']:+.2f}%)")
        print(f"  Trades en plus     : {excess_trades:+,}  "
              f"(model {res_model['n_trades']:,} vs oracle {res_oracle['n_trades']:,})")
        print(f"  Fees supplémentaires: {res_model['fees_total_pct'] - res_oracle['fees_total_pct']:+.2f}%")
        print(f"  Δ PnL Brut         : {res_model['pnl_gross_pct'] - res_oracle['pnl_gross_pct']:+.2f}%")


if __name__ == '__main__':
    main()
