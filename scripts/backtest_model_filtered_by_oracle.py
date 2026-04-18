#!/usr/bin/env python3
"""
DIAGNOSTIC : backtest du modèle filtré par l'oracle.

À chaque row 5min :
  - signal_model  = sign(proba - 0.5) → ±1
  - signal_oracle = sign(y_test_continuous) → ±1
  - Si signal_model == signal_oracle : trade (slope = signal)
  - Sinon                             : conserve position (slope = 0)

⚠️ NON UTILISABLE EN PRODUCTION : requiert l'oracle en temps réel.
C'est un outil de diagnostic pour isoler :
  - Si PnL remonte vers Oracle → problème = SWITCH (flips parasites)
  - Si PnL reste négatif       → problème = TIMING/LAG

Affiche 3 stratégies côte à côte :
  - Oracle pur (référence)
  - Model pur (threshold 0.5, baseline)
  - Model ∩ Oracle (diagnostic)

Usage :
    python scripts/backtest_model_filtered_by_oracle.py --npz ... --preds ...
    python scripts/backtest_model_filtered_by_oracle.py --npz ... --preds ... --split test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import (
    backtest_5min_progressive,
    backtest_5min_filtered_by_oracle,
)


def buy_and_hold_5m(closes):
    first, last = closes[0], closes[-1]
    if np.isnan(first) or np.isnan(last) or first == 0:
        return 0.0
    return (last - first) / first * 100


def print_row(name, r, bh, fees):
    """Affiche une ligne de stats."""
    fees_pct = r['n_trades'] * 2 * fees * 100
    pnl_gross = r['pnl_pct'] + fees_pct
    alpha = r['pnl_pct'] - bh
    if 'n_filtered' in r:
        extra = f" {r['n_filtered']:>7,}"
    else:
        extra = f" {'—':>7}"
    print(f"  {name:<28}"
          f"{r['n_trades']:>8,} "
          f"{r['win_rate']:>6.1f}% "
          f"{r['profit_factor']:>6.2f} "
          f"{r['sharpe']:>7.3f} "
          f"{pnl_gross:>+9.2f}% "
          f"{fees_pct:>9.2f}% "
          f"{r['pnl_pct']:>+10.2f}% "
          f"{alpha:>+10.2f}%"
          f"{extra}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', required=True,
                        help='Path vers dataset_..._progressive.npz')
    parser.add_argument('--preds', required=True,
                        help='Path vers preds NPZ (XGBoost ou CNN-LSTM)')
    parser.add_argument('--split', default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--fees', type=float, default=0.001)
    args = parser.parse_args()

    print("=" * 110)
    print(f"DIAGNOSTIC : Model ∩ Oracle  —  split={args.split}  "
          f"fees={args.fees*100:.2f}%  threshold={args.threshold}")
    print("=" * 110)

    # Load
    ds = np.load(args.npz, allow_pickle=True)
    preds = np.load(args.preds, allow_pickle=True)

    closes = ds[f'closes_{args.split}']
    dates = pd.to_datetime(ds[f'dates_{args.split}'])
    y_cont = ds[f'y_{args.split}_continuous']  # signal oracle brut (pente)
    p = preds[f'{args.split}_preds_proba']

    period_days = (dates[-1] - dates[0]).total_seconds() / 86400
    print(f"\n{args.split}: {len(closes):,} rows  |  "
          f"{dates[0]} → {dates[-1]}  ({period_days:.0f} jours)")

    # Signaux
    slopes_oracle = y_cont
    slopes_model = np.where(p > args.threshold, 1.0, -1.0)

    # 3 backtests
    print(f"\n[1] Oracle pur (référence, fonction backtest_5min_progressive)")
    r_oracle = backtest_5min_progressive(slopes_oracle, closes, fees=args.fees)

    print(f"[2] Model pur (threshold={args.threshold}, même fonction)")
    r_model = backtest_5min_progressive(slopes_model, closes, fees=args.fees)

    print(f"[3] Model ∩ Oracle (diagnostic, fonction backtest_5min_filtered_by_oracle)")
    r_filtered = backtest_5min_filtered_by_oracle(
        slopes_model, slopes_oracle, closes, fees=args.fees)

    bh = buy_and_hold_5m(closes)

    # Diagnostic désaccord model vs oracle
    sign_model = np.sign(slopes_model)
    sign_oracle = np.sign(slopes_oracle)
    mask = (sign_oracle != 0)
    agreement_rate = (sign_model[mask] == sign_oracle[mask]).mean() * 100
    print(f"\nAccord model vs oracle (sign identique, hors oracle=0): "
          f"{agreement_rate:.2f}% ({mask.sum():,} rows oracle ≠ 0)")

    # Tableau
    print(f"\n{'=' * 110}")
    print(f"RÉSULTATS — {args.split}  ({period_days:.0f} jours)")
    print(f"{'=' * 110}")
    header = (f"  {'Stratégie':<28}"
              f"{'Trades':>8} {'WR':>7} {'PF':>6} {'Sharpe':>8}"
              f"{'Brut':>11}{'Fees':>10}{'Net':>11}{'αB&H':>11} {'Filtr':>7}")
    print(header)
    print(f"  {'-' * 106}")
    print_row('Oracle pur', r_oracle, bh, args.fees)
    print_row(f'Model pur (t={args.threshold})', r_model, bh, args.fees)
    print_row('Model ∩ Oracle', r_filtered, bh, args.fees)
    print(f"  {'-' * 106}")
    print(f"  {'Buy & Hold':<28}{'—':>8} {'—':>7} {'—':>6} {'—':>8}"
          f"{'—':>11}{'—':>10}{bh:>+10.2f}%")

    # Diagnostic final
    print(f"\n{'=' * 110}")
    print("DIAGNOSTIC")
    print(f"{'=' * 110}")
    print(f"  Model filtré: {r_filtered['n_trades']:,} trades  "
          f"({r_filtered['n_filtered']:,} trades bloqués par désaccord oracle)")
    print(f"  PnL Oracle pur     : {r_oracle['pnl_pct']:+10.2f}%")
    print(f"  PnL Model pur      : {r_model['pnl_pct']:+10.2f}%")
    print(f"  PnL Model ∩ Oracle : {r_filtered['pnl_pct']:+10.2f}%")

    capture_filtered = (r_filtered['pnl_pct'] / r_oracle['pnl_pct'] * 100
                        if r_oracle['pnl_pct'] != 0 else 0)
    improvement_vs_model = r_filtered['pnl_pct'] - r_model['pnl_pct']

    print(f"\n  Capture Filtered vs Oracle: {capture_filtered:+.1f}%")
    print(f"  Gain vs Model pur         : {improvement_vs_model:+.2f}%")

    print(f"\n  VERDICT :")
    if capture_filtered > 50:
        print(f"    → Capture > 50%: filtre oracle sauve l'essentiel du PnL")
        print(f"    → Problème = SWITCH (flips parasites quand sign_model ≠ sign_oracle)")
        print(f"    → Piste : stabiliser le signal (3-classes, cadence 30min, loss PnL-aware)")
    elif capture_filtered > 0:
        print(f"    → Capture positive mais < 50%: filtre oracle aide partiellement")
        print(f"    → Problème mixte: switch + timing sub-optimal")
    else:
        print(f"    → Capture négative: même en accord oracle, entrées/sorties mauvaises")
        print(f"    → Problème = TIMING/LAG (la cible binaire ou l'exec timing ne colle pas)")
        print(f"    → Piste : régression rendement, multi-horizon, redéfinir t_ref")


if __name__ == '__main__':
    main()
