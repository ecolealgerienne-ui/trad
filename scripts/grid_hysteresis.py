#!/usr/bin/env python3
"""
Grid search sur hysteresis asymétrique pour transformer les probas XGBoost
en signal de trading avec zone morte.

Règle hysteresis :
  - proba > high  → LONG  (+1)
  - proba < low   → SHORT (-1)
  - low <= proba <= high → zone morte → slope = 0 → conserve position actuelle

Le backtest existant (backtest_5min_progressive dans core.py) traite déjà
slope == 0 comme "conserver la position", donc pas de modification de core
nécessaire.

Usage :
    python scripts/grid_hysteresis.py --split test
    python scripts/grid_hysteresis.py --split val --lows 0.30 0.35 0.40 0.45 --highs 0.55 0.60 0.65 0.70
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
    first, last = closes[0], closes[-1]
    if np.isnan(first) or np.isnan(last) or first == 0:
        return 0.0
    return (last - first) / first * 100


def apply_hysteresis(proba, low, high):
    """Convertit probas en slopes ternaires {-1, 0, +1} selon hysteresis."""
    return np.where(proba > high, 1.0,
                     np.where(proba < low, -1.0, 0.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz',
                        default='data/prepared/dataset_macd_30m_full_progressive.npz')
    parser.add_argument('--preds',
                        default='data/prepared/preds_macd_30m_full_progressive.npz')
    parser.add_argument('--split', default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--fees', type=float, default=0.001)
    parser.add_argument('--lows', type=float, nargs='+',
                        default=[0.30, 0.35, 0.40, 0.45])
    parser.add_argument('--highs', type=float, nargs='+',
                        default=[0.55, 0.60, 0.65, 0.70])
    parser.add_argument('--top', type=int, default=20,
                        help='Top N configs à afficher (trié par PnL Net)')
    args = parser.parse_args()

    print("=" * 100)
    print(f"GRID HYSTERESIS — split={args.split}  fees={args.fees*100:.2f}%")
    print(f"  lows  : {args.lows}")
    print(f"  highs : {args.highs}")
    print("=" * 100)

    # Load
    ds = np.load(args.npz, allow_pickle=True)
    preds = np.load(args.preds, allow_pickle=True)

    closes = ds[f'closes_{args.split}']
    dates = pd.to_datetime(ds[f'dates_{args.split}'])
    y_cont = ds[f'y_{args.split}_continuous']
    p = preds[f'{args.split}_preds_proba']

    period_days = (dates[-1] - dates[0]).total_seconds() / 86400
    print(f"\n{args.split}: {len(closes):,} rows  |  "
          f"{dates[0]} → {dates[-1]}  ({period_days:.0f} jours)")

    # Distribution des probas (diagnostic)
    print(f"\n[Proba distribution]")
    bins_edges = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                   0.60, 0.70, 0.80, 0.90, 0.95, 1.0]
    counts, _ = np.histogram(p, bins=bins_edges)
    for i in range(len(bins_edges) - 1):
        pct = counts[i] / len(p) * 100
        bar = '█' * int(pct / 2)
        print(f"  [{bins_edges[i]:.2f}, {bins_edges[i+1]:.2f}) "
              f"{counts[i]:>10,} ({pct:>5.2f}%) {bar}")
    # Stats clés
    in_03_07 = ((p >= 0.3) & (p <= 0.7)).sum() / len(p) * 100
    in_01_09 = ((p >= 0.1) & (p <= 0.9)).sum() / len(p) * 100
    in_005_095 = ((p >= 0.05) & (p <= 0.95)).sum() / len(p) * 100
    print(f"  → Dans [0.30, 0.70]: {in_03_07:.2f}%  "
          f"| [0.10, 0.90]: {in_01_09:.2f}%  "
          f"| [0.05, 0.95]: {in_005_095:.2f}%")

    # Baseline Oracle
    print(f"\n[Baseline] Oracle (référence)")
    r_oracle = backtest_5min_progressive(y_cont, closes, fees=args.fees)
    bh = buy_and_hold_5m(closes)

    # Baseline Model simple (threshold 0.5, pas d'hysteresis)
    print(f"[Baseline] Model threshold=0.5 (pas d'hysteresis)")
    slopes_base = np.where(p > 0.5, 1.0, -1.0)
    r_base = backtest_5min_progressive(slopes_base, closes, fees=args.fees)

    # Grid search
    print(f"\n[Grid] {sum(1 for l in args.lows for h in args.highs if l < h)} "
          f"configurations hysteresis ...")
    results = []
    for low in args.lows:
        for high in args.highs:
            if low >= high:
                continue
            slopes = apply_hysteresis(p, low, high)
            n_flat = int((slopes == 0).sum())
            r = backtest_5min_progressive(slopes, closes, fees=args.fees)
            results.append({
                'low': low,
                'high': high,
                'dead_zone_pct': n_flat / len(slopes) * 100,
                **r,
            })

    # Tri par PnL Net décroissant
    results.sort(key=lambda x: -x['pnl_pct'])

    # Affichage
    print(f"\n{'=' * 100}")
    print(f"TOP {min(args.top, len(results))} configurations — triées par PnL Net")
    print(f"{'=' * 100}")
    print(f"  {'Low':>6} {'High':>6} {'DeadZn':>7}  "
          f"{'Trades':>8} {'WR':>7} {'PF':>6} {'Sharpe':>7} "
          f"{'Brut':>10} {'Fees':>10} {'Net':>11} {'αB&H':>11}")
    print(f"  {'-' * 96}")

    # Oracle d'abord
    fees_o = r_oracle['n_trades'] * 2 * args.fees * 100
    print(f"  {'ORACLE':<13} {'—':>7}  "
          f"{r_oracle['n_trades']:>8,} "
          f"{r_oracle['win_rate']:>6.1f}% "
          f"{r_oracle['profit_factor']:>6.2f} "
          f"{r_oracle['sharpe']:>7.3f} "
          f"{r_oracle['pnl_pct']+fees_o:>+9.2f}% "
          f"{fees_o:>9.2f}% "
          f"{r_oracle['pnl_pct']:>+10.2f}% "
          f"{r_oracle['pnl_pct']-bh:>+10.2f}%")

    # Baseline t=0.5
    fees_b = r_base['n_trades'] * 2 * args.fees * 100
    print(f"  {'Model t=0.5':<13} {'—':>7}  "
          f"{r_base['n_trades']:>8,} "
          f"{r_base['win_rate']:>6.1f}% "
          f"{r_base['profit_factor']:>6.2f} "
          f"{r_base['sharpe']:>7.3f} "
          f"{r_base['pnl_pct']+fees_b:>+9.2f}% "
          f"{fees_b:>9.2f}% "
          f"{r_base['pnl_pct']:>+10.2f}% "
          f"{r_base['pnl_pct']-bh:>+10.2f}%")

    print(f"  {'-' * 96}")

    # Grid trié
    for r in results[:args.top]:
        fees_pct = r['n_trades'] * 2 * args.fees * 100
        print(f"  {r['low']:>6.2f} {r['high']:>6.2f} {r['dead_zone_pct']:>6.1f}%  "
              f"{r['n_trades']:>8,} "
              f"{r['win_rate']:>6.1f}% "
              f"{r['profit_factor']:>6.2f} "
              f"{r['sharpe']:>7.3f} "
              f"{r['pnl_pct']+fees_pct:>+9.2f}% "
              f"{fees_pct:>9.2f}% "
              f"{r['pnl_pct']:>+10.2f}% "
              f"{r['pnl_pct']-bh:>+10.2f}%")

    # Best
    if results:
        best = results[0]
        print(f"\n  ★ BEST: low={best['low']}  high={best['high']}  "
              f"→ PnL Net {best['pnl_pct']:+.2f}%  "
              f"vs Oracle {r_oracle['pnl_pct']:+.2f}%  "
              f"(capture {best['pnl_pct']/r_oracle['pnl_pct']*100:+.1f}%)")
        print(f"    Trades {best['n_trades']:,} vs Oracle {r_oracle['n_trades']:,}  "
              f"({best['n_trades']-r_oracle['n_trades']:+,})  "
              f"| Dead zone: {best['dead_zone_pct']:.1f}%")


if __name__ == '__main__':
    main()
