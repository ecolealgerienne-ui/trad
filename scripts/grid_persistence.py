#!/usr/bin/env python3
"""
Grid search : persistence temporelle + holding minimum.

Objectif : réduire les flips parasites haute fréquence causés par les
erreurs ponctuelles du modèle (probas quasi-binaires qui bouncent
0.98 → 0.03 → 0.97 sur 3 rows 5min consécutives).

Deux filtres appliqués sur le signal brut (sign(proba - 0.5)) :

1. CONFIRM : le signe ne change que si N pas consécutifs montrent
   le nouveau signe. Élimine les flips à 1-2 pas.
2. MIN_HOLD : après un changement à i, tout nouveau changement est
   bloqué avant i + min_hold pas 5min. Impose une durée minimale par trade.

Grid défaut :
  --min-hold : [0, 3, 6, 12, 24]   (0min, 15min, 30min, 1h, 2h)
  --confirm  : [1, 2, 3, 6]        (1 = aucune confirmation)

20 combinaisons par run. Comparaison avec Oracle + baseline model t=0.5.

Usage :
    python scripts/grid_persistence.py --split test
    python scripts/grid_persistence.py --split test --min-hold 0 6 12 --confirm 1 3
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


def apply_persistence(proba, confirm_steps, min_hold, threshold=0.5):
    """
    Transforme probas → slopes ±1 avec persistence + holding minimum.

    Règle :
      - raw_sign[i] = +1 si proba[i] > threshold, -1 sinon
      - current_sign ne change que si les `confirm_steps` dernières
        raw_signs sont toutes égales au nouveau signe
      - après un changement à i, prochain changement autorisé à i + min_hold

    Args:
        proba : np.ndarray (n,)
        confirm_steps : int >= 1 (1 = pas de confirmation)
        min_hold : int >= 0 (0 = pas de holding min)
        threshold : float (default 0.5)

    Returns:
        np.ndarray (n,) avec valeurs ±1.
    """
    n = len(proba)
    raw = np.where(proba > threshold, 1, -1)
    slopes = np.zeros(n, dtype=np.int8)
    if n == 0:
        return slopes.astype(np.float64)

    current_sign = raw[0]
    slopes[0] = current_sign
    last_change_i = -min_hold  # pour autoriser changement dès i=0 si besoin

    for i in range(1, n):
        if raw[i] != current_sign:
            # Holding min check
            if (i - last_change_i) < min_hold:
                slopes[i] = current_sign
                continue
            # Confirmation check
            if confirm_steps > 1:
                if i + 1 < confirm_steps:
                    slopes[i] = current_sign
                    continue
                window = raw[i - confirm_steps + 1:i + 1]
                if np.all(window == raw[i]):
                    current_sign = raw[i]
                    last_change_i = i
            else:
                current_sign = raw[i]
                last_change_i = i
        slopes[i] = current_sign

    return slopes.astype(np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz',
                        default='data/prepared/dataset_macd_30m_full_progressive.npz')
    parser.add_argument('--preds',
                        default='data/prepared/preds_macd_30m_full_progressive.npz')
    parser.add_argument('--split', default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--fees', type=float, default=0.001)
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold pour raw_sign (default 0.5)')
    parser.add_argument('--min-hold', type=int, nargs='+',
                        default=[0, 3, 6, 12, 24],
                        help='Pas 5min min entre 2 trades (0=aucun)')
    parser.add_argument('--confirm', type=int, nargs='+',
                        default=[1, 2, 3, 6],
                        help='Pas consécutifs requis pour flip (1=aucun)')
    parser.add_argument('--top', type=int, default=30)
    args = parser.parse_args()

    print("=" * 105)
    print(f"GRID PERSISTENCE — split={args.split}  fees={args.fees*100:.2f}%  "
          f"threshold={args.threshold}")
    print(f"  min-hold (5min): {args.min_hold}  (en minutes: "
          f"{[h*5 for h in args.min_hold]})")
    print(f"  confirm (5min) : {args.confirm}  (en minutes: "
          f"{[c*5 for c in args.confirm]})")
    print("=" * 105)

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

    # Baselines
    print(f"\n[Baselines]")
    r_oracle = backtest_5min_progressive(y_cont, closes, fees=args.fees)
    bh = buy_and_hold_5m(closes)
    slopes_base = np.where(p > args.threshold, 1.0, -1.0)
    r_base = backtest_5min_progressive(slopes_base, closes, fees=args.fees)

    # Grid
    print(f"\n[Grid] {len(args.min_hold) * len(args.confirm)} configurations ...")
    results = []
    for mh in args.min_hold:
        for cf in args.confirm:
            slopes = apply_persistence(p, cf, mh, args.threshold)
            r = backtest_5min_progressive(slopes, closes, fees=args.fees)
            results.append({
                'min_hold': mh,
                'confirm': cf,
                **r,
            })
    results.sort(key=lambda x: -x['pnl_pct'])

    # Affichage
    print(f"\n{'=' * 105}")
    print(f"TOP {min(args.top, len(results))} configs — triées par PnL Net")
    print(f"{'=' * 105}")
    header = (f"  {'mHold':>6} {'conf':>5}  "
              f"{'Trades':>8} {'WR':>7} {'PF':>6} {'Sharpe':>7} "
              f"{'Brut':>10} {'Fees':>10} {'Net':>11} {'αB&H':>11} {'Capt%':>8}")
    print(header)
    print(f"  {'-' * 101}")

    def fmt_row(label, r, suffix=''):
        fees_pct = r['n_trades'] * 2 * args.fees * 100
        capture = (r['pnl_pct'] / r_oracle['pnl_pct'] * 100
                   if r_oracle['pnl_pct'] != 0 else 0)
        return (f"  {label:<13}  "
                f"{r['n_trades']:>8,} "
                f"{r['win_rate']:>6.1f}% "
                f"{r['profit_factor']:>6.2f} "
                f"{r['sharpe']:>7.3f} "
                f"{r['pnl_pct']+fees_pct:>+9.2f}% "
                f"{fees_pct:>9.2f}% "
                f"{r['pnl_pct']:>+10.2f}% "
                f"{r['pnl_pct']-bh:>+10.2f}% "
                f"{capture:>+7.1f}%{suffix}")

    print(fmt_row('ORACLE', r_oracle))
    print(fmt_row(f'Model t={args.threshold}', r_base))
    print(f"  {'-' * 101}")

    for r in results[:args.top]:
        label = f"{r['min_hold']:>3} {r['confirm']:>3}"
        fees_pct = r['n_trades'] * 2 * args.fees * 100
        capture = (r['pnl_pct'] / r_oracle['pnl_pct'] * 100
                   if r_oracle['pnl_pct'] != 0 else 0)
        print(f"  {r['min_hold']:>6} {r['confirm']:>5}  "
              f"{r['n_trades']:>8,} "
              f"{r['win_rate']:>6.1f}% "
              f"{r['profit_factor']:>6.2f} "
              f"{r['sharpe']:>7.3f} "
              f"{r['pnl_pct']+fees_pct:>+9.2f}% "
              f"{fees_pct:>9.2f}% "
              f"{r['pnl_pct']:>+10.2f}% "
              f"{r['pnl_pct']-bh:>+10.2f}% "
              f"{capture:>+7.1f}%")

    # Best
    if results:
        best = results[0]
        print(f"\n  ★ BEST: min_hold={best['min_hold']} ({best['min_hold']*5}min)  "
              f"confirm={best['confirm']} ({best['confirm']*5}min)")
        print(f"    PnL Net {best['pnl_pct']:+.2f}%  "
              f"vs Oracle {r_oracle['pnl_pct']:+.2f}%  "
              f"(capture {best['pnl_pct']/r_oracle['pnl_pct']*100:+.1f}%)")
        print(f"    Trades {best['n_trades']:,} vs Oracle {r_oracle['n_trades']:,}  "
              f"| vs Model t={args.threshold}: "
              f"{best['n_trades']-r_base['n_trades']:+,}")


if __name__ == '__main__':
    main()
