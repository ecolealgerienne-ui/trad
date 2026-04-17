#!/usr/bin/env python3
"""
Backtest à résolution 5min d'un dataset "progressive".

Signal = slope_progressive à chaque ligne 5min → sign(slope) = UP/DOWN/FLAT
Execution à close_5m[i+1] (lag 1 tick 5min, réaliste : on ne trade pas à xx:00/xx:30)

Modes :
  --mode oracle : utilise y_test_continuous (ffill oracle.slope[t_ref]) → borne sup
  --mode model  : utilise preds XGBoost (à charger depuis preds NPZ)

Usage :
    python scripts/backtest_progressive.py --npz data/prepared/dataset_macd_30m_180d_progressive.npz
    python scripts/backtest_progressive.py --npz ... --split test --fees 0.001
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def backtest_5min(slopes, closes, fees=0.001):
    """
    Backtest simple à résolution 5min.
    Signal > 0 → LONG, < 0 → SHORT, == 0 → pas de changement.
    Exec à closes[i+1] (lag 1 tick).

    Returns:
        dict avec stats.
    """
    n = len(slopes)
    position = 0    # 0=FLAT, 1=LONG, -1=SHORT
    entry_price = 0.0
    trades = []

    for i in range(n - 1):
        s = slopes[i]
        if np.isnan(s) or s == 0:
            target = position  # pas de signal → on garde la position
        elif s > 0:
            target = 1
        else:
            target = -1

        if target == position:
            continue

        exec_price = closes[i + 1]
        if np.isnan(exec_price):
            continue

        # Sortie de la position actuelle
        if position != 0:
            if position == 1:
                pnl = (exec_price - entry_price) / entry_price
            else:
                pnl = (entry_price - exec_price) / entry_price
            pnl -= 2 * fees
            trades.append({
                'exit_i': i + 1,
                'pnl': pnl,
                'position': position,
                'entry_price': entry_price,
                'exit_price': exec_price,
            })

        # Nouvelle entrée (flip ou depuis FLAT)
        if target != 0:
            entry_price = exec_price
        position = target

    # Close final
    if position != 0:
        exec_price = closes[-1]
        if not np.isnan(exec_price):
            if position == 1:
                pnl = (exec_price - entry_price) / entry_price
            else:
                pnl = (entry_price - exec_price) / entry_price
            pnl -= 2 * fees
            trades.append({
                'exit_i': n - 1, 'pnl': pnl, 'position': position,
                'entry_price': entry_price, 'exit_price': exec_price,
            })

    if not trades:
        return dict(n_trades=0, pnl_pct=0.0, win_rate=0.0,
                    profit_factor=0.0, sharpe=0.0, n_long=0, n_short=0)

    pnls = np.array([t['pnl'] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    return dict(
        n_trades=len(trades),
        pnl_pct=pnls.sum() * 100,
        win_rate=len(wins) / len(pnls) * 100,
        profit_factor=(wins.sum() / abs(losses.sum())
                       if len(losses) > 0 and losses.sum() != 0 else np.inf),
        sharpe=(pnls.mean() / pnls.std()
                if pnls.std() > 1e-10 else 0.0),
        n_long=sum(1 for t in trades if t['position'] == 1),
        n_short=sum(1 for t in trades if t['position'] == -1),
    )


def buy_and_hold(closes):
    first = closes[0]
    last = closes[-1]
    if np.isnan(first) or np.isnan(last) or first == 0:
        return 0.0
    return (last - first) / first * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', required=True,
                        help='Path vers dataset_..._progressive.npz')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--mode', default='oracle', choices=['oracle', 'model'])
    parser.add_argument('--preds', default=None,
                        help='Path vers preds NPZ (requis si --mode model)')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--fees', type=float, default=0.001)
    args = parser.parse_args()

    print("=" * 80)
    print(f"BACKTEST PROGRESSIVE — mode={args.mode}  split={args.split}  "
          f"fees={args.fees*100:.2f}%")
    print("=" * 80)

    npz_path = Path(args.npz)
    if not npz_path.exists():
        print(f"❌ NPZ introuvable: {npz_path}")
        return
    ds = np.load(npz_path, allow_pickle=True)
    print(f"\n✅ NPZ chargé: {npz_path}")

    # Sélection du split
    closes = ds[f'closes_{args.split}']
    dates = pd.to_datetime(ds[f'dates_{args.split}'])
    y_cont = ds[f'y_{args.split}_continuous']
    y_bin = ds[f'y_{args.split}_binary']
    print(f"   Split {args.split}: {len(closes):,} rows")
    print(f"   {dates[0]} → {dates[-1]}  "
          f"({(dates[-1] - dates[0]).total_seconds() / 86400:.1f} jours)")

    # Signal selon mode
    if args.mode == 'oracle':
        # y_cont = ffill oracle.slope[t_ref] → utilisable directement
        slopes = y_cont
        print(f"\n→ Signal = oracle slope (ffill)")
    else:
        if not args.preds:
            print("❌ --preds requis en mode model")
            return
        preds = np.load(args.preds, allow_pickle=True)
        p = preds[f'{args.split}_preds_proba']
        # Seuil → ±1
        slopes = np.where(p > args.threshold, 1.0, -1.0)
        print(f"\n→ Signal = model (threshold={args.threshold})")
        print(f"   UP={(slopes>0).sum():,} DOWN={(slopes<0).sum():,}")

    # Backtest
    print(f"\n→ Backtest à 5min (exec à close_5m[i+1]) ...")
    res = backtest_5min(slopes, closes, fees=args.fees)
    bh = buy_and_hold(closes)

    # Durée du split en jours
    period_days = (dates[-1] - dates[0]).total_seconds() / 86400

    print(f"\n{'='*80}")
    print(f"RÉSULTATS — {args.mode.upper()} sur {args.split}  "
          f"({period_days:.0f} jours)")
    print(f"{'='*80}")
    print(f"  Trades        : {res['n_trades']:,}  "
          f"(L={res['n_long']} / S={res['n_short']})")
    print(f"  Win Rate      : {res['win_rate']:.1f}%")
    print(f"  Profit Factor : {res['profit_factor']:.2f}")
    print(f"  Sharpe        : {res['sharpe']:.3f}")
    print(f"  PnL Net       : {res['pnl_pct']:+.2f}%  (fees={args.fees*100:.2f}%/trade)")
    # Fees total
    fees_tot = res['n_trades'] * 2 * args.fees * 100
    print(f"  Fees totaux   : {fees_tot:.2f}%  "
          f"(PnL brut = {res['pnl_pct'] + fees_tot:+.2f}%)")
    print(f"  Buy & Hold    : {bh:+.2f}%")
    print(f"  Alpha vs B&H  : {res['pnl_pct'] - bh:+.2f}%")


if __name__ == '__main__':
    main()
