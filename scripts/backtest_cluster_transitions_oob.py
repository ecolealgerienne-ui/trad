#!/usr/bin/env python3
"""
Backtest OOB des transitions de clusters Kalman AQ-KF sur RSI (étape 4).

Applique les transitions identifiées comme pertinentes (sur 1/3 train)
au test set complet pour mesurer le PnL Net out-of-sample.

2 stratégies comparées :
  - 'good'   : seulement les transitions avec p-value < 0.05 (STRICT)
  - 'top_N'  : top N transitions par PnL total sur train (PERMISSIF)

Granularité : 30m (cohérent avec clustering). Entry/Exit au close[i+1]
de la transition (lag 1 tick 30m).

Usage :
    python scripts/backtest_cluster_transitions_oob.py --K 10
    python scripts/backtest_cluster_transitions_oob.py --K 10 --top-ns 1 3 5 10 20
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CLUSTERS_DIR = Path('models/kalman_clusters')
TRANS_DIR = Path('results/kalman_clusters')


def detect_transitions(cluster_ids):
    """Retourne (indices, from_c, to_c) des transitions."""
    diff = np.diff(cluster_ids)
    mask = diff != 0
    indices = np.where(mask)[0] + 1
    from_c = cluster_ids[indices - 1]
    to_c = cluster_ids[indices]
    return indices, from_c, to_c


def backtest_strategy(allowed_transitions, trans_indices, from_c, to_c,
                        closes, fees):
    """
    Backtest 30m en appliquant les transitions allowed.

    allowed_transitions : list de tuples (A, B, direction) avec direction ∈ {'LONG','SHORT'}

    Logique :
      - FLAT : à chaque transition, si (from, to) ∈ allowed → entry direction
      - EN POSITION : à la PROCHAINE transition (quelconque), exit au close[i+1]
        puis vérifier si cette nouvelle transition est allowed → entry immédiat

    Returns :
        dict avec n_trades, pnl_brut_pct, pnl_net_pct, win_rate, pf, sharpe
    """
    # Index rapide : set de (A, B) allowed
    allowed_dict = {(A, B): dir for A, B, dir in allowed_transitions}

    n_closes = len(closes)
    position = 0
    entry_price = 0.0
    trades = []

    for k in range(len(trans_indices)):
        i = trans_indices[k]
        if i + 1 >= n_closes:
            continue
        price = closes[i + 1]
        if np.isnan(price):
            continue

        # Si en position, on sort
        if position != 0:
            if position == 1:
                pnl_brut = (price - entry_price) / entry_price
            else:
                pnl_brut = (entry_price - price) / entry_price
            pnl_net = pnl_brut - 2 * fees
            trades.append({'pnl_brut': pnl_brut, 'pnl_net': pnl_net,
                             'position': position})
            position = 0

        # Check si cette transition est allowed
        from_cluster = int(from_c[k])
        to_cluster = int(to_c[k])
        key = (from_cluster, to_cluster)
        if key in allowed_dict:
            direction = allowed_dict[key]
            position = 1 if direction == 'LONG' else -1
            entry_price = price

    # Close final si encore en position
    if position != 0:
        last_price = closes[-1]
        if not np.isnan(last_price) and entry_price > 0:
            if position == 1:
                pnl_brut = (last_price - entry_price) / entry_price
            else:
                pnl_brut = (entry_price - last_price) / entry_price
            pnl_net = pnl_brut - 2 * fees
            trades.append({'pnl_brut': pnl_brut, 'pnl_net': pnl_net,
                             'position': position})

    if not trades:
        return dict(n_trades=0, pnl_brut_pct=0.0, pnl_net_pct=0.0,
                    fees_pct=0.0, win_rate=0.0, pf=0.0, sharpe=0.0,
                    n_long=0, n_short=0)

    pnls_net = np.array([t['pnl_net'] for t in trades])
    pnls_brut = np.array([t['pnl_brut'] for t in trades])
    wins = pnls_net[pnls_net > 0]
    losses = pnls_net[pnls_net < 0]
    pf = (wins.sum() / abs(losses.sum())
          if len(losses) > 0 and losses.sum() != 0 else np.inf)
    sharpe = (pnls_net.mean() / pnls_net.std()
              if pnls_net.std() > 1e-10 else 0.0)

    return dict(
        n_trades=len(trades),
        pnl_brut_pct=pnls_brut.sum() * 100,
        pnl_net_pct=pnls_net.sum() * 100,
        fees_pct=len(trades) * 2 * fees * 100,
        win_rate=len(wins) / len(pnls_net) * 100,
        pf=pf,
        sharpe=sharpe,
        n_long=sum(1 for t in trades if t['position'] == 1),
        n_short=sum(1 for t in trades if t['position'] == -1),
    )


def buy_and_hold(closes):
    first, last = closes[0], closes[-1]
    if np.isnan(first) or np.isnan(last) or first == 0:
        return 0.0
    return (last - first) / first * 100


def print_result(label, r, bh=None):
    alpha = (r['pnl_net_pct'] - bh) if bh is not None else r['pnl_net_pct']
    print(f"  {label:<40}"
          f"{r['n_trades']:>8,} "
          f"{r['n_long']:>6,}/{r['n_short']:<6,} "
          f"{r['win_rate']:>6.1f}% "
          f"{r['pf']:>6.2f} "
          f"{r['sharpe']:>7.3f} "
          f"{r['pnl_brut_pct']:>+9.2f}% "
          f"{r['fees_pct']:>8.2f}% "
          f"{r['pnl_net_pct']:>+10.2f}% "
          f"{alpha:>+10.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-npz',
                        default='data/prepared/kalman_rsi_features_30m.npz')
    parser.add_argument('--K', type=int, default=10,
                        help='K du clustering à utiliser (default 10)')
    parser.add_argument('--tf', type=int, default=30)
    parser.add_argument('--fees', type=float, default=0.001)
    parser.add_argument('--top-ns', type=int, nargs='+',
                        default=[1, 3, 5, 10, 20],
                        help='Top N transitions à tester (permissif)')
    parser.add_argument('--split', default='test',
                        choices=['train', 'val', 'test'])
    args = parser.parse_args()

    tf_label = f'{args.tf}m' if args.tf < 60 else '1h'
    print("=" * 115)
    print(f"BACKTEST OOB — TRANSITIONS CLUSTERS K={args.K}  "
          f"split={args.split}  fees={args.fees*100:.2f}%")
    print("=" * 115)

    # 1. Charger closes test
    ds = np.load(args.features_npz, allow_pickle=True)
    closes = ds[f'closes_{args.split}_tf']
    dates = pd.to_datetime(ds[f'dates_{args.split}_tf'])
    period_days = (dates[-1] - dates[0]).total_seconds() / 86400
    print(f"\n[1] {args.split}: {len(closes):,} bougies 30m  |  "
          f"{dates[0]} → {dates[-1]}  ({period_days:.0f} jours)")

    # 2. Charger cluster_ids pour ce split
    pkl_cluster = CLUSTERS_DIR / f'kmeans_rsi_k{args.K}_{tf_label}.pkl'
    if not pkl_cluster.exists():
        print(f"❌ Cluster pickle introuvable : {pkl_cluster}")
        return
    with open(pkl_cluster, 'rb') as f:
        cluster_data = pickle.load(f)
    cluster_ids = cluster_data[f'cluster_ids_{args.split}']
    print(f"[2] Cluster IDs chargés : {pkl_cluster.name}  "
          f"K={args.K}  ({len(cluster_ids):,} rows)")

    # 3. Charger analyse transitions (sur train)
    pkl_trans = TRANS_DIR / f'transitions_analysis_{tf_label}.pkl'
    if not pkl_trans.exists():
        print(f"❌ Analyse transitions introuvable : {pkl_trans}")
        return
    with open(pkl_trans, 'rb') as f:
        trans_data = pickle.load(f)
    if args.K not in trans_data['results_per_K']:
        print(f"❌ K={args.K} pas dans l'analyse (lancez analyze_cluster_transitions --ks {args.K})")
        return
    K_results = trans_data['results_per_K'][args.K]
    all_results = K_results['all_results']
    good_transitions_raw = K_results['good_transitions']
    print(f"[3] Analyse transitions chargée : {pkl_trans.name}")
    print(f"    All results    : {len(all_results)} transitions")
    print(f"    Good (p<0.05)  : {len(good_transitions_raw)} transitions")

    # 4. Détecter transitions sur split
    trans_idx, from_c, to_c = detect_transitions(cluster_ids)
    print(f"[4] Transitions détectées sur {args.split} : {len(trans_idx):,}")

    # 5. Baseline
    bh = buy_and_hold(closes)
    print(f"\n[5] Baseline Buy & Hold : {bh:+.2f}%")

    # 6. Stratégies à tester
    print(f"\n{'=' * 115}")
    print(f"STRATÉGIES TESTÉES — tri par PnL Net décroissant")
    print(f"{'=' * 115}")
    header = (f"  {'Stratégie':<40}"
              f"{'Trades':>8} "
              f"{'L/S':>13} "
              f"{'WR':>7} "
              f"{'PF':>6} "
              f"{'Sharpe':>7} "
              f"{'Brut':>10} "
              f"{'Fees':>9} "
              f"{'Net':>11} "
              f"{'αB&H':>11}")
    print(header)
    print(f"  {'-' * 113}")
    print(f"  {'Buy & Hold':<40}{'—':>8} {'—':>13} {'—':>7} {'—':>6} {'—':>7} "
          f"{'—':>10} {'—':>9} {bh:>+10.2f}% {'0.00':>+10}%")
    print(f"  {'-' * 113}")

    results_collected = []

    # Stratégie good
    good_allowed = [(r['A'], r['B'], r['best_direction'])
                     for r in good_transitions_raw]
    if good_allowed:
        r_good = backtest_strategy(good_allowed, trans_idx, from_c, to_c,
                                      closes, args.fees)
        label_good = f"good (p<0.05, N={len(good_allowed)})"
        results_collected.append((label_good, r_good))
    else:
        print(f"  ⚠️ good : aucune transition → skip")

    # Stratégies top_N
    for N in args.top_ns:
        if N > len(all_results):
            continue
        top_allowed = [(r['A'], r['B'], r['best_direction'])
                        for r in all_results[:N]]
        r = backtest_strategy(top_allowed, trans_idx, from_c, to_c,
                                closes, args.fees)
        label = f"top_{N} (N={N})"
        results_collected.append((label, r))

    # Tri par PnL Net décroissant
    results_collected.sort(key=lambda x: -x[1]['pnl_net_pct'])

    for label, r in results_collected:
        print_result(label, r, bh=bh)

    # Best
    if results_collected:
        best_label, best_r = results_collected[0]
        print(f"\n  ★ BEST: {best_label}")
        print(f"    PnL Net {best_r['pnl_net_pct']:+.2f}%  "
              f"vs B&H {bh:+.2f}%  "
              f"(alpha {best_r['pnl_net_pct'] - bh:+.2f}%)")
        print(f"    Trades: {best_r['n_trades']:,}  "
              f"WR: {best_r['win_rate']:.1f}%  "
              f"PF: {best_r['pf']:.2f}")
        if best_r['pnl_net_pct'] > 0:
            print(f"    🏆 PnL POSITIF sur test OOB !")
        elif best_r['pnl_net_pct'] > bh:
            print(f"    ⚠️ Négatif mais mieux que B&H (alpha positif)")
        else:
            print(f"    ❌ Pas d'edge détectable")


if __name__ == '__main__':
    main()
