#!/usr/bin/env python3
"""
Analyse exhaustive des transitions entre clusters K-means (étape 3).

Pour chaque K et chaque paire (A → B) de clusters (A ≠ B) :
  - Détecte toutes les occurrences de transition A→B sur 1/3 du train
  - Pour chaque occurrence : entry à la transition, exit à la transition suivante
  - Teste 2 directions (LONG et SHORT)
  - Stats : n_occurrences, WR, PnL net, Sharpe-like, significativité t-test
  - Identifie les transitions pertinentes (PnL net positif + significatif)

Granularité : 30m (bougies du Kalman RSI). Entry = close[i+1], Exit = close à
la prochaine transition. PnL = (exit - entry) / entry pour LONG, inverse pour SHORT.

Sortie : pickle avec les bonnes transitions par K (pour étape 4 backtest OOB).

Usage :
    python scripts/analyze_cluster_transitions.py
    python scripts/analyze_cluster_transitions.py --ks 10 15 20 --sample-frac 0.333
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from scipy.stats import ttest_1samp
except ImportError:
    ttest_1samp = None

CLUSTERS_DIR = Path('models/kalman_clusters')
OUT_DIR = Path('results/kalman_clusters')


def detect_transitions(cluster_ids):
    """
    Retourne un array des indices de transition (position dans cluster_ids
    où cluster_ids[i] != cluster_ids[i-1]).

    Returns:
        indices (n_trans,) : positions i des transitions
        from_cluster (n_trans,) : cluster_ids[i-1]
        to_cluster (n_trans,) : cluster_ids[i]
    """
    diff = np.diff(cluster_ids)
    mask = diff != 0
    indices = np.where(mask)[0] + 1  # i où change
    from_c = cluster_ids[indices - 1]
    to_c = cluster_ids[indices]
    return indices, from_c, to_c


def compute_transition_pnl(trans_indices, from_c, to_c, closes, fees,
                              A, B):
    """
    Pour toutes les occurrences de transition A→B :
      - Entry au close[trans_i + 1] (lag 1 tick 30m — skip la bougie de transition)
      - Exit au close[next_trans_i + 1]

    Returns :
        pnl_brut_long, pnl_brut_short (arrays) : PnL brut par occurrence
    """
    mask = (from_c == A) & (to_c == B)
    occ = trans_indices[mask]
    n = len(closes)
    pnl_brut_long = []
    pnl_brut_short = []
    for k, i in enumerate(occ):
        if i + 1 >= n:
            continue
        # Entry au close[i+1]
        entry = closes[i + 1]
        # Trouver prochaine transition après i
        # next transitions = trans_indices > i
        next_trans_mask = trans_indices > i
        if not next_trans_mask.any():
            # Pas de prochaine transition : exit à la dernière close
            exit_i = n - 1
        else:
            next_i = trans_indices[next_trans_mask][0]
            if next_i + 1 < n:
                exit_i = next_i + 1
            else:
                exit_i = n - 1
        exit_p = closes[exit_i]
        if np.isnan(entry) or np.isnan(exit_p) or entry == 0:
            continue
        ret = (exit_p - entry) / entry
        pnl_brut_long.append(ret - 2 * fees)
        pnl_brut_short.append(-ret - 2 * fees)
    return (np.array(pnl_brut_long), np.array(pnl_brut_short))


def significance_test(pnls):
    """T-test 1-sample contre 0. Retourne (mean, p_value)."""
    if len(pnls) < 3 or ttest_1samp is None:
        return float(np.mean(pnls)) if len(pnls) > 0 else 0.0, 1.0
    mean = float(np.mean(pnls))
    # t-test : H0 mean = 0
    _, p_value = ttest_1samp(pnls, 0.0)
    return mean, float(p_value)


def analyze_K(K, pkl_path, closes_train, sample_frac, fees,
                min_occurrences, p_threshold):
    """Analyse toutes les transitions A→B × 2 directions pour K donné."""
    print(f"\n{'=' * 115}")
    print(f"K = {K}  —  analyse transitions")
    print(f"{'=' * 115}")

    with open(pkl_path, 'rb') as f:
        cluster_data = pickle.load(f)
    cluster_ids_train = cluster_data['cluster_ids_train']

    # 1/3 du train (premier tiers chronologique)
    n_total = len(cluster_ids_train)
    n_sample = int(n_total * sample_frac)
    cluster_ids_sample = cluster_ids_train[:n_sample]
    closes_sample = closes_train[:n_sample]
    print(f"  Sample : {n_sample:,} bougies / {n_total:,} train "
          f"({sample_frac*100:.1f}%)")

    # Détecter transitions
    trans_idx, from_c, to_c = detect_transitions(cluster_ids_sample)
    print(f"  Transitions détectées : {len(trans_idx):,}  "
          f"({len(trans_idx)/n_sample*100:.2f}% rows)")

    # Matrice (K×K) × 2 directions : pour chaque (A, B) → stats
    results = []
    for A in range(K):
        for B in range(K):
            if A == B:
                continue
            pnl_l, pnl_s = compute_transition_pnl(
                trans_idx, from_c, to_c, closes_sample, fees, A, B)
            n_occ = len(pnl_l)
            if n_occ < min_occurrences:
                continue
            # Direction LONG
            mean_l, p_l = significance_test(pnl_l)
            # Direction SHORT
            mean_s, p_s = significance_test(pnl_s)
            # Stats
            wr_l = float((pnl_l > 0).mean() * 100)
            wr_s = float((pnl_s > 0).mean() * 100)
            results.append({
                'A': A, 'B': B,
                'n_occ': n_occ,
                'long_mean': mean_l * 100, 'long_p': p_l, 'long_wr': wr_l,
                'long_total': float(pnl_l.sum() * 100),
                'short_mean': mean_s * 100, 'short_p': p_s, 'short_wr': wr_s,
                'short_total': float(pnl_s.sum() * 100),
            })

    # Trier par PnL total (best direction entre LONG/SHORT)
    for r in results:
        if r['long_total'] >= r['short_total']:
            r['best_direction'] = 'LONG'
            r['best_total'] = r['long_total']
            r['best_mean'] = r['long_mean']
            r['best_p'] = r['long_p']
            r['best_wr'] = r['long_wr']
        else:
            r['best_direction'] = 'SHORT'
            r['best_total'] = r['short_total']
            r['best_mean'] = r['short_mean']
            r['best_p'] = r['short_p']
            r['best_wr'] = r['short_wr']

    results.sort(key=lambda r: -r['best_total'])

    # Affichage top 20
    print(f"\n  TOP 20 transitions A→B par PnL total (best direction) :")
    print(f"    {'A':>3} {'B':>3} {'N_occ':>6}  "
          f"{'Dir':>5} {'Mean%':>9} {'Total%':>10} "
          f"{'WR%':>6} {'p-val':>8}  Sig")
    for r in results[:20]:
        sig = '✅' if (r['best_p'] < p_threshold and r['best_total'] > 0) else ''
        print(f"    {r['A']:>3} {r['B']:>3} {r['n_occ']:>6,}  "
              f"{r['best_direction']:>5} {r['best_mean']:>+8.4f}% "
              f"{r['best_total']:>+9.2f}% {r['best_wr']:>5.1f}% "
              f"{r['best_p']:>8.4f}  {sig}")

    # Filtrer les "bonnes" transitions
    good_transitions = [
        r for r in results
        if r['best_total'] > 0 and r['best_p'] < p_threshold
    ]
    print(f"\n  Bonnes transitions (PnL > 0 ET p < {p_threshold}) : "
          f"{len(good_transitions)}")

    # PnL total combiné si on prend toutes les bonnes
    if good_transitions:
        total_combined = sum(r['best_total'] for r in good_transitions)
        total_occ = sum(r['n_occ'] for r in good_transitions)
        print(f"  → Sur 1/3 train : PnL total combiné = "
              f"{total_combined:+.2f}%  "
              f"({total_occ:,} trades)")

    return {
        'K': K,
        'n_total_trans': len(trans_idx),
        'all_results': results,
        'good_transitions': good_transitions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-npz',
                        default='data/prepared/kalman_rsi_features_30m.npz')
    parser.add_argument('--ks', type=int, nargs='+',
                        default=[10, 15, 20],
                        help='Ks à analyser')
    parser.add_argument('--sample-frac', type=float, default=1/3,
                        help='Fraction du train pour analyse (default 1/3)')
    parser.add_argument('--fees', type=float, default=0.001,
                        help='Fees par côté')
    parser.add_argument('--min-occurrences', type=int, default=10,
                        help='Nombre min d\'occurrences pour considérer une transition')
    parser.add_argument('--p-threshold', type=float, default=0.05,
                        help='Seuil p-value t-test pour significativité')
    args = parser.parse_args()

    print("=" * 115)
    print(f"ANALYSE EXHAUSTIVE DES TRANSITIONS DE CLUSTERS")
    print(f"  Ks        : {args.ks}")
    print(f"  Sample    : {args.sample_frac*100:.1f}% du train")
    print(f"  Fees      : {args.fees*100:.2f}%/côté")
    print(f"  Min N_occ : {args.min_occurrences}")
    print(f"  p-threshold : {args.p_threshold}")
    print("=" * 115)

    # Charger closes train
    ds = np.load(args.features_npz, allow_pickle=True)
    closes_train = ds['closes_train_tf']
    tf_label = f"{int(ds['tf_minutes'])}m"
    print(f"\n  Closes train : {len(closes_train):,} bougies {tf_label}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_K_results = {}
    for K in args.ks:
        pkl_path = CLUSTERS_DIR / f'kmeans_rsi_k{K}_{tf_label}.pkl'
        if not pkl_path.exists():
            print(f"❌ Pickle manquant : {pkl_path}")
            continue
        res = analyze_K(K, pkl_path, closes_train, args.sample_frac, args.fees,
                          args.min_occurrences, args.p_threshold)
        all_K_results[K] = res

    # Sauvegarde
    out_path = OUT_DIR / f'transitions_analysis_{tf_label}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump({
            'results_per_K': all_K_results,
            'sample_frac': args.sample_frac,
            'fees': args.fees,
            'min_occurrences': args.min_occurrences,
            'p_threshold': args.p_threshold,
            'tf_label': tf_label,
        }, f)
    print(f"\n✅ Sauvé : {out_path}")

    # Synthèse
    print(f"\n{'=' * 115}")
    print(f"SYNTHÈSE — nb bonnes transitions par K")
    print(f"{'=' * 115}")
    print(f"   {'K':>3}  {'N_trans':>10}  {'N_good':>8}  "
          f"{'PnL_total_combined':>20}  {'N_occ_good':>12}")
    for K, res in all_K_results.items():
        n_good = len(res['good_transitions'])
        total = sum(r['best_total'] for r in res['good_transitions'])
        n_occ = sum(r['n_occ'] for r in res['good_transitions'])
        print(f"   {K:>3}  {res['n_total_trans']:>10,}  {n_good:>8}  "
              f"{total:>+18.2f}%  {n_occ:>12,}")

    print(f"\n  → Prochaine étape : backtest_cluster_transitions_oob.py "
          f"sur test set")


if __name__ == '__main__':
    main()
