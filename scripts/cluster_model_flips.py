#!/usr/bin/env python3
"""
Clustering non-supervisé des flips du modèle pour identifier les régimes
pertinents vs parasites.

Idée utilisateur (validée) :
  1. Prendre les ~10k flips du modèle avec leurs 9 features contextuelles
  2. Clustering K-means (grid K=5,7,10,15)
  3. Pour chaque cluster : mesurer rate is_profitable_flip / is_good_flip
  4. Clusters avec rate élevé = régimes pertinents
  5. Clusters avec rate bas = parasites (à filtrer en backtest)

Le clustering est NON-SUPERVISÉ (features seulement). Les labels
(is_profitable_flip, is_good_flip) servent uniquement à ÉVALUER la
qualité des clusters A POSTERIORI, pas à les découvrir.

Méthodologie OOB :
  - Fit clustering sur flips VAL (out-of-sample modèle direction)
  - Sauvegarder scaler + centroïdes
  - Le backtest (étape suivante) assignera les flips TEST au cluster
    le plus proche (sans refit)

Usage :
    python scripts/cluster_model_flips.py \\
        --long-csv  results/flips/flips_to_long_rsi_30m_full_cnnlstm_lag0_val.csv \\
        --short-csv results/flips/flips_to_short_rsi_30m_full_cnnlstm_lag0_val.csv

    # Grid personnalisé
    python scripts/cluster_model_flips.py --long-csv ... --short-csv ... --ks 5 7 10 15 20
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
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from scipy.stats import chi2_contingency
except ImportError as e:
    print(f"❌ Dépendance manquante: {e}")
    sys.exit(1)

MODELS_DIR = Path('models/clusters')
RESULTS_DIR = Path('results/clusters')

FEATURES_CLUSTERING = [
    # Top discriminantes (Cohen's d >= 0.2)
    'atr_14_norm',
    'atr_ratio_sl',
    'distance_to_ma20',
    'proba_distance_to_extreme',
    'proba_trend_3rows',
    'volume_relative',
    # Complémentaires (Cohen's d modéré)
    'close_slope_1h',
    'proba_std_12rows',
    'model_proba',
]


def evaluate_cluster(k_id, mask, df_flips, base_rate_profit, base_rate_good):
    """Calcule les stats détaillées d'un cluster."""
    size = int(mask.sum())
    pct = size / len(df_flips) * 100

    sub = df_flips[mask]
    rate_profit = sub['is_profitable_flip'].mean()
    rate_good = sub['is_good_flip'].mean()
    pnl_mean = sub['pnl_net_flip'].mean() * 100  # en %
    pnl_sum = sub['pnl_net_flip'].sum() * 100

    # Direction split
    n_long = int((sub['new_signal_model'] == 1).sum())
    n_short = int((sub['new_signal_model'] == -1).sum())

    # Rate profitable par direction
    long_mask = sub['new_signal_model'] == 1
    short_mask = sub['new_signal_model'] == -1
    rate_profit_long = (sub.loc[long_mask, 'is_profitable_flip'].mean()
                         if n_long > 0 else 0.0)
    rate_profit_short = (sub.loc[short_mask, 'is_profitable_flip'].mean()
                          if n_short > 0 else 0.0)

    # Chi-square test : is_profitable_flip significativement différent de base rate ?
    # Contingency : [in_cluster_profit, in_cluster_no_profit], [out_cluster_profit, out_cluster_no_profit]
    other = df_flips[~mask]
    tab = np.array([
        [sub['is_profitable_flip'].sum(), size - sub['is_profitable_flip'].sum()],
        [other['is_profitable_flip'].sum(), len(other) - other['is_profitable_flip'].sum()],
    ])
    try:
        _, p_value, _, _ = chi2_contingency(tab)
    except Exception:
        p_value = 1.0

    # Lift vs base
    lift_profit = rate_profit / base_rate_profit if base_rate_profit > 0 else 0
    lift_good = rate_good / base_rate_good if base_rate_good > 0 else 0

    return {
        'k_id': k_id,
        'size': size,
        'pct': pct,
        'rate_profit': rate_profit,
        'rate_good': rate_good,
        'pnl_mean_pct': pnl_mean,
        'pnl_sum_pct': pnl_sum,
        'n_long': n_long,
        'n_short': n_short,
        'rate_profit_long': rate_profit_long,
        'rate_profit_short': rate_profit_short,
        'p_value': p_value,
        'lift_profit': lift_profit,
        'lift_good': lift_good,
    }


def run_kmeans(X_scaled, df_flips, K, base_rate_profit, base_rate_good,
                feature_cols, seed=42):
    """K-means avec K clusters + évaluation de chaque cluster."""
    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    labels = km.fit_predict(X_scaled)

    cluster_stats = []
    for k in range(K):
        mask = labels == k
        if mask.sum() == 0:
            continue
        stats = evaluate_cluster(k, mask, df_flips, base_rate_profit,
                                    base_rate_good)
        cluster_stats.append(stats)

    # Trier par rate_profit décroissant
    cluster_stats.sort(key=lambda s: -s['rate_profit'])

    return km, labels, cluster_stats


def classify_clusters(cluster_stats, base_rate_profit,
                        min_size=100, p_threshold=0.05,
                        relevant_lift=1.3, parasite_lift=0.5):
    """
    Classifie chaque cluster :
      - 'relevant'  : rate_profit >= relevant_lift * base, size ok, p-value ok
      - 'parasite'  : rate_profit <= parasite_lift * base, size ok, p-value ok
      - 'neutral'   : entre les 2 ou non significatif
    """
    for s in cluster_stats:
        if s['size'] < min_size:
            s['category'] = 'too_small'
        elif s['p_value'] > p_threshold:
            s['category'] = 'non_significant'
        elif s['lift_profit'] >= relevant_lift:
            s['category'] = 'relevant'
        elif s['lift_profit'] <= parasite_lift:
            s['category'] = 'parasite'
        else:
            s['category'] = 'neutral'
    return cluster_stats


def print_cluster_table(K, cluster_stats, base_rate_profit):
    """Affiche un tableau détaillé des clusters."""
    print(f"\n{'=' * 115}")
    print(f"K = {K} — stats par cluster (triées par rate_profit décroissant)")
    print(f"{'=' * 115}")

    header = (f"  {'id':>3} {'Size':>6} {'%':>6} "
              f"{'Rate_prof':>10} {'Lift':>6} {'Rate_good':>10} "
              f"{'Long':>6} {'Short':>6} "
              f"{'Prof_L':>8} {'Prof_S':>8} "
              f"{'PnL_mean':>10} {'PnL_sum':>10} "
              f"{'p-val':>8} {'Category':>15}")
    print(header)
    print(f"  {'-' * 113}")

    for s in cluster_stats:
        marker_profit = ''
        if s['category'] == 'relevant':
            marker_profit = '🔥'
        elif s['category'] == 'parasite':
            marker_profit = '❌'
        elif s['category'] == 'too_small':
            marker_profit = '⚠️'

        print(f"  {s['k_id']:>3} "
              f"{s['size']:>6,} "
              f"{s['pct']:>5.1f}% "
              f"{s['rate_profit']*100:>9.2f}% "
              f"{s['lift_profit']:>5.2f}× "
              f"{s['rate_good']*100:>9.2f}% "
              f"{s['n_long']:>6,} "
              f"{s['n_short']:>6,} "
              f"{s['rate_profit_long']*100:>7.2f}% "
              f"{s['rate_profit_short']*100:>7.2f}% "
              f"{s['pnl_mean_pct']:>+9.4f}% "
              f"{s['pnl_sum_pct']:>+9.2f}% "
              f"{s['p_value']:>8.4f} "
              f"{s['category']:>15} "
              f"{marker_profit}")


def print_cluster_centroids(km, feature_cols, scaler, K):
    """Affiche les centroïdes (dénormalisés)."""
    print(f"\n  Centroïdes (features moyennes par cluster, valeurs dé-normalisées) :")
    centroids_raw = scaler.inverse_transform(km.cluster_centers_)

    header = f"    {'Cluster':>8}"
    for f in feature_cols:
        header += f" {f[:16]:>17}"
    print(header)
    for k in range(K):
        row = f"    {k:>8}"
        for i, f in enumerate(feature_cols):
            row += f" {centroids_raw[k, i]:>+17.5f}"
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--long-csv', required=True,
                        help='CSV flips_to_long_*.csv (val recommandé pour OOB)')
    parser.add_argument('--short-csv', required=True,
                        help='CSV flips_to_short_*.csv')
    parser.add_argument('--ks', type=int, nargs='+',
                        default=[5, 7, 10, 15],
                        help='Grid de K à tester')
    parser.add_argument('--features', nargs='+', default=None,
                        help='Features pour clustering (défaut: FEATURES_CLUSTERING)')
    parser.add_argument('--min-size', type=int, default=100,
                        help='Taille min pour être considéré significatif')
    parser.add_argument('--relevant-lift', type=float, default=1.3,
                        help='Lift min (rate / base) pour cluster "relevant"')
    parser.add_argument('--parasite-lift', type=float, default=0.5,
                        help='Lift max pour cluster "parasite"')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 115)
    print(f"CLUSTERING K-MEANS SUR FLIPS MODÈLE")
    print(f"  LONG  : {args.long_csv}")
    print(f"  SHORT : {args.short_csv}")
    print(f"  Ks    : {args.ks}")
    print("=" * 115)

    # Load
    long_csv = Path(args.long_csv)
    short_csv = Path(args.short_csv)
    if not long_csv.exists() or not short_csv.exists():
        print("❌ CSV introuvable")
        return

    df_long = pd.read_csv(long_csv, parse_dates=['flip_dt'])
    df_short = pd.read_csv(short_csv, parse_dates=['flip_dt'])
    df_flips = pd.concat([df_long, df_short], ignore_index=True)
    df_flips = df_flips.sort_values('flip_dt').reset_index(drop=True)
    print(f"\n[1] Flips chargés : {len(df_flips):,} "
          f"(LONG {len(df_long):,} / SHORT {len(df_short):,})")
    print(f"    Période : {df_flips['flip_dt'].min()} → {df_flips['flip_dt'].max()}")

    # Features
    feature_cols = args.features if args.features else FEATURES_CLUSTERING
    missing = [f for f in feature_cols if f not in df_flips.columns]
    if missing:
        print(f"❌ Features manquantes : {missing}")
        return
    print(f"\n[2] Features clustering ({len(feature_cols)}) : {feature_cols}")

    # Vérifier NaN
    X = df_flips[feature_cols].values.astype(np.float64)
    nan_mask = np.isnan(X).any(axis=1)
    n_nan = int(nan_mask.sum())
    if n_nan > 0:
        print(f"    ⚠️ {n_nan} rows avec NaN → exclues du clustering")
        df_flips = df_flips.loc[~nan_mask].reset_index(drop=True)
        X = df_flips[feature_cols].values.astype(np.float64)

    # StandardScaler
    print(f"\n[3] StandardScaler (fit sur {len(df_flips):,} rows)")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Base rates
    base_rate_profit = df_flips['is_profitable_flip'].mean()
    base_rate_good = df_flips['is_good_flip'].mean()
    print(f"\n[4] Base rates :")
    print(f"    is_profitable_flip : {base_rate_profit*100:.2f}%")
    print(f"    is_good_flip       : {base_rate_good*100:.2f}%")
    print(f"    Relevant threshold  (lift ≥ {args.relevant_lift}) : "
          f"rate_profit ≥ {base_rate_profit * args.relevant_lift * 100:.2f}%")
    print(f"    Parasite threshold  (lift ≤ {args.parasite_lift}) : "
          f"rate_profit ≤ {base_rate_profit * args.parasite_lift * 100:.2f}%")

    # Grid K-means
    print(f"\n[5] Grid K-means")
    all_results = {}
    for K in args.ks:
        km, labels, cluster_stats = run_kmeans(
            X_scaled, df_flips, K, base_rate_profit, base_rate_good,
            feature_cols, seed=args.seed)
        cluster_stats = classify_clusters(
            cluster_stats, base_rate_profit,
            min_size=args.min_size,
            relevant_lift=args.relevant_lift,
            parasite_lift=args.parasite_lift)

        print_cluster_table(K, cluster_stats, base_rate_profit)
        print_cluster_centroids(km, feature_cols, scaler, K)

        all_results[K] = {
            'km': km,
            'labels': labels,
            'stats': cluster_stats,
        }

    # Synthèse : pour chaque K, combien de clusters relevant/parasite ?
    print(f"\n{'=' * 115}")
    print(f"SYNTHÈSE — K vs # clusters par catégorie")
    print(f"{'=' * 115}")
    print(f"  {'K':>3} {'Total':>7} {'Relevant':>10} {'Parasite':>10} "
          f"{'Neutral':>10} {'TooSmall':>10} {'NonSig':>10}  "
          f"{'Coverage_rel':>14} {'Rate_rel':>10}")
    for K in args.ks:
        stats = all_results[K]['stats']
        n_rel = sum(1 for s in stats if s['category'] == 'relevant')
        n_par = sum(1 for s in stats if s['category'] == 'parasite')
        n_neu = sum(1 for s in stats if s['category'] == 'neutral')
        n_sm = sum(1 for s in stats if s['category'] == 'too_small')
        n_ns = sum(1 for s in stats if s['category'] == 'non_significant')
        size_rel = sum(s['size'] for s in stats if s['category'] == 'relevant')
        pct_rel = size_rel / len(df_flips) * 100
        if size_rel > 0:
            rate_rel = sum(s['rate_profit'] * s['size'] for s in stats
                            if s['category'] == 'relevant') / size_rel * 100
        else:
            rate_rel = 0
        print(f"  {K:>3} {len(stats):>7} {n_rel:>10} {n_par:>10} "
              f"{n_neu:>10} {n_sm:>10} {n_ns:>10}  "
              f"{pct_rel:>13.2f}% {rate_rel:>9.2f}%")

    # Sauvegarde pour backtest
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Tag basé sur le nom des CSV (sans "flips_to_long_")
    tag = long_csv.stem.replace('flips_to_long_', '')

    # Sauver pour chaque K le scaler + centroïdes + stats
    for K in args.ks:
        res = all_results[K]
        save_path = MODELS_DIR / f'kmeans_k{K}_{tag}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump({
                'scaler': scaler,
                'kmeans': res['km'],
                'feature_cols': feature_cols,
                'K': K,
                'labels': res['labels'],
                'cluster_stats': res['stats'],
                'base_rate_profit': base_rate_profit,
                'base_rate_good': base_rate_good,
                'tag': tag,
            }, f)
        print(f"  ✅ K={K} sauvé: {save_path}  "
              f"({save_path.stat().st_size / 1024:.1f} KB)")

    # Best K suggestion
    print(f"\n{'=' * 115}")
    print("RECOMMANDATION")
    print(f"{'=' * 115}")
    best_K = None
    best_score = -1
    for K in args.ks:
        stats = all_results[K]['stats']
        n_rel = sum(1 for s in stats if s['category'] == 'relevant')
        size_rel = sum(s['size'] for s in stats if s['category'] == 'relevant')
        # Score composite : nombre de clusters relevants × size coverage
        score = n_rel * size_rel / len(df_flips)
        if n_rel > 0 and size_rel / len(df_flips) > 0.2:
            if score > best_score:
                best_K = K
                best_score = score
    if best_K is not None:
        print(f"  K optimal suggéré : {best_K}")
        print(f"  → étape suivante : backtest_with_cluster_filter.py "
              f"avec models/clusters/kmeans_k{best_K}_{tag}.pkl")
    else:
        print(f"  ⚠️ Aucun K ne donne de clusters relevants significatifs")
        print(f"  → réessayer avec paramètres différents ou ajouter features")


if __name__ == '__main__':
    main()
