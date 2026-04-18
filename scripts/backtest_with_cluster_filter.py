#!/usr/bin/env python3
"""
Backtest avec filtrage des flips par clusters K-means (étape 2 de l'idée
utilisateur — clustering non-supervisé des régimes).

Pipeline :
  1. Charge le modèle K-means fit sur flips val (scaler + centroïdes + stats)
  2. Charge les flips test (CSV avec features déjà calculées)
  3. Standardise features avec le scaler train (val) — PAS de refit
  4. Prédit le cluster de chaque flip test (centroïde le plus proche)
  5. Filtre selon catégorie cluster :
     - Mode 'relevant_only' : garde les flips dans clusters 'relevant'
     - Mode 'non_parasite'  : garde tout SAUF clusters 'parasite'
  6. Reconstitue les slopes filtrées (row par row, 5min)
  7. Backtest via core.backtest_5min_progressive
  8. Compare Oracle / Model pur / Model filtré par cluster

Usage :
    python scripts/backtest_with_cluster_filter.py \\
        --npz data/prepared/dataset_rsi_30m_full_progressive_lag0.npz \\
        --preds data/prepared/preds_rsi_30m_full_progressive_cnnlstm_lag0.npz \\
        --long-test-csv  results/flips/flips_to_long_rsi_30m_full_cnnlstm_lag0_test.csv \\
        --short-test-csv results/flips/flips_to_short_rsi_30m_full_cnnlstm_lag0_test.csv \\
        --kmeans-pkls models/clusters/kmeans_k10_*.pkl models/clusters/kmeans_k15_*.pkl
"""

import argparse
import pickle
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


def detect_flips(proba, threshold=0.5):
    sig = np.where(proba > threshold, 1, -1)
    diff = np.diff(sig)
    flip_mask = np.concatenate([[False], diff != 0])
    flip_indices = np.where(flip_mask)[0]
    return flip_indices, sig


def reconstruct_slopes_from_flips(n, flip_indices, sig, accepted_mask):
    """
    Reconstitue la série sig_filtered en appliquant uniquement les flips acceptés.

    Si flip accepté à i : sig_filtered[i:] = sig[i] jusqu'au prochain flip accepté
    Si flip rejeté : on conserve current_sig
    """
    sig_filtered = np.empty(n, dtype=np.int8)
    current_sig = int(sig[0])
    flip_pos = 0
    n_flips = len(flip_indices)

    for i in range(n):
        if flip_pos < n_flips and i == flip_indices[flip_pos]:
            if accepted_mask[flip_pos]:
                current_sig = int(sig[i])
            flip_pos += 1
        sig_filtered[i] = current_sig

    return sig_filtered.astype(np.float64)


def assign_clusters(df_flips, scaler, km, feature_cols):
    """Standardize features et prédit le cluster pour chaque flip."""
    X = df_flips[feature_cols].values.astype(np.float64)
    nan_mask = np.isnan(X).any(axis=1)
    # Pour les NaN, on remplace par 0 (ne sera pas filtré)
    X_clean = np.where(np.isnan(X), 0.0, X)
    X_scaled = scaler.transform(X_clean)
    cluster_ids = km.predict(X_scaled)
    cluster_ids[nan_mask] = -1  # -1 = invalide
    return cluster_ids, nan_mask


def fmt_row(label, r, fees, bh, oracle_pnl=None):
    fees_pct = r['n_trades'] * 2 * fees * 100
    capture = (r['pnl_pct'] / oracle_pnl * 100) if oracle_pnl else 0
    return (f"  {label:<40}"
            f"{r['n_trades']:>8,} "
            f"{r['win_rate']:>6.1f}% "
            f"{r['profit_factor']:>6.2f} "
            f"{r['sharpe']:>7.3f} "
            f"{r['pnl_pct']+fees_pct:>+9.2f}% "
            f"{fees_pct:>9.2f}% "
            f"{r['pnl_pct']:>+10.2f}% "
            f"{r['pnl_pct']-bh:>+10.2f}% "
            f"{capture:>+7.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', required=True)
    parser.add_argument('--preds', required=True)
    parser.add_argument('--long-test-csv', required=True,
                        help='CSV flips test LONG (généré par extract_model_flips.py --split test)')
    parser.add_argument('--short-test-csv', required=True,
                        help='CSV flips test SHORT')
    parser.add_argument('--kmeans-pkls', nargs='+', required=True,
                        help='1+ paths vers kmeans_k*_*.pkl à tester')
    parser.add_argument('--split', default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--fees', type=float, default=0.001)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--min-lifts', type=float, nargs='+',
                        default=[1.3, 1.5, 1.8, 2.0],
                        help='Seuils min_lift à tester (profitable_rate / base_rate). '
                             'Plus haut = plus sélectif.')
    parser.add_argument('--no-non-parasite', action='store_true',
                        help='Désactive mode non_parasite (garde seulement relevant_only)')
    args = parser.parse_args()

    print("=" * 115)
    print(f"BACKTEST CLUSTER FILTER — split={args.split}  fees={args.fees*100:.2f}%")
    print(f"  kmeans models: {len(args.kmeans_pkls)}")
    print("=" * 115)

    # 1. Load NPZ + preds
    ds = np.load(args.npz, allow_pickle=True)
    preds = np.load(args.preds, allow_pickle=True)
    closes = ds[f'closes_{args.split}']
    dates = pd.to_datetime(ds[f'dates_{args.split}'])
    y_cont = ds[f'y_{args.split}_continuous']
    p = preds[f'{args.split}_preds_proba']
    period_days = (dates[-1] - dates[0]).total_seconds() / 86400
    print(f"\n[1] {args.split}: {len(closes):,} rows  |  "
          f"{dates[0]} → {dates[-1]}  ({period_days:.0f} jours)")

    # 2. Charger les flips test CSV
    df_long_test = pd.read_csv(args.long_test_csv, parse_dates=['flip_dt'])
    df_short_test = pd.read_csv(args.short_test_csv, parse_dates=['flip_dt'])
    df_flips_test = pd.concat([df_long_test, df_short_test], ignore_index=True)
    df_flips_test = df_flips_test.sort_values('flip_i').reset_index(drop=True)
    print(f"\n[2] Flips test: LONG {len(df_long_test):,} + SHORT {len(df_short_test):,} "
          f"= {len(df_flips_test):,} total")

    # 3. Détecter les flips dans les preds (sanity check : même count que CSV)
    flip_indices, sig = detect_flips(p, threshold=args.threshold)
    print(f"   Flips détectés dans preds: {len(flip_indices):,}")
    assert len(flip_indices) == len(df_flips_test), \
        f"Mismatch flip count: preds={len(flip_indices)} vs CSV={len(df_flips_test)}"
    # Vérifier alignement flip_i
    assert np.array_equal(flip_indices, df_flips_test['flip_i'].values), \
        "Mismatch flip_i entre preds et CSV"
    print(f"   ✅ Alignement preds ↔ CSV OK")

    # 4. Backtests baselines
    print(f"\n[3] Baselines")
    r_oracle = backtest_5min_progressive(y_cont, closes, fees=args.fees)
    slopes_model = sig.astype(np.float64)
    r_model = backtest_5min_progressive(slopes_model, closes, fees=args.fees)
    bh = buy_and_hold_5m(closes)
    print(f"   Oracle PnL Net: {r_oracle['pnl_pct']:+.2f}%  "
          f"({r_oracle['n_trades']} trades)")
    print(f"   Model pur PnL Net: {r_model['pnl_pct']:+.2f}%  "
          f"({r_model['n_trades']} trades)")

    # 5. Pour chaque kmeans model, tester les 2 modes (relevant_only, non_parasite)
    all_results = []
    for pkl_path in args.kmeans_pkls:
        pkl_file = Path(pkl_path)
        if not pkl_file.exists():
            print(f"⚠️ Pickle introuvable: {pkl_file}")
            continue
        with open(pkl_file, 'rb') as f:
            cluster_data = pickle.load(f)
        scaler = cluster_data['scaler']
        km = cluster_data['kmeans']
        cluster_stats = cluster_data['cluster_stats']
        feature_cols = cluster_data['feature_cols']
        K = cluster_data['K']

        print(f"\n{'=' * 115}")
        print(f"K-MEANS K={K}  ({pkl_file.name})")
        print(f"{'=' * 115}")

        # Catégorisation des clusters
        relevant_ids = [s['k_id'] for s in cluster_stats
                         if s['category'] == 'relevant']
        parasite_ids = [s['k_id'] for s in cluster_stats
                         if s['category'] == 'parasite']
        print(f"  Relevant (cat='relevant') : {relevant_ids}")
        print(f"  Parasite (cat='parasite') : {parasite_ids}")

        # Assigner les flips test aux clusters
        cluster_ids_test, nan_mask = assign_clusters(
            df_flips_test, scaler, km, feature_cols)
        n_nan = int(nan_mask.sum())
        if n_nan > 0:
            print(f"  ⚠️ {n_nan} flips avec NaN → cluster=-1 (non-filtré)")

        # Distribution des flips test par cluster
        print(f"\n  Distribution flips test par cluster :")
        unique, counts = np.unique(cluster_ids_test, return_counts=True)
        # Map cluster_id → stats pour lookup
        cluster_stats_by_id = {s['k_id']: s for s in cluster_stats}
        for cid, cnt in zip(unique, counts):
            cat = 'unknown'
            rate_val = 0
            lift_val = 0
            if int(cid) in cluster_stats_by_id:
                s = cluster_stats_by_id[int(cid)]
                cat = s['category']
                rate_val = s['rate_profit'] * 100
                lift_val = s['lift_profit']
            marker = ('🔥' if cat == 'relevant'
                       else '❌' if cat == 'parasite'
                       else '')
            print(f"    Cluster {int(cid):>3} : {int(cnt):>5,} flips  "
                  f"category={cat:<15} val_rate={rate_val:>5.2f}% "
                  f"lift={lift_val:>4.2f}×  {marker}")

        # Grid de min_lifts pour relevant_only
        for min_lift in args.min_lifts:
            relevant_ids_at_lift = [
                s['k_id'] for s in cluster_stats
                if s['category'] == 'relevant' and s['lift_profit'] >= min_lift
            ]
            accepted_mask = np.isin(cluster_ids_test, relevant_ids_at_lift)
            n_acc = int(accepted_mask.sum())
            if n_acc == 0:
                continue  # skip si pas de clusters sélectionnés
            sig_filt = reconstruct_slopes_from_flips(
                len(p), flip_indices, sig, accepted_mask)
            r = backtest_5min_progressive(sig_filt, closes, fees=args.fees)
            all_results.append({
                'K': K, 'mode': f'rel_lift>={min_lift}',
                'min_lift': min_lift,
                'n_clusters': len(relevant_ids_at_lift),
                'result': r,
                'n_accepted': n_acc,
                'n_rejected': len(cluster_ids_test) - n_acc,
            })

        # Mode non_parasite (inchangé, optionnel)
        if not args.no_non_parasite and parasite_ids:
            accepted_mask_np = ~np.isin(cluster_ids_test, parasite_ids)
            n_acc_np = int(accepted_mask_np.sum())
            sig_filt_np = reconstruct_slopes_from_flips(
                len(p), flip_indices, sig, accepted_mask_np)
            r_np = backtest_5min_progressive(sig_filt_np, closes, fees=args.fees)
            all_results.append({
                'K': K, 'mode': 'non_parasite',
                'min_lift': None,
                'n_clusters': K - len(parasite_ids),
                'result': r_np,
                'n_accepted': n_acc_np,
                'n_rejected': len(cluster_ids_test) - n_acc_np,
            })

    # Affichage comparatif
    print(f"\n{'=' * 115}")
    print(f"RÉSULTATS — {args.split}  ({period_days:.0f} jours)")
    print(f"{'=' * 115}")
    header = (f"  {'Stratégie':<40}{'Trades':>8}{'WR':>7}{'PF':>7}{'Sharpe':>8}"
              f"{'Brut':>10}{'Fees':>10}{'Net':>11}{'αB&H':>11}{'Capt%':>8}")
    print(header)
    print(f"  {'-' * 113}")
    print(fmt_row('Oracle', r_oracle, args.fees, bh, oracle_pnl=r_oracle['pnl_pct']))
    print(fmt_row('Model pur', r_model, args.fees, bh, oracle_pnl=r_oracle['pnl_pct']))
    print(f"  {'-' * 113}")

    # Trier par PnL Net décroissant pour lisibilité
    all_results.sort(key=lambda x: -x['result']['pnl_pct'])
    for res in all_results:
        nc = res.get('n_clusters', '?')
        label = (f"K={res['K']:>2} {res['mode']:<18} "
                  f"(C={nc} kept={res['n_accepted']:,})")
        print(fmt_row(label, res['result'], args.fees, bh,
                       oracle_pnl=r_oracle['pnl_pct']))

    # Best
    if all_results:
        best = all_results[0]
        print(f"\n  ★ BEST: K={best['K']} mode={best['mode']} "
              f"(C={best.get('n_clusters', '?')})")
        print(f"    PnL Net {best['result']['pnl_pct']:+.2f}%  "
              f"vs Model pur {r_model['pnl_pct']:+.2f}%  "
              f"(gain {best['result']['pnl_pct'] - r_model['pnl_pct']:+.2f})")
        print(f"    vs Oracle {r_oracle['pnl_pct']:+.2f}%  "
              f"(capture {best['result']['pnl_pct']/r_oracle['pnl_pct']*100:+.1f}%)")
        print(f"    Trades: {best['result']['n_trades']:,}  "
              f"(accepted flips: {best['n_accepted']:,})")
        if best['result']['pnl_pct'] > 0:
            print(f"    🏆 PnL POSITIF — filter cluster validé !")
        elif best['result']['pnl_pct'] > r_model['pnl_pct']:
            print(f"    ⚠️ Amélioration mais encore négatif")
        else:
            print(f"    ❌ Pas d'amélioration vs Model pur")


if __name__ == '__main__':
    main()
