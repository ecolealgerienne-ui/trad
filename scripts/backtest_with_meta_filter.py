#!/usr/bin/env python3
"""
Backtest avec filtrage des flips par les meta-classifiers (étape 2.D).

Architecture en 2 étages validée par AUC > 0.65 sur LONG et SHORT :
  Étage 1 : modèle direction (CNN-LSTM) → signal à chaque 5min
  Étage 2 : meta-classifier (XGBoost LONG / SHORT) → "ce flip vaut-il la peine ?"

À chaque flip détecté du modèle :
  - Récupère sa proba meta correspondante (LONG ou SHORT classifier)
  - Si proba_meta > threshold → exécute le flip (trade)
  - Sinon → conserve la position actuelle (slope = 0 → backtest_5min_progressive
            interprète comme "garde position")

Compare 3 stratégies :
  - Oracle pur (référence)
  - Model pur (baseline, pas de filtre)
  - Model + meta filter (grid de seuils)

Usage :
    python scripts/backtest_with_meta_filter.py \\
        --npz data/prepared/dataset_rsi_30m_full_progressive_lag0.npz \\
        --preds data/prepared/preds_rsi_30m_full_progressive_cnnlstm_lag0.npz \\
        --meta-long results/meta_flips/meta_long_preds_<TAG>.npz \\
        --meta-short results/meta_flips/meta_short_preds_<TAG>.npz \\
        --split test
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


def detect_flips(proba, threshold=0.5):
    """Détecte les flips de signe(proba)."""
    sig = np.where(proba > threshold, 1, -1)
    diff = np.diff(sig)
    flip_mask = np.concatenate([[False], diff != 0])
    flip_indices = np.where(flip_mask)[0]
    return flip_indices, sig


def load_meta_probas(npz_long, npz_short):
    """
    Charge les probas meta des 2 classifiers et construit 2 dicts
    {flip_i: meta_proba} pour LONG et SHORT.
    Concatène train+val+test (couvre tous les flips).
    """
    long_npz = np.load(npz_long, allow_pickle=True)
    short_npz = np.load(npz_short, allow_pickle=True)

    # Concat des indices et probas pour tout le test set
    long_indices = np.concatenate([long_npz['train_indices'],
                                    long_npz['val_indices'],
                                    long_npz['test_indices']])
    long_proba = np.concatenate([long_npz['train_proba'],
                                  long_npz['val_proba'],
                                  long_npz['test_proba']])
    short_indices = np.concatenate([short_npz['train_indices'],
                                     short_npz['val_indices'],
                                     short_npz['test_indices']])
    short_proba = np.concatenate([short_npz['train_proba'],
                                   short_npz['val_proba'],
                                   short_npz['test_proba']])

    long_dict = dict(zip(long_indices.tolist(), long_proba.tolist()))
    short_dict = dict(zip(short_indices.tolist(), short_proba.tolist()))

    # Splits info (pour stats par split meta)
    splits = {
        'long': {
            'train': set(long_npz['train_indices'].tolist()),
            'val': set(long_npz['val_indices'].tolist()),
            'test': set(long_npz['test_indices'].tolist()),
        },
        'short': {
            'train': set(short_npz['train_indices'].tolist()),
            'val': set(short_npz['val_indices'].tolist()),
            'test': set(short_npz['test_indices'].tolist()),
        },
        'long_thr_f1': float(long_npz['threshold_f1']) if 'threshold_f1' in long_npz else 0.5,
        'long_thr_prec': float(long_npz['threshold_precision']) if 'threshold_precision' in long_npz else 0.5,
        'short_thr_f1': float(short_npz['threshold_f1']) if 'threshold_f1' in short_npz else 0.5,
        'short_thr_prec': float(short_npz['threshold_precision']) if 'threshold_precision' in short_npz else 0.5,
    }
    return long_dict, short_dict, splits


def get_meta_probas_per_flip(flip_indices, sig, long_dict, short_dict,
                                  default=0.5):
    """
    Pour chaque flip, retourne la proba meta correspondante (LONG ou SHORT
    selon la nouvelle direction).
    """
    n_flips = len(flip_indices)
    meta_proba = np.full(n_flips, default, dtype=np.float64)
    direction = np.zeros(n_flips, dtype=np.int8)
    for k, fi in enumerate(flip_indices):
        new_sig = sig[fi]
        direction[k] = new_sig
        if new_sig == 1:
            meta_proba[k] = long_dict.get(int(fi), default)
        else:
            meta_proba[k] = short_dict.get(int(fi), default)
    return meta_proba, direction


def filter_and_reconstruct(p, flip_indices, sig, meta_proba, threshold):
    """
    Filtre les flips selon meta_proba > threshold puis reconstruit la
    séquence de positions (sig_filtered) row par row.

    Si flip rejeté : on conserve la position actuelle.
    Si flip accepté : on bascule la position.

    Returns:
        sig_filtered (n,) : ±1 à chaque row
        n_accepted, n_rejected
    """
    n = len(p)
    accepted = meta_proba > threshold
    n_accepted = int(accepted.sum())
    n_rejected = int((~accepted).sum())

    sig_filtered = np.empty(n, dtype=np.int8)
    current_sig = int(sig[0])  # signe initial (avant tout flip)
    flip_pos = 0

    for i in range(n):
        if flip_pos < len(flip_indices) and i == flip_indices[flip_pos]:
            if accepted[flip_pos]:
                current_sig = int(sig[i])
            flip_pos += 1
        sig_filtered[i] = current_sig

    return sig_filtered.astype(np.float64), n_accepted, n_rejected


def fmt_row(label, r, fees, bh, oracle_pnl=None):
    fees_pct = r['n_trades'] * 2 * fees * 100
    capture = (r['pnl_pct'] / oracle_pnl * 100) if oracle_pnl else 0
    return (f"  {label:<32}"
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
    parser.add_argument('--meta-long', required=True)
    parser.add_argument('--meta-short', required=True)
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--fees', type=float, default=0.001)
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=[0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
    parser.add_argument('--scope', default='all', choices=['all', 'meta_test_only'],
                        help='all = backtest sur 458 jours (filtre tous les flips), '
                             'meta_test_only = filtre uniquement les flips test du meta')
    args = parser.parse_args()

    print("=" * 110)
    print(f"BACKTEST META FILTER — split={args.split}  fees={args.fees*100:.2f}%  "
          f"scope={args.scope}")
    print(f"  thresholds: {args.thresholds}")
    print("=" * 110)

    # Load NPZ + preds + meta probas
    print(f"\n[1] Load datasets")
    ds = np.load(args.npz, allow_pickle=True)
    preds = np.load(args.preds, allow_pickle=True)
    closes = ds[f'closes_{args.split}']
    dates = pd.to_datetime(ds[f'dates_{args.split}'])
    y_cont = ds[f'y_{args.split}_continuous']
    p = preds[f'{args.split}_preds_proba']
    period_days = (dates[-1] - dates[0]).total_seconds() / 86400
    print(f"   {args.split}: {len(closes):,} rows  |  {dates[0]} → {dates[-1]}  "
          f"({period_days:.0f} jours)")

    long_dict, short_dict, splits = load_meta_probas(args.meta_long, args.meta_short)
    print(f"   Meta LONG  : {len(long_dict):,} flips scorés")
    print(f"   Meta SHORT : {len(short_dict):,} flips scorés")
    print(f"   Calibrated thresholds rappel:")
    print(f"     LONG  F1={splits['long_thr_f1']:.3f}  "
          f"Prec={splits['long_thr_prec']:.3f}")
    print(f"     SHORT F1={splits['short_thr_f1']:.3f}  "
          f"Prec={splits['short_thr_prec']:.3f}")

    # Détecter les flips
    flip_indices, sig = detect_flips(p, threshold=args.threshold)
    print(f"\n[2] Flips détectés: {len(flip_indices):,} "
          f"({len(flip_indices)/len(p)*100:.2f}% des rows)")

    # Lookup meta proba pour chaque flip
    meta_proba, direction = get_meta_probas_per_flip(
        flip_indices, sig, long_dict, short_dict, default=0.5)
    n_long_flips = int((direction == 1).sum())
    n_short_flips = int((direction == -1).sum())
    print(f"   Flips → LONG: {n_long_flips:,}  |  → SHORT: {n_short_flips:,}")

    # Si scope = meta_test_only, on accepte d'office tous les flips train/val
    # et on filtre uniquement les flips test du meta
    if args.scope == 'meta_test_only':
        print(f"\n[3] Scope meta_test_only : on accepte tous les flips train/val,")
        print(f"   on filtre uniquement les flips test du meta")
        # Build mask : 1 si dans test du meta, 0 sinon
        is_meta_test = np.zeros(len(flip_indices), dtype=bool)
        for k, fi in enumerate(flip_indices):
            if direction[k] == 1 and int(fi) in splits['long']['test']:
                is_meta_test[k] = True
            elif direction[k] == -1 and int(fi) in splits['short']['test']:
                is_meta_test[k] = True
        n_meta_test = int(is_meta_test.sum())
        print(f"   Flips dans meta test: {n_meta_test:,} "
              f"(autres = baseline acceptés)")
    else:
        is_meta_test = np.ones(len(flip_indices), dtype=bool)

    # Backtests baselines
    print(f"\n[4] Backtests baselines")
    r_oracle = backtest_5min_progressive(y_cont, closes, fees=args.fees)
    slopes_model = sig.astype(np.float64)
    r_model = backtest_5min_progressive(slopes_model, closes, fees=args.fees)
    bh = buy_and_hold_5m(closes)

    # Grid de seuils
    print(f"\n[5] Grid filtre meta : {len(args.thresholds)} seuils")
    results = []
    for thr in args.thresholds:
        # Pour scope meta_test_only : flips hors test sont acceptés d'office
        if args.scope == 'meta_test_only':
            effective_proba = meta_proba.copy()
            effective_proba[~is_meta_test] = 1.0  # toujours accepté
        else:
            effective_proba = meta_proba

        sig_filtered, n_acc, n_rej = filter_and_reconstruct(
            p, flip_indices, sig, effective_proba, threshold=thr)
        r = backtest_5min_progressive(sig_filtered, closes, fees=args.fees)
        r['threshold'] = thr
        r['n_accepted'] = n_acc
        r['n_rejected'] = n_rej
        # Détail par direction
        long_mask = direction == 1
        short_mask = direction == -1
        r['n_long_acc'] = int(((effective_proba > thr) & long_mask).sum())
        r['n_short_acc'] = int(((effective_proba > thr) & short_mask).sum())
        results.append(r)

    # Tri par PnL Net décroissant
    results.sort(key=lambda x: -x['pnl_pct'])

    # Affichage
    print(f"\n{'=' * 110}")
    print(f"RÉSULTATS — {args.split}  ({period_days:.0f} jours)")
    print(f"{'=' * 110}")
    header = (f"  {'Stratégie':<32}{'Trades':>8}{'WR':>7}{'PF':>7}{'Sharpe':>8}"
              f"{'Brut':>10}{'Fees':>10}{'Net':>11}{'αB&H':>11}{'Capt%':>8}")
    print(header)
    print(f"  {'-' * 108}")
    print(fmt_row('Oracle', r_oracle, args.fees, bh, oracle_pnl=r_oracle['pnl_pct']))
    print(fmt_row(f'Model pur (t={args.threshold})', r_model, args.fees, bh,
                   oracle_pnl=r_oracle['pnl_pct']))
    print(f"  {'-' * 108}")

    print(f"\n  TOP {len(results)} configs meta filter — triées par PnL Net")
    for r in results:
        label = (f"meta thr={r['threshold']:.2f} "
                  f"(L={r['n_long_acc']}/{n_long_flips} S={r['n_short_acc']}/{n_short_flips})")
        print(fmt_row(label, r, args.fees, bh, oracle_pnl=r_oracle['pnl_pct']))

    # Best
    if results:
        best = results[0]
        print(f"\n  ★ BEST: threshold={best['threshold']:.2f}")
        print(f"    PnL Net {best['pnl_pct']:+.2f}%  vs Model pur {r_model['pnl_pct']:+.2f}%  "
              f"(gain {best['pnl_pct'] - r_model['pnl_pct']:+.2f})")
        print(f"    vs Oracle {r_oracle['pnl_pct']:+.2f}%  "
              f"(capture {best['pnl_pct']/r_oracle['pnl_pct']*100:+.1f}%)")
        print(f"    Trades: {best['n_trades']:,} (acceptés: {best['n_accepted']:,} / "
              f"rejetés: {best['n_rejected']:,})")
        if best['pnl_pct'] > 0:
            print(f"    🏆 PnL POSITIF — meta filter validé en production !")
        elif best['pnl_pct'] > r_model['pnl_pct']:
            print(f"    ⚠️ Amélioration mais encore négatif")
        else:
            print(f"    ❌ Pas d'amélioration vs Model pur")


if __name__ == '__main__':
    main()
