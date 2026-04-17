#!/usr/bin/env python3
"""
Cross-validation entre les 3 indicateurs (MACD, RSI, CCI).

Objectif : mesurer si les erreurs des 3 modèles sont corrélées ou décorrélées,
pour décider si un consensus (majorité 2/3 ou unanimité 3/3) a du sens.

Prérequis : alignement confirmé par validate_indicator_alignment.py
(mêmes dates_5min, mêmes closes, mêmes df_5m/df_tf entre les 3 NPZ).

Sections :
  [1] Load + alignment sanity check
  [2] Corrélation des probas (Pearson) entre paires
  [3] Accord binaire (sign match) entre paires
  [4] Diversité des erreurs (conditionnelles vs oracle respectif)
  [5] Accuracy du consensus (majorité vs oracle-consensus)
  [6] Backtest comparatif 6 stratégies :
        Oracle MACD / Oracle CCI / Oracle RSI
        Model MACD / Model CCI / Model RSI
        Consensus Majorité (2/3)
        Consensus Unanimité (3/3, sinon conserve position)

Usage :
    python scripts/cross_validation_indicators.py --split test
    python scripts/cross_validation_indicators.py --split val --fees 0.001
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import backtest_5min_progressive

PREP_DIR = Path('data/prepared')
INDICATORS = ['macd', 'cci', 'rsi']


def buy_and_hold_5m(closes):
    first, last = closes[0], closes[-1]
    if np.isnan(first) or np.isnan(last) or first == 0:
        return 0.0
    return (last - first) / first * 100


def fmt_row(label, r, bh, oracle_ref_pnl=None):
    """Format une ligne de résultat backtest."""
    fees_pct = r['n_trades'] * 2 * 0.001 * 100  # fees assumés 0.1%
    alpha = r['pnl_pct'] - bh
    capture = ''
    if oracle_ref_pnl is not None and oracle_ref_pnl != 0:
        capture = f"{r['pnl_pct']/oracle_ref_pnl*100:+7.1f}%"
    else:
        capture = f"{'—':>8}"
    return (f"  {label:<28}"
            f"{r['n_trades']:>8,} "
            f"{r['win_rate']:>6.1f}% "
            f"{r['profit_factor']:>6.2f} "
            f"{r['sharpe']:>7.3f} "
            f"{r['pnl_pct']+fees_pct:>+9.2f}% "
            f"{fees_pct:>9.2f}% "
            f"{r['pnl_pct']:>+10.2f}% "
            f"{alpha:>+10.2f}% "
            f"{capture:>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', type=int, default=30)
    parser.add_argument('--period', default='full')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--fees', type=float, default=0.001)
    args = parser.parse_args()

    tf_label = f'{args.tf}m' if args.tf < 60 else '1h'
    print("=" * 100)
    print(f"CROSS-VALIDATION INDICATORS — {INDICATORS}  "
          f"tf={tf_label}  period={args.period}  split={args.split}  "
          f"fees={args.fees*100:.2f}%")
    print("=" * 100)

    # ========================================================================
    # [1] Load
    # ========================================================================
    print(f"\n[1] Load datasets + preds")
    dsets = {}
    preds = {}
    for ind in INDICATORS:
        ds_path = PREP_DIR / f'dataset_{ind}_{tf_label}_{args.period}_progressive.npz'
        pr_path = PREP_DIR / f'preds_{ind}_{tf_label}_{args.period}_progressive.npz'
        if not ds_path.exists() or not pr_path.exists():
            print(f"  ❌ Manquant pour {ind}: {ds_path.name} ou {pr_path.name}")
            return
        dsets[ind] = np.load(ds_path, allow_pickle=True)
        preds[ind] = np.load(pr_path, allow_pickle=True)
        print(f"  ✅ {ind.upper()}: dataset {ds_path.stat().st_size/1024/1024:.1f} MB, "
              f"preds {pr_path.stat().st_size/1024:.0f} KB")

    # Sanity alignment quick check
    closes_ref = dsets[INDICATORS[0]][f'closes_{args.split}']
    for ind in INDICATORS[1:]:
        if not np.array_equal(dsets[ind][f'closes_{args.split}'], closes_ref):
            print(f"  ❌ ERREUR: closes_{args.split} désalignés entre "
                  f"{INDICATORS[0]} et {ind} → stoppe")
            return
    print(f"  ✅ Alignement {args.split}: closes identiques entre les 3 indicateurs "
          f"(n={len(closes_ref):,})")

    # Extract arrays
    p = {ind: preds[ind][f'{args.split}_preds_proba'] for ind in INDICATORS}
    y_true = {ind: dsets[ind][f'y_{args.split}_binary'] for ind in INDICATORS}
    y_cont = {ind: dsets[ind][f'y_{args.split}_continuous'] for ind in INDICATORS}
    closes = closes_ref
    dates = pd.to_datetime(dsets[INDICATORS[0]][f'dates_{args.split}'])
    period_days = (dates[-1] - dates[0]).total_seconds() / 86400

    # Signes binaires ±1 pour chaque modèle
    sig_bin = {ind: np.where(p[ind] > 0.5, 1.0, -1.0) for ind in INDICATORS}

    print(f"  {args.split}: {len(closes):,} rows  |  "
          f"{dates[0]} → {dates[-1]}  ({period_days:.0f} jours)")

    # ========================================================================
    # [2] Corrélation des probas (Pearson)
    # ========================================================================
    print(f"\n[2] Corrélation des probas (Pearson)")
    print(f"              {INDICATORS[0].upper():>8} {INDICATORS[1].upper():>8} "
          f"{INDICATORS[2].upper():>8}")
    corr_matrix = np.corrcoef([p[ind] for ind in INDICATORS])
    for i, ind_i in enumerate(INDICATORS):
        row = f"  {ind_i.upper():<10}  "
        for j in range(3):
            row += f"{corr_matrix[i, j]:>+8.4f} "
        print(row)
    mean_off = (corr_matrix.sum() - 3) / 6  # moyenne hors diagonale
    print(f"  → Corrélation moyenne hors diagonale: {mean_off:+.4f}")

    # ========================================================================
    # [3] Accord binaire entre paires
    # ========================================================================
    print(f"\n[3] Accord binaire (sign identique)")
    for i, a in enumerate(INDICATORS):
        for b in INDICATORS[i+1:]:
            agree = (sig_bin[a] == sig_bin[b]).mean() * 100
            print(f"  {a.upper()} == {b.upper():<6}: {agree:.2f}% "
                  f"({int((sig_bin[a] == sig_bin[b]).sum()):,}/{len(p[a]):,})")

    # Accord triple
    all_up = (sig_bin['macd'] > 0) & (sig_bin['cci'] > 0) & (sig_bin['rsi'] > 0)
    all_dn = (sig_bin['macd'] < 0) & (sig_bin['cci'] < 0) & (sig_bin['rsi'] < 0)
    unanim_pct = (all_up | all_dn).mean() * 100
    print(f"  Unanimité 3/3  : {unanim_pct:.2f}%  "
          f"(UP: {all_up.mean()*100:.2f}%, DOWN: {all_dn.mean()*100:.2f}%)")

    # ========================================================================
    # [4] Diversité des erreurs (vs son propre oracle)
    # ========================================================================
    print(f"\n[4] Diversité des erreurs (chacun vs son propre oracle)")
    err = {}
    for ind in INDICATORS:
        pred_bin = (p[ind] > 0.5).astype(int)
        err[ind] = (pred_bin != y_true[ind])
        print(f"  {ind.upper()}: {err[ind].mean()*100:.2f}% erreurs "
              f"({err[ind].sum():,}/{len(err[ind]):,})")

    # Conditional error rate : P(err_b | err_a)
    print(f"\n  Erreur conditionnelle : P(err[b] = 1 | err[a] = 1)")
    print(f"  (Si faible → erreurs décorrélées → consensus utile)")
    print(f"  (Si élevé  → erreurs corrélées   → consensus inefficace)")
    for a in INDICATORS:
        for b in INDICATORS:
            if a == b:
                continue
            mask = err[a]
            cond = err[b][mask].mean() * 100 if mask.sum() > 0 else 0
            baseline = err[b].mean() * 100
            ratio = cond / baseline if baseline > 0 else 0
            print(f"    P(err[{b}] | err[{a}]) = {cond:.2f}%  "
                  f"(baseline {baseline:.2f}%, ratio {ratio:.2f}×)")

    # Erreurs simultanées 3/3
    all_err = err['macd'] & err['cci'] & err['rsi']
    print(f"\n  Erreur simultanée des 3 modèles: {all_err.mean()*100:.2f}% "
          f"({all_err.sum():,} rows)")

    # ========================================================================
    # [5] Consensus : accuracy vs oracle-consensus
    # ========================================================================
    print(f"\n[5] Accuracy du vote majoritaire (2/3) vs oracle-consensus")
    # Oracle consensus : majorité des 3 oracles
    oracle_votes = (y_true['macd'].astype(int)
                     + y_true['cci'].astype(int)
                     + y_true['rsi'].astype(int))
    oracle_majority = (oracle_votes >= 2).astype(int)

    # Model consensus : majorité des 3 prédictions
    pred_votes = ((p['macd'] > 0.5).astype(int)
                  + (p['cci'] > 0.5).astype(int)
                  + (p['rsi'] > 0.5).astype(int))
    pred_majority = (pred_votes >= 2).astype(int)

    # Accuracy individuelle vs oracle-majority
    for ind in INDICATORS:
        pred_bin = (p[ind] > 0.5).astype(int)
        acc = (pred_bin == oracle_majority).mean() * 100
        print(f"  Acc({ind.upper()} vs oracle-maj): {acc:.2f}%")
    acc_cons = (pred_majority == oracle_majority).mean() * 100
    print(f"  Acc(CONSENSUS-MAJ vs oracle-maj): {acc_cons:.2f}%")

    # ========================================================================
    # [6] Backtest comparatif
    # ========================================================================
    print(f"\n[6] Backtest — comparaison Oracles / Models individuels / Consensus")

    bh = buy_and_hold_5m(closes)
    results = {}

    # 3 Oracles
    for ind in INDICATORS:
        results[f'Oracle {ind.upper()}'] = backtest_5min_progressive(
            y_cont[ind], closes, fees=args.fees)

    # 3 Models individuels
    for ind in INDICATORS:
        results[f'Model {ind.upper()}'] = backtest_5min_progressive(
            sig_bin[ind], closes, fees=args.fees)

    # Consensus Majorité (2/3) : vote signé
    vote_sum = sig_bin['macd'] + sig_bin['cci'] + sig_bin['rsi']
    slopes_maj = np.sign(vote_sum)  # ±1 (pas de 0 avec 3 votes impairs)
    results['Consensus Majorité 2/3'] = backtest_5min_progressive(
        slopes_maj, closes, fees=args.fees)

    # Consensus Unanimité (3/3) : 0 sinon → conserve position
    slopes_unanim = np.where(np.abs(vote_sum) == 3, np.sign(vote_sum), 0.0)
    # Info sur le % de rows avec action
    n_action = (slopes_unanim != 0).sum()
    results['Consensus Unanimité 3/3'] = backtest_5min_progressive(
        slopes_unanim, closes, fees=args.fees)

    # Affichage tableau
    print(f"\n{'=' * 100}")
    print(f"RÉSULTATS — {args.split}  ({period_days:.0f} jours)")
    print(f"{'=' * 100}")
    header = (f"  {'Stratégie':<28}{'Trades':>8}  {'WR':>7}{'PF':>7}"
              f"{'Sharpe':>8}{'Brut':>10}{'Fees':>10}{'Net':>11}{'αB&H':>11}"
              f"{'Capt':>8}")
    print(header)
    print(f"  {'-' * 95}")

    # Choisir une référence Oracle pour "Capt" → moyenne des 3
    mean_oracle_pnl = np.mean([results[f'Oracle {ind.upper()}']['pnl_pct']
                                 for ind in INDICATORS])

    for label, r in results.items():
        print(fmt_row(label, r, bh, oracle_ref_pnl=mean_oracle_pnl))

    print(f"  {'-' * 95}")
    print(f"  {'Buy & Hold':<28}{'—':>8}  {'—':>7}{'—':>7}{'—':>8}"
          f"{'—':>10}{'—':>10}{bh:>+10.2f}%")

    # Info action unanimité
    print(f"\n  Unanimité 3/3 : action sur {n_action:,}/{len(slopes_unanim):,} rows "
          f"({n_action/len(slopes_unanim)*100:.2f}%)")

    # ========================================================================
    # Synthèse
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("SYNTHÈSE")
    print(f"{'=' * 100}")
    best_individual = max(((f'Model {i.upper()}', results[f'Model {i.upper()}']['pnl_pct'])
                           for i in INDICATORS), key=lambda x: x[1])
    cons_maj = results['Consensus Majorité 2/3']['pnl_pct']
    cons_un = results['Consensus Unanimité 3/3']['pnl_pct']
    print(f"  Meilleur modèle individuel : {best_individual[0]}  "
          f"PnL Net {best_individual[1]:+.2f}%")
    print(f"  Consensus Majorité 2/3     : PnL Net {cons_maj:+.2f}%  "
          f"(Δ vs best individuel: {cons_maj - best_individual[1]:+.2f}%)")
    print(f"  Consensus Unanimité 3/3    : PnL Net {cons_un:+.2f}%  "
          f"(Δ vs best individuel: {cons_un - best_individual[1]:+.2f}%)")
    print(f"\n  → Corrélation moyenne probas : {mean_off:+.4f}")
    print(f"  → Erreur simultanée 3/3       : {all_err.mean()*100:.2f}%")
    if mean_off > 0.8 and all_err.mean() > 0.5:
        print(f"  → VERDICT: erreurs fortement corrélées → consensus peu utile")
    elif mean_off < 0.5 or all_err.mean() < 0.2:
        print(f"  → VERDICT: erreurs décorrélées → consensus potentiellement utile")
    else:
        print(f"  → VERDICT: corrélation modérée → tester consensus pondéré/meta")


if __name__ == '__main__':
    main()
