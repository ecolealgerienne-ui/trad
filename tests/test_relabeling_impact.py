#!/usr/bin/env python3
"""
TEST DE VALIDATION: Impact du Relabeling sur Oracle

Objectif: Valider que le relabeling améliore effectivement la performance
AVANT de perdre du temps à réentraîner.

Test:
1. Charger données TEST uniquement
2. Relabeling EN MÉMOIRE (pas de sauvegarde)
3. Comparer Oracle AVANT vs APRÈS relabeling
4. Métriques: Accuracy, Prédictivité, Trading simulé

Si Oracle APRÈS > Oracle AVANT → Le relabeling est valide ✅
Sinon → Revoir l'approche ❌
"""

import numpy as np
import pandas as pd
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuration relabeling (ajustable via CLI)
CONFIG = None  # Sera initialisé dans main()


def load_dataset(indicator: str):
    """Charge le dataset de test"""
    path = Path(f"data/prepared/dataset_btc_eth_bnb_ada_ltc_{indicator}_dual_binary_kalman.npz")

    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {path}")

    data = np.load(path, allow_pickle=True)

    return {
        'X_test': data['X_test'],
        'Y_test': data['Y_test'],
        'Y_test_pred': data.get('Y_test_pred', None)
    }


def compute_features(returns, force_labels):
    """Calcule vol_rolling et duration"""
    vol_rolling = pd.Series(returns).abs().rolling(window=20).mean().fillna(0).values

    duration = np.zeros_like(force_labels, dtype=int)
    count = 0
    for i in range(len(force_labels)):
        if force_labels[i] == 1:
            count += 1
        else:
            count = 0
        duration[i] = count

    return vol_rolling, duration


def apply_relabeling(X, Y, indicator):
    """
    Applique le relabeling EN MÉMOIRE (pas de sauvegarde)

    Returns:
        Y_relabeled: Labels Force relabelés (Direction inchangée)
        mask_trap: Masque des samples relabelés
    """
    Y_relabeled = Y.copy()

    # Extraction c_ret
    idx_ret = 2 if indicator == 'cci' else 0
    returns = X[:, -1, idx_ret]

    force_labels = Y[:, 1]

    # Calcul métriques
    vol, duration = compute_features(returns, force_labels)

    # Masques pièges
    mask_duration_trap = np.isin(duration, CONFIG['universal']['trap_duration'])

    mask_vol_trap = np.zeros(len(X), dtype=bool)
    if CONFIG['conditional'][indicator]['relabel_high_vol']:
        q4_threshold = np.percentile(vol[vol > 0], 75)
        mask_vol_trap = vol > q4_threshold

    # Mode conditionnel: AND au lieu de OR
    if CONFIG.get('vol_conditional', False):
        mask_trap = mask_duration_trap & mask_vol_trap  # AND
    else:
        mask_trap = mask_duration_trap | mask_vol_trap  # OR

    # Relabeling
    relabeled_count = 0
    for i in np.where(mask_trap)[0]:
        if Y_relabeled[i, 1] == 1:  # Si STRONG
            Y_relabeled[i, 1] = 0   # → WEAK
            relabeled_count += 1

    logger.info(f"   Relabeling: {relabeled_count} labels Force 1→0")
    logger.info(f"   Samples modifiés: {mask_trap.sum()} ({mask_trap.mean()*100:.1f}%)")

    return Y_relabeled, mask_trap


def compute_metrics(Y_true, Y_pred, name):
    """Calcule accuracy Direction et Force"""
    dir_acc = (Y_true[:, 0] == Y_pred[:, 0]).mean()
    force_acc = (Y_true[:, 1] == Y_pred[:, 1]).mean()

    logger.info(f"\n{name}:")
    logger.info(f"   Direction Accuracy: {dir_acc*100:.2f}%")
    logger.info(f"   Force Accuracy:     {force_acc*100:.2f}%")
    logger.info(f"   Moyenne:            {(dir_acc+force_acc)/2*100:.2f}%")

    return dir_acc, force_acc


def compute_predictiveness(Y_labels, X, indicator):
    """
    Calcule la prédictivité: correlation entre labels et returns futurs

    Simule: Si on trade selon Y_labels, quelle serait la correlation avec les gains?
    """
    # Extraire returns
    idx_ret = 2 if indicator == 'cci' else 0
    returns = X[:, -1, idx_ret]

    # Future returns (approximation: on utilise returns[i+1] si disponible)
    future_returns = np.roll(returns, -1)
    future_returns[-1] = 0  # Pas de futur pour le dernier sample

    # Direction prédite
    pred_direction = Y_labels[:, 0]  # 0=DOWN, 1=UP

    # Force prédite
    pred_force = Y_labels[:, 1]  # 0=WEAK, 1=STRONG

    # Calcul prédictivité Direction
    # Si prédit UP (1), on s'attend à future_return > 0
    # Si prédit DOWN (0), on s'attend à future_return < 0
    direction_signal = (pred_direction * 2 - 1)  # 1 ou -1
    dir_correlation = np.corrcoef(direction_signal, future_returns)[0, 1]

    # Calcul prédictivité Force=STRONG
    strong_mask = (pred_force == 1)
    if strong_mask.sum() > 0:
        # Sur les STRONG uniquement, quelle est la correlation?
        strong_correlation = np.corrcoef(
            direction_signal[strong_mask],
            future_returns[strong_mask]
        )[0, 1]
    else:
        strong_correlation = 0.0

    logger.info(f"   Prédictivité Direction:     {dir_correlation:.4f}")
    logger.info(f"   Prédictivité STRONG:        {strong_correlation:.4f}")

    return dir_correlation, strong_correlation


def simulate_trading(Y_labels, X, indicator, name):
    """
    Simule un trading simple selon les labels

    Logique:
    - Si Direction=UP et Force=STRONG → LONG
    - Si Direction=DOWN et Force=STRONG → SHORT
    - Sinon → HOLD
    """
    idx_ret = 2 if indicator == 'cci' else 0
    returns = X[:, -1, idx_ret]
    future_returns = np.roll(returns, -1)
    future_returns[-1] = 0

    direction = Y_labels[:, 0]
    force = Y_labels[:, 1]

    # Signals
    long_signal = (direction == 1) & (force == 1)
    short_signal = (direction == 0) & (force == 1)

    # PnL simulé
    pnl = np.zeros(len(Y_labels))
    pnl[long_signal] = future_returns[long_signal]
    pnl[short_signal] = -future_returns[short_signal]

    # Métriques
    n_trades = long_signal.sum() + short_signal.sum()
    n_long = long_signal.sum()
    n_short = short_signal.sum()

    if n_trades > 0:
        wins = (pnl > 0).sum()
        losses = (pnl < 0).sum()
        win_rate = wins / n_trades if n_trades > 0 else 0

        avg_win = pnl[pnl > 0].mean() if wins > 0 else 0
        avg_loss = pnl[pnl < 0].mean() if losses > 0 else 0

        total_pnl = pnl.sum()

        logger.info(f"\n{name} - Trading Simulé:")
        logger.info(f"   Trades totaux:    {n_trades} (LONG: {n_long}, SHORT: {n_short})")
        logger.info(f"   Win Rate:         {win_rate*100:.2f}%")
        logger.info(f"   PnL Total:        {total_pnl*100:+.2f}%")
        logger.info(f"   Avg Win:          {avg_win*100:+.3f}%")
        logger.info(f"   Avg Loss:         {avg_loss*100:.3f}%")

        if avg_loss != 0:
            profit_factor = -avg_win / avg_loss if avg_loss < 0 else 0
            logger.info(f"   Profit Factor:    {profit_factor:.2f}")

        return {
            'n_trades': n_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    else:
        logger.info(f"\n{name} - Trading Simulé: AUCUN trade (Force=STRONG jamais prédit)")
        return None


def main():
    global CONFIG

    parser = argparse.ArgumentParser(
        description='Test validation impact relabeling'
    )
    parser.add_argument(
        '--indicator',
        default='macd',
        choices=['macd', 'rsi', 'cci'],
        help='Indicateur à tester (default: macd)'
    )
    parser.add_argument(
        '--duration-trap',
        nargs='+',
        type=int,
        default=[3, 4, 5],
        help='Durées STRONG à relabeler (default: 3 4 5). Ex: --duration-trap 3 pour test conservateur'
    )
    parser.add_argument(
        '--vol-conditional',
        action='store_true',
        default=False,
        help='Relabeler uniquement si Duration ET Vol Q4 (AND au lieu de OR)'
    )
    args = parser.parse_args()

    # Initialiser CONFIG
    CONFIG = {
        'universal': {
            'trap_duration': args.duration_trap
        },
        'conditional': {
            'macd': {'relabel_high_vol': True},
            'cci':  {'relabel_high_vol': True},
            'rsi':  {'relabel_high_vol': False}
        },
        'vol_conditional': args.vol_conditional
    }

    logger.info("=" * 80)
    logger.info("TEST DE VALIDATION: Impact Relabeling sur Oracle")
    logger.info("=" * 80)
    logger.info(f"\nIndicateur: {args.indicator.upper()}")
    logger.info("Split: TEST uniquement (out-of-sample)")
    logger.info(f"Durées relabelées: {args.duration_trap}")
    logger.info(f"Mode: {'Vol AND Duration' if args.vol_conditional else 'Vol OR Duration'}")
    logger.info("")

    # 1. Charger données test
    logger.info("📁 Chargement données test...")
    data = load_dataset(args.indicator)

    X_test = data['X_test']
    Y_test_original = data['Y_test']
    Y_test_pred = data['Y_test_pred']

    if Y_test_pred is None:
        logger.error("❌ Y_test_pred absent - exécuter evaluate.py d'abord")
        return

    logger.info(f"   Samples test: {len(X_test)}")

    # 2. Appliquer relabeling EN MÉMOIRE
    logger.info("\n🔄 Application relabeling...")
    Y_test_relabeled, mask_trap = apply_relabeling(X_test, Y_test_original, args.indicator)

    # 3. Comparaison Oracle AVANT vs APRÈS
    logger.info("\n" + "=" * 80)
    logger.info("📊 COMPARAISON ORACLE AVANT vs APRÈS")
    logger.info("=" * 80)

    # Oracle AVANT (original)
    compute_metrics(Y_test_original, Y_test_original, "Oracle AVANT (baseline)")
    compute_predictiveness(Y_test_original, X_test, args.indicator)
    results_before = simulate_trading(Y_test_original, X_test, args.indicator, "Oracle AVANT")

    # Oracle APRÈS (relabeled)
    compute_metrics(Y_test_relabeled, Y_test_relabeled, "Oracle APRÈS (relabeled)")
    compute_predictiveness(Y_test_relabeled, X_test, args.indicator)
    results_after = simulate_trading(Y_test_relabeled, X_test, args.indicator, "Oracle APRÈS")

    # IA (pour référence)
    compute_metrics(Y_test_original, Y_test_pred, "IA (référence - inchangée)")
    compute_predictiveness(Y_test_pred, X_test, args.indicator)
    results_ia = simulate_trading(Y_test_pred, X_test, args.indicator, "IA")

    # 4. Synthèse
    logger.info("\n" + "=" * 80)
    logger.info("🎯 SYNTHÈSE")
    logger.info("=" * 80)

    if results_before and results_after:
        delta_wr = (results_after['win_rate'] - results_before['win_rate']) * 100
        delta_pnl = (results_after['total_pnl'] - results_before['total_pnl']) * 100
        delta_trades = results_after['n_trades'] - results_before['n_trades']

        logger.info(f"\nImpact Relabeling:")
        logger.info(f"   ΔWin Rate:   {delta_wr:+.2f}%")
        logger.info(f"   ΔPnL Total:  {delta_pnl:+.2f}%")
        logger.info(f"   ΔTrades:     {delta_trades:+d}")

        if delta_wr > 0 and delta_pnl > 0:
            logger.info("\n✅ VALIDATION POSITIVE: Relabeling améliore Oracle")
            logger.info("   → GO pour réentraînement avec datasets relabelés")
        elif delta_trades < 0 and delta_pnl > 0:
            logger.info("\n✅ VALIDATION POSITIVE: Moins de trades mais meilleur PnL")
            logger.info("   → GO pour réentraînement (meilleure sélectivité)")
        else:
            logger.info("\n⚠️  VALIDATION MITIGÉE: Relabeling n'améliore pas clairement")
            logger.info("   → Revoir seuils ou approche")

    logger.info("")


if __name__ == "__main__":
    main()
