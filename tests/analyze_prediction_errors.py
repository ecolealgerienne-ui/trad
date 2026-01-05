#!/usr/bin/env python3
"""
Analyse de concordance Oracle vs Prédictions pour stratégie Dual-Binary.

Compare les positions générées par Oracle (labels parfaits) vs Prédictions (modèle ML)
pour identifier les désaccords et diagnostiquer les problèmes de trading.
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
from typing import Dict, Tuple

# Ajouter src/ au path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from test_dual_binary_trading import (
    load_dataset,
    run_dual_binary_strategy,
    Position,
    logger
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def analyze_position_concordance(
    positions_oracle: np.ndarray,
    positions_pred: np.ndarray,
    returns: np.ndarray,
    Y_oracle: np.ndarray,
    Y_pred: np.ndarray
) -> Dict:
    """
    Analyse détaillée de concordance entre positions Oracle et Prédictions.

    Args:
        positions_oracle: Positions Oracle (n_samples,) {-1, 0, 1}
        positions_pred: Positions Prédictions (n_samples,) {-1, 0, 1}
        returns: Rendements c_ret (n_samples,)
        Y_oracle: Labels Oracle (n_samples, 2)
        Y_pred: Prédictions modèle (n_samples, 2)

    Returns:
        Dictionnaire de statistiques
    """
    n_samples = len(positions_oracle)

    # 1. Concordance globale
    concordance = (positions_oracle == positions_pred).sum() / n_samples

    # 2. Concordance par type de position
    mask_oracle_long = (positions_oracle == Position.LONG.value)
    mask_oracle_short = (positions_oracle == Position.SHORT.value)
    mask_oracle_flat = (positions_oracle == Position.FLAT.value)

    mask_pred_long = (positions_pred == Position.LONG.value)
    mask_pred_short = (positions_pred == Position.SHORT.value)
    mask_pred_flat = (positions_pred == Position.FLAT.value)

    # Concordance LONG
    n_oracle_long = mask_oracle_long.sum()
    n_pred_agrees_long = (positions_pred[mask_oracle_long] == Position.LONG.value).sum() if n_oracle_long > 0 else 0
    concordance_long = n_pred_agrees_long / n_oracle_long if n_oracle_long > 0 else 0.0

    # Concordance SHORT
    n_oracle_short = mask_oracle_short.sum()
    n_pred_agrees_short = (positions_pred[mask_oracle_short] == Position.SHORT.value).sum() if n_oracle_short > 0 else 0
    concordance_short = n_pred_agrees_short / n_oracle_short if n_oracle_short > 0 else 0.0

    # Concordance FLAT
    n_oracle_flat = mask_oracle_flat.sum()
    n_pred_agrees_flat = (positions_pred[mask_oracle_flat] == Position.FLAT.value).sum() if n_oracle_flat > 0 else 0
    concordance_flat = n_pred_agrees_flat / n_oracle_flat if n_oracle_flat > 0 else 0.0

    # 3. Analyse des désaccords
    disagreements = (positions_oracle != positions_pred)
    n_disagreements = disagreements.sum()

    # Types de désaccords
    # Oracle=LONG, Pred!=LONG
    oracle_long_pred_wrong = mask_oracle_long & (positions_pred != Position.LONG.value)
    n_oracle_long_missed = oracle_long_pred_wrong.sum()

    # Oracle=SHORT, Pred!=SHORT
    oracle_short_pred_wrong = mask_oracle_short & (positions_pred != Position.SHORT.value)
    n_oracle_short_missed = oracle_short_pred_wrong.sum()

    # Oracle=FLAT, Pred!=FLAT (faux trades)
    oracle_flat_pred_wrong = mask_oracle_flat & (positions_pred != Position.FLAT.value)
    n_false_trades = oracle_flat_pred_wrong.sum()

    # 4. Impact P&L des désaccords
    # P&L théorique des positions Oracle manquées
    pnl_missed_long = returns[oracle_long_pred_wrong].sum() if n_oracle_long_missed > 0 else 0.0
    pnl_missed_short = (-returns[oracle_short_pred_wrong]).sum() if n_oracle_short_missed > 0 else 0.0

    # P&L théorique des faux trades
    false_long = oracle_flat_pred_wrong & mask_pred_long
    false_short = oracle_flat_pred_wrong & mask_pred_short
    pnl_false_long = returns[false_long].sum() if false_long.sum() > 0 else 0.0
    pnl_false_short = (-returns[false_short]).sum() if false_short.sum() > 0 else 0.0
    pnl_false_total = pnl_false_long + pnl_false_short

    # 5. Analyse Direction vs Force
    # Convertir probabilités en binaire
    oracle_dir = (Y_oracle[:, 0] > 0.5).astype(int)
    oracle_force = (Y_oracle[:, 1] > 0.5).astype(int)
    pred_dir = (Y_pred[:, 0] > 0.5).astype(int)
    pred_force = (Y_pred[:, 1] > 0.5).astype(int)

    # Concordance Direction
    concordance_dir = (oracle_dir == pred_dir).sum() / n_samples

    # Concordance Force
    concordance_force = (oracle_force == pred_force).sum() / n_samples

    # Dans les désaccords de position, quelle est la cause?
    # Direction correcte mais Force incorrecte
    dir_ok_force_wrong = disagreements & (oracle_dir == pred_dir) & (oracle_force != pred_force)
    n_dir_ok_force_wrong = dir_ok_force_wrong.sum()

    # Force correcte mais Direction incorrecte
    force_ok_dir_wrong = disagreements & (oracle_force == pred_force) & (oracle_dir != pred_dir)
    n_force_ok_dir_wrong = force_ok_dir_wrong.sum()

    # Les deux incorrects
    both_wrong = disagreements & (oracle_dir != pred_dir) & (oracle_force != pred_force)
    n_both_wrong = both_wrong.sum()

    return {
        # Concordance globale
        'n_samples': n_samples,
        'concordance_global': concordance,
        'n_disagreements': n_disagreements,

        # Concordance par type
        'n_oracle_long': n_oracle_long,
        'n_oracle_short': n_oracle_short,
        'n_oracle_flat': n_oracle_flat,
        'concordance_long': concordance_long,
        'concordance_short': concordance_short,
        'concordance_flat': concordance_flat,

        # Désaccords détaillés
        'n_oracle_long_missed': n_oracle_long_missed,
        'n_oracle_short_missed': n_oracle_short_missed,
        'n_false_trades': n_false_trades,

        # Impact P&L
        'pnl_missed_long': pnl_missed_long,
        'pnl_missed_short': pnl_missed_short,
        'pnl_false_total': pnl_false_total,

        # Direction vs Force
        'concordance_direction': concordance_dir,
        'concordance_force': concordance_force,
        'n_dir_ok_force_wrong': n_dir_ok_force_wrong,
        'n_force_ok_dir_wrong': n_force_ok_dir_wrong,
        'n_both_wrong': n_both_wrong,
    }


def print_analysis(stats: Dict, indicator: str, split: str):
    """Affiche les statistiques d'analyse."""
    logger.info("\n" + "="*70)
    logger.info(f"📊 ANALYSE CONCORDANCE - {indicator.upper()} ({split})")
    logger.info("="*70)

    # Concordance globale
    logger.info("\n🎯 Concordance Globale:")
    logger.info(f"   Total samples:        {stats['n_samples']:,}")
    logger.info(f"   Concordance:          {stats['concordance_global']*100:.2f}%")
    logger.info(f"   Désaccords:           {stats['n_disagreements']:,} ({stats['n_disagreements']/stats['n_samples']*100:.2f}%)")

    # Distribution positions Oracle
    logger.info("\n📈 Distribution Positions Oracle:")
    logger.info(f"   LONG:                 {stats['n_oracle_long']:,} ({stats['n_oracle_long']/stats['n_samples']*100:.1f}%)")
    logger.info(f"   SHORT:                {stats['n_oracle_short']:,} ({stats['n_oracle_short']/stats['n_samples']*100:.1f}%)")
    logger.info(f"   FLAT:                 {stats['n_oracle_flat']:,} ({stats['n_oracle_flat']/stats['n_samples']*100:.1f}%)")

    # Concordance par type
    logger.info("\n✅ Concordance par Type de Position:")
    logger.info(f"   LONG:                 {stats['concordance_long']*100:.2f}%")
    logger.info(f"   SHORT:                {stats['concordance_short']*100:.2f}%")
    logger.info(f"   FLAT:                 {stats['concordance_flat']*100:.2f}%")

    # Désaccords détaillés
    logger.info("\n❌ Analyse des Désaccords:")
    logger.info(f"   Oracle LONG manqués:  {stats['n_oracle_long_missed']:,} ({stats['n_oracle_long_missed']/stats['n_oracle_long']*100:.1f}% des LONG Oracle)")
    logger.info(f"   Oracle SHORT manqués: {stats['n_oracle_short_missed']:,} ({stats['n_oracle_short_missed']/stats['n_oracle_short']*100:.1f}% des SHORT Oracle)")
    logger.info(f"   Faux trades:          {stats['n_false_trades']:,} (trade alors que Oracle=FLAT)")

    # Impact P&L
    logger.info("\n💰 Impact P&L Théorique des Désaccords:")
    logger.info(f"   P&L LONG manqués:     {stats['pnl_missed_long']:+.2f}%")
    logger.info(f"   P&L SHORT manqués:    {stats['pnl_missed_short']:+.2f}%")
    logger.info(f"   P&L Faux trades:      {stats['pnl_false_total']:+.2f}%")
    logger.info(f"   P&L Total manqué:     {stats['pnl_missed_long'] + stats['pnl_missed_short']:+.2f}%")

    # Direction vs Force
    logger.info("\n🔍 Cause des Désaccords (Direction vs Force):")
    logger.info(f"   Concordance Direction: {stats['concordance_direction']*100:.2f}%")
    logger.info(f"   Concordance Force:     {stats['concordance_force']*100:.2f}%")
    logger.info(f"   Dir OK, Force WRONG:   {stats['n_dir_ok_force_wrong']:,} ({stats['n_dir_ok_force_wrong']/stats['n_disagreements']*100:.1f}% des désaccords)")
    logger.info(f"   Force OK, Dir WRONG:   {stats['n_force_ok_dir_wrong']:,} ({stats['n_force_ok_dir_wrong']/stats['n_disagreements']*100:.1f}% des désaccords)")
    logger.info(f"   Les deux WRONG:        {stats['n_both_wrong']:,} ({stats['n_both_wrong']/stats['n_disagreements']*100:.1f}% des désaccords)")

    # Diagnostic
    logger.info("\n🔬 Diagnostic:")

    # Problème principal?
    if stats['concordance_direction'] < 0.85:
        logger.info("   ⚠️  PROBLÈME PRINCIPAL: Direction mal prédite (<85%)")
    elif stats['concordance_force'] < 0.75:
        logger.info("   ⚠️  PROBLÈME PRINCIPAL: Force mal prédite (<75%)")
    elif stats['n_false_trades'] > stats['n_oracle_long_missed'] + stats['n_oracle_short_missed']:
        logger.info("   ⚠️  PROBLÈME PRINCIPAL: Trop de faux trades (over-trading)")
    else:
        logger.info("   ⚠️  PROBLÈME PRINCIPAL: Opportunités manquées (sous-trading)")

    # Cause dominante des désaccords?
    if stats['n_dir_ok_force_wrong'] > stats['n_force_ok_dir_wrong']:
        pct = stats['n_dir_ok_force_wrong'] / stats['n_disagreements'] * 100
        logger.info(f"   ⚠️  CAUSE DOMINANTE: Erreurs Force ({pct:.1f}% des désaccords)")
    else:
        pct = stats['n_force_ok_dir_wrong'] / stats['n_disagreements'] * 100
        logger.info(f"   ⚠️  CAUSE DOMINANTE: Erreurs Direction ({pct:.1f}% des désaccords)")

    logger.info("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyse concordance Oracle vs Prédictions pour Dual-Binary"
    )
    parser.add_argument(
        '--indicator',
        required=True,
        choices=['rsi', 'macd', 'cci'],
        help="Indicateur à analyser"
    )
    parser.add_argument(
        '--split',
        default='test',
        choices=['train', 'val', 'test'],
        help="Split à analyser (défaut: test)"
    )
    parser.add_argument(
        '--fees',
        type=float,
        default=0.1,
        help="Frais par trade en %% (défaut: 0.1%%)"
    )
    parser.add_argument(
        '--threshold-force',
        type=float,
        default=0.5,
        help="Seuil Force (défaut: 0.5)"
    )

    args = parser.parse_args()

    # Convertir fees
    fees_decimal = args.fees / 100.0

    # Charger données
    logger.info(f"\n📂 Chargement dataset: {args.indicator.upper()} ({args.split})")
    data = load_dataset(args.indicator, args.split)

    if data['Y_pred'] is None:
        logger.error("❌ Pas de prédictions disponibles dans le dataset")
        logger.error("   Exécuter d'abord: python src/evaluate.py --data <dataset>")
        sys.exit(1)

    # Extraire returns
    X = data['X']
    if args.indicator in ['rsi', 'macd']:
        returns = X[:, -1, 0]  # c_ret à index 0 pour RSI/MACD
    else:  # cci
        returns = X[:, -1, 2]  # c_ret à index 2 pour CCI

    logger.info(f"   ✅ Chargé: {len(data['Y']):,} samples")

    # Générer positions Oracle
    logger.info("\n🎯 Génération positions Oracle...")
    positions_oracle, _ = run_dual_binary_strategy(
        Y=data['Y'],
        returns=returns,
        fees=fees_decimal,
        use_predictions=False,
        threshold_force=args.threshold_force,
        verbose=False
    )

    # Générer positions Prédictions
    logger.info("🎯 Génération positions Prédictions...")
    positions_pred, _ = run_dual_binary_strategy(
        Y=data['Y'],
        returns=returns,
        fees=fees_decimal,
        use_predictions=True,
        Y_pred=data['Y_pred'],
        threshold_force=args.threshold_force,
        verbose=False
    )

    # Analyser concordance
    logger.info("🔍 Analyse concordance...")
    stats = analyze_position_concordance(
        positions_oracle=positions_oracle,
        positions_pred=positions_pred,
        returns=returns,
        Y_oracle=data['Y'],
        Y_pred=data['Y_pred']
    )

    # Afficher résultats
    print_analysis(stats, args.indicator, args.split)

    logger.info("\n✅ Analyse terminée")


if __name__ == '__main__':
    main()
