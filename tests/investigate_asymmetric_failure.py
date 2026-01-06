#!/usr/bin/env python3
"""
Investigation du PnL Brut négatif après fix asymétrique.

Questions à répondre :
1. Quels types de transitions causent les pertes ?
2. La logique d'inversion (LONG → SHORT) crée-t-elle des whipsaws ?
3. Les positions HOLD (Force WEAK) accumulent-elles des pertes ?
4. Quelle est la distribution des durées de trades ?
"""

import numpy as np
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_transitions(trades_log_path: str):
    """
    Analyser les transitions de position et identifier les pertes.

    TODO: Instrumenter test_dual_binary_trading.py pour logger :
    - Chaque changement de position
    - Raison du changement (Force WEAK, Direction change, etc.)
    - PnL au moment de la sortie
    """
    logger.info("⚠️  Nécessite instrumentation du script backtest")
    logger.info("   Ajouter logging détaillé des transitions dans test_dual_binary_trading.py")

def main():
    logger.info("=" * 70)
    logger.info("🔍 INVESTIGATION : Pourquoi PnL Brut négatif ?")
    logger.info("=" * 70)

    # Hypothèses à tester
    hypotheses = [
        {
            'id': 1,
            'question': "La logique HOLD garde-t-elle des positions perdantes trop longtemps ?",
            'test': "Comparer PnL moyen des trades avec durée < 10 vs durée > 20",
            'expected': "Si HOLD est mauvais, les trades longs perdent plus"
        },
        {
            'id': 2,
            'question': "Les inversions (LONG → SHORT) créent-elles des whipsaws ?",
            'test': "Compter combien de reversals vs exits FLAT, et leur PnL",
            'expected': "Si whipsaw, les reversals ont PnL négatif"
        },
        {
            'id': 3,
            'question': "Le Win Rate 28% est-il concentré sur certains types de trades ?",
            'test': "Segmenter par : entrée FLAT→LONG, reversal SHORT→LONG, etc.",
            'expected': "Identifier quels patterns gagnent vs perdent"
        },
        {
            'id': 4,
            'question': "Y a-t-il des moments où Force WEAK = perte systématique ?",
            'test': "Analyser la corrélation Force WEAK (step suivant) et rendement",
            'expected': "Si corrélation négative forte, HOLD est toxique"
        }
    ]

    logger.info("\n📋 HYPOTHÈSES À TESTER :\n")
    for h in hypotheses:
        logger.info(f"❓ #{h['id']}: {h['question']}")
        logger.info(f"   🧪 Test: {h['test']}")
        logger.info(f"   🎯 Attendu: {h['expected']}\n")

    logger.info("=" * 70)
    logger.info("🛠️  PROCHAINES ÉTAPES :")
    logger.info("=" * 70)
    logger.info("1. Instrumenter test_dual_binary_trading.py avec logging détaillé")
    logger.info("2. Générer trades_log.csv avec colonnes :")
    logger.info("   - trade_id, start, end, duration")
    logger.info("   - position_from, position_to (FLAT/LONG/SHORT)")
    logger.info("   - transition_type (ENTRY/EXIT/REVERSAL)")
    logger.info("   - exit_reason (FORCE_WEAK/DIR_CHANGE/DIR_CHANGE_STRONG)")
    logger.info("   - direction_at_entry, force_at_entry")
    logger.info("   - direction_at_exit, force_at_exit")
    logger.info("   - pnl_brut, pnl_after_fees")
    logger.info("3. Analyser ce fichier pour répondre aux 4 questions")
    logger.info("4. Identifier la cause racine du PnL Brut négatif")

    logger.info("\n💡 HYPOTHÈSE PRÉLIMINAIRE :")
    logger.info("   La logique 'HOLD on WEAK' garde des positions perdantes")
    logger.info("   Les reversals (LONG→SHORT sur DIR_CHANGE_STRONG) créent des whipsaws")
    logger.info("   Solution potentielle : EXIT to FLAT au lieu de REVERSAL")

if __name__ == '__main__':
    main()
