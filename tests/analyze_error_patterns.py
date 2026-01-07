#!/usr/bin/env python3
"""
Analyse des Patterns d'Erreurs - Témoins vs Décideur

OBJECTIF:
Quand un indicateur (décideur) se trompe, que disent les autres (témoins)?
Identifier les patterns récurrents pour créer des règles de veto.

Usage:
    python tests/analyze_error_patterns.py --decider macd --split test --max-samples 20000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import argparse
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import Counter
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class ErrorContext:
    """Contexte d'une erreur du décideur."""
    step: int

    # Décideur (celui qui se trompe)
    decider_pred_dir: int
    decider_pred_force: int
    decider_oracle_dir: int
    decider_oracle_force: int

    # Témoin 1
    witness1_name: str
    witness1_pred_dir: int
    witness1_pred_force: int
    witness1_oracle_dir: int

    # Témoin 2
    witness2_name: str
    witness2_pred_dir: int
    witness2_pred_force: int
    witness2_oracle_dir: int


@dataclass
class InsightPattern:
    """Pattern d'insight identifié."""
    name: str
    description: str
    count: int
    total_errors: int
    percentage: float
    examples: List[int]  # Steps où le pattern apparaît


# =============================================================================
# CHARGEMENT DONNÉES
# =============================================================================

def load_dataset(indicator: str, filter_type: str, split: str = 'test') -> Dict:
    """Charge dataset."""
    filter_suffix = 'octave20' if filter_type == 'octave' else 'kalman'
    path = f'data/prepared/dataset_btc_eth_bnb_ada_ltc_{indicator}_dual_binary_{filter_suffix}.npz'

    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset introuvable: {path}")

    data = np.load(path, allow_pickle=True)

    return {
        'Y': data[f'Y_{split}'],
        'Y_pred': data.get(f'Y_{split}_pred', None),
    }


# =============================================================================
# ANALYSE ERREURS
# =============================================================================

def extract_error_contexts(
    decider_name: str,
    decider_data: Dict,
    witness1_name: str,
    witness1_data: Dict,
    witness2_name: str,
    witness2_data: Dict,
    max_samples: int = None
) -> List[ErrorContext]:
    """
    Extrait le contexte de toutes les erreurs du décideur.

    Une erreur = décideur prédit Direction différente de l'Oracle.
    """
    decider_oracle = (decider_data['Y'] > 0.5).astype(int)
    decider_pred = (decider_data['Y_pred'] > 0.5).astype(int)

    witness1_oracle = (witness1_data['Y'] > 0.5).astype(int)
    witness1_pred = (witness1_data['Y_pred'] > 0.5).astype(int)

    witness2_oracle = (witness2_data['Y'] > 0.5).astype(int)
    witness2_pred = (witness2_data['Y_pred'] > 0.5).astype(int)

    if max_samples:
        decider_oracle = decider_oracle[:max_samples]
        decider_pred = decider_pred[:max_samples]
        witness1_oracle = witness1_oracle[:max_samples]
        witness1_pred = witness1_pred[:max_samples]
        witness2_oracle = witness2_oracle[:max_samples]
        witness2_pred = witness2_pred[:max_samples]

    n_samples = len(decider_oracle)
    errors = []

    for i in range(n_samples):
        # Erreur = prédiction Direction différente de Oracle Direction
        decider_error = decider_pred[i, 0] != decider_oracle[i, 0]

        if decider_error:
            errors.append(ErrorContext(
                step=i,
                decider_pred_dir=int(decider_pred[i, 0]),
                decider_pred_force=int(decider_pred[i, 1]),
                decider_oracle_dir=int(decider_oracle[i, 0]),
                decider_oracle_force=int(decider_oracle[i, 1]),
                witness1_name=witness1_name,
                witness1_pred_dir=int(witness1_pred[i, 0]),
                witness1_pred_force=int(witness1_pred[i, 1]),
                witness1_oracle_dir=int(witness1_oracle[i, 0]),
                witness2_name=witness2_name,
                witness2_pred_dir=int(witness2_pred[i, 0]),
                witness2_pred_force=int(witness2_pred[i, 1]),
                witness2_oracle_dir=int(witness2_oracle[i, 0]),
            ))

    return errors


# =============================================================================
# DÉTECTION PATTERNS
# =============================================================================

def detect_patterns(errors: List[ErrorContext], decider_name: str) -> List[InsightPattern]:
    """
    Détecte les patterns récurrents dans les erreurs.
    """
    if not errors:
        return []

    total_errors = len(errors)
    patterns = []

    # Pattern #1: Témoin 1 prédit correctement quand décideur se trompe
    witness1_correct = [
        e for e in errors
        if e.witness1_pred_dir == e.decider_oracle_dir
    ]
    if witness1_correct:
        patterns.append(InsightPattern(
            name=f"{errors[0].witness1_name}_CORRECT_WHEN_{decider_name}_WRONG",
            description=f"{errors[0].witness1_name} prédit correctement quand {decider_name} se trompe",
            count=len(witness1_correct),
            total_errors=total_errors,
            percentage=len(witness1_correct) / total_errors * 100,
            examples=[e.step for e in witness1_correct[:5]]
        ))

    # Pattern #2: Témoin 2 prédit correctement quand décideur se trompe
    witness2_correct = [
        e for e in errors
        if e.witness2_pred_dir == e.decider_oracle_dir
    ]
    if witness2_correct:
        patterns.append(InsightPattern(
            name=f"{errors[0].witness2_name}_CORRECT_WHEN_{decider_name}_WRONG",
            description=f"{errors[0].witness2_name} prédit correctement quand {decider_name} se trompe",
            count=len(witness2_correct),
            total_errors=total_errors,
            percentage=len(witness2_correct) / total_errors * 100,
            examples=[e.step for e in witness2_correct[:5]]
        ))

    # Pattern #3: LES DEUX témoins prédisent correctement (veto fort)
    both_correct = [
        e for e in errors
        if e.witness1_pred_dir == e.decider_oracle_dir
        and e.witness2_pred_dir == e.decider_oracle_dir
    ]
    if both_correct:
        patterns.append(InsightPattern(
            name="BOTH_WITNESSES_CORRECT",
            description=f"{errors[0].witness1_name}+{errors[0].witness2_name} d'accord avec Oracle",
            count=len(both_correct),
            total_errors=total_errors,
            percentage=len(both_correct) / total_errors * 100,
            examples=[e.step for e in both_correct[:5]]
        ))

    # Pattern #4: Témoin 1 contredit décideur (désaccord Direction)
    witness1_disagrees = [
        e for e in errors
        if e.witness1_pred_dir != e.decider_pred_dir
    ]
    if witness1_disagrees:
        patterns.append(InsightPattern(
            name=f"{errors[0].witness1_name}_DISAGREES_DIR",
            description=f"{errors[0].witness1_name} Direction opposée à {decider_name}",
            count=len(witness1_disagrees),
            total_errors=total_errors,
            percentage=len(witness1_disagrees) / total_errors * 100,
            examples=[e.step for e in witness1_disagrees[:5]]
        ))

    # Pattern #5: Témoin 2 contredit décideur
    witness2_disagrees = [
        e for e in errors
        if e.witness2_pred_dir != e.decider_pred_dir
    ]
    if witness2_disagrees:
        patterns.append(InsightPattern(
            name=f"{errors[0].witness2_name}_DISAGREES_DIR",
            description=f"{errors[0].witness2_name} Direction opposée à {decider_name}",
            count=len(witness2_disagrees),
            total_errors=total_errors,
            percentage=len(witness2_disagrees) / total_errors * 100,
            examples=[e.step for e in witness2_disagrees[:5]]
        ))

    # Pattern #6: LES DEUX témoins contredisent décideur (warning fort)
    both_disagree = [
        e for e in errors
        if e.witness1_pred_dir != e.decider_pred_dir
        and e.witness2_pred_dir != e.decider_pred_dir
    ]
    if both_disagree:
        patterns.append(InsightPattern(
            name="BOTH_WITNESSES_DISAGREE",
            description=f"{errors[0].witness1_name}+{errors[0].witness2_name} contredisent {decider_name}",
            count=len(both_disagree),
            total_errors=total_errors,
            percentage=len(both_disagree) / total_errors * 100,
            examples=[e.step for e in both_disagree[:5]]
        ))

    # Pattern #7: Témoin 1 Force=WEAK (signal d'incertitude)
    witness1_weak = [
        e for e in errors
        if e.witness1_pred_force == 0
    ]
    if witness1_weak:
        patterns.append(InsightPattern(
            name=f"{errors[0].witness1_name}_FORCE_WEAK",
            description=f"{errors[0].witness1_name} Force=WEAK au moment de l'erreur",
            count=len(witness1_weak),
            total_errors=total_errors,
            percentage=len(witness1_weak) / total_errors * 100,
            examples=[e.step for e in witness1_weak[:5]]
        ))

    # Pattern #8: Témoin 2 Force=WEAK
    witness2_weak = [
        e for e in errors
        if e.witness2_pred_force == 0
    ]
    if witness2_weak:
        patterns.append(InsightPattern(
            name=f"{errors[0].witness2_name}_FORCE_WEAK",
            description=f"{errors[0].witness2_name} Force=WEAK au moment de l'erreur",
            count=len(witness2_weak),
            total_errors=total_errors,
            percentage=len(witness2_weak) / total_errors * 100,
            examples=[e.step for e in witness2_weak[:5]]
        ))

    # Pattern #9: Décideur lui-même Force=WEAK (trading incertain)
    decider_weak = [
        e for e in errors
        if e.decider_pred_force == 0
    ]
    if decider_weak:
        patterns.append(InsightPattern(
            name=f"{decider_name}_FORCE_WEAK",
            description=f"{decider_name} Force=WEAK au moment de l'erreur (ne devrait pas trader)",
            count=len(decider_weak),
            total_errors=total_errors,
            percentage=len(decider_weak) / total_errors * 100,
            examples=[e.step for e in decider_weak[:5]]
        ))

    # Trier par pourcentage décroissant
    patterns.sort(key=lambda p: p.percentage, reverse=True)

    return patterns


# =============================================================================
# AFFICHAGE
# =============================================================================

def print_report(
    decider_name: str,
    errors: List[ErrorContext],
    patterns: List[InsightPattern],
    total_samples: int
):
    """Affiche le rapport d'analyse."""
    logger.info("\n" + "="*120)
    logger.info(f"ANALYSE PATTERNS D'ERREURS - Décideur: {decider_name.upper()}")
    logger.info("="*120)

    logger.info(f"\n📊 STATISTIQUES GLOBALES:")
    logger.info(f"  Total samples: {total_samples:,}")
    logger.info(f"  Erreurs {decider_name}: {len(errors):,} ({len(errors)/total_samples*100:.2f}%)")
    logger.info(f"  Accuracy {decider_name}: {(1 - len(errors)/total_samples)*100:.2f}%")

    if not patterns:
        logger.info("\n❌ Aucun pattern détecté.")
        return

    logger.info(f"\n🔍 PATTERNS IDENTIFIÉS ({len(patterns)} patterns):")
    logger.info("-"*120)
    logger.info(f"{'#':<4} {'Pattern':<40} {'Count':>8} {'Percentage':>12} {'Exemples'}")
    logger.info("-"*120)

    for i, p in enumerate(patterns, 1):
        examples_str = ', '.join([f"#{s}" for s in p.examples[:3]])
        logger.info(f"{i:<4} {p.name:<40} {p.count:>8,} {p.percentage:>11.2f}% {examples_str}")

    logger.info("\n" + "="*120)
    logger.info("📝 INSIGHTS ACTIONNABLES (seuil >50%):")
    logger.info("="*120)

    actionable = [p for p in patterns if p.percentage >= 50.0]

    if not actionable:
        logger.info("\n⚠️  Aucun pattern >50% détecté.")
        logger.info("→ Les témoins ne sont pas suffisamment prédictifs des erreurs du décideur.")
    else:
        for i, p in enumerate(actionable, 1):
            logger.info(f"\n{i}. {p.description}")
            logger.info(f"   Fréquence: {p.percentage:.1f}% ({p.count:,}/{len(errors):,} erreurs)")

            # Recommandation
            if "CORRECT" in p.name:
                logger.info(f"   💡 Règle: Utiliser {p.name.split('_')[0]} comme veto si contredit {decider_name}")
            elif "DISAGREE" in p.name and "BOTH" in p.name:
                logger.info(f"   💡 Règle: Ne PAS trader si les 2 témoins contredisent {decider_name}")
            elif "WEAK" in p.name:
                logger.info(f"   💡 Règle: Ne PAS trader si {p.name.split('_')[0]} Force=WEAK")

    logger.info("\n" + "="*120 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyse des patterns d\'erreurs avec témoins',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--decider', type=str, default='macd',
                        choices=['macd', 'rsi', 'cci'],
                        help='Indicateur décideur à analyser (défaut: macd)')

    parser.add_argument('--filter', type=str, default='kalman',
                        choices=['kalman', 'octave'],
                        help='Filtre à utiliser pour TOUS les indicateurs (défaut: kalman)')

    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Split (défaut: test)')

    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limiter le nombre de samples (défaut: tous)')

    args = parser.parse_args()

    logger.info("="*120)
    logger.info("ANALYSE PATTERNS D'ERREURS")
    logger.info("="*120)
    logger.info(f"Décideur: {args.decider.upper()}")
    logger.info(f"Filtre: {args.filter}")
    logger.info(f"Split: {args.split}")
    logger.info("="*120 + "\n")

    # Charger les 3 indicateurs
    logger.info("📂 Chargement des datasets...\n")

    macd_data = load_dataset('macd', args.filter, args.split)
    rsi_data = load_dataset('rsi', args.filter, args.split)
    cci_data = load_dataset('cci', args.filter, args.split)

    logger.info("✅ Datasets chargés\n")

    # Définir décideur et témoins
    if args.decider == 'macd':
        decider_data = macd_data
        witness1_name, witness1_data = 'RSI', rsi_data
        witness2_name, witness2_data = 'CCI', cci_data
    elif args.decider == 'rsi':
        decider_data = rsi_data
        witness1_name, witness1_data = 'MACD', macd_data
        witness2_name, witness2_data = 'CCI', cci_data
    else:  # cci
        decider_data = cci_data
        witness1_name, witness1_data = 'MACD', macd_data
        witness2_name, witness2_data = 'RSI', rsi_data

    # Extraire erreurs
    logger.info(f"🔍 Extraction des erreurs {args.decider.upper()}...\n")

    errors = extract_error_contexts(
        args.decider.upper(),
        decider_data,
        witness1_name,
        witness1_data,
        witness2_name,
        witness2_data,
        max_samples=args.max_samples
    )

    total_samples = args.max_samples if args.max_samples else len(decider_data['Y'])

    logger.info(f"✅ {len(errors):,} erreurs extraites sur {total_samples:,} samples\n")

    # Détecter patterns
    logger.info("🔍 Détection des patterns...\n")

    patterns = detect_patterns(errors, args.decider.upper())

    logger.info(f"✅ {len(patterns)} patterns détectés\n")

    # Afficher rapport
    print_report(args.decider.upper(), errors, patterns, total_samples)


if __name__ == '__main__':
    main()
