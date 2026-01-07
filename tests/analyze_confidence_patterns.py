#!/usr/bin/env python3
"""
Analyse Chirurgicale des Erreurs avec Scores de Confiance

OBJECTIF:
Au lieu de binariser (0/1), analyser les probabilités brutes pour détecter:
- Prédictions incertaines (zone grise 0.45-0.55)
- Vetos forts (témoin très confiant contredit décideur faible)
- Zones d'incertitude collective (tous les indicateurs hésitent)

Usage:
    python tests/analyze_confidence_patterns.py --decider macd --filter kalman --max-samples 20000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import argparse
from typing import Dict, List, Tuple
from dataclasses import dataclass
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
class ConfidenceContext:
    """Contexte d'erreur avec scores de confiance."""
    step: int

    # Décideur
    decider_prob_dir: float
    decider_prob_force: float
    decider_conf_dir: float  # abs(prob - 0.5) * 2
    decider_conf_force: float
    decider_error: bool

    # Témoin 1
    witness1_name: str
    witness1_prob_dir: float
    witness1_prob_force: float
    witness1_conf_dir: float
    witness1_conf_force: float
    witness1_correct: bool

    # Témoin 2
    witness2_name: str
    witness2_prob_dir: float
    witness2_prob_force: float
    witness2_conf_dir: float
    witness2_conf_force: float
    witness2_correct: bool


@dataclass
class ConfidencePattern:
    """Pattern basé sur la confiance."""
    name: str
    description: str
    count: int
    total_errors: int
    percentage: float
    avg_decider_conf: float
    avg_witness_conf: float
    examples: List[int]


# =============================================================================
# CHARGEMENT DONNÉES
# =============================================================================

def load_dataset(indicator: str, filter_type: str, split: str = 'test') -> Dict:
    """Charge dataset avec probabilités brutes."""
    filter_suffix = 'octave20' if filter_type == 'octave' else 'kalman'
    path = f'data/prepared/dataset_btc_eth_bnb_ada_ltc_{indicator}_dual_binary_{filter_suffix}.npz'

    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset introuvable: {path}")

    data = np.load(path, allow_pickle=True)

    return {
        'Y': data[f'Y_{split}'],              # Oracle (0/1)
        'Y_pred': data.get(f'Y_{split}_pred', None),  # Probabilités [0.0, 1.0]
    }


# =============================================================================
# CALCUL CONFIANCE
# =============================================================================

def compute_confidence(prob: float) -> float:
    """
    Calcule le score de confiance d'une probabilité.

    Score = 0: très incertain (prob = 0.5)
    Score = 1: très confiant (prob = 0.0 ou 1.0)
    """
    return abs(prob - 0.5) * 2.0


def classify_confidence(conf: float) -> str:
    """Classifie un score de confiance."""
    if conf < 0.20:
        return "ZONE_GRISE"
    elif conf < 0.40:
        return "FAIBLE"
    elif conf < 0.60:
        return "MOYEN"
    else:
        return "FORT"


# =============================================================================
# EXTRACTION CONTEXTE
# =============================================================================

def extract_confidence_contexts(
    decider_name: str,
    decider_data: Dict,
    witness1_name: str,
    witness1_data: Dict,
    witness2_name: str,
    witness2_data: Dict,
    max_samples: int = None
) -> List[ConfidenceContext]:
    """
    Extrait contexte avec scores de confiance pour toutes les erreurs.
    """
    # Oracle binarisé
    decider_oracle = (decider_data['Y'] > 0.5).astype(int)
    witness1_oracle = (witness1_data['Y'] > 0.5).astype(int)
    witness2_oracle = (witness2_data['Y'] > 0.5).astype(int)

    # Prédictions brutes (probabilités)
    decider_pred = decider_data['Y_pred']
    witness1_pred = witness1_data['Y_pred']
    witness2_pred = witness2_data['Y_pred']

    if max_samples:
        decider_oracle = decider_oracle[:max_samples]
        decider_pred = decider_pred[:max_samples]
        witness1_oracle = witness1_oracle[:max_samples]
        witness1_pred = witness1_pred[:max_samples]
        witness2_oracle = witness2_oracle[:max_samples]
        witness2_pred = witness2_pred[:max_samples]

    n_samples = len(decider_oracle)
    contexts = []

    for i in range(n_samples):
        # Erreur = Direction prédite différente de Oracle
        decider_pred_bin = int(decider_pred[i, 0] > 0.5)
        decider_error = decider_pred_bin != decider_oracle[i, 0]

        if decider_error:
            # Probabilités brutes
            dec_prob_dir = float(decider_pred[i, 0])
            dec_prob_force = float(decider_pred[i, 1])
            w1_prob_dir = float(witness1_pred[i, 0])
            w1_prob_force = float(witness1_pred[i, 1])
            w2_prob_dir = float(witness2_pred[i, 0])
            w2_prob_force = float(witness2_pred[i, 1])

            # Scores de confiance
            dec_conf_dir = compute_confidence(dec_prob_dir)
            dec_conf_force = compute_confidence(dec_prob_force)
            w1_conf_dir = compute_confidence(w1_prob_dir)
            w1_conf_force = compute_confidence(w1_prob_force)
            w2_conf_dir = compute_confidence(w2_prob_dir)
            w2_conf_force = compute_confidence(w2_prob_force)

            # Témoins corrects?
            w1_pred_bin = int(w1_prob_dir > 0.5)
            w2_pred_bin = int(w2_prob_dir > 0.5)
            w1_correct = w1_pred_bin == decider_oracle[i, 0]
            w2_correct = w2_pred_bin == decider_oracle[i, 0]

            contexts.append(ConfidenceContext(
                step=i,
                decider_prob_dir=dec_prob_dir,
                decider_prob_force=dec_prob_force,
                decider_conf_dir=dec_conf_dir,
                decider_conf_force=dec_conf_force,
                decider_error=True,
                witness1_name=witness1_name,
                witness1_prob_dir=w1_prob_dir,
                witness1_prob_force=w1_prob_force,
                witness1_conf_dir=w1_conf_dir,
                witness1_conf_force=w1_conf_force,
                witness1_correct=w1_correct,
                witness2_name=witness2_name,
                witness2_prob_dir=w2_prob_dir,
                witness2_prob_force=w2_prob_force,
                witness2_conf_dir=w2_conf_dir,
                witness2_conf_force=w2_conf_force,
                witness2_correct=w2_correct,
            ))

    return contexts


# =============================================================================
# DÉTECTION PATTERNS CONFIANCE
# =============================================================================

def detect_confidence_patterns(
    contexts: List[ConfidenceContext],
    decider_name: str
) -> List[ConfidencePattern]:
    """
    Détecte patterns basés sur les scores de confiance.
    """
    if not contexts:
        return []

    total = len(contexts)
    patterns = []

    # Pattern #1: Décideur en zone grise (<0.2 confiance)
    decider_uncertain = [c for c in contexts if c.decider_conf_dir < 0.20]
    if decider_uncertain:
        avg_dec_conf = np.mean([c.decider_conf_dir for c in decider_uncertain])
        avg_wit_conf = np.mean([
            (c.witness1_conf_dir + c.witness2_conf_dir) / 2
            for c in decider_uncertain
        ])
        patterns.append(ConfidencePattern(
            name=f"{decider_name}_ZONE_GRISE",
            description=f"{decider_name} confiance Direction <0.20 (zone grise)",
            count=len(decider_uncertain),
            total_errors=total,
            percentage=len(decider_uncertain) / total * 100,
            avg_decider_conf=avg_dec_conf,
            avg_witness_conf=avg_wit_conf,
            examples=[c.step for c in decider_uncertain[:5]]
        ))

    # Pattern #2: Témoin 1 fort (>0.5) ET décideur faible (<0.3)
    w1_strong_dec_weak = [
        c for c in contexts
        if c.witness1_conf_dir > 0.50 and c.decider_conf_dir < 0.30
    ]
    if w1_strong_dec_weak:
        avg_dec_conf = np.mean([c.decider_conf_dir for c in w1_strong_dec_weak])
        avg_wit_conf = np.mean([c.witness1_conf_dir for c in w1_strong_dec_weak])
        patterns.append(ConfidencePattern(
            name=f"{contexts[0].witness1_name}_FORT_VS_{decider_name}_FAIBLE",
            description=f"{contexts[0].witness1_name} très confiant (>0.5) vs {decider_name} faible (<0.3)",
            count=len(w1_strong_dec_weak),
            total_errors=total,
            percentage=len(w1_strong_dec_weak) / total * 100,
            avg_decider_conf=avg_dec_conf,
            avg_witness_conf=avg_wit_conf,
            examples=[c.step for c in w1_strong_dec_weak[:5]]
        ))

    # Pattern #3: Témoin 2 fort ET décideur faible
    w2_strong_dec_weak = [
        c for c in contexts
        if c.witness2_conf_dir > 0.50 and c.decider_conf_dir < 0.30
    ]
    if w2_strong_dec_weak:
        avg_dec_conf = np.mean([c.decider_conf_dir for c in w2_strong_dec_weak])
        avg_wit_conf = np.mean([c.witness2_conf_dir for c in w2_strong_dec_weak])
        patterns.append(ConfidencePattern(
            name=f"{contexts[0].witness2_name}_FORT_VS_{decider_name}_FAIBLE",
            description=f"{contexts[0].witness2_name} très confiant (>0.5) vs {decider_name} faible (<0.3)",
            count=len(w2_strong_dec_weak),
            total_errors=total,
            percentage=len(w2_strong_dec_weak) / total * 100,
            avg_decider_conf=avg_dec_conf,
            avg_witness_conf=avg_wit_conf,
            examples=[c.step for c in w2_strong_dec_weak[:5]]
        ))

    # Pattern #4: Tous en zone grise (incertitude collective)
    all_uncertain = [
        c for c in contexts
        if c.decider_conf_dir < 0.20
        and c.witness1_conf_dir < 0.20
        and c.witness2_conf_dir < 0.20
    ]
    if all_uncertain:
        avg_conf = np.mean([
            (c.decider_conf_dir + c.witness1_conf_dir + c.witness2_conf_dir) / 3
            for c in all_uncertain
        ])
        patterns.append(ConfidencePattern(
            name="INCERTITUDE_COLLECTIVE",
            description="Tous les indicateurs en zone grise (<0.20)",
            count=len(all_uncertain),
            total_errors=total,
            percentage=len(all_uncertain) / total * 100,
            avg_decider_conf=avg_conf,
            avg_witness_conf=avg_conf,
            examples=[c.step for c in all_uncertain[:5]]
        ))

    # Pattern #5: Décideur faible Force (<0.3)
    dec_force_weak = [c for c in contexts if c.decider_conf_force < 0.30]
    if dec_force_weak:
        avg_dec_conf = np.mean([c.decider_conf_force for c in dec_force_weak])
        patterns.append(ConfidencePattern(
            name=f"{decider_name}_FORCE_INCERTAINE",
            description=f"{decider_name} confiance Force <0.30 (incertaine)",
            count=len(dec_force_weak),
            total_errors=total,
            percentage=len(dec_force_weak) / total * 100,
            avg_decider_conf=avg_dec_conf,
            avg_witness_conf=0.0,
            examples=[c.step for c in dec_force_weak[:5]]
        ))

    # Pattern #6: Témoin 1 correct ET confiant (>0.4)
    w1_correct_confident = [
        c for c in contexts
        if c.witness1_correct and c.witness1_conf_dir > 0.40
    ]
    if w1_correct_confident:
        avg_dec_conf = np.mean([c.decider_conf_dir for c in w1_correct_confident])
        avg_wit_conf = np.mean([c.witness1_conf_dir for c in w1_correct_confident])
        patterns.append(ConfidencePattern(
            name=f"{contexts[0].witness1_name}_CORRECT_CONFIANT",
            description=f"{contexts[0].witness1_name} correct ET confiant (>0.4)",
            count=len(w1_correct_confident),
            total_errors=total,
            percentage=len(w1_correct_confident) / total * 100,
            avg_decider_conf=avg_dec_conf,
            avg_witness_conf=avg_wit_conf,
            examples=[c.step for c in w1_correct_confident[:5]]
        ))

    # Pattern #7: Témoin 2 correct ET confiant
    w2_correct_confident = [
        c for c in contexts
        if c.witness2_correct and c.witness2_conf_dir > 0.40
    ]
    if w2_correct_confident:
        avg_dec_conf = np.mean([c.decider_conf_dir for c in w2_correct_confident])
        avg_wit_conf = np.mean([c.witness2_conf_dir for c in w2_correct_confident])
        patterns.append(ConfidencePattern(
            name=f"{contexts[0].witness2_name}_CORRECT_CONFIANT",
            description=f"{contexts[0].witness2_name} correct ET confiant (>0.4)",
            count=len(w2_correct_confident),
            total_errors=total,
            percentage=len(w2_correct_confident) / total * 100,
            avg_decider_conf=avg_dec_conf,
            avg_witness_conf=avg_wit_conf,
            examples=[c.step for c in w2_correct_confident[:5]]
        ))

    # Trier par pourcentage
    patterns.sort(key=lambda p: p.percentage, reverse=True)

    return patterns


# =============================================================================
# AFFICHAGE
# =============================================================================

def print_report(
    decider_name: str,
    contexts: List[ConfidenceContext],
    patterns: List[ConfidencePattern],
    total_samples: int
):
    """Affiche rapport avec scores de confiance."""
    logger.info("\n" + "="*120)
    logger.info(f"ANALYSE CHIRURGICALE - Décideur: {decider_name.upper()}")
    logger.info("="*120)

    logger.info(f"\n📊 STATISTIQUES:")
    logger.info(f"  Total samples: {total_samples:,}")
    logger.info(f"  Erreurs: {len(contexts):,} ({len(contexts)/total_samples*100:.2f}%)")

    if not contexts:
        logger.info("\n❌ Aucune erreur trouvée.")
        return

    # Distribution confiance décideur
    dec_confs = [c.decider_conf_dir for c in contexts]
    logger.info(f"\n📏 CONFIANCE {decider_name} Direction (sur erreurs):")
    logger.info(f"  Moyenne: {np.mean(dec_confs):.3f}")
    logger.info(f"  Min/Max: {np.min(dec_confs):.3f} / {np.max(dec_confs):.3f}")
    logger.info(f"  Zone grise (<0.20): {sum(1 for c in dec_confs if c < 0.20):,} ({sum(1 for c in dec_confs if c < 0.20)/len(dec_confs)*100:.1f}%)")

    if not patterns:
        logger.info("\n❌ Aucun pattern détecté.")
        return

    logger.info(f"\n🔍 PATTERNS CONFIANCE ({len(patterns)} patterns):")
    logger.info("-"*120)
    logger.info(f"{'#':<3} {'Pattern':<45} {'Count':>8} {'%':>7} {'Conf Déc':>10} {'Conf Tém':>10}")
    logger.info("-"*120)

    for i, p in enumerate(patterns, 1):
        logger.info(
            f"{i:<3} {p.name:<45} {p.count:>8,} {p.percentage:>6.1f}% "
            f"{p.avg_decider_conf:>10.3f} {p.avg_witness_conf:>10.3f}"
        )

    logger.info("\n" + "="*120)
    logger.info("💡 RÈGLES CHIRURGICALES (>50%):")
    logger.info("="*120)

    actionable = [p for p in patterns if p.percentage >= 50.0]

    if not actionable:
        logger.info("\n⚠️  Aucun pattern >50%.")
    else:
        for i, p in enumerate(actionable, 1):
            logger.info(f"\n{i}. {p.description}")
            logger.info(f"   Fréquence: {p.percentage:.1f}% ({p.count:,}/{len(contexts):,})")
            logger.info(f"   Confiance {decider_name}: {p.avg_decider_conf:.3f}")

            if "ZONE_GRISE" in p.name:
                logger.info(f"   💡 Règle: Ne PAS trader si {decider_name} confiance <0.20")
            elif "FORT_VS" in p.name:
                witness = p.name.split('_')[0]
                logger.info(f"   💡 Règle: Veto {witness} si confiance >{p.avg_witness_conf:.2f} et contredit {decider_name} <0.30")
            elif "INCERTITUDE_COLLECTIVE" in p.name:
                logger.info(f"   💡 Règle: Ne JAMAIS trader si tous indicateurs confiance <0.20")
            elif "CORRECT_CONFIANT" in p.name:
                witness = p.name.split('_')[0]
                logger.info(f"   💡 Règle: Utiliser {witness} comme veto si confiant (>0.4)")

    logger.info("\n" + "="*120 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyse chirurgicale avec scores de confiance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--decider', type=str, default='macd',
                        choices=['macd', 'rsi', 'cci'],
                        help='Indicateur décideur (défaut: macd)')

    parser.add_argument('--filter', type=str, default='kalman',
                        choices=['kalman', 'octave'],
                        help='Filtre (défaut: kalman)')

    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Split (défaut: test)')

    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limiter samples (défaut: tous)')

    args = parser.parse_args()

    logger.info("="*120)
    logger.info("ANALYSE CHIRURGICALE - SCORES DE CONFIANCE")
    logger.info("="*120)
    logger.info(f"Décideur: {args.decider.upper()}")
    logger.info(f"Filtre: {args.filter}")
    logger.info("="*120 + "\n")

    # Charger datasets
    logger.info("📂 Chargement...\n")
    macd_data = load_dataset('macd', args.filter, args.split)
    rsi_data = load_dataset('rsi', args.filter, args.split)
    cci_data = load_dataset('cci', args.filter, args.split)

    # Définir rôles
    if args.decider == 'macd':
        decider_data = macd_data
        w1_name, w1_data = 'RSI', rsi_data
        w2_name, w2_data = 'CCI', cci_data
    elif args.decider == 'rsi':
        decider_data = rsi_data
        w1_name, w1_data = 'MACD', macd_data
        w2_name, w2_data = 'CCI', cci_data
    else:
        decider_data = cci_data
        w1_name, w1_data = 'MACD', macd_data
        w2_name, w2_data = 'RSI', rsi_data

    # Extraire contextes
    logger.info("🔍 Extraction contextes confiance...\n")
    contexts = extract_confidence_contexts(
        args.decider.upper(), decider_data,
        w1_name, w1_data,
        w2_name, w2_data,
        max_samples=args.max_samples
    )

    total = args.max_samples if args.max_samples else len(decider_data['Y'])
    logger.info(f"✅ {len(contexts):,} erreurs extraites\n")

    # Détecter patterns
    logger.info("🔍 Détection patterns confiance...\n")
    patterns = detect_confidence_patterns(contexts, args.decider.upper())
    logger.info(f"✅ {len(patterns)} patterns détectés\n")

    # Rapport
    print_report(args.decider.upper(), contexts, patterns, total)


if __name__ == '__main__':
    main()
