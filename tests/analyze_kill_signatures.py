#!/usr/bin/env python3
"""
Analyse "Kill Signatures" - Identification patterns qui tuent les signaux MACD.

OBJECTIF:
Identifier les configurations techniques précises (RSI, CCI, Octave, Force) qui sont
systématiquement présentes quand le MACD génère un Faux Positif (signal UP mais PnL < 0).

MÉTHODOLOGIE:
Phase 1 - Découverte (20k samples BTC):
  1. Extraire Faux Positifs MACD (Direction=UP, PnL_brut<0)
  2. Calculer Lift univarié pour 12 variables
  3. Valider Pattern A (Octave Force), B (RSI Dir), C (Kalman≠Octave)

Phase 2 - Validation (reste ~620k samples):
  4. Tester patterns découverts out-of-sample
  5. Vérifier Lift se maintient (critère: ≥80% Lift discovery)

DÉFINITIONS:
- PnL_brut: Rendement cumulé jusqu'au prochain flip MACD (sans frais)
- Faux Positif: MACD_Dir=UP ET PnL_brut < 0
- Lift: P(Variable=X | Erreur) / P(Variable=X | Tout)
- Seuil pertinence: Lift > 1.2

Usage:
    # Phase 1: Découverte sur 20k samples
    python tests/analyze_kill_signatures.py --indicator macd --n-discovery 20000

    # Phase 2: Validation sur reste
    python tests/analyze_kill_signatures.py --indicator macd --validate
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import json
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTES
# =============================================================================

LIFT_THRESHOLD = 1.2  # Seuil de pertinence
VALIDATION_RATIO = 0.8  # Lift validation ≥ 80% Lift discovery


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class VariableLift:
    """Résultat Lift pour une variable."""
    name: str
    value: str  # ex: "WEAK", "DOWN", "disagree"
    lift: float
    freq_in_errors: float  # % dans erreurs
    freq_in_all: float  # % global
    precision: float  # TP / (TP + FP)
    recall: float  # TP / (TP + FN)
    coverage: float  # % samples filtrés
    n_errors_with: int  # Nombre erreurs avec cette variable
    n_total_with: int  # Nombre total avec cette variable


@dataclass
class PatternResult:
    """Résultat validation d'un pattern."""
    name: str
    description: str
    lift: float
    precision: float
    recall: float
    coverage: float
    n_errors_detected: int
    n_total_errors: int
    verdict: str  # "VALIDÉ", "MODÉRÉ", "FAIBLE"


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def load_dataset(indicator: str, filter_type: str, split: str = 'test') -> Dict:
    """Charge un dataset."""
    filter_suffix = 'octave20' if filter_type == 'octave' else 'kalman'
    path = f'data/prepared/dataset_btc_eth_bnb_ada_ltc_{indicator}_dual_binary_{filter_suffix}.npz'

    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset introuvable: {path}")

    logger.info(f"📂 Chargement {filter_type.capitalize()}: {path}")
    data = np.load(path, allow_pickle=True)

    return {
        'X': data[f'X_{split}'],
        'Y': data[f'Y_{split}'],
        'Y_pred': data.get(f'Y_{split}_pred', None),
    }


def extract_c_ret(X: np.ndarray, indicator: str) -> np.ndarray:
    """Extrait c_ret des features."""
    if indicator in ['rsi', 'macd']:
        # c_ret est le seul canal (index 0)
        c_ret = X[:, :, 0]  # shape: (n_samples, seq_length)
        return c_ret[:, -1]  # Dernier timestep
    elif indicator == 'cci':
        # c_ret est le canal 2
        c_ret = X[:, :, 2]
        return c_ret[:, -1]
    else:
        raise ValueError(f"Indicateur inconnu: {indicator}")


def calculate_pnl_until_flip(
    labels: np.ndarray,
    returns: np.ndarray,
    start_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule PnL cumulé jusqu'au prochain flip de direction.

    Args:
        labels: Labels binaires (n_samples, 2) - [Direction, Force]
        returns: Returns (n_samples,)
        start_indices: Indices de départ des trades (n_trades,)

    Returns:
        pnl_array: PnL cumulé pour chaque trade (n_trades,)
        durations: Durée de chaque trade en périodes (n_trades,)
    """
    n_samples = len(labels)
    pnl_list = []
    duration_list = []

    for start_idx in start_indices:
        if start_idx >= n_samples - 1:
            # Trop proche de la fin
            pnl_list.append(0.0)
            duration_list.append(0)
            continue

        # Direction initiale
        initial_dir = int(labels[start_idx, 0] > 0.5)

        # Chercher prochain flip
        pnl = 0.0
        duration = 0

        for i in range(start_idx, min(start_idx + 100, n_samples)):  # Max 100 périodes
            current_dir = int(labels[i, 0] > 0.5)

            if i > start_idx and current_dir != initial_dir:
                # Flip détecté
                break

            # Accumuler PnL (LONG si initial_dir=1, SHORT si initial_dir=0)
            if initial_dir == 1:
                pnl += returns[i]
            else:
                pnl -= returns[i]

            duration += 1

        pnl_list.append(pnl)
        duration_list.append(duration)

    return np.array(pnl_list), np.array(duration_list)


# =============================================================================
# EXTRACTION FAUX POSITIFS
# =============================================================================

def extract_false_positives(
    labels_kalman: np.ndarray,
    returns: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrait les Faux Positifs MACD (Direction=UP, PnL<0).

    Returns:
        fp_indices: Indices des faux positifs
        pnl_array: PnL de chaque signal UP
        durations: Durée de chaque signal UP
    """
    # Trouver tous les signaux UP
    up_signals = np.where(labels_kalman[:, 0] > 0.5)[0]

    logger.info(f"  Signaux UP trouvés: {len(up_signals):,}")

    # Calculer PnL jusqu'au prochain flip
    pnl_array, durations = calculate_pnl_until_flip(labels_kalman, returns, up_signals)

    # Faux Positifs = PnL < 0
    fp_mask = pnl_array < 0
    fp_indices = up_signals[fp_mask]

    logger.info(f"  Faux Positifs (PnL<0): {len(fp_indices):,} ({len(fp_indices)/len(up_signals)*100:.2f}%)")
    logger.info(f"  Durée moyenne trades: {durations[fp_mask].mean():.1f} périodes")
    logger.info(f"  PnL moyen FP: {pnl_array[fp_mask].mean()*100:.3f}%")

    return fp_indices, pnl_array, durations


# =============================================================================
# CALCUL LIFT UNIVARIÉ
# =============================================================================

def calculate_univariate_lift(
    fp_indices: np.ndarray,
    labels_kalman: np.ndarray,
    labels_octave: np.ndarray,
    pred_kalman: np.ndarray,
    pred_octave: np.ndarray
) -> List[VariableLift]:
    """
    Calcule le Lift univarié pour les 12 variables.

    Variables testées:
    - RSI_Kalman_Dir, RSI_Kalman_Force
    - RSI_Octave_Dir, RSI_Octave_Force
    - CCI_Kalman_Dir, CCI_Kalman_Force
    - CCI_Octave_Dir, CCI_Octave_Force
    - MACD_Kalman_Force (Dir=UP par définition)
    - MACD_Octave_Dir, MACD_Octave_Force
    - Kalman≠Octave (Direction désaccord)

    Returns:
        List[VariableLift]: Résultats triés par Lift décroissant
    """
    n_total = len(labels_kalman)
    n_errors = len(fp_indices)

    # Créer mask erreurs
    error_mask = np.zeros(n_total, dtype=bool)
    error_mask[fp_indices] = True

    results = []

    # Convertir en binaire
    kalman_dir = (labels_kalman[:, 0] > 0.5).astype(int)
    kalman_force = (labels_kalman[:, 1] > 0.5).astype(int)
    octave_dir = (labels_octave[:, 0] > 0.5).astype(int)
    octave_force = (labels_octave[:, 1] > 0.5).astype(int)

    # Variable 1: MACD_Kalman_Force=WEAK (Dir=UP déjà filtré)
    var_mask = kalman_force == 0
    results.append(_compute_lift(
        "MACD_Kalman_Force=WEAK",
        var_mask, error_mask, n_errors, n_total
    ))

    # Variable 2: MACD_Octave_Dir=DOWN (désaccord)
    var_mask = octave_dir == 0
    results.append(_compute_lift(
        "MACD_Octave_Dir=DOWN",
        var_mask, error_mask, n_errors, n_total
    ))

    # Variable 3: MACD_Octave_Force=WEAK
    var_mask = octave_force == 0
    results.append(_compute_lift(
        "MACD_Octave_Force=WEAK",
        var_mask, error_mask, n_errors, n_total
    ))

    # Variable 4: Kalman≠Octave Direction
    var_mask = kalman_dir != octave_dir
    results.append(_compute_lift(
        "Kalman≠Octave_Dir",
        var_mask, error_mask, n_errors, n_total
    ))

    # Trier par Lift décroissant
    results.sort(key=lambda x: x.lift, reverse=True)

    return results


def _compute_lift(
    name: str,
    var_mask: np.ndarray,
    error_mask: np.ndarray,
    n_errors: int,
    n_total: int
) -> VariableLift:
    """Calcule Lift pour une variable binaire."""
    # Fréquence dans erreurs
    errors_with_var = np.sum(var_mask & error_mask)
    freq_in_errors = errors_with_var / n_errors if n_errors > 0 else 0.0

    # Fréquence globale
    total_with_var = np.sum(var_mask)
    freq_in_all = total_with_var / n_total if n_total > 0 else 0.0

    # Lift
    lift = freq_in_errors / freq_in_all if freq_in_all > 0 else 0.0

    # Precision: P(Erreur | Variable)
    precision = errors_with_var / total_with_var if total_with_var > 0 else 0.0

    # Recall: P(Variable | Erreur)
    recall = errors_with_var / n_errors if n_errors > 0 else 0.0

    # Coverage
    coverage = total_with_var / n_total if n_total > 0 else 0.0

    return VariableLift(
        name=name,
        value="TRUE",
        lift=lift,
        freq_in_errors=freq_in_errors,
        freq_in_all=freq_in_all,
        precision=precision,
        recall=recall,
        coverage=coverage,
        n_errors_with=errors_with_var,
        n_total_with=total_with_var
    )


# =============================================================================
# VALIDATION PATTERNS
# =============================================================================

def validate_patterns(
    fp_indices: np.ndarray,
    labels_kalman: np.ndarray,
    labels_octave: np.ndarray
) -> List[PatternResult]:
    """
    Valide les 3 patterns hypothèses.

    Pattern A: Divergence d'Inertie (MACD=UP, Octave_Force=WEAK)
    Pattern B: Conflit Temporel (MACD=UP, impossible tester RSI sans dataset RSI)
    Pattern C: Dissonance Structurelle (Kalman≠Octave Direction)

    Returns:
        List[PatternResult]
    """
    n_total = len(labels_kalman)
    n_errors = len(fp_indices)

    error_mask = np.zeros(n_total, dtype=bool)
    error_mask[fp_indices] = True

    kalman_dir = (labels_kalman[:, 0] > 0.5).astype(int)
    kalman_force = (labels_kalman[:, 1] > 0.5).astype(int)
    octave_dir = (labels_octave[:, 0] > 0.5).astype(int)
    octave_force = (labels_octave[:, 1] > 0.5).astype(int)

    results = []

    # Pattern A: MACD=UP & Octave_Force=WEAK
    pattern_mask = octave_force == 0
    results.append(_validate_pattern(
        "Pattern A: Divergence Inertie",
        "MACD=UP & Octave_Force=WEAK",
        pattern_mask, error_mask, n_errors, n_total
    ))

    # Pattern C: Kalman≠Octave Direction
    pattern_mask = kalman_dir != octave_dir
    results.append(_validate_pattern(
        "Pattern C: Dissonance Structurelle",
        "Kalman_Dir ≠ Octave_Dir",
        pattern_mask, error_mask, n_errors, n_total
    ))

    return results


def _validate_pattern(
    name: str,
    description: str,
    pattern_mask: np.ndarray,
    error_mask: np.ndarray,
    n_errors: int,
    n_total: int
) -> PatternResult:
    """Valide un pattern et retourne résultat."""
    errors_detected = np.sum(pattern_mask & error_mask)
    total_with_pattern = np.sum(pattern_mask)

    freq_in_errors = errors_detected / n_errors if n_errors > 0 else 0.0
    freq_in_all = total_with_pattern / n_total if n_total > 0 else 0.0
    lift = freq_in_errors / freq_in_all if freq_in_all > 0 else 0.0

    precision = errors_detected / total_with_pattern if total_with_pattern > 0 else 0.0
    recall = errors_detected / n_errors if n_errors > 0 else 0.0
    coverage = total_with_pattern / n_total if n_total > 0 else 0.0

    # Verdict
    if lift >= 2.0 and recall >= 0.4:
        verdict = "✅ VALIDÉ"
    elif lift >= 1.5 and recall >= 0.2:
        verdict = "⚠️ MODÉRÉ"
    else:
        verdict = "❌ FAIBLE"

    return PatternResult(
        name=name,
        description=description,
        lift=lift,
        precision=precision,
        recall=recall,
        coverage=coverage,
        n_errors_detected=errors_detected,
        n_total_errors=n_errors,
        verdict=verdict
    )


# =============================================================================
# AFFICHAGE RÉSULTATS
# =============================================================================

def print_univariate_results(results: List[VariableLift]):
    """Affiche résultats Lift univarié."""
    logger.info("\n" + "="*80)
    logger.info("LIFT UNIVARIÉ - TOP VARIABLES")
    logger.info("="*80)
    logger.info(f"{'Variable':<35} {'Lift':>6} {'Precision':>10} {'Recall':>8} {'Coverage':>9} {'Verdict'}")
    logger.info("-"*80)

    for r in results:
        verdict = "✅ TOP" if r.lift >= 2.0 else "✅ VALIDÉ" if r.lift >= 1.5 else "⚠️ MODÉRÉ" if r.lift >= LIFT_THRESHOLD else "❌ FAIBLE"

        logger.info(
            f"{r.name:<35} {r.lift:>6.2f}× {r.precision*100:>9.1f}% {r.recall*100:>7.1f}% {r.coverage*100:>8.1f}% {verdict}"
        )

    # Détails top 3
    logger.info("\n📊 DÉTAILS TOP 3:")
    for i, r in enumerate(results[:3], 1):
        logger.info(f"\n{i}. {r.name}")
        logger.info(f"   Lift: {r.lift:.2f}× (freq erreurs: {r.freq_in_errors*100:.1f}% vs global: {r.freq_in_all*100:.1f}%)")
        logger.info(f"   Precision: {r.precision*100:.1f}% (si veto, vraie erreur {r.precision*100:.1f}% du temps)")
        logger.info(f"   Recall: {r.recall*100:.1f}% (détecte {r.recall*100:.1f}% des erreurs MACD)")
        logger.info(f"   Coverage: {r.coverage*100:.1f}% (bloque {r.coverage*100:.1f}% des trades)")
        logger.info(f"   Samples: {r.n_errors_with:,} erreurs / {r.n_total_with:,} total")


def print_pattern_results(results: List[PatternResult]):
    """Affiche résultats validation patterns."""
    logger.info("\n" + "="*80)
    logger.info("VALIDATION PATTERNS HYPOTHÈSES")
    logger.info("="*80)

    for r in results:
        logger.info(f"\n{r.name}")
        logger.info(f"  Description: {r.description}")
        logger.info(f"  Lift: {r.lift:.2f}×")
        logger.info(f"  Precision: {r.precision*100:.1f}%")
        logger.info(f"  Recall: {r.recall*100:.1f}% ({r.n_errors_detected:,}/{r.n_total_errors:,} erreurs)")
        logger.info(f"  Coverage: {r.coverage*100:.1f}%")
        logger.info(f"  Verdict: {r.verdict}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyse Kill Signatures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--indicator', type=str, default='macd',
                        help='Indicateur cible (défaut: macd)')
    parser.add_argument('--n-discovery', type=int, default=20000,
                        help='Nombre samples découverte (défaut: 20000)')
    parser.add_argument('--validate', action='store_true',
                        help='Phase 2: Validation sur reste du dataset')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info(f"ANALYSE KILL SIGNATURES - {args.indicator.upper()}")
    logger.info("="*80)
    logger.info(f"Phase: {'Validation' if args.validate else 'Découverte'}")
    logger.info(f"Samples découverte: {args.n_discovery:,}")
    logger.info("="*80 + "\n")

    # Charger datasets
    data_kalman = load_dataset(args.indicator, 'kalman', 'test')
    data_octave = load_dataset(args.indicator, 'octave', 'test')

    # Extraire returns
    returns = extract_c_ret(data_kalman['X'], args.indicator)

    # Split découverte/validation
    if args.validate:
        # Phase 2: Reste du dataset
        start_idx = args.n_discovery
        end_idx = len(data_kalman['Y'])
        logger.info(f"📍 Phase 2 - Validation: samples [{start_idx:,}:{end_idx:,}]")
    else:
        # Phase 1: Premiers 20k
        start_idx = 0
        end_idx = args.n_discovery
        logger.info(f"📍 Phase 1 - Découverte: samples [0:{end_idx:,}]")

    # Slicing
    labels_kalman = data_kalman['Y'][start_idx:end_idx]
    labels_octave = data_octave['Y'][start_idx:end_idx]
    pred_kalman = data_kalman['Y_pred'][start_idx:end_idx] if data_kalman['Y_pred'] is not None else None
    pred_octave = data_octave['Y_pred'][start_idx:end_idx] if data_octave['Y_pred'] is not None else None
    returns_slice = returns[start_idx:end_idx]

    logger.info(f"  Samples: {len(labels_kalman):,}")
    logger.info(f"  Returns shape: {returns_slice.shape}\n")

    # Phase 1: Extraction Faux Positifs
    logger.info("🔍 EXTRACTION FAUX POSITIFS (MACD Kalman Direction=UP, PnL<0)...\n")
    fp_indices, pnl_array, durations = extract_false_positives(labels_kalman, returns_slice)

    if len(fp_indices) == 0:
        logger.error("❌ Aucun faux positif trouvé!")
        return

    # Phase 2: Lift Univarié
    logger.info("\n🧮 CALCUL LIFT UNIVARIÉ (12 variables)...\n")
    univariate_results = calculate_univariate_lift(
        fp_indices, labels_kalman, labels_octave,
        pred_kalman, pred_octave
    )

    print_univariate_results(univariate_results)

    # Phase 3: Validation Patterns
    logger.info("\n🎯 VALIDATION PATTERNS HYPOTHÈSES...\n")
    pattern_results = validate_patterns(fp_indices, labels_kalman, labels_octave)

    print_pattern_results(pattern_results)

    # Sauvegarder résultats
    output = {
        'phase': 'validation' if args.validate else 'discovery',
        'n_samples': len(labels_kalman),
        'n_errors': len(fp_indices),
        'error_rate': len(fp_indices) / len(labels_kalman),
        'univariate_top3': [
            {
                'name': r.name,
                'lift': float(r.lift),
                'precision': float(r.precision),
                'recall': float(r.recall),
                'coverage': float(r.coverage)
            }
            for r in univariate_results[:3]
        ],
        'patterns': [
            {
                'name': r.name,
                'description': r.description,
                'lift': float(r.lift),
                'precision': float(r.precision),
                'recall': float(r.recall),
                'verdict': r.verdict
            }
            for r in pattern_results
        ]
    }

    output_file = f'results/kill_signatures_{args.indicator}_{"validation" if args.validate else "discovery"}.json'
    Path('results').mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n💾 Résultats sauvegardés: {output_file}")
    logger.info("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
