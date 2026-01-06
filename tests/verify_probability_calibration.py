#!/usr/bin/env python3
"""
Test CRITIQUE : Vérifier calibration des probabilités

Si toutes les probas sont > 0.9 ou < 0.1, le meta-modèle n'aura rien à apprendre.
On a besoin de variabilité dans [0.3, 0.7] pour que ça marche.
"""

import numpy as np
import argparse
import logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_dataset(indicator: str, split: str):
    """Charge le dataset."""
    dataset_path = Path(f"data/prepared/dataset_btc_eth_bnb_ada_ltc_{indicator}_dual_binary_kalman.npz")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)

    return {
        'Y_pred': data.get(f'Y_{split}_pred', None),
    }


def analyze_probability_distribution(probs, name):
    """Analyse distribution des probabilités."""
    logger.info(f"\n📊 {name}:")
    logger.info(f"   Min:     {probs.min():.4f}")
    logger.info(f"   Max:     {probs.max():.4f}")
    logger.info(f"   Moyenne: {probs.mean():.4f}")
    logger.info(f"   Médiane: {np.median(probs):.4f}")
    logger.info(f"   Std:     {probs.std():.4f}")

    # Distribution par bins
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(probs, bins=bins)

    logger.info(f"\n   Distribution:")
    for i in range(len(bins)-1):
        pct = hist[i] / len(probs) * 100
        logger.info(f"      [{bins[i]:.1f}, {bins[i+1]:.1f}): {hist[i]:,} ({pct:.1f}%)")

    # Zone utile (0.3-0.7)
    useful = ((probs >= 0.3) & (probs <= 0.7)).sum()
    useful_pct = useful / len(probs) * 100
    logger.info(f"\n   Zone utile [0.3, 0.7]: {useful:,} ({useful_pct:.1f}%)")

    # Trop confiantes (< 0.1 ou > 0.9)
    overconfident = ((probs < 0.1) | (probs > 0.9)).sum()
    overconfident_pct = overconfident / len(probs) * 100
    logger.info(f"   Trop confiantes (<0.1 ou >0.9): {overconfident:,} ({overconfident_pct:.1f}%)")

    return {
        'mean': probs.mean(),
        'std': probs.std(),
        'useful_pct': useful_pct,
        'overconfident_pct': overconfident_pct,
        'hist': hist,
    }


def main():
    parser = argparse.ArgumentParser(description="Vérifier calibration des probabilités")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info(f"🔬 VÉRIFICATION CALIBRATION PROBABILITÉS ({args.split})")
    logger.info("=" * 80)

    results = {}

    for indicator in ['macd', 'rsi', 'cci']:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"📊 {indicator.upper()}")
        logger.info("=" * 80)

        data = load_dataset(indicator, args.split)
        Y_pred = data['Y_pred']

        if Y_pred is None:
            logger.error(f"❌ Prédictions non disponibles pour {indicator}")
            continue

        # Direction
        dir_stats = analyze_probability_distribution(Y_pred[:, 0], "Direction Probability")

        # Force
        force_stats = analyze_probability_distribution(Y_pred[:, 1], "Force Probability")

        results[indicator] = {
            'direction': dir_stats,
            'force': force_stats,
        }

    # Synthèse
    logger.info(f"\n{'=' * 80}")
    logger.info(f"💡 SYNTHÈSE")
    logger.info("=" * 80)

    for indicator in ['macd', 'rsi', 'cci']:
        if indicator not in results:
            continue

        logger.info(f"\n{indicator.upper()}:")

        dir_useful = results[indicator]['direction']['useful_pct']
        force_useful = results[indicator]['force']['useful_pct']

        logger.info(f"   Direction zone utile [0.3-0.7]: {dir_useful:.1f}%")
        logger.info(f"   Force zone utile [0.3-0.7]:     {force_useful:.1f}%")

        if force_useful < 10:
            logger.info(f"   ⚠️  PROBLÈME: Moins de 10% des Force probas dans [0.3-0.7]")
            logger.info(f"      → Modèle trop confiant, peu de variabilité pour meta-modèle")
        elif force_useful < 30:
            logger.info(f"   ⚠️  Variabilité faible ({force_useful:.1f}%), mais utilisable")
        else:
            logger.info(f"   ✅ Bonne variabilité ({force_useful:.1f}%), idéal pour meta-modèle")

    logger.info(f"\n{'=' * 80}")
    logger.info(f"📊 RECOMMANDATION")
    logger.info("=" * 80)

    avg_force_useful = np.mean([results[ind]['force']['useful_pct'] for ind in results])

    logger.info(f"\n   Variabilité moyenne Force: {avg_force_useful:.1f}%")

    if avg_force_useful < 15:
        logger.info(f"\n   ❌ VARIABILITÉ TROP FAIBLE")
        logger.info(f"      → Les probas Force sont trop polarisées (0 ou 1)")
        logger.info(f"      → Meta-modèle aura peu d'information à exploiter")
        logger.info(f"\n   💊 SOLUTIONS:")
        logger.info(f"      1. Temperature scaling des probabilités (diviser logits par T>1)")
        logger.info(f"      2. Calibration post-hoc (Platt scaling)")
        logger.info(f"      3. Utiliser logits bruts au lieu de probabilités")
    elif avg_force_useful < 30:
        logger.info(f"\n   ⚠️  VARIABILITÉ MOYENNE")
        logger.info(f"      → Utilisable, mais pas idéal")
        logger.info(f"      → Meta-modèle aura de l'information, mais limitée")
    else:
        logger.info(f"\n   ✅ VARIABILITÉ EXCELLENTE")
        logger.info(f"      → Parfait pour meta-modèle")
        logger.info(f"      → Les probabilités sont bien calibrées")

    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    main()
