#!/usr/bin/env python3
"""
Test CRITIQUE : Est-ce que Y[i] (pente passée) prédit returns[i+1] (futur) ?

Compare :
1. Oracle Y[i] → returns[i+1] (corrélation baseline)
2. Prédictions Y_pred[i] → returns[i+1] (corrélation avec bruit)

Si corrélation Oracle >> corrélation IA → Problème : 8% erreur détruit l'utilité
Si corrélation Oracle ≈ corrélation IA → Autre problème (logique trading, etc.)
"""

import numpy as np
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_dataset(indicator: str, split: str):
    """Charge le dataset."""
    dataset_path = Path(f"data/prepared/dataset_btc_eth_bnb_ada_ltc_{indicator}_dual_binary_kalman.npz")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)

    return {
        'X': data[f'X_{split}'],
        'Y': data[f'Y_{split}'],
        'Y_pred': data.get(f'Y_{split}_pred', None),
    }


def extract_c_ret(X: np.ndarray, indicator: str) -> np.ndarray:
    """Extrait c_ret de X."""
    if indicator in ['rsi', 'macd']:
        c_ret = X[:, -1, 0]
    elif indicator == 'cci':
        c_ret = X[:, -1, 2]
    else:
        raise ValueError(f"Indicateur inconnu: {indicator}")

    return c_ret


def compute_directional_accuracy(direction_labels, future_returns):
    """
    Calcule l'accuracy directionnelle :
    - Si label=UP (1), on prédit que future_return > 0
    - Si label=DOWN (0), on prédit que future_return < 0

    Returns:
        accuracy: % de prédictions correctes
    """
    predicted_up = (direction_labels == 1)
    actual_up = (future_returns > 0)

    correct = (predicted_up == actual_up)
    accuracy = correct.mean()

    return accuracy


def compute_correlation(direction_labels, future_returns):
    """
    Calcule corrélation entre labels Direction et returns futurs.

    Convertit Direction (0/1) en (-1/+1) pour corrélation linéaire.
    """
    direction_signed = (direction_labels * 2) - 1  # 0→-1, 1→+1
    correlation = np.corrcoef(direction_signed, future_returns)[0, 1]

    return correlation


def analyze_force_filter(direction_labels, force_labels, future_returns):
    """
    Analyse l'impact du filtre Force=STRONG.

    Compare accuracy sur :
    - Tous les samples
    - Seulement Force=STRONG
    """
    # Tous
    acc_all = compute_directional_accuracy(direction_labels, future_returns)
    corr_all = compute_correlation(direction_labels, future_returns)

    # Force=STRONG uniquement
    strong_mask = (force_labels == 1)
    if strong_mask.sum() > 0:
        acc_strong = compute_directional_accuracy(
            direction_labels[strong_mask],
            future_returns[strong_mask]
        )
        corr_strong = compute_correlation(
            direction_labels[strong_mask],
            future_returns[strong_mask]
        )
    else:
        acc_strong = 0
        corr_strong = 0

    return {
        'accuracy_all': acc_all,
        'accuracy_strong': acc_strong,
        'correlation_all': corr_all,
        'correlation_strong': corr_strong,
        'n_strong': strong_mask.sum(),
        'pct_strong': strong_mask.mean(),
    }


def main():
    parser = argparse.ArgumentParser(description="Test prédictivité des labels")
    parser.add_argument('--indicator', type=str, required=True, choices=['macd', 'rsi', 'cci'])
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])

    args = parser.parse_args()

    # Charger données
    data = load_dataset(args.indicator, args.split)
    returns = extract_c_ret(data['X'], args.indicator)

    # Extraire labels
    Y_oracle = data['Y']
    Y_pred = data['Y_pred']

    if Y_pred is None:
        logger.error("❌ Prédictions non disponibles. Exécuter train.py + evaluate.py d'abord.")
        return

    # Futurs returns (décalage de +1)
    future_returns = returns[1:]
    Y_oracle = Y_oracle[:-1]  # Aligner
    Y_pred = Y_pred[:-1]

    n_samples = len(future_returns)

    logger.info("=" * 80)
    logger.info(f"🔬 TEST PRÉDICTIVITÉ DES LABELS - {args.indicator.upper()} ({args.split})")
    logger.info("=" * 80)
    logger.info(f"\n📊 Dataset:")
    logger.info(f"   Samples: {n_samples:,}")
    logger.info(f"   Test: Y[i] (pente passée) → returns[i+1] (futur)")

    # Analyser Oracle
    logger.info(f"\n" + "=" * 80)
    logger.info(f"📈 ORACLE (Ground Truth Labels)")
    logger.info("=" * 80)

    oracle_dir = Y_oracle[:, 0]
    oracle_force = Y_oracle[:, 1]

    oracle_stats = analyze_force_filter(oracle_dir, oracle_force, future_returns)

    logger.info(f"\n   Tous les samples:")
    logger.info(f"      Accuracy directionnelle: {oracle_stats['accuracy_all']*100:.2f}%")
    logger.info(f"      Corrélation:             {oracle_stats['correlation_all']:.4f}")

    logger.info(f"\n   Force=STRONG uniquement ({oracle_stats['n_strong']:,} samples, {oracle_stats['pct_strong']*100:.1f}%):")
    logger.info(f"      Accuracy directionnelle: {oracle_stats['accuracy_strong']*100:.2f}%")
    logger.info(f"      Corrélation:             {oracle_stats['correlation_strong']:.4f}")

    # Analyser Prédictions
    logger.info(f"\n" + "=" * 80)
    logger.info(f"🤖 PRÉDICTIONS (Modèle ML)")
    logger.info("=" * 80)

    # Convertir prédictions en binaire
    pred_dir = (Y_pred[:, 0] > 0.5).astype(int)
    pred_force = (Y_pred[:, 1] > 0.5).astype(int)

    pred_stats = analyze_force_filter(pred_dir, pred_force, future_returns)

    logger.info(f"\n   Tous les samples:")
    logger.info(f"      Accuracy directionnelle: {pred_stats['accuracy_all']*100:.2f}%")
    logger.info(f"      Corrélation:             {pred_stats['correlation_all']:.4f}")

    logger.info(f"\n   Force=STRONG uniquement ({pred_stats['n_strong']:,} samples, {pred_stats['pct_strong']*100:.1f}%):")
    logger.info(f"      Accuracy directionnelle: {pred_stats['accuracy_strong']*100:.2f}%")
    logger.info(f"      Corrélation:             {pred_stats['correlation_strong']:.4f}")

    # Comparaison
    logger.info(f"\n" + "=" * 80)
    logger.info(f"📊 COMPARAISON ORACLE vs PRÉDICTIONS")
    logger.info("=" * 80)

    acc_delta_all = pred_stats['accuracy_all'] - oracle_stats['accuracy_all']
    acc_delta_strong = pred_stats['accuracy_strong'] - oracle_stats['accuracy_strong']
    corr_delta_all = pred_stats['correlation_all'] - oracle_stats['correlation_all']
    corr_delta_strong = pred_stats['correlation_strong'] - oracle_stats['correlation_strong']

    logger.info(f"\n   Accuracy directionnelle (TOUS):")
    logger.info(f"      Oracle:      {oracle_stats['accuracy_all']*100:.2f}%")
    logger.info(f"      Prédictions: {pred_stats['accuracy_all']*100:.2f}%")
    logger.info(f"      Delta:       {acc_delta_all*100:+.2f}%")

    logger.info(f"\n   Accuracy directionnelle (Force=STRONG):")
    logger.info(f"      Oracle:      {oracle_stats['accuracy_strong']*100:.2f}%")
    logger.info(f"      Prédictions: {pred_stats['accuracy_strong']*100:.2f}%")
    logger.info(f"      Delta:       {acc_delta_strong*100:+.2f}%")

    logger.info(f"\n   Corrélation (TOUS):")
    logger.info(f"      Oracle:      {oracle_stats['correlation_all']:.4f}")
    logger.info(f"      Prédictions: {pred_stats['correlation_all']:.4f}")
    logger.info(f"      Delta:       {corr_delta_all:+.4f}")

    logger.info(f"\n   Corrélation (Force=STRONG):")
    logger.info(f"      Oracle:      {oracle_stats['correlation_strong']:.4f}")
    logger.info(f"      Prédictions: {pred_stats['correlation_strong']:.4f}")
    logger.info(f"      Delta:       {corr_delta_strong:+.4f}")

    # Interprétation
    logger.info(f"\n" + "=" * 80)
    logger.info(f"💡 INTERPRÉTATION")
    logger.info("=" * 80)

    if acc_delta_strong < -5.0:
        logger.info(f"\n   ⚠️  PROBLÈME MAJEUR:")
        logger.info(f"      Les prédictions ont {abs(acc_delta_strong):.1f}% d'accuracy EN MOINS pour prédire le futur!")
        logger.info(f"      Les 8% d'erreur du modèle DÉTRUISENT l'utilité des labels.")
        logger.info(f"\n   💊 SOLUTIONS:")
        logger.info(f"      1. Améliorer l'accuracy du modèle (92% → 98%+)")
        logger.info(f"      2. Changer les labels pour prédire le futur direct (Y = returns[i+1] > 0)")
        logger.info(f"      3. Utiliser un ensemble de modèles pour réduire l'erreur")
    elif acc_delta_strong < -2.0:
        logger.info(f"\n   ⚠️  Prédictions légèrement moins prédictives ({abs(acc_delta_strong):.1f}%)")
        logger.info(f"      Impact modéré, mais contribue à la dégradation.")
    else:
        logger.info(f"\n   ✅ Prédictions aussi prédictives que l'Oracle ({acc_delta_strong:+.1f}%)")
        logger.info(f"      Le problème n'est PAS dans la qualité des prédictions.")
        logger.info(f"      Chercher ailleurs : logique de trading, frais, etc.")

    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    main()
