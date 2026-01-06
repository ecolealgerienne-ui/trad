#!/usr/bin/env python3
"""
Analyse OVERLAP Force=STRONG : Oracle vs Prédictions

Question logique:
- Oracle Force=STRONG: 196,969 samples → 65.99% prédictivité futur
- IA Force=STRONG: 149,309 samples → 49.17% prédictivité futur

Est-ce que ce sont les MÊMES samples ou des samples DIFFÉRENTS?

Test:
1. Overlap (les deux d'accord Force=STRONG) → quelle prédictivité?
2. Oracle STRONG mais IA WEAK → quelle prédictivité?
3. IA STRONG mais Oracle WEAK → quelle prédictivité?

Cela dira si le problème vient de:
- L'IA rate les bons samples (cas 2 a bonne prédictivité)
- L'IA sélectionne de mauvais samples (cas 3 a mauvaise prédictivité)
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
    Calcule l'accuracy directionnelle.
    Si label=UP (1), on prédit que future_return > 0
    """
    predicted_up = (direction_labels == 1)
    actual_up = (future_returns > 0)
    correct = (predicted_up == actual_up)
    return correct.mean()


def compute_correlation(direction_labels, future_returns):
    """Calcule corrélation entre labels Direction et returns futurs."""
    direction_signed = (direction_labels * 2) - 1  # 0→-1, 1→+1
    correlation = np.corrcoef(direction_signed, future_returns)[0, 1]
    return correlation


def analyze_overlap(oracle_dir, oracle_force, pred_dir, pred_force, future_returns):
    """
    Analyse l'overlap entre Oracle et Prédictions Force=STRONG.

    4 cas mutuellement exclusifs:
    1. BOTH_STRONG: Oracle=1 ET Pred=1
    2. ORACLE_ONLY: Oracle=1 ET Pred=0
    3. PRED_ONLY: Oracle=0 ET Pred=1
    4. BOTH_WEAK: Oracle=0 ET Pred=0
    """
    oracle_strong = (oracle_force == 1)
    pred_strong = (pred_force == 1)

    # 4 cas
    both_strong = oracle_strong & pred_strong
    oracle_only = oracle_strong & ~pred_strong
    pred_only = ~oracle_strong & pred_strong
    both_weak = ~oracle_strong & ~pred_strong

    results = {}

    for name, mask in [
        ('BOTH_STRONG', both_strong),
        ('ORACLE_ONLY', oracle_only),
        ('PRED_ONLY', pred_only),
        ('BOTH_WEAK', both_weak),
    ]:
        n = mask.sum()
        pct = mask.mean() * 100

        if n > 0:
            # Prédictivité Direction
            acc = compute_directional_accuracy(oracle_dir[mask], future_returns[mask])
            corr = compute_correlation(oracle_dir[mask], future_returns[mask])

            results[name] = {
                'n': n,
                'pct': pct,
                'accuracy': acc,
                'correlation': corr,
            }
        else:
            results[name] = {
                'n': 0,
                'pct': 0,
                'accuracy': 0,
                'correlation': 0,
            }

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyse overlap Force=STRONG Oracle vs Prédictions")
    parser.add_argument('--indicator', type=str, required=True, choices=['macd', 'rsi', 'cci'])
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])

    args = parser.parse_args()

    # Charger données
    data = load_dataset(args.indicator, args.split)
    returns = extract_c_ret(data['X'], args.indicator)

    Y_oracle = data['Y']
    Y_pred = data['Y_pred']

    if Y_pred is None:
        logger.error("❌ Prédictions non disponibles. Exécuter train.py + evaluate.py d'abord.")
        return

    # Futurs returns (décalage de +1)
    future_returns = returns[1:]
    Y_oracle = Y_oracle[:-1]
    Y_pred = Y_pred[:-1]

    n_samples = len(future_returns)

    # Extraire labels
    oracle_dir = Y_oracle[:, 0]
    oracle_force = Y_oracle[:, 1]
    pred_dir = (Y_pred[:, 0] > 0.5).astype(int)
    pred_force = (Y_pred[:, 1] > 0.5).astype(int)

    # Statistiques de base
    oracle_strong_count = (oracle_force == 1).sum()
    pred_strong_count = (pred_force == 1).sum()

    logger.info("=" * 80)
    logger.info(f"🔬 ANALYSE OVERLAP FORCE=STRONG - {args.indicator.upper()} ({args.split})")
    logger.info("=" * 80)
    logger.info(f"\n📊 Dataset:")
    logger.info(f"   Total samples: {n_samples:,}")
    logger.info(f"   Oracle Force=STRONG: {oracle_strong_count:,} ({oracle_strong_count/n_samples*100:.1f}%)")
    logger.info(f"   Prédictions Force=STRONG: {pred_strong_count:,} ({pred_strong_count/n_samples*100:.1f}%)")

    # Analyser overlap
    results = analyze_overlap(oracle_dir, oracle_force, pred_dir, pred_force, future_returns)

    # Afficher résultats
    logger.info(f"\n" + "=" * 80)
    logger.info(f"📊 RÉSULTATS PAR CATÉGORIE")
    logger.info("=" * 80)

    logger.info(f"\n1️⃣  BOTH_STRONG (Oracle=1 ET Pred=1) - L'OVERLAP")
    logger.info(f"   Samples:     {results['BOTH_STRONG']['n']:,} ({results['BOTH_STRONG']['pct']:.1f}%)")
    logger.info(f"   Accuracy:    {results['BOTH_STRONG']['accuracy']*100:.2f}%")
    logger.info(f"   Corrélation: {results['BOTH_STRONG']['correlation']:.4f}")

    logger.info(f"\n2️⃣  ORACLE_ONLY (Oracle=1 ET Pred=0) - IA RATE")
    logger.info(f"   Samples:     {results['ORACLE_ONLY']['n']:,} ({results['ORACLE_ONLY']['pct']:.1f}%)")
    logger.info(f"   Accuracy:    {results['ORACLE_ONLY']['accuracy']*100:.2f}%")
    logger.info(f"   Corrélation: {results['ORACLE_ONLY']['correlation']:.4f}")

    logger.info(f"\n3️⃣  PRED_ONLY (Oracle=0 ET Pred=1) - IA FAUX POSITIFS")
    logger.info(f"   Samples:     {results['PRED_ONLY']['n']:,} ({results['PRED_ONLY']['pct']:.1f}%)")
    logger.info(f"   Accuracy:    {results['PRED_ONLY']['accuracy']*100:.2f}%")
    logger.info(f"   Corrélation: {results['PRED_ONLY']['correlation']:.4f}")

    logger.info(f"\n4️⃣  BOTH_WEAK (Oracle=0 ET Pred=0) - ACCORD SUR WEAK")
    logger.info(f"   Samples:     {results['BOTH_WEAK']['n']:,} ({results['BOTH_WEAK']['pct']:.1f}%)")
    logger.info(f"   Accuracy:    {results['BOTH_WEAK']['accuracy']*100:.2f}%")
    logger.info(f"   Corrélation: {results['BOTH_WEAK']['correlation']:.4f}")

    # Analyse logique
    logger.info(f"\n" + "=" * 80)
    logger.info(f"💡 ANALYSE LOGIQUE")
    logger.info("=" * 80)

    overlap_pct = results['BOTH_STRONG']['n'] / oracle_strong_count * 100 if oracle_strong_count > 0 else 0

    logger.info(f"\n📊 Overlap Oracle/Prédictions:")
    logger.info(f"   L'IA détecte {overlap_pct:.1f}% des vrais Force=STRONG de l'Oracle")

    if overlap_pct < 50:
        logger.info(f"\n   ⚠️  OVERLAP FAIBLE ({overlap_pct:.1f}%)")
        logger.info(f"      → L'IA et l'Oracle ne sont PAS d'accord sur Force=STRONG")
        logger.info(f"      → Problème de FEATURES ou APPRENTISSAGE")
    else:
        logger.info(f"\n   ✅ Overlap élevé ({overlap_pct:.1f}%)")
        logger.info(f"      → L'IA détecte bien les vrais Force=STRONG")

    # Analyse des erreurs IA
    pred_only_pct = results['PRED_ONLY']['n'] / pred_strong_count * 100 if pred_strong_count > 0 else 0

    logger.info(f"\n📊 Qualité des prédictions IA Force=STRONG:")
    logger.info(f"   Faux Positifs: {pred_only_pct:.1f}% des prédictions IA sont Oracle=WEAK")

    if results['PRED_ONLY']['accuracy'] < 0.52:
        logger.info(f"\n   ⚠️  FAUX POSITIFS = HASARD ({results['PRED_ONLY']['accuracy']*100:.2f}%)")
        logger.info(f"      → L'IA sélectionne de MAUVAIS samples Force=STRONG")
        logger.info(f"      → Ces samples n'ont AUCUN momentum vers le futur")
        logger.info(f"\n   💊 SOLUTION:")
        logger.info(f"      1. Réduire les faux positifs (seuil Force > 0.5 → 0.6 ou 0.7)")
        logger.info(f"      2. Ajouter features (Volume, ATR, confirmation temporelle)")
        logger.info(f"      3. Loss function avec pénalité sur faux positifs")
    else:
        logger.info(f"\n   ✅ Faux positifs acceptables ({results['PRED_ONLY']['accuracy']*100:.2f}%)")

    # Analyse des vrais positifs ratés
    if results['ORACLE_ONLY']['accuracy'] > 0.60:
        logger.info(f"\n📊 Opportunités manquées:")
        logger.info(f"   L'IA rate {results['ORACLE_ONLY']['n']:,} vrais Force=STRONG")
        logger.info(f"   Ces samples ont {results['ORACLE_ONLY']['accuracy']*100:.2f}% prédictivité")
        logger.info(f"\n   💊 SOLUTION:")
        logger.info(f"      → Réduire le seuil Force (0.5 → 0.4) pour capturer plus de vrais positifs")
        logger.info(f"      → Trade-off: plus de faux positifs, mais moins d'opportunités manquées")

    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    main()
