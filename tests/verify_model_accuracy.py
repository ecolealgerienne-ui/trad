#!/usr/bin/env python3
"""
Test SIMPLE : Vérifier les 92% Direction et 86% Force

Compare directement Y (oracle) vs Y_pred (modèle) sample par sample.

Si on ne retrouve PAS ~92% et ~86%, alors il y a un problème :
- Indexation incorrecte ([:, 0] et [:, 1] inversés ?)
- Split incorrect (92% sur train, pas sur test ?)
- Conversion incorrecte (seuillage > 0.5 pas bon ?)
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
        'Y': data[f'Y_{split}'],
        'Y_pred': data.get(f'Y_{split}_pred', None),
    }


def compute_accuracy(y_true, y_pred_binary):
    """Calcule accuracy simple."""
    correct = (y_true == y_pred_binary).astype(float)
    return correct.mean()


def compute_confusion_matrix(y_true, y_pred_binary):
    """Calcule matrice de confusion."""
    tp = ((y_true == 1) & (y_pred_binary == 1)).sum()
    tn = ((y_true == 0) & (y_pred_binary == 0)).sum()
    fp = ((y_true == 0) & (y_pred_binary == 1)).sum()
    fn = ((y_true == 1) & (y_pred_binary == 0)).sum()

    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Vérifier accuracy réelle du modèle")
    parser.add_argument('--indicator', type=str, required=True, choices=['macd', 'rsi', 'cci'])
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])

    args = parser.parse_args()

    # Charger données
    data = load_dataset(args.indicator, args.split)

    Y_oracle = data['Y']
    Y_pred = data['Y_pred']

    if Y_pred is None:
        logger.error("❌ Prédictions non disponibles. Exécuter train.py + evaluate.py d'abord.")
        return

    n_samples = len(Y_oracle)

    logger.info("=" * 80)
    logger.info(f"🔬 VÉRIFICATION ACCURACY MODÈLE - {args.indicator.upper()} ({args.split})")
    logger.info("=" * 80)
    logger.info(f"\n📊 Dataset: {n_samples:,} samples")

    # Extraire labels Oracle (déjà binaires)
    oracle_dir = Y_oracle[:, 0]
    oracle_force = Y_oracle[:, 1]

    # Convertir prédictions en binaire avec seuil 0.5
    pred_dir = (Y_pred[:, 0] > 0.5).astype(int)
    pred_force = (Y_pred[:, 1] > 0.5).astype(int)

    # Vérifier que Y_oracle est bien binaire
    assert np.all((oracle_dir == 0) | (oracle_dir == 1)), "❌ Oracle Direction pas binaire!"
    assert np.all((oracle_force == 0) | (oracle_force == 1)), "❌ Oracle Force pas binaire!"

    # Distributions
    logger.info(f"\n📊 Distributions Oracle:")
    logger.info(f"   Direction UP:     {(oracle_dir == 1).sum():,} ({(oracle_dir == 1).mean()*100:.1f}%)")
    logger.info(f"   Direction DOWN:   {(oracle_dir == 0).sum():,} ({(oracle_dir == 0).mean()*100:.1f}%)")
    logger.info(f"   Force STRONG:     {(oracle_force == 1).sum():,} ({(oracle_force == 1).mean()*100:.1f}%)")
    logger.info(f"   Force WEAK:       {(oracle_force == 0).sum():,} ({(oracle_force == 0).mean()*100:.1f}%)")

    logger.info(f"\n📊 Distributions Prédictions:")
    logger.info(f"   Direction UP:     {(pred_dir == 1).sum():,} ({(pred_dir == 1).mean()*100:.1f}%)")
    logger.info(f"   Direction DOWN:   {(pred_dir == 0).sum():,} ({(pred_dir == 0).mean()*100:.1f}%)")
    logger.info(f"   Force STRONG:     {(pred_force == 1).sum():,} ({(pred_force == 1).mean()*100:.1f}%)")
    logger.info(f"   Force WEAK:       {(pred_force == 0).sum():,} ({(pred_force == 0).mean()*100:.1f}%)")

    # Calculer accuracies
    logger.info(f"\n" + "=" * 80)
    logger.info(f"📊 ACCURACY SAMPLE PAR SAMPLE")
    logger.info("=" * 80)

    acc_dir = compute_accuracy(oracle_dir, pred_dir)
    acc_force = compute_accuracy(oracle_force, pred_force)

    logger.info(f"\n   Direction Accuracy: {acc_dir*100:.2f}%")
    logger.info(f"   Force Accuracy:     {acc_force*100:.2f}%")

    # Confusion matrices
    logger.info(f"\n" + "=" * 80)
    logger.info(f"📊 MATRICE DE CONFUSION - DIRECTION")
    logger.info("=" * 80)

    cm_dir = compute_confusion_matrix(oracle_dir, pred_dir)
    logger.info(f"\n   True Positives (UP correctement prédit):    {cm_dir['tp']:,}")
    logger.info(f"   True Negatives (DOWN correctement prédit):  {cm_dir['tn']:,}")
    logger.info(f"   False Positives (UP prédit, réel DOWN):     {cm_dir['fp']:,}")
    logger.info(f"   False Negatives (DOWN prédit, réel UP):     {cm_dir['fn']:,}")
    logger.info(f"\n   Precision: {cm_dir['precision']*100:.2f}%")
    logger.info(f"   Recall:    {cm_dir['recall']*100:.2f}%")

    logger.info(f"\n" + "=" * 80)
    logger.info(f"📊 MATRICE DE CONFUSION - FORCE")
    logger.info("=" * 80)

    cm_force = compute_confusion_matrix(oracle_force, pred_force)
    logger.info(f"\n   True Positives (STRONG correctement prédit):  {cm_force['tp']:,}")
    logger.info(f"   True Negatives (WEAK correctement prédit):    {cm_force['tn']:,}")
    logger.info(f"   False Positives (STRONG prédit, réel WEAK):   {cm_force['fp']:,}")
    logger.info(f"   False Negatives (WEAK prédit, réel STRONG):   {cm_force['fn']:,}")
    logger.info(f"\n   Precision: {cm_force['precision']*100:.2f}%")
    logger.info(f"   Recall:    {cm_force['recall']*100:.2f}%")

    # Validation
    logger.info(f"\n" + "=" * 80)
    logger.info(f"✅ VALIDATION")
    logger.info("=" * 80)

    expected_dir = 92.4 if args.indicator == 'macd' else 87.0
    expected_force = 81.5 if args.indicator == 'macd' else 74.0

    delta_dir = acc_dir * 100 - expected_dir
    delta_force = acc_force * 100 - expected_force

    logger.info(f"\n   Direction:")
    logger.info(f"      Attendu:  ~{expected_dir:.1f}%")
    logger.info(f"      Obtenu:   {acc_dir*100:.2f}%")
    logger.info(f"      Delta:    {delta_dir:+.2f}%")

    logger.info(f"\n   Force:")
    logger.info(f"      Attendu:  ~{expected_force:.1f}%")
    logger.info(f"      Obtenu:   {acc_force*100:.2f}%")
    logger.info(f"      Delta:    {delta_force:+.2f}%")

    if abs(delta_dir) > 5 or abs(delta_force) > 5:
        logger.info(f"\n   ⚠️  ÉCART SIGNIFICATIF (> 5%)")
        logger.info(f"\n   Hypothèses possibles:")
        logger.info(f"      1. Les 92%/86% sont sur TRAIN, pas sur {args.split.upper()}")
        logger.info(f"      2. Problème d'indexation ([:, 0] et [:, 1] inversés ?)")
        logger.info(f"      3. Problème de conversion (seuil 0.5 incorrect ?)")
        logger.info(f"      4. Overfitting massif (train >> test)")
        logger.info(f"\n   💊 À FAIRE:")
        logger.info(f"      → Tester sur TRAIN: python tests/verify_model_accuracy.py --indicator {args.indicator} --split train")
        logger.info(f"      → Comparer train vs test")
    else:
        logger.info(f"\n   ✅ Accuracies confirmées (écart < 5%)")

    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    main()
