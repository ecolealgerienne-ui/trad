"""
Script d'évaluation du modèle CNN-LSTM sur le test set.

Évalue le meilleur modèle sauvegardé et calcule les métriques détaillées.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import json
import argparse
from typing import Dict

logger = logging.getLogger(__name__)

# Import modules locaux
from constants import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    RESULTS_DIR
)
from model import create_model, compute_metrics
from train import IndicatorDataset
from prepare_data import load_prepared_data
from data_utils import normalize_labels_for_single_output
from utils import log_dataset_metadata


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: str
) -> Dict[str, float]:
    """
    Évalue le modèle sur un dataset.

    Args:
        model: Modèle
        dataloader: DataLoader
        loss_fn: Loss function
        device: Device

    Returns:
        Dictionnaire avec toutes les métriques
    """
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            # Déplacer sur device
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # Forward
            outputs = model(X_batch)

            # Loss
            loss = loss_fn(outputs, Y_batch)

            # Accumuler
            total_loss += loss.item() * X_batch.size(0)
            all_predictions.append(outputs.cpu())
            all_targets.append(Y_batch.cpu())

    # Moyennes
    avg_loss = total_loss / len(dataloader.dataset)

    # Métriques
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_predictions, all_targets)
    metrics['loss'] = avg_loss

    return metrics


def compute_vote_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calcule les métriques du vote majoritaire (moyenne des 3 prédictions).

    Note: BOL retiré car impossible à synchroniser (toujours lag +1).

    Args:
        predictions: Probabilités (batch, 3)
        targets: Labels (batch, 3)
        threshold: Seuil de décision

    Returns:
        Dictionnaire avec métriques du vote
    """
    # Vote: moyenne des 3 probabilités
    vote_probs = predictions.mean(dim=1)  # (batch,)

    # Vote binaire
    vote_preds = (vote_probs >= threshold).float()

    # Target du vote: majorité des labels (>=2 sur 3)
    vote_targets = (targets.sum(dim=1) >= 2).float()

    # Métriques
    correct = (vote_preds == vote_targets).sum().item()
    total = len(vote_targets)
    accuracy = correct / total

    # TP, TN, FP, FN
    tp = ((vote_preds == 1) & (vote_targets == 1)).sum().item()
    tn = ((vote_preds == 0) & (vote_targets == 0)).sum().item()
    fp = ((vote_preds == 1) & (vote_targets == 0)).sum().item()
    fn = ((vote_preds == 0) & (vote_targets == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'vote_accuracy': accuracy,
        'vote_precision': precision,
        'vote_recall': recall,
        'vote_f1': f1,
        'vote_tp': tp,
        'vote_tn': tn,
        'vote_fp': fp,
        'vote_fn': fn
    }


def print_metrics_table(metrics: Dict[str, float], indicator_names: list = None):
    """
    Affiche un tableau formaté des métriques.

    Args:
        metrics: Dictionnaire de métriques
        indicator_names: Liste des noms d'indicateurs (auto-détecté si None)
    """
    logger.info("\n" + "="*80)
    logger.info("MÉTRIQUES PAR INDICATEUR")
    logger.info("="*80)

    # Header
    logger.info(f"{'Indicateur':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    logger.info("-"*80)

    # Déterminer les indicateurs à afficher
    if indicator_names is None:
        # Détecter depuis les métriques disponibles
        if 'RSI_accuracy' in metrics:
            indicator_names = ['RSI', 'CCI', 'MACD']
        elif 'INDICATOR_accuracy' in metrics:
            indicator_names = ['INDICATOR']
        else:
            indicator_names = []

    # Lignes par indicateur
    for name in indicator_names:
        acc = metrics.get(f'{name}_accuracy', 0.0)
        prec = metrics.get(f'{name}_precision', 0.0)
        rec = metrics.get(f'{name}_recall', 0.0)
        f1 = metrics.get(f'{name}_f1', 0.0)

        # Ne pas afficher si pas de données
        if acc == 0.0 and prec == 0.0 and rec == 0.0 and f1 == 0.0:
            continue

        logger.info(f"{name:<12} {acc:<10.3f} {prec:<10.3f} {rec:<10.3f} {f1:<10.3f}")

    # Moyennes (seulement si plus d'un indicateur)
    if len(indicator_names) > 1:
        logger.info("-"*80)
        avg_acc = metrics.get('avg_accuracy', 0.0)
        avg_prec = metrics.get('avg_precision', 0.0)
        avg_rec = metrics.get('avg_recall', 0.0)
        avg_f1 = metrics.get('avg_f1', 0.0)

        logger.info(f"{'MOYENNE':<12} {avg_acc:<10.3f} {avg_prec:<10.3f} {avg_rec:<10.3f} {avg_f1:<10.3f}")

    # Vote majoritaire
    if 'vote_accuracy' in metrics:
        logger.info("="*80)
        logger.info("VOTE MAJORITAIRE (Moyenne des 3 prédictions)")
        logger.info("="*80)

        vote_acc = metrics['vote_accuracy']
        vote_prec = metrics['vote_precision']
        vote_rec = metrics['vote_recall']
        vote_f1 = metrics['vote_f1']

        logger.info(f"{'VOTE':<12} {vote_acc:<10.3f} {vote_prec:<10.3f} {vote_rec:<10.3f} {vote_f1:<10.3f}")


def parse_args():
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(
        description='Évaluation du modèle CNN-LSTM sur le test set',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--data', '-d', type=str, required=True,
                        help='Chemin vers les données préparées (.npz). '
                             'IMPORTANT: Doit être le même dataset utilisé pour l\'entraînement!')

    parser.add_argument('--indicator', '-i', type=str, default='all',
                        choices=['all', 'rsi', 'cci', 'macd'],
                        help='Indicateur à évaluer (all=multi-output, rsi/cci/macd=single-output)')

    return parser.parse_args()


# Mapping indicateur -> index
INDICATOR_INDEX = {'rsi': 0, 'cci': 1, 'macd': 2}
INDICATOR_NAMES = ['RSI', 'CCI', 'MACD']


def main():
    """Pipeline complet d'évaluation."""
    # Parser arguments
    args = parse_args()

    # Configurer logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    logger.info("="*80)
    logger.info("ÉVALUATION DU MODÈLE CNN-LSTM")
    logger.info("="*80)

    # Déterminer mode (multi-output ou single-output)
    single_indicator = args.indicator != 'all'
    if single_indicator:
        indicator_idx = INDICATOR_INDEX[args.indicator]
        indicator_name = INDICATOR_NAMES[indicator_idx]
        num_outputs = 1
        logger.info(f"\n🎯 Mode SINGLE-OUTPUT: {indicator_name}")
    else:
        indicator_idx = None
        indicator_name = None
        num_outputs = 3
        logger.info(f"\n🎯 Mode MULTI-OUTPUT: RSI, CCI, MACD")

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"\nDevice: {device}")

    # Chemin du modèle (inclut l'indicateur si single-output)
    if single_indicator:
        model_path = BEST_MODEL_PATH.replace('.pth', f'_{args.indicator}.pth')
    else:
        model_path = BEST_MODEL_PATH

    # Vérifier que le modèle existe
    if not Path(model_path).exists():
        logger.error(f"❌ Modèle non trouvé: {model_path}")
        if single_indicator:
            logger.error(f"   Entraîner d'abord: python src/train.py --data <dataset> --indicator {args.indicator}")
        else:
            logger.error(f"   Entraîner d'abord le modèle: python src/train.py --data <dataset>")
        return

    # =========================================================================
    # 1. CHARGER LES DONNÉES
    # =========================================================================
    # Charger données préparées (même dataset que l'entraînement)
    logger.info(f"\n1. Chargement des données préparées: {args.data}")
    prepared = load_prepared_data(args.data)
    X_test, Y_test = prepared['test']
    metadata = prepared['metadata']
    log_dataset_metadata(metadata, logger)

    # Filtrer les labels si mode single-output
    if single_indicator:
        Y_test = normalize_labels_for_single_output(Y_test, indicator_idx, indicator_name)

    logger.info(f"  Test: X={X_test.shape}, Y={Y_test.shape}")

    # =========================================================================
    # 3. CRÉER DATALOADER
    # =========================================================================
    logger.info("\n3. Création du DataLoader...")

    test_dataset = IndicatorDataset(X_test, Y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    logger.info(f"  Test batches: {len(test_loader)}")

    # =========================================================================
    # 4. CHARGER LE MODÈLE
    # =========================================================================
    logger.info(f"\n4. Chargement du modèle depuis {model_path}...")

    # Charger checkpoint pour récupérer la config du modèle
    checkpoint = torch.load(model_path, map_location=device)

    # Récupérer config du modèle (ou utiliser défauts si ancien checkpoint)
    model_config = checkpoint.get('model_config', {})

    # Détecter le nombre de features depuis les données
    num_features = X_test.shape[2]

    # Utiliser num_outputs de la config ou celui déterminé par --indicator
    saved_num_outputs = model_config.get('num_outputs', 3)
    if saved_num_outputs != num_outputs:
        logger.warning(f"  ⚠️ num_outputs mismatch: modèle={saved_num_outputs}, demandé={num_outputs}")
        num_outputs = saved_num_outputs

    model, loss_fn = create_model(
        device=device,
        num_indicators=num_features,
        num_outputs=num_outputs,
        cnn_filters=model_config.get('cnn_filters', 64),
        lstm_hidden_size=model_config.get('lstm_hidden_size', 64),
        lstm_num_layers=model_config.get('lstm_num_layers', 2),
        lstm_dropout=model_config.get('lstm_dropout', 0.2),
        dense_hidden_size=model_config.get('dense_hidden_size', 32),
        dense_dropout=model_config.get('dense_dropout', 0.3),
    )

    # Charger poids
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"  ✅ Modèle chargé (époque {checkpoint['epoch']})")
    logger.info(f"     Val Loss: {checkpoint['val_loss']:.4f}")
    logger.info(f"     Val Acc: {checkpoint['val_accuracy']:.3f}")
    if single_indicator:
        logger.info(f"     Indicateur: {indicator_name}")
    if model_config:
        logger.info(f"     Config: CNN={model_config.get('cnn_filters')}, "
                   f"LSTM={model_config.get('lstm_hidden_size')}x{model_config.get('lstm_num_layers')}")

    # =========================================================================
    # 5. ÉVALUATION
    # =========================================================================
    logger.info("\n5. Évaluation sur test set...")

    metrics = evaluate_model(model, test_loader, loss_fn, device)

    logger.info(f"\n  Test Loss: {metrics['loss']:.4f}")

    # Afficher tableau
    print_metrics_table(metrics)

    # =========================================================================
    # 6. VOTE MAJORITAIRE (seulement en mode multi-output)
    # =========================================================================
    if not single_indicator:
        logger.info("\n6. Calcul du vote majoritaire...")

        # Prédictions complètes
        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                all_predictions.append(outputs.cpu())
                all_targets.append(Y_batch.cpu())

        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Métriques vote
        vote_metrics = compute_vote_metrics(all_predictions, all_targets)
        metrics.update(vote_metrics)

        # Afficher
        print_metrics_table(metrics)
    else:
        logger.info("\n6. Vote majoritaire: N/A (mode single-output)")

    # =========================================================================
    # 7. SAUVEGARDER RÉSULTATS
    # =========================================================================
    logger.info("\n7. Sauvegarde des résultats...")

    results_path = Path(RESULTS_DIR) / 'test_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"  Résultats sauvegardés: {results_path}")

    # =========================================================================
    # RÉSUMÉ FINAL
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("✅ ÉVALUATION TERMINÉE")
    logger.info("="*80)

    logger.info(f"\nRésultats clés:")
    logger.info(f"  Test Loss: {metrics['loss']:.4f}")
    if single_indicator:
        logger.info(f"  Indicateur: {indicator_name}")
        logger.info(f"  Accuracy: {metrics['avg_accuracy']:.3f}")
        logger.info(f"  F1: {metrics['avg_f1']:.3f}")
    else:
        logger.info(f"  Accuracy moyenne: {metrics['avg_accuracy']:.3f}")
        logger.info(f"  F1 moyen: {metrics['avg_f1']:.3f}")
        logger.info(f"  Vote majoritaire accuracy: {metrics['vote_accuracy']:.3f}")

    # Comparaison avec baseline (50% = hasard)
    baseline = 0.50
    improvement = (metrics['avg_accuracy'] - baseline) / baseline * 100

    logger.info(f"\n📈 Amélioration vs baseline (hasard):")
    logger.info(f"  Baseline: {baseline:.1%}")
    logger.info(f"  Modèle: {metrics['avg_accuracy']:.1%}")
    logger.info(f"  Gain: {improvement:+.1f}%")

    if metrics['avg_accuracy'] >= 0.70:
        logger.info(f"\n🎯 Objectif 70%+ atteint ! ✅")
    else:
        logger.info(f"\n⚠️ Objectif 70%+ pas encore atteint")
        logger.info(f"   Suggestions:")
        logger.info(f"   - Augmenter NUM_EPOCHS")
        logger.info(f"   - Ajuster hyperparamètres (CNN_FILTERS, LSTM_HIDDEN_SIZE)")
        logger.info(f"   - Vérifier qualité des labels")


if __name__ == '__main__':
    main()
