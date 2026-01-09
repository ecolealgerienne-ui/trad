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
from prepare_data import load_prepared_data, filter_by_assets
from data_utils import normalize_labels_for_single_output
from utils import log_dataset_metadata


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: str,
    indicator_names: list = None
) -> Dict[str, float]:
    """
    Évalue le modèle sur un dataset.

    Args:
        model: Modèle
        dataloader: DataLoader
        loss_fn: Loss function
        device: Device
        indicator_names: Noms des outputs (ex: ['Direction', 'Force'] pour dual-binary)

    Returns:
        Dictionnaire avec toutes les métriques
    """
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            # Unpacking flexible: (X, Y) ou (X, Y, T)
            if len(batch) == 3:
                X_batch, Y_batch, T_batch = batch
                # T_batch non utilisé en évaluation (seulement pour training loss)
            else:
                X_batch, Y_batch = batch

            # Déplacer sur device
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # Forward (retourne logits ou probabilités selon use_bce_with_logits)
            model_outputs = model(X_batch)

            # Loss (applique sigmoid si BCEWithLogitsLoss, sinon attend probabilités)
            loss = loss_fn(model_outputs, Y_batch)

            # Obtenir probabilités pour métriques (gère sigmoid conditionnellement)
            outputs = model.predict_proba(X_batch)

            # Accumuler
            total_loss += loss.item() * X_batch.size(0)
            all_predictions.append(outputs.cpu())
            all_targets.append(Y_batch.cpu())

    # Moyennes
    avg_loss = total_loss / len(dataloader.dataset)

    # Métriques
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_predictions, all_targets, indicator_names=indicator_names)
    metrics['loss'] = avg_loss

    return metrics


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

    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Chemin vers le modèle (.pth). Si non spécifié, utilise le chemin par défaut.')

    parser.add_argument('--indicator', '-i', type=str, default='all',
                        choices=['all', 'rsi', 'cci', 'macd', 'close', 'macd40', 'macd26', 'macd13'],
                        help='Indicateur à évaluer (all=multi-output, autres=single-output)')

    parser.add_argument('--filter', '-f', type=str, default=None,
                        help='Nom du filtre utilisé (ex: octave20, kalman). '
                             'Utilisé pour trouver le modèle automatiquement.')

    # Assets filtering
    parser.add_argument('--assets', type=str, nargs='+', default=None,
                        help='Assets à utiliser (ex: --assets BTC ETH). '
                             'Si non spécifié, utilise tous les assets du dataset.')

    return parser.parse_args()


# Mapping indicateur -> index (pour datasets multi-output)
# Pour les single-output (close, macd40, etc.), l'index est None
INDICATOR_INDEX = {
    'rsi': 0, 'cci': 1, 'macd': 2,
    'close': None, 'macd40': None, 'macd26': None, 'macd13': None
}
INDICATOR_NAMES = {
    'rsi': 'RSI', 'cci': 'CCI', 'macd': 'MACD',
    'close': 'CLOSE', 'macd40': 'MACD40', 'macd26': 'MACD26', 'macd13': 'MACD13'
}


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

    # NOTE: Mode sera déterminé APRÈS détection auto de l'indicateur depuis le nom du fichier

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"\nDevice: {device}")

    # =========================================================================
    # CHARGEMENT PRÉLIMINAIRE DES MÉTADONNÉES (pour détection filtre)
    # =========================================================================
    filter_type_metadata = None
    if args.data and not args.model:
        # Charger uniquement les métadonnées (rapide)
        try:
            preliminary_data = load_prepared_data(args.data)
            preliminary_metadata = preliminary_data.get('metadata', {})
            if preliminary_metadata and 'filter_type' in preliminary_metadata:
                filter_type_metadata = preliminary_metadata['filter_type']
        except Exception as e:
            logger.warning(f"⚠️ Impossible de charger les métadonnées: {e}")

    # =========================================================================
    # AUTO-DÉTECTION DU CHEMIN DU MODÈLE (logique identique à train.py)
    # =========================================================================
    if args.model:
        model_path = args.model
        # Si modèle spécifié manuellement, déterminer mode depuis args.indicator
        single_indicator = args.indicator != 'all'
        if single_indicator:
            indicator_idx = INDICATOR_INDEX[args.indicator]
            indicator_name = INDICATOR_NAMES[args.indicator]
            logger.info(f"\n🎯 Mode SINGLE-OUTPUT: {indicator_name}")
        else:
            indicator_idx = None
            indicator_name = None
            logger.info(f"\n🎯 Mode MULTI-OUTPUT: RSI, CCI, MACD")
    else:
        # Détecter l'indicateur et le filtre depuis le nom du fichier dataset
        detected_indicator = None
        detected_filter = None

        if args.data:
            data_name = Path(args.data).stem.lower()

            # Détecter indicateur (ex: dataset_..._rsi_dual_binary_kalman.npz → 'rsi')
            for ind in ['rsi', 'cci', 'macd', 'close']:
                if f'_{ind}_' in data_name or data_name.endswith(f'_{ind}'):
                    detected_indicator = ind
                    break

            # Détecter filtre (fallback si pas dans metadata)
            for filt in ['kalman', 'octave20', 'octave', 'decycler']:
                if filt in data_name:
                    detected_filter = filt
                    break

        # Priorité: CLI > filename
        if args.indicator and args.indicator != 'all':
            detected_indicator = args.indicator

        # Priorité pour le filtre: metadata > CLI argument > filename
        if filter_type_metadata:
            detected_filter = filter_type_metadata
        elif args.filter:
            detected_filter = args.filter

        # =========================================================================
        # DÉTERMINER MODE (single vs multi) APRÈS détection indicateur
        # =========================================================================
        # Si indicateur détecté (filename) OU CLI != 'all' → SINGLE-OUTPUT
        single_indicator = detected_indicator is not None or args.indicator != 'all'

        if single_indicator:
            if detected_indicator:
                indicator_idx = INDICATOR_INDEX.get(detected_indicator)
                indicator_name = INDICATOR_NAMES.get(detected_indicator, detected_indicator.upper())
            else:
                indicator_idx = INDICATOR_INDEX[args.indicator]
                indicator_name = INDICATOR_NAMES[args.indicator]
            num_outputs = 1
            logger.info(f"\n🎯 Mode SINGLE-OUTPUT: {indicator_name}")
            logger.info(f"   Indicateur détecté: {detected_indicator or args.indicator}")
        else:
            indicator_idx = None
            indicator_name = None
            num_outputs = 3
            logger.info(f"\n🎯 Mode MULTI-OUTPUT: RSI, CCI, MACD")

        # Construire le nom du modèle
        suffix_parts = []
        if detected_indicator:
            suffix_parts.append(detected_indicator)
        if detected_filter:
            suffix_parts.append(detected_filter)

        # Détecter si c'est dual-binary depuis le nom du fichier
        data_name_lower = Path(args.data).stem.lower()
        if args.data and 'dual_binary' in data_name_lower:
            suffix_parts.append('dual_binary')

        # Phase 2.11: Détecter si c'est un dataset avec transitions (_wt)
        if args.data and '_wt' in data_name_lower:
            suffix_parts.append('wt')

        if suffix_parts:
            suffix = '_'.join(suffix_parts)
            model_path = BEST_MODEL_PATH.replace('.pth', f'_{suffix}.pth')
        else:
            model_path = BEST_MODEL_PATH

        logger.info(f"\n🔍 Détection auto du modèle:")
        logger.info(f"  Indicateur détecté: {detected_indicator or 'aucun'}")
        logger.info(f"  Filtre détecté: {detected_filter or 'aucun'}")
        if filter_type_metadata:
            logger.info(f"  Source filtre: métadonnées")
        logger.info(f"  Chemin modèle: {model_path}")

    # Vérifier que le modèle existe
    if not Path(model_path).exists():
        logger.error(f"❌ Modèle non trouvé: {model_path}")
        if single_indicator:
            filter_hint = f" --filter {args.filter}" if args.filter else ""
            logger.error(f"   Entraîner d'abord: python src/train.py --data {args.data} --indicator {args.indicator}{filter_hint}")
        else:
            logger.error(f"   Entraîner d'abord le modèle: python src/train.py --data {args.data}")
        return

    # =========================================================================
    # 1. CHARGER LES DONNÉES
    # =========================================================================
    # Charger données préparées (même dataset que l'entraînement)
    logger.info(f"\n1. Chargement des données préparées: {args.data}")
    prepared = load_prepared_data(args.data)

    # Unpacking flexible: (X, Y) ou (X, Y, T)
    if len(prepared['test']) == 3:
        X_test, Y_test, T_test = prepared['test']
        has_transitions = True
        logger.info("  ✅ Dataset avec transitions détecté (Phase 2.11)")
    else:
        X_test, Y_test = prepared['test']
        T_test = None
        has_transitions = False

    metadata = prepared['metadata']
    log_dataset_metadata(metadata, logger)

    # FILTRAGE PAR ASSETS (optionnel)
    if args.assets:
        logger.info(f"\n🔍 Filtrage des assets...")

        # Charger OHLCV depuis le fichier .npz pour le filtrage
        data_npz = np.load(args.data, allow_pickle=True)

        # Filtrer test
        X_test, Y_test, T_test, _ = filter_by_assets(
            X_test, Y_test, T_test, data_npz['OHLCV_test'],
            args.assets, metadata
        )

        logger.info(f"  ✅ Filtrage terminé pour {len(args.assets)} asset(s)")

    # Filtrer les labels si mode single-output
    if single_indicator:
        Y_test = normalize_labels_for_single_output(Y_test, indicator_idx, indicator_name)

    logger.info(f"  Test: X={X_test.shape}, Y={Y_test.shape}")
    if has_transitions:
        logger.info(f"        T={T_test.shape} (transitions: {T_test.mean()*100:.1f}%)")

    # =========================================================================
    # 3. CRÉER DATALOADER
    # =========================================================================
    logger.info("\n3. Création du DataLoader...")

    test_dataset = IndicatorDataset(X_test, Y_test, T_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    logger.info(f"  Test batches: {len(test_loader)}")

    # =========================================================================
    # 4. CHARGER LE MODÈLE ET AUTO-DÉTECTER L'ARCHITECTURE
    # =========================================================================
    logger.info(f"\n4. Chargement du modèle depuis {model_path}...")

    # Charger checkpoint pour récupérer la config du modèle
    checkpoint = torch.load(model_path, map_location=device)

    # Récupérer config du modèle (ou utiliser défauts si ancien checkpoint)
    model_config = checkpoint.get('model_config', {})

    # =========================================================================
    # AUTO-DÉTECTION DE L'ARCHITECTURE (comme train.py)
    # =========================================================================

    # Détecter depuis le checkpoint
    is_dual_binary = model_config.get('is_dual_binary', False)
    indicator_for_metrics_saved = model_config.get('indicator_for_metrics', None)

    # Détecter depuis les metadata du dataset
    if metadata and 'label_names' in metadata and len(metadata['label_names']) == 2:
        is_dual_binary = True
        if not indicator_for_metrics_saved:
            label_name = metadata['label_names'][0]  # Ex: 'rsi_dir'
            indicator_for_metrics_saved = label_name.split('_')[0].upper()  # 'RSI'

    # Détecter le nombre de features et outputs depuis les données
    n_features_detected = X_test.shape[2]
    n_outputs_detected = Y_test.shape[1]

    logger.info(f"\n🔍 Architecture détectée:")
    logger.info(f"  Features: {n_features_detected}")
    logger.info(f"  Outputs: {n_outputs_detected}")
    logger.info(f"  Dual-Binary: {is_dual_binary}")
    if indicator_for_metrics_saved:
        logger.info(f"  Indicateur: {indicator_for_metrics_saved}")
    if metadata and 'filter_type' in metadata:
        logger.info(f"  Filtre: {metadata['filter_type'].upper()}")

    # Utiliser num_outputs de la config ou celui détecté depuis les données
    num_features = n_features_detected
    saved_num_outputs = model_config.get('num_outputs', n_outputs_detected)

    if saved_num_outputs != n_outputs_detected:
        logger.warning(f"  ⚠️ num_outputs mismatch: modèle={saved_num_outputs}, données={n_outputs_detected}")
        num_outputs = saved_num_outputs
    else:
        num_outputs = n_outputs_detected

    # Préparer les noms d'indicateurs pour les métriques
    if is_dual_binary:
        # Dual-binary: ['Direction', 'Force']
        indicator_names_for_metrics = ['Direction', 'Force']
        logger.info(f"  Mode Dual-Binary détecté: {indicator_names_for_metrics}")
    elif single_indicator:
        # Single-output: ['MACD'] ou ['RSI'] etc.
        indicator_names_for_metrics = [indicator_name]
    else:
        # Multi-output: ['RSI', 'CCI', 'MACD'] (défaut)
        indicator_names_for_metrics = None  # compute_metrics utilisera les défauts

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
        use_layer_norm=model_config.get('use_layer_norm', True),  # Par défaut True pour rétrocompatibilité
        use_bce_with_logits=model_config.get('use_bce_with_logits', True),  # Par défaut True pour rétrocompatibilité
        use_shortcut=model_config.get('use_shortcut', False),
        shortcut_steps=model_config.get('shortcut_steps', 5),
        use_temporal_gate=model_config.get('use_temporal_gate', False),
    )

    # Charger poids
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"\n✅ Modèle chargé:")
    logger.info(f"  Époque: {checkpoint['epoch']}")
    logger.info(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    logger.info(f"  Val Acc: {checkpoint['val_accuracy']:.3f}")
    if indicator_for_metrics_saved:
        logger.info(f"  Indicateur: {indicator_for_metrics_saved}")
    if model_config:
        logger.info(f"  Config: CNN={model_config.get('cnn_filters')}, "
                   f"LSTM={model_config.get('lstm_hidden_size')}x{model_config.get('lstm_num_layers')}")

    # =========================================================================
    # 5. ÉVALUATION
    # =========================================================================
    logger.info("\n5. Évaluation sur test set...")

    metrics = evaluate_model(model, test_loader, loss_fn, device, indicator_names=indicator_names_for_metrics)

    # Affichage des métriques selon le mode
    if is_dual_binary:
        # Dual-binary: afficher Direction et Force séparément
        logger.info(f"\n📊 Résultats Test:")
        logger.info(f"  Loss: {metrics['loss']:.4f}, Avg Acc: {metrics['avg_accuracy']:.3f}")
        logger.info(f"  Direction: Acc={metrics.get('Direction_accuracy', 0):.3f}, "
                   f"F1={metrics.get('Direction_f1', 0):.3f}, "
                   f"Prec={metrics.get('Direction_precision', 0):.3f}, "
                   f"Rec={metrics.get('Direction_recall', 0):.3f}")
        logger.info(f"  Force:     Acc={metrics.get('Force_accuracy', 0):.3f}, "
                   f"F1={metrics.get('Force_f1', 0):.3f}, "
                   f"Prec={metrics.get('Force_precision', 0):.3f}, "
                   f"Rec={metrics.get('Force_recall', 0):.3f}")
    else:
        logger.info(f"\n  Test Loss: {metrics['loss']:.4f}")

    # Afficher tableau complet
    print_metrics_table(metrics, indicator_names=indicator_names_for_metrics)

    # =========================================================================
    # 6. SAUVEGARDER RÉSULTATS
    # =========================================================================
    logger.info("\n6. Sauvegarde des résultats...")

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

    if is_dual_binary:
        # Mode dual-binary: afficher Direction et Force
        logger.info(f"  Mode: Dual-Binary ({indicator_for_metrics_saved or 'INDICATOR'})")
        logger.info(f"  Avg Accuracy: {metrics['avg_accuracy']:.3f}")
        logger.info(f"  Direction - Acc: {metrics.get('Direction_accuracy', 0):.3f}, F1: {metrics.get('Direction_f1', 0):.3f}")
        logger.info(f"  Force - Acc: {metrics.get('Force_accuracy', 0):.3f}, F1: {metrics.get('Force_f1', 0):.3f}")
    elif single_indicator:
        # Mode single-output
        logger.info(f"  Indicateur: {indicator_name}")
        logger.info(f"  Accuracy: {metrics['avg_accuracy']:.3f}")
        logger.info(f"  F1: {metrics['avg_f1']:.3f}")
    else:
        # Mode multi-output
        logger.info(f"  Accuracy moyenne: {metrics['avg_accuracy']:.3f}")
        logger.info(f"  F1 moyen: {metrics['avg_f1']:.3f}")

    # Comparaison avec baseline (50% = hasard)
    baseline = 0.50
    improvement = (metrics['avg_accuracy'] - baseline) / baseline * 100

    logger.info(f"\n📈 Amélioration vs baseline (hasard):")
    logger.info(f"  Baseline: {baseline:.1%}")
    logger.info(f"  Modèle: {metrics['avg_accuracy']:.1%}")
    logger.info(f"  Gain: {improvement:+.1f}%")

    # Objectif selon le mode
    if is_dual_binary:
        # Objectif dual-binary: Direction 85%+, Force 65-70%+
        dir_acc = metrics.get('Direction_accuracy', 0)
        force_acc = metrics.get('Force_accuracy', 0)

        logger.info(f"\n🎯 Objectifs Dual-Binary:")
        if dir_acc >= 0.85:
            logger.info(f"  Direction: {dir_acc:.1%} ✅ (objectif 85%+)")
        else:
            logger.info(f"  Direction: {dir_acc:.1%} ⚠️ (objectif 85%+)")

        if force_acc >= 0.65:
            logger.info(f"  Force: {force_acc:.1%} ✅ (objectif 65-70%+)")
        else:
            logger.info(f"  Force: {force_acc:.1%} ⚠️ (objectif 65-70%+)")
    else:
        # Objectif classique: 70%+
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
