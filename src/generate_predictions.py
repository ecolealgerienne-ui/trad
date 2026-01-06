#!/usr/bin/env python3
"""
Générer et sauvegarder les prédictions du modèle dans le fichier .npz

Usage:
    python src/generate_predictions.py --data data/prepared/dataset_..._macd_dual_binary_kalman.npz
"""

import argparse
import logging
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from constants import BATCH_SIZE, DEVICE, MODELS_DIR
from model import create_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_predictions_for_split(model, X, Y, device, batch_size=128):
    """
    Génère les prédictions pour un split donné.

    Args:
        model: Modèle PyTorch
        X: Features (n_samples, seq_len, n_features)
        Y: Labels (n_samples, n_outputs)
        device: CPU ou CUDA
        batch_size: Taille des batchs

    Returns:
        Y_pred: Prédictions probabilistes (n_samples, n_outputs)
    """
    # Créer DataLoader
    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(Y)
    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Générer prédictions
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            outputs = torch.sigmoid(logits)  # Convertir logits en probabilités
            all_predictions.append(outputs.cpu().numpy())

    # Concatener
    Y_pred = np.concatenate(all_predictions, axis=0)

    return Y_pred


def main():
    parser = argparse.ArgumentParser(description='Générer et sauvegarder les prédictions')
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Chemin vers le dataset .npz'
    )
    parser.add_argument(
        '--indicator',
        type=str,
        help='Indicateur (auto-détecté depuis le nom de fichier si omis)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device à utiliser (défaut: auto)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help=f'Taille des batchs (défaut: {BATCH_SIZE})'
    )

    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    logger.info("="*80)
    logger.info("GÉNÉRATION DES PRÉDICTIONS")
    logger.info("="*80)
    logger.info(f"Device: {device}")
    logger.info(f"Dataset: {args.data}")

    # =========================================================================
    # 1. CHARGER LES DONNÉES
    # =========================================================================
    logger.info("\n1. Chargement des données...")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {data_path}")

    data = np.load(data_path, allow_pickle=True)

    # Extraire les splits
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    Y_train, Y_val, Y_test = data['Y_train'], data['Y_val'], data['Y_test']

    logger.info(f"  Train: X={X_train.shape}, Y={Y_train.shape}")
    logger.info(f"  Val:   X={X_val.shape}, Y={Y_val.shape}")
    logger.info(f"  Test:  X={X_test.shape}, Y={Y_test.shape}")

    # Détecter architecture
    n_features = X_train.shape[2]
    n_outputs = Y_train.shape[1]
    is_dual_binary = (n_outputs == 2)

    logger.info(f"\n  Architecture détectée:")
    logger.info(f"    Features: {n_features}")
    logger.info(f"    Outputs: {n_outputs}")
    logger.info(f"    Dual-Binary: {is_dual_binary}")

    # Auto-détecter l'indicateur depuis le nom de fichier
    if args.indicator is None:
        filename = data_path.stem.lower()
        if 'macd' in filename:
            indicator = 'macd'
        elif 'rsi' in filename:
            indicator = 'rsi'
        elif 'cci' in filename:
            indicator = 'cci'
        else:
            raise ValueError(f"Impossible de détecter l'indicateur depuis {filename}")
        logger.info(f"    Indicateur auto-détecté: {indicator.upper()}")
    else:
        indicator = args.indicator.lower()
        logger.info(f"    Indicateur: {indicator.upper()}")

    # =========================================================================
    # 2. CHARGER LE MODÈLE
    # =========================================================================
    logger.info("\n2. Chargement du modèle...")

    # Chercher le meilleur modèle
    model_name = f"best_model_{indicator}_kalman_dual_binary.pth"
    model_path = Path(MODELS_DIR) / model_name

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get('model_config', {})

    logger.info(f"  Modèle: {model_path}")
    logger.info(f"  Époque: {checkpoint['epoch']}")
    logger.info(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    logger.info(f"  Val Acc: {checkpoint['val_accuracy']:.3f}")

    # Créer le modèle
    model, _ = create_model(
        device=device,
        num_indicators=n_features,
        num_outputs=n_outputs,
        cnn_filters=model_config.get('cnn_filters', 64),
        lstm_hidden_size=model_config.get('lstm_hidden_size', 64),
        lstm_num_layers=model_config.get('lstm_num_layers', 2),
        lstm_dropout=model_config.get('lstm_dropout', 0.2),
        dense_hidden_size=model_config.get('dense_hidden_size', 32),
        dense_dropout=model_config.get('dense_dropout', 0.3),
        use_layer_norm=model_config.get('use_layer_norm', False),
        use_bce_with_logits=model_config.get('use_bce_with_logits', False),
    )

    # Charger les poids
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info("  ✅ Modèle chargé et prêt")

    # =========================================================================
    # 3. GÉNÉRER LES PRÉDICTIONS
    # =========================================================================
    logger.info("\n3. Génération des prédictions...")

    logger.info("  Train split...")
    Y_pred_train = generate_predictions_for_split(
        model, X_train, Y_train, device, args.batch_size
    )
    logger.info(f"    ✅ Y_pred_train: {Y_pred_train.shape}")

    logger.info("  Val split...")
    Y_pred_val = generate_predictions_for_split(
        model, X_val, Y_val, device, args.batch_size
    )
    logger.info(f"    ✅ Y_pred_val: {Y_pred_val.shape}")

    logger.info("  Test split...")
    Y_pred_test = generate_predictions_for_split(
        model, X_test, Y_test, device, args.batch_size
    )
    logger.info(f"    ✅ Y_pred_test: {Y_pred_test.shape}")

    # =========================================================================
    # 4. SAUVEGARDER LES PRÉDICTIONS
    # =========================================================================
    logger.info("\n4. Sauvegarde des prédictions dans le .npz...")

    # Charger toutes les données existantes
    existing_data = dict(data.items())

    # Ajouter les prédictions
    existing_data['Y_pred_train'] = Y_pred_train
    existing_data['Y_pred_val'] = Y_pred_val
    existing_data['Y_pred_test'] = Y_pred_test

    # Sauvegarder (écrase le fichier avec les nouvelles données)
    np.savez_compressed(data_path, **existing_data)

    logger.info(f"  ✅ Prédictions sauvegardées dans: {data_path}")

    # =========================================================================
    # 5. VÉRIFICATION
    # =========================================================================
    logger.info("\n5. Vérification...")

    # Recharger pour vérifier
    data_check = np.load(data_path, allow_pickle=True)

    logger.info("  Clés dans le fichier .npz:")
    for key in sorted(data_check.keys()):
        shape = data_check[key].shape if hasattr(data_check[key], 'shape') else 'N/A'
        logger.info(f"    - {key}: {shape}")

    # Vérifier que Y_pred existe
    assert 'Y_pred_train' in data_check, "Y_pred_train manquant!"
    assert 'Y_pred_val' in data_check, "Y_pred_val manquant!"
    assert 'Y_pred_test' in data_check, "Y_pred_test manquant!"

    logger.info("\n" + "="*80)
    logger.info("✅ PRÉDICTIONS GÉNÉRÉES ET SAUVEGARDÉES")
    logger.info("="*80)
    logger.info(f"\nVous pouvez maintenant lancer l'analyse:")
    logger.info(f"  python tests/analyze_why_8percent_kills.py \\")
    logger.info(f"      --data {args.data} \\")
    logger.info(f"      --indicator {indicator} \\")
    logger.info(f"      --split test")


if __name__ == '__main__':
    main()
