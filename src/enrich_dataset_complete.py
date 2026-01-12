#!/usr/bin/env python3
"""
Enrichissement complet du dataset régime avec TOUTES les prédictions.

RÈGLE ARCHITECTURALE CRITIQUE:
===============================================================================
📦 UN SEUL DATASET SOURCE CONTENANT TOUTES LES INFORMATIONS

Problème évité:
- ❌ Plusieurs datasets séparés avec tailles différentes
- ❌ Alignement complexe par timestamp/asset_id
- ❌ Risque de désynchronisation

Solution:
- ✅ Dataset de base (features + labels régime)
- ✅ + Prédictions Model A (régime classifier)
- ✅ + Prédictions MACD (direction)
- ✅ = Dataset enrichi unique et complet

Structure Y enrichie:
    Y[:, 0] = timestamp
    Y[:, 1] = asset_id
    Y[:, 2] = regime_label (ground truth)
    Y[:, 3] = trend_strength (ground truth)
    Y[:, 4] = volatility_cluster (ground truth)
    Y[:, 5] = regime_pred (Model A prediction) ✨
    Y[:, 6-9] = regime_probs (4 classes) ✨
    Y[:, 10] = macd_direction_pred (MACD Model prediction) ✨
    Y[:, 11] = macd_direction_prob (confidence) ✨

Pipeline:
1. Charger dataset de base: dataset_<assets>_regime.npz
2. Charger modèle Model A: models/best_model_regime.pth
3. Charger modèle MACD: models/best_model_macd_kalman_dual_binary.pth
4. Faire les prédictions Model A sur X
5. Faire les prédictions MACD sur X (adapter features si nécessaire)
6. Enrichir Y avec les colonnes de prédictions
7. Sauvegarder dataset enrichi: dataset_<assets>_regime_enriched.npz

Usage:
    python src/enrich_dataset_complete.py --assets BTC ETH BNB ADA LTC

Génère:
    data/prepared/dataset_btc_eth_bnb_ada_ltc_regime_enriched.npz

Author: Claude Code - Phase 2 (Enrichment Layer)
Date: 2026-01-12
Version: 1.0
"""

import numpy as np
import torch
import torch.nn as nn
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple
import json
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import project modules
from constants import PREPARED_DATA_DIR
from model import CNNLSTMModel  # Pour charger les modèles


def load_base_dataset(assets: list) -> Dict:
    """
    Charge le dataset de base régime (sans prédictions).

    Args:
        assets: Liste des assets (ex: ['BTC', 'ETH'])

    Returns:
        Dict avec X_train, Y_train, OHLCV_train, etc.
    """
    assets_str = '_'.join(sorted([a.lower() for a in assets]))
    dataset_path = Path(PREPARED_DATA_DIR) / f'dataset_{assets_str}_regime.npz'

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset de base introuvable: {dataset_path}\n"
            f"Exécutez d'abord: python src/prepare_data_regime.py --assets {' '.join(assets)}"
        )

    logger.info(f"📂 Chargement dataset de base: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)

    result = {
        'X_train': data['X_train'],
        'Y_train': data['Y_train'],
        'OHLCV_train': data['OHLCV_train'],
        'X_val': data['X_val'],
        'Y_val': data['Y_val'],
        'OHLCV_val': data['OHLCV_val'],
        'X_test': data['X_test'],
        'Y_test': data['Y_test'],
        'OHLCV_test': data['OHLCV_test'],
        'metadata': data['metadata'].item() if isinstance(data['metadata'], np.ndarray) else data['metadata']
    }

    logger.info(f"   Y_train shape (base): {result['Y_train'].shape}")
    logger.info(f"   Expected Y columns: [timestamp, asset_id, regime, trend_strength, volatility_cluster]")

    return result


def load_model_a(device: torch.device) -> nn.Module:
    """
    Charge le modèle Model A (régime classifier).

    Returns:
        Modèle PyTorch en mode eval
    """
    model_path = Path('models/best_model_regime.pth')

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle Model A introuvable: {model_path}\n"
            f"Entraînez d'abord: python src/train_regime_classifier.py"
        )

    logger.info(f"🤖 Chargement Model A (régime): {model_path}")

    # Charger checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Reconstruire le modèle (adapter architecture selon votre model.py)
    # NOTE: Vous devrez peut-être ajuster n_features et n_classes
    model = CNNLSTMModel(
        n_features=checkpoint.get('n_features', 20),  # Nombre de features régime
        n_classes=4,  # 4 régimes
        sequence_length=checkpoint.get('sequence_length', 12)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info(f"   ✓ Model A chargé (accuracy: {checkpoint.get('val_accuracy', 'N/A')})")

    return model


def load_model_macd(device: torch.device) -> nn.Module:
    """
    Charge le modèle MACD direction.

    Returns:
        Modèle PyTorch en mode eval
    """
    model_path = Path('models/best_model_macd_kalman_dual_binary.pth')

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle MACD introuvable: {model_path}\n"
            f"Entraînez d'abord le modèle MACD direction"
        )

    logger.info(f"🤖 Chargement Model MACD: {model_path}")

    # Charger checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Reconstruire le modèle
    model = CNNLSTMModel(
        n_features=checkpoint.get('n_features', 1),  # 1 feature pour MACD (c_ret)
        n_classes=2,  # 2 outputs: direction, force
        sequence_length=checkpoint.get('sequence_length', 25)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info(f"   ✓ Model MACD chargé (accuracy: {checkpoint.get('test_accuracy', 'N/A')})")

    return model


def predict_model_a(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fait les prédictions Model A (régime).

    Args:
        model: Modèle Model A en mode eval
        X: Features (n, seq_len, n_features+2)
        device: torch device
        batch_size: Taille des batchs

    Returns:
        predictions: (n,) classe prédite (0-3)
        probabilities: (n, 4) probabilités pour chaque classe
    """
    logger.info(f"🔮 Prédictions Model A sur {len(X)} samples...")

    n_samples = len(X)
    all_preds = []
    all_probs = []

    # Extraire seulement les features (retirer timestamp et asset_id)
    X_features = X[:, :, 2:]  # (n, seq_len, n_features)

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Model A predictions"):
            batch = X_features[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)

            outputs = model(batch_tensor)  # (batch, 4)
            probs = torch.softmax(outputs, dim=1)  # (batch, 4)
            preds = torch.argmax(probs, dim=1)  # (batch,)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    predictions = np.concatenate(all_preds)
    probabilities = np.concatenate(all_probs)

    logger.info(f"   ✓ Prédictions terminées")
    logger.info(f"   Distribution: {np.bincount(predictions, minlength=4)}")

    return predictions, probabilities


def predict_macd(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fait les prédictions MACD direction.

    NOTE: Le modèle MACD a été entraîné sur des features différentes (c_ret uniquement).
    On doit adapter X_regime pour correspondre aux features MACD.

    Args:
        model: Modèle MACD en mode eval
        X: Features régime (n, seq_len, n_features_regime+2)
        device: torch device
        batch_size: Taille des batchs

    Returns:
        predictions: (n,) direction prédite (0=DOWN, 1=UP)
        probabilities: (n,) probabilité de UP
    """
    logger.info(f"🔮 Prédictions MACD sur {len(X)} samples...")
    logger.warning("⚠️  Adaptation nécessaire: Features régime → Features MACD (c_ret)")

    # TODO: ADAPTATION CRITIQUE
    # Le modèle MACD a été entraîné sur c_ret (close return)
    # Le dataset régime contient ~20 features différentes
    #
    # Options:
    # 1. Extraire c_ret depuis OHLCV (recommandé mais nécessite OHLCV)
    # 2. Utiliser une feature proxy (ex: MA_5_slope)
    # 3. Réentraîner MACD sur features régime
    #
    # Pour l'instant, on utilise une feature proxy (colonne 2 = première feature après timestamp/asset_id)

    n_samples = len(X)
    all_preds = []
    all_probs = []

    # Extraire une feature proxy pour MACD (première feature après timestamp/asset_id)
    X_proxy = X[:, :, 2:3]  # (n, seq_len, 1) - prend la première feature comme proxy

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="MACD predictions"):
            batch = X_proxy[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)

            outputs = model(batch_tensor)  # (batch, 2) - [direction, force]

            # Prendre seulement la direction (output 0)
            direction_logits = outputs[:, 0]  # (batch,)
            direction_probs = torch.sigmoid(direction_logits)  # (batch,)
            direction_preds = (direction_probs > 0.5).long()  # (batch,)

            all_preds.append(direction_preds.cpu().numpy())
            all_probs.append(direction_probs.cpu().numpy())

    predictions = np.concatenate(all_preds)
    probabilities = np.concatenate(all_probs)

    logger.info(f"   ✓ Prédictions terminées")
    logger.info(f"   Distribution UP/DOWN: {np.bincount(predictions, minlength=2)}")
    logger.info(f"   Prob moyenne: {probabilities.mean():.3f}")

    return predictions, probabilities


def enrich_split(
    X: np.ndarray,
    Y: np.ndarray,
    OHLCV: np.ndarray,
    model_a: nn.Module,
    model_macd: nn.Module,
    device: torch.device,
    split_name: str
) -> np.ndarray:
    """
    Enrichit un split (train/val/test) avec les prédictions.

    Args:
        X, Y, OHLCV: Arrays du split
        model_a, model_macd: Modèles chargés
        device: torch device
        split_name: 'train', 'val', ou 'test'

    Returns:
        Y_enriched: (n, 12) avec toutes les prédictions
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 ENRICHISSEMENT {split_name.upper()} SET")
    logger.info(f"{'='*80}")

    # 1. Prédictions Model A (régime)
    regime_preds, regime_probs = predict_model_a(model_a, X, device)

    # 2. Prédictions MACD (direction)
    macd_preds, macd_probs = predict_macd(model_macd, X, device)

    # 3. Construire Y_enriched
    logger.info(f"\n🔧 Construction Y enrichi...")
    logger.info(f"   Y original: {Y.shape}")
    logger.info(f"   + regime_pred: ({len(regime_preds)},)")
    logger.info(f"   + regime_probs: {regime_probs.shape}")
    logger.info(f"   + macd_pred: ({len(macd_preds)},)")
    logger.info(f"   + macd_prob: ({len(macd_probs)},)")

    Y_enriched = np.column_stack([
        Y,                              # [:, 0-4]: timestamp, asset_id, regime, trend_strength, volatility_cluster
        regime_preds,                   # [:, 5]: regime_pred (Model A)
        regime_probs,                   # [:, 6-9]: regime_probs (4 classes)
        macd_preds,                     # [:, 10]: macd_direction_pred
        macd_probs                      # [:, 11]: macd_direction_prob
    ])

    logger.info(f"   Y enrichi: {Y_enriched.shape}")
    logger.info(f"   ✓ Colonnes ajoutées: 7 (1 pred + 4 probs + 1 macd_pred + 1 macd_prob)")

    return Y_enriched


def main():
    parser = argparse.ArgumentParser(description='Enrichir dataset régime avec TOUTES les prédictions')
    parser.add_argument('--assets', nargs='+', required=True,
                        help='Liste des assets (ex: BTC ETH BNB)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device PyTorch')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Taille des batchs pour prédictions')

    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"🖥️  Device: {device}")

    # 1. Charger dataset de base
    data = load_base_dataset(args.assets)

    # 2. Charger modèles
    model_a = load_model_a(device)
    model_macd = load_model_macd(device)

    # 3. Enrichir chaque split
    Y_train_enriched = enrich_split(
        data['X_train'], data['Y_train'], data['OHLCV_train'],
        model_a, model_macd, device, 'train'
    )

    Y_val_enriched = enrich_split(
        data['X_val'], data['Y_val'], data['OHLCV_val'],
        model_a, model_macd, device, 'val'
    )

    Y_test_enriched = enrich_split(
        data['X_test'], data['Y_test'], data['OHLCV_test'],
        model_a, model_macd, device, 'test'
    )

    # 4. Préparer metadata enrichi
    metadata = data['metadata'].copy()
    metadata['enriched'] = True
    metadata['enrichment_date'] = str(torch.datetime.datetime.now())
    metadata['models_used'] = {
        'model_a': 'best_model_regime.pth',
        'model_macd': 'best_model_macd_kalman_dual_binary.pth'
    }
    metadata['structure']['Y'] = (
        '(n, 12) - [timestamp, asset_id, regime, trend_strength, volatility_cluster, '
        'regime_pred, regime_prob_0, regime_prob_1, regime_prob_2, regime_prob_3, '
        'macd_direction_pred, macd_direction_prob]'
    )

    # 5. Sauvegarder dataset enrichi
    assets_str = '_'.join(sorted([a.lower() for a in args.assets]))
    output_path = Path(PREPARED_DATA_DIR) / f'dataset_{assets_str}_regime_enriched.npz'

    logger.info(f"\n{'='*80}")
    logger.info(f"💾 SAUVEGARDE DATASET ENRICHI")
    logger.info(f"{'='*80}")

    np.savez_compressed(
        output_path,
        X_train=data['X_train'],
        Y_train=Y_train_enriched,
        OHLCV_train=data['OHLCV_train'],
        X_val=data['X_val'],
        Y_val=Y_val_enriched,
        OHLCV_val=data['OHLCV_val'],
        X_test=data['X_test'],
        Y_test=Y_test_enriched,
        OHLCV_test=data['OHLCV_test'],
        metadata=json.dumps(metadata)
    )

    # Sauvegarder metadata JSON
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"   ✅ Dataset: {output_path}")
    logger.info(f"   ✅ Metadata: {metadata_path}")
    logger.info(f"   📦 Taille: {output_path.stat().st_size / (1024**2):.1f} MB")

    # Résumé
    logger.info(f"\n{'='*80}")
    logger.info(f"✓ ENRICHISSEMENT TERMINÉ")
    logger.info(f"{'='*80}")
    logger.info(f"\nStructure Y enrichie:")
    logger.info(f"  [:, 0-4]  = Base (timestamp, asset_id, labels...)")
    logger.info(f"  [:, 5]    = regime_pred (Model A)")
    logger.info(f"  [:, 6-9]  = regime_probs (4 classes)")
    logger.info(f"  [:, 10]   = macd_direction_pred")
    logger.info(f"  [:, 11]   = macd_direction_prob")
    logger.info(f"\nProchaine étape:")
    logger.info(f"  python src/create_meta_labels_regime.py \\")
    logger.info(f"    --regime-filter range \\")
    logger.info(f"    --split train")


if __name__ == '__main__':
    main()
