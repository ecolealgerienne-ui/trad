#!/usr/bin/env python3
"""
Génération des Méta-Features pour Stacking / Ensemble Learning

Objectif: Combiner les prédictions de 3 modèles experts (MACD, RSI, CCI)
pour améliorer la prédiction de Direction (Kalman original).

Pipeline:
  1. Charger les 3 modèles entraînés (.pth)
  2. Charger les 3 datasets correspondants
  3. Générer les prédictions (probabilités) pour Train/Val/Test
  4. Sauvegarder les méta-features

Output:
  X_meta: (n, 6) - [p_macd_dir, p_macd_force, p_rsi_dir, p_rsi_force, p_cci_dir, p_cci_force]
  Y_meta: (n, 1) - Direction Kalman original (cible commune)

Usage:
  python src/generate_meta_features.py --assets BTC ETH BNB ADA LTC
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import logging
import argparse
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Imports locaux
from constants import BATCH_SIZE
from model import create_model


def load_model_and_predict(
    model_path: str,
    X: np.ndarray,
    device: str,
    batch_size: int = 256
) -> np.ndarray:
    """
    Charge un modèle et génère les prédictions (probabilités).

    Args:
        model_path: Chemin vers le modèle .pth
        X: Features (n, seq_len, n_features)
        device: Device ('cuda' ou 'cpu')
        batch_size: Taille des batchs

    Returns:
        Prédictions (n, 2) - [proba_direction, proba_force]
    """
    # Charger checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get('model_config', {})

    # Créer modèle
    n_features = X.shape[2]

    # Retirer n_features et num_outputs de model_config s'ils existent
    # (on les passe explicitement)
    model_config_clean = {k: v for k, v in model_config.items()
                          if k not in ['n_features', 'num_outputs']}

    model = create_model(
        n_features=n_features,
        num_outputs=2,  # Direction + Force
        **model_config_clean
    ).to(device)

    # Charger poids
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"   Modèle chargé: {Path(model_path).name}")
    logger.info(f"   Features: {n_features}, Outputs: 2")

    # Créer DataLoader
    X_tensor = torch.FloatTensor(X)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Prédictions
    all_predictions = []
    with torch.no_grad():
        for (X_batch,) in dataloader:
            X_batch = X_batch.to(device)
            # predict_proba retourne probabilités (gère sigmoid automatiquement)
            probs = model.predict_proba(X_batch)  # (batch, 2)
            all_predictions.append(probs.cpu().numpy())

    predictions = np.vstack(all_predictions)  # (n, 2)
    logger.info(f"   Prédictions générées: {predictions.shape}")

    return predictions


def generate_meta_features_for_split(
    split_name: str,
    models_paths: Dict[str, str],
    datasets_paths: Dict[str, str],
    device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Génère les méta-features pour un split (train/val/test).

    Args:
        split_name: 'train', 'val', ou 'test'
        models_paths: {'macd': path, 'rsi': path, 'cci': path}
        datasets_paths: {'macd': path, 'rsi': path, 'cci': path}
        device: Device

    Returns:
        X_meta: (n, 6) - Prédictions des 3 modèles
        Y_meta: (n, 1) - Direction Kalman (cible commune)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Génération Méta-Features: {split_name.upper()}")
    logger.info('='*80)

    predictions = {}
    Y_meta = None

    for indicator in ['macd', 'rsi', 'cci']:
        logger.info(f"\n{indicator.upper()}:")

        # Charger dataset
        data = np.load(datasets_paths[indicator], allow_pickle=True)
        X = data[f'X_{split_name}']
        Y = data[f'Y_{split_name}']

        logger.info(f"   Dataset: {Path(datasets_paths[indicator]).name}")
        logger.info(f"   X shape: {X.shape}, Y shape: {Y.shape}")

        # Générer prédictions
        preds = load_model_and_predict(
            models_paths[indicator],
            X,
            device
        )
        predictions[indicator] = preds  # (n, 2)

        # Sauvegarder Y_meta (Direction, colonne 0)
        if Y_meta is None:
            Y_meta = Y[:, 0:1]  # (n, 1) - Direction uniquement

    # Vérifier cohérence tailles
    n_samples = Y_meta.shape[0]
    for indicator, preds in predictions.items():
        if preds.shape[0] != n_samples:
            raise ValueError(
                f"{indicator} predictions shape mismatch: "
                f"{preds.shape[0]} vs {n_samples} expected"
            )

    # Concaténer prédictions (6 colonnes)
    X_meta = np.concatenate([
        predictions['macd'],  # (n, 2) - [dir, force]
        predictions['rsi'],   # (n, 2)
        predictions['cci'],   # (n, 2)
    ], axis=1)  # (n, 6)

    logger.info(f"\n✅ Méta-Features générées:")
    logger.info(f"   X_meta shape: {X_meta.shape}")
    logger.info(f"   Y_meta shape: {Y_meta.shape}")
    logger.info(f"   Distribution Direction:")
    logger.info(f"     UP (1):   {np.sum(Y_meta == 1)} ({np.mean(Y_meta)*100:.1f}%)")
    logger.info(f"     DOWN (0): {np.sum(Y_meta == 0)} ({(1-np.mean(Y_meta))*100:.1f}%)")

    return X_meta, Y_meta


def main():
    parser = argparse.ArgumentParser(
        description="Génère les méta-features pour Stacking"
    )
    parser.add_argument(
        '--assets',
        nargs='+',
        default=['BTC', 'ETH', 'BNB', 'ADA', 'LTC'],
        help='Assets utilisés (pour nom de fichier)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/meta',
        help='Répertoire de sortie'
    )
    parser.add_argument(
        '--device',
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='Device à utiliser'
    )

    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    logger.info("="*80)
    logger.info("GÉNÉRATION MÉTA-FEATURES POUR STACKING")
    logger.info("="*80)
    logger.info(f"\nDevice: {device}")
    logger.info(f"Assets: {', '.join(args.assets)}")

    # Construire noms de fichiers
    assets_str = '_'.join(args.assets).lower()

    # Chemins des modèles
    models_paths = {
        'macd': f'models/best_model_macd_kalman_dual_binary.pth',
        'rsi': f'models/best_model_rsi_kalman_dual_binary.pth',
        'cci': f'models/best_model_cci_kalman_dual_binary.pth',
    }

    # Chemins des datasets
    datasets_paths = {
        'macd': f'data/prepared/dataset_{assets_str}_macd_dual_binary_kalman.npz',
        'rsi': f'data/prepared/dataset_{assets_str}_rsi_dual_binary_kalman.npz',
        'cci': f'data/prepared/dataset_{assets_str}_cci_dual_binary_kalman.npz',
    }

    # Vérifier existence
    logger.info("\n📁 Vérification fichiers...")
    for indicator, path in models_paths.items():
        if not Path(path).exists():
            logger.error(f"❌ Modèle manquant: {path}")
            logger.error(f"   Entraînez d'abord le modèle {indicator.upper()}:")
            logger.error(f"   python src/train.py --data {datasets_paths[indicator]} --epochs 50")
            return 1

    for indicator, path in datasets_paths.items():
        if not Path(path).exists():
            logger.error(f"❌ Dataset manquant: {path}")
            logger.error(f"   Générez d'abord les datasets:")
            logger.error(f"   python src/prepare_data_purified_dual_binary.py --assets {' '.join(args.assets)}")
            return 1

    logger.info("✅ Tous les fichiers requis sont présents")

    # Générer méta-features pour chaque split
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ['train', 'val', 'test']:
        X_meta, Y_meta = generate_meta_features_for_split(
            split_name,
            models_paths,
            datasets_paths,
            device
        )

        # Sauvegarder
        output_path = output_dir / f'meta_features_{split_name}.npz'
        np.savez_compressed(
            output_path,
            X_meta=X_meta,
            Y_meta=Y_meta
        )
        logger.info(f"\n💾 Sauvegardé: {output_path}")

    logger.info("\n" + "="*80)
    logger.info("✅ TERMINÉ - Méta-Features générées pour Train/Val/Test")
    logger.info("="*80)
    logger.info("\nProchaine étape:")
    logger.info("  python src/train_stacking.py")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
