"""
Régénère les prédictions (probabilités) pour un dataset existant.

Ce script:
1. Charge un dataset préparé (.npz)
2. Charge le modèle correspondant
3. Génère les prédictions (probabilités [0,1]) pour train/val/test
4. Sauvegarde les prédictions dans le dataset

Usage:
    python src/regenerate_predictions.py --data data/prepared/dataset_..._octave20.npz --indicator macd

    # Traiter plusieurs datasets
    for ind in rsi cci macd; do
        python src/regenerate_predictions.py \
            --data data/prepared/dataset_btc_eth_bnb_ada_ltc_ohlcv2_${ind}_octave20.npz \
            --indicator $ind
    done
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import json
import argparse

logger = logging.getLogger(__name__)

# Import modules locaux
from constants import BEST_MODEL_PATH
from model import create_model
from train import IndicatorDataset
from prepare_data import load_prepared_data
from data_utils import normalize_labels_for_single_output


def generate_predictions(model, X, device, batch_size=512):
    """
    Génère les prédictions du modèle sur un dataset.

    Returns:
        Probabilités continues [0,1] (n_samples, n_outputs)
    """
    model.eval()
    dataset = IndicatorDataset(X, np.zeros((len(X), 1)))  # Y factice
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            # Garder les probabilités brutes [0,1]
            all_preds.append(outputs.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Régénère les prédictions (probabilités) pour un dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--data', '-d', type=str, required=True,
                        help='Chemin vers le dataset (.npz)')

    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Chemin vers le modèle (.pth). Auto-détecté si non spécifié.')

    parser.add_argument('--indicator', '-i', type=str, default='macd',
                        choices=['rsi', 'cci', 'macd', 'close'],
                        help='Indicateur (pour trouver le modèle)')

    parser.add_argument('--filter', '-f', type=str, default='octave20',
                        help='Nom du filtre (octave20, kalman)')

    parser.add_argument('--batch-size', type=int, default=512,
                        help='Taille des batches')

    return parser.parse_args()


def main():
    args = parse_args()

    # Configurer logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    logger.info("="*80)
    logger.info("RÉGÉNÉRATION DES PRÉDICTIONS (PROBABILITÉS)")
    logger.info("="*80)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"\nDevice: {device}")

    # Chemin du modèle
    if args.model:
        model_path = args.model
    else:
        # Auto-détecter depuis le nom du dataset
        suffix = f"ohlcv2_{args.filter}_{args.indicator}"
        model_path = BEST_MODEL_PATH.replace('.pth', f'_{suffix}.pth')

    # Vérifier existence
    if not Path(model_path).exists():
        logger.error(f"❌ Modèle non trouvé: {model_path}")
        return

    if not Path(args.data).exists():
        logger.error(f"❌ Dataset non trouvé: {args.data}")
        return

    logger.info(f"\n📂 Dataset: {args.data}")
    logger.info(f"📦 Modèle: {model_path}")

    # =========================================================================
    # 1. CHARGER LES DONNÉES
    # =========================================================================
    logger.info("\n1. Chargement des données...")
    prepared = load_prepared_data(args.data)

    X_train, Y_train = prepared['train']
    X_val, Y_val = prepared['val']
    X_test, Y_test = prepared['test']
    metadata = prepared['metadata']

    logger.info(f"   Train: {X_train.shape}")
    logger.info(f"   Val: {X_val.shape}")
    logger.info(f"   Test: {X_test.shape}")

    # =========================================================================
    # 2. CHARGER LE MODÈLE
    # =========================================================================
    logger.info("\n2. Chargement du modèle...")

    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get('model_config', {})

    num_features = X_test.shape[2]
    num_outputs = model_config.get('num_outputs', 1)

    model, _ = create_model(
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

    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"   ✅ Modèle chargé (époque {checkpoint['epoch']})")

    # =========================================================================
    # 3. GÉNÉRER LES PRÉDICTIONS
    # =========================================================================
    logger.info("\n3. Génération des prédictions (probabilités)...")

    Y_train_pred = generate_predictions(model, X_train, device, args.batch_size)
    Y_val_pred = generate_predictions(model, X_val, device, args.batch_size)
    Y_test_pred = generate_predictions(model, X_test, device, args.batch_size)

    logger.info(f"   Train: shape={Y_train_pred.shape}, mean={Y_train_pred.mean():.4f}, "
                f"min={Y_train_pred.min():.4f}, max={Y_train_pred.max():.4f}")
    logger.info(f"   Val:   shape={Y_val_pred.shape}, mean={Y_val_pred.mean():.4f}, "
                f"min={Y_val_pred.min():.4f}, max={Y_val_pred.max():.4f}")
    logger.info(f"   Test:  shape={Y_test_pred.shape}, mean={Y_test_pred.mean():.4f}, "
                f"min={Y_test_pred.min():.4f}, max={Y_test_pred.max():.4f}")

    # Vérifier que ce sont bien des probabilités
    if Y_test_pred.min() < 0 or Y_test_pred.max() > 1:
        logger.warning("   ⚠️ Valeurs hors [0,1] détectées!")

    # Distribution
    logger.info(f"\n   Distribution (test):")
    logger.info(f"     < 0.3: {(Y_test_pred < 0.3).mean()*100:.1f}%")
    logger.info(f"     0.3-0.5: {((Y_test_pred >= 0.3) & (Y_test_pred < 0.5)).mean()*100:.1f}%")
    logger.info(f"     0.5-0.7: {((Y_test_pred >= 0.5) & (Y_test_pred < 0.7)).mean()*100:.1f}%")
    logger.info(f"     >= 0.7: {(Y_test_pred >= 0.7).mean()*100:.1f}%")

    # =========================================================================
    # 4. SAUVEGARDER
    # =========================================================================
    logger.info("\n4. Sauvegarde des prédictions...")

    # Charger le dataset existant
    existing = dict(np.load(args.data, allow_pickle=True))

    # Ajouter/remplacer les prédictions
    existing['Y_train_pred'] = Y_train_pred
    existing['Y_val_pred'] = Y_val_pred
    existing['Y_test_pred'] = Y_test_pred

    # Mettre à jour les métadonnées
    if 'metadata' in existing:
        try:
            meta = json.loads(str(existing['metadata']))
            meta['predictions_regenerated'] = True
            meta['predictions_are_probabilities'] = True
            meta['predictions_train_mean'] = float(Y_train_pred.mean())
            meta['predictions_val_mean'] = float(Y_val_pred.mean())
            meta['predictions_test_mean'] = float(Y_test_pred.mean())
            existing['metadata'] = json.dumps(meta)
        except:
            pass

    # Sauvegarder
    np.savez_compressed(args.data, **existing)

    logger.info(f"   ✅ Sauvegardé: {args.data}")

    # =========================================================================
    # RÉSUMÉ
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("RÉSUMÉ")
    logger.info("="*80)
    logger.info(f"\n✅ Prédictions régénérées avec succès!")
    logger.info(f"   - Format: Probabilités continues [0,1]")
    logger.info(f"   - Train: {Y_train_pred.shape[0]:,} samples")
    logger.info(f"   - Val: {Y_val_pred.shape[0]:,} samples")
    logger.info(f"   - Test: {Y_test_pred.shape[0]:,} samples")
    logger.info(f"\n   Moyenne des prédictions:")
    logger.info(f"   - Train: {Y_train_pred.mean():.4f}")
    logger.info(f"   - Val: {Y_val_pred.mean():.4f}")
    logger.info(f"   - Test: {Y_test_pred.mean():.4f}")


if __name__ == '__main__':
    main()
