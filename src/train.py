"""
Script d'entraînement du modèle CNN-LSTM Multi-Output.

Pipeline complet:
    1. Charger les données (BTC + ETH)
    2. Préparer les datasets (indicateurs + labels)
    3. Créer DataLoaders PyTorch
    4. Entraîner le modèle avec early stopping
    5. Sauvegarder le meilleur modèle
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
import json
from typing import Dict, Tuple
import sys
import argparse

logger = logging.getLogger(__name__)

# Import modules locaux
from constants import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    RANDOM_SEED,
    BEST_MODEL_PATH,
    MODELS_DIR,
    CHECKPOINTS_DIR
)
from indicators import prepare_datasets
from model import create_model, compute_metrics
from prepare_data import load_prepared_data
from data_utils import normalize_labels_for_single_output
from utils import log_dataset_metadata
from datetime import datetime


class IndicatorDataset(Dataset):
    """
    Dataset PyTorch pour les séquences d'indicateurs.

    Args:
        X: Features (n_sequences, sequence_length, n_indicators)
        Y: Labels (n_sequences, n_outputs)
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


class EarlyStopping:
    """
    Early stopping pour arrêter l'entraînement si validation loss ne s'améliore pas.

    Args:
        patience: Nombre d'époques sans amélioration avant d'arrêter
        min_delta: Amélioration minimale pour considérer comme amélioration
        mode: 'min' pour loss (lower is better), 'max' pour accuracy (higher is better)
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Vérifier si on doit arrêter.

        Args:
            score: Métrique à surveiller (loss ou accuracy)

        Returns:
            True si on doit arrêter, False sinon
        """
        if self.best_score is None:
            self.best_score = score
            return False

        # Mode 'min': lower is better (loss)
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        # Mode 'max': higher is better (accuracy)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    indicator_names: list = None
) -> Dict[str, float]:
    """
    Entraîne le modèle sur une époque.

    Args:
        model: Modèle
        dataloader: DataLoader
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device
        indicator_names: Noms des outputs (ex: ['Direction', 'Force'] pour dual-binary)

    Returns:
        Dictionnaire avec loss et métriques
    """
    model.train()

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    for X_batch, Y_batch in dataloader:
        # Déplacer sur device
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(X_batch)

        # Loss
        loss = loss_fn(outputs, Y_batch)

        # Backward
        loss.backward()
        optimizer.step()

        # Accumuler
        total_loss += loss.item() * X_batch.size(0)
        all_predictions.append(outputs.detach().cpu())
        all_targets.append(Y_batch.detach().cpu())

    # Moyennes
    avg_loss = total_loss / len(dataloader.dataset)

    # Métriques
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_predictions, all_targets, indicator_names=indicator_names)
    metrics['loss'] = avg_loss

    return metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    indicator_names: list = None
) -> Dict[str, float]:
    """
    Valide le modèle sur une époque.

    Args:
        model: Modèle
        dataloader: DataLoader
        loss_fn: Loss function
        device: Device
        indicator_names: Noms des outputs (ex: ['Direction', 'Force'] pour dual-binary)

    Returns:
        Dictionnaire avec loss et métriques
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
    metrics = compute_metrics(all_predictions, all_targets, indicator_names=indicator_names)
    metrics['loss'] = avg_loss

    return metrics


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    num_epochs: int = NUM_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    save_path: str = BEST_MODEL_PATH,
    model_config: Dict = None,
    indicator_names: list = None
) -> Dict:
    """
    Boucle d'entraînement complète avec early stopping.

    Args:
        train_loader: DataLoader train
        val_loader: DataLoader validation
        model: Modèle
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device
        num_epochs: Nombre max d'époques
        patience: Patience pour early stopping
        save_path: Chemin pour sauvegarder le meilleur modèle
        model_config: Configuration du modèle
        indicator_names: Noms des outputs (ex: ['Direction', 'Force'] pour dual-binary)

    Returns:
        Historique de l'entraînement
    """
    logger.info("="*80)
    logger.info("DÉBUT DE L'ENTRAÎNEMENT")
    logger.info("="*80)

    # Early stopping
    early_stopping = EarlyStopping(patience=patience, mode='min')

    # Historique
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'best_epoch': 0,
        'best_val_loss': float('inf')
    }

    # Boucle d'entraînement
    for epoch in range(num_epochs):
        logger.info(f"\nÉpoque {epoch+1}/{num_epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device, indicator_names)

        # Validation
        val_metrics = validate_epoch(model, val_loader, loss_fn, device, indicator_names)

        # Sauvegarder historique
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['avg_accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['avg_accuracy'])

        # Logger
        if indicator_names and len(indicator_names) == 2:
            # Dual-binary: afficher Direction et Force séparément
            logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Avg Acc: {train_metrics['avg_accuracy']:.3f}, "
                       f"Avg F1: {train_metrics['avg_f1']:.3f}")
            logger.info(f"          Direction: Acc={train_metrics['Direction_accuracy']:.3f}, "
                       f"F1={train_metrics['Direction_f1']:.3f}")
            logger.info(f"          Force:     Acc={train_metrics['Force_accuracy']:.3f}, "
                       f"F1={train_metrics['Force_f1']:.3f}")

            logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                       f"Avg Acc: {val_metrics['avg_accuracy']:.3f}, "
                       f"Avg F1: {val_metrics['avg_f1']:.3f}")
            logger.info(f"          Direction: Acc={val_metrics['Direction_accuracy']:.3f}, "
                       f"F1={val_metrics['Direction_f1']:.3f}")
            logger.info(f"          Force:     Acc={val_metrics['Force_accuracy']:.3f}, "
                       f"F1={val_metrics['Force_f1']:.3f}")
        else:
            # Affichage standard
            logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Acc: {train_metrics['avg_accuracy']:.3f}, "
                       f"F1: {train_metrics['avg_f1']:.3f}")
            logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                       f"Acc: {val_metrics['avg_accuracy']:.3f}, "
                       f"F1: {val_metrics['avg_f1']:.3f}")

        # Sauvegarder meilleur modèle
        if val_metrics['loss'] < history['best_val_loss']:
            history['best_val_loss'] = val_metrics['loss']
            history['best_epoch'] = epoch + 1

            # Créer dossier si nécessaire
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            # Sauvegarder
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['avg_accuracy'],
                'model_config': model_config,
            }, save_path)

            logger.info(f"  ✅ Meilleur modèle sauvegardé (val_loss: {val_metrics['loss']:.4f})")

        # Early stopping
        if early_stopping(val_metrics['loss']):
            logger.info(f"\n⏹️ Early stopping à l'époque {epoch+1}")
            break

    logger.info("="*80)
    logger.info("FIN DE L'ENTRAÎNEMENT")
    logger.info("="*80)
    logger.info(f"Meilleur modèle: Époque {history['best_epoch']}, "
               f"Val Loss: {history['best_val_loss']:.4f}")

    return history


def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description='Entraînement du modèle CNN-LSTM Multi-Output',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Hyperparamètres d'entraînement
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Taille des batches')
    parser.add_argument('--lr', '--learning-rate', type=float, default=LEARNING_RATE,
                        dest='learning_rate', help='Learning rate')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Nombre maximum d\'époques')
    parser.add_argument('--patience', type=int, default=EARLY_STOPPING_PATIENCE,
                        help='Patience pour early stopping')

    # Hyperparamètres du modèle
    parser.add_argument('--cnn-filters', type=int, default=64,
                        help='Nombre de filtres CNN')
    parser.add_argument('--lstm-hidden', type=int, default=64,
                        help='Taille hidden LSTM')
    parser.add_argument('--lstm-layers', type=int, default=2,
                        help='Nombre de couches LSTM')
    parser.add_argument('--lstm-dropout', type=float, default=0.2,
                        help='Dropout LSTM (entre couches)')
    parser.add_argument('--dense-hidden', type=int, default=32,
                        help='Taille couche dense partagée')
    parser.add_argument('--dense-dropout', type=float, default=0.3,
                        help='Dropout après dense')

    # Chemins
    parser.add_argument('--save-path', type=str, default=BEST_MODEL_PATH,
                        help='Chemin pour sauvegarder le meilleur modèle')

    # Données préparées
    parser.add_argument('--data', '-d', type=str, default=None,
                        help='Chemin vers les données préparées (.npz). Si non spécifié, prépare les données à la volée.')

    # Note: --filter supprimé car --data est maintenant requis
    # Le filtre est défini lors de la préparation des données avec prepare_data_30min.py

    # Indicateur spécifique (optionnel)
    parser.add_argument('--indicator', '-i', type=str, default='all',
                        choices=['all', 'rsi', 'cci', 'macd', 'close', 'macd40', 'macd26', 'macd13'],
                        help='Indicateur à entraîner (all=multi-output, autres=single-output)')

    # Nom du filtre (pour le nom du modèle)
    parser.add_argument('--filter', '-f', type=str, default=None,
                        help='Nom du filtre utilisé (ex: octave20, kalman). Inclus dans le nom du modèle.')

    # Autres
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help='Random seed pour reproductibilité')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device à utiliser (auto détecte automatiquement)')

    return parser.parse_args()


def generate_predictions(model: nn.Module, X: np.ndarray, device: str, batch_size: int = 512) -> np.ndarray:
    """
    Génère les prédictions du modèle sur un dataset.

    Args:
        model: Modèle entraîné
        X: Features (n_samples, seq_length, n_features)
        device: Device
        batch_size: Taille des batches

    Returns:
        Probabilités continues [0,1] (n_samples, n_outputs)
        NOTE: Les probabilités sont sauvegardées brutes, pas binarisées.
              La binarisation (seuil 0.5) se fait dans la state machine.
    """
    model.eval()
    dataset = IndicatorDataset(X, np.zeros((len(X), 1)))  # Y factice
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            # Le modèle applique déjà sigmoid dans forward(), outputs sont des probabilités [0,1]
            # IMPORTANT: Sauvegarder les probabilités brutes, pas binarisées!
            all_preds.append(outputs.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def save_predictions_to_npz(
    npz_path: str,
    model: nn.Module,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    device: str,
    model_path: str
):
    """
    Génère les prédictions et met à jour le fichier .npz avec Y_train_pred, Y_val_pred, Y_test_pred.

    Args:
        npz_path: Chemin vers le fichier .npz original
        model: Modèle entraîné
        X_train, X_val, X_test: Features
        device: Device
        model_path: Chemin du modèle sauvegardé (pour metadata)
    """
    logger.info("\n📊 Génération des prédictions...")

    # Générer prédictions
    Y_train_pred = generate_predictions(model, X_train, device)
    Y_val_pred = generate_predictions(model, X_val, device)
    Y_test_pred = generate_predictions(model, X_test, device)

    logger.info(f"  Train: {Y_train_pred.shape}, mean={Y_train_pred.mean():.3f}")
    logger.info(f"  Val:   {Y_val_pred.shape}, mean={Y_val_pred.mean():.3f}")
    logger.info(f"  Test:  {Y_test_pred.shape}, mean={Y_test_pred.mean():.3f}")

    # Charger le fichier .npz existant
    logger.info(f"\n💾 Mise à jour du fichier: {npz_path}")
    existing_data = dict(np.load(npz_path, allow_pickle=True))

    # Mettre à jour metadata
    if 'metadata' in existing_data:
        metadata = json.loads(str(existing_data['metadata']))
    else:
        metadata = {}

    metadata['predictions_added_at'] = datetime.now().isoformat()
    metadata['predictions_model'] = str(model_path)
    metadata['predictions_train_mean'] = float(Y_train_pred.mean())
    metadata['predictions_val_mean'] = float(Y_val_pred.mean())
    metadata['predictions_test_mean'] = float(Y_test_pred.mean())

    # Ajouter les prédictions
    existing_data['Y_train_pred'] = Y_train_pred
    existing_data['Y_val_pred'] = Y_val_pred
    existing_data['Y_test_pred'] = Y_test_pred
    existing_data['metadata'] = json.dumps(metadata)

    # Sauvegarder
    np.savez_compressed(npz_path, **existing_data)
    logger.info(f"  ✅ Prédictions sauvegardées dans {npz_path}")
    logger.info(f"     Nouvelles clés: Y_train_pred, Y_val_pred, Y_test_pred")


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


def validate_args_vs_filename(args) -> None:
    """
    Vérifie la cohérence entre les paramètres --filter et --indicator et le nom du fichier de données.

    Args:
        args: Arguments parsés

    Raises:
        SystemExit: Si incohérence détectée
    """
    if not args.data:
        return  # Pas de fichier, pas de validation

    filename = Path(args.data).stem.lower()  # ex: dataset_btc_eth_bnb_ada_ltc_ohlcv2_rsi_kalman

    # Vérifier le filtre
    if args.filter:
        filter_name = args.filter.lower()
        if filter_name not in filename:
            logger.error(f"❌ Incohérence détectée!")
            logger.error(f"   --filter '{args.filter}' ne correspond pas au fichier")
            logger.error(f"   Fichier: {Path(args.data).name}")
            logger.error(f"   Le filtre '{filter_name}' n'est pas présent dans le nom du fichier")
            raise SystemExit(1)

    # Vérifier l'indicateur (sauf 'all')
    if args.indicator != 'all':
        indicator_name = args.indicator.lower()
        if indicator_name not in filename:
            logger.error(f"❌ Incohérence détectée!")
            logger.error(f"   --indicator '{args.indicator}' ne correspond pas au fichier")
            logger.error(f"   Fichier: {Path(args.data).name}")
            logger.error(f"   L'indicateur '{indicator_name}' n'est pas présent dans le nom du fichier")
            raise SystemExit(1)

    logger.info(f"✅ Paramètres cohérents avec le fichier de données")


def main():
    """Pipeline complet d'entraînement."""

    # Parser arguments
    args = parse_args()

    # Configurer logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    logger.info("="*80)
    logger.info("PIPELINE D'ENTRAÎNEMENT CNN-LSTM")
    logger.info("="*80)

    # Valider la cohérence des arguments avec le fichier de données
    validate_args_vs_filename(args)

    # Seed pour reproductibilité
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    logger.info(f"\nDevice: {device}")

    # Afficher hyperparamètres
    # Déterminer mode (multi-output ou single-output)
    single_indicator = args.indicator != 'all'
    if single_indicator:
        indicator_idx = INDICATOR_INDEX[args.indicator]  # None pour close, macd40, etc.
        indicator_name = INDICATOR_NAMES[args.indicator]
        num_outputs = 1
        logger.info(f"\n🎯 Mode SINGLE-OUTPUT: {indicator_name}")
    else:
        indicator_idx = None
        indicator_name = None
        num_outputs = 3
        logger.info(f"\n🎯 Mode MULTI-OUTPUT: RSI, CCI, MACD")

    logger.info(f"\n⚙️ Hyperparamètres d'entraînement:")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Max epochs: {args.epochs}")
    logger.info(f"  Early stopping patience: {args.patience}")
    logger.info(f"  Random seed: {args.seed}")

    logger.info(f"\n🏗️ Architecture du modèle:")
    logger.info(f"  CNN filters: {args.cnn_filters}")
    logger.info(f"  LSTM hidden: {args.lstm_hidden}")
    logger.info(f"  LSTM layers: {args.lstm_layers}")
    logger.info(f"  LSTM dropout: {args.lstm_dropout}")
    logger.info(f"  Dense hidden: {args.dense_hidden}")
    logger.info(f"  Dense dropout: {args.dense_dropout}")

    # =========================================================================
    # 1. CHARGER LES DONNÉES
    # =========================================================================
    if args.data:
        # Charger données préparées (rapide)
        logger.info(f"\n1. Chargement des données préparées: {args.data}")
        prepared = load_prepared_data(args.data)
        X_train, Y_train = prepared['train']
        X_val, Y_val = prepared['val']
        X_test, Y_test = prepared['test']
        metadata = prepared['metadata']
        log_dataset_metadata(metadata, logger)
    else:
        # Données préparées requises (ancienne méthode avait du data leakage)
        logger.error("❌ Argument --data requis!")
        logger.error("")
        logger.error("Préparez d'abord les données avec:")
        logger.error("  python src/prepare_data_purified_dual_binary.py --assets BTC ETH BNB ADA LTC")
        logger.error("")
        logger.error("Puis entraînez avec:")
        logger.error("  python src/train.py --data data/prepared/dataset_..._rsi_dual_binary_kalman.npz")
        raise SystemExit(1)

    # =========================================================================
    # AUTO-DÉTECTION ARCHITECTURE (Pure Signal)
    # =========================================================================
    # Détecter n_features et n_outputs depuis les données
    n_features_detected = X_train.shape[2]  # 1 pour RSI/MACD, 3 pour CCI
    n_outputs_detected = Y_train.shape[1]   # 2 pour dual-binary (direction + force)

    # Détecter si dual-binary depuis metadata
    is_dual_binary = False
    indicator_for_metrics = None

    if metadata:
        # Détection dual-binary
        if 'label_names' in metadata and len(metadata['label_names']) == 2:
            is_dual_binary = True
            # Extraire nom indicateur (ex: ['rsi_dir', 'rsi_force'] -> 'rsi')
            label_name = metadata['label_names'][0]
            indicator_for_metrics = label_name.split('_')[0].upper()

        # Log architecture détectée
        logger.info(f"\n🔍 Architecture détectée:")
        logger.info(f"  Features: {n_features_detected}")
        logger.info(f"  Outputs: {n_outputs_detected}")

        if is_dual_binary:
            logger.info(f"  Type: DUAL-BINARY ({indicator_for_metrics})")
            logger.info(f"  Labels: Direction + Force")
        else:
            logger.info(f"  Type: SINGLE-OUTPUT")

    # Filtrer les labels si mode single-output (ancien pipeline)
    if single_indicator and not is_dual_binary:
        # Ancien pipeline (3 outputs -> 1)
        Y_train = normalize_labels_for_single_output(Y_train, indicator_idx, indicator_name)
        Y_val = normalize_labels_for_single_output(Y_val, indicator_idx, indicator_name)
        Y_test = normalize_labels_for_single_output(Y_test, indicator_idx, indicator_name)

    logger.info(f"\n📊 Datasets:")
    logger.info(f"  Train: X={X_train.shape}, Y={Y_train.shape}")
    logger.info(f"  Val:   X={X_val.shape}, Y={Y_val.shape}")
    logger.info(f"  Test:  X={X_test.shape}, Y={Y_test.shape}")

    # =========================================================================
    # 2. CRÉER DATALOADERS
    # =========================================================================
    logger.info("\n2. Création des DataLoaders...")

    train_dataset = IndicatorDataset(X_train, Y_train)
    val_dataset = IndicatorDataset(X_val, Y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,  # Chargement parallèle des données
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if device == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if device == 'cuda' else False
    )

    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")

    # =========================================================================
    # 3. CRÉER MODÈLE (Architecture Auto-Adaptative)
    # =========================================================================
    logger.info("\n3. Création du modèle...")

    # Utiliser valeurs détectées au lieu de num_outputs manuel
    num_outputs_final = n_outputs_detected

    logger.info(f"  num_features={n_features_detected}, num_outputs={num_outputs_final}")

    model, loss_fn = create_model(
        device=device,
        num_indicators=n_features_detected,
        num_outputs=num_outputs_final,
        cnn_filters=args.cnn_filters,
        lstm_hidden_size=args.lstm_hidden,
        lstm_num_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        dense_hidden_size=args.dense_hidden,
        dense_dropout=args.dense_dropout
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # =========================================================================
    # 4. ENTRAÎNEMENT
    # =========================================================================
    logger.info(f"\n4. Entraînement ({args.epochs} époques max)...")

    # Préparer noms indicateurs pour métriques
    if is_dual_binary:
        # Dual-binary: ['Direction', 'Force']
        indicator_names_for_metrics = ['Direction', 'Force']
    elif single_indicator:
        # Single-output ancien pipeline
        indicator_names_for_metrics = [indicator_name] if indicator_name else None
    else:
        # Multi-output ancien pipeline
        indicator_names_for_metrics = None  # Défaut: RSI, CCI, MACD

    # Config du modèle pour sauvegarde
    model_config = {
        'cnn_filters': args.cnn_filters,
        'lstm_hidden_size': args.lstm_hidden,
        'lstm_num_layers': args.lstm_layers,
        'lstm_dropout': args.lstm_dropout,
        'dense_hidden_size': args.dense_hidden,
        'dense_dropout': args.dense_dropout,
        'num_outputs': num_outputs_final,
        'num_features': n_features_detected,
        'indicator': args.indicator,
        'is_dual_binary': is_dual_binary,
        'indicator_for_metrics': indicator_for_metrics,
    }

    # Chemin de sauvegarde (inclut le préfixe dataset + filtre + indicateur)
    # Extraire le préfixe du dataset (ex: "ohlcv2" de "dataset_..._ohlcv2_cci_octave20.npz")
    dataset_prefix = ""
    if args.data:
        data_name = Path(args.data).stem  # dataset_btc_eth_bnb_ada_ltc_ohlcv2_cci_octave20
        # Chercher des préfixes connus dans le nom
        known_prefixes = ['ohlcv2', 'ohlc', '5min_30min', '5min', '30min']
        for prefix in known_prefixes:
            if prefix in data_name:
                dataset_prefix = prefix
                break

    # Construire le suffixe du nom de fichier
    suffix_parts = []
    if dataset_prefix:
        suffix_parts.append(dataset_prefix)
    if args.filter:
        suffix_parts.append(args.filter)
    if single_indicator:
        suffix_parts.append(args.indicator)

    if suffix_parts:
        suffix = '_'.join(suffix_parts)
        save_path = args.save_path.replace('.pth', f'_{suffix}.pth')
    else:
        save_path = args.save_path

    logger.info(f"  Modèle sera sauvegardé: {save_path}")

    history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        patience=args.patience,
        save_path=save_path,
        model_config=model_config,
        indicator_names=indicator_names_for_metrics
    )

    # =========================================================================
    # 5. SAUVEGARDER HISTORIQUE
    # =========================================================================
    logger.info("\n5. Sauvegarde de l'historique...")

    history_path = Path(MODELS_DIR) / 'training_history.json'
    history_path.parent.mkdir(parents=True, exist_ok=True)

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"  Historique sauvegardé: {history_path}")

    # =========================================================================
    # 6. GÉNÉRER ET SAUVEGARDER LES PRÉDICTIONS
    # =========================================================================
    if args.data:
        logger.info("\n6. Génération des prédictions...")

        # Charger le meilleur modèle
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"  Meilleur modèle chargé: {save_path}")

        # Sauvegarder les prédictions dans le .npz
        save_predictions_to_npz(
            npz_path=args.data,
            model=model,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            device=device,
            model_path=save_path
        )

    # =========================================================================
    # RÉSUMÉ FINAL
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("✅ ENTRAÎNEMENT TERMINÉ")
    logger.info("="*80)
    logger.info(f"\nMeilleur modèle:")
    logger.info(f"  Époque: {history['best_epoch']}")
    logger.info(f"  Val Loss: {history['best_val_loss']:.4f}")
    logger.info(f"  Sauvegardé: {save_path}")

    if is_dual_binary:
        logger.info(f"  Type: DUAL-BINARY ({indicator_for_metrics})")
        logger.info(f"  Features: {n_features_detected}")
        logger.info(f"  Outputs: Direction + Force")
    elif single_indicator:
        logger.info(f"  Indicateur: {indicator_name}")
    else:
        logger.info(f"  Type: MULTI-OUTPUT (RSI, CCI, MACD)")

    if args.data:
        logger.info(f"\n📊 Prédictions sauvegardées dans: {args.data}")
        logger.info(f"   Nouvelles clés: Y_train_pred, Y_val_pred, Y_test_pred")

    logger.info(f"\nProchaines étapes:")
    if is_dual_binary:
        logger.info(f"  - Évaluer: python src/evaluate.py --data {args.data}")
    elif single_indicator:
        logger.info(f"  - Évaluer: python src/evaluate.py --data <dataset> --indicator {args.indicator}")
    else:
        logger.info(f"  - Évaluer sur test set: python src/evaluate.py --data <dataset>")
    logger.info(f"  - Visualiser historique: voir {history_path}")


if __name__ == '__main__':
    main()
