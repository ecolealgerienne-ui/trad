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
from data_utils import load_and_split_btc_eth
from indicators import prepare_datasets
from model import create_model, compute_metrics


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
    device: str
) -> Dict[str, float]:
    """
    Entraîne le modèle sur une époque.

    Args:
        model: Modèle
        dataloader: DataLoader
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device

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
    metrics = compute_metrics(all_predictions, all_targets)
    metrics['loss'] = avg_loss

    return metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str
) -> Dict[str, float]:
    """
    Valide le modèle sur une époque.

    Args:
        model: Modèle
        dataloader: DataLoader
        loss_fn: Loss function
        device: Device

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
    metrics = compute_metrics(all_predictions, all_targets)
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
    save_path: str = BEST_MODEL_PATH
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
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device)

        # Validation
        val_metrics = validate_epoch(model, val_loader, loss_fn, device)

        # Sauvegarder historique
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['avg_accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['avg_accuracy'])

        # Logger
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

    # Chemins
    parser.add_argument('--save-path', type=str, default=BEST_MODEL_PATH,
                        help='Chemin pour sauvegarder le meilleur modèle')

    # Autres
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help='Random seed pour reproductibilité')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device à utiliser (auto détecte automatiquement)')

    return parser.parse_args()


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
    logger.info(f"\n⚙️ Hyperparamètres:")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Max epochs: {args.epochs}")
    logger.info(f"  Early stopping patience: {args.patience}")
    logger.info(f"  Random seed: {args.seed}")

    # =========================================================================
    # 1. CHARGER LES DONNÉES
    # =========================================================================
    logger.info("\n1. Chargement des données BTC + ETH...")
    train_df, val_df, test_df = load_and_split_btc_eth()

    # =========================================================================
    # 2. PRÉPARER LES DATASETS
    # =========================================================================
    logger.info("\n2. Préparation des datasets (indicateurs + labels)...")
    datasets = prepare_datasets(train_df, val_df, test_df)

    X_train, Y_train = datasets['train']
    X_val, Y_val = datasets['val']
    X_test, Y_test = datasets['test']

    logger.info(f"\n📊 Datasets:")
    logger.info(f"  Train: X={X_train.shape}, Y={Y_train.shape}")
    logger.info(f"  Val:   X={X_val.shape}, Y={Y_val.shape}")
    logger.info(f"  Test:  X={X_test.shape}, Y={Y_test.shape}")

    # =========================================================================
    # 3. CRÉER DATALOADERS
    # =========================================================================
    logger.info("\n3. Création des DataLoaders...")

    train_dataset = IndicatorDataset(X_train, Y_train)
    val_dataset = IndicatorDataset(X_val, Y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 0 pour éviter problèmes multiprocessing
        pin_memory=True if device == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )

    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")

    # =========================================================================
    # 4. CRÉER MODÈLE
    # =========================================================================
    logger.info("\n4. Création du modèle...")
    model, loss_fn = create_model(device=device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # =========================================================================
    # 5. ENTRAÎNEMENT
    # =========================================================================
    logger.info(f"\n5. Entraînement ({args.epochs} époques max)...")

    history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        patience=args.patience,
        save_path=args.save_path
    )

    # =========================================================================
    # 6. SAUVEGARDER HISTORIQUE
    # =========================================================================
    logger.info("\n6. Sauvegarde de l'historique...")

    history_path = Path(MODELS_DIR) / 'training_history.json'
    history_path.parent.mkdir(parents=True, exist_ok=True)

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"  Historique sauvegardé: {history_path}")

    # =========================================================================
    # RÉSUMÉ FINAL
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("✅ ENTRAÎNEMENT TERMINÉ")
    logger.info("="*80)
    logger.info(f"\nMeilleur modèle:")
    logger.info(f"  Époque: {history['best_epoch']}")
    logger.info(f"  Val Loss: {history['best_val_loss']:.4f}")
    logger.info(f"  Sauvegardé: {args.save_path}")

    logger.info(f"\nProchaines étapes:")
    logger.info(f"  - Évaluer sur test set: python src/evaluate.py")
    logger.info(f"  - Visualiser historique: voir {history_path}")


if __name__ == '__main__':
    main()
