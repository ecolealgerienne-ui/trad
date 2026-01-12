"""
Script d'entraînement du modèle CNN-LSTM Multi-Output.

Pipeline complet:
    1. Charger les données (BTC + ETH)
    2. Préparer les datasets (indicateurs + labels)
    3. Créer DataLoaders PyTorch
    4. Entraîner le modèle avec early stopping
    5. Sauvegarder le meilleur modèle

═══════════════════════════════════════════════════════════════════════════════
DONNÉES D'ENTRAÎNEMENT - Structure détaillée
═══════════════════════════════════════════════════════════════════════════════

INPUT: X_train (Dataset Régime)
──────────────────────────────
Shape originale: (n_train, 25, 25)
  - n_train: Nombre d'échantillons train
  - 25: Longueur séquence (25 timesteps × 5min = 2h05 de contexte)
  - 25: Nombre de features (2 metadata + 3 pure signal + 20 regime features)

Colonnes X_train[:, :, i]:
  Index 0-1: METADATA
    [0] timestamp    - Unix timestamp (int64)
    [1] asset_id     - ID asset 0-4 (BTC=0, ETH=1, BNB=2, ADA=3, LTC=4)

  Index 2-4: PURE SIGNAL FEATURES (3) - Pour modèle direction
    [2]  h_ret - Rendement High (High - Close_prev) / Close_prev
    [3]  l_ret - Rendement Low  (Low - Close_prev) / Close_prev
    [4]  c_ret - Rendement Close (Close - Close_prev) / Close_prev ⭐ UTILISÉ

  Index 5-11: TREND FEATURES (7) - Pour classification régime
    [5]  ma20_slope          - Pente MA20 normalisée
    [6]  ma50_slope          - Pente MA50 normalisée
    [7]  regression_slope    - Pente régression linéaire
    [8]  regression_r2       - R² régression (qualité tendance)
    [9]  adx                 - Average Directional Index
    [10] macd_histogram_norm - Histogram MACD normalisé
    [11] hurst_exponent      - Exposant de Hurst (persistance tendance)

  Index 12-20: VOLATILITY FEATURES (9)
    [12] atr_normalized         - ATR normalisé par prix
    [13] bb_upper               - Bande de Bollinger supérieure
    [14] bb_middle              - Bande de Bollinger moyenne (SMA20)
    [15] bb_lower               - Bande de Bollinger inférieure
    [16] bb_width               - Largeur bandes Bollinger
    [17] percent_b              - Position prix dans bandes (0-1)
    [18] realized_volatility    - Volatilité réalisée (std returns)
    [19] volatility_compression - Ratio volatilité courte/longue
    [20] range_atr_ratio        - Ratio (High-Low)/ATR

  Index 21-24: VOLUME & MICROSTRUCTURE FEATURES (4)
    [21] volume_ratio     - Volume / MA20 volume
    [22] volume_spike     - Détection spike volume (bool)
    [23] vwap_deviation   - Écart prix vs VWAP
    [24] obv_derivative   - Dérivée On-Balance Volume

EXTRACTION POUR DIRECTION (--indicator macd/rsi/cci):
─────────────────────────────────────────────────────
Quand on entraîne pour direction depuis dataset régime:

  MACD et RSI (Close-based):
    → X_train[:, :, [0, 1, 4]] extrait [timestamp, asset_id, c_ret]
    → Shape finale: (n_train, 25, 3)
    → Modèle input: X[:, :, 2:] = c_ret uniquement (1 feature)

  CCI (Typical Price-based = (H+L+C)/3):
    → X_train[:, :, [0, 1, 2, 3, 4]] extrait [timestamp, asset_id, h_ret, l_ret, c_ret]
    → Shape finale: (n_train, 25, 5)
    → Modèle input: X[:, :, 2:] = h_ret, l_ret, c_ret (3 features)

  → C'est l'architecture "Pure Signal" qui donne 92% accuracy

Source: regime_features.py - calculate_all_regime_features()
Référence: regime_features.py::get_regime_feature_names()

TARGET: Y_train
────────────────
Shape: (n_train, 13) - APRÈS enrichissement par train_regime_classifier.py

Colonnes Y_train[:, i]:
  [0]  timestamp         - Unix timestamp (int64)
  [1]  asset_id          - ID asset 0-4
  [2]  regime            - Régime 0-3 (4 classes)
  [3]  trend_strength    - Score tendance 0.0-1.0
  [4]  volatility        - Score volatilité 0.0-1.0
  [5]  macd_direction    - Direction MACD Kalman 0/1 (0=DOWN, 1=UP) - TARGET
  [6]  rsi_direction     - Direction RSI Kalman 0/1 (0=DOWN, 1=UP) - TARGET
  [7]  cci_direction     - Direction CCI Kalman 0/1 (0=DOWN, 1=UP) - TARGET
  [8]  regime_prob_0     - P(regime=0) XGBoost [0.0-1.0]
  [9]  regime_prob_1     - P(regime=1) XGBoost [0.0-1.0]
  [10] regime_prob_2     - P(regime=2) XGBoost [0.0-1.0]
  [11] regime_prob_3     - P(regime=3) XGBoost [0.0-1.0]
  [12] regime_pred       - Régime prédit (argmax probs) [0-3]

Note: Colonnes 8-12 ajoutées par train_regime_classifier.py (enrichissement)

EXTRACTION POUR TRAINING DIRECTION:
  Si --indicator macd:
    Y_train = Y_train[:, [0, 1, 5]]  # [timestamp, asset_id, macd_direction]
    → Shape finale: (n_train, 3)
    → Target: Y_train[:, 2] = macd_direction (binaire 0/1)

  Si --indicator rsi:
    Y_train = Y_train[:, [0, 1, 6]]  # [timestamp, asset_id, rsi_direction]
    → Shape finale: (n_train, 3)
    → Target: Y_train[:, 2] = rsi_direction (binaire 0/1)

  Si --indicator cci:
    Y_train = Y_train[:, [0, 1, 7]]  # [timestamp, asset_id, cci_direction]
    → Shape finale: (n_train, 3)
    → Target: Y_train[:, 2] = cci_direction (binaire 0/1)

MODÈLE: CNN-LSTM Binary Classifier
  Input:  X_train[:, :, 2:] (sans timestamp/asset_id) = (n, 25, 20) features
  Output: P(direction=UP) pour l'indicateur choisi [0.0-1.0]

PRÉDICTIONS SAUVEGARDÉES (après training):
  Le script ajoute 3 nouvelles clés NPZ (n'enrichit PAS Y):
    - Y_train_pred: (n_train,) - probabilités prédites train
    - Y_val_pred:   (n_val,)   - probabilités prédites val
    - Y_test_pred:  (n_test,)  - probabilités prédites test

Source dataset: data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz
Généré par: src/prepare_data_regime.py
Enrichi par: src/train_regime_classifier.py (colonnes 8-12)
═══════════════════════════════════════════════════════════════════════════════
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
from prepare_data import load_prepared_data, filter_by_assets
from data_utils import normalize_labels_for_single_output
from utils import log_dataset_metadata
from datetime import datetime


class IndicatorDataset(Dataset):
    """
    Dataset PyTorch pour les séquences d'indicateurs.

    Args:
        X: Features (n_sequences, sequence_length, n_indicators)
        Y: Labels (n_sequences, n_outputs)
        T: Transition indicators (n_sequences,) - optionnel (Phase 2.11)
            - 1.0 si transition (label[i] != label[i-1])
            - 0.0 si continuation
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, T: np.ndarray = None):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        self.T = torch.FloatTensor(T) if T is not None else None
        self.has_transitions = (T is not None)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        if self.has_transitions:
            return self.X[idx], self.Y[idx], self.T[idx]
        else:
            # Backward compatibility - return only X, Y
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
    indicator_names: list = None,
    grad_clip: float = None
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
        grad_clip: Valeur max du gradient (None = pas de clipping)

    Returns:
        Dictionnaire avec loss et métriques
    """
    model.train()

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    for batch in dataloader:
        # Unpacking flexible: (X, Y) ou (X, Y, T)
        if len(batch) == 3:
            X_batch, Y_batch, T_batch = batch
            T_batch = T_batch.to(device)
        else:
            X_batch, Y_batch = batch
            T_batch = None

        # Déplacer sur device
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(X_batch)

        # Loss (passer transitions SEULEMENT si WeightedTransitionBCELoss)
        # Note: T_batch peut exister mais loss_fn peut être baseline (--no-weighted-loss)
        if T_batch is not None and hasattr(loss_fn, 'transition_weight'):
            loss = loss_fn(outputs, Y_batch, T_batch)
        else:
            loss = loss_fn(outputs, Y_batch)

        # Backward
        loss.backward()

        # 🛡️ Gradient clipping pour stabilité
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

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
        for batch in dataloader:
            # Unpacking flexible: (X, Y) ou (X, Y, T)
            if len(batch) == 3:
                X_batch, Y_batch, T_batch = batch
                T_batch = T_batch.to(device)
            else:
                X_batch, Y_batch = batch
                T_batch = None

            # Déplacer sur device
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # Forward
            outputs = model(X_batch)

            # Loss (passer transitions SEULEMENT si WeightedTransitionBCELoss)
            # Note: T_batch peut exister mais loss_fn peut être baseline (--no-weighted-loss)
            if T_batch is not None and hasattr(loss_fn, 'transition_weight'):
                loss = loss_fn(outputs, Y_batch, T_batch)
            else:
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

    # 🔍 DIAGNOSTIC: Afficher distribution des prédictions vs targets
    with torch.no_grad():
        pred_binary = (all_predictions > 0.5).float()
        n_pred_0 = (pred_binary == 0).sum().item()
        n_pred_1 = (pred_binary == 1).sum().item()
        n_target_0 = (all_targets == 0).sum().item()
        n_target_1 = (all_targets == 1).sum().item()
        logger.info(f"  [DEBUG] Prédictions: 0={n_pred_0} ({n_pred_0/len(all_predictions)*100:.1f}%), "
                   f"1={n_pred_1} ({n_pred_1/len(all_predictions)*100:.1f}%)")
        logger.info(f"  [DEBUG] Targets:     0={n_target_0} ({n_target_0/len(all_targets)*100:.1f}%), "
                   f"1={n_target_1} ({n_target_1/len(all_targets)*100:.1f}%)")

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
    indicator_names: list = None,
    grad_clip: float = None
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
        grad_clip: Gradient clipping max norm (None = désactivé)

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
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device, indicator_names, grad_clip)

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
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping max norm (None = désactivé, 1.0 recommandé pour stabilité)')

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

    # Assets filtering
    parser.add_argument('--assets', type=str, nargs='+', default=None,
                        help='Assets à utiliser (ex: --assets BTC ETH). '
                             'Si non spécifié, utilise tous les assets du dataset.')

    # Autres
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help='Random seed pour reproductibilité')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device à utiliser (auto détecte automatiquement)')

    # Weighted Loss Control
    parser.add_argument('--no-weighted-loss', action='store_true',
                        help='Désactiver WeightedTransitionBCELoss même si transitions disponibles (baseline mode)')
    parser.add_argument('--transition-weight', type=float, default=5.0,
                        help='Poids pour les transitions dans WeightedTransitionBCELoss (défaut: 5.0)')

    # Shortcut Last-N Steps
    parser.add_argument('--shortcut', action='store_true',
                        help='Activer shortcut last-5 steps (améliore détection transitions)')
    parser.add_argument('--shortcut-steps', type=int, default=5,
                        help='Nombre de steps pour le shortcut (défaut: 5)')

    # Temporal Gate (poids learnable par timestep)
    parser.add_argument('--temporal-gate', action='store_true',
                        help='Activer temporal gate (poids learnable par timestep, favorise récents)')

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
            # Utiliser predict_proba() qui gère sigmoid conditionnellement
            # (logits → sigmoid si use_bce_with_logits=True, sinon déjà en [0,1])
            outputs = model.predict_proba(X_batch)
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
        # metadata est un numpy scalar dict, utiliser .item() pour extraire le dict Python
        metadata = existing_data['metadata'].item()
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
#
# STRUCTURE DATASET UNIVERSEL (dataset_*_regime.npz):
# Y = [timestamp, asset_id, regime, trend_strength, volatility_cluster,
#      macd_direction, rsi_direction, cci_direction]
# Index: 0        1         2       3               4
#        5              6             7
#
# ⚠️ ATTENTION: Les anciens datasets 3-colonnes utilisaient:
#    Y = [rsi_dir, cci_dir, macd_dir] → indices 0, 1, 2
# ⚠️ Les nouveaux datasets universels 8+ colonnes utilisent:
#    Y[:, 5] = macd_direction, Y[:, 6] = rsi_direction, Y[:, 7] = cci_direction
#
INDICATOR_INDEX = {
    'macd': 5,  # Y[:, 5] = macd_direction (binary: 0, 1)
    'rsi': 6,   # Y[:, 6] = rsi_direction (binary: 0, 1)
    'cci': 7,   # Y[:, 7] = cci_direction (binary: 0, 1)
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
    is_universal_dataset = ('regime' in filename or 'universal' in filename)  # Dataset universel avec tous les labels

    if args.indicator != 'all' and not is_universal_dataset:
        indicator_name = args.indicator.lower()
        if indicator_name not in filename:
            logger.error(f"❌ Incohérence détectée!")
            logger.error(f"   --indicator '{args.indicator}' ne correspond pas au fichier")
            logger.error(f"   Fichier: {Path(args.data).name}")
            logger.error(f"   L'indicateur '{indicator_name}' n'est pas présent dans le nom du fichier")
            raise SystemExit(1)

    if is_universal_dataset:
        logger.info(f"✅ Dataset universel détecté (regime/universal) - contient tous les indicateurs")
    else:
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

    # NOTE: Mode (single vs multi) sera déterminé APRÈS le chargement des données
    # pour permettre la détection automatique de l'indicateur depuis le nom du fichier

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

        # Unpacking avec détection automatique des transitions (Phase 2.11)
        if len(prepared['train']) == 3:
            # Nouveau format: (X, Y, T) avec transitions
            X_train, Y_train, T_train = prepared['train']
            X_val, Y_val, T_val = prepared['val']
            X_test, Y_test, T_test = prepared['test']
            has_transitions = True
            logger.info(f"  ✅ Dataset avec transitions détecté (Phase 2.11 - Weighted Loss)")
        else:
            # Ancien format: (X, Y) sans transitions
            X_train, Y_train = prepared['train']
            X_val, Y_val = prepared['val']
            X_test, Y_test = prepared['test']
            T_train = T_val = T_test = None
            has_transitions = False
            logger.info(f"  ℹ️ Dataset sans transitions (backward compatibility)")

        metadata = prepared['metadata']
        log_dataset_metadata(metadata, logger)

        # =====================================================================
        # EXTRACTION LABEL DEPUIS DATASET UNIVERSEL (si applicable)
        # =====================================================================
        # Dataset universel (regime): Y shape (n, 8) avec toutes les directions
        # Structure: [timestamp, asset_id, regime, ts, vc, macd_dir, rsi_dir, cci_dir]
        logger.info(f"\n🔍 DEBUG - Y shape before extraction: train={Y_train.shape}, val={Y_val.shape}, test={Y_test.shape}")
        is_universal_dataset_extracted = False  # Flag pour éviter écrasement n_outputs_detected

        if Y_train.shape[1] == 8:
            logger.info(f"\n📦 Dataset universel détecté (Y shape: {Y_train.shape})")

            # Mapping indicateur -> colonne Y
            indicator_column_map = {
                'macd': 5,  # Y[:, 5] = macd_direction
                'rsi': 6,   # Y[:, 6] = rsi_direction
                'cci': 7    # Y[:, 7] = cci_direction
            }

            if args.indicator in indicator_column_map:
                col_idx = indicator_column_map[args.indicator]
                logger.info(f"  Extraction label {args.indicator.upper()} (colonne {col_idx})")

                # Extraire [timestamp, asset_id, label_indicateur]
                Y_train = Y_train[:, [0, 1, col_idx]]
                Y_val = Y_val[:, [0, 1, col_idx]]
                Y_test = Y_test[:, [0, 1, col_idx]]

                logger.info(f"  ✅ Y extrait: {Y_train.shape}")

                # =====================================================================
                # EXTRACTION Pure Signal Features pour modèle Direction
                # =====================================================================
                # Dataset régime X: (n, 25, 25) = [timestamp, asset_id, h_ret, l_ret, c_ret, ...]
                # Indices: 0=timestamp, 1=asset_id, 2=h_ret, 3=l_ret, 4=c_ret
                #
                # Features par indicateur (Pure Signal):
                #   - MACD → c_ret uniquement (1 feature)
                #   - RSI  → c_ret uniquement (1 feature)
                #   - CCI  → h_ret, l_ret, c_ret (3 features) car CCI = (TP - MA) / (0.015 * MD)
                #            où TP = (High + Low + Close) / 3
                logger.info(f"     X shape avant: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

                if args.indicator == 'cci':
                    # CCI utilise Typical Price = (H+L+C)/3 → besoin h_ret, l_ret, c_ret
                    logger.info(f"  🎯 Extraction h_ret, l_ret, c_ret (indices 2,3,4) pour CCI...")
                    X_train = X_train[:, :, [0, 1, 2, 3, 4]]  # timestamp, asset_id, h_ret, l_ret, c_ret
                    X_val = X_val[:, :, [0, 1, 2, 3, 4]]
                    X_test = X_test[:, :, [0, 1, 2, 3, 4]]
                    features_name = "h_ret, l_ret, c_ret (3 features)"
                else:
                    # MACD et RSI utilisent uniquement Close → c_ret
                    logger.info(f"  🎯 Extraction c_ret (index 4) pour {args.indicator.upper()}...")
                    X_train = X_train[:, :, [0, 1, 4]]  # timestamp, asset_id, c_ret
                    X_val = X_val[:, :, [0, 1, 4]]
                    X_test = X_test[:, :, [0, 1, 4]]
                    features_name = "c_ret (1 feature)"

                logger.info(f"     X shape après: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
                logger.info(f"  ✅ Features extraites: {features_name}")

                # IMPORTANT: Recalculer n_outputs_detected après extraction
                # Y shape (n, 3) mais seule la dernière colonne est le label
                # Les 2 premières sont timestamp et asset_id (metadata)
                n_outputs_detected = 1  # Direction binaire (UP/DOWN)
                indicator_for_metrics = args.indicator.upper()  # Pour auto-détection architecture
                is_universal_dataset_extracted = True  # Flag pour éviter écrasement ligne 778
                logger.info(f"  🎯 n_outputs mis à jour: {n_outputs_detected} (direction binaire)")
                logger.info(f"  🎯 indicator_for_metrics: {indicator_for_metrics}")

                # Validation post-extraction
                if Y_train.shape[1] != 3:
                    logger.error(f"❌ ERREUR: Y shape après extraction devrait être (n, 3) mais est {Y_train.shape}")
                    raise SystemExit(1)
            else:
                logger.error(f"❌ Indicateur '{args.indicator}' non supporté pour dataset universel")
                logger.error(f"   Indicateurs disponibles: macd, rsi, cci")
                raise SystemExit(1)
        else:
            # Dataset non-universel ou shape incorrect
            logger.warning(f"⚠️ Y shape n'est pas 8 colonnes (shape={Y_train.shape[1]})")
            logger.warning(f"   Ce n'est PAS un dataset universel (regime)")
            logger.warning(f"   Pour entraîner sur directions depuis dataset universel:")
            logger.warning(f"   1. Vérifier que le dataset a 8 colonnes: [timestamp, asset_id, regime, ts, vc, macd_dir, rsi_dir, cci_dir]")
            logger.warning(f"   2. Utiliser --indicator macd|rsi|cci pour extraire la direction appropriée")

        # =====================================================================
        # FILTRAGE PAR ASSETS (optionnel)
        # =====================================================================
        if args.assets:
            logger.info(f"\n🔍 Filtrage des assets...")

            # Charger OHLCV depuis le fichier .npz pour le filtrage
            data_npz = np.load(args.data, allow_pickle=True)

            # Filtrer train
            X_train, Y_train, T_train, _ = filter_by_assets(
                X_train, Y_train, T_train, data_npz['OHLCV_train'],
                args.assets, metadata
            )

            # Filtrer val
            X_val, Y_val, T_val, _ = filter_by_assets(
                X_val, Y_val, T_val, data_npz['OHLCV_val'],
                args.assets, metadata
            )

            # Filtrer test
            X_test, Y_test, T_test, _ = filter_by_assets(
                X_test, Y_test, T_test, data_npz['OHLCV_test'],
                args.assets, metadata
            )

            logger.info(f"  ✅ Filtrage terminé pour {len(args.assets)} asset(s)")

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

    # Ne pas écraser n_outputs_detected s'il a déjà été défini lors de l'extraction du dataset universel
    if not is_universal_dataset_extracted:
        n_outputs_detected = Y_train.shape[1]   # 2 pour dual-binary (direction + force)
    # Sinon, garder la valeur définie ligne 724 (n_outputs_detected = 1 pour direction binaire)

    # Détecter si dual-binary depuis metadata
    is_dual_binary = False
    indicator_for_metrics = None
    filter_type_metadata = None

    if metadata:
        # Détection dual-binary
        if 'label_names' in metadata and len(metadata['label_names']) == 2:
            is_dual_binary = True
            # Extraire nom indicateur (ex: ['rsi_dir', 'rsi_force'] -> 'rsi')
            label_name = metadata['label_names'][0]
            indicator_for_metrics = label_name.split('_')[0].upper()

        # Détecter le type de filtre depuis les métadonnées
        if 'filter_type' in metadata:
            filter_type_metadata = metadata['filter_type']

        # Log architecture détectée
        logger.info(f"\n🔍 Architecture détectée:")
        logger.info(f"  Features: {n_features_detected}")
        logger.info(f"  Outputs: {n_outputs_detected}")

        if is_dual_binary:
            logger.info(f"  Type: DUAL-BINARY ({indicator_for_metrics})")
            logger.info(f"  Labels: Direction + Force")
        else:
            logger.info(f"  Type: SINGLE-OUTPUT")

        if filter_type_metadata:
            logger.info(f"  Filtre: {filter_type_metadata.upper()}")

    # =========================================================================
    # AUTO-DÉTECTION INDICATEUR (depuis filename ou metadata)
    # =========================================================================
    detected_indicator = None
    detected_filter = None

    if args.data:
        data_name = Path(args.data).stem.lower()  # dataset_btc_macd_direction_only_kalman_wt

        # Détecter indicateur depuis le nom du fichier
        for ind in ['rsi', 'cci', 'macd', 'close']:
            if f'_{ind}_' in data_name or data_name.endswith(f'_{ind}'):
                detected_indicator = ind
                break

        # Détecter filtre depuis le nom du fichier (fallback si pas dans metadata)
        for filt in ['kalman', 'octave20', 'octave', 'decycler']:
            if filt in data_name:
                detected_filter = filt
                break

    # Priorité: metadata > filename > CLI
    if is_dual_binary and indicator_for_metrics:
        detected_indicator = indicator_for_metrics.lower()

    # Priorité pour le filtre: metadata > CLI argument > filename
    if filter_type_metadata:
        detected_filter = filter_type_metadata
    elif args.filter:
        detected_filter = args.filter

    # Fallback sur CLI pour ancien pipeline (si aucune détection)
    if not detected_indicator and args.indicator != 'all':
        detected_indicator = args.indicator

    # =========================================================================
    # DÉTERMINER MODE (single vs multi) APRÈS détection indicateur
    # =========================================================================
    # Si indicateur détecté (filename/metadata) OU CLI != 'all' → SINGLE-OUTPUT
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

    # Filtrer les labels si mode single-output (ancien pipeline)
    if single_indicator and not is_dual_binary:
        # Ancien pipeline (3 outputs -> 1)
        Y_train = normalize_labels_for_single_output(Y_train, indicator_idx, indicator_name)
        Y_val = normalize_labels_for_single_output(Y_val, indicator_idx, indicator_name)
        Y_test = normalize_labels_for_single_output(Y_test, indicator_idx, indicator_name)

        # CORRECTION CRITIQUE: Mettre à jour n_outputs_detected après filtrage
        # Le filtrage a réduit Y de (n, 13) à (n, 1)
        n_outputs_detected = Y_train.shape[1]  # Devrait être 1
        logger.info(f"  ✅ n_outputs_detected mis à jour après filtrage single-output: {n_outputs_detected}")

    logger.info(f"\n📊 Datasets:")
    logger.info(f"  Train: X={X_train.shape}, Y={Y_train.shape}")
    logger.info(f"  Val:   X={X_val.shape}, Y={Y_val.shape}")
    logger.info(f"  Test:  X={X_test.shape}, Y={Y_test.shape}")

    # =========================================================================
    # 2. CRÉER DATALOADERS
    # =========================================================================
    logger.info("\n2. Création des DataLoaders...")

    # Passer les transitions si disponibles (Phase 2.11)
    train_dataset = IndicatorDataset(X_train, Y_train, T_train)
    val_dataset = IndicatorDataset(X_val, Y_val, T_val)

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

    # DEBUG: Vérifier les valeurs avant création du modèle
    logger.info(f"\n🔍 DEBUG - Valeurs avant création modèle:")
    logger.info(f"   is_universal_dataset_extracted: {is_universal_dataset_extracted}")
    logger.info(f"   n_outputs_detected: {n_outputs_detected}")
    logger.info(f"   n_features_detected: {n_features_detected}")
    logger.info(f"   indicator_for_metrics: {indicator_for_metrics}")
    logger.info(f"   Y_train.shape: {Y_train.shape}")
    logger.info(f"   Y_train.shape[1] (nb colonnes): {Y_train.shape[1]}")

    # Validation finale avant création modèle
    # Accepter: 1 (single-output filtré), 2 (dual-binary), 3 (universel extrait)
    if not is_universal_dataset_extracted and Y_train.shape[1] not in [1, 2, 3]:
        logger.error(f"\n❌ ERREUR CRITIQUE: Y shape invalide!")
        logger.error(f"   Y_train.shape: {Y_train.shape}")
        logger.error(f"   Y_train.shape[1]: {Y_train.shape[1]} colonnes")
        logger.error(f"   Attendu: 1 (single-output), 2 (dual-binary) ou 3 (universel extrait)")
        logger.error(f"   is_universal_dataset_extracted: {is_universal_dataset_extracted}")
        logger.error(f"")
        logger.error(f"   Soit:")
        logger.error(f"   1. Le dataset a été filtré single-output → shape[1]=1 ✅")
        logger.error(f"   2. Le dataset est dual-binary → shape[1]=2 ✅")
        logger.error(f"   3. Le dataset EST universel (regime) extrait → shape[1]=3 ✅")
        logger.error(f"")
        logger.error(f"   Vérifier que le fichier {args.data} est bien le dataset attendu")
        raise SystemExit(1)

    # Utiliser valeurs détectées au lieu de num_outputs manuel
    num_outputs_final = n_outputs_detected

    # =========================================================================
    # AUTO-DÉTECTION LayerNorm + BCEWithLogitsLoss (architecture hybride)
    # =========================================================================
    # Configuration optimale par indicateur (validée empiriquement)
    use_layer_norm = False  # Par défaut: désactivé
    use_bce_with_logits = False  # Par défaut: désactivé (BCELoss baseline)

    if indicator_for_metrics:
        indicator_lower = indicator_for_metrics.lower()
        if indicator_lower == 'macd':
            # MACD: Les deux optimisations aident (86.9%)
            # Indicateur de tendance lourde (double EMA) → stabilisation bénéfique
            use_layer_norm = True
            use_bce_with_logits = True
            logger.info(f"  🎯 Indicateur MACD détecté → LayerNorm + BCEWithLogitsLoss ACTIVÉS")
        elif indicator_lower == 'cci':
            # CCI: BCEWithLogitsLoss seul optimal (83.3%)
            # 3 features (h,l,c) → BCE aide (+3.8%), LayerNorm nuit (-0.5%)
            use_layer_norm = False
            use_bce_with_logits = True
            logger.info(f"  🎯 Indicateur CCI détecté → BCEWithLogitsLoss ACTIVÉ, LayerNorm DÉSACTIVÉ (optimal)")
        elif indicator_lower == 'rsi':
            # RSI: Baseline optimal (80.7%)
            # Oscillateur simple → optimisations neutres
            use_layer_norm = False
            use_bce_with_logits = False
            logger.info(f"  🎯 Indicateur RSI détecté → Architecture baseline (optimal)")
        else:
            logger.info(f"  🎯 Indicateur {indicator_for_metrics} détecté → Architecture baseline")

    logger.info(f"  num_features={n_features_detected}, num_outputs={num_outputs_final}")
    logger.info(f"  use_layer_norm={use_layer_norm}, use_bce_with_logits={use_bce_with_logits}")

    # Phase 2.11: Utiliser WeightedTransitionBCELoss si transitions disponibles
    # Flag --no-weighted-loss permet de forcer le mode baseline
    use_weighted_loss = has_transitions and not args.no_weighted_loss
    if args.no_weighted_loss and has_transitions:
        logger.info(f"  ⚠️ WeightedTransitionBCELoss DÉSACTIVÉ (--no-weighted-loss)")

    if use_weighted_loss:
        logger.info(f"  🎯 Phase 2.11: WeightedTransitionBCELoss ACTIVÉ (transition_weight={args.transition_weight}×)")
        from model import WeightedTransitionBCELoss

        # Créer le modèle (sans loss, on la remplace)
        model, _ = create_model(
            device=device,
            num_indicators=n_features_detected,
            num_outputs=num_outputs_final,
            cnn_filters=args.cnn_filters,
            lstm_hidden_size=args.lstm_hidden,
            lstm_num_layers=args.lstm_layers,
            lstm_dropout=args.lstm_dropout,
            dense_hidden_size=args.dense_hidden,
            dense_dropout=args.dense_dropout,
            use_layer_norm=use_layer_norm,
            use_bce_with_logits=use_bce_with_logits,
            use_shortcut=args.shortcut,
            shortcut_steps=args.shortcut_steps,
            use_temporal_gate=args.temporal_gate
        )

        # Remplacer par WeightedTransitionBCELoss
        loss_fn = WeightedTransitionBCELoss(
            num_outputs=num_outputs_final,
            transition_weight=args.transition_weight,
            use_bce_with_logits=use_bce_with_logits
        )
    else:
        # Backward compatibility: loss classique
        model, loss_fn = create_model(
            device=device,
            num_indicators=n_features_detected,
            num_outputs=num_outputs_final,
            cnn_filters=args.cnn_filters,
            lstm_hidden_size=args.lstm_hidden,
            lstm_num_layers=args.lstm_layers,
            lstm_dropout=args.lstm_dropout,
            dense_hidden_size=args.dense_hidden,
            dense_dropout=args.dense_dropout,
            use_layer_norm=use_layer_norm,
            use_bce_with_logits=use_bce_with_logits,
            use_shortcut=args.shortcut,
            shortcut_steps=args.shortcut_steps,
            use_temporal_gate=args.temporal_gate
        )

    # Log features actives
    if args.shortcut:
        logger.info(f"🔗 Shortcut Last-{args.shortcut_steps} Steps ACTIVÉ (skip connection)")
    if args.temporal_gate:
        logger.info(f"⏱️ Temporal Gate ACTIVÉ (poids learnable par timestep)")

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
        'use_layer_norm': use_layer_norm,
        'use_bce_with_logits': use_bce_with_logits,
        'use_shortcut': args.shortcut,
        'shortcut_steps': args.shortcut_steps,
        'use_temporal_gate': args.temporal_gate,
    }

    # =========================================================================
    # NOMMAGE AUTOMATIQUE DU MODÈLE
    # =========================================================================
    # detected_indicator et detected_filter ont été détectés plus tôt (lignes 715-748)

    # Construire le nom du modèle
    suffix_parts = []
    if detected_indicator:
        suffix_parts.append(detected_indicator)
    if detected_filter:
        suffix_parts.append(detected_filter)
    if is_dual_binary:
        suffix_parts.append('dual_binary')

    if suffix_parts:
        suffix = '_'.join(suffix_parts)
        save_path = args.save_path.replace('.pth', f'_{suffix}.pth')
    else:
        save_path = args.save_path

    logger.info(f"\n💾 Modèle sauvegardé:")
    logger.info(f"  Indicateur détecté: {detected_indicator or 'aucun'}")
    logger.info(f"  Filtre détecté: {detected_filter or 'aucun'}")

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
        indicator_names=indicator_names_for_metrics,
        grad_clip=args.grad_clip
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
