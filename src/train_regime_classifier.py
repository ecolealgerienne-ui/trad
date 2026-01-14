#!/usr/bin/env python3
"""
Regime Classifier Training - Model A (Meta-Regime Phase 1)

Entraîne un classifieur CNN-LSTM multiclass pour prédire le régime de marché (3 classes).

Architecture:
    3 raw returns features (c_ret, h_ret, l_ret)
    → CNN 1D (extraction patterns)
    → LSTM (contexte temporel)
    → Dense + Softmax
    → Probabilités 3 régimes [0, 1, 2]

Avantage vs XGBoost:
    - Pas d'agrégation (mean, std, min, max) nécessaire
    - Le modèle apprend directement sur les séquences (n, 25, 3)
    - Meilleure capture des patterns temporels

Régimes (basés sur Trend Strength × Volatility Cluster):
    0: RANGE_LOW_VOL  - Consolidation calme (TS < 0.45, vol ≤ P50)
    1: RANGE_HIGH_VOL - Consolidation agitée (TS < 0.45, vol > P50)
    2: TREND          - Tendance (TS ≥ 0.45)

Note: En crypto, TREND = VOLATILITÉ (Oxford-Man Institute, BIS 2020).

Features (3 raw returns):
    [0] c_ret - Close return (close[t] - close[t-1]) / close[t-1]
    [1] h_ret - High return (high[t] - close[t-1]) / close[t-1]
    [2] l_ret - Low return (low[t] - close[t-1]) / close[t-1]

Target:
    regime = 0, 1, ou 2 (3 classes)

Référence:
    - Ang & Bekaert (2002) - Regime Switches
    - López de Prado (2018) - Feature Engineering
    - Documentation: docs/META_REGIME_TRADING_SPECS.md

═══════════════════════════════════════════════════════════════════════════════
DONNÉES D'ENTRAÎNEMENT - Structure détaillée
═══════════════════════════════════════════════════════════════════════════════

INPUT: X_train
────────────────
Shape: (n_train, 25, 5)
  - n_train: Nombre d'échantillons train
  - 25: Longueur séquence (25 timesteps × 5min = 2h05 de contexte)
  - 5: Nombre de colonnes (2 metadata + 3 raw returns)

Colonnes X_train[:, :, i]:
  Index 0-1: METADATA (non utilisés par le modèle)
    [0] timestamp    - Unix timestamp (int64)
    [1] asset_id     - ID asset 0-4 (BTC=0, ETH=1, BNB=2, ADA=3, LTC=4)

  Index 2-4: RAW RETURNS FEATURES (3) - UTILISÉS PAR CNN-LSTM
    [2] c_ret - Close return
    [3] h_ret - High return
    [4] l_ret - Low return

Source: prepare_data_regime.py

TARGET: Y_train
────────────────
Shape: (n_train, 6)

Colonnes Y_train[:, i]:
  [0] timestamp       - Unix timestamp (int64)
  [1] asset_id        - ID asset 0-4
  [2] regime          - Régime 0-2 (TARGET PRINCIPAL)
  [3] macd_dir        - Direction MACD Kalman 0/1 (0=DOWN, 1=UP)
  [4] rsi_dir         - Direction RSI Kalman 0/1
  [5] cci_dir         - Direction CCI Kalman 0/1

RÉGIMES (3 classes):
  0: RANGE_LOW_VOL  - Consolidation calme (TS < 0.45, vol ≤ P50)
  1: RANGE_HIGH_VOL - Consolidation agitée (TS < 0.45, vol > P50)
  2: TREND          - Tendance (TS ≥ 0.45)

UTILISATION PAR CNN-LSTM:
  Ce script prend directement les séquences (pas d'agrégation!):
  X_features shape: (n_train, 25, 3) = séquences de raw returns
  Target: regimes_train = Y_train[:, 2]

ENRICHISSEMENT POST-TRAINING:
  Après entraînement, Y est enrichi avec les prédictions du modèle:
  Y enrichi shape: (n, 10) = Y original (6) + regime_pred (1) + probs (3)
  Nouvelles colonnes:
    [6] regime_pred - Prédiction du classifieur
    [7] prob_R0     - Probabilité RANGE_LOW_VOL
    [8] prob_R1     - Probabilité RANGE_HIGH_VOL
    [9] prob_R2     - Probabilité TREND

Source dataset: data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz
Généré par: src/prepare_data_regime.py
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import json
import shutil
from typing import Dict, Tuple
import logging

# SMOTE (optional)
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CNN-LSTM MULTICLASS MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class RegimeCNNLSTM(nn.Module):
    """
    CNN-LSTM pour classification multiclass de régimes (3 classes).

    Architecture:
        Input: (batch, 25, 3) - 25 timesteps × 3 raw returns
        → CNN 1D (extraction features temporelles)
        → LayerNorm (stabilisation)
        → LSTM (contexte séquentiel)
        → Dense + Softmax
        → Output: (batch, 3) - probabilités pour 3 régimes

    Args:
        sequence_length: Longueur des séquences (défaut: 25)
        num_features: Nombre de features (défaut: 3 = c_ret, h_ret, l_ret)
        num_classes: Nombre de classes (défaut: 3 régimes)
        cnn_filters: Nombre de filtres CNN (défaut: 64)
        cnn_kernel_size: Taille kernel CNN (défaut: 3)
        lstm_hidden_size: Taille hidden LSTM (défaut: 64)
        lstm_num_layers: Nombre de couches LSTM (défaut: 2)
        lstm_dropout: Dropout LSTM (défaut: 0.2)
        dense_hidden_size: Taille couche dense (défaut: 32)
        dense_dropout: Dropout dense (défaut: 0.3)
    """

    def __init__(
        self,
        sequence_length: int = 25,
        num_features: int = 3,
        num_classes: int = 3,
        cnn_filters: int = 64,
        cnn_kernel_size: int = 3,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.2,
        dense_hidden_size: int = 32,
        dense_dropout: float = 0.3
    ):
        super(RegimeCNNLSTM, self).__init__()

        self.sequence_length = sequence_length
        self.num_features = num_features
        self.num_classes = num_classes

        # CNN Layer
        self.cnn = nn.Conv1d(
            in_channels=num_features,
            out_channels=cnn_filters,
            kernel_size=cnn_kernel_size,
            stride=1,
            padding=cnn_kernel_size // 2  # Same padding
        )
        self.cnn_activation = nn.ReLU()
        self.cnn_batchnorm = nn.BatchNorm1d(cnn_filters)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(cnn_filters)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )

        # Dense layers
        self.dense1 = nn.Linear(lstm_hidden_size, dense_hidden_size)
        self.dense_activation = nn.ReLU()
        self.dense_dropout = nn.Dropout(dense_dropout)

        # Output layer (num_classes logits)
        self.output = nn.Linear(dense_hidden_size, num_classes)

        logger.info(f"✅ RegimeCNNLSTM créé:")
        logger.info(f"  Input: ({sequence_length}, {num_features})")
        logger.info(f"  CNN: {cnn_filters} filters, kernel={cnn_kernel_size}")
        logger.info(f"  LSTM: {lstm_hidden_size} hidden × {lstm_num_layers} layers")
        logger.info(f"  Dense: {dense_hidden_size}")
        logger.info(f"  Output: {num_classes} classes (softmax)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, sequence_length, num_features)

        Returns:
            Logits (batch, num_classes) - appliquer softmax pour probabilités
        """
        # Input: (batch, seq_len, features)

        # CNN expects (batch, channels, length)
        x = x.transpose(1, 2)  # (batch, features, seq_len)

        # CNN
        x = self.cnn(x)  # (batch, cnn_filters, seq_len)
        x = self.cnn_activation(x)
        x = self.cnn_batchnorm(x)

        # Back to (batch, seq_len, cnn_filters)
        x = x.transpose(1, 2)

        # Layer Norm
        x = self.layer_norm(x)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_hidden)

        # Take last timestep
        x = lstm_out[:, -1, :]  # (batch, lstm_hidden)

        # Dense
        x = self.dense1(x)
        x = self.dense_activation(x)
        x = self.dense_dropout(x)

        # Output logits
        logits = self.output(x)  # (batch, num_classes)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne probabilités softmax."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne classes prédites."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_regime_dataset(npz_path: Path) -> Dict:
    """
    Charge le dataset de régimes préparé.

    Structure attendue du NPZ:
        - X_train, Y_train, OHLCV_train
        - X_val, Y_val, OHLCV_val
        - X_test, Y_test, OHLCV_test
        - metadata: JSON avec infos

    Args:
        npz_path: Chemin vers le fichier .npz

    Returns:
        Dict avec splits séparés et metadata
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    print(f"Loading dataset: {npz_path.name}")
    data = np.load(npz_path, allow_pickle=True)

    # Extraire les splits
    X_train = data['X_train']
    Y_train = data['Y_train']
    OHLCV_train = data['OHLCV_train']

    X_val = data['X_val']
    Y_val = data['Y_val']
    OHLCV_val = data['OHLCV_val']

    X_test = data['X_test']
    Y_test = data['Y_test']
    OHLCV_test = data['OHLCV_test']

    # Metadata peut être stocké de différentes façons dans un npz
    if 'metadata' in data:
        try:
            meta_raw = data['metadata']
            # Cas 1: numpy array contenant un dict directement
            if hasattr(meta_raw, 'item'):
                meta_item = meta_raw.item()
                if isinstance(meta_item, dict):
                    metadata = meta_item
                elif isinstance(meta_item, str):
                    metadata = json.loads(meta_item)
                else:
                    metadata = {}
            # Cas 2: string JSON directe
            elif isinstance(meta_raw, str):
                metadata = json.loads(meta_raw)
            else:
                metadata = {}
        except (AttributeError, ValueError, json.JSONDecodeError, TypeError):
            metadata = {}
    else:
        metadata = {}

    # Extraire les régimes (colonne 2 de Y)
    regimes_train = Y_train[:, 2].astype(int)
    regimes_val = Y_val[:, 2].astype(int)
    regimes_test = Y_test[:, 2].astype(int)

    print(f"\n  Split sizes:")
    print(f"    Train: {len(regimes_train):,} samples")
    print(f"    Val:   {len(regimes_val):,} samples")
    print(f"    Test:  {len(regimes_test):,} samples")
    print(f"  Sequences shape: {X_train.shape}")
    print(f"  Feature columns: {X_train.shape[2] - 2}")  # -2 pour timestamp, asset_id

    # Distribution des régimes (Train uniquement)
    print(f"\n  Train regime distribution:")
    regime_names = {
        0: 'RANGE LOW VOL',
        1: 'RANGE HIGH VOL',
        2: 'TREND'
    }
    for regime_id in range(3):
        count = np.sum(regimes_train == regime_id)
        pct = 100 * count / len(regimes_train)
        print(f"    Regime {regime_id} ({regime_names[regime_id]:15s}): {count:,} ({pct:.1f}%)")

    return {
        'X_train': X_train,
        'Y_train': Y_train,
        'OHLCV_train': OHLCV_train,
        'regimes_train': regimes_train,
        'X_val': X_val,
        'Y_val': Y_val,
        'OHLCV_val': OHLCV_val,
        'regimes_val': regimes_val,
        'X_test': X_test,
        'Y_test': Y_test,
        'OHLCV_test': OHLCV_test,
        'regimes_test': regimes_test,
        'metadata': metadata
    }


def prepare_features_for_cnn_lstm(X: np.ndarray) -> np.ndarray:
    """
    Extrait les features pour CNN-LSTM depuis les séquences.

    Contrairement à XGBoost, pas d'agrégation - on prend directement les séquences!

    Args:
        X: Séquences (n, 25, 5) avec [timestamp, asset_id, c_ret, h_ret, l_ret]

    Returns:
        Features (n, 25, 3) = séquences de [c_ret, h_ret, l_ret]
    """
    # Extraire uniquement les features (skip timestamp et asset_id)
    features = X[:, :, 2:]  # (n, 25, 3) = [c_ret, h_ret, l_ret]
    return features.astype(np.float32)


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 256,
    device: str = 'cpu'
) -> Tuple[DataLoader, DataLoader]:
    """
    Crée les DataLoaders PyTorch.

    Args:
        X_train, y_train: Données train
        X_val, y_val: Données validation
        batch_size: Taille des batches
        device: Device ('cpu' ou 'cuda')

    Returns:
        (train_loader, val_loader)
    """
    # Convertir en tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    # Créer datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    # Créer loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == 'cuda')
    )

    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_cnn_lstm_regime_classifier(
    model: RegimeCNNLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = 'cpu',
    epochs: int = 50,
    learning_rate: float = 0.001,
    patience: int = 10
) -> Tuple[RegimeCNNLSTM, Dict]:
    """
    Entraîne le classifieur CNN-LSTM multiclass pour les régimes.

    Args:
        model: Modèle CNN-LSTM
        train_loader: DataLoader train
        val_loader: DataLoader validation
        device: Device
        epochs: Nombre d'époques max
        learning_rate: Learning rate
        patience: Early stopping patience

    Returns:
        (model, history)
    """
    print("\n" + "="*80)
    print("TRAINING CNN-LSTM REGIME CLASSIFIER (Multiclass)")
    print("="*80)

    print(f"\nDevice: {device}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Early stopping patience: {patience}")

    model = model.to(device)

    # Loss et optimizer
    criterion = nn.CrossEntropyLoss()  # Multiclass classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    print("\nTraining...")
    for epoch in range(1, epochs + 1):
        # ===== TRAIN =====
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += X_batch.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += X_batch.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Update scheduler
        scheduler.step(val_loss)

        # Track history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Print progress
        print(f"Epoch {epoch:3d}/{epochs}: "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            print(f"  → New best model saved (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\n⚠️ Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Restored best model (val_loss={best_val_loss:.4f})")

    return model, history


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_regime_classifier(
    model: RegimeCNNLSTM,
    X: np.ndarray,
    y: np.ndarray,
    device: str,
    split_name: str
) -> Dict:
    """
    Évalue le classifieur de régimes sur un split.

    Args:
        model: Modèle CNN-LSTM entraîné
        X: Features (n, 25, 3)
        y: Régimes (n,) - valeurs [0, 1, 2]
        device: Device
        split_name: Nom du split (train/val/test)

    Returns:
        Métriques: {accuracy, precision_macro, recall_macro, f1_macro, roc_auc_ovr}
    """
    print("\n" + "="*80)
    print(f"EVALUATION - {split_name.upper()} SET")
    print("="*80)

    print(f"Samples: {len(X):,}")

    model.eval()

    # Prédictions par batch pour éviter OOM
    batch_size = 4096
    n_samples = len(X)
    all_preds = []
    all_proba = []

    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = torch.tensor(X[start_idx:end_idx], dtype=torch.float32).to(device)

            logits = model(X_batch)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_proba.append(proba)
            all_preds.append(preds)

            # Libérer mémoire GPU
            del X_batch, logits
            torch.cuda.empty_cache()

    y_pred_proba = np.concatenate(all_proba, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)

    # Métriques
    acc = accuracy_score(y, y_pred)
    prec_macro = precision_score(y, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y, y_pred, average='macro', zero_division=0)

    # ROC AUC (One-vs-Rest pour multiclass)
    try:
        auc_ovr = roc_auc_score(y, y_pred_proba, multi_class='ovr', average='macro')
    except ValueError:
        auc_ovr = 0.0  # Si une classe manque dans y

    print(f"\nMetrics:")
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  Precision (macro): {prec_macro:.4f}")
    print(f"  Recall (macro):    {rec_macro:.4f}")
    print(f"  F1-Score (macro):  {f1_macro:.4f}")
    print(f"  ROC AUC (OvR):     {auc_ovr:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print("     ", "  ".join([f"R{i}" for i in range(3)]))
    for i, row in enumerate(cm):
        print(f"  R{i}:", "  ".join([f"{val:4d}" for val in row]))

    # Per-class metrics
    print(f"\nPer-class metrics:")
    regime_names = {
        0: 'RANGE LOW VOL',
        1: 'RANGE HIGH VOL',
        2: 'TREND'
    }

    prec_per_class = precision_score(y, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y, y_pred, average=None, zero_division=0)

    for i in range(3):
        print(f"  Regime {i} ({regime_names[i]:15s}): "
              f"Prec={prec_per_class[i]:.3f}, "
              f"Rec={rec_per_class[i]:.3f}, "
              f"F1={f1_per_class[i]:.3f}")

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, digits=4, target_names=[
        'R0: RANGE LOW VOL',
        'R1: RANGE HIGH VOL',
        'R2: TREND'
    ]))

    return {
        'accuracy': acc,
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'f1_macro': f1_macro,
        'roc_auc_ovr': auc_ovr,
        'confusion_matrix': cm.tolist()
    }


def generate_predictions(
    model: RegimeCNNLSTM,
    X: np.ndarray,
    device: str,
    batch_size: int = 1024
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Génère prédictions et probabilités pour un dataset.

    Args:
        model: Modèle entraîné
        X: Features (n, 25, 3)
        device: Device
        batch_size: Taille batch pour inférence

    Returns:
        (predictions, probabilities) - (n,) et (n, 3)
    """
    model.eval()

    all_preds = []
    all_probs = []

    n_samples = len(X)
    n_batches = (n_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_samples)

            X_batch = torch.tensor(X[start:end], dtype=torch.float32).to(device)
            logits = model(X_batch)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.append(preds)
            all_probs.append(probs)

    return np.concatenate(all_preds), np.concatenate(all_probs)


# ═══════════════════════════════════════════════════════════════════════════════
# XGBOOST TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_xgboost(full_data: Dict, args) -> Dict:
    """
    Entraîne un modèle XGBoost pour classification de régimes.

    Args:
        full_data: Dictionnaire avec X_train, X_val, X_test, y_train, y_val, y_test
        args: Arguments CLI

    Returns:
        Dict avec modèle, métriques, etc.
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost n'est pas installé. Installez avec: pip install xgboost")

    logger.info("\n" + "="*80)
    logger.info("XGBOOST TRAINING")
    logger.info("="*80)

    # Extraire features (colonnes 2-4: raw returns uniquement)
    X_train_seq = full_data['X_train'][:, :, 2:5]  # (n, 25, 3)
    X_val_seq = full_data['X_val'][:, :, 2:5]
    X_test_seq = full_data['X_test'][:, :, 2:5]

    # Agréger features (mean, std, min, max, last)
    def aggregate_features(X_seq):
        """Agrège séquences (n,25,3) → (n,15) avec 5 stats × 3 features."""
        n = X_seq.shape[0]
        X_agg = np.zeros((n, 15))  # 3 features × 5 stats
        for i in range(3):
            X_agg[:, i*5 + 0] = np.mean(X_seq[:, :, i], axis=1)
            X_agg[:, i*5 + 1] = np.std(X_seq[:, :, i], axis=1)
            X_agg[:, i*5 + 2] = np.min(X_seq[:, :, i], axis=1)
            X_agg[:, i*5 + 3] = np.max(X_seq[:, :, i], axis=1)
            X_agg[:, i*5 + 4] = X_seq[:, -1, i]  # last
        return X_agg

    X_train = aggregate_features(X_train_seq)
    X_val = aggregate_features(X_val_seq)
    X_test = aggregate_features(X_test_seq)

    y_train = full_data['regimes_train']
    y_val = full_data['regimes_val']
    y_test = full_data['regimes_test']

    logger.info(f"X_train aggregated: {X_train.shape} (15 features)")
    logger.info(f"X_val aggregated: {X_val.shape}")
    logger.info(f"X_test aggregated: {X_test.shape}")

    # Distribution labels
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    logger.info("\n📊 Distribution train:")
    for cls, count in zip(unique_train, counts_train):
        pct = 100.0 * count / len(y_train)
        logger.info(f"  Regime {cls}: {count:,} ({pct:.1f}%)")

    # SMOTE si demandé
    if args.use_smote:
        if not SMOTE_AVAILABLE:
            raise ImportError(
                "SMOTE requested but imbalanced-learn not installed. "
                "Install with: pip install imbalanced-learn"
            )
        logger.info(f"\n🔄 SMOTE oversampling (target ratio={args.smote_ratio})...")

        # Calculer sampling_strategy
        n_samples = len(y_train)
        target_count_trend = int(n_samples * args.smote_ratio)

        sampling_strategy = {2: target_count_trend}  # Classe TREND=2

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=args.smote_k_neighbors,
            random_state=42
        )

        X_train, y_train = smote.fit_resample(X_train, y_train)

        logger.info(f"✅ SMOTE done: {X_train.shape[0]:,} samples")
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        logger.info("📊 Distribution après SMOTE:")
        for cls, count in zip(unique_train, counts_train):
            pct = 100.0 * count / len(y_train)
            logger.info(f"  Regime {cls}: {count:,} ({pct:.1f}%)")

    # Entraînement XGBoost
    logger.info("\n🚀 Training XGBoost...")

    # Calculer scale_pos_weight pour déséquilibre
    unique, counts = np.unique(y_train, return_counts=True)
    scale_pos_weight = counts[0] / counts[2] if len(counts) > 2 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=getattr(args, 'n_estimators', 200),
        max_depth=getattr(args, 'max_depth', 8),
        learning_rate=getattr(args, 'learning_rate', 0.1),
        objective='multi:softmax',
        num_class=3,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    logger.info("✅ Training complete")

    # Évaluation
    logger.info("\n" + "="*80)
    logger.info("EVALUATION")
    logger.info("="*80)

    # Prédictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    y_train_proba = model.predict_proba(X_train)
    y_val_proba = model.predict_proba(X_val)
    y_test_proba = model.predict_proba(X_test)

    # Métriques
    results = {}

    for split_name, y_true, y_pred, y_proba in [
        ('train', y_train, y_train_pred, y_train_proba),
        ('val', y_val, y_val_pred, y_val_proba),
        ('test', y_test, y_test_pred, y_test_proba)
    ]:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        try:
            auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except:
            auc = 0.0

        results[split_name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'auc': float(auc)
        }

        logger.info(f"\n{split_name.upper()}: Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f} | AUC={auc:.4f}")

    # Sauvegarder modèle
    model_path = args.output_dir / 'regime_xgboost.json'
    model.save_model(str(model_path))
    logger.info(f"\n💾 Modèle sauvegardé: {model_path}")

    # Sauvegarder métriques
    results_path = args.output_dir / 'xgboost_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"💾 Résultats sauvegardés: {results_path}")

    return {
        'model': model,
        'results': results,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Train Regime Classifier - CNN-LSTM or XGBoost')

    # Model selection
    parser.add_argument('--model', type=str, default='cnn-lstm',
                        choices=['cnn-lstm', 'xgboost'],
                        help='Model type to train (default: cnn-lstm)')

    parser.add_argument('--data', type=Path, required=True,
                        help='Path to prepared regime dataset (.npz)')
    parser.add_argument('--output-dir', type=Path, default=Path('models/regime'),
                        help='Output directory for regime classifier')

    # CNN-LSTM specific arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Max epochs for CNN-LSTM (default: 50)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for CNN-LSTM (default: 256)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for CNN-LSTM (default: 0.001)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience for CNN-LSTM (default: 10)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device (default: auto)')

    # XGBoost specific arguments
    parser.add_argument('--n-estimators', type=int, default=200,
                        help='Number of trees for XGBoost (default: 200)')
    parser.add_argument('--max-depth', type=int, default=8,
                        help='Max tree depth for XGBoost (default: 8)')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='Learning rate for XGBoost (default: 0.1)')

    # SMOTE options (for TREND class oversampling) - works for both models
    parser.add_argument('--use-smote', action='store_true',
                        help='Use SMOTE oversampling for minority classes')
    parser.add_argument('--smote-ratio', type=float, default=0.20,
                        help='Target ratio for TREND class after SMOTE (default: 0.20 = 20%% of dataset)')
    parser.add_argument('--smote-k-neighbors', type=int, default=5,
                        help='Number of nearest neighbors for SMOTE (default: 5)')

    args = parser.parse_args()

    print("="*80)
    print(f"REGIME CLASSIFIER TRAINING - {args.model.upper()}")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Output: {args.output_dir}")

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Device: {device}")

    # Créer répertoire output
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Charger dataset
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)

    full_data = load_regime_dataset(args.data)

    # ═══════════════════════════════════════════════════════════════════════════════
    # ROUTING: Train CNN-LSTM or XGBoost based on --model argument
    # ═══════════════════════════════════════════════════════════════════════════════

    if args.model == 'xgboost':
        # XGBoost training path
        train_xgboost(full_data, args)
        return  # Exit after XGBoost training

    # CNN-LSTM training path (default)
    # Continue with existing CNN-LSTM logic below

    # Préparer features (extraire colonnes 2-4 = c_ret, h_ret, l_ret)
    print("\n" + "="*80)
    print("PREPARING FEATURES FOR CNN-LSTM")
    print("="*80)

    X_train = prepare_features_for_cnn_lstm(full_data['X_train'])
    X_val = prepare_features_for_cnn_lstm(full_data['X_val'])
    X_test = prepare_features_for_cnn_lstm(full_data['X_test'])

    y_train = full_data['regimes_train']
    y_val = full_data['regimes_val']
    y_test = full_data['regimes_test']

    print(f"\nFeatures shapes:")
    print(f"  X_train: {X_train.shape} (n, seq_len, features)")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # SMOTE OVERSAMPLING (if enabled)
    # ═══════════════════════════════════════════════════════════════════════════════
    if args.use_smote:
        if not SMOTE_AVAILABLE:
            raise ImportError(
                "SMOTE requested but imbalanced-learn not installed. "
                "Install with: pip install imbalanced-learn"
            )
        print("\n" + "="*80)
        print("SMOTE OVERSAMPLING - Improving TREND detection")
        print("="*80)

        # Print original class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        counts_orig = counts.copy()  # Store for later comparison
        total = len(y_train)
        print(f"\n📊 BEFORE SMOTE - Class distribution (Train):")
        for cls, count in zip(unique, counts):
            regime_name = ['RANGE_LOW_VOL', 'RANGE_HIGH_VOL', 'TREND'][cls]
            print(f"  Regime {cls} ({regime_name:15s}): {count:8,} ({count/total*100:5.2f}%)")

        # Calculate target samples for TREND (regime 2)
        n_train = len(y_train)
        target_trend_samples = int(args.smote_ratio * n_train)

        print(f"\n🎯 Target configuration:")
        print(f"  TREND target ratio: {args.smote_ratio:.1%}")
        print(f"  TREND target samples: {target_trend_samples:,}")
        print(f"  k_neighbors: {args.smote_k_neighbors}")

        # Save original shape
        original_shape = X_train.shape  # (n, seq_len, features)
        seq_len, n_features = original_shape[1], original_shape[2]

        # Flatten sequences for SMOTE (SMOTE requires 2D input)
        print(f"\n🔄 Reshaping for SMOTE...")
        X_train_flat = X_train.reshape(len(X_train), -1)  # (n, seq_len * features)
        print(f"  Flattened shape: {X_train_flat.shape}")

        # Apply SMOTE to oversample TREND (regime 2) only
        sampling_strategy = {2: target_trend_samples}  # Only oversample TREND

        try:
            print(f"\n⏳ Applying SMOTE (this may take a few minutes)...")
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=args.smote_k_neighbors,
                random_state=42,
                n_jobs=-1  # Use all CPU cores
            )
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, y_train)

            # Reshape back to sequence format
            X_train = X_train_resampled.reshape(-1, seq_len, n_features)
            y_train = y_train_resampled

            print(f"  ✅ SMOTE completed successfully!")

            # Print new class distribution
            unique, counts = np.unique(y_train, return_counts=True)
            total = len(y_train)
            print(f"\n📊 AFTER SMOTE - Class distribution (Train):")
            for cls, count in zip(unique, counts):
                regime_name = ['RANGE_LOW_VOL', 'RANGE_HIGH_VOL', 'TREND'][cls]
                gain = f"(+{count - counts_orig[cls]:,})" if cls == 2 else ""
                print(f"  Regime {cls} ({regime_name:15s}): {count:8,} ({count/total*100:5.2f}%) {gain}")

            print(f"\n📈 SMOTE impact:")
            print(f"  Original samples: {original_shape[0]:,}")
            print(f"  Resampled samples: {len(y_train):,}")
            print(f"  Synthetic samples added: {len(y_train) - original_shape[0]:,}")
            print(f"  Final X_train shape: {X_train.shape}")

        except Exception as e:
            print(f"\n❌ ERROR during SMOTE: {e}")
            print(f"  Continuing with original (non-resampled) data...")

    else:
        print(f"\n⏩ SMOTE disabled (use --use-smote to enable)")

    # Créer DataLoaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val,
        batch_size=args.batch_size,
        device=device
    )

    # Créer modèle
    model = RegimeCNNLSTM(
        sequence_length=X_train.shape[1],  # 25
        num_features=X_train.shape[2],     # 3
        num_classes=3                       # 3 régimes
    )

    # Compter paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Entraîner
    model, history = train_cnn_lstm_regime_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience
    )

    # Évaluer sur les 3 splits
    results = {}
    for split_name, X_split, y_split in [
        ('train', X_train, y_train),
        ('val', X_val, y_val),
        ('test', X_test, y_test)
    ]:
        results[split_name] = evaluate_regime_classifier(
            model=model,
            X=X_split,
            y=y_split,
            device=device,
            split_name=split_name
        )

    # Générer prédictions pour enrichir le dataset
    print("\n" + "="*80)
    print("ENRICHING DATASET WITH REGIME PREDICTIONS")
    print("="*80)

    regime_preds_train, regime_probs_train = generate_predictions(model, X_train, device)
    regime_preds_val, regime_probs_val = generate_predictions(model, X_val, device)
    regime_preds_test, regime_probs_test = generate_predictions(model, X_test, device)

    # Enrichir Y avec les prédictions
    # Y original: (n, 6) - [timestamp, asset_id, regime, macd_dir, rsi_dir, cci_dir]
    # Y enrichi: (n, 10) - [Y_original (6), regime_pred (1), prob_R0, prob_R1, prob_R2 (3)]
    Y_train_enriched = np.column_stack([
        full_data['Y_train'],
        regime_preds_train.reshape(-1, 1),
        regime_probs_train
    ])

    Y_val_enriched = np.column_stack([
        full_data['Y_val'],
        regime_preds_val.reshape(-1, 1),
        regime_probs_val
    ])

    Y_test_enriched = np.column_stack([
        full_data['Y_test'],
        regime_preds_test.reshape(-1, 1),
        regime_probs_test
    ])

    # Créer backup de l'original (seulement la première fois)
    backup_path = args.data.parent / f"{args.data.stem}_original.npz"
    if not backup_path.exists():
        print(f"\n📦 Creating backup of original dataset...")
        shutil.copy(args.data, backup_path)
        print(f"  ✅ Backup saved: {backup_path.name}")
    else:
        print(f"\n📦 Backup already exists: {backup_path.name}")

    # Remplacer le fichier original avec la version enrichie
    print(f"\n💾 Enriching and saving dataset: {args.data.name}")
    print(f"  Added columns: regime_pred, prob_R0, prob_R1, prob_R2")
    print(f"  Y shape: {full_data['Y_train'].shape} → {Y_train_enriched.shape}")

    np.savez_compressed(
        args.data,  # Remplace l'original
        X_train=full_data['X_train'],
        Y_train=Y_train_enriched,
        OHLCV_train=full_data['OHLCV_train'],
        X_val=full_data['X_val'],
        Y_val=Y_val_enriched,
        OHLCV_val=full_data['OHLCV_val'],
        X_test=full_data['X_test'],
        Y_test=Y_test_enriched,
        OHLCV_test=full_data['OHLCV_test'],
        metadata=full_data['metadata']
    )

    print(f"✅ Dataset enriched and saved!")
    print(f"  Original backup: {backup_path.name}")

    # Sauvegarder modèle
    model_path = args.output_dir / 'regime_classifier_cnn_lstm.pth'
    print(f"\nSaving regime classifier to: {model_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'sequence_length': model.sequence_length,
            'num_features': model.num_features,
            'num_classes': model.num_classes
        },
        'history': history
    }, model_path)

    # Sauvegarder résultats
    results_path = args.output_dir / 'regime_classifier_results.json'
    print(f"Saving results to: {results_path}")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("✅ REGIME CLASSIFIER TRAINING COMPLETED")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Results: {results_path}")
    print(f"\nTest Metrics:")
    print(f"  Accuracy:    {results['test']['accuracy']:.4f}")
    print(f"  F1 (macro):  {results['test']['f1_macro']:.4f}")
    print(f"  ROC AUC (OvR): {results['test']['roc_auc_ovr']:.4f}")


if __name__ == '__main__':
    main()
