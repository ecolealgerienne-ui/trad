#!/usr/bin/env python3
"""
Unified Regime Classifier Training - Comparaison équitable CNN-LSTM vs XGBoost

BUT: Comparer CNN-LSTM et XGBoost avec les MÊMES INPUTS (raw returns)
pour une évaluation équitable sans data leakage.

Features utilisées (3 raw returns UNIQUEMENT):
    - h_ret (high return)
    - l_ret (low return)
    - c_ret (close return)

Modèles:
    1. CNN-LSTM: Utilise directement les séquences (n, 25, 3)
    2. XGBoost: Agrège les séquences en 15 features (3 returns × 5 stats)
       - mean, std, min, max, last pour chaque return

Régimes (3 classes):
    0: RANGE_LOW_VOL  - Consolidation calme
    1: RANGE_HIGH_VOL - Consolidation agitée
    2: TREND          - Tendance

Usage:
    # CNN-LSTM (séquences)
    python src/train_regime_unified.py --model cnn-lstm

    # XGBoost (agrégé)
    python src/train_regime_unified.py --model xgboost

    # XGBoost avec SMOTE
    python src/train_regime_unified.py --model xgboost --use-smote --smote-ratio 0.20
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import json
import logging
import sys

# ML libs
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import joblib

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# X structure: (n, 25, 25) - colonnes 0-24
COL_TIMESTAMP = 0
COL_ASSET_ID = 1
COL_H_RET = 2  # Raw returns start
COL_L_RET = 3
COL_C_RET = 4  # Raw returns end

# Y structure: (n, 6+)
COL_Y_REGIME = 2

REGIME_NAMES = {
    0: 'RANGE_LOW_VOL',
    1: 'RANGE_HIGH_VOL',
    2: 'TREND'
}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
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

def load_dataset(npz_path: Path) -> Dict:
    """
    Charge le dataset et extrait UNIQUEMENT les raw returns (colonnes 2-4).

    Returns:
        Dict avec X_raw_returns (n, 25, 3) et régimes
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    logger.info(f"\n{'='*80}")
    logger.info(f"LOADING DATASET: {npz_path.name}")
    logger.info(f"{'='*80}")

    data = np.load(npz_path, allow_pickle=True)

    # Charger les splits
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_val = data['X_val']
    Y_val = data['Y_val']
    X_test = data['X_test']
    Y_test = data['Y_test']

    logger.info(f"\n📦 Original shapes:")
    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  Y_train: {Y_train.shape}")

    # Extraire UNIQUEMENT les raw returns (colonnes 2-4)
    X_train_raw = X_train[:, :, COL_H_RET:COL_C_RET+1].astype(np.float32)
    X_val_raw = X_val[:, :, COL_H_RET:COL_C_RET+1].astype(np.float32)
    X_test_raw = X_test[:, :, COL_H_RET:COL_C_RET+1].astype(np.float32)

    logger.info(f"\n✂️  Extracted RAW RETURNS ONLY:")
    logger.info(f"  X_train_raw: {X_train_raw.shape} (n, seq=25, features=3)")
    logger.info(f"  Features: h_ret, l_ret, c_ret")

    # Extraire régimes (colonne 2 de Y)
    regimes_train = Y_train[:, COL_Y_REGIME].astype(int)
    regimes_val = Y_val[:, COL_Y_REGIME].astype(int)
    regimes_test = Y_test[:, COL_Y_REGIME].astype(int)

    # Distribution
    logger.info(f"\n📊 Train regime distribution:")
    total = len(regimes_train)
    for regime_id in range(3):
        count = np.sum(regimes_train == regime_id)
        pct = 100 * count / total
        logger.info(f"  Regime {regime_id} ({REGIME_NAMES[regime_id]:15s}): {count:8,} ({pct:5.1f}%)")

    return {
        'X_train': X_train_raw,
        'X_val': X_val_raw,
        'X_test': X_test_raw,
        'regimes_train': regimes_train,
        'regimes_val': regimes_val,
        'regimes_test': regimes_test,
        'Y_train': Y_train,
        'Y_val': Y_val,
        'Y_test': Y_test
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING FOR XGBOOST
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_sequences(X_sequences: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Agrège les séquences de raw returns pour XGBoost.

    Input: (n, 25, 3) - séquences de raw returns
    Output: (n, 15) - 3 returns × 5 stats

    Agrégations: mean, std, min, max, last
    """
    n_samples, seq_len, n_features = X_sequences.shape

    logger.info(f"\n🔄 Aggregating sequences for XGBoost...")
    logger.info(f"  Input:  {X_sequences.shape} (n, seq=25, features=3)")

    # Calculer agrégations
    agg_mean = np.mean(X_sequences, axis=1)   # (n, 3)
    agg_std = np.std(X_sequences, axis=1)     # (n, 3)
    agg_min = np.min(X_sequences, axis=1)     # (n, 3)
    agg_max = np.max(X_sequences, axis=1)     # (n, 3)
    agg_last = X_sequences[:, -1, :]          # (n, 3)

    # Concaténer
    X_agg = np.concatenate([
        agg_mean, agg_std, agg_min, agg_max, agg_last
    ], axis=1)  # (n, 15)

    # Noms des features
    return_names = ['h_ret', 'l_ret', 'c_ret']
    feature_names = []
    for agg in ['mean', 'std', 'min', 'max', 'last']:
        for ret in return_names:
            feature_names.append(f"{ret}_{agg}")

    logger.info(f"  Output: {X_agg.shape} (n, 15)")
    logger.info(f"  Features: {len(feature_names)}")

    # Handle NaN/Inf
    n_nan = np.sum(np.isnan(X_agg))
    n_inf = np.sum(np.isinf(X_agg))
    if n_nan > 0 or n_inf > 0:
        logger.warning(f"  Found {n_nan} NaN and {n_inf} Inf - replacing with 0")
        X_agg = np.nan_to_num(X_agg, nan=0.0, posinf=0.0, neginf=0.0)

    return X_agg, feature_names


# ═══════════════════════════════════════════════════════════════════════════════
# SMOTE OVERSAMPLING
# ═══════════════════════════════════════════════════════════════════════════════

def apply_smote(X: np.ndarray, y: np.ndarray, ratio: float, k_neighbors: int,
                is_sequence: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applique SMOTE pour oversampler la classe TREND (regime 2).

    Args:
        X: Features (sequences ou agrégées)
        y: Labels régimes
        ratio: Ratio cible pour TREND (ex: 0.20 = 20%)
        k_neighbors: Nombre de voisins pour SMOTE
        is_sequence: True si X est (n, 25, 3), False si (n, 15)

    Returns:
        (X_resampled, y_resampled)
    """
    if not SMOTE_AVAILABLE:
        logger.error("❌ SMOTE not available. Install: pip install imbalanced-learn")
        return X, y

    logger.info(f"\n{'='*80}")
    logger.info("SMOTE OVERSAMPLING")
    logger.info(f"{'='*80}")

    # Distribution avant
    unique, counts = np.unique(y, return_counts=True)
    counts_orig = counts.copy()
    total = len(y)

    logger.info(f"\n📊 BEFORE SMOTE:")
    for cls, count in zip(unique, counts):
        logger.info(f"  Regime {cls} ({REGIME_NAMES[cls]:15s}): {count:8,} ({count/total*100:5.2f}%)")

    # Target samples pour TREND
    target_trend = int(ratio * total)
    logger.info(f"\n🎯 Target TREND samples: {target_trend:,} ({ratio:.0%} of dataset)")

    # Flatten si séquences
    original_shape = X.shape
    if is_sequence:
        logger.info(f"  Flattening sequences: {X.shape} → ({X.shape[0]}, {X.shape[1]*X.shape[2]})")
        X_flat = X.reshape(len(X), -1)
    else:
        X_flat = X

    # SMOTE
    try:
        logger.info(f"\n⏳ Applying SMOTE (k_neighbors={k_neighbors})...")
        smote = SMOTE(
            sampling_strategy={2: target_trend},
            k_neighbors=k_neighbors,
            random_state=42,
            n_jobs=-1
        )
        X_resampled, y_resampled = smote.fit_resample(X_flat, y)

        # Reshape si séquences
        if is_sequence:
            X_resampled = X_resampled.reshape(-1, original_shape[1], original_shape[2])

        logger.info(f"  ✅ SMOTE completed!")

        # Distribution après
        unique, counts = np.unique(y_resampled, return_counts=True)
        total = len(y_resampled)

        logger.info(f"\n📊 AFTER SMOTE:")
        for cls, count in zip(unique, counts):
            gain = f"(+{count - counts_orig[cls]:,})" if cls == 2 else ""
            logger.info(f"  Regime {cls} ({REGIME_NAMES[cls]:15s}): {count:8,} ({count/total*100:5.2f}%) {gain}")

        logger.info(f"\n📈 Impact:")
        logger.info(f"  Original samples: {len(y):,}")
        logger.info(f"  Resampled samples: {len(y_resampled):,}")
        logger.info(f"  Synthetic added: {len(y_resampled) - len(y):,}")

        return X_resampled, y_resampled

    except Exception as e:
        logger.error(f"\n❌ SMOTE failed: {e}")
        logger.info("  Continuing with original data...")
        return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING - CNN-LSTM
# ═══════════════════════════════════════════════════════════════════════════════

def train_cnn_lstm(dataset: Dict, args: argparse.Namespace) -> Dict:
    """
    Entraîne le modèle CNN-LSTM sur séquences (n, 25, 3).
    """
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING CNN-LSTM")
    logger.info(f"{'='*80}")

    # Data
    X_train = dataset['X_train']
    y_train = dataset['regimes_train']
    X_val = dataset['X_val']
    y_val = dataset['regimes_val']

    # SMOTE si demandé
    if args.use_smote:
        X_train, y_train = apply_smote(
            X_train, y_train,
            ratio=args.smote_ratio,
            k_neighbors=args.smote_k_neighbors,
            is_sequence=True
        )

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\n🔧 Device: {device}")

    # DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    model = CNNLSTMRegimeClassifier(
        input_size=3,
        num_classes=3,
        sequence_length=25
    ).to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0

    logger.info(f"\n🏋️  Training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(y_batch.numpy())

        val_acc = accuracy_score(val_true, val_preds)
        scheduler.step(val_acc)

        logger.info(f"  Epoch {epoch+1:2d}/{args.epochs}: "
                   f"Train Loss={train_loss/len(train_loader):.4f}, "
                   f"Val Acc={val_acc:.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), args.output_dir / 'best_cnn_lstm.pth')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"\n⏸️  Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load(args.output_dir / 'best_cnn_lstm.pth'))

    return {
        'model': model,
        'model_type': 'cnn-lstm',
        'best_val_acc': best_val_acc,
        'device': device
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING - XGBOOST
# ═══════════════════════════════════════════════════════════════════════════════

def train_xgboost(dataset: Dict, args: argparse.Namespace) -> Dict:
    """
    Entraîne XGBoost sur features agrégées (n, 15).
    """
    if not XGBOOST_AVAILABLE:
        logger.error("❌ XGBoost not available. Install: pip install xgboost")
        sys.exit(1)

    logger.info(f"\n{'='*80}")
    logger.info("TRAINING XGBOOST")
    logger.info(f"{'='*80}")

    # Aggregate sequences
    X_train_agg, feature_names = aggregate_sequences(dataset['X_train'])
    X_val_agg, _ = aggregate_sequences(dataset['X_val'])

    y_train = dataset['regimes_train']
    y_val = dataset['regimes_val']

    # SMOTE si demandé
    if args.use_smote:
        X_train_agg, y_train = apply_smote(
            X_train_agg, y_train,
            ratio=args.smote_ratio,
            k_neighbors=args.smote_k_neighbors,
            is_sequence=False
        )

    # Class weights pour le déséquilibre
    unique, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    weights = total / (len(unique) * counts)
    scale_pos_weight = {int(cls): float(w) for cls, w in zip(unique, weights)}

    logger.info(f"\n⚖️  Class weights: {scale_pos_weight}")

    # XGBoost parameters
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    }

    logger.info(f"\n🏋️  Training XGBoost...")

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train_agg, y_train,
        eval_set=[(X_val_agg, y_val)],
        early_stopping_rounds=20,
        verbose=False
    )

    # Validation accuracy
    val_preds = model.predict(X_val_agg)
    val_acc = accuracy_score(y_val, val_preds)

    logger.info(f"\n✅ Best Val Accuracy: {val_acc:.4f}")

    # Save model
    joblib.dump(model, args.output_dir / 'best_xgboost.pkl')

    # Feature importance
    importance = model.feature_importances_
    top_indices = np.argsort(importance)[-10:][::-1]

    logger.info(f"\n🔝 Top 10 Feature Importance:")
    for idx in top_indices:
        logger.info(f"  {feature_names[idx]:20s}: {importance[idx]:.4f}")

    return {
        'model': model,
        'model_type': 'xgboost',
        'best_val_acc': val_acc,
        'feature_names': feature_names
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model_info: Dict, dataset: Dict) -> Dict:
    """
    Évalue le modèle sur le test set.
    """
    logger.info(f"\n{'='*80}")
    logger.info("EVALUATION - TEST SET")
    logger.info(f"{'='*80}")

    model = model_info['model']
    model_type = model_info['model_type']

    X_test = dataset['X_test']
    y_test = dataset['regimes_test']

    # Prédictions
    if model_type == 'cnn-lstm':
        device = model_info['device']
        model.eval()

        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

        y_pred = []
        y_proba = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                proba = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(proba, axis=1)

                y_pred.extend(preds)
                y_proba.extend(proba)

        y_pred = np.array(y_pred)
        y_proba = np.array(y_proba)

    else:  # xgboost
        X_test_agg, _ = aggregate_sequences(X_test)
        y_pred = model.predict(X_test_agg)
        y_proba = model.predict_proba(X_test_agg)

    # Métriques
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # ROC AUC (OvR)
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    except:
        auc = 0.0

    logger.info(f"\n📊 GLOBAL METRICS:")
    logger.info(f"  Accuracy:          {acc:.4f}")
    logger.info(f"  Precision (macro): {precision:.4f}")
    logger.info(f"  Recall (macro):    {recall:.4f}")
    logger.info(f"  F1-Score (macro):  {f1:.4f}")
    logger.info(f"  ROC AUC (OvR):     {auc:.4f}")

    # Per-class metrics
    logger.info(f"\n📊 PER-CLASS METRICS:")
    for regime_id in range(3):
        mask = (y_test == regime_id)
        n_true = np.sum(mask)
        n_pred = np.sum(y_pred == regime_id)

        if n_true > 0:
            prec = precision_score(y_test[mask], y_pred[mask],
                                  labels=[regime_id], average='macro', zero_division=0)
            rec = recall_score(y_test == regime_id, y_pred == regime_id, zero_division=0)
            f1_cls = f1_score(y_test == regime_id, y_pred == regime_id, zero_division=0)

            logger.info(f"\n  Regime {regime_id} - {REGIME_NAMES[regime_id]}:")
            logger.info(f"    Samples:   {n_true:,}")
            logger.info(f"    Precision: {prec:.4f}")
            logger.info(f"    Recall:    {rec:.4f}")
            logger.info(f"    F1-Score:  {f1_cls:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\n🔢 CONFUSION MATRIX:")
    logger.info(f"  True\\Pred  R0      R1      R2")
    for i in range(3):
        row = cm[i]
        logger.info(f"  R{i}       {row[0]:6d}  {row[1]:6d}  {row[2]:6d}")

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'y_pred': y_pred,
        'y_proba': y_proba
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Unified Regime Classifier Training')

    # Data
    parser.add_argument('--data', type=str,
                       default='data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz',
                       help='Path to dataset NPZ')

    # Model selection
    parser.add_argument('--model', type=str, required=True,
                       choices=['cnn-lstm', 'xgboost'],
                       help='Model to train')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Max epochs for CNN-LSTM')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size for CNN-LSTM')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate for CNN-LSTM')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')

    # SMOTE
    parser.add_argument('--use-smote', action='store_true',
                       help='Use SMOTE oversampling')
    parser.add_argument('--smote-ratio', type=float, default=0.20,
                       help='Target ratio for TREND after SMOTE')
    parser.add_argument('--smote-k-neighbors', type=int, default=5,
                       help='K neighbors for SMOTE')

    # Output
    parser.add_argument('--output-dir', type=str,
                       default='models/regime_unified',
                       help='Output directory')

    args = parser.parse_args()

    # Create output dir
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    dataset = load_dataset(Path(args.data))

    # Train
    if args.model == 'cnn-lstm':
        model_info = train_cnn_lstm(dataset, args)
    else:
        model_info = train_xgboost(dataset, args)

    # Evaluate
    results = evaluate_model(model_info, dataset)

    # Save results
    results_file = args.output_dir / f'results_{args.model}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'model': args.model,
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'auc': results['auc'],
            'confusion_matrix': results['confusion_matrix'],
            'best_val_acc': model_info['best_val_acc'],
            'use_smote': args.use_smote,
            'smote_ratio': args.smote_ratio if args.use_smote else None
        }, f, indent=2)

    logger.info(f"\n💾 Results saved to: {results_file}")
    logger.info(f"\n{'='*80}")
    logger.info("DONE")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
