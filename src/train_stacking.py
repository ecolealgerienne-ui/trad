#!/usr/bin/env python3
"""
Entraînement du Meta-Modèle pour Stacking / Ensemble Learning

Objectif: Combiner les prédictions de 3 modèles experts (MACD, RSI, CCI)
pour améliorer la prédiction de Direction (Kalman original).

Cible: Direction Kalman (label original, pas de relabeling)

Hypothèse: Si le Stacking améliore l'Accuracy Direction (92% → 95-96%),
la rentabilité devrait suivre naturellement car on colle mieux au Kalman.

Modèles testés:
  1. Logistic Regression (baseline)
  2. Random Forest (si non-linéaire)
  3. MLP (si très non-linéaire)

Usage:
  python src/train_stacking.py --model logistic
  python src/train_stacking.py --model rf
  python src/train_stacking.py --model mlp
"""

import numpy as np
from pathlib import Path
import logging
import argparse
from typing import Dict
import json

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcule les métriques de classification."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }


def train_logistic_regression(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray
) -> LogisticRegression:
    """Entraîne une Régression Logistique."""
    logger.info("\n🎯 Modèle: Logistic Regression")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )

    logger.info("   Entraînement...")
    model.fit(X_train, Y_train.ravel())

    # Évaluation
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    metrics_train = compute_metrics(Y_train.ravel(), y_train_pred)
    metrics_val = compute_metrics(Y_val.ravel(), y_val_pred)

    logger.info(f"\n   Train Accuracy: {metrics_train['accuracy']*100:.2f}%")
    logger.info(f"   Val Accuracy:   {metrics_val['accuracy']*100:.2f}%")

    # Poids des features (interprétabilité)
    logger.info(f"\n   Poids des features:")
    feature_names = ['MACD_dir', 'MACD_force', 'RSI_dir', 'RSI_force', 'CCI_dir', 'CCI_force']
    for name, weight in zip(feature_names, model.coef_[0]):
        logger.info(f"     {name:12s}: {weight:+.4f}")

    return model, metrics_train, metrics_val


def train_random_forest(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray
) -> RandomForestClassifier:
    """Entraîne un Random Forest."""
    logger.info("\n🎯 Modèle: Random Forest")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    logger.info("   Entraînement...")
    model.fit(X_train, Y_train.ravel())

    # Évaluation
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    metrics_train = compute_metrics(Y_train.ravel(), y_train_pred)
    metrics_val = compute_metrics(Y_val.ravel(), y_val_pred)

    logger.info(f"\n   Train Accuracy: {metrics_train['accuracy']*100:.2f}%")
    logger.info(f"   Val Accuracy:   {metrics_val['accuracy']*100:.2f}%")

    # Feature importance
    logger.info(f"\n   Feature Importance:")
    feature_names = ['MACD_dir', 'MACD_force', 'RSI_dir', 'RSI_force', 'CCI_dir', 'CCI_force']
    importances = sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)
    for name, imp in importances:
        logger.info(f"     {name:12s}: {imp:.4f}")

    return model, metrics_train, metrics_val


class SimpleMLP(nn.Module):
    """MLP simple pour meta-learning."""
    def __init__(self, input_size=6, hidden_size=32, dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def train_mlp(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 0.001,
    device: str = 'cuda'
) -> SimpleMLP:
    """Entraîne un MLP."""
    logger.info("\n🎯 Modèle: MLP (Multi-Layer Perceptron)")

    # Créer datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(Y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(Y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Créer modèle
    model = SimpleMLP().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logger.info(f"   Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    logger.info(f"   Device: {device}")

    best_val_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Val
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                preds = (outputs.cpu() > 0.5).float()
                val_preds.append(preds)
                val_targets.append(Y_batch)

        val_preds = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_acc = accuracy_score(val_targets.ravel(), val_preds.ravel())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            logger.info(f"   Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

    # Charger meilleur modèle
    model.load_state_dict(best_model_state)

    # Évaluation finale
    model.eval()
    with torch.no_grad():
        train_preds = (model(torch.FloatTensor(X_train).to(device)).cpu() > 0.5).numpy()
        val_preds = (model(torch.FloatTensor(X_val).to(device)).cpu() > 0.5).numpy()

    metrics_train = compute_metrics(Y_train.ravel(), train_preds.ravel())
    metrics_val = compute_metrics(Y_val.ravel(), val_preds.ravel())

    logger.info(f"\n   Best Val Accuracy: {best_val_acc*100:.2f}%")
    logger.info(f"   Final Train Accuracy: {metrics_train['accuracy']*100:.2f}%")

    return model, metrics_train, metrics_val


def main():
    parser = argparse.ArgumentParser(
        description="Entraîne le meta-modèle pour Stacking"
    )
    parser.add_argument(
        '--model',
        choices=['logistic', 'rf', 'mlp'],
        default='logistic',
        help='Type de meta-modèle'
    )
    parser.add_argument(
        '--meta-dir',
        default='data/meta',
        help='Répertoire des méta-features'
    )
    parser.add_argument(
        '--output-dir',
        default='models',
        help='Répertoire de sortie'
    )
    parser.add_argument(
        '--device',
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='Device (pour MLP uniquement)'
    )

    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    logger.info("="*80)
    logger.info("ENTRAÎNEMENT META-MODÈLE (STACKING)")
    logger.info("="*80)
    logger.info(f"\nModèle: {args.model.upper()}")

    # Charger méta-features
    meta_dir = Path(args.meta_dir)

    logger.info(f"\n📁 Chargement méta-features...")
    data_train = np.load(meta_dir / 'meta_features_train.npz')
    data_val = np.load(meta_dir / 'meta_features_val.npz')
    data_test = np.load(meta_dir / 'meta_features_test.npz')

    X_train, Y_train = data_train['X_meta'], data_train['Y_meta']
    X_val, Y_val = data_val['X_meta'], data_val['Y_meta']
    X_test, Y_test = data_test['X_meta'], data_test['Y_meta']

    logger.info(f"   Train: X={X_train.shape}, Y={Y_train.shape}")
    logger.info(f"   Val:   X={X_val.shape}, Y={Y_val.shape}")
    logger.info(f"   Test:  X={X_test.shape}, Y={Y_test.shape}")

    # Entraîner modèle
    if args.model == 'logistic':
        model, metrics_train, metrics_val = train_logistic_regression(
            X_train, Y_train, X_val, Y_val
        )
    elif args.model == 'rf':
        model, metrics_train, metrics_val = train_random_forest(
            X_train, Y_train, X_val, Y_val
        )
    else:  # mlp
        model, metrics_train, metrics_val = train_mlp(
            X_train, Y_train, X_val, Y_val, device=device
        )

    # Évaluation sur Test
    logger.info(f"\n{'='*80}")
    logger.info("ÉVALUATION SUR TEST SET")
    logger.info('='*80)

    if args.model == 'mlp':
        model.eval()
        with torch.no_grad():
            y_test_pred = (model(torch.FloatTensor(X_test).to(device)).cpu() > 0.5).numpy()
    else:
        y_test_pred = model.predict(X_test)

    metrics_test = compute_metrics(Y_test.ravel(), y_test_pred.ravel())

    logger.info(f"\nTest Accuracy:  {metrics_test['accuracy']*100:.2f}%")
    logger.info(f"Test Precision: {metrics_test['precision']*100:.2f}%")
    logger.info(f"Test Recall:    {metrics_test['recall']*100:.2f}%")
    logger.info(f"Test F1:        {metrics_test['f1']*100:.2f}%")

    # Sauvegarder modèle et métriques
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f'meta_model_{args.model}.pkl'

    if args.model == 'mlp':
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {'input_size': 6, 'hidden_size': 32, 'dropout': 0.3}
        }, model_path)
    else:
        import joblib
        joblib.dump(model, model_path)

    logger.info(f"\n💾 Modèle sauvegardé: {model_path}")

    # Sauvegarder métriques
    metrics_path = output_dir / f'meta_model_{args.model}_metrics.json'
    metrics_all = {
        'train': {k: v for k, v in metrics_train.items() if k != 'confusion_matrix'},
        'val': {k: v for k, v in metrics_val.items() if k != 'confusion_matrix'},
        'test': {k: v for k, v in metrics_test.items() if k != 'confusion_matrix'}
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_all, f, indent=2)

    logger.info(f"💾 Métriques sauvegardées: {metrics_path}")

    logger.info("\n" + "="*80)
    logger.info("✅ TERMINÉ")
    logger.info("="*80)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
