#!/usr/bin/env python3
"""
Entraînement du Meta-Modèle pour Stacking / Ensemble Learning

Objectif: Combiner les prédictions de 3 modèles experts (MACD, RSI, CCI)
pour améliorer la prédiction de Direction (Kalman original).

Cible: Direction Kalman (label original, pas de relabeling)

Hypothèse: Si le Stacking améliore l'Accuracy Direction (92% → 95-96%),
la rentabilité devrait suivre naturellement car on colle mieux au Kalman.

DONNÉES D'ENTRÉE:
    Les .npz doivent contenir Y_pred (prédictions des modèles):
    - dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_kalman.npz (Y_train_pred, Y_val_pred, Y_test_pred)
    - dataset_btc_eth_bnb_ada_ltc_rsi_dual_binary_kalman.npz (Y_train_pred, Y_val_pred, Y_test_pred)
    - dataset_btc_eth_bnb_ada_ltc_cci_dual_binary_kalman.npz (Y_train_pred, Y_val_pred, Y_test_pred)

    Si Y_pred manquant → Exécuter: python src/evaluate.py --data <dataset>

Modèles testés:
  1. Logistic Regression (baseline - RECOMMANDÉ)
  2. Random Forest (si non-linéaire)
  3. MLP (si très non-linéaire)

Usage:
  python src/train_stacking.py --model logistic
  python src/train_stacking.py --model rf
  python src/train_stacking.py --model mlp
"""

import sys
import numpy as np
from pathlib import Path
import logging
import argparse
from typing import Dict, Tuple

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


# =============================================================================
# CHARGEMENT DONNÉES
# =============================================================================

DATASET_PATHS = {
    'macd': 'data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_kalman.npz',
    'rsi': 'data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_dual_binary_kalman.npz',
    'cci': 'data/prepared/dataset_btc_eth_bnb_ada_ltc_cci_dual_binary_kalman.npz',
}


def load_predictions_from_npz(split: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge les prédictions des 3 modèles depuis les .npz.

    Args:
        split: 'train', 'val', ou 'test'

    Returns:
        X_meta: (n, 6) - Prédictions des 3 modèles [macd_dir, macd_force, rsi_dir, rsi_force, cci_dir, cci_force]
        Y_meta: (n, 1) - Direction Kalman (cible commune)
    """
    logger.info(f"\n📂 Chargement prédictions split '{split}'...")

    predictions = {}
    Y_meta = None

    for indicator in ['macd', 'rsi', 'cci']:
        path = DATASET_PATHS[indicator]

        if not Path(path).exists():
            raise FileNotFoundError(
                f"❌ Dataset introuvable: {path}\n"
                f"   Exécuter: python src/prepare_data_purified_dual_binary.py --assets BTC ETH BNB ADA LTC"
            )

        logger.info(f"   {indicator.upper()}...")
        data = np.load(path, allow_pickle=True)

        # Vérifier que Y_pred existe
        y_pred_key = f'Y_{split}_pred'
        if y_pred_key not in data:
            raise ValueError(
                f"❌ Prédictions manquantes dans {path}\n"
                f"   Clé manquante: {y_pred_key}\n"
                f"   Exécuter: python src/evaluate.py --data {path}"
            )

        Y_pred = data[y_pred_key]  # Shape: (n, 2) - [direction, force]
        Y = data[f'Y_{split}']     # Shape: (n, 2) - [direction, force]

        logger.info(f"      Y_pred shape: {Y_pred.shape}")

        predictions[indicator] = Y_pred

        # Utiliser Y du premier indicateur comme cible (tous identiques)
        if Y_meta is None:
            Y_meta = Y[:, 0:1]  # Direction uniquement (shape: n, 1)

    # Concaténer prédictions (6 features)
    X_meta = np.concatenate([
        predictions['macd'],  # (n, 2)
        predictions['rsi'],   # (n, 2)
        predictions['cci'],   # (n, 2)
    ], axis=1)  # (n, 6)

    logger.info(f"\n✅ Méta-features créées:")
    logger.info(f"   X_meta shape: {X_meta.shape}")
    logger.info(f"   Y_meta shape: {Y_meta.shape}")
    logger.info(f"   Features: [MACD_dir, MACD_force, RSI_dir, RSI_force, CCI_dir, CCI_force]")
    logger.info(f"   Cible: Direction Kalman")

    return X_meta, Y_meta


# =============================================================================
# MÉTRIQUES
# =============================================================================

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


# =============================================================================
# MODÈLES
# =============================================================================

def train_logistic_regression(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray
) -> Tuple[LogisticRegression, Dict, Dict, Dict]:
    """Entraîne une Régression Logistique."""
    logger.info("\n" + "="*80)
    logger.info("🎯 Modèle: Logistic Regression (Baseline)")
    logger.info("="*80)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )

    logger.info("\n⏳ Entraînement...")
    model.fit(X_train, Y_train.ravel())

    # Évaluation
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    metrics_train = compute_metrics(Y_train.ravel(), y_train_pred)
    metrics_val = compute_metrics(Y_val.ravel(), y_val_pred)
    metrics_test = compute_metrics(Y_test.ravel(), y_test_pred)

    logger.info(f"\n📊 Résultats:")
    logger.info(f"   Train Accuracy: {metrics_train['accuracy']*100:.2f}%")
    logger.info(f"   Val Accuracy:   {metrics_val['accuracy']*100:.2f}%")
    logger.info(f"   Test Accuracy:  {metrics_test['accuracy']*100:.2f}%")

    gap_train_val = abs(metrics_train['accuracy'] - metrics_val['accuracy']) * 100
    gap_val_test = abs(metrics_val['accuracy'] - metrics_test['accuracy']) * 100
    logger.info(f"\n   Gap Train/Val: {gap_train_val:.2f}%")
    logger.info(f"   Gap Val/Test:  {gap_val_test:.2f}%")

    # Poids des features (interprétabilité)
    logger.info(f"\n📈 Poids des features (interprétabilité):")
    feature_names = ['MACD_dir', 'MACD_force', 'RSI_dir', 'RSI_force', 'CCI_dir', 'CCI_force']
    for name, weight in zip(feature_names, model.coef_[0]):
        logger.info(f"     {name:12s}: {weight:+.4f}")

    return model, metrics_train, metrics_val, metrics_test


def train_random_forest(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray
) -> Tuple[RandomForestClassifier, Dict, Dict, Dict]:
    """Entraîne un Random Forest."""
    logger.info("\n" + "="*80)
    logger.info("🌲 Modèle: Random Forest (Non-Linéaire)")
    logger.info("="*80)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    logger.info("\n⏳ Entraînement...")
    model.fit(X_train, Y_train.ravel())

    # Évaluation
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    metrics_train = compute_metrics(Y_train.ravel(), y_train_pred)
    metrics_val = compute_metrics(Y_val.ravel(), y_val_pred)
    metrics_test = compute_metrics(Y_test.ravel(), y_test_pred)

    logger.info(f"\n📊 Résultats:")
    logger.info(f"   Train Accuracy: {metrics_train['accuracy']*100:.2f}%")
    logger.info(f"   Val Accuracy:   {metrics_val['accuracy']*100:.2f}%")
    logger.info(f"   Test Accuracy:  {metrics_test['accuracy']*100:.2f}%")

    gap_train_val = abs(metrics_train['accuracy'] - metrics_val['accuracy']) * 100
    gap_val_test = abs(metrics_val['accuracy'] - metrics_test['accuracy']) * 100
    logger.info(f"\n   Gap Train/Val: {gap_train_val:.2f}%")
    logger.info(f"   Gap Val/Test:  {gap_val_test:.2f}%")

    # Feature importance
    logger.info(f"\n📈 Feature Importance:")
    feature_names = ['MACD_dir', 'MACD_force', 'RSI_dir', 'RSI_force', 'CCI_dir', 'CCI_force']
    importances = sorted(zip(feature_names, model.feature_importances_),
                        key=lambda x: x[1], reverse=True)
    for name, importance in importances:
        logger.info(f"     {name:12s}: {importance:.4f}")

    return model, metrics_train, metrics_val, metrics_test


class SimpleMLP(nn.Module):
    """MLP simple pour méta-apprentissage."""
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
    X_test: np.ndarray,
    Y_test: np.ndarray,
    device: str = 'cpu',
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 0.001
) -> Tuple[SimpleMLP, Dict, Dict, Dict]:
    """Entraîne un MLP."""
    logger.info("\n" + "="*80)
    logger.info("🧠 Modèle: MLP (Deep Learning)")
    logger.info("="*80)

    # Préparation données
    X_train_t = torch.FloatTensor(X_train).to(device)
    Y_train_t = torch.FloatTensor(Y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    Y_val_t = torch.FloatTensor(Y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    Y_test_t = torch.FloatTensor(Y_test).to(device)

    train_dataset = TensorDataset(X_train_t, Y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Modèle
    model = SimpleMLP(input_size=6, hidden_size=32, dropout=0.3).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logger.info(f"\n⏳ Entraînement ({epochs} époques)...")

    best_val_acc = 0
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Évaluation
        model.eval()
        with torch.no_grad():
            y_val_pred = (model(X_val_t) > 0.5).float().cpu().numpy()
            val_acc = accuracy_score(Y_val.ravel(), y_val_pred.ravel())

        if (epoch + 1) % 10 == 0:
            logger.info(f"   Époque {epoch+1:3d}: Val Acc = {val_acc*100:.2f}%")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"   Early stopping à l'époque {epoch+1}")
                break

    # Évaluation finale
    model.eval()
    with torch.no_grad():
        y_train_pred = (model(X_train_t) > 0.5).float().cpu().numpy()
        y_val_pred = (model(X_val_t) > 0.5).float().cpu().numpy()
        y_test_pred = (model(X_test_t) > 0.5).float().cpu().numpy()

    metrics_train = compute_metrics(Y_train.ravel(), y_train_pred.ravel())
    metrics_val = compute_metrics(Y_val.ravel(), y_val_pred.ravel())
    metrics_test = compute_metrics(Y_test.ravel(), y_test_pred.ravel())

    logger.info(f"\n📊 Résultats:")
    logger.info(f"   Train Accuracy: {metrics_train['accuracy']*100:.2f}%")
    logger.info(f"   Val Accuracy:   {metrics_val['accuracy']*100:.2f}%")
    logger.info(f"   Test Accuracy:  {metrics_test['accuracy']*100:.2f}%")

    gap_train_val = abs(metrics_train['accuracy'] - metrics_val['accuracy']) * 100
    gap_val_test = abs(metrics_val['accuracy'] - metrics_test['accuracy']) * 100
    logger.info(f"\n   Gap Train/Val: {gap_train_val:.2f}%")
    logger.info(f"   Gap Val/Test:  {gap_val_test:.2f}%")

    return model, metrics_train, metrics_val, metrics_test


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Stacking - Entraînement Meta-Modèle')
    parser.add_argument(
        '--model',
        type=str,
        choices=['logistic', 'rf', 'mlp'],
        default='logistic',
        help="Modèle à entraîner (défaut: logistic)"
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help="Device pour MLP (défaut: cpu)"
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("🤖 STACKING - Entraînement Meta-Modèle")
    logger.info("="*80)
    logger.info(f"\n🎯 Objectif: Combiner MACD, RSI, CCI pour améliorer Direction")
    logger.info(f"📊 Cible: Direction Kalman (original, pas de relabeling)")
    logger.info(f"📈 Attendu: Accuracy 92% → 95-96%, Win Rate 14% → 55-65%")

    # Charger données
    logger.info("\n" + "="*80)
    logger.info("📂 CHARGEMENT DONNÉES")
    logger.info("="*80)

    X_train, Y_train = load_predictions_from_npz('train')
    X_val, Y_val = load_predictions_from_npz('val')
    X_test, Y_test = load_predictions_from_npz('test')

    # Entraîner modèle
    logger.info("\n" + "="*80)
    logger.info("⏳ ENTRAÎNEMENT")
    logger.info("="*80)

    if args.model == 'logistic':
        model, metrics_train, metrics_val, metrics_test = train_logistic_regression(
            X_train, Y_train, X_val, Y_val, X_test, Y_test
        )
    elif args.model == 'rf':
        model, metrics_train, metrics_val, metrics_test = train_random_forest(
            X_train, Y_train, X_val, Y_val, X_test, Y_test
        )
    elif args.model == 'mlp':
        model, metrics_train, metrics_val, metrics_test = train_mlp(
            X_train, Y_train, X_val, Y_val, X_test, Y_test,
            device=args.device
        )

    # Critères de succès
    logger.info("\n" + "="*80)
    logger.info("✅ CRITÈRES DE SUCCÈS")
    logger.info("="*80)

    test_acc = metrics_test['accuracy'] * 100
    gap_train_test = abs(metrics_train['accuracy'] - metrics_test['accuracy']) * 100

    success_criteria = {
        'Test Accuracy ≥ 95%': test_acc >= 95,
        'Gap Train/Test < 5%': gap_train_test < 5,
    }

    for criterion, passed in success_criteria.items():
        status = "✅" if passed else "❌"
        logger.info(f"   {status} {criterion}")

    all_passed = all(success_criteria.values())

    if all_passed:
        logger.info(f"\n🏆 SUCCÈS! Tous les critères passés!")
        logger.info(f"   → Prochaine étape: Backtest pour vérifier Win Rate > 50%")
    else:
        logger.info(f"\n⚠️  Critères non atteints. Diagnostiquer:")
        logger.info(f"   - Vérifier diversité des 3 modèles de base")
        logger.info(f"   - Tester avec d'autres features (volatilité, volume)")

    logger.info("\n" + "="*80)
    logger.info("🏁 FIN")
    logger.info("="*80)


if __name__ == '__main__':
    sys.exit(main())
