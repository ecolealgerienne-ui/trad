"""
Modèle CNN-LSTM Multi-Output pour Prédiction de Pente d'Indicateurs.

Architecture:
    Input: (batch, 12, 4) - 12 timesteps × 4 indicateurs
    → CNN (extraction features)
    → LSTM (patterns temporels)
    → Dense partagé
    → 4 têtes de sortie indépendantes (RSI, CCI, BOL, MACD)
    Output: (batch, 4) - 4 probabilités binaires

Loss:
    BCE moyenne sur les 4 outputs (poids égaux)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# Import constants
from constants import (
    SEQUENCE_LENGTH,
    NUM_INDICATORS,
    NUM_OUTPUTS,
    CNN_FILTERS,
    CNN_KERNEL_SIZE,
    CNN_STRIDE,
    CNN_PADDING,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    LSTM_DROPOUT,
    DENSE_HIDDEN_SIZE,
    DENSE_DROPOUT,
    LOSS_WEIGHT_RSI,
    LOSS_WEIGHT_CCI,
    LOSS_WEIGHT_BOL,
    LOSS_WEIGHT_MACD
)


class MultiOutputCNNLSTM(nn.Module):
    """
    Modèle CNN-LSTM avec 4 sorties indépendantes.

    Architecture:
        1. CNN 1D pour extraction de features temporelles
        2. LSTM pour capturer patterns séquentiels
        3. Dense partagé
        4. 4 têtes de sortie (une par indicateur)

    Args:
        sequence_length: Longueur des séquences (défaut: 12)
        num_indicators: Nombre d'indicateurs en input (défaut: 4)
        num_outputs: Nombre de sorties (défaut: 4)
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
        sequence_length: int = SEQUENCE_LENGTH,
        num_indicators: int = NUM_INDICATORS,
        num_outputs: int = NUM_OUTPUTS,
        cnn_filters: int = CNN_FILTERS,
        cnn_kernel_size: int = CNN_KERNEL_SIZE,
        lstm_hidden_size: int = LSTM_HIDDEN_SIZE,
        lstm_num_layers: int = LSTM_NUM_LAYERS,
        lstm_dropout: float = LSTM_DROPOUT,
        dense_hidden_size: int = DENSE_HIDDEN_SIZE,
        dense_dropout: float = DENSE_DROPOUT
    ):
        super(MultiOutputCNNLSTM, self).__init__()

        self.sequence_length = sequence_length
        self.num_indicators = num_indicators
        self.num_outputs = num_outputs

        # =====================================================================
        # CNN Layer (1D Convolution sur dimension temporelle)
        # =====================================================================
        # Input: (batch, sequence_length, num_indicators)
        # Conv1d attend: (batch, channels, length)
        # On doit donc transposer: (batch, num_indicators, sequence_length)

        self.cnn = nn.Conv1d(
            in_channels=num_indicators,
            out_channels=cnn_filters,
            kernel_size=cnn_kernel_size,
            stride=CNN_STRIDE,
            padding=CNN_PADDING
        )
        self.cnn_activation = nn.ReLU()
        self.cnn_batchnorm = nn.BatchNorm1d(cnn_filters)

        # Calculer longueur après CNN (avec padding='same', longueur préservée)
        cnn_output_length = sequence_length

        # =====================================================================
        # LSTM Layer
        # =====================================================================
        # Input: (batch, seq_len, features) = (batch, cnn_output_length, cnn_filters)

        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )

        # =====================================================================
        # Dense Partagé
        # =====================================================================
        # On prend le dernier hidden state du LSTM
        # Output LSTM: (batch, seq_len, lstm_hidden_size)
        # On prend [:, -1, :] → (batch, lstm_hidden_size)

        self.dense_shared = nn.Linear(lstm_hidden_size, dense_hidden_size)
        self.dense_activation = nn.ReLU()
        self.dense_dropout = nn.Dropout(dense_dropout)

        # =====================================================================
        # Têtes de Sortie Indépendantes (4 outputs)
        # =====================================================================
        # Chaque tête prédit la pente d'un indicateur (0 ou 1)

        self.head_rsi = nn.Linear(dense_hidden_size, 1)
        self.head_cci = nn.Linear(dense_hidden_size, 1)
        self.head_bol = nn.Linear(dense_hidden_size, 1)
        self.head_macd = nn.Linear(dense_hidden_size, 1)

        logger.info("✅ Modèle CNN-LSTM créé:")
        logger.info(f"  Input: ({sequence_length}, {num_indicators})")
        logger.info(f"  CNN: {cnn_filters} filters, kernel={cnn_kernel_size}")
        logger.info(f"  LSTM: {lstm_hidden_size} hidden × {lstm_num_layers} layers")
        logger.info(f"  Dense: {dense_hidden_size}")
        logger.info(f"  Outputs: {num_outputs}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass du modèle.

        Args:
            x: Input tensor (batch, sequence_length, num_indicators)

        Returns:
            Output tensor (batch, num_outputs) avec probabilités pour chaque indicateur
        """
        # Input: (batch, sequence_length, num_indicators)
        batch_size = x.size(0)

        # =====================================================================
        # CNN
        # =====================================================================
        # Transposer pour Conv1d: (batch, num_indicators, sequence_length)
        x = x.transpose(1, 2)

        # Conv1d
        x = self.cnn(x)  # (batch, cnn_filters, sequence_length)
        x = self.cnn_activation(x)
        x = self.cnn_batchnorm(x)

        # Retransposer pour LSTM: (batch, sequence_length, cnn_filters)
        x = x.transpose(1, 2)

        # =====================================================================
        # LSTM
        # =====================================================================
        # x: (batch, sequence_length, cnn_filters)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Prendre le dernier timestep
        # lstm_out: (batch, sequence_length, lstm_hidden_size)
        x = lstm_out[:, -1, :]  # (batch, lstm_hidden_size)

        # =====================================================================
        # Dense Partagé
        # =====================================================================
        x = self.dense_shared(x)  # (batch, dense_hidden_size)
        x = self.dense_activation(x)
        x = self.dense_dropout(x)

        # =====================================================================
        # Têtes de Sortie (4 outputs indépendants)
        # =====================================================================
        # Chaque tête produit une logit → Sigmoid → probabilité

        out_rsi = torch.sigmoid(self.head_rsi(x))    # (batch, 1)
        out_cci = torch.sigmoid(self.head_cci(x))    # (batch, 1)
        out_bol = torch.sigmoid(self.head_bol(x))    # (batch, 1)
        out_macd = torch.sigmoid(self.head_macd(x))  # (batch, 1)

        # Concaténer les 4 sorties
        outputs = torch.cat([out_rsi, out_cci, out_bol, out_macd], dim=1)  # (batch, 4)

        return outputs

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prédiction de probabilités (identique à forward).

        Args:
            x: Input tensor (batch, sequence_length, num_indicators)

        Returns:
            Probabilités (batch, num_outputs)
        """
        return self.forward(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Prédiction de classes binaires (0 ou 1).

        Args:
            x: Input tensor (batch, sequence_length, num_indicators)
            threshold: Seuil de décision (défaut: 0.5)

        Returns:
            Prédictions binaires (batch, num_outputs)
        """
        probs = self.predict_proba(x)
        predictions = (probs >= threshold).long()
        return predictions


class MultiOutputBCELoss(nn.Module):
    """
    Loss BCE multi-output avec poids optionnels pour chaque sortie.

    Calcule la BCE pour chaque output et fait la moyenne pondérée.

    Args:
        weights: Poids pour chaque output [RSI, CCI, BOL, MACD] (défaut: égaux)
    """

    def __init__(
        self,
        weights: Tuple[float, float, float, float] = (
            LOSS_WEIGHT_RSI,
            LOSS_WEIGHT_CCI,
            LOSS_WEIGHT_BOL,
            LOSS_WEIGHT_MACD
        )
    ):
        super(MultiOutputBCELoss, self).__init__()

        # Convertir en tensor
        self.weights = torch.tensor(weights, dtype=torch.float32)

        # BCE sans reduction (on veut calculer séparément pour chaque output)
        self.bce = nn.BCELoss(reduction='none')

        logger.info(f"✅ Loss multi-output créée:")
        logger.info(f"  Poids: RSI={weights[0]}, CCI={weights[1]}, "
                   f"BOL={weights[2]}, MACD={weights[3]}")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule la loss BCE moyenne pondérée sur les 4 outputs.

        Args:
            predictions: Prédictions (batch, 4)
            targets: Labels (batch, 4)

        Returns:
            Loss scalaire (moyenne pondérée)
        """
        # Déplacer weights sur le même device que predictions
        if self.weights.device != predictions.device:
            self.weights = self.weights.to(predictions.device)

        # BCE pour chaque output: (batch, 4)
        bce_per_output = self.bce(predictions, targets.float())

        # Moyenne sur batch: (4,)
        bce_mean = bce_per_output.mean(dim=0)

        # Pondération: scalaire
        weighted_loss = (bce_mean * self.weights).sum() / self.weights.sum()

        return weighted_loss


def create_model(
    device: str = 'cpu',
    cnn_filters: int = CNN_FILTERS,
    lstm_hidden_size: int = LSTM_HIDDEN_SIZE,
    lstm_num_layers: int = LSTM_NUM_LAYERS,
    lstm_dropout: float = LSTM_DROPOUT,
    dense_hidden_size: int = DENSE_HIDDEN_SIZE,
    dense_dropout: float = DENSE_DROPOUT
) -> Tuple[MultiOutputCNNLSTM, MultiOutputBCELoss]:
    """
    Factory function pour créer le modèle et la loss.

    Args:
        device: Device ('cpu' ou 'cuda')
        cnn_filters: Nombre de filtres CNN
        lstm_hidden_size: Taille hidden LSTM
        lstm_num_layers: Nombre de couches LSTM
        lstm_dropout: Dropout LSTM
        dense_hidden_size: Taille couche dense
        dense_dropout: Dropout dense

    Returns:
        (model, loss_fn)
    """
    model = MultiOutputCNNLSTM(
        cnn_filters=cnn_filters,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        lstm_dropout=lstm_dropout,
        dense_hidden_size=dense_hidden_size,
        dense_dropout=dense_dropout
    )
    loss_fn = MultiOutputBCELoss()

    # Déplacer sur device
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    # Compter paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"\n📊 Paramètres du modèle:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,}")
    logger.info(f"  Device: {device}")

    return model, loss_fn


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calcule les métriques de classification pour multi-output.

    Args:
        predictions: Probabilités (batch, 4)
        targets: Labels (batch, 4)
        threshold: Seuil de décision

    Returns:
        Dictionnaire avec métriques par output + moyennes
    """
    # Convertir probabilités en prédictions binaires
    preds_binary = (predictions >= threshold).float()
    targets_float = targets.float()

    metrics = {}

    indicator_names = ['RSI', 'CCI', 'BOL', 'MACD']

    for i, name in enumerate(indicator_names):
        # Extraire prédictions et targets pour cet indicateur
        pred = preds_binary[:, i]
        target = targets_float[:, i]

        # True Positives, False Positives, etc.
        tp = ((pred == 1) & (target == 1)).sum().item()
        tn = ((pred == 0) & (target == 0)).sum().item()
        fp = ((pred == 1) & (target == 0)).sum().item()
        fn = ((pred == 0) & (target == 1)).sum().item()

        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f'{name}_accuracy'] = accuracy
        metrics[f'{name}_precision'] = precision
        metrics[f'{name}_recall'] = recall
        metrics[f'{name}_f1'] = f1

    # Moyennes
    metrics['avg_accuracy'] = sum(metrics[f'{n}_accuracy'] for n in indicator_names) / 4
    metrics['avg_precision'] = sum(metrics[f'{n}_precision'] for n in indicator_names) / 4
    metrics['avg_recall'] = sum(metrics[f'{n}_recall'] for n in indicator_names) / 4
    metrics['avg_f1'] = sum(metrics[f'{n}_f1'] for n in indicator_names) / 4

    return metrics


# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

if __name__ == '__main__':
    # Configurer logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    logger.info("="*80)
    logger.info("TEST DU MODÈLE CNN-LSTM")
    logger.info("="*80)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"\nDevice: {device}")

    # Créer modèle
    logger.info("\n1. Création du modèle...")
    model, loss_fn = create_model(device=device)

    # Test forward pass
    logger.info("\n2. Test forward pass...")
    batch_size = 32
    x_dummy = torch.randn(batch_size, SEQUENCE_LENGTH, NUM_INDICATORS).to(device)
    y_dummy = torch.randint(0, 2, (batch_size, NUM_OUTPUTS)).to(device)

    logger.info(f"  Input: {x_dummy.shape}")
    logger.info(f"  Target: {y_dummy.shape}")

    # Forward
    with torch.no_grad():
        output = model(x_dummy)

    logger.info(f"  Output: {output.shape}")
    logger.info(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Test loss
    logger.info("\n3. Test loss...")
    loss = loss_fn(output, y_dummy.float())
    logger.info(f"  Loss: {loss.item():.4f}")

    # Test metrics
    logger.info("\n4. Test metrics...")
    metrics = compute_metrics(output, y_dummy)

    logger.info("  Métriques par indicateur:")
    for name in ['RSI', 'CCI', 'BOL', 'MACD']:
        logger.info(f"    {name}: Acc={metrics[f'{name}_accuracy']:.3f}, "
                   f"F1={metrics[f'{name}_f1']:.3f}")

    logger.info(f"\n  Moyennes:")
    logger.info(f"    Accuracy: {metrics['avg_accuracy']:.3f}")
    logger.info(f"    F1: {metrics['avg_f1']:.3f}")

    logger.info("\n" + "="*80)
    logger.info("✅ Modèle opérationnel!")
    logger.info("="*80)
