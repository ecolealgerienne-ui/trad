"""
Modèle CNN-LSTM Multi-Output pour Prédiction de Pente d'Indicateurs.

Architecture:
    Input: (batch, 12, 3) - 12 timesteps × 3 indicateurs
    → CNN (extraction features)
    → LSTM (patterns temporels)
    → Dense partagé
    → 3 têtes de sortie indépendantes (RSI, CCI, MACD)
    Output: (batch, 3) - 3 probabilités binaires

Loss:
    BCE moyenne sur les 3 outputs (poids égaux)

Note:
    BOL (Bollinger Bands) a été retiré car impossible à synchroniser
    avec les autres indicateurs (toujours lag +1).
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
    LOSS_WEIGHT_MACD
)


class MultiOutputCNNLSTM(nn.Module):
    """
    Modèle CNN-LSTM avec 3 sorties indépendantes.

    Architecture:
        1. CNN 1D pour extraction de features temporelles
        2. LSTM pour capturer patterns séquentiels
        3. Dense partagé
        4. 3 têtes de sortie (RSI, CCI, MACD)

    Args:
        sequence_length: Longueur des séquences (défaut: 12)
        num_indicators: Nombre d'indicateurs en input (défaut: 3)
        num_outputs: Nombre de sorties (défaut: 3)
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
        # Têtes de Sortie Indépendantes (3 outputs)
        # =====================================================================
        # Chaque tête prédit la pente d'un indicateur (0 ou 1)
        # Note: BOL retiré car non synchronisable

        self.head_rsi = nn.Linear(dense_hidden_size, 1)
        self.head_cci = nn.Linear(dense_hidden_size, 1)
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
        # Têtes de Sortie (3 outputs indépendants)
        # =====================================================================
        # Chaque tête produit une logit → Sigmoid → probabilité

        out_rsi = torch.sigmoid(self.head_rsi(x))    # (batch, 1)
        out_cci = torch.sigmoid(self.head_cci(x))    # (batch, 1)
        out_macd = torch.sigmoid(self.head_macd(x))  # (batch, 1)

        # Concaténer les 3 sorties
        outputs = torch.cat([out_rsi, out_cci, out_macd], dim=1)  # (batch, 3)

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
        num_outputs: Nombre de sorties (1 pour single-indicator, 3 pour multi)
        weights: Poids pour chaque output (défaut: égaux)
    """

    def __init__(
        self,
        num_outputs: int = 3,
        weights: Tuple[float, ...] = None
    ):
        super(MultiOutputBCELoss, self).__init__()

        # Poids par défaut selon le nombre d'outputs
        if weights is None:
            if num_outputs == 3:
                weights = (LOSS_WEIGHT_RSI, LOSS_WEIGHT_CCI, LOSS_WEIGHT_MACD)
            else:
                weights = tuple([1.0] * num_outputs)

        # Convertir en tensor
        self.weights = torch.tensor(weights[:num_outputs], dtype=torch.float32)
        self.num_outputs = num_outputs

        # BCE sans reduction (on veut calculer séparément pour chaque output)
        self.bce = nn.BCELoss(reduction='none')

        if num_outputs == 3:
            logger.info(f"✅ Loss multi-output créée:")
            logger.info(f"  Poids: RSI={weights[0]}, CCI={weights[1]}, MACD={weights[2]}")
        else:
            logger.info(f"✅ Loss single-output créée (poids={weights[0]})")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule la loss BCE moyenne pondérée sur les outputs.

        Args:
            predictions: Prédictions (batch, num_outputs)
            targets: Labels (batch, num_outputs)

        Returns:
            Loss scalaire (moyenne pondérée)
        """
        # Déplacer weights sur le même device que predictions
        if self.weights.device != predictions.device:
            self.weights = self.weights.to(predictions.device)

        # BCE pour chaque output: (batch, num_outputs)
        bce_per_output = self.bce(predictions, targets.float())

        # Moyenne sur batch: (num_outputs,)
        bce_mean = bce_per_output.mean(dim=0)

        # Pondération: scalaire
        weighted_loss = (bce_mean * self.weights).sum() / self.weights.sum()

        return weighted_loss


def create_model(
    device: str = 'cpu',
    num_indicators: int = NUM_INDICATORS,
    num_outputs: int = NUM_OUTPUTS,
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
        num_indicators: Nombre de features en entrée (défaut: 3)
        num_outputs: Nombre de sorties/indicateurs à prédire (défaut: 3)
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
        num_indicators=num_indicators,
        num_outputs=num_outputs,
        cnn_filters=cnn_filters,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        lstm_dropout=lstm_dropout,
        dense_hidden_size=dense_hidden_size,
        dense_dropout=dense_dropout
    )
    loss_fn = MultiOutputBCELoss(num_outputs=num_outputs)

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
    threshold: float = 0.5,
    indicator_names: list = None
) -> Dict[str, float]:
    """
    Calcule les métriques de classification pour multi-output.

    Args:
        predictions: Probabilités (batch, num_outputs)
        targets: Labels (batch, num_outputs)
        threshold: Seuil de décision
        indicator_names: Liste des noms d'indicateurs (auto-détecté si None)

    Returns:
        Dictionnaire avec métriques par output + moyennes
    """
    # Convertir probabilités en prédictions binaires
    preds_binary = (predictions >= threshold).float()
    targets_float = targets.float()

    metrics = {}

    # Détecter le nombre d'outputs
    num_outputs = predictions.shape[1]

    # Noms par défaut selon le nombre d'outputs
    if indicator_names is None:
        if num_outputs == 3:
            indicator_names = ['RSI', 'CCI', 'MACD']
        elif num_outputs == 1:
            indicator_names = ['INDICATOR']  # Sera remplacé par le vrai nom dans l'affichage
        else:
            indicator_names = [f'OUT_{i}' for i in range(num_outputs)]

    for i, name in enumerate(indicator_names):
        if i >= num_outputs:
            break

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

    # Moyennes (sur les indicateurs actifs)
    active_names = indicator_names[:num_outputs]
    metrics['avg_accuracy'] = sum(metrics[f'{n}_accuracy'] for n in active_names) / len(active_names)
    metrics['avg_precision'] = sum(metrics[f'{n}_precision'] for n in active_names) / len(active_names)
    metrics['avg_recall'] = sum(metrics[f'{n}_recall'] for n in active_names) / len(active_names)
    metrics['avg_f1'] = sum(metrics[f'{n}_f1'] for n in active_names) / len(active_names)

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
    for name in ['RSI', 'CCI', 'MACD']:
        logger.info(f"    {name}: Acc={metrics[f'{name}_accuracy']:.3f}, "
                   f"F1={metrics[f'{name}_f1']:.3f}")

    logger.info(f"\n  Moyennes:")
    logger.info(f"    Accuracy: {metrics['avg_accuracy']:.3f}")
    logger.info(f"    F1: {metrics['avg_f1']:.3f}")

    logger.info("\n" + "="*80)
    logger.info("✅ Modèle opérationnel!")
    logger.info("="*80)
