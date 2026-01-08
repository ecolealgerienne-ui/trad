"""
Modèle CNN-LSTM Multi-Output pour Prédiction de Pente d'Indicateurs.

Architecture:
    Input: (batch, 12, 3) - 12 timesteps × 3 indicateurs
    → CNN (extraction features)
    → LayerNorm (stabilisation gradients)
    → LSTM (patterns temporels)
    → Dense partagé
    → 3 têtes de sortie indépendantes (RSI, CCI, MACD)
    Output: (batch, 3) - 3 probabilités binaires

Loss:
    BCEWithLogitsLoss moyenne sur les 3 outputs (poids égaux)

Optimisations:
    - BCEWithLogitsLoss pour stabilité numérique
    - LayerNorm entre CNN et LSTM pour réduire dérive de covariance

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
        2. LayerNorm pour stabiliser les gradients LSTM
        3. LSTM pour capturer patterns séquentiels
        4. Dense partagé
        5. 3 têtes de sortie (RSI, CCI, MACD)

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
        use_layer_norm: Activer LayerNorm entre CNN et LSTM (défaut: True)
                        Recommandé: True pour MACD, False pour RSI/CCI
        use_bce_with_logits: Utiliser BCEWithLogitsLoss (défaut: True)
                             Si True: forward() retourne logits bruts
                             Si False: forward() retourne probabilités via sigmoid
                             Recommandé: True pour MACD, False pour RSI/CCI
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
        dense_dropout: float = DENSE_DROPOUT,
        use_layer_norm: bool = True,
        use_bce_with_logits: bool = True
    ):
        super(MultiOutputCNNLSTM, self).__init__()

        self.sequence_length = sequence_length
        self.num_indicators = num_indicators
        self.num_outputs = num_outputs
        self.use_layer_norm = use_layer_norm
        self.use_bce_with_logits = use_bce_with_logits

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
        # Layer Normalization (entre CNN et LSTM) - OPTIONNEL
        # =====================================================================
        # Stabilise les gradients LSTM et réduit la dérive de covariance
        # Activé uniquement pour MACD (indicateurs volatils comme RSI/CCI n'en bénéficient pas)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(cnn_filters)
        else:
            self.layer_norm = None

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
        # Têtes de Sortie Indépendantes (num_outputs)
        # =====================================================================
        # Chaque tête prédit la pente d'un indicateur (0 ou 1)
        # Dynamique selon num_outputs (1 pour single-indicator, 3 pour multi)

        self.output_heads = nn.ModuleList([
            nn.Linear(dense_hidden_size, 1) for _ in range(num_outputs)
        ])

        layernorm_status = "avec LayerNorm" if use_layer_norm else "sans LayerNorm"
        logger.info(f"✅ Modèle CNN-LSTM créé ({layernorm_status}):")
        logger.info(f"  Input: ({sequence_length}, {num_indicators})")
        logger.info(f"  CNN: {cnn_filters} filters, kernel={cnn_kernel_size}")
        if use_layer_norm:
            logger.info(f"  LayerNorm: {cnn_filters} features (ACTIVÉ)")
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
        # Layer Normalization (stabilisation avant LSTM) - OPTIONNEL
        # =====================================================================
        if self.layer_norm is not None:
            x = self.layer_norm(x)  # Normalise sur la dimension features

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
        # Têtes de Sortie (num_outputs indépendants)
        # =====================================================================
        head_outputs = [head(x) for head in self.output_heads]

        # Concaténer les sorties: (batch, num_outputs)
        outputs = torch.cat(head_outputs, dim=1)

        # Appliquer sigmoid si BCELoss classique (baseline pour RSI/CCI)
        # Si BCEWithLogitsLoss (MACD), retourner logits bruts
        if not self.use_bce_with_logits:
            outputs = torch.sigmoid(outputs)  # Probabilités pour BCELoss

        return outputs

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prédiction de probabilités.

        Args:
            x: Input tensor (batch, sequence_length, num_indicators)

        Returns:
            Probabilités (batch, num_outputs)
        """
        outputs = self.forward(x)

        # Si use_bce_with_logits=True: forward() retourne logits, appliquer sigmoid
        # Si use_bce_with_logits=False: forward() retourne déjà probabilités
        if self.use_bce_with_logits:
            return torch.sigmoid(outputs)
        else:
            return outputs  # Déjà en [0, 1]

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


class MultiOutputBCEWithLogitsLoss(nn.Module):
    """
    Loss BCE avec logits multi-output avec poids optionnels pour chaque sortie.

    Utilise BCEWithLogitsLoss pour stabilité numérique (sigmoid intégré).
    Recommandé pour MACD uniquement.
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
        super(MultiOutputBCEWithLogitsLoss, self).__init__()

        # Poids par défaut selon le nombre d'outputs
        if weights is None:
            if num_outputs == 3:
                weights = (LOSS_WEIGHT_RSI, LOSS_WEIGHT_CCI, LOSS_WEIGHT_MACD)
            else:
                weights = tuple([1.0] * num_outputs)

        # Convertir en tensor
        self.weights = torch.tensor(weights[:num_outputs], dtype=torch.float32)
        self.num_outputs = num_outputs

        # BCEWithLogitsLoss pour stabilité numérique (sigmoid intégré dans la loss)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

        if num_outputs == 3:
            logger.info(f"✅ Loss multi-output créée (BCEWithLogitsLoss):")
            logger.info(f"  Poids: RSI={weights[0]}, CCI={weights[1]}, MACD={weights[2]}")
        else:
            logger.info(f"✅ Loss single-output créée (BCEWithLogitsLoss, poids={weights[0]})")

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


class MultiOutputBCELoss(nn.Module):
    """
    Loss BCE classique multi-output avec poids optionnels pour chaque sortie.

    Utilise BCELoss standard (modèle doit retourner probabilités via sigmoid).
    Recommandé pour CCI et RSI (baseline v7.0).
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

        # BCE classique (attend des probabilités [0,1])
        self.bce = nn.BCELoss(reduction='none')

        if num_outputs == 3:
            logger.info(f"✅ Loss multi-output créée (BCELoss baseline):")
            logger.info(f"  Poids: RSI={weights[0]}, CCI={weights[1]}, MACD={weights[2]}")
        else:
            logger.info(f"✅ Loss single-output créée (BCELoss baseline, poids={weights[0]})")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule la loss BCE moyenne pondérée sur les outputs.

        Args:
            predictions: Probabilités (batch, num_outputs) - DOIT être [0,1]
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


class WeightedTransitionBCELoss(nn.Module):
    """
    Loss BCE avec pondération augmentée sur les transitions.

    **Principe (Phase 2.11):**
    - Continuations (label[i] == label[i-1]): Poids = 1.0
    - Transitions (label[i] != label[i-1]): Poids = transition_weight (défaut: 5.0)

    **Objectif:**
    Forcer le modèle à mieux apprendre les retournements de marché (transitions),
    qui représentent seulement ~10% des samples mais sont critiques pour le trading.

    **Résultats attendus:**
    - Transition Accuracy: 52-58% → 75-80%
    - Gap 92% Global → 34% Win Rate réduit

    Args:
        num_outputs: Nombre de sorties (1 pour direction-only, 2 pour dual-binary)
        transition_weight: Poids pour les transitions (défaut: 5.0)
        use_bce_with_logits: Utiliser BCEWithLogitsLoss (True) ou BCELoss (False)
        output_weights: Poids optionnels par output (pour multi-output)
    """

    def __init__(
        self,
        num_outputs: int = 1,
        transition_weight: float = 5.0,
        use_bce_with_logits: bool = False,
        output_weights: Tuple[float, ...] = None
    ):
        super(WeightedTransitionBCELoss, self).__init__()

        self.num_outputs = num_outputs
        self.transition_weight = transition_weight
        self.use_bce_with_logits = use_bce_with_logits

        # Poids par output (pour multi-output)
        if output_weights is None:
            output_weights = tuple([1.0] * num_outputs)
        self.output_weights = torch.tensor(output_weights[:num_outputs], dtype=torch.float32)

        # BCE avec ou sans logits
        if use_bce_with_logits:
            self.bce = nn.BCEWithLogitsLoss(reduction='none')
            loss_type = "BCEWithLogitsLoss"
        else:
            self.bce = nn.BCELoss(reduction='none')
            loss_type = "BCELoss"

        logger.info(f"✅ Loss avec pondération transitions créée ({loss_type}):")
        logger.info(f"  Num outputs: {num_outputs}")
        logger.info(f"  Transition weight: {transition_weight}×")
        logger.info(f"  Continuations weight: 1.0×")
        if num_outputs > 1:
            logger.info(f"  Output weights: {output_weights}")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        is_transition: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Calcule la loss BCE avec pondération augmentée sur transitions.

        Args:
            predictions: Prédictions (batch, num_outputs)
                - Si use_bce_with_logits=True: logits bruts
                - Si use_bce_with_logits=False: probabilités [0,1]
            targets: Labels (batch, num_outputs)
            is_transition: Indicateur binaire transitions (batch,) - optionnel
                - 1.0 si transition (label[i] != label[i-1])
                - 0.0 si continuation (label[i] == label[i-1])
                - Si None: assume tous continuations (poids = 1.0)

        Returns:
            Loss scalaire (moyenne pondérée)
        """
        # Déplacer weights sur le même device
        if self.output_weights.device != predictions.device:
            self.output_weights = self.output_weights.to(predictions.device)

        # BCE pour chaque sample et output: (batch, num_outputs)
        bce_per_sample = self.bce(predictions, targets.float())

        # Calculer poids par sample: (batch, 1)
        if is_transition is not None:
            # is_transition: (batch,) → (batch, 1)
            is_transition = is_transition.view(-1, 1).float()
            # Poids: 1.0 + (transition_weight - 1.0) * is_transition
            # → Continuations: 1.0, Transitions: transition_weight
            sample_weights = 1.0 + (self.transition_weight - 1.0) * is_transition
        else:
            # Si pas fourni: assume tous continuations
            sample_weights = torch.ones((predictions.size(0), 1), device=predictions.device)

        # Appliquer poids par sample: (batch, num_outputs)
        weighted_bce = bce_per_sample * sample_weights

        # Moyenne sur batch: (num_outputs,)
        bce_mean = weighted_bce.mean(dim=0)

        # Pondération par output: scalaire
        weighted_loss = (bce_mean * self.output_weights).sum() / self.output_weights.sum()

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
    dense_dropout: float = DENSE_DROPOUT,
    use_layer_norm: bool = True,
    use_bce_with_logits: bool = True
) -> Tuple[MultiOutputCNNLSTM, nn.Module]:
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
        use_layer_norm: Activer LayerNorm (MACD: True, RSI/CCI: False)
        use_bce_with_logits: Utiliser BCEWithLogitsLoss (MACD: True, RSI/CCI: False)

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
        dense_dropout=dense_dropout,
        use_layer_norm=use_layer_norm,
        use_bce_with_logits=use_bce_with_logits
    )

    # Choisir la loss function selon l'indicateur
    if use_bce_with_logits:
        loss_fn = MultiOutputBCEWithLogitsLoss(num_outputs=num_outputs)
    else:
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
