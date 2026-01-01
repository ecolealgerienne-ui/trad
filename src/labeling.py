"""
Labeling: Génération des labels de prédiction (cibles).

⚠️ CONCEPT CRITIQUE: Reconstruction de Signal avec Décalage Temporel

Le modèle à l'instant t doit prédire si la pente entre t-2 et t-1 est positive.

Workflow:
1. Calculer un signal (ex: RSI)
2. Appliquer filtre d'Octave (filtfilt) → Signal propre (utilise le futur!)
3. Calculer la pente: slope[t] = signal_filtered[t-1] - signal_filtered[t-2]
4. Label[t] = 1 si slope[t] > 0, sinon 0

C'est ce décalage qui permet au modèle d'apprendre sans data leakage.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict
import logging

# Import des fonctions locales
from filters import signal_filtfilt, calculate_slope
try:
    from indicators_ta import calculate_rsi_ta as calculate_rsi
except ImportError:
    # Fallback sur l'implémentation manuelle
    from indicators import calculate_rsi

logger = logging.getLogger(__name__)


def create_labels_from_filtered_signal(signal: Union[pd.Series, np.ndarray],
                                      smoothing: float = 0.25,
                                      offset: int = 1) -> Dict[str, np.ndarray]:
    """
    Crée les labels binaires (0/1) à partir d'un signal filtré.

    Workflow:
    1. Appliquer filtre d'Octave sur le signal
    2. Calculer la pente
    3. Appliquer le décalage temporel
    4. Convertir en labels binaires

    Args:
        signal: Signal d'entrée (RSI, Close, etc.)
        smoothing: Paramètre de lissage pour le filtre d'Octave
        offset: Décalage temporel (1 = prédire t-1 depuis t)

    Returns:
        Dictionnaire avec:
        - 'labels': Labels binaires (0/1)
        - 'filtered_signal': Signal filtré
        - 'slope': Pente du signal filtré
        - 'slope_shifted': Pente décalée (utilisée pour les labels)

    Example:
        >>> rsi = calculate_rsi(df['close'])
        >>> label_data = create_labels_from_filtered_signal(rsi, smoothing=0.25)
        >>> df['label'] = label_data['labels']
    """
    if isinstance(signal, pd.Series):
        signal_array = signal.values
    else:
        signal_array = np.array(signal)

    # 1. Appliquer le filtre signal_filtfilt (NON-CAUSAL)
    logger.info(f"Application du filtre signal_filtfilt (step={smoothing})...")
    filtered_signal = signal_filtfilt(signal_array, step=smoothing)

    # 2. Calculer la pente (différence)
    slope = calculate_slope(filtered_signal, periods=1)
    # slope[t] = filtered[t] - filtered[t-1]

    # 3. Appliquer le décalage temporel
    # On veut: label[t] prédit slope[t-offset]
    slope_shifted = np.roll(slope, offset)
    # Après shift: slope_shifted[t] = slope[t-1]

    # Marquer les premiers éléments comme NaN (car ils n'ont pas de valeur t-1)
    slope_shifted[:offset] = np.nan

    # 4. Convertir en labels binaires
    labels = np.full_like(slope_shifted, np.nan)
    valid_mask = ~np.isnan(slope_shifted)

    labels[valid_mask] = (slope_shifted[valid_mask] > 0).astype(int)
    # label = 1 si pente positive, 0 si négative

    # Statistiques
    n_labels = np.sum(~np.isnan(labels))
    n_positive = np.sum(labels == 1)
    n_negative = np.sum(labels == 0)

    logger.info(f"Labels créés: {n_labels} total")
    logger.info(f"  - Positifs (1): {n_positive} ({n_positive/n_labels*100:.1f}%)")
    logger.info(f"  - Négatifs (0): {n_negative} ({n_negative/n_labels*100:.1f}%)")

    # Vérifier l'équilibre des classes
    balance_ratio = n_positive / n_labels if n_labels > 0 else 0
    if balance_ratio < 0.3 or balance_ratio > 0.7:
        logger.warning(f"⚠️  Classes déséquilibrées! Ratio positifs: {balance_ratio:.2%}")

    return {
        'labels': labels,
        'filtered_signal': filtered_signal,
        'slope': slope,
        'slope_shifted': slope_shifted
    }


def create_labels_from_rsi(prices: Union[pd.Series, np.ndarray],
                          rsi_period: int = 14,
                          smoothing: float = 0.25,
                          offset: int = 1) -> Dict[str, np.ndarray]:
    """
    Pipeline complet: Prix → RSI → Filtre → Labels.

    Args:
        prices: Prix de clôture
        rsi_period: Période du RSI
        smoothing: Paramètre de lissage
        offset: Décalage temporel

    Returns:
        Dictionnaire avec labels et données intermédiaires

    Example:
        >>> label_data = create_labels_from_rsi(df['close'])
        >>> df['label'] = label_data['labels']
        >>> df['rsi'] = label_data['rsi']
        >>> df['rsi_filtered'] = label_data['filtered_signal']
    """
    logger.info(f"Création des labels depuis RSI (période={rsi_period})...")

    # 1. Calculer RSI
    rsi = calculate_rsi(prices, period=rsi_period)

    # 2. Créer les labels
    label_data = create_labels_from_filtered_signal(rsi, smoothing=smoothing, offset=offset)

    # Ajouter le RSI aux données retournées
    label_data['rsi'] = rsi

    return label_data


def create_labels_from_close(prices: Union[pd.Series, np.ndarray],
                            smoothing: float = 0.25,
                            offset: int = 1) -> Dict[str, np.ndarray]:
    """
    Pipeline: Prix → Filtre direct → Labels.

    Alternative: utiliser directement les prix au lieu du RSI.

    Args:
        prices: Prix de clôture
        smoothing: Paramètre de lissage
        offset: Décalage temporel

    Returns:
        Dictionnaire avec labels
    """
    logger.info("Création des labels depuis Close direct...")

    label_data = create_labels_from_filtered_signal(prices, smoothing=smoothing, offset=offset)

    return label_data


def create_multiclass_labels(slope: np.ndarray,
                            thresholds: tuple = (-0.5, 0.5)) -> np.ndarray:
    """
    Crée des labels multi-classes au lieu de binaires.

    Classes:
    - 0: Pente négative forte (< threshold_low)
    - 1: Pente neutre (entre thresholds)
    - 2: Pente positive forte (> threshold_high)

    Args:
        slope: Pente du signal
        thresholds: (threshold_low, threshold_high)

    Returns:
        Labels multi-classes (0, 1, 2)
    """
    threshold_low, threshold_high = thresholds

    labels = np.full_like(slope, np.nan)
    valid_mask = ~np.isnan(slope)

    labels[valid_mask & (slope < threshold_low)] = 0    # Baisse forte
    labels[valid_mask & ((slope >= threshold_low) & (slope <= threshold_high))] = 1  # Neutre
    labels[valid_mask & (slope > threshold_high)] = 2   # Hausse forte

    n_class0 = np.sum(labels == 0)
    n_class1 = np.sum(labels == 1)
    n_class2 = np.sum(labels == 2)
    n_total = n_class0 + n_class1 + n_class2

    logger.info(f"Labels multi-classes créés:")
    logger.info(f"  - Classe 0 (baisse): {n_class0} ({n_class0/n_total*100:.1f}%)")
    logger.info(f"  - Classe 1 (neutre): {n_class1} ({n_class1/n_total*100:.1f}%)")
    logger.info(f"  - Classe 2 (hausse): {n_class2} ({n_class2/n_total*100:.1f}%)")

    return labels


def validate_labels(labels: np.ndarray,
                   features_df: pd.DataFrame,
                   check_leakage: bool = True) -> Dict:
    """
    Valide la qualité des labels.

    Vérifications:
    1. Pas de NaN excessifs
    2. Équilibre des classes
    3. Pas de data leakage (si check_leakage=True)

    Args:
        labels: Labels générés
        features_df: DataFrame avec les features
        check_leakage: Si True, vérifie la corrélation future

    Returns:
        Dictionnaire de statistiques
    """
    stats = {}

    # 1. NaN
    n_nan = np.isnan(labels).sum()
    n_total = len(labels)
    pct_nan = n_nan / n_total * 100

    stats['n_total'] = n_total
    stats['n_nan'] = n_nan
    stats['pct_nan'] = pct_nan

    if pct_nan > 20:
        logger.warning(f"⚠️  {pct_nan:.1f}% de NaN dans les labels!")

    # 2. Distribution des classes
    unique, counts = np.unique(labels[~np.isnan(labels)], return_counts=True)
    stats['class_distribution'] = dict(zip(unique.astype(int), counts))

    logger.info(f"Distribution des classes: {stats['class_distribution']}")

    # 3. Équilibre
    if len(unique) == 2:  # Binaire
        n_positive = counts[unique == 1.0][0] if 1.0 in unique else 0
        n_negative = counts[unique == 0.0][0] if 0.0 in unique else 0
        n_valid = n_positive + n_negative

        balance = n_positive / n_valid if n_valid > 0 else 0
        stats['balance_ratio'] = balance

        if balance < 0.35 or balance > 0.65:
            logger.warning(f"⚠️  Classes déséquilibrées: {balance:.2%} positifs")
        else:
            logger.info(f"✅ Classes équilibrées: {balance:.2%} positifs")

    # 4. Data leakage (optionnel)
    if check_leakage and features_df is not None:
        logger.info("Vérification du data leakage...")

        # Créer une série temporaire pour les labels
        label_series = pd.Series(labels, index=features_df.index)

        # Corrélation avec features futures
        suspicious_cols = []
        for col in features_df.columns:
            if col not in ['timestamp', 'label']:
                # Corrélation entre feature[t] et label[t+1]
                future_corr = features_df[col].corr(label_series.shift(-1))

                if abs(future_corr) > 0.7:
                    suspicious_cols.append((col, future_corr))
                    logger.warning(f"⚠️  {col} corrélation {future_corr:.3f} avec label[t+1]")

        stats['suspicious_features'] = suspicious_cols

        if not suspicious_cols:
            logger.info("✅ Pas de data leakage détecté")

    return stats


def add_labels_to_dataframe(df: pd.DataFrame,
                           label_source: str = 'rsi',
                           rsi_period: int = 14,
                           smoothing: float = 0.25,
                           offset: int = 1,
                           validate: bool = True) -> pd.DataFrame:
    """
    Ajoute les labels au DataFrame.

    Args:
        df: DataFrame avec colonnes ['close'] (et 'rsi' si label_source='rsi')
        label_source: 'rsi' ou 'close'
        rsi_period: Période RSI (si label_source='rsi')
        smoothing: Paramètre de lissage
        offset: Décalage temporel
        validate: Si True, valide les labels

    Returns:
        DataFrame avec colonnes label ajoutées

    Example:
        >>> df = add_labels_to_dataframe(df, label_source='rsi', smoothing=0.25)
        >>> # Ajoute: label, rsi_filtered, slope
    """
    df = df.copy()

    logger.info(f"Ajout des labels au DataFrame (source={label_source})...")

    if label_source == 'rsi':
        label_data = create_labels_from_rsi(
            df['close'],
            rsi_period=rsi_period,
            smoothing=smoothing,
            offset=offset
        )
        df['rsi_filtered'] = label_data['filtered_signal']

    elif label_source == 'close':
        label_data = create_labels_from_close(
            df['close'],
            smoothing=smoothing,
            offset=offset
        )
        df['close_filtered'] = label_data['filtered_signal']

    else:
        raise ValueError(f"label_source invalide: {label_source}")

    # Ajouter les colonnes
    df['label'] = label_data['labels']
    df['slope'] = label_data['slope']
    df['slope_shifted'] = label_data['slope_shifted']

    # Validation
    if validate:
        stats = validate_labels(df['label'].values, df)
        logger.info(f"Validation terminée: {stats['pct_nan']:.1f}% NaN")

    # Supprimer les lignes avec label=NaN pour l'entraînement
    n_before = len(df)
    df_clean = df.dropna(subset=['label'])
    n_after = len(df_clean)

    logger.info(f"Lignes avec labels valides: {n_after}/{n_before} ({n_after/n_before*100:.1f}%)")

    return df
