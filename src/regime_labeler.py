"""
Module de labeling des régimes de marché (4 classes).

PRINCIPE CLÉ: Régimes basés sur Trend Strength (TS) × Volatility Cluster (VC)
===============================================================================

Calcule 4 régimes de marché basés sur deux dimensions:

**Trend Strength (TS)**: Force de la tendance (0-1)
  - Combinaison: MA slopes, ADX, regression R², Hurst exponent
  - TS > 0.6 = TREND
  - TS < 0.4 = RANGE
  - 0.4 ≤ TS ≤ 0.6 = Zone neutre (assigned to closest)

**Volatility Cluster (VC)**: Niveau de volatilité
  - Combinaison: ATR normalized, BB width, realized volatility
  - VC > 70th percentile = HIGH VOL
  - VC ≤ 70th percentile = LOW VOL

**4 Régimes (TS × VC)**:
  0: RANGE LOW VOL  (Range + Low Vol)
  1: RANGE HIGH VOL (Range + High Vol)
  2: TREND LOW VOL  (Trend + Low Vol)
  3: TREND HIGH VOL (Trend + High Vol)

Usage:
    from regime_labeler import calculate_regime_labels

    # Calculer les labels de régime
    regime_labels, ts_score, vc_score = calculate_regime_labels(df)

    # Ajouter au DataFrame
    df['regime'] = regime_labels
    df['trend_strength'] = ts_score
    df['volatility_cluster'] = vc_score

Requires:
    - DataFrame avec features de regime_features.py
    - Colonnes requises: ma20_slope, ma50_slope, adx, regression_r2, hurst_exponent
                         atr_normalized, bb_width, realized_volatility

Author: Claude Code
Date: 2025-01-11
Version: 1.0
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION DES POIDS
# =============================================================================

# Poids pour Trend Strength (TS)
# Total doit être = 1.0
TS_WEIGHTS = {
    'ma20_slope': 0.20,      # Tendance court terme
    'ma50_slope': 0.20,      # Tendance moyen terme
    'adx': 0.25,             # Force directionnelle (ADX key indicator)
    'regression_r2': 0.20,   # Qualité de la tendance linéaire
    'hurst_exponent': 0.15,  # Persistance vs mean-reversion
}

# Poids pour Volatility Cluster (VC)
# Total doit être = 1.0
VC_WEIGHTS = {
    'atr_normalized': 0.40,      # ATR/close (volatilité principale)
    'bb_width': 0.30,            # Largeur Bollinger Bands
    'realized_volatility': 0.30, # Volatilité réalisée annualisée
}

# Seuils de classification
TS_TREND_THRESHOLD = 0.6    # TS > 0.6 = TREND
TS_RANGE_THRESHOLD = 0.4    # TS < 0.4 = RANGE
VC_HIGH_PERCENTILE = 70     # VC > P70 = HIGH VOL


# =============================================================================
# NORMALISATION ET AGRÉGATION
# =============================================================================

def normalize_feature(values: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalise une feature entre 0 et 1.

    Args:
        values: Array de valeurs
        method: 'minmax' ou 'percentile'

    Returns:
        Normalized array (0-1)
    """
    if method == 'minmax':
        # Min-Max normalization
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        if vmax - vmin < 1e-10:
            return np.zeros_like(values)
        return (values - vmin) / (vmax - vmin)

    elif method == 'percentile':
        # Percentile-based (plus robuste aux outliers)
        p01 = np.nanpercentile(values, 1)
        p99 = np.nanpercentile(values, 99)
        if p99 - p01 < 1e-10:
            return np.zeros_like(values)
        normalized = (values - p01) / (p99 - p01)
        return np.clip(normalized, 0, 1)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def calculate_trend_strength(df: pd.DataFrame,
                              weights: dict = None,
                              normalize_method: str = 'percentile') -> np.ndarray:
    """
    Calcule Trend Strength (TS) score (0-1).

    Combine 5 features:
    - MA20 slope (normalized absolute value)
    - MA50 slope (normalized absolute value)
    - ADX (déjà 0-100, divisé par 100)
    - Regression R² (déjà 0-1)
    - Hurst exponent (transformé: |H - 0.5| × 2 pour avoir 0-1)

    Args:
        df: DataFrame avec features de régime
        weights: Dict des poids (défaut: TS_WEIGHTS)
        normalize_method: 'minmax' ou 'percentile'

    Returns:
        Array (n,) avec scores TS (0-1)
    """
    if weights is None:
        weights = TS_WEIGHTS

    # Vérifier que les colonnes existent
    required_cols = list(weights.keys())
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns for TS calculation: {missing_cols}")

    # Normaliser chaque composante
    ts_components = {}

    # MA slopes (prendre valeur absolue pour avoir force indépendamment de direction)
    ts_components['ma20_slope'] = normalize_feature(
        np.abs(df['ma20_slope'].values),
        method=normalize_method
    )
    ts_components['ma50_slope'] = normalize_feature(
        np.abs(df['ma50_slope'].values),
        method=normalize_method
    )

    # ADX (déjà 0-100, diviser par 100)
    ts_components['adx'] = np.clip(df['adx'].values / 100.0, 0, 1)

    # Regression R² (déjà 0-1)
    ts_components['regression_r2'] = np.clip(df['regression_r2'].values, 0, 1)

    # Hurst exponent: transformer pour avoir 0=mean-reverting, 1=trending
    # H < 0.5 = mean-reverting → TS faible
    # H > 0.5 = trending → TS fort
    # Transformation: |H - 0.5| × 2 donne score 0-1
    hurst = df['hurst_exponent'].values
    ts_components['hurst_exponent'] = np.abs(hurst - 0.5) * 2.0
    ts_components['hurst_exponent'] = np.clip(ts_components['hurst_exponent'], 0, 1)

    # Calculer score pondéré
    ts_score = np.zeros(len(df))
    for feature, weight in weights.items():
        ts_score += ts_components[feature] * weight

    # Clip final pour sécurité
    ts_score = np.clip(ts_score, 0, 1)

    # Remplacer NaN par 0 (cas où features manquantes)
    ts_score = np.nan_to_num(ts_score, nan=0.0)

    return ts_score


def calculate_volatility_cluster(df: pd.DataFrame,
                                   weights: dict = None,
                                   normalize_method: str = 'percentile') -> np.ndarray:
    """
    Calcule Volatility Cluster (VC) score (0-1).

    Combine 3 features:
    - ATR normalized (ATR/close)
    - BB width (largeur Bollinger Bands)
    - Realized volatility (annualisée)

    Args:
        df: DataFrame avec features de régime
        weights: Dict des poids (défaut: VC_WEIGHTS)
        normalize_method: 'minmax' ou 'percentile'

    Returns:
        Array (n,) avec scores VC (0-1)
    """
    if weights is None:
        weights = VC_WEIGHTS

    # Vérifier que les colonnes existent
    required_cols = list(weights.keys())
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns for VC calculation: {missing_cols}")

    # Normaliser chaque composante
    vc_components = {}

    # ATR normalized (déjà un ratio, normaliser)
    vc_components['atr_normalized'] = normalize_feature(
        df['atr_normalized'].values,
        method=normalize_method
    )

    # BB width (normaliser)
    vc_components['bb_width'] = normalize_feature(
        df['bb_width'].values,
        method=normalize_method
    )

    # Realized volatility (normaliser)
    vc_components['realized_volatility'] = normalize_feature(
        df['realized_volatility'].values,
        method=normalize_method
    )

    # Calculer score pondéré
    vc_score = np.zeros(len(df))
    for feature, weight in weights.items():
        vc_score += vc_components[feature] * weight

    # Clip final pour sécurité
    vc_score = np.clip(vc_score, 0, 1)

    # Remplacer NaN par 0
    vc_score = np.nan_to_num(vc_score, nan=0.0)

    return vc_score


# =============================================================================
# CLASSIFICATION DES RÉGIMES
# =============================================================================

def classify_regime(ts_score: np.ndarray,
                     vc_score: np.ndarray,
                     ts_trend_threshold: float = TS_TREND_THRESHOLD,
                     ts_range_threshold: float = TS_RANGE_THRESHOLD,
                     vc_high_percentile: int = VC_HIGH_PERCENTILE) -> np.ndarray:
    """
    Classifie chaque sample dans un des 4 régimes basé sur TS × VC.

    Régimes:
    - 0: RANGE LOW VOL  (TS < 0.4, VC ≤ P70)
    - 1: RANGE HIGH VOL (TS < 0.4, VC > P70)
    - 2: TREND LOW VOL  (TS > 0.6, VC ≤ P70)
    - 3: TREND HIGH VOL (TS > 0.6, VC > P70)

    Zone neutre (0.4 ≤ TS ≤ 0.6): Assigné au régime le plus proche.

    Args:
        ts_score: Trend Strength scores (0-1)
        vc_score: Volatility Cluster scores (0-1)
        ts_trend_threshold: Seuil pour TREND (défaut: 0.6)
        ts_range_threshold: Seuil pour RANGE (défaut: 0.4)
        vc_high_percentile: Percentile pour HIGH VOL (défaut: 70)

    Returns:
        Array (n,) avec labels 0-3
    """
    n_samples = len(ts_score)

    # Calculer seuil de volatilité (P70)
    vc_threshold = np.nanpercentile(vc_score, vc_high_percentile)

    # Initialiser labels
    regime_labels = np.zeros(n_samples, dtype=np.int8)

    # Classification binaire Trend/Range
    is_trend = ts_score > ts_trend_threshold
    is_range = ts_score < ts_range_threshold
    is_neutral = ~is_trend & ~is_range

    # Classification binaire High/Low Vol
    is_high_vol = vc_score > vc_threshold

    # Régime 0: RANGE LOW VOL
    mask_0 = is_range & ~is_high_vol
    regime_labels[mask_0] = 0

    # Régime 1: RANGE HIGH VOL
    mask_1 = is_range & is_high_vol
    regime_labels[mask_1] = 1

    # Régime 2: TREND LOW VOL
    mask_2 = is_trend & ~is_high_vol
    regime_labels[mask_2] = 2

    # Régime 3: TREND HIGH VOL
    mask_3 = is_trend & is_high_vol
    regime_labels[mask_3] = 3

    # Zone neutre (0.4 ≤ TS ≤ 0.6): Assigner au régime le plus proche
    if is_neutral.any():
        # Calculer distance à TREND (0.6) et RANGE (0.4)
        dist_to_trend = np.abs(ts_score[is_neutral] - ts_trend_threshold)
        dist_to_range = np.abs(ts_score[is_neutral] - ts_range_threshold)
        assign_as_trend = dist_to_trend < dist_to_range

        # Assigner selon volatilité
        neutral_labels = np.where(
            assign_as_trend,
            np.where(is_high_vol[is_neutral], 3, 2),  # TREND LOW/HIGH VOL
            np.where(is_high_vol[is_neutral], 1, 0)   # RANGE LOW/HIGH VOL
        )
        regime_labels[is_neutral] = neutral_labels

    return regime_labels


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def calculate_regime_labels(df: pd.DataFrame,
                              ts_weights: dict = None,
                              vc_weights: dict = None,
                              ts_trend_threshold: float = TS_TREND_THRESHOLD,
                              ts_range_threshold: float = TS_RANGE_THRESHOLD,
                              vc_high_percentile: int = VC_HIGH_PERCENTILE,
                              normalize_method: str = 'percentile') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule les labels de régime (0-3) pour chaque sample.

    Pipeline complet:
    1. Calculer Trend Strength (TS) score
    2. Calculer Volatility Cluster (VC) score
    3. Classifier en 4 régimes

    Args:
        df: DataFrame avec features de régime (de regime_features.py)
        ts_weights: Poids pour TS (défaut: TS_WEIGHTS)
        vc_weights: Poids pour VC (défaut: VC_WEIGHTS)
        ts_trend_threshold: Seuil TS pour TREND (défaut: 0.6)
        ts_range_threshold: Seuil TS pour RANGE (défaut: 0.4)
        vc_high_percentile: Percentile pour HIGH VOL (défaut: 70)
        normalize_method: 'minmax' ou 'percentile' (défaut: 'percentile')

    Returns:
        Tuple (regime_labels, ts_score, vc_score):
        - regime_labels: Array (n,) avec labels 0-3
        - ts_score: Array (n,) avec Trend Strength scores (0-1)
        - vc_score: Array (n,) avec Volatility Cluster scores (0-1)

    Example:
        >>> from regime_features import calculate_all_regime_features
        >>> from regime_labeler import calculate_regime_labels
        >>>
        >>> # Calculer features
        >>> df = calculate_all_regime_features(df)
        >>>
        >>> # Calculer labels
        >>> regime_labels, ts_score, vc_score = calculate_regime_labels(df)
        >>>
        >>> # Ajouter au DataFrame
        >>> df['regime'] = regime_labels
        >>> df['trend_strength'] = ts_score
        >>> df['volatility_cluster'] = vc_score
        >>>
        >>> # Statistiques
        >>> print(f"Regime distribution:")
        >>> print(pd.Series(regime_labels).value_counts().sort_index())
    """
    logger.info("  Calcul Trend Strength (TS)...")
    ts_score = calculate_trend_strength(
        df,
        weights=ts_weights,
        normalize_method=normalize_method
    )

    logger.info("  Calcul Volatility Cluster (VC)...")
    vc_score = calculate_volatility_cluster(
        df,
        weights=vc_weights,
        normalize_method=normalize_method
    )

    logger.info("  Classification des régimes (4 classes)...")
    regime_labels = classify_regime(
        ts_score,
        vc_score,
        ts_trend_threshold=ts_trend_threshold,
        ts_range_threshold=ts_range_threshold,
        vc_high_percentile=vc_high_percentile
    )

    # Statistiques
    n_total = len(regime_labels)
    regime_counts = pd.Series(regime_labels).value_counts().sort_index()

    logger.info("  Distribution des régimes:")
    for regime_id, count in regime_counts.items():
        pct = (count / n_total) * 100
        regime_name = {
            0: "RANGE LOW VOL",
            1: "RANGE HIGH VOL",
            2: "TREND LOW VOL",
            3: "TREND HIGH VOL"
        }[regime_id]
        logger.info(f"    Régime {regime_id} ({regime_name}): {count}/{n_total} ({pct:.1f}%)")

    # Statistiques TS et VC
    ts_mean = np.mean(ts_score)
    vc_mean = np.mean(vc_score)
    vc_p70 = np.nanpercentile(vc_score, vc_high_percentile)

    logger.info(f"  Trend Strength - Moyenne: {ts_mean:.3f}")
    logger.info(f"  Volatility Cluster - Moyenne: {vc_mean:.3f}, P70: {vc_p70:.3f}")

    return regime_labels, ts_score, vc_score


# =============================================================================
# FONCTION UTILITAIRE - VALIDATION
# =============================================================================

def validate_regime_features(df: pd.DataFrame) -> bool:
    """
    Vérifie que le DataFrame contient toutes les features requises.

    Args:
        df: DataFrame à valider

    Returns:
        True si toutes les features sont présentes

    Raises:
        ValueError: Si des features manquent
    """
    # Features requises pour TS
    ts_required = list(TS_WEIGHTS.keys())

    # Features requises pour VC
    vc_required = list(VC_WEIGHTS.keys())

    all_required = ts_required + vc_required

    missing_cols = [col for col in all_required if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing {len(missing_cols)} features for regime labeling: {missing_cols}\n"
            f"Please run calculate_all_regime_features() first."
        )

    logger.info(f"✓ All {len(all_required)} required features present")
    return True


# =============================================================================
# MAIN - TESTS UNITAIRES
# =============================================================================

if __name__ == '__main__':
    """
    Tests unitaires du module regime_labeler.
    """
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("=" * 80)
    logger.info("TESTS UNITAIRES - regime_labeler.py")
    logger.info("=" * 80)

    # Test 1: Créer un DataFrame de test
    logger.info("\nTest 1: Création DataFrame synthétique")
    n_samples = 1000

    # Features TS
    df_test = pd.DataFrame({
        'ma20_slope': np.random.randn(n_samples) * 0.001,
        'ma50_slope': np.random.randn(n_samples) * 0.0005,
        'adx': np.random.uniform(10, 50, n_samples),
        'regression_r2': np.random.uniform(0, 1, n_samples),
        'hurst_exponent': np.random.uniform(0.3, 0.7, n_samples),
        # Features VC
        'atr_normalized': np.random.uniform(0.005, 0.05, n_samples),
        'bb_width': np.random.uniform(0.01, 0.1, n_samples),
        'realized_volatility': np.random.uniform(0.1, 0.8, n_samples),
    })

    logger.info(f"  DataFrame créé: {df_test.shape}")

    # Test 2: Validation des features
    logger.info("\nTest 2: Validation des features")
    try:
        validate_regime_features(df_test)
        logger.info("  ✓ Validation réussie")
    except ValueError as e:
        logger.error(f"  ✗ Validation échouée: {e}")
        sys.exit(1)

    # Test 3: Calcul Trend Strength
    logger.info("\nTest 3: Calcul Trend Strength")
    ts_score = calculate_trend_strength(df_test)
    logger.info(f"  TS Score - Min: {ts_score.min():.3f}, Max: {ts_score.max():.3f}, Mean: {ts_score.mean():.3f}")
    assert ts_score.min() >= 0 and ts_score.max() <= 1, "TS score hors bornes [0,1]"
    logger.info("  ✓ TS Score dans [0,1]")

    # Test 4: Calcul Volatility Cluster
    logger.info("\nTest 4: Calcul Volatility Cluster")
    vc_score = calculate_volatility_cluster(df_test)
    logger.info(f"  VC Score - Min: {vc_score.min():.3f}, Max: {vc_score.max():.3f}, Mean: {vc_score.mean():.3f}")
    assert vc_score.min() >= 0 and vc_score.max() <= 1, "VC score hors bornes [0,1]"
    logger.info("  ✓ VC Score dans [0,1]")

    # Test 5: Classification des régimes
    logger.info("\nTest 5: Classification des régimes")
    regime_labels, ts, vc = calculate_regime_labels(df_test)

    # Vérifier que tous les labels sont dans [0,3]
    unique_labels = np.unique(regime_labels)
    logger.info(f"  Labels uniques: {unique_labels}")
    assert all(0 <= label <= 3 for label in unique_labels), "Labels hors bornes [0,3]"
    logger.info("  ✓ Tous les labels dans [0,3]")

    # Test 6: Distribution des régimes
    logger.info("\nTest 6: Distribution des régimes")
    regime_counts = pd.Series(regime_labels).value_counts().sort_index()
    logger.info(f"\n{regime_counts}")

    # Vérifier que les 4 régimes sont présents (au moins 5% chacun)
    for regime_id in range(4):
        if regime_id in regime_counts.index:
            pct = (regime_counts[regime_id] / n_samples) * 100
            logger.info(f"  Régime {regime_id}: {pct:.1f}%")
            assert pct >= 5.0, f"Régime {regime_id} sous-représenté (<5%)"
        else:
            logger.warning(f"  ⚠ Régime {regime_id} absent")

    logger.info("\n" + "=" * 80)
    logger.info("✓ TOUS LES TESTS RÉUSSIS")
    logger.info("=" * 80)
