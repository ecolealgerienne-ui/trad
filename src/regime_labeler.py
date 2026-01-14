"""
Module de labeling des régimes de marché (3 classes).

PRINCIPE CLÉ: Régimes basés sur Trend Strength (TS) × Volatility Cluster (VC)
===============================================================================

Calcule 3 régimes de marché basés sur deux dimensions.

**IMPORTANT - Correction 2026-01-12**:
Le régime "TREND LOW VOL" n'existe quasiment pas en crypto (0.1% des samples).
C'est un fait de microstructure documenté : crypto TREND = VOLATILITÉ.
Références: Oxford-Man Institute Realized Library, BIS Papers 2020.

**Trend Strength (TS)**: Force de la tendance (0-1)
  - Combinaison: MA slopes, ADX, regression R², Hurst exponent
  - TS > 0.5 = TREND
  - TS < 0.45 = RANGE (seuil relevé pour plus de sécurité)
  - 0.45 ≤ TS ≤ 0.5 = Zone neutre (assigned to closest)

**Volatility Cluster (VC)**: Niveau de volatilité
  - Combinaison: ATR normalized, BB width, realized volatility
  - VC > 50th percentile = HIGH VOL (pour RANGE seulement)
  - VC ≤ 50th percentile = LOW VOL (pour RANGE seulement)
  - TREND: volatilité ignorée (toujours élevée par définition)

**3 Régimes**:
  0: RANGE LOW VOL  (TS < 0.45 ET VC ≤ P50) - Marché inactif/dormant - NO TRADE
  1: RANGE HIGH VOL (TS < 0.45 ET VC > P50) - Chop violent, piège - PRUDENT
  2: TREND          (TS > 0.5, any vol)     - Seul régime exploitable - RECOMMANDÉ

Distribution attendue: Régime 0 (~35-45%), Régime 1 (~35-45%), Régime 2 (~15-25%)

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
Version: 2.0 - 3 régimes (suppression TREND LOW VOL inexistant en crypto)
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
# NOTE: Passage à 3 régimes le 2026-01-12 (suppression TREND LOW VOL)
# TREND LOW VOL n'existe pas en crypto : trend = volatilité (fait documenté)
# Le seuil VC est maintenant utilisé UNIQUEMENT pour discriminer RANGE LOW/HIGH VOL
# UPDATE 2026-01-14: Seuils ajustés pour plus de sécurité (filtrage plus strict)
TS_TREND_THRESHOLD = 0.5    # TS > 0.5 = TREND (any volatility)
TS_RANGE_THRESHOLD = 0.45   # TS < 0.45 = RANGE (augmenté de 0.4 pour plus de filtrage)
VC_LOW_PERCENTILE = 50      # Pour RANGE: VC ≤ P50 = LOW VOL (augmenté de 40 pour plus de filtrage)


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
                     vc_low_percentile: int = VC_LOW_PERCENTILE) -> np.ndarray:
    """
    Classifie chaque sample dans un des 3 régimes basé sur TS × VC.

    **IMPORTANT**: En crypto, TREND = VOLATILITÉ par nature.
    Le régime "TREND LOW VOL" n'existe pas statistiquement.

    Régimes:
    - 0: RANGE LOW VOL  (TS < 0.45 ET VC ≤ P50) - Marché inactif - NO TRADE
    - 1: RANGE HIGH VOL (TS < 0.45 ET VC > P50) - Chop violent, piège - PRUDENT
    - 2: TREND          (TS > 0.5, any vol)     - Seul régime exploitable - RECOMMANDÉ

    Zone neutre (0.45 ≤ TS ≤ 0.5): Assigné au régime le plus proche.

    Args:
        ts_score: Trend Strength scores (0-1)
        vc_score: Volatility Cluster scores (0-1)
        ts_trend_threshold: Seuil pour TREND (défaut: 0.5)
        ts_range_threshold: Seuil pour RANGE (défaut: 0.45)
        vc_low_percentile: Percentile pour LOW VOL dans RANGE (défaut: 50)

    Returns:
        Array (n,) avec labels 0-2
    """
    n_samples = len(ts_score)

    # Calculer seuil de volatilité (P40) pour discriminer RANGE LOW/HIGH VOL
    vc_threshold = np.nanpercentile(vc_score, vc_low_percentile)

    # Initialiser labels (défaut = 1 = RANGE HIGH VOL, le plus fréquent)
    regime_labels = np.ones(n_samples, dtype=np.int8)

    # Classification TS: Trend vs Range vs Neutre
    is_trend = ts_score > ts_trend_threshold
    is_range = ts_score < ts_range_threshold
    is_neutral = ~is_trend & ~is_range

    # Classification VC: Low Vol vs High Vol (SEULEMENT pour RANGE)
    is_low_vol = vc_score <= vc_threshold

    # Régime 2: TREND (TS > 0.5, any volatility)
    # En crypto, trend = volatilité, donc on ignore VC
    regime_labels[is_trend] = 2

    # Régime 0: RANGE LOW VOL (TS < 0.4 ET VC ≤ P40)
    mask_0 = is_range & is_low_vol
    regime_labels[mask_0] = 0

    # Régime 1: RANGE HIGH VOL (TS < 0.4 ET VC > P40)
    # Déjà initialisé à 1, mais on le force explicitement pour clarté
    mask_1 = is_range & ~is_low_vol
    regime_labels[mask_1] = 1

    # Zone neutre (0.4 ≤ TS ≤ 0.5): Assigner au régime le plus proche
    if is_neutral.any():
        # Calculer distance à TREND (0.5) et RANGE (0.4)
        dist_to_trend = np.abs(ts_score[is_neutral] - ts_trend_threshold)
        dist_to_range = np.abs(ts_score[is_neutral] - ts_range_threshold)
        assign_as_trend = dist_to_trend < dist_to_range

        # Assigner selon proximité
        # Si plus proche de TREND → Régime 2
        # Si plus proche de RANGE → Régime 0 ou 1 selon volatilité
        neutral_labels = np.where(
            assign_as_trend,
            2,  # TREND (any volatility)
            np.where(is_low_vol[is_neutral], 0, 1)  # RANGE LOW/HIGH VOL
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
                              vc_low_percentile: int = VC_LOW_PERCENTILE,
                              normalize_method: str = 'percentile') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule les labels de régime (0-2) pour chaque sample.

    Pipeline complet:
    1. Calculer Trend Strength (TS) score
    2. Calculer Volatility Cluster (VC) score
    3. Classifier en 3 régimes

    **IMPORTANT**: En crypto, TREND = VOLATILITÉ.
    Le régime "TREND LOW VOL" n'existe pas (fait documenté).

    Args:
        df: DataFrame avec features de régime (de regime_features.py)
        ts_weights: Poids pour TS (défaut: TS_WEIGHTS)
        vc_weights: Poids pour VC (défaut: VC_WEIGHTS)
        ts_trend_threshold: Seuil TS pour TREND (défaut: 0.5)
        ts_range_threshold: Seuil TS pour RANGE (défaut: 0.45)
        vc_low_percentile: Percentile pour LOW VOL dans RANGE (défaut: 50)
        normalize_method: 'minmax' ou 'percentile' (défaut: 'percentile')

    Returns:
        Tuple (regime_labels, ts_score, vc_score):
        - regime_labels: Array (n,) avec labels 0-2
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

    logger.info("  Classification des régimes (3 classes)...")
    regime_labels = classify_regime(
        ts_score,
        vc_score,
        ts_trend_threshold=ts_trend_threshold,
        ts_range_threshold=ts_range_threshold,
        vc_low_percentile=vc_low_percentile
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
            2: "TREND"
        }[regime_id]
        logger.info(f"    Régime {regime_id} ({regime_name}): {count}/{n_total} ({pct:.1f}%)")

    # Statistiques TS et VC
    ts_mean = np.mean(ts_score)
    vc_mean = np.mean(vc_score)
    vc_p40 = np.nanpercentile(vc_score, vc_low_percentile)

    logger.info(f"  Trend Strength - Moyenne: {ts_mean:.3f}")
    logger.info(f"  Volatility Cluster - Moyenne: {vc_mean:.3f}, P40: {vc_p40:.3f}")

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
    logger.info("\nTest 5: Classification des régimes (3 classes)")
    regime_labels, ts, vc = calculate_regime_labels(df_test)

    # Vérifier que tous les labels sont dans [0,2]
    unique_labels = np.unique(regime_labels)
    logger.info(f"  Labels uniques: {unique_labels}")
    assert all(0 <= label <= 2 for label in unique_labels), "Labels hors bornes [0,2]"
    logger.info("  ✓ Tous les labels dans [0,2]")

    # Test 6: Distribution des régimes
    logger.info("\nTest 6: Distribution des régimes (3 classes)")
    regime_counts = pd.Series(regime_labels).value_counts().sort_index()
    logger.info(f"\n{regime_counts}")

    # Vérifier que les 3 régimes sont présents (au moins 5% chacun)
    regime_names = {0: "RANGE LOW VOL", 1: "RANGE HIGH VOL", 2: "TREND"}
    for regime_id in range(3):
        if regime_id in regime_counts.index:
            pct = (regime_counts[regime_id] / n_samples) * 100
            logger.info(f"  Régime {regime_id} ({regime_names[regime_id]}): {pct:.1f}%")
            assert pct >= 5.0, f"Régime {regime_id} sous-représenté (<5%)"
        else:
            logger.warning(f"  ⚠ Régime {regime_id} ({regime_names[regime_id]}) absent")

    logger.info("\n" + "=" * 80)
    logger.info("✓ TOUS LES TESTS RÉUSSIS")
    logger.info("=" * 80)
