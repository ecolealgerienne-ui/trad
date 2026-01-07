#!/usr/bin/env python3
"""
Structural Filters - Filtres Basés sur Conditions de Marché

Module pour filtrer les trades basé sur:
- ATR (Volatilité minimum)
- Volume Relatif (Liquidité minimum)
- ADX (Regime trending vs ranging)

Principe: Ne trader QUE dans les meilleures conditions pour maximiser edge/trade.

Objectif Phase 2.8:
- Réduire trades de 30,876 → 15,000 (-50%)
- Maintenir PnL Brut (+110%)
- Atteindre PnL Net POSITIF

Référence littérature:
- ATR: J. Welles Wilder (1978) - "New Concepts in Technical Trading Systems"
- Volume: Granville (1963) - "On Balance Volume"
- ADX: J. Welles Wilder (1978) - "Average Directional Index"
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

# Réutiliser les fonctions validées
from indicators_ta import calculate_atr_ta, calculate_adx_ta

logger = logging.getLogger(__name__)


# =============================================================================
# CALCUL INDICATEURS STRUCTURELS
# =============================================================================

def calculate_volume_ratio(
    volume: np.ndarray,
    window: int = 20
) -> np.ndarray:
    """
    Calcule ratio Volume actuel / Moyenne Mobile Volume.

    Volume relatif > 1.0 = liquidité au-dessus de la moyenne
    Volume relatif < 1.0 = liquidité en-dessous de la moyenne

    Args:
        volume: Volume brut
        window: Période MA (défaut: 20)

    Returns:
        Volume ratio (volume / MA(volume))

    Littérature:
        Granville (1963) - Volume confirme la tendance
    """
    # Calculer MA volume
    volume_ma = pd.Series(volume).rolling(window=window, min_periods=1).mean().values

    # Éviter division par zéro
    volume_ma = np.where(volume_ma == 0, 1e-8, volume_ma)

    # Ratio
    volume_ratio = volume / volume_ma

    return volume_ratio


def calculate_atr_normalized(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 14
) -> np.ndarray:
    """
    Calcule ATR normalisé par close (pour comparabilité entre assets).

    ATR_normalized = ATR / Close

    Args:
        high: Prix High
        low: Prix Low
        close: Prix Close
        window: Période ATR (défaut: 14)

    Returns:
        ATR normalisé (%)

    Littérature:
        Wilder (1978) - ATR mesure volatilité vraie
    """
    # Calculer ATR brut (réutilise fonction validée)
    atr = calculate_atr_ta(high, low, close, window=window, fillna=True)

    # Normaliser par close
    atr_norm = atr / close

    return atr_norm


def calculate_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 14
) -> np.ndarray:
    """
    Calcule ADX (Average Directional Index).

    ADX > 25: Trending market (momentum stratégies fonctionnent)
    ADX < 25: Ranging market (mean-reversion domine)

    Args:
        high: Prix High
        low: Prix Low
        close: Prix Close
        window: Période ADX (défaut: 14)

    Returns:
        ADX values

    Littérature:
        Wilder (1978) - ADX mesure force de la tendance
    """
    # Réutilise fonction validée
    adx_dict = calculate_adx_ta(high, low, close, window=window, fillna=True)

    return adx_dict['adx']


# =============================================================================
# APPLICATION FILTRES
# =============================================================================

def apply_atr_filter(
    atr_norm: np.ndarray,
    threshold: float = 0.01
) -> np.ndarray:
    """
    Filtre ATR: Ne trader QUE si volatilité suffisante.

    Logique:
    - atr_norm >= threshold → OK (marché volatil)
    - atr_norm < threshold → BLOCK (marché trop calme)

    Args:
        atr_norm: ATR normalisé (%)
        threshold: Seuil minimum (défaut: 1% = 0.01)

    Returns:
        Masque booléen (True = autorisé, False = bloqué)

    Impact attendu:
        - Réduction: -30% à -40% trades
        - Win Rate: +5-8%
        - PnL Brut: Maintenu (filtre périodes low-edge)
    """
    return atr_norm >= threshold


def apply_volume_filter(
    volume_ratio: np.ndarray,
    threshold: float = 0.8
) -> np.ndarray:
    """
    Filtre Volume: Ne trader QUE si liquidité suffisante.

    Logique:
    - volume_ratio >= threshold → OK (bonne liquidité)
    - volume_ratio < threshold → BLOCK (faible liquidité = slippage)

    Args:
        volume_ratio: Volume relatif (volume / MA)
        threshold: Seuil minimum (défaut: 0.8 = 80% de la MA)

    Returns:
        Masque booléen (True = autorisé, False = bloqué)

    Impact attendu:
        - Réduction: -15% à -25% trades
        - Slippage réduit
        - Execution quality améliorée
    """
    return volume_ratio >= threshold


def apply_adx_filter(
    adx: np.ndarray,
    threshold: float = 25.0
) -> np.ndarray:
    """
    Filtre ADX: Ne trader QUE en trending market.

    Logique:
    - adx >= threshold → OK (marché trending)
    - adx < threshold → BLOCK (marché ranging = mean-reversion)

    Args:
        adx: ADX values
        threshold: Seuil minimum (défaut: 25)

    Returns:
        Masque booléen (True = autorisé, False = bloqué)

    Impact attendu:
        - Réduction: -20% à -35% trades
        - Win Rate: +8-12% (meilleur sur trends)
        - PnL Brut: Potentiellement amélioré
    """
    return adx >= threshold


def apply_all_filters(
    atr_norm: Optional[np.ndarray] = None,
    volume_ratio: Optional[np.ndarray] = None,
    adx: Optional[np.ndarray] = None,
    atr_threshold: float = 0.01,
    volume_threshold: float = 0.8,
    adx_threshold: float = 25.0,
    enable_atr: bool = True,
    enable_volume: bool = True,
    enable_adx: bool = True
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Applique tous les filtres structurels.

    Args:
        atr_norm: ATR normalisé (optionnel)
        volume_ratio: Volume relatif (optionnel)
        adx: ADX values (optionnel)
        atr_threshold: Seuil ATR
        volume_threshold: Seuil Volume
        adx_threshold: Seuil ADX
        enable_atr: Activer filtre ATR
        enable_volume: Activer filtre Volume
        enable_adx: Activer filtre ADX

    Returns:
        (combined_mask, individual_masks)
        - combined_mask: Masque combiné (AND de tous les filtres actifs)
        - individual_masks: Dict avec chaque filtre séparément

    Logique combinée (AND):
        trade_allowed = atr_ok AND volume_ok AND adx_ok
    """
    n_samples = None

    # Détecter taille
    if atr_norm is not None:
        n_samples = len(atr_norm)
    elif volume_ratio is not None:
        n_samples = len(volume_ratio)
    elif adx is not None:
        n_samples = len(adx)
    else:
        raise ValueError("Au moins un indicateur doit être fourni")

    # Masques individuels
    masks = {}

    # Filtre ATR
    if enable_atr and atr_norm is not None:
        masks['atr'] = apply_atr_filter(atr_norm, atr_threshold)
        logger.debug(f"Filtre ATR: {masks['atr'].sum()}/{n_samples} autorisés ({masks['atr'].mean()*100:.1f}%)")
    else:
        masks['atr'] = np.ones(n_samples, dtype=bool)

    # Filtre Volume
    if enable_volume and volume_ratio is not None:
        masks['volume'] = apply_volume_filter(volume_ratio, volume_threshold)
        logger.debug(f"Filtre Volume: {masks['volume'].sum()}/{n_samples} autorisés ({masks['volume'].mean()*100:.1f}%)")
    else:
        masks['volume'] = np.ones(n_samples, dtype=bool)

    # Filtre ADX
    if enable_adx and adx is not None:
        masks['adx'] = apply_adx_filter(adx, adx_threshold)
        logger.debug(f"Filtre ADX: {masks['adx'].sum()}/{n_samples} autorisés ({masks['adx'].mean()*100:.1f}%)")
    else:
        masks['adx'] = np.ones(n_samples, dtype=bool)

    # Combinaison (AND)
    combined_mask = masks['atr'] & masks['volume'] & masks['adx']

    logger.info(f"Filtres combinés: {combined_mask.sum()}/{n_samples} autorisés ({combined_mask.mean()*100:.1f}%)")

    return combined_mask, masks


# =============================================================================
# CALCUL DEPUIS DATAFRAME OHLCV
# =============================================================================

def compute_structural_features(
    df: pd.DataFrame,
    atr_window: int = 14,
    volume_window: int = 20,
    adx_window: int = 14
) -> pd.DataFrame:
    """
    Calcule tous les indicateurs structurels depuis un DataFrame OHLCV.

    Args:
        df: DataFrame avec colonnes ['open', 'high', 'low', 'close', 'volume']
        atr_window: Période ATR
        volume_window: Période MA Volume
        adx_window: Période ADX

    Returns:
        DataFrame avec colonnes additionnelles:
        - atr_norm: ATR normalisé
        - volume_ratio: Volume relatif
        - adx: ADX

    Usage:
        df = pd.read_csv('data_trad/BTCUSD_all_5m.csv')
        df = compute_structural_features(df)
        atr_mask = apply_atr_filter(df['atr_norm'].values, 0.01)
    """
    logger.info(f"Calcul indicateurs structurels (ATR={atr_window}, Vol={volume_window}, ADX={adx_window})")

    # Vérifier colonnes requises
    required = ['high', 'low', 'close', 'volume']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    # Extraire arrays
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    volume = df['volume'].values

    # Calculer indicateurs
    df['atr_norm'] = calculate_atr_normalized(high, low, close, atr_window)
    df['volume_ratio'] = calculate_volume_ratio(volume, volume_window)
    df['adx'] = calculate_adx(high, low, close, adx_window)

    logger.info(f"✅ Indicateurs structurels calculés ({len(df)} samples)")
    logger.info(f"   ATR norm - Mean: {df['atr_norm'].mean():.4f}, Median: {df['atr_norm'].median():.4f}")
    logger.info(f"   Volume ratio - Mean: {df['volume_ratio'].mean():.2f}, Median: {df['volume_ratio'].median():.2f}")
    logger.info(f"   ADX - Mean: {df['adx'].mean():.1f}, Median: {df['adx'].median():.1f}")

    return df
