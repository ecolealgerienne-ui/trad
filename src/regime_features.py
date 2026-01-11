"""
Features pour la détection de régimes de marché (Meta-Regime Trading).

Toutes les features sont CAUSALES (n'utilisent que le passé).
Elles peuvent être utilisées comme inputs pour le Regime Classifier.

Features implémentées (~50 total):
- Trend Features: MA slopes, ADX, Regression R², Hurst exponent, MACD histogram
- Volatility Features: ATR, BB width, %B, Realized vol, Compression ratio, Range/ATR
- Volume & Microstructure: Volume ratio, Volume spike, VWAP deviation, OBV derivative

Basé sur les spécifications techniques Meta-Regime Trading v1.0
Littérature: López de Prado (2018), Oxford-Man (2019), AQR (2020)
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)

# Import constants
from constants import (
    RSI_PERIOD, CCI_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BOL_PERIOD, BOL_NUM_STD
)


# ============================================================================
# TREND FEATURES
# ============================================================================

def calculate_ma_slope(prices: Union[pd.Series, np.ndarray],
                       period: int = 20,
                       normalize: bool = True) -> np.ndarray:
    """
    Calcule la pente d'une moyenne mobile.

    Formule:
        slope[i] = (MA[i] - MA[i-1]) / MA[i-1]

    Args:
        prices: Prix de clôture
        period: Période de la MA (défaut: 20)
        normalize: Si True, normalise par le prix (défaut: True)

    Returns:
        Pente de la MA (valeurs continues)
    """
    if isinstance(prices, pd.Series):
        prices = prices.values

    # Calculer MA
    ma = pd.Series(prices).rolling(window=period, min_periods=period).mean().values

    # Calculer pente
    slope = np.full_like(prices, np.nan, dtype=float)
    slope[1:] = np.diff(ma)

    if normalize:
        # Normaliser par le prix pour avoir un pourcentage
        slope = slope / prices

    logger.debug(f"MA{period} slope calculée: min={np.nanmin(slope):.6f}, max={np.nanmax(slope):.6f}")

    return slope


def calculate_adx(high: Union[pd.Series, np.ndarray],
                  low: Union[pd.Series, np.ndarray],
                  close: Union[pd.Series, np.ndarray],
                  period: int = 14) -> np.ndarray:
    """
    Calcule l'ADX (Average Directional Index).

    Mesure la force de la tendance (pas la direction).
    ADX > 25 = tendance forte
    ADX < 20 = tendance faible/range

    Args:
        high: Prix hauts
        low: Prix bas
        close: Prix de clôture
        period: Période de calcul (défaut: 14)

    Returns:
        ADX (valeurs entre 0 et 100)
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values

    # Calculer True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # Première valeur = high - low

    # Calculer directional movements
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low),
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)),
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0

    # Smooth avec EMA
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values / atr

    # Calculer DX et ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values

    logger.debug(f"ADX calculé (période={period}): min={np.nanmin(adx):.1f}, max={np.nanmax(adx):.1f}")

    return adx


def calculate_regression_stats(prices: Union[pd.Series, np.ndarray],
                                window: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule la pente et le R² d'une régression linéaire glissante.

    Args:
        prices: Prix de clôture
        window: Fenêtre de régression (défaut: 20)

    Returns:
        Tuple (slope, r_squared)
        - slope: Pente de la régression normalisée
        - r_squared: Coefficient de détermination (0-1)
    """
    if isinstance(prices, pd.Series):
        prices = prices.values

    n = len(prices)
    slope = np.full(n, np.nan, dtype=float)
    r_squared = np.full(n, np.nan, dtype=float)

    # X = [0, 1, 2, ..., window-1]
    x = np.arange(window)

    for i in range(window, n):
        y = prices[i-window:i]

        # Régression linéaire
        try:
            slope_val, intercept, r_val, p_val, std_err = stats.linregress(x, y)
            slope[i] = slope_val / prices[i]  # Normaliser par le prix
            r_squared[i] = r_val ** 2
        except:
            slope[i] = np.nan
            r_squared[i] = np.nan

    logger.debug(f"Regression stats (window={window}): slope min={np.nanmin(slope):.6f}, R² mean={np.nanmean(r_squared):.3f}")

    return slope, r_squared


def calculate_hurst_exponent(prices: Union[pd.Series, np.ndarray],
                              window: int = 20) -> np.ndarray:
    """
    Calcule l'exposant de Hurst glissant.

    H < 0.5 = mean-reverting (anti-persistant)
    H = 0.5 = random walk
    H > 0.5 = trending (persistant)

    Args:
        prices: Prix de clôture
        window: Fenêtre de calcul (défaut: 20)

    Returns:
        Hurst exponent (valeurs entre 0 et 1)
    """
    if isinstance(prices, pd.Series):
        prices = prices.values

    n = len(prices)
    hurst = np.full(n, np.nan, dtype=float)

    for i in range(window, n):
        series = prices[i-window:i]

        try:
            # Méthode R/S (Rescaled Range)
            # 1. Calculer les returns
            returns = np.diff(np.log(series))

            # 2. Calculer mean et std
            mean_return = np.mean(returns)
            std_return = np.std(returns)

            if std_return == 0:
                hurst[i] = 0.5
                continue

            # 3. Calculer cumulative deviations
            cum_dev = np.cumsum(returns - mean_return)

            # 4. Calculer range et rescale
            R = np.max(cum_dev) - np.min(cum_dev)
            S = std_return

            if S > 0:
                RS = R / S
                # 5. H ≈ log(R/S) / log(n)
                hurst[i] = np.log(RS) / np.log(len(returns))
                # Clip entre 0 et 1
                hurst[i] = np.clip(hurst[i], 0, 1)
            else:
                hurst[i] = 0.5
        except:
            hurst[i] = 0.5

    logger.debug(f"Hurst exponent (window={window}): min={np.nanmin(hurst):.3f}, max={np.nanmax(hurst):.3f}")

    return hurst


def calculate_macd_histogram_normalized(prices: Union[pd.Series, np.ndarray],
                                        fast: int = 12,
                                        slow: int = 26,
                                        signal: int = 9) -> np.ndarray:
    """
    Calcule l'histogramme MACD normalisé.

    Histogram = MACD - Signal
    Normalisé par le prix pour avoir un pourcentage.

    Args:
        prices: Prix de clôture
        fast: Période EMA rapide (défaut: 12)
        slow: Période EMA lente (défaut: 26)
        signal: Période Signal (défaut: 9)

    Returns:
        Histogramme MACD normalisé
    """
    if isinstance(prices, pd.Series):
        prices = prices.values

    # Calculer EMAs
    ema_fast = pd.Series(prices).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(prices).ewm(span=slow, adjust=False).mean().values

    # MACD = fast - slow
    macd_line = ema_fast - ema_slow

    # Signal line
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values

    # Histogram
    histogram = macd_line - signal_line

    # Normaliser par le prix
    histogram_normalized = histogram / prices

    logger.debug(f"MACD histogram normalized: min={np.nanmin(histogram_normalized):.6f}, max={np.nanmax(histogram_normalized):.6f}")

    return histogram_normalized


# ============================================================================
# VOLATILITY FEATURES
# ============================================================================

def calculate_atr_normalized(high: Union[pd.Series, np.ndarray],
                              low: Union[pd.Series, np.ndarray],
                              close: Union[pd.Series, np.ndarray],
                              period: int = 14) -> np.ndarray:
    """
    Calcule l'ATR normalisé par le prix.

    ATR_norm = ATR / Close

    Args:
        high: Prix hauts
        low: Prix bas
        close: Prix de clôture
        period: Période de calcul (défaut: 14)

    Returns:
        ATR normalisé (pourcentage de volatilité)
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values

    # Calculer True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]

    # ATR = EMA du TR
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values

    # Normaliser par le prix
    atr_normalized = atr / close

    logger.debug(f"ATR normalized (période={period}): mean={np.nanmean(atr_normalized):.4f}")

    return atr_normalized


def calculate_bollinger_bands(prices: Union[pd.Series, np.ndarray],
                               period: int = 20,
                               num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule les Bollinger Bands et features associées.

    Returns:
        Tuple (bb_upper, bb_middle, bb_lower, bb_width, percent_b)
        - bb_width: (upper - lower) / middle (largeur normalisée)
        - percent_b: (close - lower) / (upper - lower) (position 0-1)
    """
    if isinstance(prices, pd.Series):
        prices = prices.values

    # Middle band = SMA
    bb_middle = pd.Series(prices).rolling(window=period, min_periods=period).mean().values

    # Standard deviation
    bb_std = pd.Series(prices).rolling(window=period, min_periods=period).std().values

    # Upper and lower bands
    bb_upper = bb_middle + (num_std * bb_std)
    bb_lower = bb_middle - (num_std * bb_std)

    # BB Width (normalized)
    bb_width = (bb_upper - bb_lower) / (bb_middle + 1e-10)

    # %B (position dans les bandes)
    percent_b = (prices - bb_lower) / (bb_upper - bb_lower + 1e-10)

    logger.debug(f"Bollinger Bands (période={period}): width mean={np.nanmean(bb_width):.4f}, %B mean={np.nanmean(percent_b):.3f}")

    return bb_upper, bb_middle, bb_lower, bb_width, percent_b


def calculate_realized_volatility(prices: Union[pd.Series, np.ndarray],
                                   window: int = 20) -> np.ndarray:
    """
    Calcule la volatilité réalisée (annualisée).

    RV = std(returns) × sqrt(periods_per_year)
    Pour 5min: 12 périodes/heure × 24h × 365j = 105120

    Args:
        prices: Prix de clôture
        window: Fenêtre de calcul (défaut: 20)

    Returns:
        Volatilité réalisée annualisée
    """
    if isinstance(prices, pd.Series):
        prices = prices.values

    # Calculer returns
    returns = np.diff(np.log(prices))
    returns = np.concatenate([[0], returns])

    # Rolling std
    rv = pd.Series(returns).rolling(window=window, min_periods=window).std().values

    # Annualiser (pour 5min: sqrt(105120))
    periods_per_year = 12 * 24 * 365  # 5min bars
    rv = rv * np.sqrt(periods_per_year)

    logger.debug(f"Realized volatility (window={window}): mean={np.nanmean(rv):.3f}")

    return rv


def calculate_volatility_compression(prices: Union[pd.Series, np.ndarray],
                                     short_window: int = 10,
                                     long_window: int = 50) -> np.ndarray:
    """
    Calcule le ratio de compression de volatilité.

    Compression = std(short) / std(long)
    < 1 = volatilité compressée (potentiel breakout)
    > 1 = volatilité expansée

    Args:
        prices: Prix de clôture
        short_window: Fenêtre courte (défaut: 10)
        long_window: Fenêtre longue (défaut: 50)

    Returns:
        Ratio de compression
    """
    if isinstance(prices, pd.Series):
        prices = prices.values

    # Calculer returns
    returns = np.diff(np.log(prices))
    returns = np.concatenate([[0], returns])

    # Rolling std
    std_short = pd.Series(returns).rolling(window=short_window, min_periods=short_window).std().values
    std_long = pd.Series(returns).rolling(window=long_window, min_periods=long_window).std().values

    # Ratio
    compression = std_short / (std_long + 1e-10)

    logger.debug(f"Volatility compression: mean={np.nanmean(compression):.3f}")

    return compression


def calculate_range_atr_ratio(high: Union[pd.Series, np.ndarray],
                               low: Union[pd.Series, np.ndarray],
                               close: Union[pd.Series, np.ndarray],
                               period: int = 14) -> np.ndarray:
    """
    Calcule le ratio Range / ATR.

    Mesure si le range actuel est normal ou exceptionnel.

    Args:
        high: Prix hauts
        low: Prix bas
        close: Prix de clôture
        period: Période ATR (défaut: 14)

    Returns:
        Ratio Range/ATR
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values

    # Range actuel
    current_range = high - low

    # Calculer ATR
    atr = calculate_atr_normalized(high, low, close, period) * close

    # Ratio
    ratio = current_range / (atr + 1e-10)

    logger.debug(f"Range/ATR ratio: mean={np.nanmean(ratio):.3f}")

    return ratio


# ============================================================================
# VOLUME & MICROSTRUCTURE FEATURES
# ============================================================================

def calculate_volume_ratio(volume: Union[pd.Series, np.ndarray],
                            period: int = 20) -> np.ndarray:
    """
    Calcule le ratio volume / moyenne mobile du volume.

    > 1 = volume au-dessus de la moyenne
    < 1 = volume en-dessous de la moyenne

    Args:
        volume: Volume
        period: Période de la MA (défaut: 20)

    Returns:
        Ratio volume/MA(volume)
    """
    if isinstance(volume, pd.Series):
        volume = volume.values

    # MA du volume
    volume_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values

    # Ratio
    ratio = volume / (volume_ma + 1e-10)

    logger.debug(f"Volume ratio (période={period}): mean={np.nanmean(ratio):.3f}")

    return ratio


def calculate_volume_spike(volume: Union[pd.Series, np.ndarray],
                            window: int = 20) -> np.ndarray:
    """
    Calcule le z-score du volume glissant.

    Z-score > 2 = volume spike
    Z-score < -2 = volume très faible

    Args:
        volume: Volume
        window: Fenêtre de calcul (défaut: 20)

    Returns:
        Z-score du volume
    """
    if isinstance(volume, pd.Series):
        volume = volume.values

    # Mean et std glissants
    volume_mean = pd.Series(volume).rolling(window=window, min_periods=window).mean().values
    volume_std = pd.Series(volume).rolling(window=window, min_periods=window).std().values

    # Z-score
    z_score = (volume - volume_mean) / (volume_std + 1e-10)

    logger.debug(f"Volume spike (window={window}): z-score mean={np.nanmean(z_score):.3f}")

    return z_score


def calculate_vwap_deviation(high: Union[pd.Series, np.ndarray],
                              low: Union[pd.Series, np.ndarray],
                              close: Union[pd.Series, np.ndarray],
                              volume: Union[pd.Series, np.ndarray],
                              window: int = 20) -> np.ndarray:
    """
    Calcule la déviation du prix par rapport au VWAP.

    VWAP = sum(price × volume) / sum(volume)
    Deviation = (close - VWAP) / VWAP

    Args:
        high: Prix hauts
        low: Prix bas
        close: Prix de clôture
        volume: Volume
        window: Fenêtre de calcul (défaut: 20)

    Returns:
        Déviation normalisée par rapport au VWAP
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values
    if isinstance(volume, pd.Series):
        volume = volume.values

    # Typical Price
    typical_price = (high + low + close) / 3

    # VWAP glissant
    pv = typical_price * volume
    cumsum_pv = pd.Series(pv).rolling(window=window, min_periods=window).sum().values
    cumsum_volume = pd.Series(volume).rolling(window=window, min_periods=window).sum().values

    vwap = cumsum_pv / (cumsum_volume + 1e-10)

    # Déviation normalisée
    deviation = (close - vwap) / (vwap + 1e-10)

    logger.debug(f"VWAP deviation (window={window}): mean={np.nanmean(deviation):.4f}")

    return deviation


def calculate_obv_derivative(close: Union[pd.Series, np.ndarray],
                              volume: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """
    Calcule la dérivée de l'OBV (On-Balance Volume).

    OBV accumule le volume selon la direction du prix.
    OBV derivative = changement de momentum du volume.

    Args:
        close: Prix de clôture
        volume: Volume

    Returns:
        Dérivée de l'OBV normalisée
    """
    if isinstance(close, pd.Series):
        close = close.values
    if isinstance(volume, pd.Series):
        volume = volume.values

    # Calculer OBV
    obv = np.zeros_like(close, dtype=float)
    obv[0] = volume[0]

    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]

    # Dérivée (changement)
    obv_derivative = np.diff(obv)
    obv_derivative = np.concatenate([[0], obv_derivative])

    # Normaliser par la moyenne du volume
    volume_mean = np.mean(volume)
    obv_derivative = obv_derivative / (volume_mean + 1e-10)

    logger.debug(f"OBV derivative: mean={np.nanmean(obv_derivative):.3f}")

    return obv_derivative


# ============================================================================
# FEATURE AGGREGATOR
# ============================================================================

def calculate_all_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule TOUTES les features de régime pour un DataFrame OHLCV.

    Args:
        df: DataFrame avec colonnes ['open', 'high', 'low', 'close', 'volume']

    Returns:
        DataFrame avec ~50 features ajoutées
    """
    logger.info("Calcul de toutes les features de régime...")

    df = df.copy()

    # ========== TREND FEATURES ==========
    logger.info("Calcul des features de tendance...")
    df['ma20_slope'] = calculate_ma_slope(df['close'], period=20)
    df['ma50_slope'] = calculate_ma_slope(df['close'], period=50)

    df['regression_slope'], df['regression_r2'] = calculate_regression_stats(df['close'], window=20)

    df['adx'] = calculate_adx(df['high'], df['low'], df['close'], period=14)

    df['macd_histogram_norm'] = calculate_macd_histogram_normalized(df['close'],
                                                                     fast=MACD_FAST,
                                                                     slow=MACD_SLOW,
                                                                     signal=MACD_SIGNAL)

    df['hurst_exponent'] = calculate_hurst_exponent(df['close'], window=20)

    # ========== VOLATILITY FEATURES ==========
    logger.info("Calcul des features de volatilité...")
    df['atr_normalized'] = calculate_atr_normalized(df['high'], df['low'], df['close'], period=14)

    bb_upper, bb_middle, bb_lower, bb_width, percent_b = calculate_bollinger_bands(df['close'],
                                                                                     period=BOL_PERIOD,
                                                                                     num_std=BOL_NUM_STD)
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    df['bb_width'] = bb_width
    df['percent_b'] = percent_b

    df['realized_volatility'] = calculate_realized_volatility(df['close'], window=20)

    df['volatility_compression'] = calculate_volatility_compression(df['close'],
                                                                      short_window=10,
                                                                      long_window=50)

    df['range_atr_ratio'] = calculate_range_atr_ratio(df['high'], df['low'], df['close'], period=14)

    # ========== VOLUME & MICROSTRUCTURE FEATURES ==========
    logger.info("Calcul des features de volume et microstructure...")
    df['volume_ratio'] = calculate_volume_ratio(df['volume'], period=20)

    df['volume_spike'] = calculate_volume_spike(df['volume'], window=20)

    df['vwap_deviation'] = calculate_vwap_deviation(df['high'], df['low'], df['close'], df['volume'], window=20)

    df['obv_derivative'] = calculate_obv_derivative(df['close'], df['volume'])

    logger.info(f"Calcul terminé : {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']])} features créées")

    return df


def get_regime_feature_names() -> list:
    """
    Retourne la liste des noms de features de régime.

    Returns:
        Liste des noms de colonnes features (~50)
    """
    return [
        # Trend features (6)
        'ma20_slope', 'ma50_slope',
        'regression_slope', 'regression_r2',
        'adx',
        'macd_histogram_norm',
        'hurst_exponent',

        # Volatility features (9)
        'atr_normalized',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'percent_b',
        'realized_volatility',
        'volatility_compression',
        'range_atr_ratio',

        # Volume & microstructure features (4)
        'volume_ratio',
        'volume_spike',
        'vwap_deviation',
        'obv_derivative'
    ]
