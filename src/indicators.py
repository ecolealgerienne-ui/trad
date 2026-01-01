"""
Indicateurs techniques pour le trading crypto.

Tous les indicateurs sont CAUSAUX (n'utilisent que le passé).
Ils peuvent être utilisés comme features pour le modèle.

Indicateurs implémentés:
- RSI (Relative Strength Index)
- CCI (Commodity Channel Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def calculate_rsi(prices: Union[pd.Series, np.ndarray],
                 period: int = 14) -> np.ndarray:
    """
    Calcule le RSI (Relative Strength Index).

    Formule:
        RSI = 100 - (100 / (1 + RS))
        RS = Moyenne des gains / Moyenne des pertes

    Args:
        prices: Prix de clôture
        period: Période de calcul (défaut: 14)

    Returns:
        RSI (valeurs entre 0 et 100)

    Example:
        >>> rsi = calculate_rsi(df['close'], period=14)
        >>> # RSI > 70 = surachat, RSI < 30 = survente
    """
    if isinstance(prices, pd.Series):
        prices = prices.values

    # Calculer les variations
    deltas = np.diff(prices)
    deltas = np.concatenate([[0], deltas])  # Ajouter 0 au début

    # Séparer gains et pertes
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Moyenne mobile exponentielle des gains et pertes
    avg_gains = np.full_like(prices, np.nan, dtype=float)
    avg_losses = np.full_like(prices, np.nan, dtype=float)

    # Première moyenne (SMA)
    if len(gains) >= period:
        avg_gains[period] = np.mean(gains[1:period+1])
        avg_losses[period] = np.mean(losses[1:period+1])

        # EMA pour les valeurs suivantes
        for i in range(period + 1, len(prices)):
            avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i]) / period
            avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i]) / period

    # Calculer RSI
    rsi = np.full_like(prices, np.nan, dtype=float)
    mask = avg_losses != 0

    rs = np.where(mask, avg_gains / avg_losses, 0)
    rsi = 100 - (100 / (1 + rs))

    # Gérer le cas où avg_losses = 0 (que des gains)
    rsi[avg_losses == 0] = 100

    logger.debug(f"RSI calculé (période={period}): min={np.nanmin(rsi):.1f}, max={np.nanmax(rsi):.1f}")

    return rsi


def calculate_cci(high: Union[pd.Series, np.ndarray],
                 low: Union[pd.Series, np.ndarray],
                 close: Union[pd.Series, np.ndarray],
                 period: int = 20,
                 constant: float = 0.015) -> np.ndarray:
    """
    Calcule le CCI (Commodity Channel Index).

    Formule:
        CCI = (TP - SMA(TP)) / (constant * MeanDeviation)
        TP = Typical Price = (High + Low + Close) / 3

    Args:
        high: Prix High
        low: Prix Low
        close: Prix Close
        period: Période de calcul (défaut: 20)
        constant: Constante de Lambert (défaut: 0.015)

    Returns:
        CCI (généralement entre -100 et +100)

    Example:
        >>> cci = calculate_cci(df['high'], df['low'], df['close'], period=20)
        >>> # CCI > 100 = surachat, CCI < -100 = survente
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values

    # Typical Price
    tp = (high + low + close) / 3

    # SMA du Typical Price
    tp_series = pd.Series(tp)
    sma_tp = tp_series.rolling(window=period, min_periods=period).mean().values

    # Mean Deviation
    mad = np.full_like(tp, np.nan, dtype=float)

    for i in range(period - 1, len(tp)):
        mad[i] = np.mean(np.abs(tp[i-period+1:i+1] - sma_tp[i]))

    # CCI
    cci = (tp - sma_tp) / (constant * mad)

    logger.debug(f"CCI calculé (période={period}): min={np.nanmin(cci):.1f}, max={np.nanmax(cci):.1f}")

    return cci


def calculate_macd(prices: Union[pd.Series, np.ndarray],
                  fast_period: int = 12,
                  slow_period: int = 26,
                  signal_period: int = 9) -> Dict[str, np.ndarray]:
    """
    Calcule le MACD (Moving Average Convergence Divergence).

    Formule:
        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA(MACD Line, signal_period)
        Histogram = MACD Line - Signal Line

    Args:
        prices: Prix de clôture
        fast_period: Période EMA rapide (défaut: 12)
        slow_period: Période EMA lente (défaut: 26)
        signal_period: Période signal (défaut: 9)

    Returns:
        Dictionnaire {'macd': array, 'signal': array, 'histogram': array}

    Example:
        >>> macd_data = calculate_macd(df['close'])
        >>> df['macd'] = macd_data['macd']
        >>> df['macd_signal'] = macd_data['signal']
        >>> df['macd_hist'] = macd_data['histogram']
    """
    if isinstance(prices, pd.Series):
        prices_series = prices
    else:
        prices_series = pd.Series(prices)

    # EMA rapide et lente
    ema_fast = prices_series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices_series.ewm(span=slow_period, adjust=False).mean()

    # MACD Line
    macd_line = (ema_fast - ema_slow).values

    # Signal Line (EMA de la MACD Line)
    macd_series = pd.Series(macd_line)
    signal_line = macd_series.ewm(span=signal_period, adjust=False).mean().values

    # Histogram
    histogram = macd_line - signal_line

    logger.debug(f"MACD calculé ({fast_period}/{slow_period}/{signal_period})")

    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def calculate_bollinger_bands(prices: Union[pd.Series, np.ndarray],
                             period: int = 20,
                             num_std: float = 2.0) -> Dict[str, np.ndarray]:
    """
    Calcule les Bandes de Bollinger.

    Formule:
        Middle Band = SMA(period)
        Upper Band = Middle Band + (num_std * STD)
        Lower Band = Middle Band - (num_std * STD)

    Args:
        prices: Prix de clôture
        period: Période de calcul (défaut: 20)
        num_std: Nombre d'écarts-types (défaut: 2.0)

    Returns:
        Dictionnaire {'upper': array, 'middle': array, 'lower': array, 'bandwidth': array}

    Example:
        >>> bb = calculate_bollinger_bands(df['close'], period=20, num_std=2)
        >>> df['bb_upper'] = bb['upper']
        >>> df['bb_middle'] = bb['middle']
        >>> df['bb_lower'] = bb['lower']
    """
    if isinstance(prices, pd.Series):
        prices_series = prices
    else:
        prices_series = pd.Series(prices)

    # Middle Band (SMA)
    middle = prices_series.rolling(window=period, min_periods=period).mean()

    # Écart-type
    std = prices_series.rolling(window=period, min_periods=period).std()

    # Upper et Lower Bands
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)

    # Bandwidth (mesure de la volatilité)
    bandwidth = ((upper - lower) / middle).values

    logger.debug(f"Bollinger Bands calculées (période={period}, std={num_std})")

    return {
        'upper': upper.values,
        'middle': middle.values,
        'lower': lower.values,
        'bandwidth': bandwidth
    }


def calculate_atr(high: Union[pd.Series, np.ndarray],
                 low: Union[pd.Series, np.ndarray],
                 close: Union[pd.Series, np.ndarray],
                 period: int = 14) -> np.ndarray:
    """
    Calcule l'ATR (Average True Range) - mesure de volatilité.

    Args:
        high: Prix High
        low: Prix Low
        close: Prix Close
        period: Période de calcul

    Returns:
        ATR
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values

    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))

    # Prendre le max des 3
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # Premier élément = high - low

    # ATR = EMA du True Range
    tr_series = pd.Series(tr)
    atr = tr_series.ewm(span=period, adjust=False).mean().values

    logger.debug(f"ATR calculé (période={period})")

    return atr


def calculate_stochastic(high: Union[pd.Series, np.ndarray],
                        low: Union[pd.Series, np.ndarray],
                        close: Union[pd.Series, np.ndarray],
                        k_period: int = 14,
                        d_period: int = 3) -> Dict[str, np.ndarray]:
    """
    Calcule l'oscillateur Stochastic.

    Formule:
        %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
        %D = SMA(%K, d_period)

    Args:
        high: Prix High
        low: Prix Low
        close: Prix Close
        k_period: Période pour %K
        d_period: Période pour %D

    Returns:
        Dictionnaire {'k': array, 'd': array}
    """
    if isinstance(high, pd.Series):
        high_series = high
    else:
        high_series = pd.Series(high)

    if isinstance(low, pd.Series):
        low_series = low
    else:
        low_series = pd.Series(low)

    if isinstance(close, pd.Series):
        close_series = close
    else:
        close_series = pd.Series(close)

    # Highest High et Lowest Low sur k_period
    highest_high = high_series.rolling(window=k_period, min_periods=k_period).max()
    lowest_low = low_series.rolling(window=k_period, min_periods=k_period).min()

    # %K
    k = 100 * (close_series - lowest_low) / (highest_high - lowest_low)

    # %D (SMA de %K)
    d = k.rolling(window=d_period, min_periods=d_period).mean()

    logger.debug(f"Stochastic calculé (%K={k_period}, %D={d_period})")

    return {
        'k': k.values,
        'd': d.values
    }


def add_all_indicators(df: pd.DataFrame,
                      rsi_periods: list = [14],
                      cci_periods: list = [20],
                      macd_params: list = [(12, 26, 9)],
                      bb_periods: list = [20]) -> pd.DataFrame:
    """
    Ajoute tous les indicateurs au DataFrame avec plusieurs périodes.

    Args:
        df: DataFrame avec colonnes ['open', 'high', 'low', 'close']
        rsi_periods: Liste des périodes RSI à calculer
        cci_periods: Liste des périodes CCI à calculer
        macd_params: Liste de tuples (fast, slow, signal) pour MACD
        bb_periods: Liste des périodes Bollinger Bands

    Returns:
        DataFrame avec tous les indicateurs ajoutés

    Example:
        >>> df = add_all_indicators(df, rsi_periods=[14, 21], cci_periods=[20, 40])
    """
    df = df.copy()

    logger.info("Ajout de tous les indicateurs...")

    # RSI pour différentes périodes
    for period in rsi_periods:
        col_name = f'rsi_{period}'
        df[col_name] = calculate_rsi(df['close'], period=period)
        logger.info(f"  ✓ {col_name} ajouté")

    # CCI pour différentes périodes
    for period in cci_periods:
        col_name = f'cci_{period}'
        df[col_name] = calculate_cci(df['high'], df['low'], df['close'], period=period)
        logger.info(f"  ✓ {col_name} ajouté")

    # MACD pour différents paramètres
    for fast, slow, signal in macd_params:
        macd_data = calculate_macd(df['close'], fast, slow, signal)
        prefix = f'macd_{fast}_{slow}_{signal}'
        df[f'{prefix}_line'] = macd_data['macd']
        df[f'{prefix}_signal'] = macd_data['signal']
        df[f'{prefix}_hist'] = macd_data['histogram']
        logger.info(f"  ✓ {prefix} ajouté")

    # Bollinger Bands pour différentes périodes
    for period in bb_periods:
        bb_data = calculate_bollinger_bands(df['close'], period=period)
        prefix = f'bb_{period}'
        df[f'{prefix}_upper'] = bb_data['upper']
        df[f'{prefix}_middle'] = bb_data['middle']
        df[f'{prefix}_lower'] = bb_data['lower']
        df[f'{prefix}_bandwidth'] = bb_data['bandwidth']
        logger.info(f"  ✓ {prefix} ajouté")

    # ATR
    df['atr_14'] = calculate_atr(df['high'], df['low'], df['close'], period=14)
    logger.info("  ✓ atr_14 ajouté")

    # Stochastic
    stoch = calculate_stochastic(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch['k']
    df['stoch_d'] = stoch['d']
    logger.info("  ✓ stochastic ajouté")

    logger.info(f"Tous les indicateurs ajoutés. Total colonnes: {len(df.columns)}")

    return df
