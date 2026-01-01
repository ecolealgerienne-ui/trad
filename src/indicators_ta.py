"""
Indicateurs techniques utilisant la bibliothèque `ta`.

⚠️ PRÉFÉRER CE MODULE à indicators.py (plus optimisé et testé).

La bibliothèque `ta` (Technical Analysis Library) fournit:
- Implémentations optimisées
- Tests unitaires complets
- Maintenance active
- Plus de 130 indicateurs

Installation: pip install ta

Indicateurs disponibles:
- RSI, CCI, MACD, Bollinger Bands
- Stochastic, Williams %R, ROC
- ADX, ATR, Ichimoku
- Et bien d'autres...
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("Bibliothèque 'ta' non installée. Pip install ta")


def check_ta_available():
    """Vérifie que la bibliothèque ta est installée."""
    if not TA_AVAILABLE:
        raise ImportError(
            "La bibliothèque 'ta' n'est pas installée.\n"
            "Installation: pip install ta\n"
            "Fallback: Utilisez indicators.py à la place."
        )


def calculate_rsi_ta(close: Union[pd.Series, np.ndarray],
                    window: int = 14,
                    fillna: bool = False) -> np.ndarray:
    """
    Calcule le RSI (Relative Strength Index) avec la bibliothèque ta.

    Args:
        close: Prix de clôture
        window: Période de calcul (défaut: 14)
        fillna: Si True, remplit les NaN avec 50

    Returns:
        RSI (valeurs entre 0 et 100)

    Example:
        >>> rsi = calculate_rsi_ta(df['close'], window=14)
    """
    check_ta_available()

    if isinstance(close, np.ndarray):
        close = pd.Series(close)

    rsi = ta.momentum.RSIIndicator(close=close, window=window, fillna=fillna).rsi()

    logger.debug(f"RSI calculé (ta library, window={window})")

    return rsi.values


def calculate_cci_ta(high: Union[pd.Series, np.ndarray],
                    low: Union[pd.Series, np.ndarray],
                    close: Union[pd.Series, np.ndarray],
                    window: int = 20,
                    constant: float = 0.015,
                    fillna: bool = False) -> np.ndarray:
    """
    Calcule le CCI (Commodity Channel Index) avec ta.

    Args:
        high: Prix High
        low: Prix Low
        close: Prix Close
        window: Période de calcul (défaut: 20)
        constant: Constante de Lambert (défaut: 0.015)
        fillna: Si True, remplit les NaN

    Returns:
        CCI
    """
    check_ta_available()

    if isinstance(high, np.ndarray):
        high = pd.Series(high)
    if isinstance(low, np.ndarray):
        low = pd.Series(low)
    if isinstance(close, np.ndarray):
        close = pd.Series(close)

    cci = ta.trend.CCIIndicator(
        high=high,
        low=low,
        close=close,
        window=window,
        constant=constant,
        fillna=fillna
    ).cci()

    logger.debug(f"CCI calculé (ta library, window={window})")

    return cci.values


def calculate_macd_ta(close: Union[pd.Series, np.ndarray],
                     window_fast: int = 12,
                     window_slow: int = 26,
                     window_sign: int = 9,
                     fillna: bool = False) -> Dict[str, np.ndarray]:
    """
    Calcule le MACD avec ta.

    Args:
        close: Prix de clôture
        window_fast: Période EMA rapide (défaut: 12)
        window_slow: Période EMA lente (défaut: 26)
        window_sign: Période signal (défaut: 9)
        fillna: Si True, remplit les NaN

    Returns:
        Dictionnaire {'macd': array, 'signal': array, 'diff': array}
    """
    check_ta_available()

    if isinstance(close, np.ndarray):
        close = pd.Series(close)

    macd_indicator = ta.trend.MACD(
        close=close,
        window_fast=window_fast,
        window_slow=window_slow,
        window_sign=window_sign,
        fillna=fillna
    )

    logger.debug(f"MACD calculé (ta library, {window_fast}/{window_slow}/{window_sign})")

    return {
        'macd': macd_indicator.macd().values,
        'signal': macd_indicator.macd_signal().values,
        'diff': macd_indicator.macd_diff().values  # Histogram
    }


def calculate_bollinger_bands_ta(close: Union[pd.Series, np.ndarray],
                                window: int = 20,
                                window_dev: int = 2,
                                fillna: bool = False) -> Dict[str, np.ndarray]:
    """
    Calcule les Bandes de Bollinger avec ta.

    Args:
        close: Prix de clôture
        window: Période de calcul (défaut: 20)
        window_dev: Nombre d'écarts-types (défaut: 2)
        fillna: Si True, remplit les NaN

    Returns:
        Dictionnaire {'upper': array, 'middle': array, 'lower': array,
                      'bandwidth': array, 'pband': array}
    """
    check_ta_available()

    if isinstance(close, np.ndarray):
        close = pd.Series(close)

    bb_indicator = ta.volatility.BollingerBands(
        close=close,
        window=window,
        window_dev=window_dev,
        fillna=fillna
    )

    logger.debug(f"Bollinger Bands calculées (ta library, window={window})")

    return {
        'upper': bb_indicator.bollinger_hband().values,
        'middle': bb_indicator.bollinger_mavg().values,
        'lower': bb_indicator.bollinger_lband().values,
        'bandwidth': bb_indicator.bollinger_wband().values,  # Bandwidth
        'pband': bb_indicator.bollinger_pband().values  # %B indicator
    }


def calculate_atr_ta(high: Union[pd.Series, np.ndarray],
                    low: Union[pd.Series, np.ndarray],
                    close: Union[pd.Series, np.ndarray],
                    window: int = 14,
                    fillna: bool = False) -> np.ndarray:
    """
    Calcule l'ATR (Average True Range) avec ta.

    Args:
        high: Prix High
        low: Prix Low
        close: Prix Close
        window: Période de calcul
        fillna: Si True, remplit les NaN

    Returns:
        ATR
    """
    check_ta_available()

    if isinstance(high, np.ndarray):
        high = pd.Series(high)
    if isinstance(low, np.ndarray):
        low = pd.Series(low)
    if isinstance(close, np.ndarray):
        close = pd.Series(close)

    atr = ta.volatility.AverageTrueRange(
        high=high,
        low=low,
        close=close,
        window=window,
        fillna=fillna
    ).average_true_range()

    logger.debug(f"ATR calculé (ta library, window={window})")

    return atr.values


def calculate_stochastic_ta(high: Union[pd.Series, np.ndarray],
                           low: Union[pd.Series, np.ndarray],
                           close: Union[pd.Series, np.ndarray],
                           window: int = 14,
                           smooth_window: int = 3,
                           fillna: bool = False) -> Dict[str, np.ndarray]:
    """
    Calcule l'oscillateur Stochastic avec ta.

    Args:
        high: Prix High
        low: Prix Low
        close: Prix Close
        window: Période pour %K
        smooth_window: Période pour %D
        fillna: Si True, remplit les NaN

    Returns:
        Dictionnaire {'k': array, 'd': array}
    """
    check_ta_available()

    if isinstance(high, np.ndarray):
        high = pd.Series(high)
    if isinstance(low, np.ndarray):
        low = pd.Series(low)
    if isinstance(close, np.ndarray):
        close = pd.Series(close)

    stoch_indicator = ta.momentum.StochasticOscillator(
        high=high,
        low=low,
        close=close,
        window=window,
        smooth_window=smooth_window,
        fillna=fillna
    )

    logger.debug(f"Stochastic calculé (ta library, {window}/{smooth_window})")

    return {
        'k': stoch_indicator.stoch().values,
        'd': stoch_indicator.stoch_signal().values
    }


def calculate_adx_ta(high: Union[pd.Series, np.ndarray],
                    low: Union[pd.Series, np.ndarray],
                    close: Union[pd.Series, np.ndarray],
                    window: int = 14,
                    fillna: bool = False) -> Dict[str, np.ndarray]:
    """
    Calcule l'ADX (Average Directional Index) avec ta.

    Mesure la force de la tendance.

    Args:
        high: Prix High
        low: Prix Low
        close: Prix Close
        window: Période de calcul
        fillna: Si True, remplit les NaN

    Returns:
        Dictionnaire {'adx': array, 'adx_pos': array, 'adx_neg': array}
    """
    check_ta_available()

    if isinstance(high, np.ndarray):
        high = pd.Series(high)
    if isinstance(low, np.ndarray):
        low = pd.Series(low)
    if isinstance(close, np.ndarray):
        close = pd.Series(close)

    adx_indicator = ta.trend.ADXIndicator(
        high=high,
        low=low,
        close=close,
        window=window,
        fillna=fillna
    )

    logger.debug(f"ADX calculé (ta library, window={window})")

    return {
        'adx': adx_indicator.adx().values,
        'adx_pos': adx_indicator.adx_pos().values,  # +DI
        'adx_neg': adx_indicator.adx_neg().values   # -DI
    }


def add_all_ta_features(df: pd.DataFrame,
                       rsi_windows: List[int] = [14, 21],
                       cci_windows: List[int] = [20],
                       macd_params: List[tuple] = [(12, 26, 9)],
                       bb_windows: List[int] = [20],
                       atr_windows: List[int] = [14],
                       stoch_params: List[tuple] = [(14, 3)],
                       add_adx: bool = True,
                       fillna: bool = False) -> pd.DataFrame:
    """
    Ajoute tous les indicateurs techniques au DataFrame avec la bibliothèque ta.

    Args:
        df: DataFrame avec colonnes ['high', 'low', 'close']
        rsi_windows: Liste des périodes RSI
        cci_windows: Liste des périodes CCI
        macd_params: Liste de tuples (fast, slow, signal) pour MACD
        bb_windows: Liste des périodes Bollinger Bands
        atr_windows: Liste des périodes ATR
        stoch_params: Liste de tuples (window, smooth_window) pour Stochastic
        add_adx: Si True, ajoute ADX
        fillna: Si True, remplit les NaN

    Returns:
        DataFrame avec tous les indicateurs ajoutés

    Example:
        >>> df = add_all_ta_features(df, rsi_windows=[14, 21])
    """
    check_ta_available()

    df = df.copy()

    logger.info("Ajout de tous les indicateurs (ta library)...")

    # RSI
    for window in rsi_windows:
        col_name = f'rsi_{window}'
        df[col_name] = calculate_rsi_ta(df['close'], window=window, fillna=fillna)
        logger.info(f"  ✓ {col_name}")

    # CCI
    for window in cci_windows:
        col_name = f'cci_{window}'
        df[col_name] = calculate_cci_ta(df['high'], df['low'], df['close'],
                                        window=window, fillna=fillna)
        logger.info(f"  ✓ {col_name}")

    # MACD
    for fast, slow, signal in macd_params:
        macd_data = calculate_macd_ta(df['close'], fast, slow, signal, fillna=fillna)
        prefix = f'macd_{fast}_{slow}_{signal}'
        df[f'{prefix}_line'] = macd_data['macd']
        df[f'{prefix}_signal'] = macd_data['signal']
        df[f'{prefix}_diff'] = macd_data['diff']
        logger.info(f"  ✓ {prefix}")

    # Bollinger Bands
    for window in bb_windows:
        bb_data = calculate_bollinger_bands_ta(df['close'], window=window, fillna=fillna)
        prefix = f'bb_{window}'
        df[f'{prefix}_upper'] = bb_data['upper']
        df[f'{prefix}_middle'] = bb_data['middle']
        df[f'{prefix}_lower'] = bb_data['lower']
        df[f'{prefix}_bandwidth'] = bb_data['bandwidth']
        df[f'{prefix}_pband'] = bb_data['pband']
        logger.info(f"  ✓ {prefix}")

    # ATR
    for window in atr_windows:
        col_name = f'atr_{window}'
        df[col_name] = calculate_atr_ta(df['high'], df['low'], df['close'],
                                        window=window, fillna=fillna)
        logger.info(f"  ✓ {col_name}")

    # Stochastic
    for window, smooth in stoch_params:
        stoch_data = calculate_stochastic_ta(df['high'], df['low'], df['close'],
                                             window=window, smooth_window=smooth,
                                             fillna=fillna)
        prefix = f'stoch_{window}_{smooth}'
        df[f'{prefix}_k'] = stoch_data['k']
        df[f'{prefix}_d'] = stoch_data['d']
        logger.info(f"  ✓ {prefix}")

    # ADX
    if add_adx:
        adx_data = calculate_adx_ta(df['high'], df['low'], df['close'],
                                   window=14, fillna=fillna)
        df['adx_14'] = adx_data['adx']
        df['adx_14_pos'] = adx_data['adx_pos']
        df['adx_14_neg'] = adx_data['adx_neg']
        logger.info("  ✓ adx_14")

    logger.info(f"Tous les indicateurs ajoutés (ta). Total colonnes: {len(df.columns)}")

    return df


def add_all_ta_features_bulk(df: pd.DataFrame,
                             fillna: bool = False) -> pd.DataFrame:
    """
    Ajoute TOUS les indicateurs disponibles dans ta (méthode bulk).

    ⚠️ Génère beaucoup de colonnes (~80+).

    Args:
        df: DataFrame avec colonnes ['open', 'high', 'low', 'close', 'volume']
        fillna: Si True, remplit les NaN

    Returns:
        DataFrame avec tous les indicateurs
    """
    check_ta_available()

    df = df.copy()

    logger.info("Ajout de TOUS les indicateurs ta (bulk mode)...")

    # Ajouter toutes les features d'un coup
    df = ta.add_all_ta_features(
        df,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume" if "volume" in df.columns else None,
        fillna=fillna
    )

    logger.info(f"Indicateurs bulk ajoutés. Total colonnes: {len(df.columns)}")

    return df
