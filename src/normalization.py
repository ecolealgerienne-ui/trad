"""
Normalisation des données pour universalité multi-actifs.

⚠️ IMPORTANT: JAMAIS utiliser de prix bruts comme features!
Le modèle doit être universel (BTC, ETH, etc.)

Méthodes de normalisation:
1. Z-Score glissant (rolling)
2. Relative Open (prix en % de l'Open de la bougie)
3. Min-Max scaling
4. Log returns
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def rolling_zscore(data: Union[pd.Series, np.ndarray],
                  window: int = 50,
                  min_periods: Optional[int] = None) -> np.ndarray:
    """
    Calcule le Z-Score glissant (rolling).

    Formule:
        z = (x - rolling_mean) / rolling_std

    ⚠️ CAUSAL: N'utilise que le passé (fenêtre glissante)

    Args:
        data: Données à normaliser
        window: Taille de la fenêtre glissante
        min_periods: Nombre minimum de périodes (défaut: window)

    Returns:
        Z-Score normalisé

    Example:
        >>> normalized_price = rolling_zscore(df['close'], window=50)
        >>> # Valeurs typiquement entre -3 et +3
    """
    if isinstance(data, np.ndarray):
        data_series = pd.Series(data)
    else:
        data_series = data

    if min_periods is None:
        min_periods = window

    # Calculer mean et std glissants
    rolling_mean = data_series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = data_series.rolling(window=window, min_periods=min_periods).std()

    # Z-Score
    zscore = (data_series - rolling_mean) / rolling_std

    # Remplacer les inf/nan par 0
    zscore = zscore.replace([np.inf, -np.inf], np.nan)

    logger.debug(f"Z-Score calculé (window={window}): mean={zscore.mean():.3f}, std={zscore.std():.3f}")

    return zscore.values


def relative_to_open(open_price: Union[float, np.ndarray],
                    high: Union[float, np.ndarray],
                    low: Union[float, np.ndarray],
                    close: Union[float, np.ndarray]) -> dict:
    """
    Exprime H, L, C en pourcentage de l'Open.

    Formule:
        relative_high = (high - open) / open * 100
        relative_low = (low - open) / open * 100
        relative_close = (close - open) / open * 100

    Args:
        open_price: Prix d'ouverture
        high: Prix High
        low: Prix Low
        close: Prix Close

    Returns:
        Dictionnaire {'high': %, 'low': %, 'close': %}

    Example:
        >>> # Pour la bougie fantôme 30m
        >>> rel = relative_to_open(ghost_open, ghost_high, ghost_low, ghost_close)
        >>> df['ghost_high_rel'] = rel['high']
    """
    open_price = np.asarray(open_price)
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)

    # Éviter division par zéro
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_high = (high - open_price) / open_price * 100
        rel_low = (low - open_price) / open_price * 100
        rel_close = (close - open_price) / open_price * 100

    # Remplacer inf/nan par 0
    rel_high = np.where(np.isfinite(rel_high), rel_high, 0)
    rel_low = np.where(np.isfinite(rel_low), rel_low, 0)
    rel_close = np.where(np.isfinite(rel_close), rel_close, 0)

    logger.debug(f"Relative to Open calculé: high={np.mean(rel_high):.2f}%, low={np.mean(rel_low):.2f}%, close={np.mean(rel_close):.2f}%")

    return {
        'high': rel_high,
        'low': rel_low,
        'close': rel_close
    }


def log_returns(prices: Union[pd.Series, np.ndarray],
               periods: int = 1) -> np.ndarray:
    """
    Calcule les returns logarithmiques.

    Formule:
        return = log(price[t] / price[t-1])

    Args:
        prices: Série de prix
        periods: Nombre de périodes

    Returns:
        Log returns

    Example:
        >>> returns = log_returns(df['close'])
        >>> # Stationnaire et centré autour de 0
    """
    if isinstance(prices, pd.Series):
        prices = prices.values

    # Log returns
    log_ret = np.diff(np.log(prices), n=periods)

    # Ajouter NaN au début
    log_ret = np.concatenate([np.full(periods, np.nan), log_ret])

    logger.debug(f"Log returns calculés (periods={periods}): mean={np.nanmean(log_ret):.6f}, std={np.nanstd(log_ret):.6f}")

    return log_ret


def percent_change(prices: Union[pd.Series, np.ndarray],
                  periods: int = 1) -> np.ndarray:
    """
    Calcule le pourcentage de variation.

    Formule:
        pct_change = (price[t] - price[t-1]) / price[t-1] * 100

    Args:
        prices: Série de prix
        periods: Nombre de périodes

    Returns:
        Variation en %
    """
    if isinstance(prices, pd.Series):
        prices_series = prices
    else:
        prices_series = pd.Series(prices)

    pct = prices_series.pct_change(periods=periods) * 100

    return pct.values


def minmax_scaling(data: Union[pd.Series, np.ndarray],
                  feature_range: Tuple[float, float] = (0, 1),
                  window: Optional[int] = None) -> np.ndarray:
    """
    Normalisation Min-Max.

    Formule:
        scaled = (x - min) / (max - min) * (max_range - min_range) + min_range

    Args:
        data: Données à normaliser
        feature_range: Range de sortie (min, max)
        window: Si spécifié, utilise une fenêtre glissante (CAUSAL)

    Returns:
        Données normalisées
    """
    if isinstance(data, pd.Series):
        data_series = data
    else:
        data_series = pd.Series(data)

    min_val, max_val = feature_range

    if window is not None:
        # Min-Max glissant (CAUSAL)
        rolling_min = data_series.rolling(window=window, min_periods=window).min()
        rolling_max = data_series.rolling(window=window, min_periods=window).max()

        scaled = (data_series - rolling_min) / (rolling_max - rolling_min)
        scaled = scaled * (max_val - min_val) + min_val

    else:
        # Min-Max global (pour le dataset entier)
        data_min = data_series.min()
        data_max = data_series.max()

        scaled = (data_series - data_min) / (data_max - data_min)
        scaled = scaled * (max_val - min_val) + min_val

    return scaled.values


def robust_scaling(data: Union[pd.Series, np.ndarray],
                  window: Optional[int] = None) -> np.ndarray:
    """
    Robust Scaling (utilise médiane et IQR au lieu de mean/std).

    Formule:
        scaled = (x - median) / IQR

    Moins sensible aux outliers que le Z-Score.

    Args:
        data: Données à normaliser
        window: Si spécifié, utilise une fenêtre glissante

    Returns:
        Données normalisées
    """
    if isinstance(data, pd.Series):
        data_series = data
    else:
        data_series = pd.Series(data)

    if window is not None:
        # Robust scaling glissant
        rolling_median = data_series.rolling(window=window, min_periods=window).median()
        rolling_q25 = data_series.rolling(window=window, min_periods=window).quantile(0.25)
        rolling_q75 = data_series.rolling(window=window, min_periods=window).quantile(0.75)

        iqr = rolling_q75 - rolling_q25
        scaled = (data_series - rolling_median) / iqr

    else:
        # Robust scaling global
        median = data_series.median()
        q25 = data_series.quantile(0.25)
        q75 = data_series.quantile(0.75)
        iqr = q75 - q25

        scaled = (data_series - median) / iqr

    return scaled.values


def normalize_ohlc_ghost(df: pd.DataFrame,
                        ghost_prefix: str = 'ghost',
                        method: str = 'relative_open') -> pd.DataFrame:
    """
    Normalise les colonnes OHLC de la bougie fantôme.

    Args:
        df: DataFrame avec colonnes ghost_open, ghost_high, ghost_low, ghost_close
        ghost_prefix: Préfixe des colonnes bougie fantôme
        method: 'relative_open' ou 'zscore'

    Returns:
        DataFrame avec colonnes normalisées ajoutées

    Example:
        >>> df = normalize_ohlc_ghost(df, method='relative_open')
        >>> # Ajoute: ghost_high_norm, ghost_low_norm, ghost_close_norm
    """
    df = df.copy()

    o = f'{ghost_prefix}_open'
    h = f'{ghost_prefix}_high'
    l = f'{ghost_prefix}_low'
    c = f'{ghost_prefix}_close'

    # Vérifier que les colonnes existent
    required_cols = [o, h, l, c]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    if method == 'relative_open':
        # Méthode Relative Open (recommandée pour bougie fantôme)
        rel = relative_to_open(
            df[o].values,
            df[h].values,
            df[l].values,
            df[c].values
        )

        df[f'{ghost_prefix}_high_norm'] = rel['high']
        df[f'{ghost_prefix}_low_norm'] = rel['low']
        df[f'{ghost_prefix}_close_norm'] = rel['close']

        logger.info(f"Normalisation Relative Open appliquée à {ghost_prefix}")

    elif method == 'zscore':
        # Méthode Z-Score glissant
        df[f'{ghost_prefix}_open_norm'] = rolling_zscore(df[o], window=50)
        df[f'{ghost_prefix}_high_norm'] = rolling_zscore(df[h], window=50)
        df[f'{ghost_prefix}_low_norm'] = rolling_zscore(df[l], window=50)
        df[f'{ghost_prefix}_close_norm'] = rolling_zscore(df[c], window=50)

        logger.info(f"Normalisation Z-Score appliquée à {ghost_prefix}")

    else:
        raise ValueError(f"Méthode inconnue: {method}")

    return df


def normalize_features(df: pd.DataFrame,
                      feature_cols: list,
                      method: str = 'zscore',
                      window: int = 50,
                      suffix: str = '_norm') -> pd.DataFrame:
    """
    Normalise une liste de features.

    Args:
        df: DataFrame
        feature_cols: Liste des colonnes à normaliser
        method: 'zscore', 'minmax', 'robust'
        window: Taille de la fenêtre (pour méthodes glissantes)
        suffix: Suffixe à ajouter aux colonnes normalisées

    Returns:
        DataFrame avec colonnes normalisées ajoutées

    Example:
        >>> feature_cols = ['rsi_14', 'cci_20', 'macd_12_26_9_line']
        >>> df = normalize_features(df, feature_cols, method='zscore')
    """
    df = df.copy()

    for col in feature_cols:
        if col not in df.columns:
            logger.warning(f"Colonne {col} introuvable, ignorée")
            continue

        new_col = f'{col}{suffix}'

        if method == 'zscore':
            df[new_col] = rolling_zscore(df[col], window=window)

        elif method == 'minmax':
            df[new_col] = minmax_scaling(df[col], window=window)

        elif method == 'robust':
            df[new_col] = robust_scaling(df[col], window=window)

        else:
            raise ValueError(f"Méthode inconnue: {method}")

        logger.debug(f"  ✓ {new_col} créée")

    logger.info(f"{len(feature_cols)} features normalisées avec {method}")

    return df


def clip_outliers(data: Union[pd.Series, np.ndarray],
                 lower_quantile: float = 0.01,
                 upper_quantile: float = 0.99) -> np.ndarray:
    """
    Clip les outliers en utilisant les quantiles.

    Args:
        data: Données
        lower_quantile: Quantile inférieur
        upper_quantile: Quantile supérieur

    Returns:
        Données clippées
    """
    if isinstance(data, pd.Series):
        data_series = data
    else:
        data_series = pd.Series(data)

    lower = data_series.quantile(lower_quantile)
    upper = data_series.quantile(upper_quantile)

    clipped = data_series.clip(lower=lower, upper=upper)

    n_clipped = ((data_series < lower) | (data_series > upper)).sum()
    logger.debug(f"Outliers clippés: {n_clipped} valeurs ({n_clipped/len(data_series)*100:.2f}%)")

    return clipped.values
