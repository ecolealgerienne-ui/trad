"""
Filtres de signal pour le lissage et la reconstruction.

Le filtre d'Octave (filtfilt) est utilisé pour créer un signal "propre" qui sert de cible.
⚠️ IMPORTANT: Ce filtre utilise les données futures (non-causal) - UNIQUEMENT pour le label!
"""

import numpy as np
import pandas as pd
from scipy import signal
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


def octave_filter(data: Union[pd.Series, np.ndarray],
                 smoothing: float = 0.25,
                 order: int = 2) -> np.ndarray:
    """
    Applique un filtre d'Octave (Butterworth lowpass + filtfilt).

    ⚠️ ATTENTION: Ce filtre est NON-CAUSAL (utilise le futur).
    Il doit être utilisé UNIQUEMENT pour générer les labels, JAMAIS pour les features!

    Args:
        data: Signal d'entrée (RSI, Close, etc.)
        smoothing: Paramètre de lissage (0.0 à 1.0)
                  0.25 = filtre modéré (recommandé)
                  0.1 = filtre fort
                  0.5 = filtre léger
        order: Ordre du filtre Butterworth

    Returns:
        Signal filtré (même longueur que l'entrée)

    Example:
        >>> rsi = calculate_rsi(close_prices)
        >>> filtered_rsi = octave_filter(rsi, smoothing=0.25)
        >>> slope = np.diff(filtered_rsi)
    """
    if isinstance(data, pd.Series):
        data = data.values

    # Gérer les NaN
    mask = ~np.isnan(data)
    if not mask.any():
        logger.warning("Toutes les valeurs sont NaN, retour de l'entrée")
        return data

    # Extraire les données valides
    valid_data = data[mask]

    # Normalisation de la fréquence de coupure
    # smoothing = 0.25 signifie fc = 0.25 * (fs / 2) où fs = 1 (normalisé)
    # Donc fc = 0.25 / 2 = 0.125
    cutoff_freq = smoothing / 2.0

    # Vérifier que la fréquence est valide
    if not (0 < cutoff_freq < 0.5):
        raise ValueError(f"cutoff_freq doit être entre 0 et 0.5, obtenu {cutoff_freq}")

    # Créer le filtre Butterworth
    b, a = signal.butter(order, cutoff_freq, btype='low', analog=False)

    # Appliquer filtfilt (filtre bidirectionnel sans déphasage)
    # ⚠️ Utilise les données futures!
    filtered_valid = signal.filtfilt(b, a, valid_data)

    # Reconstruire le tableau complet avec NaN préservés
    filtered = np.full_like(data, np.nan, dtype=float)
    filtered[mask] = filtered_valid

    logger.debug(f"Filtre d'Octave appliqué: smoothing={smoothing}, order={order}")

    return filtered


def savgol_filter(data: Union[pd.Series, np.ndarray],
                 window_length: int = 11,
                 polyorder: int = 2) -> np.ndarray:
    """
    Filtre Savitzky-Golay (alternative au filtre d'Octave).

    ⚠️ ATTENTION: Également NON-CAUSAL (utilise le futur).

    Args:
        data: Signal d'entrée
        window_length: Longueur de la fenêtre (doit être impair)
        polyorder: Ordre du polynôme

    Returns:
        Signal filtré
    """
    if isinstance(data, pd.Series):
        data = data.values

    # S'assurer que window_length est impair
    if window_length % 2 == 0:
        window_length += 1

    # Gérer les NaN
    mask = ~np.isnan(data)
    valid_data = data[mask]

    if len(valid_data) < window_length:
        logger.warning(f"Données insuffisantes pour Savgol ({len(valid_data)} < {window_length})")
        return data

    # Appliquer le filtre
    filtered_valid = signal.savgol_filter(valid_data, window_length, polyorder)

    # Reconstruire
    filtered = np.full_like(data, np.nan, dtype=float)
    filtered[mask] = filtered_valid

    return filtered


def exponential_smoothing(data: Union[pd.Series, np.ndarray],
                         alpha: float = 0.3) -> np.ndarray:
    """
    Lissage exponentiel simple (CAUSAL - peut être utilisé pour features).

    ⚠️ Ce filtre est CAUSAL (n'utilise que le passé).

    Args:
        data: Signal d'entrée
        alpha: Facteur de lissage (0 à 1)
              alpha proche de 0 = fort lissage
              alpha proche de 1 = suit le signal

    Returns:
        Signal lissé
    """
    if isinstance(data, pd.Series):
        data = data.values

    smoothed = np.full_like(data, np.nan, dtype=float)

    # Initialiser avec la première valeur non-NaN
    mask = ~np.isnan(data)
    if not mask.any():
        return smoothed

    first_idx = np.where(mask)[0][0]
    smoothed[first_idx] = data[first_idx]

    # Appliquer le lissage exponentiel
    for i in range(first_idx + 1, len(data)):
        if not np.isnan(data[i]):
            if np.isnan(smoothed[i-1]):
                smoothed[i] = data[i]
            else:
                smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]

    return smoothed


def rolling_mean_causal(data: Union[pd.Series, np.ndarray],
                       window: int = 10) -> np.ndarray:
    """
    Moyenne mobile CAUSALE (n'utilise que le passé).

    Args:
        data: Signal d'entrée
        window: Taille de la fenêtre

    Returns:
        Moyenne mobile
    """
    if isinstance(data, pd.Series):
        series = data
    else:
        series = pd.Series(data)

    # Rolling avec min_periods pour éviter les NaN excessifs
    rolled = series.rolling(window=window, min_periods=1).mean()

    return rolled.values


def calculate_slope(data: Union[pd.Series, np.ndarray],
                   periods: int = 1) -> np.ndarray:
    """
    Calcule la pente (différence) sur N périodes.

    Args:
        data: Signal d'entrée
        periods: Nombre de périodes pour la différence

    Returns:
        Pente (slope)

    Example:
        >>> slope = calculate_slope(filtered_signal, periods=1)
        >>> # slope[t] = signal[t] - signal[t-1]
    """
    if isinstance(data, pd.Series):
        data = data.values

    slope = np.diff(data, n=periods)

    # Ajouter NaN au début pour garder la longueur
    slope = np.concatenate([np.full(periods, np.nan), slope])

    return slope


def apply_filter_with_validation(data: Union[pd.Series, np.ndarray],
                                 filter_type: str = 'octave',
                                 **kwargs) -> dict:
    """
    Applique un filtre avec validation et statistiques.

    Args:
        data: Signal d'entrée
        filter_type: 'octave', 'savgol', 'exp', 'rolling'
        **kwargs: Paramètres spécifiques au filtre

    Returns:
        Dictionnaire avec 'filtered', 'stats', 'warnings'
    """
    result = {
        'filtered': None,
        'stats': {},
        'warnings': []
    }

    # Vérifier les données
    if isinstance(data, pd.Series):
        input_data = data.values
    else:
        input_data = np.array(data)

    nan_count = np.isnan(input_data).sum()
    nan_pct = nan_count / len(input_data) * 100

    if nan_pct > 50:
        result['warnings'].append(f"⚠️ {nan_pct:.1f}% de NaN dans les données")

    # Appliquer le filtre
    if filter_type == 'octave':
        smoothing = kwargs.get('smoothing', 0.25)
        filtered = octave_filter(input_data, smoothing=smoothing)

    elif filter_type == 'savgol':
        window_length = kwargs.get('window_length', 11)
        polyorder = kwargs.get('polyorder', 2)
        filtered = savgol_filter(input_data, window_length, polyorder)

    elif filter_type == 'exp':
        alpha = kwargs.get('alpha', 0.3)
        filtered = exponential_smoothing(input_data, alpha=alpha)

    elif filter_type == 'rolling':
        window = kwargs.get('window', 10)
        filtered = rolling_mean_causal(input_data, window=window)

    else:
        raise ValueError(f"Type de filtre inconnu: {filter_type}")

    result['filtered'] = filtered

    # Calculer les statistiques
    valid_original = input_data[~np.isnan(input_data)]
    valid_filtered = filtered[~np.isnan(filtered)]

    if len(valid_original) > 0 and len(valid_filtered) > 0:
        result['stats'] = {
            'original_mean': np.mean(valid_original),
            'filtered_mean': np.mean(valid_filtered),
            'original_std': np.std(valid_original),
            'filtered_std': np.std(valid_filtered),
            'smoothing_ratio': np.std(valid_filtered) / np.std(valid_original),
            'nan_count': np.isnan(filtered).sum()
        }

    logger.info(f"Filtre {filter_type} appliqué - Ratio lissage: {result['stats'].get('smoothing_ratio', 0):.3f}")

    return result
