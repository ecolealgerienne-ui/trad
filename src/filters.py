"""
Filtres de signal pour le lissage et la reconstruction.

⚠️ IMPORTANT: Ces filtres utilisent les données futures (non-causaux) - UNIQUEMENT pour les labels!

Filtres disponibles:
- signal_filtfilt: Filtre d'Octave (Butterworth + filtfilt) - MÉTHODE PRINCIPALE
- kalman_filter: Filtre de Kalman
- hp_filter: Hodrick-Prescott filter (tendance/cycle)
- wavelet_denoise: Débruitage par ondelettes
- loess_smoothing: LOESS smoothing
- emd_filter: Empirical Mode Decomposition
- ensemble_filter: Combinaison de plusieurs filtres
"""

import numpy as np
import pandas as pd
import scipy.signal
from typing import Union, Optional, Dict, List
import logging
import warnings

logger = logging.getLogger(__name__)


def signal_filtfilt(signal_data: Union[pd.Series, np.ndarray],
                    step: float = 0.25,
                    order: int = 3) -> np.ndarray:
    """
    Filtre d'Octave principal (Butterworth + filtfilt).

    ⚠️ ATTENTION: Ce filtre est NON-CAUSAL (utilise le futur).
    Il doit être utilisé UNIQUEMENT pour générer les labels, JAMAIS pour les features!

    Cette méthode est la référence du projet.

    Args:
        signal_data: Signal d'entrée (RSI, Close, etc.)
        step: Paramètre de lissage (0.0 à 1.0)
              - 0.2 = filtre fort (plus lisse)
              - 0.25 = filtre modéré (recommandé par défaut)
              - 0.3 = filtre léger
        order: Ordre du filtre Butterworth (défaut: 3)

    Returns:
        Signal filtré (même longueur que l'entrée)

    Example:
        >>> from indicators import calculate_rsi
        >>> rsi = calculate_rsi(df['close'])
        >>> filtered_rsi = signal_filtfilt(rsi, step=0.25)
        >>> slope = np.diff(filtered_rsi)
    """
    if isinstance(signal_data, pd.Series):
        signal_data = signal_data.values

    # Gérer les NaN
    mask = ~np.isnan(signal_data)
    if not mask.any():
        logger.warning("Toutes les valeurs sont NaN, retour de l'entrée")
        return signal_data

    # Extraire les données valides
    valid_data = signal_data[mask]

    # Créer le filtre Butterworth
    B, A = scipy.signal.butter(order, step, output='ba')

    # Appliquer filtfilt (filtre bidirectionnel sans déphasage)
    # ⚠️ Utilise les données futures!
    filtered_valid = scipy.signal.filtfilt(B, A, valid_data)

    # Reconstruire le tableau complet avec NaN préservés
    filtered = np.full_like(signal_data, np.nan, dtype=float)
    filtered[mask] = filtered_valid

    logger.debug(f"signal_filtfilt appliqué: step={step}, order={order}")

    return filtered


# Alias pour compatibilité
def octave_filter(data: Union[pd.Series, np.ndarray],
                 smoothing: float = 0.25,
                 order: int = 3) -> np.ndarray:
    """
    Alias de signal_filtfilt pour compatibilité.

    ⚠️ Utilisez signal_filtfilt() de préférence.
    """
    return signal_filtfilt(data, step=smoothing, order=order)


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


def kalman_filter(signal_data: Union[pd.Series, np.ndarray],
                 process_variance: float = 0.01,
                 measurement_variance: float = 0.1) -> np.ndarray:
    """
    Filtre de Kalman pour lissage du signal.

    ⚠️ NON-CAUSAL si utilisé avec smoother (utilise le futur).

    Args:
        signal_data: Signal d'entrée
        process_variance: Variance du processus (Q)
        measurement_variance: Variance de mesure (R)

    Returns:
        Signal filtré

    Example:
        >>> filtered = kalman_filter(rsi, process_variance=0.01)
    """
    try:
        from pykalman import KalmanFilter
    except ImportError:
        logger.error("pykalman non installé. Pip install pykalman")
        raise ImportError("pip install pykalman")

    if isinstance(signal_data, pd.Series):
        signal_data = signal_data.values

    # Gérer les NaN
    mask = ~np.isnan(signal_data)
    if not mask.any():
        return signal_data

    valid_data = signal_data[mask].reshape(-1, 1)

    # Initialiser le filtre de Kalman
    kf = KalmanFilter(
        initial_state_mean=valid_data[0],
        n_dim_obs=1,
        n_dim_state=1,
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_covariance=1,
        observation_covariance=measurement_variance,
        transition_covariance=process_variance
    )

    # Appliquer le smoother (utilise passé + futur)
    state_means, _ = kf.smooth(valid_data)

    # Reconstruire
    filtered = np.full_like(signal_data, np.nan, dtype=float)
    filtered[mask] = state_means.flatten()

    logger.debug(f"Filtre de Kalman appliqué: Q={process_variance}, R={measurement_variance}")

    return filtered


def hp_filter(signal_data: Union[pd.Series, np.ndarray],
             lamb: float = 1600) -> Dict[str, np.ndarray]:
    """
    Filtre Hodrick-Prescott (HP Filter).

    Sépare un signal en tendance + cycle.

    Args:
        signal_data: Signal d'entrée
        lamb: Paramètre de lissage (lambda)
              - 1600 = données mensuelles (défaut)
              - 6.25 = données annuelles
              - 129600 = données trimestrielles
              - Pour crypto intraday: utiliser lamb plus petit (100-400)

    Returns:
        Dictionnaire avec 'trend' et 'cycle'

    Example:
        >>> result = hp_filter(close_prices, lamb=400)
        >>> trend = result['trend']
        >>> cycle = result['cycle']
    """
    try:
        from statsmodels.tsa.filters.hp_filter import hpfilter
    except ImportError:
        logger.error("statsmodels non installé. Pip install statsmodels")
        raise ImportError("pip install statsmodels")

    if isinstance(signal_data, pd.Series):
        signal_series = signal_data
    else:
        signal_series = pd.Series(signal_data)

    # Appliquer HP filter
    cycle, trend = hpfilter(signal_series.dropna(), lamb=lamb)

    # Aligner avec l'index original
    trend_full = np.full_like(signal_data, np.nan, dtype=float)
    cycle_full = np.full_like(signal_data, np.nan, dtype=float)

    mask = ~signal_series.isna()
    trend_full[mask] = trend.values
    cycle_full[mask] = cycle.values

    logger.debug(f"HP Filter appliqué: lambda={lamb}")

    return {
        'trend': trend_full,
        'cycle': cycle_full
    }


def wavelet_denoise(signal_data: Union[pd.Series, np.ndarray],
                   wavelet: str = 'db4',
                   level: int = 3,
                   mode: str = 'soft') -> np.ndarray:
    """
    Débruitage par ondelettes (Wavelet Denoising).

    Excellent pour signaux crypto (multi-échelle).

    Args:
        signal_data: Signal d'entrée
        wavelet: Type d'ondelette ('db4', 'sym4', 'coif3', etc.)
        level: Niveau de décomposition
        mode: 'soft' ou 'hard' thresholding

    Returns:
        Signal débruité

    Example:
        >>> denoised = wavelet_denoise(rsi, wavelet='db4', level=3)
    """
    try:
        import pywt
    except ImportError:
        logger.error("PyWavelets non installé. Pip install PyWavelets")
        raise ImportError("pip install PyWavelets")

    if isinstance(signal_data, pd.Series):
        signal_data = signal_data.values

    # Gérer les NaN
    mask = ~np.isnan(signal_data)
    if not mask.any():
        return signal_data

    valid_data = signal_data[mask]

    # Décomposition en ondelettes
    coeffs = pywt.wavedec(valid_data, wavelet, level=level)

    # Calculer le seuil (Universal Threshold)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(valid_data)))

    # Appliquer le seuil à tous les détails
    coeffs_thresh = [coeffs[0]]  # Garder les approximations
    for i in range(1, len(coeffs)):
        if mode == 'soft':
            coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
        else:
            coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='hard'))

    # Reconstruction
    denoised_valid = pywt.waverec(coeffs_thresh, wavelet)

    # Ajuster la longueur si nécessaire
    if len(denoised_valid) != len(valid_data):
        denoised_valid = denoised_valid[:len(valid_data)]

    # Reconstruire
    denoised = np.full_like(signal_data, np.nan, dtype=float)
    denoised[mask] = denoised_valid

    logger.debug(f"Wavelet denoising appliqué: {wavelet}, level={level}")

    return denoised


def loess_smoothing(signal_data: Union[pd.Series, np.ndarray],
                   frac: float = 0.1) -> np.ndarray:
    """
    LOESS (Locally Weighted Scatterplot Smoothing).

    Robuste aux outliers.

    Args:
        signal_data: Signal d'entrée
        frac: Fraction de données pour chaque fit local (0-1)

    Returns:
        Signal lissé
    """
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError:
        logger.error("statsmodels non installé")
        raise ImportError("pip install statsmodels")

    if isinstance(signal_data, pd.Series):
        signal_data = signal_data.values

    # Gérer les NaN
    mask = ~np.isnan(signal_data)
    if not mask.any():
        return signal_data

    valid_data = signal_data[mask]
    x = np.arange(len(valid_data))

    # Appliquer LOESS
    smoothed_valid = lowess(valid_data, x, frac=frac, return_sorted=False)

    # Reconstruire
    smoothed = np.full_like(signal_data, np.nan, dtype=float)
    smoothed[mask] = smoothed_valid

    logger.debug(f"LOESS smoothing appliqué: frac={frac}")

    return smoothed


def emd_filter(signal_data: Union[pd.Series, np.ndarray],
              n_imfs: int = 3) -> Dict[str, np.ndarray]:
    """
    Empirical Mode Decomposition (EMD).

    Décompose signal en modes intrinsèques (IMFs).

    Args:
        signal_data: Signal d'entrée
        n_imfs: Nombre d'IMFs à garder pour reconstruction

    Returns:
        Dictionnaire avec 'filtered', 'imfs', 'residue'

    Example:
        >>> result = emd_filter(close_prices, n_imfs=3)
        >>> filtered = result['filtered']
    """
    try:
        from PyEMD import EMD
    except ImportError:
        logger.error("PyEMD non installé. Pip install EMD-signal")
        raise ImportError("pip install EMD-signal")

    if isinstance(signal_data, pd.Series):
        signal_data = signal_data.values

    # Gérer les NaN
    mask = ~np.isnan(signal_data)
    if not mask.any():
        return {'filtered': signal_data, 'imfs': None, 'residue': None}

    valid_data = signal_data[mask]

    # EMD
    emd = EMD()
    IMFs = emd.emd(valid_data)

    # Reconstruire avec les n premiers IMFs + résidu
    if len(IMFs) > n_imfs:
        filtered_valid = np.sum(IMFs[:n_imfs], axis=0) + IMFs[-1]
    else:
        filtered_valid = np.sum(IMFs, axis=0)

    # Reconstruire
    filtered = np.full_like(signal_data, np.nan, dtype=float)
    filtered[mask] = filtered_valid

    logger.debug(f"EMD appliqué: {len(IMFs)} IMFs, {n_imfs} utilisés")

    return {
        'filtered': filtered,
        'imfs': IMFs,
        'residue': IMFs[-1] if len(IMFs) > 0 else None
    }


def ensemble_filter(signal_data: Union[pd.Series, np.ndarray],
                   filters: List[str] = ['signal_filtfilt', 'kalman', 'hp'],
                   weights: Optional[List[float]] = None,
                   **filter_params) -> np.ndarray:
    """
    Ensemble de filtres (combinaison pondérée).

    Combine plusieurs filtres pour plus de robustesse.

    Args:
        signal_data: Signal d'entrée
        filters: Liste des filtres à combiner
        weights: Poids de chaque filtre (None = poids égaux)
        **filter_params: Paramètres pour chaque filtre

    Returns:
        Signal filtré (moyenne pondérée)

    Example:
        >>> filtered = ensemble_filter(
        ...     rsi,
        ...     filters=['signal_filtfilt', 'kalman', 'wavelet'],
        ...     weights=[0.5, 0.3, 0.2]
        ... )
    """
    if weights is None:
        weights = [1.0 / len(filters)] * len(filters)

    if len(weights) != len(filters):
        raise ValueError("Nombre de poids != nombre de filtres")

    if abs(sum(weights) - 1.0) > 1e-6:
        logger.warning(f"Somme des poids = {sum(weights)}, normalisation à 1.0")
        weights = [w / sum(weights) for w in weights]

    filtered_signals = []

    for filter_name in filters:
        if filter_name == 'signal_filtfilt' or filter_name == 'octave':
            step = filter_params.get('step', 0.25)
            filtered_signals.append(signal_filtfilt(signal_data, step=step))

        elif filter_name == 'kalman':
            pv = filter_params.get('process_variance', 0.01)
            mv = filter_params.get('measurement_variance', 0.1)
            filtered_signals.append(kalman_filter(signal_data, pv, mv))

        elif filter_name == 'hp':
            lamb = filter_params.get('lamb', 400)
            hp_result = hp_filter(signal_data, lamb=lamb)
            filtered_signals.append(hp_result['trend'])

        elif filter_name == 'wavelet':
            wavelet = filter_params.get('wavelet', 'db4')
            level = filter_params.get('level', 3)
            filtered_signals.append(wavelet_denoise(signal_data, wavelet, level))

        elif filter_name == 'loess':
            frac = filter_params.get('frac', 0.1)
            filtered_signals.append(loess_smoothing(signal_data, frac))

        else:
            logger.warning(f"Filtre inconnu: {filter_name}, ignoré")
            continue

    # Moyenne pondérée
    ensemble = np.zeros_like(signal_data, dtype=float)
    for i, (filtered, weight) in enumerate(zip(filtered_signals, weights)):
        ensemble += weight * filtered

    logger.info(f"Ensemble filter: {filters} avec poids {weights}")

    return ensemble


def apply_filter_with_validation(data: Union[pd.Series, np.ndarray],
                                 filter_type: str = 'signal_filtfilt',
                                 **kwargs) -> dict:
    """
    Applique un filtre avec validation et statistiques.

    Args:
        data: Signal d'entrée
        filter_type: 'signal_filtfilt', 'kalman', 'hp', 'wavelet', 'loess', 'emd', 'ensemble'
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
    try:
        if filter_type == 'signal_filtfilt' or filter_type == 'octave':
            step = kwargs.get('step', 0.25)
            filtered = signal_filtfilt(input_data, step=step)

        elif filter_type == 'kalman':
            filtered = kalman_filter(input_data, **kwargs)

        elif filter_type == 'hp':
            hp_result = hp_filter(input_data, **kwargs)
            filtered = hp_result['trend']

        elif filter_type == 'wavelet':
            filtered = wavelet_denoise(input_data, **kwargs)

        elif filter_type == 'loess':
            filtered = loess_smoothing(input_data, **kwargs)

        elif filter_type == 'emd':
            emd_result = emd_filter(input_data, **kwargs)
            filtered = emd_result['filtered']

        elif filter_type == 'ensemble':
            filtered = ensemble_filter(input_data, **kwargs)

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

    except ImportError as e:
        logger.error(f"Erreur d'import pour {filter_type}: {e}")
        result['warnings'].append(str(e))
        result['filtered'] = input_data
        return result

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
