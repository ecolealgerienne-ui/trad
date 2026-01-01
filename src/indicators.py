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

# Import constants
from constants import (
    RSI_PERIOD, CCI_PERIOD, CCI_CONSTANT,
    BOL_PERIOD, BOL_NUM_STD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    CCI_RAW_MIN, CCI_RAW_MAX,
    INDICATOR_MIN, INDICATOR_MAX,
    MACD_NORM_WINDOW,
    DECYCLER_CUTOFF,
    SEQUENCE_LENGTH,
    NUM_INDICATORS
)


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


# =============================================================================
# NORMALIZATION FUNCTIONS (Pour input du modèle IA)
# =============================================================================

def normalize_cci(cci: np.ndarray,
                  raw_min: float = CCI_RAW_MIN,
                  raw_max: float = CCI_RAW_MAX,
                  target_min: float = INDICATOR_MIN,
                  target_max: float = INDICATOR_MAX) -> np.ndarray:
    """
    Normalise le CCI de [-200, +200] vers [0, 100].

    Formule: (value - raw_min) / (raw_max - raw_min) * (target_max - target_min) + target_min

    Args:
        cci: Valeurs CCI brutes
        raw_min: Minimum attendu du CCI (défaut: -200)
        raw_max: Maximum attendu du CCI (défaut: +200)
        target_min: Minimum cible (défaut: 0)
        target_max: Maximum cible (défaut: 100)

    Returns:
        CCI normalisé entre 0 et 100
    """
    cci_clipped = np.clip(cci, raw_min, raw_max)
    cci_normalized = (cci_clipped - raw_min) / (raw_max - raw_min) * (target_max - target_min) + target_min

    logger.debug(f"CCI normalisé: {raw_min}/{raw_max} → {target_min}/{target_max}")

    return cci_normalized


def calculate_bollinger_percent_b(close: Union[pd.Series, np.ndarray],
                                   period: int = BOL_PERIOD,
                                   num_std: float = BOL_NUM_STD) -> np.ndarray:
    """
    Calcule le Bollinger %B (position du prix dans les bandes).

    Formule:
        %B = (Close - Lower Band) / (Upper Band - Lower Band) × 100
        %B = 0   → Prix sur bande basse
        %B = 50  → Prix sur bande moyenne
        %B = 100 → Prix sur bande haute

    Args:
        close: Prix de clôture
        period: Période Bollinger (défaut: 20)
        num_std: Nombre d'écarts-types (défaut: 2)

    Returns:
        %B normalisé entre 0 et 100
    """
    bb = calculate_bollinger_bands(close, period=period, num_std=num_std)

    upper = bb['upper']
    lower = bb['lower']

    if isinstance(close, pd.Series):
        close = close.values

    # %B formula
    percent_b = (close - lower) / (upper - lower)

    # Convertir en 0-100 et clipper
    percent_b = percent_b * 100
    percent_b = np.clip(percent_b, 0, 100)

    logger.debug(f"Bollinger %B calculé (période={period}, std={num_std})")

    return percent_b


def normalize_macd_histogram(histogram: np.ndarray,
                             window: int = MACD_NORM_WINDOW,
                             target_min: float = INDICATOR_MIN,
                             target_max: float = INDICATOR_MAX) -> np.ndarray:
    """
    Normalise l'histogramme MACD avec min-max dynamique sur une fenêtre glissante.

    Stratégie: Pour chaque point, calculer min/max sur les 1000 dernières bougies,
    puis normaliser vers [0, 100].

    Args:
        histogram: Histogramme MACD brut
        window: Fenêtre pour calculer min/max (défaut: 1000)
        target_min: Minimum cible (défaut: 0)
        target_max: Maximum cible (défaut: 100)

    Returns:
        MACD normalisé entre 0 et 100
    """
    histogram_series = pd.Series(histogram)

    # Calculer min/max glissant
    rolling_min = histogram_series.rolling(window=window, min_periods=1).min()
    rolling_max = histogram_series.rolling(window=window, min_periods=1).max()

    # Normalisation
    range_val = rolling_max - rolling_min

    # Éviter division par zéro
    range_val = np.where(range_val == 0, 1, range_val)

    normalized = (histogram_series - rolling_min) / range_val * (target_max - target_min) + target_min

    logger.debug(f"MACD histogram normalisé (window={window})")

    return normalized.values


# =============================================================================
# DECYCLER PARFAIT (Pour génération des labels)
# =============================================================================

def ehlers_decycler(prices: np.ndarray, cutoff: float = DECYCLER_CUTOFF) -> np.ndarray:
    """
    Filtre Decycler de John Ehlers (version causale - forward only).

    Le Decycler est un filtre passe-haut qui enlève les cycles courts.

    Formule:
        α = (cos(2π × cutoff) + sin(2π × cutoff) - 1) / cos(2π × cutoff)
        HP[i] = (1 - α/2)² × (Price[i] - 2×Price[i-1] + Price[i-2]) + 2×(1-α)×HP[i-1] - (1-α)²×HP[i-2]
        Decycler[i] = Price[i] - HP[i]

    Args:
        prices: Série de prix
        cutoff: Fréquence de coupure (défaut: 0.1)

    Returns:
        Signal filtré (Decycler)
    """
    alpha = (np.cos(2 * np.pi * cutoff) + np.sin(2 * np.pi * cutoff) - 1) / np.cos(2 * np.pi * cutoff)

    hp = np.zeros_like(prices)

    for i in range(2, len(prices)):
        hp[i] = ((1 - alpha / 2) ** 2) * (prices[i] - 2 * prices[i-1] + prices[i-2]) + \
                2 * (1 - alpha) * hp[i-1] - \
                ((1 - alpha) ** 2) * hp[i-2]

    decycler = prices - hp

    return decycler


def apply_decycler_perfect(signal: np.ndarray, cutoff: float = DECYCLER_CUTOFF) -> np.ndarray:
    """
    Applique le Decycler en mode PARFAIT (forward-backward) pour génération de labels.

    ⚠️ ATTENTION : Cette version est NON-CAUSALE (utilise le futur)!
    Elle sert UNIQUEMENT pour générer les labels (vérité terrain).

    Process:
        1. Filtre forward (début → fin)
        2. Filtre backward (fin → début)
        3. Résultat = signal parfaitement lissé sans lag

    Args:
        signal: Signal d'entrée (indicateur ou prix)
        cutoff: Fréquence de coupure

    Returns:
        Signal filtré parfait (sans lag temporel)
    """
    # Forward pass
    forward = ehlers_decycler(signal, cutoff=cutoff)

    # Backward pass (inverser, filtrer, ré-inverser)
    backward = ehlers_decycler(forward[::-1], cutoff=cutoff)
    perfect = backward[::-1]

    logger.debug(f"Decycler parfait appliqué (cutoff={cutoff})")

    return perfect


# =============================================================================
# LABEL GENERATION (Pente du filtre parfait)
# =============================================================================

def generate_labels(filtered_indicator: np.ndarray) -> np.ndarray:
    """
    Génère les labels binaires à partir d'un indicateur filtré (Decycler parfait).

    Règle:
        Label[t] = 1  si filtered[t-1] > filtered[t-2]  (pente haussière → BUY)
        Label[t] = 0  si filtered[t-1] <= filtered[t-2] (pente baissière → SELL)

    ⚠️ IMPORTANT: On compare filtered[t-1] vs filtered[t-2] pour synchronisation
    avec open[t+1] (trade exécuté au timestep suivant).

    Args:
        filtered_indicator: Indicateur filtré avec Decycler parfait

    Returns:
        Labels binaires (0 ou 1)
    """
    labels = np.zeros(len(filtered_indicator), dtype=int)

    # À partir de t=2 (besoin de t-1 et t-2)
    for t in range(2, len(filtered_indicator)):
        if filtered_indicator[t-1] > filtered_indicator[t-2]:
            labels[t] = 1  # Pente haussière
        else:
            labels[t] = 0  # Pente baissière

    logger.debug(f"Labels générés: {np.sum(labels)} BUY ({np.sum(labels)/len(labels)*100:.1f}%), "
                f"{len(labels) - np.sum(labels)} SELL ({(len(labels)-np.sum(labels))/len(labels)*100:.1f}%)")

    return labels


# =============================================================================
# SEQUENCE CREATION (Pour modèle CNN-LSTM)
# =============================================================================

def create_sequences(indicators: np.ndarray,
                    labels: np.ndarray,
                    sequence_length: int = SEQUENCE_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crée des séquences de longueur fixe pour l'entraînement du modèle.

    Format:
        X[i] = [indicators[i-sequence_length:i]]  → Séquence de 12 timesteps
        Y[i] = labels[i]                          → Label au temps i

    Args:
        indicators: Array (n_samples, n_indicators) - 4 indicateurs normalisés
        labels: Array (n_samples, n_outputs) - 4 labels binaires (un par indicateur)
        sequence_length: Longueur des séquences (défaut: 12)

    Returns:
        X: Array (n_sequences, sequence_length, n_indicators) - Shape (N, 12, 4)
        Y: Array (n_sequences, n_outputs) - Shape (N, 4)

    Example:
        >>> X, Y = create_sequences(indicators, labels, sequence_length=12)
        >>> print(X.shape)  # (N, 12, 4)
        >>> print(Y.shape)  # (N, 4)
    """
    n_samples = len(indicators)
    n_indicators = indicators.shape[1]
    n_outputs = labels.shape[1] if len(labels.shape) > 1 else 1

    # Nombre de séquences possibles
    n_sequences = n_samples - sequence_length

    # Initialiser arrays
    X = np.zeros((n_sequences, sequence_length, n_indicators))
    Y = np.zeros((n_sequences, n_outputs)) if n_outputs > 1 else np.zeros(n_sequences)

    # Créer séquences
    for i in range(n_sequences):
        X[i] = indicators[i:i+sequence_length]
        Y[i] = labels[i+sequence_length]

    logger.info(f"Séquences créées: X={X.shape}, Y={Y.shape}")

    return X, Y


# =============================================================================
# PIPELINE COMPLET
# =============================================================================

def calculate_all_indicators_for_model(df: pd.DataFrame) -> np.ndarray:
    """
    Calcule les 4 indicateurs normalisés pour le modèle IA.

    Indicateurs (tous normalisés 0-100):
        1. RSI(14)           → Déjà 0-100
        2. CCI(20)           → -200/+200 → 0-100
        3. Bollinger %B(20)  → 0-100
        4. MACD Histogram    → Normalisé dynamiquement → 0-100

    Args:
        df: DataFrame avec colonnes ['open', 'high', 'low', 'close']

    Returns:
        Array (n_samples, 4) avec les 4 indicateurs normalisés
    """
    logger.info("Calcul des 4 indicateurs pour le modèle IA...")

    # 1. RSI (déjà 0-100)
    rsi = calculate_rsi(df['close'], period=RSI_PERIOD)
    logger.info(f"  ✓ RSI({RSI_PERIOD}) calculé")

    # 2. CCI normalisé
    cci_raw = calculate_cci(df['high'], df['low'], df['close'],
                            period=CCI_PERIOD, constant=CCI_CONSTANT)
    cci_norm = normalize_cci(cci_raw)
    logger.info(f"  ✓ CCI({CCI_PERIOD}) calculé et normalisé")

    # 3. Bollinger %B
    bol_percentb = calculate_bollinger_percent_b(df['close'],
                                                  period=BOL_PERIOD,
                                                  num_std=BOL_NUM_STD)
    logger.info(f"  ✓ Bollinger %B({BOL_PERIOD}, {BOL_NUM_STD}σ) calculé")

    # 4. MACD Histogram normalisé
    macd_data = calculate_macd(df['close'],
                               fast_period=MACD_FAST,
                               slow_period=MACD_SLOW,
                               signal_period=MACD_SIGNAL)
    macd_hist_norm = normalize_macd_histogram(macd_data['histogram'])
    logger.info(f"  ✓ MACD({MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL}) histogram normalisé")

    # Combiner en array (n_samples, 4)
    indicators = np.column_stack([rsi, cci_norm, bol_percentb, macd_hist_norm])

    # Gérer les NaN (warm-up des indicateurs)
    # Stratégie: Forward-fill puis remplacer les NaN restants par 50 (valeur neutre)
    indicators_df = pd.DataFrame(indicators, columns=['RSI', 'CCI', 'BOL', 'MACD'])
    indicators_df = indicators_df.ffill().fillna(50.0)
    indicators = indicators_df.values

    n_nan_before = np.sum(np.isnan(np.column_stack([rsi, cci_norm, bol_percentb, macd_hist_norm])))
    if n_nan_before > 0:
        logger.info(f"  ℹ️ {n_nan_before} NaN gérés (warm-up des indicateurs)")

    logger.info(f"Indicateurs combinés: shape={indicators.shape}")

    return indicators


def generate_all_labels(indicators: np.ndarray) -> np.ndarray:
    """
    Génère les labels pour les 4 indicateurs avec Decycler parfait.

    Process:
        1. Pour chaque indicateur (RSI, CCI, BOL, MACD)
        2. Appliquer Decycler parfait (forward-backward)
        3. Générer labels binaires (pente haussière = 1, baissière = 0)

    Args:
        indicators: Array (n_samples, 4) - Les 4 indicateurs normalisés

    Returns:
        Array (n_samples, 4) - Les 4 labels binaires
    """
    logger.info("Génération des labels avec Decycler parfait...")

    n_samples = indicators.shape[0]
    labels = np.zeros((n_samples, NUM_INDICATORS), dtype=int)

    indicator_names = ['RSI', 'CCI', 'BOL', 'MACD']

    for i in range(NUM_INDICATORS):
        # Appliquer Decycler parfait
        filtered = apply_decycler_perfect(indicators[:, i])

        # Générer labels
        labels[:, i] = generate_labels(filtered)

        buy_pct = np.sum(labels[:, i]) / n_samples * 100
        logger.info(f"  ✓ {indicator_names[i]}: {np.sum(labels[:, i])} BUY ({buy_pct:.1f}%), "
                   f"{n_samples - np.sum(labels[:, i])} SELL ({100-buy_pct:.1f}%)")

    return labels


def prepare_datasets(train_df: pd.DataFrame,
                    val_df: pd.DataFrame,
                    test_df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Pipeline COMPLET: des DataFrames raw aux datasets prêts pour l'entraînement.

    Process:
        1. Calculer les 4 indicateurs normalisés (RSI, CCI, BOL, MACD)
        2. Générer les 4 labels avec Decycler parfait
        3. Créer séquences de 12 timesteps
        4. Retourner X, Y pour train/val/test

    Args:
        train_df: DataFrame train (avec colonnes OHLC)
        val_df: DataFrame validation
        test_df: DataFrame test

    Returns:
        Dictionnaire avec:
            'train': (X_train, Y_train)
            'val': (X_val, Y_val)
            'test': (X_test, Y_test)

        Où:
            X shape: (n_sequences, 12, 4) - 12 timesteps × 4 indicateurs
            Y shape: (n_sequences, 4) - 4 labels binaires

    Example:
        >>> datasets = prepare_datasets(train_df, val_df, test_df)
        >>> X_train, Y_train = datasets['train']
        >>> print(X_train.shape)  # (N_train, 12, 4)
        >>> print(Y_train.shape)  # (N_train, 4)
    """
    logger.info("="*80)
    logger.info("PRÉPARATION COMPLÈTE DES DATASETS")
    logger.info("="*80)

    results = {}

    for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        logger.info(f"\n📊 Processing {split_name.upper()} set ({len(df):,} bougies)...")

        # 1. Calculer indicateurs
        indicators = calculate_all_indicators_for_model(df)

        # 2. Générer labels
        labels = generate_all_labels(indicators)

        # 3. Créer séquences
        X, Y = create_sequences(indicators, labels, sequence_length=SEQUENCE_LENGTH)

        logger.info(f"✅ {split_name.upper()}: X={X.shape}, Y={Y.shape}")

        results[split_name] = (X, Y)

    logger.info("="*80)
    logger.info("✅ DATASETS PRÊTS POUR L'ENTRAÎNEMENT")
    logger.info("="*80)

    # Afficher stats finales
    logger.info(f"\n📊 STATS FINALES:")
    for split_name, (X, Y) in results.items():
        logger.info(f"  {split_name.upper():5s}: X={X.shape}, Y={Y.shape}")

    return results


# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

if __name__ == '__main__':
    # Configurer logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    logger.info("="*80)
    logger.info("TEST DU MODULE INDICATORS")
    logger.info("="*80)

    # Charger les données avec data_utils
    from data_utils import load_and_split_btc_eth

    logger.info("\n1. Chargement des données...")
    train_df, val_df, test_df = load_and_split_btc_eth()

    logger.info(f"\n2. Préparation des datasets...")
    datasets = prepare_datasets(train_df, val_df, test_df)

    # Récupérer les datasets
    X_train, Y_train = datasets['train']
    X_val, Y_val = datasets['val']
    X_test, Y_test = datasets['test']

    # Afficher résumé
    logger.info("\n" + "="*80)
    logger.info("RÉSUMÉ FINAL")
    logger.info("="*80)
    logger.info(f"\n📊 SHAPES:")
    logger.info(f"  Train: X={X_train.shape}, Y={Y_train.shape}")
    logger.info(f"  Val:   X={X_val.shape}, Y={Y_val.shape}")
    logger.info(f"  Test:  X={X_test.shape}, Y={X_test.shape}")

    # Vérifier les valeurs
    logger.info(f"\n🔍 VALIDATION:")
    logger.info(f"  X range: [{X_train.min():.2f}, {X_train.max():.2f}] (attendu: [0, 100])")
    logger.info(f"  Y values: {np.unique(Y_train)} (attendu: [0, 1])")

    # Stats labels
    logger.info(f"\n📈 DISTRIBUTION LABELS (Train):")
    for i, name in enumerate(['RSI', 'CCI', 'BOL', 'MACD']):
        buy_count = np.sum(Y_train[:, i])
        buy_pct = buy_count / len(Y_train) * 100
        logger.info(f"  {name}: {buy_count:,} BUY ({buy_pct:.1f}%), "
                   f"{len(Y_train) - buy_count:,} SELL ({100-buy_pct:.1f}%)")

    logger.info("\n✅ Module indicators.py opérationnel!")
    logger.info(f"✅ Prêt pour l'entraînement du modèle CNN-LSTM")
    logger.info("="*80)
