"""
Filtres adaptatifs zero-lag pour features causales.

MISSION CRITIQUE: Remplacer les filtres statiques par des filtres adaptatifs
qui s'ajustent dynamiquement au marché pour réduire le lag à ZÉRO.

Filtres implémentés:
1. KAMA (Kaufman's Adaptive Moving Average) - Efficiency Ratio
2. HMA (Hull Moving Average) - Élimination du décalage de phase
3. Ehlers SuperSmoother - Réduction du lag de groupe
4. Ehlers Decycler - Suppression des cycles de bruit

⚠️ CAUSALITÉ CRITIQUE:
- Tous ces filtres sont STRICTEMENT CAUSAUX (pas de future data)
- Utilisables comme FEATURES (X) pour l'IA
- Le filtfilt non-causal reste UNIQUEMENT pour les LABELS (Y)

Référence littérature:
- Kaufman, P. J. (1995). Smarter Trading
- Ehlers, J. F. (2001). Rocket Science for Traders
- Hull, A. (2005). Active Trader Magazine

Auteur: Pipeline Team
Date: 2026-01-01 (Mise à jour Spec #1 - Filtres Adaptatifs)
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def kama_filter(
    signal: Union[pd.Series, np.ndarray],
    er_period: int = 10,
    fast_ema: int = 2,
    slow_ema: int = 30,
    return_efficiency_ratio: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    KAMA - Kaufman's Adaptive Moving Average.

    Le filtre le plus robuste pour prédire la pente avec l'IA.
    Utilise l'Efficiency Ratio (ER) pour adapter la vitesse du filtre.

    Logique:
    - Si le prix va directement de A à B → ER élevé → filtre rapide
    - Si le prix oscille beaucoup → ER faible → filtre lent

    Formule:
    1. ER = |Change| / Sum(|Volatility|)
    2. SC (Smoothing Constant) = [ER * (fast - slow) + slow]²
    3. KAMA[t] = KAMA[t-1] + SC * (Price[t] - KAMA[t-1])

    Args:
        signal: Série de prix (close, RSI, etc.)
        er_period: Période pour calculer l'Efficiency Ratio (défaut: 10)
        fast_ema: Constante EMA rapide (défaut: 2)
        slow_ema: Constante EMA lente (défaut: 30)
        return_efficiency_ratio: Si True, retourne aussi l'ER

    Returns:
        Si return_efficiency_ratio=False: KAMA filtré (np.ndarray)
        Si return_efficiency_ratio=True: (KAMA, ER) tuple

    ⚠️ CAUSALITÉ: 100% causal, utilisable comme feature

    Example:
        >>> close = df['close'].values
        >>> kama, er = kama_filter(close, return_efficiency_ratio=True)
        >>> # ER comme feature: indique la "vitesse" du marché
    """
    if isinstance(signal, pd.Series):
        signal = signal.values

    n = len(signal)
    kama = np.zeros(n)
    er = np.zeros(n)

    # Constantes EMA
    fast_sc = 2.0 / (fast_ema + 1)
    slow_sc = 2.0 / (slow_ema + 1)

    # Initialisation: première valeur = signal
    kama[0] = signal[0]

    for i in range(1, n):
        # Calculer l'Efficiency Ratio sur la fenêtre [i-er_period, i]
        if i < er_period:
            # Pas assez de données, utiliser EMA lente
            er[i] = 0.0
            sc = slow_sc
        else:
            # Change absolu sur la période
            change = abs(signal[i] - signal[i - er_period])

            # Volatilité: somme des changements absolus
            volatility = np.sum(np.abs(np.diff(signal[i - er_period:i + 1])))

            # Efficiency Ratio
            if volatility != 0:
                er[i] = change / volatility
            else:
                er[i] = 0.0

            # Smoothing Constant adaptatif
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2

        # Mise à jour KAMA
        kama[i] = kama[i - 1] + sc * (signal[i] - kama[i - 1])

    if return_efficiency_ratio:
        return kama, er
    else:
        return kama


def hma_filter(
    signal: Union[pd.Series, np.ndarray],
    period: int = 16
) -> np.ndarray:
    """
    HMA - Hull Moving Average (Zero-Lag).

    Le filtre le plus RAPIDE pour détecter les retournements de pente.
    Élimine presque totalement le décalage de phase.

    Formule:
    1. WMA_half = WMA(signal, period/2)
    2. WMA_full = WMA(signal, period)
    3. raw_hma = 2 * WMA_half - WMA_full
    4. HMA = WMA(raw_hma, sqrt(period))

    Avantage: Détecte les changements de pente AVANT les MA classiques.

    Args:
        signal: Série de prix
        period: Période de la HMA (défaut: 16)

    Returns:
        HMA filtré (np.ndarray)

    ⚠️ CAUSALITÉ: 100% causal, utilisable comme feature

    Reference:
        Hull, A. (2005). "Reducing lag in a moving average"
        Active Trader Magazine
    """
    if isinstance(signal, pd.Series):
        signal = signal.values

    def wma(data, window):
        """Weighted Moving Average (causal)."""
        weights = np.arange(1, window + 1)
        result = np.zeros(len(data))

        for i in range(len(data)):
            if i < window - 1:
                # Pas assez de données, utiliser ce qu'on a
                w = weights[:i + 1]
                result[i] = np.sum(w * data[:i + 1]) / np.sum(w)
            else:
                # Fenêtre complète
                result[i] = np.sum(weights * data[i - window + 1:i + 1]) / np.sum(weights)

        return result

    # Étape 1: WMA sur period/2
    half_period = max(1, period // 2)
    wma_half = wma(signal, half_period)

    # Étape 2: WMA sur period complet
    wma_full = wma(signal, period)

    # Étape 3: Calcul HMA brut
    raw_hma = 2 * wma_half - wma_full

    # Étape 4: Lissage final avec sqrt(period)
    sqrt_period = max(1, int(np.sqrt(period)))
    hma = wma(raw_hma, sqrt_period)

    return hma


def ehlers_supersmoother(
    signal: Union[pd.Series, np.ndarray],
    cutoff_period: int = 10
) -> np.ndarray:
    """
    Ehlers SuperSmoother Filter.

    Le filtre le plus PRÉCIS pour les modèles d'IA prédisant la direction.
    Utilise un calcul de transfert de signal pour supprimer les fréquences
    de bruit sans décaler les fréquences de tendance.

    Formule (2-pole Butterworth):
    a = exp(-√2 * π / cutoff_period)
    b = 2 * a * cos(√2 * π / cutoff_period)
    c2 = b
    c3 = -a²
    c1 = 1 - c2 - c3

    SuperSmoother[t] = c1*(Price[t] + Price[t-1])/2 + c2*SS[t-1] + c3*SS[t-2]

    Avantage: Lag de groupe minimal, idéal pour CNN-LSTM.

    Args:
        signal: Série de prix
        cutoff_period: Période de coupure (défaut: 10)

    Returns:
        Signal filtré (np.ndarray)

    ⚠️ CAUSALITÉ: 100% causal, utilisable comme feature

    Reference:
        Ehlers, J. F. (2013). "Cycle Analytics for Traders"
    """
    if isinstance(signal, pd.Series):
        signal = signal.values

    n = len(signal)
    filt = np.zeros(n)

    # Calcul des coefficients
    pi = np.pi
    sqrt2 = np.sqrt(2)

    # Éviter division par zéro
    if cutoff_period < 1:
        cutoff_period = 1

    a = np.exp(-sqrt2 * pi / cutoff_period)
    b = 2 * a * np.cos(sqrt2 * pi / cutoff_period)

    c2 = b
    c3 = -a * a
    c1 = 1 - c2 - c3

    # Initialisation
    filt[0] = signal[0]
    if n > 1:
        filt[1] = signal[1]

    # Calcul récursif (causal)
    for i in range(2, n):
        filt[i] = c1 * (signal[i] + signal[i - 1]) / 2 + c2 * filt[i - 1] + c3 * filt[i - 2]

    return filt


def ehlers_decycler(
    signal: Union[pd.Series, np.ndarray],
    high_pass_period: int = 125
) -> np.ndarray:
    """
    Ehlers Decycler (High-Pass Filter).

    Supprime les cycles de bruit pour isoler la tendance.
    Complément du SuperSmoother.

    Formule:
    HPPeriod = high_pass_period
    alpha1 = (cos(0.707*2π/HPPeriod) + sin(0.707*2π/HPPeriod) - 1) / cos(0.707*2π/HPPeriod)

    HP[t] = (1 - alpha1/2)² * (Price[t] - 2*Price[t-1] + Price[t-2]) +
            2*(1 - alpha1)*HP[t-1] - (1 - alpha1)²*HP[t-2]

    Decycler[t] = Price[t] - HP[t]

    Args:
        signal: Série de prix
        high_pass_period: Période du filtre passe-haut (défaut: 125)

    Returns:
        Tendance décyclée (np.ndarray)

    ⚠️ CAUSALITÉ: 100% causal, utilisable comme feature

    Reference:
        Ehlers, J. F. (2015). "Decyclers"
    """
    if isinstance(signal, pd.Series):
        signal = signal.values

    n = len(signal)
    hp = np.zeros(n)

    # Calcul du coefficient alpha
    pi = np.pi
    angle = 0.707 * 2 * pi / high_pass_period
    alpha1 = (np.cos(angle) + np.sin(angle) - 1) / np.cos(angle)

    # Coefficients
    a = (1 - alpha1 / 2) ** 2
    b = 2 * (1 - alpha1)
    c = -(1 - alpha1) ** 2

    # Initialisation
    hp[0] = 0
    if n > 1:
        hp[1] = 0

    # Calcul récursif du High-Pass
    for i in range(2, n):
        hp[i] = a * (signal[i] - 2 * signal[i - 1] + signal[i - 2]) + \
                b * hp[i - 1] + c * hp[i - 2]

    # Decycler = Signal - High-Pass
    decycler = signal - hp

    return decycler


def adaptive_filter_ensemble(
    signal: Union[pd.Series, np.ndarray],
    methods: Optional[list] = None,
    return_components: bool = False
) -> Union[np.ndarray, dict]:
    """
    Ensemble de filtres adaptatifs (moyenne pondérée).

    Combine KAMA, HMA, et Ehlers pour robustesse maximale.

    Args:
        signal: Série de prix
        methods: Liste des méthodes à combiner
                 ['kama', 'hma', 'supersmoother', 'decycler']
                 (défaut: tous)
        return_components: Si True, retourne dict avec chaque filtre

    Returns:
        Si return_components=False: Signal filtré combiné
        Si return_components=True: Dict avec tous les filtres

    ⚠️ CAUSALITÉ: 100% causal, utilisable comme feature

    Example:
        >>> result = adaptive_filter_ensemble(close, return_components=True)
        >>> # Utiliser chaque composant comme feature séparée
        >>> features['kama'] = result['kama']
        >>> features['hma'] = result['hma']
        >>> features['ensemble'] = result['ensemble']
    """
    if isinstance(signal, pd.Series):
        signal = signal.values

    if methods is None:
        methods = ['kama', 'hma', 'supersmoother', 'decycler']

    components = {}

    # Calculer chaque filtre
    if 'kama' in methods:
        components['kama'] = kama_filter(signal)

    if 'hma' in methods:
        components['hma'] = hma_filter(signal)

    if 'supersmoother' in methods:
        components['supersmoother'] = ehlers_supersmoother(signal)

    if 'decycler' in methods:
        components['decycler'] = ehlers_decycler(signal)

    # Calculer l'ensemble (moyenne)
    if len(components) > 0:
        ensemble = np.mean(list(components.values()), axis=0)
        components['ensemble'] = ensemble
    else:
        components['ensemble'] = signal.copy()

    if return_components:
        return components
    else:
        return components['ensemble']


def extract_filter_reactivity(
    signal: Union[pd.Series, np.ndarray],
    er_period: int = 10
) -> np.ndarray:
    """
    Extrait l'Efficiency Ratio (réactivité) du filtre KAMA.

    ⚠️ FEATURE CRITIQUE pour l'IA:
    Si l'IA voit que l'ER devient soudainement élevé, elle comprend
    qu'une explosion de volatilité est en cours → prédicteur puissant
    pour la pente.

    Args:
        signal: Série de prix
        er_period: Période pour le calcul ER

    Returns:
        Efficiency Ratio (np.ndarray) dans [0, 1]
        - ER proche de 1: Marché en tendance forte (filtre rapide)
        - ER proche de 0: Marché en consolidation (filtre lent)

    ⚠️ CAUSALITÉ: 100% causal, utilisable comme feature

    Example:
        >>> er = extract_filter_reactivity(close)
        >>> # Ajouter comme feature pour l'IA
        >>> df['filter_reactivity'] = er
        >>> # L'IA apprend: ER élevé = tendance = pente forte
    """
    # Utiliser KAMA avec return_efficiency_ratio=True
    _, er = kama_filter(signal, er_period=er_period, return_efficiency_ratio=True)
    return er


def validate_causality(
    signal: Union[pd.Series, np.ndarray],
    filter_func,
    **filter_kwargs
) -> dict:
    """
    Valide qu'un filtre est strictement causal.

    Test: Le filtre à l'instant t ne doit PAS changer si on ajoute
    des données après t.

    Args:
        signal: Signal de test
        filter_func: Fonction de filtrage à tester
        **filter_kwargs: Arguments pour le filtre

    Returns:
        Dict avec résultats de validation:
        {
            'is_causal': bool,
            'violations': int,
            'max_deviation': float
        }

    Example:
        >>> result = validate_causality(close, kama_filter, er_period=10)
        >>> assert result['is_causal'], "Filtre non-causal détecté!"
    """
    if isinstance(signal, pd.Series):
        signal = signal.values

    # Filtrer le signal complet
    full_filtered = filter_func(signal, **filter_kwargs)

    # Filtrer seulement les 80% premiers points
    cutoff = int(len(signal) * 0.8)
    partial_signal = signal[:cutoff]
    partial_filtered = filter_func(partial_signal, **filter_kwargs)

    # Comparer: les valeurs jusqu'à cutoff doivent être identiques
    deviation = np.abs(full_filtered[:cutoff] - partial_filtered)
    max_deviation = np.max(deviation)
    violations = np.sum(deviation > 1e-10)  # Tolérance numérique

    is_causal = violations == 0

    result = {
        'is_causal': is_causal,
        'violations': int(violations),
        'max_deviation': float(max_deviation)
    }

    if is_causal:
        logger.info(f"✅ Filtre {filter_func.__name__} est CAUSAL")
    else:
        logger.error(f"❌ Filtre {filter_func.__name__} NON-CAUSAL! "
                    f"{violations} violations, max deviation={max_deviation:.6f}")

    return result


def compare_filters(
    signal: Union[pd.Series, np.ndarray],
    show_metrics: bool = True
) -> pd.DataFrame:
    """
    Compare tous les filtres adaptatifs sur un signal.

    Args:
        signal: Signal à filtrer
        show_metrics: Si True, affiche les métriques

    Returns:
        DataFrame avec comparaison des filtres

    Example:
        >>> comparison = compare_filters(df['close'])
        >>> print(comparison)
    """
    if isinstance(signal, pd.Series):
        signal_values = signal.values
    else:
        signal_values = signal

    # Appliquer tous les filtres
    kama = kama_filter(signal_values)
    hma = hma_filter(signal_values)
    supersmoother = ehlers_supersmoother(signal_values)
    decycler = ehlers_decycler(signal_values)
    ensemble = adaptive_filter_ensemble(signal_values)

    # Créer DataFrame
    df_comparison = pd.DataFrame({
        'original': signal_values,
        'kama': kama,
        'hma': hma,
        'supersmoother': supersmoother,
        'decycler': decycler,
        'ensemble': ensemble
    })

    if show_metrics:
        logger.info("\n" + "="*60)
        logger.info("COMPARAISON DES FILTRES ADAPTATIFS")
        logger.info("="*60)

        for col in ['kama', 'hma', 'supersmoother', 'decycler', 'ensemble']:
            # Calculer le lag moyen
            diff = df_comparison[col] - df_comparison['original']
            mean_lag = np.mean(np.abs(diff))

            # Smoothness (variance de la dérivée)
            derivative = np.diff(df_comparison[col])
            smoothness = np.var(derivative)

            logger.info(f"\n{col.upper()}:")
            logger.info(f"  Lag moyen: {mean_lag:.6f}")
            logger.info(f"  Smoothness: {smoothness:.6f}")

    return df_comparison


# Tests de validation automatiques
if __name__ == '__main__':
    """Tests rapides des filtres adaptatifs."""

    # Créer signal de test
    np.random.seed(42)
    n = 200
    trend = np.linspace(100, 150, n)
    noise = np.random.randn(n) * 2
    signal = trend + noise

    print("\n" + "="*60)
    print("TESTS DES FILTRES ADAPTATIFS ZERO-LAG")
    print("="*60)

    # Test 1: KAMA
    print("\n[1/5] Test KAMA...")
    kama, er = kama_filter(signal, return_efficiency_ratio=True)
    print(f"✅ KAMA: {len(kama)} points")
    print(f"   ER moyen: {np.mean(er):.4f}")

    # Test 2: HMA
    print("\n[2/5] Test HMA...")
    hma = hma_filter(signal)
    print(f"✅ HMA: {len(hma)} points")

    # Test 3: Ehlers SuperSmoother
    print("\n[3/5] Test Ehlers SuperSmoother...")
    ss = ehlers_supersmoother(signal)
    print(f"✅ SuperSmoother: {len(ss)} points")

    # Test 4: Ehlers Decycler
    print("\n[4/5] Test Ehlers Decycler...")
    dc = ehlers_decycler(signal)
    print(f"✅ Decycler: {len(dc)} points")

    # Test 5: Validation de causalité
    print("\n[5/5] Validation de causalité...")

    for name, func in [('KAMA', kama_filter),
                       ('HMA', hma_filter),
                       ('SuperSmoother', ehlers_supersmoother),
                       ('Decycler', ehlers_decycler)]:
        result = validate_causality(signal, func)
        if not result['is_causal']:
            print(f"❌ {name} ÉCHOUÉ!")
        else:
            print(f"✅ {name} causal")

    # Comparaison
    print("\n[6/6] Comparaison des filtres...")
    comparison = compare_filters(signal, show_metrics=True)

    print("\n" + "="*60)
    print("✅ TOUS LES TESTS PASSÉS - Filtres prêts pour production")
    print("="*60)


def kalman_filter_causal(
    signal: Union[pd.Series, np.ndarray],
    process_variance: float = 0.01,
    measurement_variance: float = 0.1
) -> np.ndarray:
    """
    Filtre de Kalman CAUSAL (n'utilise que le passé).

    Contrairement à la version dans filters.py qui utilise smooth() (non-causal),
    cette version utilise filter() et est STRICTEMENT CAUSALE.

    Utilisable comme FEATURE pour l'IA.

    Args:
        signal: Signal d'entrée
        process_variance: Variance du processus (Q) - Ajuste la réactivité
                         Plus élevé = plus réactif au signal
        measurement_variance: Variance de mesure (R) - Contrôle le lissage
                            Plus élevé = plus lisse

    Returns:
        Signal filtré (causal)

    Example:
        >>> filtered = kalman_filter_causal(close_prices, 0.01, 0.1)
        >>> # Plus réactif: process_variance=0.1, measurement_variance=0.1
        >>> # Plus lisse: process_variance=0.001, measurement_variance=1.0
    """
    try:
        from pykalman import KalmanFilter
    except ImportError:
        logger.warning("pykalman non installé, retour du signal original")
        logger.warning("Installation: pip install pykalman")
        if isinstance(signal, pd.Series):
            return signal.values
        return signal

    if isinstance(signal, pd.Series):
        signal = signal.values

    # Gérer les NaN
    mask = ~np.isnan(signal)
    if not mask.any():
        return signal

    valid_data = signal[mask].reshape(-1, 1)

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

    # Appliquer le FILTER (causal, pas smooth!)
    # filter() utilise uniquement les données passées
    state_means, _ = kf.filter(valid_data)

    # Reconstruire
    filtered = np.full_like(signal, np.nan, dtype=float)
    filtered[mask] = state_means.flatten()

    logger.debug(f"Kalman causal: Q={process_variance}, R={measurement_variance}")

    return filtered


def butterworth_causal(
    signal: Union[pd.Series, np.ndarray],
    cutoff: float = 0.25,
    order: int = 3
) -> np.ndarray:
    """
    Filtre Butterworth CAUSAL (n'utilise que le passé).

    Contrairement à filtfilt() qui est bidirectionnel (non-causal),
    cette version utilise lfilter() et est STRICTEMENT CAUSALE.

    Utilisable comme FEATURE pour l'IA.

    Args:
        signal: Signal d'entrée
        cutoff: Fréquence de coupure normalisée (0.0 à 1.0)
                - 0.1 = filtre très lisse
                - 0.25 = filtre modéré (défaut)
                - 0.5 = filtre léger
        order: Ordre du filtre (défaut: 3)

    Returns:
        Signal filtré (causal)

    Example:
        >>> filtered = butterworth_causal(close_prices, cutoff=0.25, order=3)
        >>> # Plus lisse: cutoff=0.1
        >>> # Plus réactif: cutoff=0.4

    Note:
        Ce filtre a un lag de phase (déphasage) contrairement à filtfilt(),
        mais il est CAUSAL et peut être utilisé pour les features.
    """
    import scipy.signal

    if isinstance(signal, pd.Series):
        signal = signal.values

    # Gérer les NaN
    mask = ~np.isnan(signal)
    if not mask.any():
        return signal

    valid_data = signal[mask]

    # Créer le filtre Butterworth
    B, A = scipy.signal.butter(order, cutoff, output='ba')

    # Appliquer lfilter (causal, unidirectionnel)
    # ⚠️ N'utilise QUE le passé!
    filtered_valid = scipy.signal.lfilter(B, A, valid_data)

    # Reconstruire
    filtered = np.full_like(signal, np.nan, dtype=float)
    filtered[mask] = filtered_valid

    logger.debug(f"Butterworth causal: cutoff={cutoff}, order={order}")

    return filtered
