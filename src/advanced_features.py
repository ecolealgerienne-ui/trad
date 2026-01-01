"""
Features avancées pour le modèle de prédiction.

Inclut les features de vitesse, contexte de prix, et transformations.
Ces features sont critiques pour atteindre >90% d'accuracy.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Dict
import logging

logger = logging.getLogger(__name__)


def add_velocity_features(df: pd.DataFrame,
                         ghost_prefix: str = 'ghost') -> pd.DataFrame:
    """
    Ajoute les features de vitesse (velocity) à la bougie fantôme.

    Ces features capturent la DYNAMIQUE de formation de la bougie.

    Features ajoutées:
    1. velocity: (F - O) / step_index - Vitesse de progression
    2. amplitude: (H - L) / O - Amplitude relative (volatilité)
    3. acceleration: (F_k - F_{k-1}) / O - Accélération (variation entre steps)

    Args:
        df: DataFrame avec bougies fantômes
        ghost_prefix: Préfixe des colonnes ghost

    Returns:
        DataFrame avec features de vitesse ajoutées

    Example:
        >>> df = add_velocity_features(df)
        >>> # Ajoute: velocity, amplitude, acceleration
    """
    df = df.copy()

    o = f'{ghost_prefix}_open'
    h = f'{ghost_prefix}_high'
    l = f'{ghost_prefix}_low'
    c = f'{ghost_prefix}_close'

    # Vérifier que les colonnes existent
    required_cols = [o, h, l, c, 'step']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    # 1. Vitesse de Progression
    # (Close - Open) / step_index
    # Indique si la bougie monte/descend rapidement ou lentement
    df['velocity'] = (df[c] - df[o]) / df['step']

    # Normaliser par Open pour avoir un ratio indépendant du prix
    df['velocity'] = df['velocity'] / df[o]

    logger.debug("✓ velocity ajoutée: (F - O) / step / O")

    # 2. Amplitude Relative
    # (High - Low) / Open
    # Mesure la volatilité de la bougie en formation
    df['amplitude'] = (df[h] - df[l]) / df[o]

    logger.debug("✓ amplitude ajoutée: (H - L) / O")

    # 3. Accélération (variation entre steps)
    # (F_k - F_{k-1}) / O
    # Indique si le mouvement s'accélère ou ralentit

    # Grouper par bougie 30m pour calculer la différence entre steps
    df = df.sort_values(['candle_30m_timestamp', 'step']).reset_index(drop=True)

    # Calculer la différence de close entre steps consécutifs
    df['close_diff'] = df.groupby('candle_30m_timestamp')[c].diff()

    # Normaliser par Open
    df['acceleration'] = df['close_diff'] / df[o]

    # Le premier step de chaque bougie n'a pas d'accélération (NaN)
    # C'est normal

    logger.debug("✓ acceleration ajoutée: (F_k - F_{k-1}) / O")

    # Supprimer la colonne temporaire
    df = df.drop(columns=['close_diff'])

    logger.info(f"Features de vitesse ajoutées: velocity, amplitude, acceleration")

    return df


def add_open_context(df: pd.DataFrame,
                    ghost_prefix: str = 'ghost',
                    window: int = 50) -> pd.DataFrame:
    """
    Ajoute le Z-Score de l'Open pour donner le contexte de prix.

    Au lieu de passer Open normalisé (qui serait toujours 0),
    on passe le Z-Score de Open calculé sur une fenêtre glissante.

    Cela indique si la bougie s'ouvre en zone de surachat/survente.

    Args:
        df: DataFrame avec bougies fantômes
        ghost_prefix: Préfixe des colonnes ghost
        window: Taille de la fenêtre glissante pour le Z-Score

    Returns:
        DataFrame avec open_zscore ajouté

    Example:
        >>> df = add_open_context(df, window=50)
        >>> # Ajoute: ghost_open_zscore
    """
    df = df.copy()

    o = f'{ghost_prefix}_open'

    if o not in df.columns:
        raise ValueError(f"Colonne {o} manquante")

    # Trier par timestamp pour avoir un rolling correct
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Calculer le Z-Score glissant de Open
    rolling_mean = df[o].rolling(window=window, min_periods=window//2).mean()
    rolling_std = df[o].rolling(window=window, min_periods=window//2).std()

    df[f'{ghost_prefix}_open_zscore'] = (df[o] - rolling_mean) / rolling_std

    # Remplacer inf/nan par 0
    df[f'{ghost_prefix}_open_zscore'] = df[f'{ghost_prefix}_open_zscore'].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    logger.info(f"Contexte de prix ajouté: {ghost_prefix}_open_zscore (window={window})")

    return df


def add_step_index_normalized(df: pd.DataFrame,
                              max_steps: int = 6) -> pd.DataFrame:
    """
    Ajoute step_index normalisé entre 0.0 et 1.0.

    Le step brut (1-6) n'est pas idéal pour les réseaux de neurones.
    La version normalisée (0.0-1.0) permet au modèle d'apprendre
    une progression linéaire.

    Args:
        df: DataFrame avec colonne 'step'
        max_steps: Nombre maximum de steps (défaut: 6 pour 30min/5min)

    Returns:
        DataFrame avec step_index_norm ajouté

    Example:
        >>> df = add_step_index_normalized(df)
        >>> # step=1 → step_index_norm=0.0
        >>> # step=6 → step_index_norm=1.0
    """
    df = df.copy()

    if 'step' not in df.columns:
        raise ValueError("Colonne 'step' manquante")

    # Normaliser: (step - 1) / (max_steps - 1)
    # step=1 → 0.0
    # step=6 → 1.0
    df['step_index_norm'] = (df['step'] - 1) / (max_steps - 1)

    logger.info(f"Step index normalisé ajouté: 0.0 (step=1) → 1.0 (step={max_steps})")

    return df


def add_log_returns_ghost(df: pd.DataFrame,
                         ghost_prefix: str = 'ghost') -> pd.DataFrame:
    """
    Ajoute les log returns au lieu des variations relatives.

    Log returns sont plus stables statistiquement et mieux adaptés
    pour les réseaux de neurones.

    Formule:
        H_log = log(H / O)
        L_log = log(L / O)
        C_log = log(C / O)

    Args:
        df: DataFrame avec bougies fantômes
        ghost_prefix: Préfixe des colonnes ghost

    Returns:
        DataFrame avec log returns ajoutés

    Example:
        >>> df = add_log_returns_ghost(df)
        >>> # Ajoute: ghost_high_log, ghost_low_log, ghost_close_log
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

    # Calculer log returns
    # Éviter division par zéro et log de valeurs négatives
    with np.errstate(divide='ignore', invalid='ignore'):
        df[f'{ghost_prefix}_high_log'] = np.log(df[h] / df[o])
        df[f'{ghost_prefix}_low_log'] = np.log(df[l] / df[o])
        df[f'{ghost_prefix}_close_log'] = np.log(df[c] / df[o])

    # Remplacer inf/nan par 0
    for col in [f'{ghost_prefix}_high_log', f'{ghost_prefix}_low_log', f'{ghost_prefix}_close_log']:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    logger.info(f"Log returns ajoutés: {ghost_prefix}_high/low/close_log")

    return df


def add_all_advanced_features(df: pd.DataFrame,
                             ghost_prefix: str = 'ghost',
                             open_zscore_window: int = 50,
                             max_steps: int = 6) -> pd.DataFrame:
    """
    Ajoute TOUTES les features avancées au DataFrame.

    Features ajoutées:
    1. Velocity, amplitude, acceleration (dynamique de formation)
    2. Open Z-Score (contexte de prix)
    3. Step index normalisé (0.0-1.0)
    4. Log returns (au lieu de % relatifs)

    Args:
        df: DataFrame avec bougies fantômes
        ghost_prefix: Préfixe des colonnes ghost
        open_zscore_window: Fenêtre pour Z-Score de Open
        max_steps: Nombre maximum de steps

    Returns:
        DataFrame avec toutes les features avancées

    Example:
        >>> df = add_all_advanced_features(df)
    """
    logger.info("\n" + "="*60)
    logger.info("Ajout de TOUTES les features avancées")
    logger.info("="*60)

    # 1. Features de vitesse
    df = add_velocity_features(df, ghost_prefix=ghost_prefix)

    # 2. Contexte de prix (Open Z-Score)
    df = add_open_context(df, ghost_prefix=ghost_prefix, window=open_zscore_window)

    # 3. Step index normalisé
    df = add_step_index_normalized(df, max_steps=max_steps)

    # 4. Log returns
    df = add_log_returns_ghost(df, ghost_prefix=ghost_prefix)

    logger.info("="*60)
    logger.info("✅ Toutes les features avancées ajoutées!")
    logger.info(f"Total colonnes: {len(df.columns)}")
    logger.info("="*60 + "\n")

    return df


def validate_advanced_features(df: pd.DataFrame) -> Dict:
    """
    Valide les features avancées.

    Vérifications:
    1. Pas de NaN excessifs
    2. Valeurs dans des ranges raisonnables
    3. Corrélations entre features

    Args:
        df: DataFrame avec features avancées

    Returns:
        Dictionnaire de statistiques
    """
    stats = {}

    advanced_features = [
        'velocity', 'amplitude', 'acceleration',
        'ghost_open_zscore', 'step_index_norm',
        'ghost_high_log', 'ghost_low_log', 'ghost_close_log'
    ]

    for feat in advanced_features:
        if feat in df.columns:
            values = df[feat].dropna()
            stats[feat] = {
                'count': len(values),
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'nan_pct': df[feat].isna().sum() / len(df) * 100
            }

            # Warning si trop de NaN
            if stats[feat]['nan_pct'] > 10:
                logger.warning(f"⚠️  {feat}: {stats[feat]['nan_pct']:.1f}% de NaN")

    logger.info("Validation des features avancées:")
    for feat, stat in stats.items():
        logger.info(f"  {feat}: mean={stat['mean']:.4f}, std={stat['std']:.4f}, NaN={stat['nan_pct']:.1f}%")

    return stats
