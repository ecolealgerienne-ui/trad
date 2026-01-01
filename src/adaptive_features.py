"""
Features basées sur les filtres adaptatifs zero-lag.

ARCHITECTURE MISE À JOUR (2026-01-01):

FEATURES (X) - CAUSALES:
- Filtres adaptatifs: KAMA, HMA, SuperSmoother, Decycler
- Efficiency Ratio (vitesse de l'alpha)
- Dérivées des filtres (pente)

LABELS (Y) - NON-CAUSALES:
- filtfilt (Butterworth bidirectionnel) - TARGET IDÉAL

Cette séparation est CRITIQUE pour atteindre >90% accuracy.

Référence: Mise à jour Spec #1 (2025-12-20)
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, List
import logging

from adaptive_filters import (
    kama_filter,
    hma_filter,
    ehlers_supersmoother,
    ehlers_decycler,
    extract_filter_reactivity,
    adaptive_filter_ensemble
)

logger = logging.getLogger(__name__)


def add_adaptive_filter_features(
    df: pd.DataFrame,
    source_col: str = 'current_5m_close',
    filters: Optional[List[str]] = None,
    add_slopes: bool = True,
    add_reactivity: bool = True
) -> pd.DataFrame:
    """
    Ajoute les features basées sur les filtres adaptatifs.

    ⚠️ IMPORTANT: Ces filtres sont CAUSAUX, utilisables comme features (X).

    Args:
        df: DataFrame avec données
        source_col: Colonne source pour filtrage
        filters: Liste des filtres à ajouter
                 ['kama', 'hma', 'supersmoother', 'decycler', 'ensemble']
                 (défaut: tous)
        add_slopes: Si True, ajoute les pentes (dérivées)
        add_reactivity: Si True, ajoute l'Efficiency Ratio (vitesse alpha)

    Returns:
        DataFrame avec nouvelles features ajoutées

    Features ajoutées:
    - {filter}_filtered: Signal filtré
    - {filter}_slope: Pente du signal filtré (optionnel)
    - filter_reactivity: Efficiency Ratio KAMA (optionnel)

    Example:
        >>> df = add_adaptive_filter_features(df, source_col='current_5m_close')
        >>> # Maintenant df contient:
        >>> # - kama_filtered, hma_filtered, etc.
        >>> # - kama_slope, hma_slope, etc.
        >>> # - filter_reactivity (vitesse du marché)
    """
    logger.info("\n" + "="*60)
    logger.info("AJOUT DES FEATURES FILTRES ADAPTATIFS ZERO-LAG")
    logger.info("="*60)

    if source_col not in df.columns:
        logger.error(f"❌ Colonne source '{source_col}' non trouvée!")
        return df

    signal = df[source_col].values

    if filters is None:
        filters = ['kama', 'hma', 'supersmoother', 'decycler', 'ensemble']

    # Ajouter chaque filtre
    for filter_name in filters:
        col_name = f'{filter_name}_filtered'

        logger.info(f"\n[{filter_name.upper()}] Application du filtre...")

        if filter_name == 'kama':
            filtered = kama_filter(signal)
        elif filter_name == 'hma':
            filtered = hma_filter(signal)
        elif filter_name == 'supersmoother':
            filtered = ehlers_supersmoother(signal)
        elif filter_name == 'decycler':
            filtered = ehlers_decycler(signal)
        elif filter_name == 'ensemble':
            filtered = adaptive_filter_ensemble(signal)
        else:
            logger.warning(f"Filtre inconnu: {filter_name}, ignoré")
            continue

        df[col_name] = filtered
        logger.info(f"✅ {col_name} ajouté")

        # Ajouter la pente (dérivée)
        if add_slopes:
            slope_col = f'{filter_name}_slope'
            # Pente = différence entre t et t-1
            df[slope_col] = df[col_name].diff()
            logger.info(f"✅ {slope_col} ajouté")

    # Ajouter l'Efficiency Ratio (réactivité)
    if add_reactivity:
        logger.info("\n[REACTIVITY] Extraction de l'Efficiency Ratio...")
        er = extract_filter_reactivity(signal)
        df['filter_reactivity'] = er
        logger.info("✅ filter_reactivity ajouté")
        logger.info(f"   ER moyen: {np.mean(er[~np.isnan(er)]):.4f}")

    logger.info("\n" + "="*60)
    logger.info(f"✅ Features filtres adaptatifs ajoutées: {len([f for f in filters])} filtres")
    logger.info("="*60)

    return df


def add_rsi_adaptive_features(
    df: pd.DataFrame,
    rsi_col: str = 'rsi_14',
    filters: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Applique les filtres adaptatifs sur le RSI.

    Le RSI filtré est souvent PLUS prédictif que le prix filtré pour
    détecter les retournements.

    Args:
        df: DataFrame avec RSI
        rsi_col: Colonne RSI source
        filters: Filtres à appliquer (défaut: ['kama', 'supersmoother'])

    Returns:
        DataFrame avec RSI filtré

    Features ajoutées:
    - rsi_{filter}_filtered
    - rsi_{filter}_slope

    Example:
        >>> df = add_rsi_adaptive_features(df, rsi_col='rsi_14')
        >>> # Utiliser rsi_kama_filtered comme feature
    """
    if rsi_col not in df.columns:
        logger.warning(f"Colonne RSI '{rsi_col}' non trouvée, skip RSI filtering")
        return df

    logger.info("\n[RSI] Application des filtres adaptatifs sur RSI...")

    rsi_values = df[rsi_col].values

    if filters is None:
        filters = ['kama', 'supersmoother']

    for filter_name in filters:
        col_name = f'rsi_{filter_name}_filtered'

        if filter_name == 'kama':
            filtered = kama_filter(rsi_values)
        elif filter_name == 'hma':
            filtered = hma_filter(rsi_values)
        elif filter_name == 'supersmoother':
            filtered = ehlers_supersmoother(rsi_values)
        elif filter_name == 'decycler':
            filtered = ehlers_decycler(rsi_values)
        else:
            continue

        df[col_name] = filtered

        # Pente du RSI filtré
        slope_col = f'rsi_{filter_name}_slope'
        df[slope_col] = df[col_name].diff()

        logger.info(f"✅ {col_name} + {slope_col} ajoutés")

    return df


def create_adaptive_label(
    df: pd.DataFrame,
    source_col: str = 'current_5m_close',
    filter_method: str = 'kama',
    slope_threshold: float = 0.0,
    temporal_offset: int = 1
) -> pd.DataFrame:
    """
    Crée des labels basés sur les filtres adaptatifs CAUSAUX.

    ALTERNATIVE aux labels non-causaux (filtfilt):
    Utilise un filtre adaptatif + décalage temporel pour créer
    un label "presque aussi bon" que filtfilt mais TOTALEMENT causal.

    Args:
        df: DataFrame
        source_col: Colonne source
        filter_method: Méthode de filtrage ('kama', 'hma', 'supersmoother')
        slope_threshold: Seuil pour binariser (défaut: 0.0)
        temporal_offset: Décalage temporel (défaut: 1)
                        Label[t] = signe(pente[t+offset])

    Returns:
        DataFrame avec colonne 'label_adaptive'

    ⚠️ NOTE: Cette approche est une ALTERNATIVE au filtfilt.
    Pour la Spec #1, on garde filtfilt pour les labels (Y).
    Mais cette méthode peut être testée pour comparaison.

    Example:
        >>> df = create_adaptive_label(df, filter_method='kama')
        >>> # Comparer label_adaptive vs label (filtfilt)
    """
    logger.info(f"\n[LABEL ADAPTATIF] Création avec filtre {filter_method}...")

    signal = df[source_col].values

    # Appliquer le filtre
    if filter_method == 'kama':
        filtered = kama_filter(signal)
    elif filter_method == 'hma':
        filtered = hma_filter(signal)
    elif filter_method == 'supersmoother':
        filtered = ehlers_supersmoother(signal)
    else:
        filtered = signal

    # Calculer la pente
    slope = np.diff(filtered, prepend=filtered[0])

    # Décalage temporel (prédire pente future)
    slope_shifted = np.roll(slope, -temporal_offset)
    slope_shifted[-temporal_offset:] = np.nan

    # Binariser
    label = np.where(slope_shifted > slope_threshold, 1, 0)
    label = np.where(np.isnan(slope_shifted), np.nan, label)

    df['label_adaptive'] = label

    # Statistiques
    valid_labels = label[~np.isnan(label)]
    if len(valid_labels) > 0:
        pos_pct = np.sum(valid_labels == 1) / len(valid_labels) * 100
        logger.info(f"✅ label_adaptive créé: {pos_pct:.1f}% hausse")

    return df


def validate_adaptive_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None
) -> dict:
    """
    Valide que les features adaptatives sont correctement calculées.

    Vérifications:
    - Pas de NaN excessifs (>10%)
    - Slopes dans un range raisonnable
    - Reactivity dans [0, 1]

    Args:
        df: DataFrame avec features
        feature_cols: Colonnes à valider (défaut: auto-detect)

    Returns:
        Dict avec résultats de validation
    """
    logger.info("\n" + "="*60)
    logger.info("VALIDATION DES FEATURES ADAPTATIVES")
    logger.info("="*60)

    issues = []

    # Auto-detect si non spécifié
    if feature_cols is None:
        feature_cols = [col for col in df.columns
                       if any(x in col for x in ['kama', 'hma', 'supersmoother',
                                                 'decycler', 'ensemble', 'reactivity'])]

    for col in feature_cols:
        if col not in df.columns:
            continue

        values = df[col].dropna()

        # Vérif 1: NaN
        nan_pct = (df[col].isna().sum() / len(df)) * 100
        if nan_pct > 10:
            issue = f"{col}: {nan_pct:.1f}% NaN (>10%)"
            issues.append(issue)
            logger.warning(f"⚠️  {issue}")

        # Vérif 2: Reactivity dans [0, 1]
        if 'reactivity' in col:
            if not ((values >= 0) & (values <= 1)).all():
                issue = f"{col}: valeurs hors [0, 1]"
                issues.append(issue)
                logger.error(f"❌ {issue}")
            else:
                logger.info(f"✅ {col}: range [0, 1] OK")

        # Vérif 3: Slopes raisonnables
        if 'slope' in col:
            slope_std = values.std()
            if slope_std > 100:  # Threshold arbitraire
                issue = f"{col}: std très élevée ({slope_std:.2f})"
                issues.append(issue)
                logger.warning(f"⚠️  {issue}")

    logger.info("\n" + "="*60)
    if not issues:
        logger.info("✅ VALIDATION RÉUSSIE - Toutes les features adaptatives OK")
    else:
        logger.warning(f"⚠️  {len(issues)} issues détectées")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'features_checked': len(feature_cols)
    }


# Tests
if __name__ == '__main__':
    """Test rapide des features adaptatives."""

    # Créer données test
    n = 100
    timestamps = pd.date_range('2024-01-01', periods=n, freq='5min')
    prices = 50000 + np.cumsum(np.random.randn(n) * 100)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'current_5m_close': prices,
        'rsi_14': 50 + np.random.randn(n) * 10
    })

    print("\n" + "="*60)
    print("TEST DES FEATURES ADAPTATIVES")
    print("="*60)

    # Test 1: Ajouter features filtres adaptatifs
    df = add_adaptive_filter_features(df, source_col='current_5m_close')

    print(f"\n✅ Features ajoutées: {len(df.columns)} colonnes")
    print(f"Colonnes: {list(df.columns)}")

    # Test 2: RSI adaptatif
    df = add_rsi_adaptive_features(df, rsi_col='rsi_14')

    # Test 3: Validation
    result = validate_adaptive_features(df)

    print("\n" + "="*60)
    if result['valid']:
        print("✅ VALIDATION RÉUSSIE")
    else:
        print(f"⚠️  {len(result['issues'])} issues")

    print("="*60)
