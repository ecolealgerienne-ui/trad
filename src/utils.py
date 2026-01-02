"""
Fonctions utilitaires communes pour le pipeline de données crypto.

IMPORTANT: Ce module contient TOUTES les fonctions partagées entre scripts.
TOUJOURS vérifier si une fonction existe ici avant d'en créer une nouvelle.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_ohlcv_data(filepath: str,
                    parse_dates: bool = True,
                    date_column: Optional[str] = None) -> pd.DataFrame:
    """
    Charge les données OHLCV depuis un fichier CSV.

    Args:
        filepath: Chemin vers le fichier CSV
        parse_dates: Si True, parse la colonne date
        date_column: Nom de la colonne date (auto-détecté si None)

    Returns:
        DataFrame avec colonnes [timestamp, open, high, low, close, volume]

    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        ValueError: Si les colonnes requises sont manquantes
    """
    logger.info(f"Chargement des données depuis {filepath}")

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        logger.error(f"Fichier introuvable: {filepath}")
        raise

    # Normaliser les noms de colonnes (lowercase)
    df.columns = df.columns.str.lower().str.strip()

    # Vérifier les colonnes OHLCV
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans {filepath}: {missing_cols}")

    # Détecter et parser la colonne de date
    if parse_dates:
        date_cols = [col for col in df.columns if any(
            keyword in col for keyword in ['time', 'date', 'timestamp']
        )]

        if date_cols:
            date_col = date_column or date_cols[0]
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.rename(columns={date_col: 'timestamp'})
            df = df.sort_values('timestamp').reset_index(drop=True)
        else:
            logger.warning("Aucune colonne de date détectée")

    logger.info(f"Données chargées: {len(df)} lignes, période: {df['timestamp'].min()} à {df['timestamp'].max()}")

    return df


def validate_ohlc_integrity(df: pd.DataFrame,
                           col_prefix: str = '') -> bool:
    """
    Valide l'intégrité des données OHLC (High >= Low, etc.).

    Args:
        df: DataFrame avec colonnes OHLC
        col_prefix: Préfixe des colonnes (ex: 'ghost_' pour ghost_open)

    Returns:
        True si les données sont valides

    Raises:
        ValueError: Si les données sont invalides
    """
    o = f'{col_prefix}open'
    h = f'{col_prefix}high'
    l = f'{col_prefix}low'
    c = f'{col_prefix}close'

    # Vérifier que High >= Low
    invalid_hl = df[df[h] < df[l]]
    if len(invalid_hl) > 0:
        raise ValueError(f"{len(invalid_hl)} lignes avec High < Low détectées")

    # Vérifier que High >= Open et Close
    invalid_h = df[(df[h] < df[o]) | (df[h] < df[c])]
    if len(invalid_h) > 0:
        raise ValueError(f"{len(invalid_h)} lignes avec High < Open/Close détectées")

    # Vérifier que Low <= Open et Close
    invalid_l = df[(df[l] > df[o]) | (df[l] > df[c])]
    if len(invalid_l) > 0:
        raise ValueError(f"{len(invalid_l)} lignes avec Low > Open/Close détectées")

    # Vérifier les valeurs nulles
    null_count = df[[o, h, l, c]].isnull().sum().sum()
    if null_count > 0:
        raise ValueError(f"{null_count} valeurs nulles détectées dans OHLC")

    logger.info("Validation OHLC: OK")
    return True


def resample_to_timeframe(df: pd.DataFrame,
                         timeframe: str = '30min',
                         ohlcv_only: bool = True) -> pd.DataFrame:
    """
    Resampling des données vers un timeframe plus large.

    Args:
        df: DataFrame avec timestamp en index ou colonne
        timeframe: Timeframe cible (ex: '30min' pour 30 minutes)
        ohlcv_only: Si True, ne garde que les colonnes OHLCV

    Returns:
        DataFrame resampé
    """
    logger.info(f"Resampling vers {timeframe}")

    # S'assurer que timestamp est l'index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    # Agrégation OHLCV
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }

    if 'volume' in df.columns:
        agg_dict['volume'] = 'sum'

    # Ajouter les autres colonnes si demandé
    if not ohlcv_only:
        for col in df.columns:
            if col not in agg_dict:
                agg_dict[col] = 'last'  # Prendre la dernière valeur

    df_resampled = df.resample(timeframe).agg(agg_dict)
    df_resampled = df_resampled.dropna().reset_index()

    logger.info(f"Resampé: {len(df_resampled)} bougies")

    return df_resampled


def calculate_returns(prices: Union[pd.Series, np.ndarray],
                     periods: int = 1) -> np.ndarray:
    """
    Calcule les returns (rendements) logarithmiques ou simples.

    Args:
        prices: Série de prix
        periods: Nombre de périodes pour le calcul

    Returns:
        Array des returns
    """
    if isinstance(prices, pd.Series):
        prices = prices.values

    # Returns logarithmiques
    returns = np.diff(np.log(prices), n=periods)

    # Ajouter des NaN au début pour garder la même longueur
    returns = np.concatenate([np.full(periods, np.nan), returns])

    return returns


def check_data_leakage(df: pd.DataFrame,
                      feature_cols: list,
                      label_col: str = 'label') -> dict:
    """
    Vérifie qu'il n'y a pas de data leakage (corrélation future).

    Args:
        df: DataFrame avec features et label
        feature_cols: Liste des colonnes de features
        label_col: Nom de la colonne label

    Returns:
        Dictionnaire avec statistiques de vérification
    """
    logger.info("Vérification du data leakage...")

    results = {
        'future_correlation': {},
        'suspicious_features': []
    }

    # Vérifier la corrélation entre features[t] et label[t+1]
    for col in feature_cols:
        if col in df.columns:
            # Corrélation avec label futur
            future_corr = df[col].corr(df[label_col].shift(-1))
            results['future_correlation'][col] = future_corr

            # Si corrélation > 0.7 avec le futur, c'est suspect
            if abs(future_corr) > 0.7:
                results['suspicious_features'].append(col)
                logger.warning(f"⚠️  {col} a une corrélation de {future_corr:.3f} avec label[t+1]")

    if not results['suspicious_features']:
        logger.info("✅ Aucun data leakage détecté")
    else:
        logger.error(f"❌ {len(results['suspicious_features'])} features suspectes détectées")

    return results


def split_train_test_temporal(df: pd.DataFrame,
                              test_size: float = 0.2,
                              validation_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split temporal (JAMAIS de shuffle pour les séries temporelles).

    Args:
        df: DataFrame trié par timestamp
        test_size: Proportion du test set
        validation_size: Proportion du validation set

    Returns:
        (train_df, val_df, test_df)
    """
    n = len(df)

    test_idx = int(n * (1 - test_size))
    val_idx = int(test_idx * (1 - validation_size))

    train_df = df[:val_idx].copy()
    val_df = df[val_idx:test_idx].copy()
    test_df = df[test_idx:].copy()

    logger.info(f"Split temporal: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    return train_df, val_df, test_df


def split_train_val_test_with_gap(df: pd.DataFrame,
                                  train_end_date: str,
                                  val_start_date: str,
                                  val_end_date: str,
                                  test_start_date: str,
                                  timestamp_col: str = 'timestamp') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split temporal avec GAP PERIOD pour éviter la contamination.

    ⚠️ CRITIQUE pour éviter le data leakage avec les filtres non-causaux!

    Le gap period (période de buffer) évite que les données de fin de train
    soient influencées par le début de validation à cause des filtres bidirectionnels.

    Args:
        df: DataFrame avec timestamp
        train_end_date: Date de fin du train (ex: '2023-10-31')
        val_start_date: Date de début validation APRÈS gap (ex: '2023-11-07')
        val_end_date: Date de fin validation (ex: '2023-11-30')
        test_start_date: Date de début test (ex: '2023-12-01')
        timestamp_col: Nom de la colonne timestamp

    Returns:
        (train_df, val_df, test_df)

    Example:
        >>> train, val, test = split_train_val_test_with_gap(
        ...     df,
        ...     train_end_date='2023-10-31',
        ...     val_start_date='2023-11-07',  # GAP de 7 jours (01-06 Nov supprimés)
        ...     val_end_date='2023-11-30',
        ...     test_start_date='2023-12-01'
        ... )
    """
    logger.info("\n" + "="*60)
    logger.info("Split Temporal avec GAP PERIOD Anti-Contamination")
    logger.info("="*60)

    # S'assurer que timestamp est datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Convertir les dates
    train_end = pd.to_datetime(train_end_date)
    val_start = pd.to_datetime(val_start_date)
    val_end = pd.to_datetime(val_end_date)
    test_start = pd.to_datetime(test_start_date)

    # Calculer le gap
    gap_days = (val_start - train_end).days - 1
    logger.info(f"GAP PERIOD: {gap_days} jours ({train_end.date()} → {val_start.date()})")

    if gap_days < 1:
        logger.warning(f"⚠️  GAP insuffisant ({gap_days} jours)! Recommandé: ≥7 jours")

    # Split
    train_df = df[df[timestamp_col] <= train_end].copy()
    val_df = df[(df[timestamp_col] >= val_start) & (df[timestamp_col] <= val_end)].copy()
    test_df = df[df[timestamp_col] >= test_start].copy()

    # Calculer les lignes supprimées dans le gap
    gap_rows = len(df[(df[timestamp_col] > train_end) & (df[timestamp_col] < val_start)])

    logger.info(f"Train: {len(train_df)} lignes (jusqu'au {train_end.date()})")
    logger.info(f"GAP: {gap_rows} lignes SUPPRIMÉES")
    logger.info(f"Validation: {len(val_df)} lignes ({val_start.date()} → {val_end.date()})")
    logger.info(f"Test: {len(test_df)} lignes (à partir du {test_start.date()})")
    logger.info(f"Total utilisé: {len(train_df) + len(val_df) + len(test_df)} / {len(df)}")
    logger.info("="*60 + "\n")

    return train_df, val_df, test_df


def trim_filter_edges(df: pd.DataFrame,
                      n_trim: int = 30,
                      timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Enlève les bords du dataset après filtrage.

    ⚠️ RÈGLE CRITIQUE: Les filtres ont besoin de warm-up (début) et
    peuvent avoir des artifacts (fin). Il faut enlever ces zones
    AVANT de créer les splits train/val/test.

    Tests empiriques (dataset 200 points avec KAMA):
    - Erreur début (0-30):    569.44 ❌ ÉLEVÉE (warm-up)
    - Erreur milieu (30-170): 488.31 ✅ FAIBLE (zone propre)
    - Erreur fin (170-200):   349.42 ❌ ÉLEVÉE (artifacts)

    Args:
        df: DataFrame avec données filtrées
        n_trim: Nombre de valeurs à enlever au début ET à la fin (défaut: 30)
        timestamp_col: Nom de la colonne timestamp

    Returns:
        DataFrame sans les bords

    Raises:
        ValueError: Si le dataset est trop petit

    Example:
        >>> # Après application des filtres
        >>> df_filtered = add_adaptive_filter_features(df, ...)
        >>>
        >>> # AVANT de créer train/val/test
        >>> df_clean = trim_filter_edges(df_filtered, n_trim=30)
        >>>
        >>> # Maintenant créer les splits
        >>> train, val, test = split_train_val_test(df_clean, ...)

    Workflow complet:
        1. Charger données brutes
        2. Créer bougies fantômes
        3. Ajouter features avancées
        4. Ajouter filtres adaptatifs
        5. Ajouter indicateurs
        6. Ajouter labels
        7. ⚠️ TRIM (cette fonction) ← CRITIQUE!
        8. Split train/val/test

    Voir: REGLES_CRITIQUES_FILTRES.md pour documentation complète
    Voir: tests/test_visualization.py:test_filter_edge_effects()
    """
    if len(df) <= 2 * n_trim:
        raise ValueError(
            f"Dataset trop petit ({len(df)} lignes) pour enlever "
            f"{n_trim} valeurs de chaque côté. "
            f"Minimum requis: {2 * n_trim + 1} lignes"
        )

    # Enlever les n_trim premières et n_trim dernières lignes
    df_trimmed = df.iloc[n_trim:-n_trim].copy()

    # Réinitialiser l'index
    df_trimmed = df_trimmed.reset_index(drop=True)

    logger.info("="*60)
    logger.info("TRIM DES BORDS (Warm-up & Artifacts)")
    logger.info("="*60)
    logger.info(f"Dataset original: {len(df)} lignes")
    logger.info(f"Enlevé DÉBUT: {n_trim} lignes (warm-up)")
    logger.info(f"Enlevé FIN: {n_trim} lignes (artifacts)")
    logger.info(f"Dataset clean: {len(df_trimmed)} lignes")

    if timestamp_col in df.columns:
        logger.info(f"Période originale: {df[timestamp_col].iloc[0]} → {df[timestamp_col].iloc[-1]}")
        logger.info(f"Période clean: {df_trimmed[timestamp_col].iloc[0]} → {df_trimmed[timestamp_col].iloc[-1]}")

    logger.info("="*60)

    return df_trimmed


def save_dataset(df: pd.DataFrame,
                filepath: str,
                compress: bool = False) -> None:
    """
    Sauvegarde le dataset en CSV avec métadonnées.

    Args:
        df: DataFrame à sauvegarder
        filepath: Chemin de destination
        compress: Si True, compresse en .csv.gz
    """
    if compress and not filepath.endswith('.gz'):
        filepath += '.gz'

    compression = 'gzip' if compress else None

    df.to_csv(filepath, index=False, compression=compression)
    logger.info(f"Dataset sauvegardé: {filepath} ({len(df)} lignes, {len(df.columns)} colonnes)")

    # Afficher les premières lignes pour vérification
    logger.info(f"Aperçu:\n{df.head()}")
    logger.info(f"Colonnes: {list(df.columns)}")


def get_column_stats(df: pd.DataFrame, col: str) -> dict:
    """
    Statistiques descriptives d'une colonne.

    Args:
        df: DataFrame
        col: Nom de la colonne

    Returns:
        Dictionnaire de statistiques
    """
    if col not in df.columns:
        raise ValueError(f"Colonne {col} introuvable")

    series = df[col].dropna()

    return {
        'count': len(series),
        'mean': series.mean(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'median': series.median(),
        'q25': series.quantile(0.25),
        'q75': series.quantile(0.75),
        'null_count': df[col].isnull().sum()
    }


def log_dataset_metadata(metadata: dict, logger_instance=None) -> None:
    """
    Affiche les métadonnées d'un dataset de manière uniforme.

    Supporte les deux formats:
        - Ancien: timeframe (5min, all, etc.)
        - Nouveau: feature_timeframe + label_timeframe (pour labels 30min)

    Args:
        metadata: Dictionnaire de métadonnées du dataset
        logger_instance: Logger à utiliser (défaut: logger du module)
    """
    log = logger_instance or logger

    # Support ancien format (timeframe) et nouveau format (feature_timeframe/label_timeframe)
    if 'timeframe' in metadata:
        tf = metadata['timeframe']
        tf_str = f"{tf}m" if tf != 'all' else "all (1m+5m train, 5m val/test)"
        log.info(f"     Timeframe: {tf_str}")
    else:
        # Nouveau format avec labels 30min
        feat_tf = metadata.get('feature_timeframe', 'unknown')
        label_tf = metadata.get('label_timeframe', 'unknown')
        log.info(f"     Features: {feat_tf}")
        log.info(f"     Labels: {label_tf}")

    log.info(f"     Filtre: {metadata.get('filter_type', 'unknown')}")

    if 'created_at' in metadata:
        log.info(f"     Créé: {metadata['created_at']}")
