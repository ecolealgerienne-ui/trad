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
                         timeframe: str = '30T',
                         ohlcv_only: bool = True) -> pd.DataFrame:
    """
    Resampling des données vers un timeframe plus large.

    Args:
        df: DataFrame avec timestamp en index ou colonne
        timeframe: Timeframe cible (ex: '30T' pour 30 minutes)
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
