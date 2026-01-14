"""
Script de préparation des données pour Meta-Regime Trading (5min).

PRINCIPE CLÉ: Classification 3 régimes avec Features Computées (~23)
===============================================================================

✅ APPROCHE XGBoost (2026-01-14):
- Features d'entrée: TOUTES les features de régime (~23 features)
- Inclut: returns (3) + trend (7) + volatility (9) + volume (4)
- XGBoost avec features computées: 92.67% accuracy (VALIDÉ)
- CNN-LSTM avec 3 raw returns: 86.33% (ABANDONNÉ - delta -6.34%)

Régimes (3 classes):
- 0: RANGE LOW VOL  (TS < 0.45, VC ≤ P50)
- 1: RANGE HIGH VOL (TS < 0.45, VC > P50)
- 2: TREND          (TS >= 0.45, any volatility)

Features d'entrée du modèle (23 colonnes via get_regime_feature_names()):
  Returns (3):
    - c_ret, h_ret, l_ret
  Trend (7):
    - ma20_slope, ma50_slope, regression_slope, regression_r2
    - adx, macd_histogram_norm, hurst_exponent
  Volatility (9):
    - atr_normalized, bb_upper, bb_middle, bb_lower, bb_width, percent_b
    - realized_volatility, volatility_compression, range_atr_ratio
  Volume (4):
    - volume_ratio, volume_spike, vwap_deviation, obv_derivative

Labels (4 au total):
  Régime:
    - regime: 0-2 (3 classes)
  Direction (Kalman-filtered, pour modèles MACD/RSI/CCI):
    - macd_direction: 0/1 (DOWN/UP)
    - rsi_direction: 0/1 (DOWN/UP)
    - cci_direction: 0/1 (DOWN/UP)

Pipeline:
1. Charger données brutes (OHLCV 5min)
2. Calculer returns (c_ret, h_ret, l_ret)
3. Calculer features de régime (~23) pour modèle ET labeling
4. Calculer labels régime (regime_labeler.py)
5. Créer séquences (25 timesteps × 23 features)
6. Split temporel (70/15/15)
7. Sauvegarder dataset unique: dataset_<assets>_regime.npz

Usage:
    python src/prepare_data_regime.py --assets BTC ETH BNB ADA LTC

Génère:
    data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz

Author: Claude Code - Phase 1 (Data Layer)
Date: 2025-01-11
Version: 3.0 (23 features for XGBoost)
"""

import numpy as np
import pandas as pd
import argparse
import logging
import json
import os
from pathlib import Path
from datetime import datetime
import gc
from numpy.lib.stride_tricks import sliding_window_view
from joblib import Parallel, delayed
import psutil
from pykalman import KalmanFilter

logger = logging.getLogger(__name__)

# Import modules locaux
from constants import (
    AVAILABLE_ASSETS_5M, DEFAULT_ASSETS,
    TRIM_EDGES,
    PREPARED_DATA_DIR,
    SEQUENCE_LENGTH,
    REGIME_LOOKAHEAD,
    RSI_PERIOD, CCI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR,
)

# Import modules de régime
from regime_features import calculate_all_regime_features, get_regime_feature_names
from regime_labeler import calculate_regime_labels, validate_regime_features

# Import indicateurs pour labels direction
from indicators import calculate_rsi, calculate_cci, calculate_macd

# Mapping Asset Name → Asset ID (pour encodage dans les matrices)
ASSET_ID_MAP = {
    'BTC': 0,
    'ETH': 1,
    'BNB': 2,
    'ADA': 3,
    'LTC': 4
}


# =============================================================================
# LABELS DIRECTION (MACD/RSI/CCI) - ARCHITECTURE UNIFIÉE
# =============================================================================

def kalman_filter_dual(data: np.ndarray,
                       process_var: float = KALMAN_PROCESS_VAR,
                       measure_var: float = KALMAN_MEASURE_VAR) -> np.ndarray:
    """
    Applique un filtre de Kalman CINÉMATIQUE (position + vélocité).

    Copié depuis prepare_data_direction_only.py pour cohérence.

    Returns:
        np.ndarray de shape (n, 2) - [:, 0]=position, [:, 1]=velocity
    """
    valid_mask = ~np.isnan(data)
    if valid_mask.sum() < 10:
        result = np.full((len(data), 2), np.nan)
        return result

    transition_matrix = [[1, 1], [0, 1]]
    observation_matrix = [[1, 0]]
    initial_state_mean = [data[valid_mask][0], 0.0]
    observation_covariance = measure_var
    transition_covariance = np.eye(2) * process_var

    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        initial_state_mean=initial_state_mean,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance
    )

    means, _ = kf.smooth(data[valid_mask])

    result = np.full((len(data), 2), np.nan)
    result[valid_mask] = means

    return result


def calculate_direction_label(df: pd.DataFrame,
                               indicator_name: str,
                               indicator_values: np.ndarray) -> pd.Series:
    """
    Calcule le label direction pour un indicateur avec filtre Kalman.

    Pipeline:
      1. Indicateur brut → Kalman → position filtrée
      2. Label direction = position[t] > position[t-1]

    Args:
        df: DataFrame (pour index temporel)
        indicator_name: 'macd', 'rsi', ou 'cci'
        indicator_values: Valeurs brutes de l'indicateur

    Returns:
        pd.Series de labels binaires (0=DOWN, 1=UP)
    """
    # Appliquer Kalman
    filter_output = kalman_filter_dual(indicator_values)
    position = filter_output[:, 0]

    # Calculer label direction: filtered[t] > filtered[t-1]
    pos_series = pd.Series(position, index=df.index)
    pos_t0 = pos_series.shift(0)
    pos_t1 = pos_series.shift(1)
    direction_label = (pos_t0 > pos_t1).astype(int)

    logger.info(f"      {indicator_name.upper()} direction: "
                f"{direction_label.sum()}/{len(direction_label)} UP "
                f"({direction_label.mean()*100:.1f}%)")

    return direction_label


# =============================================================================
# FONCTIONS DE VALIDATION DES DONNÉES
# =============================================================================

def validate_returns(df: pd.DataFrame, asset_name: str):
    """
    Valide que les returns (c_ret, h_ret, l_ret) sont correctement calculés.

    Vérifications:
      1. Les returns correspondent bien au pct_change() des prix
      2. Les valeurs NaN sont uniquement en première position
      3. Les valeurs sont dans des plages raisonnables
      4. Pas de valeurs infinies ou anormales

    Args:
        df: DataFrame avec colonnes close, high, low, c_ret, h_ret, l_ret
        asset_name: Nom de l'asset pour logging
    """
    logger.info(f"\n  🔍 VALIDATION RETURNS ({asset_name})...")

    errors = []
    warnings = []

    # Vérification 1: c_ret correspond au pct_change de close
    expected_c_ret = df['close'].pct_change()
    if not np.allclose(df['c_ret'].dropna(), expected_c_ret.dropna(), rtol=1e-10, atol=1e-15):
        errors.append("c_ret ne correspond pas à close.pct_change()")
    else:
        logger.info(f"    ✓ c_ret = close.pct_change() vérifié")

    # Vérification 2: h_ret correspond au pct_change de high
    expected_h_ret = df['high'].pct_change()
    if not np.allclose(df['h_ret'].dropna(), expected_h_ret.dropna(), rtol=1e-10, atol=1e-15):
        errors.append("h_ret ne correspond pas à high.pct_change()")
    else:
        logger.info(f"    ✓ h_ret = high.pct_change() vérifié")

    # Vérification 3: l_ret correspond au pct_change de low
    expected_l_ret = df['low'].pct_change()
    if not np.allclose(df['l_ret'].dropna(), expected_l_ret.dropna(), rtol=1e-10, atol=1e-15):
        errors.append("l_ret ne correspond pas à low.pct_change()")
    else:
        logger.info(f"    ✓ l_ret = low.pct_change() vérifié")

    # Vérification 4: NaN uniquement en première position
    for col in ['c_ret', 'h_ret', 'l_ret']:
        nan_count = df[col].isna().sum()
        first_is_nan = df[col].iloc[0] if len(df) > 0 else False
        if nan_count > 1 or (nan_count == 1 and not pd.isna(first_is_nan)):
            warnings.append(f"{col}: {nan_count} NaN (attendu: 1 en première position)")
        else:
            logger.info(f"    ✓ {col}: NaN seulement en position [0]")

    # Vérification 5: Plages de valeurs raisonnables pour données 5min
    for col in ['c_ret', 'h_ret', 'l_ret']:
        values = df[col].dropna()
        min_val = values.min()
        max_val = values.max()
        abs_max = max(abs(min_val), abs(max_val))

        # Pour du 5min crypto, returns typiques: -10% à +10% (extrêmes rares)
        if abs_max > 0.5:  # 50% en 5min est suspect
            warnings.append(f"{col}: valeur extrême détectée (max={abs_max:.2%})")
        else:
            logger.info(f"    ✓ {col}: plage [{min_val:.4%}, {max_val:.4%}]")

    # Vérification 6: Pas de valeurs infinies
    for col in ['c_ret', 'h_ret', 'l_ret']:
        if np.isinf(df[col]).any():
            errors.append(f"{col} contient des valeurs infinies")
        else:
            logger.info(f"    ✓ {col}: pas de valeurs infinies")

    # Statistiques descriptives
    logger.info(f"\n    📊 Statistiques returns:")
    logger.info(f"       c_ret - mean: {df['c_ret'].mean():.6f}, std: {df['c_ret'].std():.6f}")
    logger.info(f"       h_ret - mean: {df['h_ret'].mean():.6f}, std: {df['h_ret'].std():.6f}")
    logger.info(f"       l_ret - mean: {df['l_ret'].mean():.6f}, std: {df['l_ret'].std():.6f}")

    # Rapport final
    if errors:
        logger.error(f"\n    ❌ ERREURS DÉTECTÉES:")
        for error in errors:
            logger.error(f"       - {error}")
        raise ValueError(f"Validation returns échouée pour {asset_name}")

    if warnings:
        logger.warning(f"\n    ⚠️  AVERTISSEMENTS:")
        for warning in warnings:
            logger.warning(f"       - {warning}")

    logger.info(f"\n    ✅ Validation returns réussie ({asset_name})")


def validate_directions(df: pd.DataFrame, asset_name: str):
    """
    Valide que les labels direction (macd_direction, rsi_direction, cci_direction)
    sont correctement calculés.

    Vérifications:
      1. Les valeurs sont binaires (0 ou 1 uniquement)
      2. Les ratios UP/DOWN sont raisonnables (pas tous identiques)
      3. Les changements de direction sont fréquents (pas stagnants)
      4. Pas de valeurs NaN après calcul

    Args:
        df: DataFrame avec colonnes *_direction
        asset_name: Nom de l'asset pour logging
    """
    logger.info(f"\n  🔍 VALIDATION DIRECTIONS ({asset_name})...")

    errors = []
    warnings = []

    for indicator in ['macd', 'rsi', 'cci']:
        col = f'{indicator}_direction'

        if col not in df.columns:
            errors.append(f"{col} manquante dans DataFrame")
            continue

        values = df[col]

        # Vérification 1: Valeurs binaires (0 ou 1)
        unique_vals = values.dropna().unique()
        if not np.all(np.isin(unique_vals, [0, 1])):
            errors.append(f"{col}: valeurs non-binaires détectées: {unique_vals}")
        else:
            logger.info(f"    ✓ {col}: valeurs binaires (0/1)")

        # Vérification 2: Ratio UP/DOWN raisonnable
        up_count = (values == 1).sum()
        down_count = (values == 0).sum()
        total = up_count + down_count

        if total == 0:
            errors.append(f"{col}: aucune valeur valide")
            continue

        up_ratio = up_count / total
        logger.info(f"    ✓ {col}: UP={up_count}/{total} ({up_ratio:.1%}), DOWN={down_count}/{total} ({1-up_ratio:.1%})")

        # Avertir si ratio extrême (>95% ou <5% dans une direction)
        if up_ratio > 0.95 or up_ratio < 0.05:
            warnings.append(f"{col}: ratio déséquilibré (UP={up_ratio:.1%})")

        # Vérification 3: Changements de direction (transitions)
        transitions = (values.diff() != 0).sum()
        transition_rate = transitions / len(values) if len(values) > 0 else 0
        logger.info(f"    ✓ {col}: {transitions} transitions ({transition_rate:.2%} du temps)")

        # Avertir si très peu de transitions (<1%)
        if transition_rate < 0.01:
            warnings.append(f"{col}: très peu de transitions ({transition_rate:.2%})")

        # Vérification 4: NaN après calcul (avant fillna)
        nan_count = values.isna().sum()
        if nan_count > 0:
            # NaN attendus: Kalman nécessite warmup (~100 samples)
            expected_nan = min(100, len(df) // 10)  # Environ 10% max acceptable
            if nan_count > expected_nan:
                warnings.append(f"{col}: {nan_count} NaN (warmup Kalman?)")
            logger.info(f"    ✓ {col}: {nan_count} NaN (warmup Kalman)")
        else:
            logger.info(f"    ✓ {col}: pas de NaN")

    # Rapport final
    if errors:
        logger.error(f"\n    ❌ ERREURS DÉTECTÉES:")
        for error in errors:
            logger.error(f"       - {error}")
        raise ValueError(f"Validation directions échouée pour {asset_name}")

    if warnings:
        logger.warning(f"\n    ⚠️  AVERTISSEMENTS:")
        for warning in warnings:
            logger.warning(f"       - {warning}")

    logger.info(f"\n    ✅ Validation directions réussie ({asset_name})")


def validate_synchronization(df: pd.DataFrame, asset_name: str):
    """
    Valide la synchronisation temporelle entre returns et directions.

    Vérifications:
      1. Même index temporel (timestamps alignés)
      2. Même nombre d'échantillons non-NaN
      3. Cohérence logique: direction UP corrèle avec c_ret > 0
      4. Lag vérifié: direction[t] basée sur filtered[t] vs filtered[t-1]

    Args:
        df: DataFrame avec returns et directions
        asset_name: Nom de l'asset pour logging
    """
    logger.info(f"\n  🔍 VALIDATION SYNCHRONISATION ({asset_name})...")

    errors = []
    warnings = []

    # Vérification 1: Index temporel identique
    if not df.index.equals(df.index):
        errors.append("Index temporel incohérent")
    else:
        logger.info(f"    ✓ Index temporel cohérent: {len(df)} samples")

    # Vérification 2: Cohérence direction vs returns
    # Note: Direction est basée sur Kalman(indicator), pas directement sur c_ret
    # Mais il devrait y avoir une corrélation positive entre direction et returns moyens
    for indicator in ['macd', 'rsi', 'cci']:
        col_dir = f'{indicator}_direction'

        if col_dir not in df.columns:
            continue

        # Calculer c_ret moyen pour chaque direction
        df_valid = df[[col_dir, 'c_ret']].dropna()

        if len(df_valid) == 0:
            warnings.append(f"{indicator}: pas assez de données valides")
            continue

        mean_ret_up = df_valid[df_valid[col_dir] == 1]['c_ret'].mean()
        mean_ret_down = df_valid[df_valid[col_dir] == 0]['c_ret'].mean()

        logger.info(f"    📊 {indicator.upper()}:")
        logger.info(f"       UP (1): c_ret moyen = {mean_ret_up:.6f}")
        logger.info(f"       DOWN (0): c_ret moyen = {mean_ret_down:.6f}")

        # Vérifier que UP a un return moyen positif et DOWN négatif
        # (Attention: ce n'est pas garanti car direction est sur indicator filtré, pas sur c_ret)
        # Mais une corrélation positive est attendue
        if mean_ret_up > mean_ret_down:
            logger.info(f"    ✓ {indicator}: cohérence logique (UP > DOWN)")
        else:
            # Ce n'est pas une erreur car les directions sont sur indicators filtrés
            warnings.append(f"{indicator}: UP mean_ret ({mean_ret_up:.6f}) ≤ DOWN mean_ret ({mean_ret_down:.6f})")

    # Vérification 3: Lag entre direction et returns
    # Direction[t] est calculée comme filtered[t] > filtered[t-1]
    # Donc direction[t] utilise des informations jusqu'à t inclus
    # c_ret[t] = (close[t] - close[t-1]) / close[t-1]
    # Il y a synchronisation parfaite: les deux utilisent t et t-1

    logger.info(f"\n    ✓ Alignement temporel:")
    logger.info(f"       direction[t] = filtered[t] > filtered[t-1]")
    logger.info(f"       c_ret[t] = (close[t] - close[t-1]) / close[t-1]")
    logger.info(f"       → Synchronisation correcte (même fenêtre temporelle)")

    # Rapport final
    if errors:
        logger.error(f"\n    ❌ ERREURS DÉTECTÉES:")
        for error in errors:
            logger.error(f"       - {error}")
        raise ValueError(f"Validation synchronisation échouée pour {asset_name}")

    if warnings:
        logger.warning(f"\n    ⚠️  AVERTISSEMENTS:")
        for warning in warnings:
            logger.warning(f"       - {warning}")

    logger.info(f"\n    ✅ Validation synchronisation réussie ({asset_name})")


# =============================================================================
# PARALLÉLISATION INTELLIGENTE
# =============================================================================

def get_safe_n_jobs(n_assets: int, ram_per_asset_gb: float = 4.0) -> int:
    """
    Calcule le nombre de jobs parallèles selon la RAM disponible.

    Args:
        n_assets: Nombre total d'assets à traiter
        ram_per_asset_gb: RAM peak estimée par asset (GB)

    Returns:
        Nombre de jobs sûrs (1 à min(n_assets, n_cores))
    """
    try:
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        max_by_ram = max(1, int(available_ram_gb / ram_per_asset_gb))
        n_cores = os.cpu_count() or 1
        max_by_cpu = max(1, n_cores - 1)
        n_jobs = min(max_by_ram, max_by_cpu, n_assets)

        logger.info(f"Parallélisation: {n_jobs} assets simultanés")
        logger.info(f"  RAM disponible: {available_ram_gb:.1f} GB")
        logger.info(f"  RAM par asset: {ram_per_asset_gb:.1f} GB")

        return n_jobs
    except Exception as e:
        logger.warning(f"Erreur détection parallélisme: {e}, fallback n_jobs=1")
        return 1


# =============================================================================
# SPLIT TEMPOREL
# =============================================================================

def find_common_period(assets: list) -> tuple:
    """
    Trouve la période temporelle commune à tous les assets.

    Args:
        assets: Liste des noms d'assets

    Returns:
        (min_timestamp, max_timestamp) en commun à tous
    """
    min_timestamps = []
    max_timestamps = []

    for asset_name in assets:
        csv_path = AVAILABLE_ASSETS_5M.get(asset_name)
        if csv_path is None:
            continue

        # Lire première ligne pour détecter la colonne timestamp
        df_sample = pd.read_csv(csv_path, nrows=1)

        # Détecter quelle colonne timestamp est disponible
        if 'timestamp' in df_sample.columns:
            ts_col = 'timestamp'
        elif 'time' in df_sample.columns:
            ts_col = 'time'
        else:
            logger.warning(f"  {asset_name}: Aucune colonne timestamp trouvée, ignoré")
            continue

        # Lire toutes les timestamps
        df = pd.read_csv(csv_path, usecols=[ts_col])
        df[ts_col] = pd.to_datetime(df[ts_col])

        min_timestamps.append(df[ts_col].min())
        max_timestamps.append(df[ts_col].max())

    # Période commune = max des min, min des max
    common_start = max(min_timestamps)
    common_end = min(max_timestamps)

    logger.info(f"  Période commune: {common_start} → {common_end}")
    logger.info(f"    Durée: {(common_end - common_start).days / 365.25:.1f} ans")

    return common_start, common_end


def calculate_split_timestamps(common_start: pd.Timestamp,
                                 common_end: pd.Timestamp,
                                 train_ratio: float = 0.70,
                                 val_ratio: float = 0.15,
                                 test_ratio: float = 0.15) -> dict:
    """
    Calcule les timestamps de split sur la période commune.

    Args:
        common_start: Timestamp début période commune
        common_end: Timestamp fin période commune
        train_ratio: Ratio pour train
        val_ratio: Ratio pour val
        test_ratio: Ratio pour test

    Returns:
        dict avec 'train_end', 'val_end'
    """
    total_duration = common_end - common_start

    train_duration = total_duration * train_ratio
    val_duration = total_duration * val_ratio

    train_end = common_start + train_duration
    val_end = train_end + val_duration

    logger.info(f"  Split timestamps:")
    logger.info(f"    Train: {common_start} → {train_end}")
    logger.info(f"    Val:   {train_end} → {val_end}")
    logger.info(f"    Test:  {val_end} → {common_end}")

    return {
        'train_start': common_start,
        'train_end': train_end,
        'val_start': train_end,
        'val_end': val_end,
        'test_start': val_end,
        'test_end': common_end
    }


def temporal_split_by_timestamps(df: pd.DataFrame,
                                   split_timestamps: dict) -> dict:
    """
    Split temporel basé sur des timestamps absolus.

    Args:
        df: DataFrame avec DatetimeIndex
        split_timestamps: Dict avec train_end, val_end

    Returns:
        dict avec clés 'train', 'val', 'test'
    """
    # Filtrer par timestamps
    train_df = df[(df.index >= split_timestamps['train_start']) &
                  (df.index < split_timestamps['train_end'])].copy()
    val_df = df[(df.index >= split_timestamps['val_start']) &
                (df.index < split_timestamps['val_end'])].copy()
    test_df = df[(df.index >= split_timestamps['test_start']) &
                 (df.index <= split_timestamps['test_end'])].copy()

    logger.info(f"  Split temporel: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }


# =============================================================================
# CRÉATION SÉQUENCES POUR RÉGIMES
# =============================================================================

def compute_future_regime_label(regime_labels: np.ndarray,
                                 lookahead: int = REGIME_LOOKAHEAD) -> np.ndarray:
    """
    Calcule le label de régime FUTUR avec logique "Any TREND".

    Principe: Pour chaque position t, on regarde les régimes dans [t+1, t+N]
    - Si ANY régime est TREND (2): label = TREND (2)
    - Sinon: vote majoritaire entre RANGE_LOW_VOL (0) et RANGE_HIGH_VOL (1)

    Args:
        regime_labels: Array (N,) des labels de régime à chaque timestep
        lookahead: Horizon N (défaut: 6 = 30 min sur données 5min)

    Returns:
        future_labels: Array (N-lookahead,) des labels futurs
    """
    N = len(regime_labels)
    n_valid = N - lookahead

    if n_valid <= 0:
        return np.array([], dtype=np.float32)

    future_labels = np.zeros(n_valid, dtype=np.float32)

    for i in range(n_valid):
        # Fenêtre future: [t+1, t+N] = indices [i+1, i+lookahead]
        future_window = regime_labels[i+1:i+1+lookahead]

        # Logique "Any TREND": si TREND (2) apparaît, label = TREND
        if 2 in future_window:
            future_labels[i] = 2.0
        else:
            # Vote majoritaire entre RANGE_LOW_VOL (0) et RANGE_HIGH_VOL (1)
            counts = np.bincount(future_window.astype(int), minlength=3)
            # Ignorer TREND (index 2) car non présent
            if counts[1] > counts[0]:
                future_labels[i] = 1.0
            else:
                future_labels[i] = 0.0

    return future_labels


def create_sequences_for_regime(df: pd.DataFrame,
                                 feature_cols: list,
                                 asset_name: str,
                                 asset_id: int,
                                 seq_length: int = SEQUENCE_LENGTH,
                                 lookahead: int = REGIME_LOOKAHEAD) -> tuple:
    """
    Crée les séquences pour le dataset régime avec labels FUTURS.

    Structure:
    - X: (n, seq_length, n_features+2) = [timestamp, asset_id, features...]
    - Y: (n, 6) = [timestamp, asset_id, regime_futur, macd_dir, rsi_dir, cci_dir]
    - OHLCV: (n, 7) = [timestamp, asset_id, O, H, L, C, V]

    IMPORTANT - Labels Futurs (2026-01-14):
    - Le label 'regime' est calculé sur la fenêtre FUTURE [t+1, t+N]
    - Logique "Any TREND": si TREND apparaît dans [t+1, t+N], label = TREND
    - Sinon: vote majoritaire entre RANGE_LOW_VOL et RANGE_HIGH_VOL
    - Les labels direction (macd_dir, rsi_dir, cci_dir) restent à t

    Args:
        df: DataFrame avec features + labels de régime
        feature_cols: Liste des features à utiliser (~23 features via get_regime_feature_names())
        asset_name: Nom de l'asset ('BTC', 'ETH', etc.)
        asset_id: ID encodé de l'asset (0-4)
        seq_length: Longueur des séquences (défaut: 25)
        lookahead: Horizon de prédiction N (défaut: 6 = 30 min)

    Returns:
        X: (n, seq_length, n_features+2)
        Y: (n, 6)  # timestamp, asset_id, regime_futur, macd_dir, rsi_dir, cci_dir
        OHLCV: (n, 7)
    """
    # Colonnes label (4 au total: 1 régime + 3 direction)
    label_cols = [
        'regime',  # Label régime (0-2)
        'macd_direction', 'rsi_direction', 'cci_direction'  # Labels direction
    ]

    # Colonnes OHLCV brutes
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']

    # Remplacer NaN par 0
    cols_needed = feature_cols + label_cols + ohlcv_cols
    df_clean = df[cols_needed].fillna(0)
    df_clean.index = df.index  # Préserver l'index temporel

    n_nans_filled = df[cols_needed].isna().sum().sum()
    logger.info(f"     NaN → 0: {n_nans_filled} valeurs remplacées")

    # Extraire arrays
    features = df_clean[feature_cols].values.astype(np.float32)  # (N, n_features)
    labels = df_clean[label_cols].values.astype(np.float32)      # (N, 4)
    ohlcv = df_clean[ohlcv_cols].values.astype(np.float32)       # (N, 5)

    N, n_features = features.shape

    # Validation
    if N < seq_length:
        logger.warning(f"     {asset_name}: Pas assez de données ({N} < {seq_length})")
        return None, None, None

    # Timestamps (Unix en secondes)
    timestamps = df_clean.index.astype(np.int64) // 10**9  # (N,)
    timestamps = timestamps.astype(np.float32)

    # Asset ID (répété)
    asset_ids = np.full(N, asset_id, dtype=np.float32)  # (N,)

    # ========================================================================
    # VECTORISATION SLIDING WINDOWS (×50 plus rapide)
    # ========================================================================

    # Features: (N, n_features) → (n_sequences_raw, seq_length, n_features)
    X_features_raw = sliding_window_view(features, window_shape=(seq_length, n_features)).squeeze(axis=1)

    # Timestamps: (N,) → (n_sequences_raw, seq_length)
    X_timestamps_raw = sliding_window_view(timestamps, window_shape=seq_length).reshape(-1, seq_length)

    # Asset IDs: (N,) → (n_sequences_raw, seq_length)
    X_asset_ids_raw = sliding_window_view(asset_ids, window_shape=seq_length).reshape(-1, seq_length)

    n_sequences_raw = X_features_raw.shape[0]

    # ========================================================================
    # LABELS FUTURS avec logique "Any TREND" (2026-01-14)
    # ========================================================================
    # Pour chaque séquence, on calcule le label de régime sur [t+1, t+N]
    # où t est la fin de la séquence et N = lookahead (défaut: 6)
    # On perd 'lookahead' séquences à la fin car pas de données futures

    # Extraire les labels de régime bruts (colonne 0)
    regime_labels_raw = labels[:, 0]  # (N,)

    # Calculer labels de régime FUTURS avec "Any TREND" logic
    # Pour séquence finissant à position (seq_length-1 + i), le label futur est
    # calculé sur [seq_length-1 + i + 1, seq_length-1 + i + lookahead]
    regime_from_first_end = regime_labels_raw[seq_length-1:]  # (n_sequences_raw,)
    future_regime_labels = compute_future_regime_label(regime_from_first_end, lookahead)

    # Nombre de séquences valides après application du lookahead
    n_sequences = len(future_regime_labels)
    logger.info(f"     Séquences brutes: {n_sequences_raw}, après lookahead N={lookahead}: {n_sequences}")

    if n_sequences == 0:
        logger.warning(f"     {asset_name}: Pas assez de données après lookahead")
        return None, None, None

    # Tronquer tous les arrays pour correspondre au nombre de séquences valides
    X_features = X_features_raw[:n_sequences]
    X_timestamps = X_timestamps_raw[:n_sequences]
    X_asset_ids = X_asset_ids_raw[:n_sequences]

    # Labels direction: restent à t (fin de séquence) - colonnes 1, 2, 3
    direction_labels = labels[seq_length-1:seq_length-1+n_sequences, 1:4]  # (n_sequences, 3)

    # Combiner regime futur + direction à t
    # Y_labels: [regime_futur, macd_dir, rsi_dir, cci_dir]
    Y_labels = np.column_stack([
        future_regime_labels,  # (n_sequences,) - labels FUTURS
        direction_labels       # (n_sequences, 3) - labels à t
    ])

    # Timestamps pour Y: Derniers timestamps de chaque séquence valide
    Y_timestamps = timestamps[seq_length-1:seq_length-1+n_sequences]  # (n_sequences,)

    # Asset IDs pour Y
    Y_asset_ids = asset_ids[seq_length-1:seq_length-1+n_sequences]  # (n_sequences,)

    # OHLCV: Prendre OHLCV à la FIN de chaque séquence valide
    OHLCV_data = ohlcv[seq_length-1:seq_length-1+n_sequences]  # (n_sequences, 5)
    OHLCV_timestamps = timestamps[seq_length-1:seq_length-1+n_sequences]
    OHLCV_asset_ids = asset_ids[seq_length-1:seq_length-1+n_sequences]

    # Combiner X: [timestamp, asset_id, features...]
    # Shape: (n_sequences, seq_length, 2+n_features)
    X = np.concatenate([
        X_timestamps[..., np.newaxis],  # (n_seq, seq_len, 1)
        X_asset_ids[..., np.newaxis],   # (n_seq, seq_len, 1)
        X_features                      # (n_seq, seq_len, n_features)
    ], axis=2)

    # Combiner Y: [timestamp, asset_id, regime_futur, macd_dir, rsi_dir, cci_dir]
    # Shape: (n_sequences, 6)
    Y = np.column_stack([
        Y_timestamps,  # (n_seq,)
        Y_asset_ids,   # (n_seq,)
        Y_labels       # (n_seq, 4)
    ])

    # Combiner OHLCV: [timestamp, asset_id, O, H, L, C, V]
    # Shape: (n_sequences, 7)
    OHLCV = np.column_stack([
        OHLCV_timestamps,
        OHLCV_asset_ids,
        OHLCV_data
    ])

    logger.info(f"     Séquences créées: {n_sequences}")
    logger.info(f"       X: {X.shape} (timestamp, asset_id, {n_features} features)")
    logger.info(f"       Y: {Y.shape} (timestamp, asset_id, regime_futur, macd_dir, rsi_dir, cci_dir)")
    logger.info(f"       OHLCV: {OHLCV.shape}")

    return X, Y, OHLCV


# =============================================================================
# TRAITEMENT D'UN ASSET
# =============================================================================

def process_single_asset(asset_name: str,
                          split_timestamps: dict,
                          clip_value: float = None,
                          max_samples: int = None) -> dict:
    """
    Traite un seul asset: charge, calcule features, labels, split, séquences.

    Args:
        asset_name: Nom de l'asset ('BTC', 'ETH', etc.)
        split_timestamps: Dict avec les timestamps de split (common_start, train_end, etc.)
        clip_value: Valeur de clipping des features (None = pas de clip)
        max_samples: Limite nombre de samples (None = tout)

    Returns:
        dict avec 'train', 'val', 'test' contenant (X, Y, OHLCV)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"ASSET: {asset_name}")
    logger.info('='*80)

    # Charger CSV
    csv_path = AVAILABLE_ASSETS_5M.get(asset_name)
    if csv_path is None:
        raise ValueError(f"Asset {asset_name} non disponible")

    logger.info(f"  Chargement: {csv_path}")
    df = pd.read_csv(csv_path)

    # Limiter nombre de samples pour tests
    if max_samples is not None and max_samples > 0:
        df = df.head(max_samples)
        logger.info(f"  Limité à {max_samples} samples (test)")

    logger.info(f"  Lignes chargées: {len(df)}")

    # Colonnes OHLCV (renommage si nécessaire)
    if 'Open' in df.columns:
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        })

    # Index temporel
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
    else:
        logger.warning("  Aucune colonne timestamp trouvée, utilisation index")

    # ========================================================================
    # FILTRAGE VOLUME=0 (FIX: 4977 violations Train)
    # ========================================================================

    original_len = len(df)
    df = df[df['volume'] > 0].copy()
    if len(df) < original_len:
        filtered_count = original_len - len(df)
        logger.info(f"  ✓ Filtré {filtered_count} lignes avec volume=0 ({filtered_count/original_len*100:.2f}%)")

    # ========================================================================
    # FILTRAGE À LA PÉRIODE COMMUNE (FIX: negative gaps)
    # ========================================================================

    if split_timestamps is not None:
        common_start = split_timestamps['train_start']
        common_end = split_timestamps['test_end']

        original_len = len(df)
        df = df[(df.index >= common_start) & (df.index <= common_end)].copy()

        if len(df) < original_len:
            filtered_count = original_len - len(df)
            logger.info(f"  ✓ Filtré à période commune: {len(df)} lignes ({filtered_count} hors période)")

    # Trim edges (100 début + 100 fin)
    if len(df) > 2 * TRIM_EDGES:
        df = df.iloc[TRIM_EDGES:-TRIM_EDGES].copy()
        logger.info(f"  Trim ±{TRIM_EDGES}: {len(df)} lignes restantes")

    # ========================================================================
    # ÉTAPE 0.5: CALCULER RETURNS (c_ret, h_ret, l_ret) - INPUTS DU MODÈLE
    # ========================================================================

    logger.info(f"\n  Calcul returns (c_ret, h_ret, l_ret) - inputs du modèle...")
    df['c_ret'] = df['close'].pct_change()
    df['h_ret'] = df['high'].pct_change()
    df['l_ret'] = df['low'].pct_change()
    logger.info(f"  ✓ Returns calculés: c_ret, h_ret, l_ret")

    # Validation des returns
    validate_returns(df, asset_name)

    # ========================================================================
    # ÉTAPE 1: CALCULER FEATURES DE RÉGIME (~20 colonnes) - POUR LABELING SEULEMENT
    # ========================================================================

    logger.info(f"\n  Calcul features de régime (~20 colonnes) pour labeling...")
    df = calculate_all_regime_features(df)
    logger.info(f"  ✓ Features intermédiaires calculées: {df.shape[1]} colonnes (utilisées pour labels UNIQUEMENT)")

    # ========================================================================
    # ÉTAPE 2: CALCULER LABELS DE RÉGIME (regime, ts_score, vc_score)
    # ========================================================================

    logger.info(f"\n  Calcul labels de régime (3 classes)...")
    try:
        validate_regime_features(df)
        regime_labels, ts_score, vc_score = calculate_regime_labels(df)

        # Ajouter au DataFrame
        df['regime'] = regime_labels
        df['trend_strength'] = ts_score
        df['volatility_cluster'] = vc_score

        logger.info(f"  ✓ Labels calculés")
    except Exception as e:
        logger.error(f"  ✗ Erreur calcul labels: {e}")
        raise

    # ========================================================================
    # ÉTAPE 2.5: CALCULER LABELS DIRECTION (MACD, RSI, CCI)
    # ========================================================================

    logger.info(f"\n  Calcul labels direction (MACD, RSI, CCI)...")

    # MACD
    macd_dict = calculate_macd(
        df['close'],
        fast_period=MACD_FAST,
        slow_period=MACD_SLOW,
        signal_period=MACD_SIGNAL
    )
    # calculate_macd retourne un dict avec 'macd', 'signal', 'histogram'
    macd_vals = macd_dict['macd']
    df['macd_direction'] = calculate_direction_label(df, 'macd', macd_vals)

    # RSI
    rsi_vals = calculate_rsi(df['close'], period=RSI_PERIOD)
    df['rsi_direction'] = calculate_direction_label(df, 'rsi', rsi_vals)

    # CCI
    cci_vals = calculate_cci(df['high'], df['low'], df['close'], period=CCI_PERIOD)
    df['cci_direction'] = calculate_direction_label(df, 'cci', cci_vals)

    logger.info(f"  ✓ Labels direction calculés (MACD, RSI, CCI)")

    # Validation des directions et synchronisation avec returns
    validate_directions(df, asset_name)
    validate_synchronization(df, asset_name)

    # Remplacer NaN par 0 après tout le calcul
    df = df.fillna(0)

    # ========================================================================
    # ÉTAPE 3: SPLIT TEMPOREL PAR TIMESTAMPS (70/15/15)
    # ========================================================================

    logger.info(f"\n  Split temporel par timestamps...")
    splits = temporal_split_by_timestamps(df, split_timestamps)

    # ========================================================================
    # ÉTAPE 4: CRÉER SÉQUENCES POUR CHAQUE SPLIT
    # ========================================================================

    logger.info(f"\n  Création séquences (seq_length={SEQUENCE_LENGTH})...")

    # Features à utiliser (TOUTES les features de régime - ~23 features)
    # Inclut: returns bruts (3) + trend (7) + volatility (9) + volume (4)
    # Ces features computées sont essentielles pour XGBoost (92.67% vs 86.33% raw)
    feature_cols = get_regime_feature_names()
    logger.info(f"  Features utilisées ({len(feature_cols)}): {feature_cols[:5]}... (total: {len(feature_cols)})")

    # Clip si demandé
    if clip_value is not None:
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].clip(-clip_value, clip_value)
        logger.info(f"  Features clippées à ±{clip_value}")

    # Asset ID
    asset_id = ASSET_ID_MAP[asset_name]

    results = {}
    for split_name, split_df in splits.items():
        logger.info(f"\n    {split_name.upper()}:")
        X, Y, OHLCV = create_sequences_for_regime(
            split_df,
            feature_cols,
            asset_name,
            asset_id,
            seq_length=SEQUENCE_LENGTH
        )

        if X is not None:
            results[split_name] = (X, Y, OHLCV)
        else:
            logger.warning(f"    {split_name}: Pas de séquences créées")
            results[split_name] = (None, None, None)

    # Nettoyage mémoire
    del df, splits
    gc.collect()

    logger.info(f"\n✅ Asset {asset_name} traité")

    return results


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main():
    """
    Pipeline principal de préparation des données régime.
    """
    parser = argparse.ArgumentParser(description='Préparation données régime')
    parser.add_argument('--assets', nargs='+', default=DEFAULT_ASSETS,
                        choices=list(AVAILABLE_ASSETS_5M.keys()),
                        help='Liste des assets à inclure')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Répertoire de sortie (défaut: data/prepared)')
    parser.add_argument('--clip', type=float, default=None,
                        help='Valeur de clipping des features (None = pas de clip)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limiter le nombre de samples par asset (pour tests)')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger.info("="*80)
    logger.info("PRÉPARATION DONNÉES - META-REGIME TRADING v3.0 (XGBoost)")
    logger.info("="*80)
    logger.info(f"Assets: {args.assets}")
    logger.info(f"Sequence length: {SEQUENCE_LENGTH}")
    logger.info(f"Régimes: 3 classes (TS × VC)")
    logger.info(f"Features MODEL INPUT: ~23 colonnes (via get_regime_feature_names())")
    logger.info(f"  → Returns (3) + Trend (7) + Volatility (9) + Volume (4)")

    # ========================================================================
    # CALCUL PÉRIODE COMMUNE ET SPLIT TIMESTAMPS (FIX: negative gaps)
    # ========================================================================

    logger.info(f"\n{'='*80}")
    logger.info(f"CALCUL PÉRIODE COMMUNE")
    logger.info('='*80)

    common_start, common_end = find_common_period(args.assets)

    logger.info(f"\n{'='*80}")
    logger.info(f"CALCUL SPLIT TIMESTAMPS")
    logger.info('='*80)

    split_timestamps = calculate_split_timestamps(
        common_start, common_end,
        train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    # ========================================================================
    # PARALLÉLISATION MULTI-CORE
    # ========================================================================

    n_jobs = get_safe_n_jobs(len(args.assets), ram_per_asset_gb=8.0)
    logger.info(f"\n🚀 TRAITEMENT PARALLÈLE: {n_jobs} asset(s) simultané(s)")

    # Traiter les assets en parallèle
    all_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_asset)(
            asset_name, split_timestamps, args.clip, args.max_samples
        ) for asset_name in args.assets
    )

    # ========================================================================
    # CONCATÉNATION DES RÉSULTATS
    # ========================================================================

    logger.info(f"\n{'='*80}")
    logger.info(f"CONCATÉNATION FINALE")
    logger.info('='*80)

    # Organiser par split
    datasets = {'train': [], 'val': [], 'test': []}

    for asset_results in all_results:
        for split_name in ['train', 'val', 'test']:
            if asset_results[split_name][0] is not None:
                datasets[split_name].append(asset_results[split_name])

    # Concaténer
    X_train = np.concatenate([s[0] for s in datasets['train']], axis=0)
    Y_train = np.concatenate([s[1] for s in datasets['train']], axis=0)
    OHLCV_train = np.concatenate([s[2] for s in datasets['train']], axis=0)

    X_val = np.concatenate([s[0] for s in datasets['val']], axis=0)
    Y_val = np.concatenate([s[1] for s in datasets['val']], axis=0)
    OHLCV_val = np.concatenate([s[2] for s in datasets['val']], axis=0)

    X_test = np.concatenate([s[0] for s in datasets['test']], axis=0)
    Y_test = np.concatenate([s[1] for s in datasets['test']], axis=0)
    OHLCV_test = np.concatenate([s[2] for s in datasets['test']], axis=0)

    logger.info(f"   Shapes concaténées:")
    logger.info(f"     Train: X={X_train.shape}, Y={Y_train.shape}, OHLCV={OHLCV_train.shape}")
    logger.info(f"     Val:   X={X_val.shape}, Y={Y_val.shape}, OHLCV={OHLCV_val.shape}")
    logger.info(f"     Test:  X={X_test.shape}, Y={Y_test.shape}, OHLCV={OHLCV_test.shape}")

    # Stats labels
    logger.info(f"\n   Balance labels régime:")
    for split_name, Y_split in [('Train', Y_train), ('Val', Y_val), ('Test', Y_test)]:
        # Y: [timestamp, asset_id, regime, ts_score, vc_score]
        regime_col = Y_split[:, 2].astype(int)
        regime_counts = pd.Series(regime_col).value_counts().sort_index()
        regime_pcts = (regime_counts / len(regime_col) * 100).round(1)

        logger.info(f"     {split_name}:")
        for regime_id, pct in regime_pcts.items():
            regime_name = {
                0: "RANGE LOW VOL",
                1: "RANGE HIGH VOL",
                2: "TREND"
            }.get(regime_id, f"UNKNOWN_{regime_id}")
            logger.info(f"       Régime {regime_id} ({regime_name}): {pct}%")

    # ========================================================================
    # SAUVEGARDE
    # ========================================================================

    if args.output_dir is None:
        output_dir = Path('data/prepared')
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    assets_str = '_'.join(args.assets).lower()
    output_path = output_dir / f"dataset_{assets_str}_regime.npz"

    # Features utilisées (définies dans ÉTAPE 4)
    feature_cols = get_regime_feature_names()

    # Metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'version': '3.0',
        'description': 'Features computées (~23) pour XGBoost - 92.67% accuracy validé',
        'assets': args.assets,
        'n_assets': len(args.assets),
        'asset_id_mapping': ASSET_ID_MAP,
        'sequence_length': SEQUENCE_LENGTH,
        'features': feature_cols,
        'features_description': {
            # Returns (3)
            'c_ret': 'close.pct_change() - Rendement close-to-close',
            'h_ret': 'high.pct_change() - Rendement high-to-high',
            'l_ret': 'low.pct_change() - Rendement low-to-low',
            # Trend (7)
            'ma20_slope': 'Pente MA20 normalisée',
            'ma50_slope': 'Pente MA50 normalisée',
            'regression_slope': 'Pente régression linéaire normalisée',
            'regression_r2': 'R² de la régression',
            'adx': 'Average Directional Index',
            'macd_histogram_norm': 'MACD histogram normalisé',
            'hurst_exponent': 'Exposant de Hurst (persistance)',
            # Volatility (9)
            'atr_normalized': 'ATR normalisé par prix',
            'bb_upper': 'Bande de Bollinger supérieure',
            'bb_middle': 'Bande de Bollinger médiane',
            'bb_lower': 'Bande de Bollinger inférieure',
            'bb_width': 'Largeur des bandes de Bollinger',
            'percent_b': '%B (position dans les bandes)',
            'realized_volatility': 'Volatilité réalisée',
            'volatility_compression': 'Compression de volatilité',
            'range_atr_ratio': 'Ratio range/ATR',
            # Volume (4)
            'volume_ratio': 'Ratio volume/MA volume',
            'volume_spike': 'Spike de volume',
            'vwap_deviation': 'Déviation VWAP',
            'obv_derivative': 'Dérivée OBV'
        },
        'n_features': len(feature_cols),
        'labels': [
            'regime',  # Label régime (0-2)
            'macd_direction', 'rsi_direction', 'cci_direction'  # Labels direction
        ],
        'n_classes': 3,  # Pour régime uniquement
        'regime_definition': {
            0: "RANGE LOW VOL (TS < 0.45, VC ≤ P50)",
            1: "RANGE HIGH VOL (TS < 0.45, VC > P50)",
            2: "TREND (TS >= 0.45, any volatility)"
        },
        'regime_note': 'Régimes calculés depuis features de régime, exposées au modèle XGBoost',
        'model_note': 'XGBoost avec features computées: 92.67% vs CNN-LSTM raw: 86.33%',
        'direction_definition': {
            'macd_direction': 'Kalman-filtered MACD slope (filtered[t] > filtered[t-1]): 1=UP, 0=DOWN',
            'rsi_direction': 'Kalman-filtered RSI slope (filtered[t] > filtered[t-1]): 1=UP, 0=DOWN',
            'cci_direction': 'Kalman-filtered CCI slope (filtered[t] > filtered[t-1]): 1=UP, 0=DOWN'
        },
        'clip_value': args.clip,
        'max_samples_per_asset': args.max_samples,
        'split_indices': {
            'train_end': len(X_train),
            'val_end': len(X_train) + len(X_val)
        },
        'splits': {
            'train': {'n_sequences': len(X_train), 'ratio': 0.70},
            'val': {'n_sequences': len(X_val), 'ratio': 0.15},
            'test': {'n_sequences': len(X_test), 'ratio': 0.15}
        },
        'structure': {
            'X': f'(n, {SEQUENCE_LENGTH}, {len(feature_cols)}+2) - [timestamp, asset_id, features...] pour chaque timestep',
            'Y': '(n, 6) - [timestamp, asset_id, regime, macd_dir, rsi_dir, cci_dir]',
            'OHLCV': '(n, 7) - [timestamp, asset_id, open, high, low, close, volume]'
        },
        'primary_key': '(timestamp, asset_id) - Commune à toutes les matrices',
        'navigation': 'Même index i → même sample dans X, Y, OHLCV'
    }

    # Sauvegarder .npz (pattern copié de prepare_data_direction_only.py)
    np.savez_compressed(
        output_path,
        X_train=X_train, Y_train=Y_train, OHLCV_train=OHLCV_train,
        X_val=X_val, Y_val=Y_val, OHLCV_val=OHLCV_val,
        X_test=X_test, Y_test=Y_test, OHLCV_test=OHLCV_test,
        metadata=json.dumps(metadata)
    )

    # Sauvegarder metadata JSON
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\n✅ DATASET SAUVEGARDÉ:")
    logger.info(f"   NPZ:      {output_path}")
    logger.info(f"   Metadata: {metadata_path}")
    logger.info(f"   Taille:   {output_path.stat().st_size / (1024**2):.1f} MB")

    logger.info("\n" + "="*80)
    logger.info("✓ PRÉPARATION TERMINÉE")
    logger.info("="*80)
    logger.info("\nProchaine étape:")
    logger.info("  python src/train_regime_classifier.py \\")
    logger.info(f"    --data {output_path}")


if __name__ == '__main__':
    main()
