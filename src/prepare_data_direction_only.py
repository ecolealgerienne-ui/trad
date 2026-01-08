"""
Script de préparation des données avec Architecture Purified + Direction-Only.

PRINCIPE CLÉ: Features purifiées par indicateur + Labels Direction seulement
============================================================================

⚠️ MOTIVATION: Force n'apporte AUCUN bénéfice (tests validés -354% à -800% PnL)
→ Simplifier le modèle en ne prédisant QUE Direction (qui fonctionne)

Pour chaque indicateur (RSI, CCI, MACD), on génère UN DATASET SÉPARÉ:

**RSI**:
  - Features: 5 Close-based (C_ret, C_ma_5, C_ma_20, C_mom_3, C_mom_10)
  - Labels: 1 (rsi_dir seulement)
  - Gain attendu: +2-5% accuracy (focus sur une tâche)

**MACD**:
  - Features: 5 Close-based (idem RSI)
  - Labels: 1 (macd_dir seulement)
  - Gain attendu: +2-5% accuracy

**CCI**:
  - Features: 5 Volatility-aware (C_ret, H_ret, L_ret, Range_ret, ATR_norm)
  - Labels: 1 (cci_dir seulement)
  - Gain attendu: +2-5% accuracy

Pipeline:
1. Charger données brutes
2. Calculer TOUS les indicateurs (RSI, CCI, MACD)
3. Calculer features Close-based ET Volatility-aware
4. Pour CHAQUE indicateur:
   - Appliquer filtre (Kalman ou Octave) → Position
   - Calculer labels Direction SEULEMENT (pas Force)
   - Créer séquences avec features appropriées
   - Sauvegarder dataset séparé

Usage:
    # Avec filtre Kalman (défaut)
    python src/prepare_data_direction_only.py --assets BTC ETH BNB ADA LTC

    # Avec filtre Octave
    python src/prepare_data_direction_only.py --assets BTC ETH BNB ADA LTC --filter octave

Génère 3 fichiers (par exemple avec Kalman):
    - dataset_..._rsi_direction_only_kalman.npz
    - dataset_..._macd_direction_only_kalman.npz
    - dataset_..._cci_direction_only_kalman.npz

Ou avec Octave:
    - dataset_..._rsi_direction_only_octave20.npz
    - dataset_..._macd_direction_only_octave20.npz
    - dataset_..._cci_direction_only_octave20.npz
"""

import numpy as np
import pandas as pd
import argparse
import logging
import json
import os
from pathlib import Path
from datetime import datetime
from pykalman import KalmanFilter
import scipy.signal as signal
import gc  # Pour nettoyage mémoire explicite
from numpy.lib.stride_tricks import sliding_window_view  # Vectorisation x50
from joblib import Parallel, delayed  # Parallélisation multi-core
import psutil  # Détection RAM disponible

logger = logging.getLogger(__name__)

# Import modules locaux
from constants import (
    AVAILABLE_ASSETS_5M, DEFAULT_ASSETS,
    TRIM_EDGES,
    PREPARED_DATA_DIR,
    SEQUENCE_LENGTH,
    KALMAN_PROCESS_VAR,
    KALMAN_MEASURE_VAR,
)

# Périodes STANDARD des indicateurs
RSI_PERIOD = 14
CCI_PERIOD = 20
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Configuration Dual-Binary
Z_SCORE_WINDOW = 100
FORCE_THRESHOLD = 1.0
COLD_START_SKIP = 100

# Configuration ATR
ATR_PERIOD = 14

# Configuration Octave
OCTAVE_STEP = 0.20  # Paramètre du filtre Octave (0.20 recommandé)

# Mapping Asset Name → Asset ID (pour encodage dans les matrices)
ASSET_ID_MAP = {
    'BTC': 0,
    'ETH': 1,
    'BNB': 2,
    'ADA': 3,
    'LTC': 4
}


# =============================================================================
# PARALLÉLISATION INTELLIGENTE
# =============================================================================

def get_safe_n_jobs(n_assets: int, ram_per_asset_gb: float = 4.0) -> int:
    """
    Calcule le nombre de jobs parallèles selon la RAM disponible.

    Évite les crashes WSL en limitant le parallélisme selon la RAM.

    Args:
        n_assets: Nombre total d'assets à traiter
        ram_per_asset_gb: RAM peak estimée par asset (GB)

    Returns:
        Nombre de jobs sûrs (1 à min(n_assets, n_cores))
    """
    try:
        # RAM disponible (GB)
        available_ram_gb = psutil.virtual_memory().available / (1024**3)

        # Limite par RAM
        max_by_ram = max(1, int(available_ram_gb / ram_per_asset_gb))

        # Limite par CPU (laisser 1 core libre pour l'OS)
        n_cores = os.cpu_count() or 1
        max_by_cpu = max(1, n_cores - 1)

        # Prendre le minimum des contraintes
        n_jobs = min(max_by_ram, max_by_cpu, n_assets)

        logger.info(f"Parallélisation: {n_jobs} assets simultanés")
        logger.info(f"  RAM disponible: {available_ram_gb:.1f} GB")
        logger.info(f"  RAM par asset: {ram_per_asset_gb:.1f} GB")
        logger.info(f"  Cores CPU: {n_cores} ({max_by_cpu} utilisables)")

        return n_jobs
    except Exception as e:
        logger.warning(f"Erreur détection parallélisme: {e}, fallback n_jobs=1")
        return 1


# =============================================================================
# FILTRE OCTAVE (Butterworth + filtfilt) - NOUVEAU
# =============================================================================

def octave_filter_dual(data: np.ndarray,
                       step: float = OCTAVE_STEP,
                       order: int = 3) -> np.ndarray:
    """
    Applique le filtre Octave et calcule position + vélocité.

    Contrairement à Kalman qui extrait position et vélocité simultanément,
    Octave calcule d'abord la position filtrée, puis dérive la vélocité.

    Args:
        data: Signal à filtrer (np.ndarray)
        step: Paramètre du filtre Butterworth (défaut: 0.20)
        order: Ordre du filtre (défaut: 3)

    Returns:
        result: (N, 2) - [position, velocity]
    """
    valid_mask = ~np.isnan(data)
    if valid_mask.sum() < 10:
        result = np.full((len(data), 2), np.nan)
        return result

    # 1. Filtrer le signal (Butterworth + filtfilt)
    B, A = signal.butter(order, step, output='ba')

    # Appliquer filtfilt sur les données valides uniquement
    valid_data = data[valid_mask]
    filtered_valid = signal.filtfilt(B, A, valid_data)

    # 2. Calculer la vélocité comme diff() de la position
    # velocity[t] = position[t] - position[t-1]
    velocity_valid = np.diff(filtered_valid, prepend=filtered_valid[0])

    # 3. Reconstruire les arrays complets avec NaN
    position = np.full(len(data), np.nan)
    velocity = np.full(len(data), np.nan)

    position[valid_mask] = filtered_valid
    velocity[valid_mask] = velocity_valid

    # 4. Combiner position + velocity
    result = np.column_stack([position, velocity])

    return result


# =============================================================================
# FILTRE KALMAN CINÉMATIQUE (copié depuis dual_binary)
# =============================================================================

def kalman_filter_dual(data: np.ndarray,
                       process_var: float = KALMAN_PROCESS_VAR,
                       measure_var: float = KALMAN_MEASURE_VAR) -> np.ndarray:
    """Applique un filtre de Kalman CINÉMATIQUE (position + vélocité)."""
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


def calculate_force_labels(velocity: np.ndarray,
                           window: int = Z_SCORE_WINDOW,
                           threshold: float = FORCE_THRESHOLD) -> tuple:
    """Calcule les labels Force à partir de la vélocité."""
    vel_series = pd.Series(velocity)
    vel_t2 = vel_series.shift(2)
    rolling_std = vel_series.rolling(window=window, min_periods=1).std()
    z_scores = vel_t2 / (rolling_std + 1e-8)
    z_scores = np.clip(z_scores, -10, 10)
    force_labels = (np.abs(z_scores) > threshold).astype(int)

    return force_labels.values, z_scores.values


# =============================================================================
# CHARGEMENT DONNÉES
# =============================================================================

def load_data_with_index(file_path: str, asset_name: str = "Asset", max_samples: int = None) -> pd.DataFrame:
    """Charge les données CSV avec DatetimeIndex."""
    df = pd.read_csv(file_path, nrows=max_samples)

    date_col = None
    for col in ['date', 'datetime', 'time', 'timestamp', 'Date', 'Datetime']:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        raise ValueError(f"Colonne date non trouvée dans {file_path}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = 'datetime'
    df.columns = df.columns.str.lower()

    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    df = df.sort_index()

    logger.info(f"  {asset_name}: {len(df):,} lignes, {df.index[0]} → {df.index[-1]}")

    return df


# =============================================================================
# CALCUL INDICATEURS
# =============================================================================

def add_indicators_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les indicateurs RSI, CCI, MACD."""
    df = df.copy()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(span=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(span=RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    # CCI
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(CCI_PERIOD).mean()
    mad = tp.rolling(CCI_PERIOD).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (tp - sma_tp) / (0.015 * mad)

    # MACD
    ema_fast = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df['macd'] = macd_line - signal_line

    return df


# =============================================================================
# FEATURES "PURE SIGNAL" (h_ret, l_ret, c_ret UNIQUEMENT)
# =============================================================================

def add_pure_signal_features(df: pd.DataFrame, clip_value: float = 0.10) -> pd.DataFrame:
    """
    Calcule les 3 features "Pure Signal" recommandées par l'expert.

    Features (3 canaux):
    - h_ret: Extension haussière (High return)
    - l_ret: Extension baissière (Low return)
    - c_ret: Rendement Close-to-Close

    Notes:
    - o_ret: BANNI (bruit microstructure)
    - range_ret: BANNI (redondant pour CCI, bruit pour RSI/MACD)
    - Stationnarité garantie (returns vs prix bruts)
    - RSI/MACD n'utiliseront que c_ret (colonne 2)
    - CCI utilisera les 3 colonnes
    """
    df = df.copy()

    prev_close = df['close'].shift(1)

    # 3 features pures (ordre: h_ret, l_ret, c_ret)
    df['h_ret'] = (df['high'] - prev_close) / prev_close
    df['l_ret'] = (df['low'] - prev_close) / prev_close
    df['c_ret'] = (df['close'] - prev_close) / prev_close

    # Clipper les outliers
    for col in ['h_ret', 'l_ret', 'c_ret']:
        df[col] = df[col].clip(-clip_value, clip_value)

    return df


# =============================================================================
# LABELS DUAL-BINARY POUR UN INDICATEUR
# =============================================================================

def add_dual_labels_for_indicator(df: pd.DataFrame, indicator: str,
                                   filter_type: str = 'kalman',
                                   z_score_window: int = Z_SCORE_WINDOW,
                                   force_threshold: float = FORCE_THRESHOLD) -> pd.DataFrame:
    """
    Calcule les labels dual-binary pour UN indicateur.

    Args:
        indicator: 'rsi', 'cci', ou 'macd'
        filter_type: 'kalman' ou 'octave' (défaut: 'kalman')

    Returns:
        df avec colonnes ajoutées:
        - {indicator}_filtered
        - {indicator}_velocity
        - {indicator}_dir
        - {indicator}_force
        - {indicator}_z_score
    """
    df = df.copy()

    logger.info(f"     Processing {indicator.upper()} avec filtre {filter_type.upper()}...")

    # 1. Appliquer le filtre (Kalman ou Octave)
    raw_signal = df[indicator].values

    if filter_type == 'octave':
        filter_output = octave_filter_dual(raw_signal, step=OCTAVE_STEP)
    elif filter_type == 'kalman':
        filter_output = kalman_filter_dual(raw_signal)
    else:
        raise ValueError(f"filter_type inconnu: {filter_type}. Choix: 'kalman', 'octave'")

    position = filter_output[:, 0]
    velocity = filter_output[:, 1]

    df[f'{indicator}_filtered'] = position
    df[f'{indicator}_velocity'] = velocity

    # 2. Label Direction
    pos_series = pd.Series(position, index=df.index)
    pos_t2 = pos_series.shift(2)
    pos_t3 = pos_series.shift(3)
    df[f'{indicator}_dir'] = (pos_t2 > pos_t3).astype(int)

    # 2.5 Détection Transitions (Phase 2.11 - Weighted Loss)
    # Transition = label[i] != label[i-1]
    # Utilisé pour pondération augmentée dans la loss function
    dir_prev = df[f'{indicator}_dir'].shift(1)
    df[f'{indicator}_is_transition'] = (df[f'{indicator}_dir'] != dir_prev).astype(float)
    # Note: is_transition sera NaN pour premier sample (pas de prev), c'est OK

    # Stats transitions
    n_transitions = df[f'{indicator}_is_transition'].sum()
    n_total = (~df[f'{indicator}_is_transition'].isna()).sum()
    if n_total > 0:
        transition_pct = (n_transitions / n_total) * 100
        logger.info(f"       Transitions: {int(n_transitions)}/{n_total} ({transition_pct:.1f}%)")

    # 3. Label Force
    force_labels, z_scores = calculate_force_labels(
        velocity, window=z_score_window, threshold=force_threshold
    )
    df[f'{indicator}_force'] = force_labels
    df[f'{indicator}_z_score'] = z_scores

    # Stats
    n_valid = (~np.isnan(df[f'{indicator}_dir'])).sum()
    if n_valid > 0:
        dir_up_pct = df[f'{indicator}_dir'].sum() / n_valid * 100
        force_strong_pct = df[f'{indicator}_force'].sum() / n_valid * 100
        logger.info(f"       Direction: {dir_up_pct:.1f}% UP")
        logger.info(f"       Force: {force_strong_pct:.1f}% STRONG")

    return df


# =============================================================================
# CRÉATION SÉQUENCES POUR UN INDICATEUR
# =============================================================================

def create_sequences_for_indicator(df: pd.DataFrame,
                                    indicator: str,
                                    feature_cols: list,
                                    asset_name: str,
                                    asset_id: int,
                                    seq_length: int = SEQUENCE_LENGTH,
                                    cold_start_skip: int = COLD_START_SKIP) -> tuple:
    """
    Crée les séquences pour UN indicateur avec features + métadonnées intégrées.

    NOUVELLE STRUCTURE (timestamp, asset_id intégrés dans CHAQUE matrice):
    - X: (n, seq_length, n_features+2) = [timestamp, asset_id, features...]
    - Y: (n, n_labels+2) = [timestamp, asset_id, direction]
    - T: (n, 3) = [timestamp, asset_id, is_transition]
    - OHLCV: (n, 7) = [timestamp, asset_id, open, high, low, close, volume]

    Args:
        indicator: 'rsi', 'cci', ou 'macd'
        feature_cols: Liste des features à utiliser
        asset_name: Nom de l'asset ('BTC', 'ETH', etc.)
        asset_id: ID encodé de l'asset (0-4)

    Returns:
        X: (n, seq_length, n_features+2) avec timestamp et asset_id intégrés
        Y: (n, 3) avec [timestamp, asset_id, direction]
        T: (n, 3) avec [timestamp, asset_id, is_transition]
        OHLCV: (n, 7) avec [timestamp, asset_id, O, H, L, C, V]
    """
    # Colonnes label (1 pour cet indicateur - Direction seulement)
    label_cols = [f'{indicator}_dir']
    transition_col = f'{indicator}_is_transition'

    # Colonnes OHLCV brutes
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']

    # Supprimer lignes avec NaN
    cols_needed = feature_cols + label_cols + [transition_col] + ohlcv_cols
    df_clean = df.dropna(subset=cols_needed)

    logger.info(f"     Lignes valides: {len(df_clean)}/{len(df)} "
                f"({len(df) - len(df_clean)} supprimées pour NaN)")

    # Extraire arrays avec types optimisés
    features = df_clean[feature_cols].values.astype(np.float32)  # (N, n_features)
    labels = df_clean[label_cols].values.astype(np.float32)      # (N, 1) - Direction seulement
    transitions = df_clean[transition_col].values.astype(np.int8)  # (N,) - Indicateur transitions
    ohlcv = df_clean[ohlcv_cols].values.astype(np.float32)       # (N, 5) - Prix et volume bruts
    # Convertir timestamps en int64 (nanosecondes depuis epoch)
    timestamps = df_clean.index.values.astype('datetime64[ns]').astype(np.int64)  # (N,) - Timestamps

    # Cold start
    start_index = seq_length + cold_start_skip

    logger.info(f"     Cold Start: skip premiers {cold_start_skip} samples")
    logger.info(f"     Start index: {start_index}")

    # ========================================================================
    # VECTORISATION avec sliding_window_view (x30-50 plus rapide) 🚀
    # ========================================================================
    n_samples = len(features)
    n_features = features.shape[1]

    # Étape 1: Créer array combiné [timestamp, asset_id, features] pour TOUT le dataset
    # Shape: (n_samples, n_features+2)
    combined = np.zeros((n_samples, n_features + 2), dtype=np.float64)
    combined[:, 0] = timestamps.astype(np.float64)
    combined[:, 1] = float(asset_id)
    combined[:, 2:] = features  # float32 → float64 (cast automatique)

    # Étape 2: Appliquer sliding_window_view (opération instantanée, pas de copie !)
    # Shape: (n_windows, seq_length, n_features+2)
    # Note: sliding_window_view retourne une vue, pas une copie → économie RAM
    X_all_windows = sliding_window_view(combined, window_shape=seq_length, axis=0)
    # Reshape pour avoir la bonne structure (bug de sliding_window_view avec 2D)
    X_all_windows = X_all_windows[:, 0, :, :]  # (n_windows, seq_length, n_features+2)

    # Étape 3: Appliquer cold start (skip premiers start_index)
    X = X_all_windows[start_index - seq_length:].copy()  # .copy() pour libérer la vue

    # Étape 4: Y, T, OHLCV (vectorisé sans boucle)
    n_sequences = len(X)

    # Y: [timestamp, asset_id, direction]
    Y = np.zeros((n_sequences, 3), dtype=np.float64)
    Y[:, 0] = timestamps[start_index:start_index + n_sequences].astype(np.float64)
    Y[:, 1] = float(asset_id)
    Y[:, 2] = labels[start_index:start_index + n_sequences, 0]

    # T: [timestamp, asset_id, is_transition]
    T = np.zeros((n_sequences, 3), dtype=np.float64)
    T[:, 0] = timestamps[start_index:start_index + n_sequences].astype(np.float64)
    T[:, 1] = float(asset_id)
    T[:, 2] = transitions[start_index:start_index + n_sequences].astype(np.float64)

    # OHLCV: [timestamp, asset_id, O, H, L, C, V]
    OHLCV = np.zeros((n_sequences, 7), dtype=np.float64)
    OHLCV[:, 0] = timestamps[start_index:start_index + n_sequences].astype(np.float64)
    OHLCV[:, 1] = float(asset_id)
    OHLCV[:, 2:] = ohlcv[start_index:start_index + n_sequences]

    # Stats transitions dans les séquences créées
    n_transitions_seqs = T[:, 2].sum()  # Colonne 2 = is_transition
    transition_pct_seqs = (n_transitions_seqs / len(T)) * 100

    logger.info(f"     Séquences créées:")
    logger.info(f"       X={X.shape} - [timestamp, asset_id, {len(feature_cols)} features] × {seq_length} steps")
    logger.info(f"       Y={Y.shape} - [timestamp, asset_id, direction]")
    logger.info(f"       T={T.shape} - [timestamp, asset_id, is_transition]")
    logger.info(f"       OHLCV={OHLCV.shape} - [timestamp, asset_id, O, H, L, C, V]")
    logger.info(f"     Transitions: {int(n_transitions_seqs)}/{len(T)} ({transition_pct_seqs:.1f}%)")
    logger.info(f"     Asset: {asset_name} (ID={asset_id})")

    return X, Y, T, OHLCV


# =============================================================================
# SPLIT CHRONOLOGIQUE
# =============================================================================

def split_chronological(X: np.ndarray, Y: np.ndarray, T: np.ndarray, OHLCV: np.ndarray,
                        train_ratio: float = 0.70,
                        val_ratio: float = 0.15,
                        gap: int = SEQUENCE_LENGTH) -> dict:
    """
    Split chronologique avec GAP.

    NOUVELLE STRUCTURE: Inclut OHLCV dans les splits.

    Args:
        X: Features avec [timestamp, asset_id, features...]
        Y: Labels avec [timestamp, asset_id, direction]
        T: Transitions avec [timestamp, asset_id, is_transition]
        OHLCV: Prix bruts avec [timestamp, asset_id, O, H, L, C, V]

    Returns:
        Dict avec splits train/val/test incluant X, Y, T, OHLCV
    """
    n = len(X)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_end_gap = train_end - gap
    val_start = train_end
    val_end_gap = val_end - gap
    test_start = val_end

    return {
        'train': (X[:train_end_gap], Y[:train_end_gap], T[:train_end_gap], OHLCV[:train_end_gap]),
        'val': (X[val_start:val_end_gap], Y[val_start:val_end_gap], T[val_start:val_end_gap], OHLCV[val_start:val_end_gap]),
        'test': (X[test_start:], Y[test_start:], T[test_start:], OHLCV[test_start:])
    }


# =============================================================================
# PRÉPARATION POUR UN INDICATEUR
# =============================================================================

def prepare_indicator_dataset(df: pd.DataFrame, asset_name: str, indicator: str,
                              feature_cols: list, filter_type: str = 'kalman',
                              clip_value: float = 0.10) -> tuple:
    """
    Prépare le dataset pour UN indicateur avec features purifiées + métadonnées intégrées.

    NOUVELLE STRUCTURE: Retourne X, Y, T, OHLCV avec (timestamp, asset_id) intégrés.

    Args:
        df: DataFrame avec OHLC + indicateurs calculés
        asset_name: Nom de l'asset ('BTC', 'ETH', etc.)
        indicator: 'rsi', 'cci', ou 'macd'
        feature_cols: Colonnes features à utiliser
        filter_type: 'kalman' ou 'octave'

    Returns:
        (X, Y, T, OHLCV) pour cet indicateur avec métadonnées intégrées
    """
    logger.info(f"\n  {asset_name} - {indicator.upper()}: Préparation avec filtre {filter_type.upper()}...")

    # Obtenir l'ID de l'asset
    asset_id = ASSET_ID_MAP.get(asset_name)
    if asset_id is None:
        raise ValueError(f"Asset inconnu: {asset_name}. Valides: {list(ASSET_ID_MAP.keys())}")

    # Calculer labels dual-binary pour cet indicateur
    df = add_dual_labels_for_indicator(df, indicator, filter_type=filter_type)

    # Créer séquences (avec transitions + OHLCV + métadonnées intégrées)
    X, Y, T, OHLCV = create_sequences_for_indicator(
        df, indicator, feature_cols, asset_name, asset_id
    )

    return X, Y, T, OHLCV


# =============================================================================
# TRAITEMENT D'UN ASSET (Pour parallélisation)
# =============================================================================

def process_single_asset(asset_name: str,
                         filter_type: str,
                         clip_value: float,
                         max_samples: int = None) -> dict:
    """
    Traite UN asset pour les 3 indicateurs (RSI, MACD, CCI).

    Fonction wrapper pour la parallélisation avec joblib.

    Args:
        asset_name: Nom de l'asset ('BTC', 'ETH', etc.)
        filter_type: 'kalman' ou 'octave'
        clip_value: Valeur de clipping
        max_samples: Limite de lignes (None = toutes)

    Returns:
        dict avec les splits (train, val, test) pour chaque indicateur
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"ASSET: {asset_name}")
    logger.info('='*80)

    file_path = AVAILABLE_ASSETS_5M[asset_name]
    asset_id = ASSET_ID_MAP[asset_name]

    # Features par indicateur (Architecture "Pure Signal")
    features_rsi = ['c_ret']  # 1 feature: Close uniquement
    features_macd = ['c_ret']  # 1 feature: Close uniquement
    features_cci = ['h_ret', 'l_ret', 'c_ret']  # 3 features: High, Low, Close

    # 1. Charger
    df = load_data_with_index(file_path, asset_name, max_samples=max_samples)
    if max_samples:
        logger.info(f"  {asset_name}: {len(df)} lignes, {df.index[0]} → {df.index[-1]} (limité à {max_samples})")
    else:
        logger.info(f"  {asset_name}: {len(df)} lignes, {df.index[0]} → {df.index[-1]}")

    # 2. Indicateurs
    df = add_indicators_to_df(df)
    logger.info(f"     Indicateurs: RSI, CCI, MACD")

    # 3. Features Pure Signal (h_ret, l_ret, c_ret)
    df = add_pure_signal_features(df, clip_value)
    logger.info(f"     Features Pure Signal: 3 canaux (h_ret, l_ret, c_ret)")

    # 4. TRIM edges
    df = df.iloc[TRIM_EDGES:-TRIM_EDGES]
    logger.info(f"     Après trim ±{TRIM_EDGES}: {len(df)} lignes")

    # Résultat pour cet asset
    asset_results = {
        'rsi': {'train': None, 'val': None, 'test': None},
        'macd': {'train': None, 'val': None, 'test': None},
        'cci': {'train': None, 'val': None, 'test': None}
    }

    # 5. Préparer pour chaque indicateur
    for indicator, feature_cols in [
        ('rsi', features_rsi),      # 1 feature: c_ret
        ('macd', features_macd),    # 1 feature: c_ret
        ('cci', features_cci)       # 3 features: h_ret, l_ret, c_ret
    ]:
        X, Y, T, OHLCV = prepare_indicator_dataset(
            df, asset_name, indicator, feature_cols, filter_type=filter_type, clip_value=clip_value
        )

        # Split chronologique (avec transitions + OHLCV)
        splits = split_chronological(X, Y, T, OHLCV)

        # Stocker les splits
        asset_results[indicator] = splits

        logger.info(f"     Split: Train={len(splits['train'][0])}, "
                   f"Val={len(splits['val'][0])}, Test={len(splits['test'][0])}")

        # Nettoyage mémoire immédiat après stockage
        del X, Y, T, OHLCV, splits
        gc.collect()

    logger.info(f"  ✅ {asset_name} traité")

    return asset_results


# =============================================================================
# PRÉPARATION ET SAUVEGARDE MULTI-INDICATEURS (AVEC PARALLÉLISATION)
# =============================================================================

def prepare_and_save_all(assets: list = None,
                         output_dir: str = None,
                         filter_type: str = 'kalman',
                         clip_value: float = 0.10,
                         max_samples: int = None) -> dict:
    """
    Prépare les 3 datasets (RSI, MACD, CCI) en une seule exécution.

    Args:
        assets: Liste des assets
        output_dir: Répertoire de sortie
        filter_type: 'kalman' ou 'octave'
        clip_value: Valeur de clipping
        max_samples: Nombre max de lignes par asset (None = toutes)

    Returns:
        dict avec les chemins des 3 fichiers .npz générés
    """
    if assets is None:
        assets = DEFAULT_ASSETS

    invalid_assets = [a for a in assets if a not in AVAILABLE_ASSETS_5M]
    if invalid_assets:
        raise ValueError(f"Assets invalides: {invalid_assets}")

    logger.info("="*80)
    logger.info("PRÉPARATION MULTI-DATASETS (Pure Signal + Direction-Only)")
    logger.info("="*80)
    logger.info(f"Assets: {', '.join(assets)}")
    logger.info(f"Filtre: {filter_type.upper()}")
    if max_samples:
        logger.info(f"⚠️  MODE TEST: {max_samples} lignes max par asset")
    logger.info(f"Génération de 3 datasets séparés:")
    logger.info(f"  1. RSI  - Features: c_ret (1)")
    logger.info(f"  2. MACD - Features: c_ret (1)")
    logger.info(f"  3. CCI  - Features: h_ret, l_ret, c_ret (3)")
    logger.info(f"Labels: Direction SEULEMENT (1 par indicateur)")
    logger.info(f"Architecture: Pure Signal (Force supprimée car inutile)")

    # ========================================================================
    # PARALLÉLISATION MULTI-CORE (x2-4 selon RAM/CPU) 🚀
    # ========================================================================

    # Calculer n_jobs selon RAM disponible
    n_jobs = get_safe_n_jobs(len(assets), ram_per_asset_gb=4.0)

    logger.info(f"\n🚀 TRAITEMENT PARALLÈLE: {n_jobs} asset(s) simultané(s)")

    # Traiter les assets en parallèle
    all_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_asset)(
            asset_name, filter_type, clip_value, max_samples
        ) for asset_name in assets
    )

    # Réorganiser les résultats par indicateur
    datasets = {
        'rsi': {'train': [], 'val': [], 'test': []},
        'macd': {'train': [], 'val': [], 'test': []},
        'cci': {'train': [], 'val': [], 'test': []}
    }

    for asset_results in all_results:
        for indicator in ['rsi', 'macd', 'cci']:
            for split_name in ['train', 'val', 'test']:
                datasets[indicator][split_name].append(asset_results[indicator][split_name])

    logger.info(f"\n✅ Tous les assets traités ({len(assets)} assets, {n_jobs} jobs parallèles)")

    # Concaténer et sauvegarder chaque indicateur
    output_paths = {}

    for indicator in ['rsi', 'macd', 'cci']:
        logger.info(f"\n{'='*80}")
        logger.info(f"SAUVEGARDE DATASET: {indicator.upper()}")
        logger.info('='*80)

        # Concaténer tous les assets (X, Y, T, OHLCV)
        X_train = np.concatenate([s[0] for s in datasets[indicator]['train']], axis=0)
        Y_train = np.concatenate([s[1] for s in datasets[indicator]['train']], axis=0)
        T_train = np.concatenate([s[2] for s in datasets[indicator]['train']], axis=0)
        OHLCV_train = np.concatenate([s[3] for s in datasets[indicator]['train']], axis=0)

        X_val = np.concatenate([s[0] for s in datasets[indicator]['val']], axis=0)
        Y_val = np.concatenate([s[1] for s in datasets[indicator]['val']], axis=0)
        T_val = np.concatenate([s[2] for s in datasets[indicator]['val']], axis=0)
        OHLCV_val = np.concatenate([s[3] for s in datasets[indicator]['val']], axis=0)

        X_test = np.concatenate([s[0] for s in datasets[indicator]['test']], axis=0)
        Y_test = np.concatenate([s[1] for s in datasets[indicator]['test']], axis=0)
        T_test = np.concatenate([s[2] for s in datasets[indicator]['test']], axis=0)
        OHLCV_test = np.concatenate([s[3] for s in datasets[indicator]['test']], axis=0)

        logger.info(f"   Shapes concaténées:")
        logger.info(f"     Train: X={X_train.shape}, Y={Y_train.shape}, T={T_train.shape}, OHLCV={OHLCV_train.shape}")
        logger.info(f"     Val:   X={X_val.shape}, Y={Y_val.shape}, T={T_val.shape}, OHLCV={OHLCV_val.shape}")
        logger.info(f"     Test:  X={X_test.shape}, Y={Y_test.shape}, T={T_test.shape}, OHLCV={OHLCV_test.shape}")

        # Stats labels (Direction-only avec nouvelle structure)
        logger.info(f"\n   Balance labels:")
        for split_name, Y_split, T_split in [('Train', Y_train, T_train), ('Val', Y_val, T_val), ('Test', Y_test, T_test)]:
            # Y: [timestamp, asset_id, direction] → colonne 2 = direction
            # T: [timestamp, asset_id, is_transition] → colonne 2 = is_transition
            dir_pct = Y_split[:, 2].astype(float).mean() * 100
            trans_pct = T_split[:, 2].astype(float).mean() * 100
            logger.info(f"     {split_name}: Direction {dir_pct:.1f}% UP, Transitions {trans_pct:.1f}%")

        # Sauvegarder
        if output_dir is None:
            output_dir = Path('data/prepared')
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        assets_str = '_'.join(assets).lower()

        # Nom de fichier avec filtre
        if filter_type == 'octave':
            filter_suffix = f"octave{int(OCTAVE_STEP*100)}"
        else:
            filter_suffix = filter_type

        # Phase 2.11: Ajouter suffixe _wt pour tests isolés (ne pas écraser existant)
        output_path = output_dir / f"dataset_{assets_str}_{indicator}_direction_only_{filter_suffix}_wt.npz"

        # Features et metadata par indicateur
        if indicator == 'rsi':
            feature_list = features_rsi
            n_features = 1
            feature_type = 'Pure Signal: c_ret uniquement (0% bruit)'
            justification = 'RSI utilise Close dans sa formule. High/Low = bruit toxique.'
        elif indicator == 'macd':
            feature_list = features_macd
            n_features = 1
            feature_type = 'Pure Signal: c_ret uniquement (0% bruit)'
            justification = 'MACD utilise Close dans sa formule. High/Low = bruit toxique.'
        else:  # cci
            feature_list = features_cci
            n_features = 3
            feature_type = 'Pure Signal: h_ret, l_ret, c_ret (Typical Price)'
            justification = 'CCI utilise (H+L+C)/3 dans sa formule. High/Low justifiés.'

        # Métadonnées du filtre
        if filter_type == 'kalman':
            filter_metadata = {
                'kalman_params': {
                    'process_var': KALMAN_PROCESS_VAR,
                    'measure_var': KALMAN_MEASURE_VAR,
                    'model': 'cinematic (position + velocity)'
                }
            }
        else:  # octave
            filter_metadata = {
                'octave_params': {
                    'step': OCTAVE_STEP,
                    'order': 3,
                    'method': 'Butterworth + filtfilt',
                    'velocity': 'diff(filtered_position)'
                }
            }

        metadata = {
            'created_at': datetime.now().isoformat(),
            'version': 'pure_signal_direction_only_v2_with_metadata',
            'architecture': 'Pure Signal + Direction-Only + Metadata Intégrées',
            'indicator': indicator.upper(),
            'assets': assets,
            'asset_id_mapping': ASSET_ID_MAP,
            'filter_type': filter_type,
            **filter_metadata,
            'cold_start_skip': COLD_START_SKIP,
            'labels': 1,
            'label_names': [f'{indicator}_dir'],
            'label_definitions': {
                'direction': 'filtered[t-2] > filtered[t-3]'
            },
            'motivation': 'Force n\'apporte aucun bénéfice (-354% à -800% PnL). Focus Direction seulement.',
            'clip_value': clip_value,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'sequence_length': SEQUENCE_LENGTH,
            'n_features': n_features,
            'features': feature_list,
            'feature_type': feature_type,
            'justification': justification,
            'features_banned': ['o_ret (microstructure)', 'range_ret (redundant/noise)'],
            'stationarity': 'Returns (stationnaires) vs prix bruts (non-stationnaires)',
            'structure': {
                'X': f'(n, {SEQUENCE_LENGTH}, {n_features}+2) - [timestamp, asset_id, features...] pour chaque timestep',
                'Y': '(n, 3) - [timestamp, asset_id, direction]',
                'T': '(n, 3) - [timestamp, asset_id, is_transition]',
                'OHLCV': '(n, 7) - [timestamp, asset_id, open, high, low, close, volume]'
            },
            'primary_key': '(timestamp, asset_id) - Commune à toutes les matrices',
            'navigation': 'Même index i → même sample dans X, Y, T, OHLCV'
        }

        np.savez_compressed(
            output_path,
            X_train=X_train, Y_train=Y_train, T_train=T_train, OHLCV_train=OHLCV_train,
            X_val=X_val, Y_val=Y_val, T_val=T_val, OHLCV_val=OHLCV_val,
            X_test=X_test, Y_test=Y_test, T_test=T_test, OHLCV_test=OHLCV_test,
            metadata=json.dumps(metadata)
        )

        logger.info(f"\n   ✅ Sauvegardé: {output_path}")

        # Metadata JSON
        metadata_path = str(output_path).replace('.npz', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        output_paths[indicator] = str(output_path)

        # Nettoyage mémoire après sauvegarde
        del X_train, Y_train, T_train, OHLCV_train
        del X_val, Y_val, T_val, OHLCV_val
        del X_test, Y_test, T_test, OHLCV_test
        del datasets[indicator]
        gc.collect()

    return output_paths


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Point d'entrée CLI."""
    available_assets = list(AVAILABLE_ASSETS_5M.keys())

    parser = argparse.ArgumentParser(
        description="Prépare les datasets avec Pure Signal + Direction-Only (Force supprimée)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Architecture "Pure Signal Direction-Only" (Simplifiée):
=======================================================

⚠️ MOTIVATION: Force n'apporte AUCUN bénéfice (-354% à -800% PnL)
→ Simplifier le modèle en ne prédisant QUE Direction (qui fonctionne)

Génère 3 fichiers .npz (un par indicateur):

1. RSI  - Features: c_ret (1) + Labels: Direction (1)
   └─ Shape: X=(n, 25, 1), Y=(n, 1)
   └─ Justification: RSI utilise Close uniquement. High/Low = bruit toxique.

2. MACD - Features: c_ret (1) + Labels: Direction (1)
   └─ Shape: X=(n, 25, 1), Y=(n, 1)
   └─ Justification: MACD utilise Close uniquement. High/Low = bruit toxique.

3. CCI  - Features: h_ret, l_ret, c_ret (3) + Labels: Direction (1)
   └─ Shape: X=(n, 25, 3), Y=(n, 1)
   └─ Justification: CCI utilise (H+L+C)/3. High/Low justifiés.

Features Bannies:
- o_ret (Open): Bruit de microstructure
- range_ret: Redondant pour CCI, bruit pour RSI/MACD

Gains Attendus (Direction-only):
- RSI/MACD: +2-5% accuracy (focus une tâche au lieu de 2)
- CCI: +2-5% accuracy
- Convergence: Plus rapide (single-task learning)
- Trading: Force inutile confirmé empiriquement

Exemples:
  # Avec filtre Kalman (défaut)
  python src/prepare_data_direction_only.py --assets BTC ETH BNB ADA LTC

  # Avec filtre Octave
  python src/prepare_data_direction_only.py --assets BTC ETH BNB ADA LTC --filter octave

Assets disponibles: {', '.join(available_assets)}
        """
    )

    parser.add_argument('--assets', '-a', type=str, nargs='+',
                        default=DEFAULT_ASSETS,
                        choices=available_assets,
                        help='Assets à inclure')
    parser.add_argument('--filter', '-f', type=str, default='kalman',
                        choices=['kalman', 'octave'],
                        help='Type de filtre (défaut: kalman)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Répertoire de sortie')
    parser.add_argument('--clip', type=float, default=0.10,
                        help='Valeur de clipping')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Nombre max de lignes par asset (pour tests rapides)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    output_paths = prepare_and_save_all(
        assets=args.assets,
        output_dir=args.output_dir,
        filter_type=args.filter,
        clip_value=args.clip,
        max_samples=args.max_samples
    )

    print(f"\n{'='*80}")
    print("✅ TERMINÉ! 3 datasets générés:")
    print('='*80)
    for indicator, path in output_paths.items():
        print(f"  {indicator.upper()}: {path}")

    print(f"\nPour entraîner:")
    for indicator, path in output_paths.items():
        print(f"  python src/train.py --data {path} --indicator {indicator}")


if __name__ == '__main__':
    main()
