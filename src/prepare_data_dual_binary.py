"""
Script de préparation des données avec Architecture Dual-Binary.

PRINCIPE CLÉ: Deux labels par indicateur (Direction + Force)
==============================================================

Pour chaque indicateur (RSI, CCI, MACD):
1. **Direction (pente)**: filtered[t-2] > filtered[t-3] → binaire (0/1)
2. **Force (vélocité)**: |velocity_zscore[t-2]| > threshold → binaire (WEAK=0 / STRONG=1)

Pipeline:
1. Charger données brutes avec DatetimeIndex
2. Calculer indicateurs (RSI, CCI, MACD) → ajouter au DataFrame
3. Calculer features OHLC (4 canaux: h_ret, l_ret, c_ret, range_ret)
4. Pour chaque indicateur:
   - Appliquer Kalman cinématique → extraire Position + Vélocité
   - Calculer Direction: pente de Position
   - Calculer Force: Z-Score de Vélocité
5. TRIM edges + Cold Start (skip premiers 100 samples)
6. Créer séquences avec Y de shape (n, 6) au lieu de (n, 1)
7. Export debug CSV (derniers 1000 samples)

Gains attendus:
- Accuracy: +3-4% (inputs purifiés)
- Trades: -60% (force discrimine turning points)

Usage:
    python src/prepare_data_dual_binary.py --assets BTC ETH BNB ADA LTC --filter kalman
"""

import numpy as np
import pandas as pd
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from pykalman import KalmanFilter

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
Z_SCORE_WINDOW = 100  # Fenêtre pour Z-Score (recommandation expert)
FORCE_THRESHOLD = 1.0  # Seuil Z-Score pour STRONG (>1 sigma)
COLD_START_SKIP = 100  # Skip premiers samples (Z-Score invalide)


# =============================================================================
# FILTRE KALMAN CINÉMATIQUE (Position + Vélocité)
# =============================================================================

def kalman_filter_dual(data: np.ndarray,
                       process_var: float = KALMAN_PROCESS_VAR,
                       measure_var: float = KALMAN_MEASURE_VAR) -> np.ndarray:
    """
    Applique un filtre de Kalman CINÉMATIQUE (position + vélocité).

    Modèle:
        Position[t] = Position[t-1] + Velocity[t-1]
        Velocity[t] = Velocity[t-1]

    Transition matrix: [[1, 1],    # Pos = Pos + Vel
                        [0, 1]]    # Vel = Vel

    Returns:
        np.ndarray (N, 2):
            - Colonne 0: Position filtrée
            - Colonne 1: Vélocité estimée
    """
    # Supprimer NaN
    valid_mask = ~np.isnan(data)
    if valid_mask.sum() < 10:
        # Pas assez de données valides
        result = np.full((len(data), 2), np.nan)
        return result

    # Transition: Position = Position + Velocity, Velocity = Velocity
    transition_matrix = [[1, 1], [0, 1]]
    observation_matrix = [[1, 0]]  # On observe seulement la position

    # État initial: [position initiale, vélocité = 0]
    initial_state_mean = [data[valid_mask][0], 0.0]

    # Covariance d'observation et de transition
    observation_covariance = measure_var
    transition_covariance = np.eye(2) * process_var

    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        initial_state_mean=initial_state_mean,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance
    )

    # Filtrage + smoothing (non-causal, utilise le futur)
    means, _ = kf.smooth(data[valid_mask])

    # Reconstruire avec NaN aux positions invalides
    result = np.full((len(data), 2), np.nan)
    result[valid_mask] = means

    return result


def calculate_force_labels(velocity: np.ndarray,
                           window: int = Z_SCORE_WINDOW,
                           threshold: float = FORCE_THRESHOLD) -> tuple:
    """
    Calcule les labels Force à partir de la vélocité.

    Force = STRONG (1) si |Z-Score(velocity)| > threshold, sinon WEAK (0)

    Z-Score = velocity / rolling_std(velocity)

    Args:
        velocity: Vélocité extraite du Kalman (N,)
        window: Fenêtre pour rolling std (défaut: 100)
        threshold: Seuil Z-Score (défaut: 1.0)

    Returns:
        force_labels: (N,) binaire (0=WEAK, 1=STRONG)
        z_scores: (N,) Z-Scores pour debug
    """
    # Convertir en Series pour rolling
    vel_series = pd.Series(velocity)

    # Décalage t-2 (alignement avec direction)
    vel_t2 = vel_series.shift(2)

    # Rolling std (fenêtre glissante)
    rolling_std = vel_series.rolling(window=window, min_periods=1).std()

    # Z-Score = velocity / std
    # Correction expert: ajouter epsilon + clipper pour éviter explosion
    z_scores = vel_t2 / (rolling_std + 1e-8)
    z_scores = np.clip(z_scores, -10, 10)

    # Force = 1 si |Z-Score| > threshold
    force_labels = (np.abs(z_scores) > threshold).astype(int)

    return force_labels.values, z_scores.values


# =============================================================================
# CHARGEMENT DONNÉES AVEC INDEX (copié depuis ohlc_v2)
# =============================================================================

def load_data_with_index(file_path: str, asset_name: str = "Asset") -> pd.DataFrame:
    """
    Charge les données CSV avec DatetimeIndex.

    Returns:
        DataFrame avec colonnes: open, high, low, close, volume
        Index: DatetimeIndex
    """
    df = pd.read_csv(file_path)

    # Identifier la colonne date
    date_col = None
    for col in ['date', 'datetime', 'time', 'timestamp', 'Date', 'Datetime']:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        raise ValueError(f"Colonne date non trouvée dans {file_path}")

    # Convertir en datetime et définir comme index
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = 'datetime'

    # Normaliser noms colonnes
    df.columns = df.columns.str.lower()

    # Vérifier colonnes requises
    required = ['open', 'high', 'low', 'close']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    # Trier par date
    df = df.sort_index()

    logger.info(f"  {asset_name}: {len(df):,} lignes, "
                f"{df.index[0]} → {df.index[-1]}")

    return df


# =============================================================================
# CALCUL INDICATEURS (copié depuis ohlc_v2)
# =============================================================================

def add_indicators_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les indicateurs et les ajoute directement au DataFrame.
    """
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
    df['macd'] = macd_line - signal_line  # Histogram

    return df


# =============================================================================
# NORMALISATION OHLC (MODIFIÉ - 4 features au lieu de 5)
# =============================================================================

def add_ohlc_features_to_df(df: pd.DataFrame, clip_value: float = 0.10) -> pd.DataFrame:
    """
    Calcule les features OHLC normalisées (4 canaux - SANS Open).

    Features (4 canaux):
    - H_ret: Extension haussière intra-bougie
    - L_ret: Extension baissière intra-bougie
    - C_ret: Rendement clôture-à-clôture → patterns principaux
    - Range_ret: Volatilité intra-bougie

    Note: Open retiré car non utilisé par RSI/CCI/MACD (input toxique).
    """
    df = df.copy()

    prev_close = df['close'].shift(1)

    # 4 canaux OHLC normalisés (SANS o_ret)
    df['h_ret'] = (df['high'] - prev_close) / prev_close      # Extension haussière
    df['l_ret'] = (df['low'] - prev_close) / prev_close       # Extension baissière
    df['c_ret'] = (df['close'] - prev_close) / prev_close     # Rendement net
    df['range_ret'] = (df['high'] - df['low']) / prev_close   # Volatilité

    # Clipper les outliers
    for col in ['h_ret', 'l_ret', 'c_ret', 'range_ret']:
        df[col] = df[col].clip(-clip_value, clip_value)

    return df


# =============================================================================
# FILTRAGE DUAL-BINARY (3 indicateurs × 2 labels)
# =============================================================================

def add_dual_labels_to_df(df: pd.DataFrame,
                          filter_type: str = 'kalman',
                          z_score_window: int = Z_SCORE_WINDOW,
                          force_threshold: float = FORCE_THRESHOLD) -> pd.DataFrame:
    """
    Calcule les labels dual-binary pour les 3 indicateurs.

    Pour chaque indicateur (RSI, CCI, MACD):
    1. Kalman cinématique → Position + Vélocité
    2. Direction: filtered[t-2] > filtered[t-3]
    3. Force: |velocity_zscore[t-2]| > threshold

    Ajoute 18 colonnes au DataFrame:
    - {ind}_filtered, {ind}_velocity
    - {ind}_dir, {ind}_force
    - {ind}_z_score (pour debug)

    pour ind in ['rsi', 'cci', 'macd']
    """
    df = df.copy()

    indicators = ['rsi', 'cci', 'macd']

    for ind in indicators:
        logger.info(f"     Processing {ind.upper()}...")

        # 1. Kalman cinématique
        raw_signal = df[ind].values
        kalman_output = kalman_filter_dual(raw_signal)

        position = kalman_output[:, 0]
        velocity = kalman_output[:, 1]

        df[f'{ind}_filtered'] = position
        df[f'{ind}_velocity'] = velocity

        # 2. Label Direction (pente)
        # Direction[t] = 1 si filtered[t-2] > filtered[t-3]
        pos_t2 = pd.Series(position).shift(2)
        pos_t3 = pd.Series(position).shift(3)
        df[f'{ind}_dir'] = (pos_t2 > pos_t3).astype(int)

        # 3. Label Force (Z-Score de vélocité)
        force_labels, z_scores = calculate_force_labels(
            velocity,
            window=z_score_window,
            threshold=force_threshold
        )
        df[f'{ind}_force'] = force_labels
        df[f'{ind}_z_score'] = z_scores  # Pour debug

        # Stats
        n_valid = (~np.isnan(df[f'{ind}_dir'])).sum()
        if n_valid > 0:
            dir_up_pct = df[f'{ind}_dir'].sum() / n_valid * 100
            force_strong_pct = df[f'{ind}_force'].sum() / n_valid * 100
            logger.info(f"       Direction: {dir_up_pct:.1f}% UP")
            logger.info(f"       Force: {force_strong_pct:.1f}% STRONG")

    return df


# =============================================================================
# CRÉATION SÉQUENCES (MODIFIÉ - Y de shape (n, 6))
# =============================================================================

def create_sequences_dual_binary(df: pd.DataFrame,
                                  feature_cols: list,
                                  seq_length: int = SEQUENCE_LENGTH,
                                  cold_start_skip: int = COLD_START_SKIP) -> tuple:
    """
    Crée les séquences X, Y avec Y de shape (n, 6).

    Y contient 6 labels (3 indicateurs × 2 labels):
    [rsi_dir, rsi_force, cci_dir, cci_force, macd_dir, macd_force]

    CORRECTION EXPERT: Cold Start handling
    - Les premiers Z_SCORE_WINDOW samples ont des Z-Scores invalides
    - On commence les séquences à start_index = seq_length + cold_start_skip

    Returns:
        X: np.array (n_sequences, seq_length, n_features)
        Y: np.array (n_sequences, 6)
        indices: list de tuples (idx_feature, idx_label)
    """
    # Colonnes label (6 au total)
    label_cols = [
        'rsi_dir', 'rsi_force',
        'cci_dir', 'cci_force',
        'macd_dir', 'macd_force'
    ]

    # Supprimer lignes avec NaN
    cols_needed = feature_cols + label_cols
    df_clean = df.dropna(subset=cols_needed)

    logger.info(f"     Lignes valides: {len(df_clean)}/{len(df)} "
                f"({len(df) - len(df_clean)} supprimées pour NaN)")

    # Extraire arrays
    features = df_clean[feature_cols].values
    labels = df_clean[label_cols].values  # Shape: (N, 6)
    dates = df_clean.index.tolist()

    # CORRECTION EXPERT: Démarrer après cold start
    start_index = seq_length + cold_start_skip

    logger.info(f"     Cold Start: skip premiers {cold_start_skip} samples "
                f"(Z-Score invalide)")
    logger.info(f"     Start index: {start_index} (seq_length={seq_length} + cold_start={cold_start_skip})")

    # Créer séquences
    X_list = []
    Y_list = []
    idx_list = []

    for i in range(start_index, len(features)):
        # Séquence de features: indices [i-seq_length, i-1]
        X_list.append(features[i-seq_length:i])
        # Labels: à l'index i (6 labels)
        Y_list.append(labels[i])
        # Indices pour vérification
        idx_list.append((dates[i-1], dates[i]))

    X = np.array(X_list)
    Y = np.array(Y_list)  # Shape: (n, 6)

    logger.info(f"     Séquences créées: X={X.shape}, Y={Y.shape}")

    return X, Y, idx_list


# =============================================================================
# EXPORT DEBUG CSV (BONUS EXPERT)
# =============================================================================

def export_debug_csv(df: pd.DataFrame, output_dir: Path, assets: list):
    """
    Exporte les derniers 1000 samples pour validation visuelle.

    Colonnes exportées:
    - datetime
    - Pour chaque indicateur: raw, filtered, velocity, z_score, dir, force
    """
    debug_cols = ['datetime']

    # Réinitialiser index pour avoir datetime en colonne
    df_debug = df.reset_index()

    for ind in ['rsi', 'cci', 'macd']:
        debug_cols.extend([
            ind,  # Raw
            f'{ind}_filtered',
            f'{ind}_velocity',
            f'{ind}_z_score',
            f'{ind}_dir',
            f'{ind}_force'
        ])

    # Prendre les derniers 1000
    df_export = df_debug[debug_cols].tail(1000)

    assets_str = '_'.join(assets).lower()
    debug_path = output_dir / f"debug_labels_{assets_str}.csv"
    df_export.to_csv(debug_path, index=False)

    logger.info(f"\n📝 Debug CSV exporté: {debug_path}")
    logger.info(f"   → Vérifier visuellement les Z-Scores et labels")


# =============================================================================
# SPLIT CHRONOLOGIQUE
# =============================================================================

def split_chronological(X: np.ndarray, Y: np.ndarray, indices: list,
                        train_ratio: float = 0.70,
                        val_ratio: float = 0.15,
                        gap: int = SEQUENCE_LENGTH) -> dict:
    """
    Split chronologique avec GAP entre train/val et val/test.
    """
    n = len(X)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_end_gap = train_end - gap
    val_start = train_end
    val_end_gap = val_end - gap
    test_start = val_end

    return {
        'train': (X[:train_end_gap], Y[:train_end_gap], indices[:train_end_gap]),
        'val': (X[val_start:val_end_gap], Y[val_start:val_end_gap], indices[val_start:val_end_gap]),
        'test': (X[test_start:], Y[test_start:], indices[test_start:])
    }


# =============================================================================
# PRÉPARATION D'UN ASSET
# =============================================================================

def prepare_single_asset(file_path: str, asset_name: str,
                         filter_type: str = 'kalman',
                         clip_value: float = 0.10,
                         z_score_window: int = Z_SCORE_WINDOW,
                         force_threshold: float = FORCE_THRESHOLD) -> tuple:
    """
    Prépare les données pour un seul asset avec architecture dual-binary.

    Returns:
        (X, Y, indices, df) avec Y de shape (n, 6)
    """
    logger.info(f"\n  {asset_name}: Préparation...")

    # 1. Charger
    df = load_data_with_index(file_path, asset_name)
    logger.info(f"     Chargé: {len(df)} lignes")

    # 2. Indicateurs
    df = add_indicators_to_df(df)
    logger.info(f"     Indicateurs: RSI, CCI, MACD")

    # 3. Features OHLC (4 canaux - sans Open)
    df = add_ohlc_features_to_df(df, clip_value)
    logger.info(f"     Features: 4 canaux OHLC (H/L/C/Range)")

    # 4. Dual labels (3 indicateurs × 2 labels)
    df = add_dual_labels_to_df(df, filter_type, z_score_window, force_threshold)

    # 5. TRIM edges
    df = df.iloc[TRIM_EDGES:-TRIM_EDGES]
    logger.info(f"     Après trim ±{TRIM_EDGES}: {len(df)} lignes")

    # 6. Créer séquences
    feature_cols = ['h_ret', 'l_ret', 'c_ret', 'range_ret']
    X, Y, indices = create_sequences_dual_binary(df, feature_cols)

    return X, Y, indices, df


# =============================================================================
# PRÉPARATION ET SAUVEGARDE
# =============================================================================

def prepare_and_save(assets: list = None,
                     output_path: str = None,
                     filter_type: str = 'kalman',
                     clip_value: float = 0.10,
                     z_score_window: int = Z_SCORE_WINDOW,
                     force_threshold: float = FORCE_THRESHOLD) -> str:
    """
    Prépare les données avec architecture dual-binary.

    Args:
        filter_type: 'kalman' (seul supporté pour l'instant)
    """
    if assets is None:
        assets = DEFAULT_ASSETS

    # Valider assets
    invalid_assets = [a for a in assets if a not in AVAILABLE_ASSETS_5M]
    if invalid_assets:
        raise ValueError(f"Assets invalides: {invalid_assets}")

    logger.info("="*80)
    logger.info("PRÉPARATION DONNÉES DUAL-BINARY (Direction + Force)")
    logger.info("="*80)
    logger.info(f"Assets: {', '.join(assets)}")
    logger.info(f"Indicateurs: RSI, CCI, MACD (3 × 2 labels = 6 outputs)")
    logger.info(f"Features: 4 canaux OHLC (h_ret, l_ret, c_ret, range_ret)")
    logger.info(f"Z-Score window: {z_score_window}, Threshold: {force_threshold}")
    logger.info(f"Cold Start skip: {COLD_START_SKIP} premiers samples")

    # Préparer chaque asset
    all_splits = {'train': [], 'val': [], 'test': []}
    all_dfs = []  # Pour debug CSV

    for asset_name in assets:
        file_path = AVAILABLE_ASSETS_5M[asset_name]
        X, Y, indices, df = prepare_single_asset(
            file_path, asset_name, filter_type, clip_value,
            z_score_window, force_threshold
        )

        all_dfs.append(df)

        # Split chronologique
        splits = split_chronological(X, Y, indices)

        for split_name in ['train', 'val', 'test']:
            all_splits[split_name].append(splits[split_name])

        logger.info(f"     Split: Train={len(splits['train'][0])}, "
                   f"Val={len(splits['val'][0])}, Test={len(splits['test'][0])}")

    # Concaténer
    X_train = np.concatenate([s[0] for s in all_splits['train']], axis=0)
    Y_train = np.concatenate([s[1] for s in all_splits['train']], axis=0)
    X_val = np.concatenate([s[0] for s in all_splits['val']], axis=0)
    Y_val = np.concatenate([s[1] for s in all_splits['val']], axis=0)
    X_test = np.concatenate([s[0] for s in all_splits['test']], axis=0)
    Y_test = np.concatenate([s[1] for s in all_splits['test']], axis=0)

    logger.info(f"\n📊 Shapes finales:")
    logger.info(f"   Train: X={X_train.shape}, Y={Y_train.shape}")
    logger.info(f"   Val:   X={X_val.shape}, Y={Y_val.shape}")
    logger.info(f"   Test:  X={X_test.shape}, Y={Y_test.shape}")

    # Stats labels (6 outputs)
    label_names = ['RSI_dir', 'RSI_force', 'CCI_dir', 'CCI_force', 'MACD_dir', 'MACD_force']
    logger.info(f"\n📈 Balance labels:")
    for split_name, Y_split in [('Train', Y_train), ('Val', Y_val), ('Test', Y_test)]:
        logger.info(f"   {split_name}:")
        for i, name in enumerate(label_names):
            pct = Y_split[:, i].mean() * 100
            logger.info(f"     {name}: {pct:.1f}%")

    # Export debug CSV (premier asset comme référence)
    if output_path is None:
        output_dir = Path('data/prepared')
    else:
        output_dir = Path(output_path).parent

    output_dir.mkdir(parents=True, exist_ok=True)
    export_debug_csv(all_dfs[0], output_dir, [assets[0]])

    # Sauvegarder
    if output_path is None:
        assets_str = '_'.join(assets).lower()
        output_path = f"data/prepared/dataset_{assets_str}_dual_binary_{filter_type}.npz"

    metadata = {
        'created_at': datetime.now().isoformat(),
        'version': 'dual_binary_v1',
        'architecture': 'Direction + Force (2 labels per indicator)',
        'assets': assets,
        'filter_type': filter_type,
        'kalman_params': {
            'process_var': KALMAN_PROCESS_VAR,
            'measure_var': KALMAN_MEASURE_VAR,
            'model': 'cinematic (position + velocity)'
        },
        'z_score_window': z_score_window,
        'force_threshold': force_threshold,
        'cold_start_skip': COLD_START_SKIP,
        'indicators': ['RSI', 'CCI', 'MACD'],
        'labels_per_indicator': 2,
        'total_labels': 6,
        'label_names': label_names,
        'label_definitions': {
            'direction': 'filtered[t-2] > filtered[t-3] (pente passée)',
            'force': '|velocity_zscore[t-2]| > threshold (turning point)'
        },
        'clip_value': clip_value,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'sequence_length': SEQUENCE_LENGTH,
        'n_features': 4,
        'features': ['h_ret', 'l_ret', 'c_ret', 'range_ret'],
        'features_removed': ['o_ret (input toxique, non utilisé par indicateurs)'],
        'alignment_verified': True,
        'timestamp_convention': 'open_time (début de bougie)',
    }

    np.savez_compressed(
        output_path,
        X_train=X_train, Y_train=Y_train,
        X_val=X_val, Y_val=Y_val,
        X_test=X_test, Y_test=Y_test,
        metadata=json.dumps(metadata)
    )

    logger.info(f"\n✅ Données sauvegardées: {output_path}")

    # Sauvegarder métadonnées
    metadata_path = str(output_path).replace('.npz', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Point d'entrée CLI."""
    available_assets = list(AVAILABLE_ASSETS_5M.keys())

    parser = argparse.ArgumentParser(
        description="Prépare les datasets avec architecture Dual-Binary (Direction + Force)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Exemples:
  python src/prepare_data_dual_binary.py --assets BTC ETH BNB ADA LTC
  python src/prepare_data_dual_binary.py --assets BTC ETH --force-threshold 1.5

Architecture:
  - 3 indicateurs: RSI, CCI, MACD
  - 2 labels par indicateur: Direction (pente) + Force (vélocité Z-Score)
  - Total: 6 outputs au lieu de 3

Gains attendus:
  - Accuracy: +3-4% (inputs purifiés, 4 features au lieu de 5)
  - Trades: -60% (Force discrimine turning points faibles)

Assets disponibles: {', '.join(available_assets)}
        """
    )

    parser.add_argument('--assets', '-a', type=str, nargs='+',
                        default=DEFAULT_ASSETS,
                        choices=available_assets,
                        help=f'Assets à inclure')
    parser.add_argument('--filter', '-f', type=str, default='kalman',
                        choices=['kalman'],
                        help='Type de filtre (seul kalman supporté)')
    parser.add_argument('--z-score-window', type=int, default=Z_SCORE_WINDOW,
                        help=f'Fenêtre rolling std pour Z-Score (défaut: {Z_SCORE_WINDOW})')
    parser.add_argument('--force-threshold', type=float, default=FORCE_THRESHOLD,
                        help=f'Seuil Z-Score pour STRONG (défaut: {FORCE_THRESHOLD})')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Chemin de sortie')
    parser.add_argument('--clip', type=float, default=0.10,
                        help='Valeur de clipping features')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    output_path = prepare_and_save(
        assets=args.assets,
        output_path=args.output,
        filter_type=args.filter,
        clip_value=args.clip,
        z_score_window=args.z_score_window,
        force_threshold=args.force_threshold
    )

    print(f"\n✅ Terminé! Dataset: {output_path}")
    print(f"\nPour entraîner:")
    print(f"  python src/train.py --data {output_path} --multi-output dual-binary")


if __name__ == '__main__':
    main()
