"""
Script de préparation des données avec Architecture Purified + Dual-Binary.

PRINCIPE CLÉ: Features purifiées par indicateur + Labels dual-binary
====================================================================

Pour chaque indicateur (RSI, CCI, MACD), on génère UN DATASET SÉPARÉ:

**RSI**:
  - Features: 5 Close-based (C_ret, C_ma_5, C_ma_20, C_mom_3, C_mom_10)
  - Labels: 2 (rsi_dir, rsi_force)
  - Gain attendu: +3-4% accuracy (retire 60% de bruit)

**MACD**:
  - Features: 5 Close-based (idem RSI)
  - Labels: 2 (macd_dir, macd_force)
  - Gain attendu: +3-4% accuracy

**CCI**:
  - Features: 5 Volatility-aware (C_ret, H_ret, L_ret, Range_ret, ATR_norm)
  - Labels: 2 (cci_dir, cci_force)
  - Gain attendu: +1-2% accuracy

Pipeline:
1. Charger données brutes
2. Calculer TOUS les indicateurs (RSI, CCI, MACD)
3. Calculer features Close-based ET Volatility-aware
4. Pour CHAQUE indicateur:
   - Appliquer Kalman dual (Position + Vélocité)
   - Calculer labels Direction + Force
   - Créer séquences avec features appropriées
   - Sauvegarder dataset séparé

Usage:
    python src/prepare_data_purified_dual_binary.py --assets BTC ETH BNB ADA LTC

Génère 3 fichiers:
    - dataset_..._rsi_dual_binary_kalman.npz
    - dataset_..._macd_dual_binary_kalman.npz
    - dataset_..._cci_dual_binary_kalman.npz
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
Z_SCORE_WINDOW = 100
FORCE_THRESHOLD = 1.0
COLD_START_SKIP = 100

# Configuration ATR
ATR_PERIOD = 14


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

def load_data_with_index(file_path: str, asset_name: str = "Asset") -> pd.DataFrame:
    """Charge les données CSV avec DatetimeIndex."""
    df = pd.read_csv(file_path)

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

    required = ['open', 'high', 'low', 'close']
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
                                   z_score_window: int = Z_SCORE_WINDOW,
                                   force_threshold: float = FORCE_THRESHOLD) -> pd.DataFrame:
    """
    Calcule les labels dual-binary pour UN indicateur.

    Args:
        indicator: 'rsi', 'cci', ou 'macd'

    Returns:
        df avec colonnes ajoutées:
        - {indicator}_filtered
        - {indicator}_velocity
        - {indicator}_dir
        - {indicator}_force
        - {indicator}_z_score
    """
    df = df.copy()

    logger.info(f"     Processing {indicator.upper()}...")

    # 1. Kalman cinématique
    raw_signal = df[indicator].values
    kalman_output = kalman_filter_dual(raw_signal)

    position = kalman_output[:, 0]
    velocity = kalman_output[:, 1]

    df[f'{indicator}_filtered'] = position
    df[f'{indicator}_velocity'] = velocity

    # 2. Label Direction
    pos_series = pd.Series(position, index=df.index)
    pos_t2 = pos_series.shift(2)
    pos_t3 = pos_series.shift(3)
    df[f'{indicator}_dir'] = (pos_t2 > pos_t3).astype(int)

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
                                    seq_length: int = SEQUENCE_LENGTH,
                                    cold_start_skip: int = COLD_START_SKIP) -> tuple:
    """
    Crée les séquences pour UN indicateur avec features spécifiques.

    Args:
        indicator: 'rsi', 'cci', ou 'macd'
        feature_cols: Liste des features à utiliser (5 colonnes)

    Returns:
        X: (n, seq_length, 5)
        Y: (n, 2) - [direction, force]
        indices: list de (idx_feature, idx_label)
    """
    # Colonnes label (2 pour cet indicateur)
    label_cols = [f'{indicator}_dir', f'{indicator}_force']

    # Supprimer lignes avec NaN
    cols_needed = feature_cols + label_cols
    df_clean = df.dropna(subset=cols_needed)

    logger.info(f"     Lignes valides: {len(df_clean)}/{len(df)} "
                f"({len(df) - len(df_clean)} supprimées pour NaN)")

    # Extraire arrays
    features = df_clean[feature_cols].values
    labels = df_clean[label_cols].values  # Shape: (N, 2)
    dates = df_clean.index.tolist()

    # Cold start
    start_index = seq_length + cold_start_skip

    logger.info(f"     Cold Start: skip premiers {cold_start_skip} samples")
    logger.info(f"     Start index: {start_index}")

    # Créer séquences
    X_list = []
    Y_list = []
    idx_list = []

    for i in range(start_index, len(features)):
        X_list.append(features[i-seq_length:i])
        Y_list.append(labels[i])
        idx_list.append((dates[i-1], dates[i]))

    X = np.array(X_list)
    Y = np.array(Y_list)

    logger.info(f"     Séquences créées: X={X.shape}, Y={Y.shape}")

    return X, Y, idx_list


# =============================================================================
# SPLIT CHRONOLOGIQUE
# =============================================================================

def split_chronological(X: np.ndarray, Y: np.ndarray, indices: list,
                        train_ratio: float = 0.70,
                        val_ratio: float = 0.15,
                        gap: int = SEQUENCE_LENGTH) -> dict:
    """Split chronologique avec GAP."""
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
# PRÉPARATION POUR UN INDICATEUR
# =============================================================================

def prepare_indicator_dataset(df: pd.DataFrame, asset_name: str, indicator: str,
                              feature_cols: list, clip_value: float = 0.10) -> tuple:
    """
    Prépare le dataset pour UN indicateur avec features purifiées.

    Args:
        df: DataFrame avec OHLC + indicateurs calculés
        indicator: 'rsi', 'cci', ou 'macd'
        feature_cols: Colonnes features à utiliser

    Returns:
        (X, Y, indices) pour cet indicateur
    """
    logger.info(f"\n  {asset_name} - {indicator.upper()}: Préparation...")

    # Calculer labels dual-binary pour cet indicateur
    df = add_dual_labels_for_indicator(df, indicator)

    # Créer séquences
    X, Y, indices = create_sequences_for_indicator(df, indicator, feature_cols)

    return X, Y, indices


# =============================================================================
# PRÉPARATION ET SAUVEGARDE MULTI-INDICATEURS
# =============================================================================

def prepare_and_save_all(assets: list = None,
                         output_dir: str = None,
                         clip_value: float = 0.10) -> dict:
    """
    Prépare les 3 datasets (RSI, MACD, CCI) en une seule exécution.

    Returns:
        dict avec les chemins des 3 fichiers .npz générés
    """
    if assets is None:
        assets = DEFAULT_ASSETS

    invalid_assets = [a for a in assets if a not in AVAILABLE_ASSETS_5M]
    if invalid_assets:
        raise ValueError(f"Assets invalides: {invalid_assets}")

    logger.info("="*80)
    logger.info("PRÉPARATION MULTI-DATASETS (Pure Signal + Dual-Binary)")
    logger.info("="*80)
    logger.info(f"Assets: {', '.join(assets)}")
    logger.info(f"Génération de 3 datasets séparés:")
    logger.info(f"  1. RSI  - Features: c_ret (1)")
    logger.info(f"  2. MACD - Features: c_ret (1)")
    logger.info(f"  3. CCI  - Features: h_ret, l_ret, c_ret (3)")
    logger.info(f"Labels: Direction + Force (2 par indicateur)")
    logger.info(f"Architecture: Pure Signal (zéro bruit toxique)")

    # Stockage par indicateur
    datasets = {
        'rsi': {'train': [], 'val': [], 'test': []},
        'macd': {'train': [], 'val': [], 'test': []},
        'cci': {'train': [], 'val': [], 'test': []}
    }

    # Features par indicateur (Architecture "Pure Signal")
    features_rsi = ['c_ret']  # 1 feature: Close uniquement
    features_macd = ['c_ret']  # 1 feature: Close uniquement
    features_cci = ['h_ret', 'l_ret', 'c_ret']  # 3 features: High, Low, Close

    # Préparer chaque asset
    for asset_name in assets:
        logger.info(f"\n{'='*80}")
        logger.info(f"ASSET: {asset_name}")
        logger.info('='*80)

        file_path = AVAILABLE_ASSETS_5M[asset_name]

        # 1. Charger
        df = load_data_with_index(file_path, asset_name)
        logger.info(f"     Chargé: {len(df)} lignes")

        # 2. Indicateurs
        df = add_indicators_to_df(df)
        logger.info(f"     Indicateurs: RSI, CCI, MACD")

        # 3. Features Pure Signal (h_ret, l_ret, c_ret)
        df = add_pure_signal_features(df, clip_value)
        logger.info(f"     Features Pure Signal: 3 canaux (h_ret, l_ret, c_ret)")

        # 4. TRIM edges
        df = df.iloc[TRIM_EDGES:-TRIM_EDGES]
        logger.info(f"     Après trim ±{TRIM_EDGES}: {len(df)} lignes")

        # 5. Préparer pour chaque indicateur
        for indicator, feature_cols in [
            ('rsi', features_rsi),      # 1 feature: c_ret
            ('macd', features_macd),    # 1 feature: c_ret
            ('cci', features_cci)       # 3 features: h_ret, l_ret, c_ret
        ]:
            X, Y, indices = prepare_indicator_dataset(
                df, asset_name, indicator, feature_cols, clip_value
            )

            # Split chronologique
            splits = split_chronological(X, Y, indices)

            for split_name in ['train', 'val', 'test']:
                datasets[indicator][split_name].append(splits[split_name])

            logger.info(f"     Split: Train={len(splits['train'][0])}, "
                       f"Val={len(splits['val'][0])}, Test={len(splits['test'][0])}")

    # Concaténer et sauvegarder chaque indicateur
    output_paths = {}

    for indicator in ['rsi', 'macd', 'cci']:
        logger.info(f"\n{'='*80}")
        logger.info(f"SAUVEGARDE DATASET: {indicator.upper()}")
        logger.info('='*80)

        # Concaténer tous les assets
        X_train = np.concatenate([s[0] for s in datasets[indicator]['train']], axis=0)
        Y_train = np.concatenate([s[1] for s in datasets[indicator]['train']], axis=0)
        X_val = np.concatenate([s[0] for s in datasets[indicator]['val']], axis=0)
        Y_val = np.concatenate([s[1] for s in datasets[indicator]['val']], axis=0)
        X_test = np.concatenate([s[0] for s in datasets[indicator]['test']], axis=0)
        Y_test = np.concatenate([s[1] for s in datasets[indicator]['test']], axis=0)

        logger.info(f"   Train: X={X_train.shape}, Y={Y_train.shape}")
        logger.info(f"   Val:   X={X_val.shape}, Y={Y_val.shape}")
        logger.info(f"   Test:  X={X_test.shape}, Y={Y_test.shape}")

        # Stats labels
        logger.info(f"\n   Balance labels:")
        for split_name, Y_split in [('Train', Y_train), ('Val', Y_val), ('Test', Y_test)]:
            dir_pct = Y_split[:, 0].mean() * 100
            force_pct = Y_split[:, 1].mean() * 100
            logger.info(f"     {split_name}: Direction {dir_pct:.1f}% UP, Force {force_pct:.1f}% STRONG")

        # Sauvegarder
        if output_dir is None:
            output_dir = Path('data/prepared')
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        assets_str = '_'.join(assets).lower()
        output_path = output_dir / f"dataset_{assets_str}_{indicator}_dual_binary_kalman.npz"

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

        metadata = {
            'created_at': datetime.now().isoformat(),
            'version': 'pure_signal_dual_binary_v1',
            'architecture': 'Pure Signal + Dual-Binary Labels',
            'indicator': indicator.upper(),
            'assets': assets,
            'filter_type': 'kalman',
            'kalman_params': {
                'process_var': KALMAN_PROCESS_VAR,
                'measure_var': KALMAN_MEASURE_VAR,
                'model': 'cinematic (position + velocity)'
            },
            'z_score_window': Z_SCORE_WINDOW,
            'force_threshold': FORCE_THRESHOLD,
            'cold_start_skip': COLD_START_SKIP,
            'labels': 2,
            'label_names': [f'{indicator}_dir', f'{indicator}_force'],
            'label_definitions': {
                'direction': 'filtered[t-2] > filtered[t-3]',
                'force': '|velocity_zscore[t-2]| > threshold'
            },
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
        }

        np.savez_compressed(
            output_path,
            X_train=X_train, Y_train=Y_train,
            X_val=X_val, Y_val=Y_val,
            X_test=X_test, Y_test=Y_test,
            metadata=json.dumps(metadata)
        )

        logger.info(f"\n   ✅ Sauvegardé: {output_path}")

        # Metadata JSON
        metadata_path = str(output_path).replace('.npz', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        output_paths[indicator] = str(output_path)

    return output_paths


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Point d'entrée CLI."""
    available_assets = list(AVAILABLE_ASSETS_5M.keys())

    parser = argparse.ArgumentParser(
        description="Prépare les datasets avec Pure Signal + Dual-Binary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Architecture "Pure Signal" (Expert-Validated):
==============================================

Génère 3 fichiers .npz (un par indicateur):

1. RSI  - Features: c_ret (1) + Labels: Direction + Force (2)
   └─ Shape: X=(n, 25, 1), Y=(n, 2)
   └─ Justification: RSI utilise Close uniquement. High/Low = bruit toxique.

2. MACD - Features: c_ret (1) + Labels: Direction + Force (2)
   └─ Shape: X=(n, 25, 1), Y=(n, 2)
   └─ Justification: MACD utilise Close uniquement. High/Low = bruit toxique.

3. CCI  - Features: h_ret, l_ret, c_ret (3) + Labels: Direction + Force (2)
   └─ Shape: X=(n, 25, 3), Y=(n, 2)
   └─ Justification: CCI utilise (H+L+C)/3. High/Low justifiés.

Features Bannies:
- o_ret (Open): Bruit de microstructure
- range_ret: Redondant pour CCI, bruit pour RSI/MACD

Gains Attendus:
- RSI/MACD: +3-4% accuracy (retire 60% de bruit)
- CCI: +1-2% accuracy
- Convergence: Plus rapide et plus propre
- Stabilité: Fonctionne sur tous les prix (10k$ ou 100k$)

Exemple:
  python src/prepare_data_purified_dual_binary.py --assets BTC ETH BNB ADA LTC

Assets disponibles: {', '.join(available_assets)}
        """
    )

    parser.add_argument('--assets', '-a', type=str, nargs='+',
                        default=DEFAULT_ASSETS,
                        choices=available_assets,
                        help='Assets à inclure')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Répertoire de sortie')
    parser.add_argument('--clip', type=float, default=0.10,
                        help='Valeur de clipping')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    output_paths = prepare_and_save_all(
        assets=args.assets,
        output_dir=args.output_dir,
        clip_value=args.clip
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
