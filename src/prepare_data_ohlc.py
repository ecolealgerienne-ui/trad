"""
Script de préparation des données OHLC normalisées.

Architecture: OHLC (5 canaux) → CNN-LSTM → Direction indicateur filtré

Canaux OHLC normalisés (Option A+):
- O_ret = (O[t] - C[t-1]) / C[t-1]
- H_ret = (H[t] - C[t-1]) / C[t-1]
- L_ret = (L[t] - C[t-1]) / C[t-1]
- C_ret = (C[t] - C[t-1]) / C[t-1]
- Range_ret = (H[t] - L[t]) / C[t-1]

Target: Direction indicateur filtré (FL_RSI, FL_CCI, FL_MACD)
Filtre: Octave (Butterworth + filtfilt)

Usage:
    python src/prepare_data_ohlc.py --target rsi --assets BTC ETH BNB ADA LTC
    python src/prepare_data_ohlc.py --target macd40 --assets BTC ETH
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Import modules locaux
from constants import (
    AVAILABLE_ASSETS_5M, DEFAULT_ASSETS,
    TRIM_EDGES,
    PREPARED_DATA_DIR,
    SEQUENCE_LENGTH,
)

# Périodes STANDARD des indicateurs (valeurs classiques)
RSI_PERIOD = 14      # Wilder's original
CCI_PERIOD = 20      # Lambert's original
MACD_FAST = 12       # Standard
MACD_SLOW = 26       # Standard
MACD_SIGNAL = 9      # Standard
from data_utils import load_crypto_data, trim_edges, split_sequences_chronological
from indicators import (
    calculate_rsi, calculate_cci, calculate_macd,
    normalize_cci, normalize_macd_histogram,
    create_sequences
)


# =============================================================================
# CONFIGURATION
# =============================================================================

OCTAVE_STEP = 0.20  # Paramètre du filtre Octave (0.20 recommandé)

# Paramètres MACD (différentes vitesses)
MACD_CONFIGS = {
    'macd': {'fast': 12, 'slow': 26, 'signal': 9},      # Standard (= macd26)
    'macd9': {'fast': 4, 'slow': 9, 'signal': 4},       # Très rapide
    'macd13': {'fast': 6, 'slow': 13, 'signal': 4},     # Rapide
    'macd26': {'fast': 12, 'slow': 26, 'signal': 9},    # Standard
    'macd40': {'fast': 18, 'slow': 40, 'signal': 13},   # Lent (recommandé)
}


# =============================================================================
# FILTRE OCTAVE
# =============================================================================

def octave_filter(data: np.ndarray, step: float = OCTAVE_STEP) -> np.ndarray:
    """
    Applique le filtre Octave (Butterworth ordre 3 + filtfilt).

    Args:
        data: Signal à filtrer
        step: Paramètre du filtre (défaut: 0.20)

    Returns:
        Signal filtré
    """
    B, A = signal.butter(3, step, output='ba')
    filtered = signal.filtfilt(B, A, data)
    return filtered


# =============================================================================
# NORMALISATION OHLC (Option A+)
# =============================================================================

def normalize_ohlc_returns(df: pd.DataFrame, clip_value: float = 0.10) -> np.ndarray:
    """
    Normalise OHLC en returns relatifs au Close précédent.

    Formules:
        O_ret = (O[t] - C[t-1]) / C[t-1]
        H_ret = (H[t] - C[t-1]) / C[t-1]
        L_ret = (L[t] - C[t-1]) / C[t-1]
        C_ret = (C[t] - C[t-1]) / C[t-1]
        Range_ret = (H[t] - L[t]) / C[t-1]

    Args:
        df: DataFrame avec colonnes open, high, low, close
        clip_value: Valeur max pour clipper les outliers (défaut: 0.10 = ±10%)

    Returns:
        np.array shape (n, 5) avec les 5 canaux normalisés et clippés
    """
    prev_close = df['close'].shift(1)

    # Éviter division par zéro
    prev_close = prev_close.replace(0, np.nan)

    # Calcul des 5 canaux
    o_ret = (df['open'] - prev_close) / prev_close
    h_ret = (df['high'] - prev_close) / prev_close
    l_ret = (df['low'] - prev_close) / prev_close
    c_ret = (df['close'] - prev_close) / prev_close
    range_ret = (df['high'] - df['low']) / prev_close

    # Combiner en array (n, 5)
    features = np.column_stack([
        o_ret.values,
        h_ret.values,
        l_ret.values,
        c_ret.values,
        range_ret.values
    ])

    # Remplacer NaN par 0 (première ligne)
    features = np.nan_to_num(features, nan=0.0)

    # Clipper les outliers pour stabiliser le training
    # Les valeurs extrêmes (>10%) causent des gradients instables
    n_clipped = np.sum((features < -clip_value) | (features > clip_value))
    if n_clipped > 0:
        pct_clipped = n_clipped / features.size * 100
        logger.info(f"     Clipping: {n_clipped:,} valeurs ({pct_clipped:.2f}%) clippées à ±{clip_value*100:.0f}%")

    features = np.clip(features, -clip_value, clip_value)

    return features


# =============================================================================
# CALCUL DES INDICATEURS POUR LABELS
# =============================================================================

def calculate_indicator_for_labels(df: pd.DataFrame, target: str) -> np.ndarray:
    """
    Calcule l'indicateur brut pour générer les labels.

    Args:
        df: DataFrame avec OHLCV
        target: 'rsi', 'cci', 'macd', 'macd13', 'macd26', 'macd40', 'close'

    Returns:
        np.array avec les valeurs de l'indicateur
    """
    target = target.lower()

    if target == 'rsi':
        indicator = calculate_rsi(df['close'], period=RSI_PERIOD)

    elif target == 'cci':
        indicator = calculate_cci(df['high'], df['low'], df['close'], period=CCI_PERIOD)

    elif target in MACD_CONFIGS:
        config = MACD_CONFIGS[target]
        macd_data = calculate_macd(
            df['close'],
            fast_period=config['fast'],
            slow_period=config['slow'],
            signal_period=config['signal']
        )
        indicator = macd_data['histogram']

    elif target == 'close':
        indicator = df['close'].values

    else:
        raise ValueError(f"Target inconnu: {target}. "
                        f"Choix: rsi, cci, macd, macd13, macd26, macd40, close")

    # Convertir en array et remplir NaN
    indicator = pd.Series(indicator).ffill().bfill().values

    return indicator


def generate_octave_labels(indicator: np.ndarray, step: float = OCTAVE_STEP,
                           delta: int = 0) -> np.ndarray:
    """
    Génère les labels depuis indicateur filtré.

    Formule simplifiée:
        Label[t] = 1 si filtered[t] > filtered[t-1]
        Label[t] = 0 sinon

    Args:
        indicator: Valeurs de l'indicateur
        step: Paramètre du filtre Octave
        delta: Non utilisé (gardé pour compatibilité)

    Returns:
        labels: np.array de 0/1
    """
    filtered = octave_filter(indicator, step)

    labels = np.zeros(len(filtered), dtype=int)

    # Formule simple: filtered[t] > filtered[t-1]
    for t in range(1, len(filtered)):
        if filtered[t] > filtered[t-1]:
            labels[t] = 1

    return labels


# =============================================================================
# PREPARATION D'UN ASSET
# =============================================================================

def prepare_single_asset(df: pd.DataFrame, target: str,
                         delta: int = 0,
                         clip_value: float = 0.10,
                         asset_name: str = "Asset") -> tuple:
    """
    Prépare OHLC normalisé + labels pour UN SEUL asset.

    Args:
        df: DataFrame avec OHLC pour un seul asset
        target: Indicateur cible ('rsi', 'cci', 'macd40', etc.)
        delta: Décalage pour labels
        asset_name: Nom pour les logs

    Returns:
        (X, Y) où X shape=(n_sequences, 12, 5), Y shape=(n_sequences, 1)
    """
    logger.info(f"  {asset_name}: Préparation ({len(df):,} bougies)...")

    # 1. Normaliser OHLC (5 canaux) avec clipping
    features = normalize_ohlc_returns(df, clip_value=clip_value)
    logger.info(f"     OHLC normalisé: shape={features.shape}")
    logger.info(f"     O_ret: [{features[:,0].min():.4f}, {features[:,0].max():.4f}]")
    logger.info(f"     C_ret: [{features[:,3].min():.4f}, {features[:,3].max():.4f}]")
    logger.info(f"     Range: [{features[:,4].min():.4f}, {features[:,4].max():.4f}]")

    # 2. Calculer indicateur pour labels
    indicator = calculate_indicator_for_labels(df, target)
    logger.info(f"     Indicateur {target.upper()}: min={indicator.min():.2f}, max={indicator.max():.2f}")

    # 3. Générer labels avec filtre Octave
    labels = generate_octave_labels(indicator, OCTAVE_STEP, delta)
    buy_pct = labels.sum() / len(labels) * 100
    logger.info(f"     Labels FL_{target.upper()}(step={OCTAVE_STEP}, delta={delta}): {buy_pct:.1f}% UP")

    # 4. Reshape labels pour create_sequences
    labels_2d = labels.reshape(-1, 1)

    # 5. Créer séquences de 12 timesteps
    X, Y = create_sequences(features, labels_2d)

    logger.info(f"     Séquences: X={X.shape}, Y={Y.shape}")

    return X, Y


# =============================================================================
# PREPARATION ET SAUVEGARDE
# =============================================================================

def prepare_and_save(target: str,
                     assets: list = None,
                     delta: int = 0,
                     output_path: str = None,
                     octave_step: float = OCTAVE_STEP,
                     clip_value: float = 0.10) -> str:
    """
    Prépare les données OHLC normalisées et les sauvegarde.

    Args:
        target: Indicateur cible ('rsi', 'cci', 'macd40', etc.)
        assets: Liste des assets (défaut: DEFAULT_ASSETS)
        delta: Décalage pour labels
        output_path: Chemin de sortie (défaut: auto-généré)
        octave_step: Paramètre du filtre Octave

    Returns:
        Chemin du fichier sauvegardé
    """
    global OCTAVE_STEP
    OCTAVE_STEP = octave_step

    if assets is None:
        assets = DEFAULT_ASSETS

    # Valider les assets
    invalid_assets = [a for a in assets if a not in AVAILABLE_ASSETS_5M]
    if invalid_assets:
        raise ValueError(f"Assets invalides: {invalid_assets}. "
                        f"Disponibles: {list(AVAILABLE_ASSETS_5M.keys())}")

    logger.info("="*80)
    logger.info("PRÉPARATION DES DONNÉES OHLC NORMALISÉES")
    logger.info("="*80)
    logger.info(f"Input: OHLC (5 canaux: O_ret, H_ret, L_ret, C_ret, Range_ret)")
    logger.info(f"Clipping: ±{clip_value*100:.0f}%")
    logger.info(f"Target: FL_{target.upper()} (Octave step={octave_step}, delta={delta})")
    logger.info(f"Assets: {', '.join(assets)}")

    # =========================================================================
    # 1. Charger données pour chaque asset
    # =========================================================================
    logger.info(f"\n1. Chargement données 5min...")

    asset_data = {}
    total_candles = 0
    for asset_name in assets:
        file_path = AVAILABLE_ASSETS_5M[asset_name]
        df = load_crypto_data(file_path, asset_name=asset_name)
        df_trimmed = trim_edges(df, trim_start=TRIM_EDGES, trim_end=TRIM_EDGES)
        asset_data[asset_name] = df_trimmed
        total_candles += len(df_trimmed)
        logger.info(f"   {asset_name}: {len(df_trimmed):,} bougies")

    # =========================================================================
    # 2. Préparer OHLC + labels pour chaque asset
    # =========================================================================
    logger.info(f"\n2. Normalisation OHLC et génération labels...")

    prepared_assets = {}
    for asset_name, df in asset_data.items():
        X, Y = prepare_single_asset(df, target, delta, clip_value, asset_name)
        prepared_assets[asset_name] = (X, Y)

    # =========================================================================
    # 3. Split chronologique avec GAP pour chaque asset
    # =========================================================================
    logger.info(f"\n3. Split chronologique avec GAP...")

    split_data = {}
    for asset_name, (X, Y) in prepared_assets.items():
        (X_train_a, Y_train_a), (X_val_a, Y_val_a), (X_test_a, Y_test_a) = \
            split_sequences_chronological(X, Y)
        split_data[asset_name] = {
            'train': (X_train_a, Y_train_a),
            'val': (X_val_a, Y_val_a),
            'test': (X_test_a, Y_test_a)
        }
        logger.info(f"   {asset_name}: Train={len(X_train_a)}, Val={len(X_val_a)}, Test={len(X_test_a)}")

    # Concaténer tous les assets
    X_train = np.concatenate([split_data[a]['train'][0] for a in assets], axis=0)
    Y_train = np.concatenate([split_data[a]['train'][1] for a in assets], axis=0)
    X_val = np.concatenate([split_data[a]['val'][0] for a in assets], axis=0)
    Y_val = np.concatenate([split_data[a]['val'][1] for a in assets], axis=0)
    X_test = np.concatenate([split_data[a]['test'][0] for a in assets], axis=0)
    Y_test = np.concatenate([split_data[a]['test'][1] for a in assets], axis=0)

    # =========================================================================
    # 4. Afficher stats finales
    # =========================================================================
    logger.info(f"\n4. Shapes des datasets:")
    logger.info(f"   Train: X={X_train.shape}, Y={Y_train.shape}")
    logger.info(f"   Val:   X={X_val.shape}, Y={Y_val.shape}")
    logger.info(f"   Test:  X={X_test.shape}, Y={Y_test.shape}")

    # Stats sur les features
    logger.info(f"\n   Stats features (Train):")
    channel_names = ['O_ret', 'H_ret', 'L_ret', 'C_ret', 'Range']
    for i, name in enumerate(channel_names):
        vals = X_train[:, :, i].flatten()
        logger.info(f"     {name}: mean={vals.mean():.6f}, std={vals.std():.6f}, "
                   f"min={vals.min():.4f}, max={vals.max():.4f}")

    # Stats sur les labels
    train_up_pct = Y_train.sum() / len(Y_train) * 100
    val_up_pct = Y_val.sum() / len(Y_val) * 100
    test_up_pct = Y_test.sum() / len(Y_test) * 100
    logger.info(f"\n   Balance labels:")
    logger.info(f"     Train: {train_up_pct:.1f}% UP")
    logger.info(f"     Val:   {val_up_pct:.1f}% UP")
    logger.info(f"     Test:  {test_up_pct:.1f}% UP")

    # =========================================================================
    # 5. Sauvegarder
    # =========================================================================
    if output_path is None:
        assets_str = '_'.join(assets).lower()
        output_path = f"data/prepared/dataset_{assets_str}_ohlc_{target}_octave{int(octave_step*100)}_delta{delta}.npz"

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Métadonnées
    metadata = {
        'created_at': datetime.now().isoformat(),
        'assets': assets,
        'n_assets': len(assets),
        'timeframe': '5min',
        'input_type': 'ohlc_normalized',
        'channels': ['O_ret', 'H_ret', 'L_ret', 'C_ret', 'Range_ret'],
        'normalization': 'returns_vs_prev_close',
        'clip_value': clip_value,
        'target': target,
        'filter_type': 'octave',
        'octave_step': octave_step,
        'delta': delta,
        'total_candles': total_candles,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'sequence_length': SEQUENCE_LENGTH,
        'n_features': 5,
        'split_strategy': 'chronological_with_gap',
        'gap_size': SEQUENCE_LENGTH
    }

    np.savez_compressed(
        output_path,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        X_test=X_test,
        Y_test=Y_test,
        metadata=json.dumps(metadata)
    )

    logger.info(f"\n5. Données sauvegardées: {output_path}")
    logger.info(f"   Taille: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")

    # Sauvegarder métadonnées
    metadata_path = str(output_path).replace('.npz', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"   Métadonnées: {metadata_path}")

    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Point d'entrée CLI."""
    available_assets = list(AVAILABLE_ASSETS_5M.keys())
    available_targets = ['rsi', 'cci', 'macd', 'macd13', 'macd26', 'macd40', 'close']

    parser = argparse.ArgumentParser(
        description="Prépare les datasets OHLC normalisés (5 canaux → indicateur filtré)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Exemples:
  # OHLC → FL_RSI
  python src/prepare_data_ohlc.py --target rsi --assets BTC ETH BNB ADA LTC

  # OHLC → FL_MACD40 (recommandé pour trading)
  python src/prepare_data_ohlc.py --target macd40 --assets BTC ETH BNB ADA LTC

  # Avec delta (décalage labels)
  python src/prepare_data_ohlc.py --target rsi --delta 1

  # Avec paramètre Octave différent
  python src/prepare_data_ohlc.py --target rsi --octave-step 0.15

Canaux OHLC normalisés:
  O_ret     = (Open[t] - Close[t-1]) / Close[t-1]
  H_ret     = (High[t] - Close[t-1]) / Close[t-1]
  L_ret     = (Low[t] - Close[t-1]) / Close[t-1]
  C_ret     = (Close[t] - Close[t-1]) / Close[t-1]
  Range_ret = (High[t] - Low[t]) / Close[t-1]

Targets disponibles: {', '.join(available_targets)}
Assets disponibles: {', '.join(available_assets)}
        """
    )

    parser.add_argument('--target', '-t', type=str, required=True,
                        choices=available_targets,
                        help=f'Indicateur cible ({", ".join(available_targets)})')
    parser.add_argument('--assets', '-a', type=str, nargs='+',
                        default=DEFAULT_ASSETS,
                        choices=available_assets,
                        help=f'Assets à inclure (défaut: {DEFAULT_ASSETS})')
    parser.add_argument('--delta', '-d', type=int, default=0,
                        help='Décalage pour labels (défaut: 0)')
    parser.add_argument('--octave-step', type=float, default=OCTAVE_STEP,
                        help=f'Paramètre du filtre Octave (défaut: {OCTAVE_STEP})')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Chemin de sortie (défaut: auto-généré)')
    parser.add_argument('--clip', type=float, default=0.10,
                        help='Valeur de clipping des outliers (défaut: 0.10 = ±10%%)')

    args = parser.parse_args()

    # Configurer logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    # Préparer et sauvegarder
    output_path = prepare_and_save(
        target=args.target,
        assets=args.assets,
        delta=args.delta,
        output_path=args.output,
        octave_step=args.octave_step,
        clip_value=args.clip
    )

    print(f"\n Terminé! Dataset prêt: {output_path}")
    print(f"\nPour entraîner:")
    print(f"  python src/train.py --data {output_path}")


if __name__ == '__main__':
    main()
