"""
Script de préparation des données avec filtre Octave.

Architecture: 1 indicateur → 1 sortie (Octave CLOSE)

Indicateurs disponibles: RSI, CCI, MACD (un seul à la fois)
Filtre: Octave (Butterworth + filtfilt)
Target: Direction du CLOSE filtré (t-1 > t-2)

Usage:
    python src/prepare_data_octave.py --indicator rsi --assets BTC ETH BNB ADA LTC
    python src/prepare_data_octave.py --indicator cci --assets BTC
    python src/prepare_data_octave.py --indicator macd --assets BTC ETH
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
    RSI_PERIOD, CCI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
)
from data_utils import load_crypto_data, trim_edges, split_sequences_chronological
from indicators import (
    calculate_rsi, calculate_cci, calculate_macd,
    normalize_cci, normalize_macd_histogram,
    create_sequences
)


# =============================================================================
# CONFIGURATION OCTAVE
# =============================================================================

OCTAVE_STEP = 0.20  # Paramètre du filtre Octave (0.20 recommandé)


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
# CALCUL DES FEATURES ET LABELS
# =============================================================================

def calculate_single_indicator(df: pd.DataFrame, indicator: str) -> np.ndarray:
    """
    Calcule un seul indicateur normalisé 0-100.

    Args:
        df: DataFrame avec OHLCV
        indicator: 'rsi', 'cci', ou 'macd'

    Returns:
        feature: np.array normalisé
    """
    if indicator == 'rsi':
        feature = calculate_rsi(df['close'], period=RSI_PERIOD)
        # RSI déjà entre 0-100

    elif indicator == 'cci':
        cci_raw = calculate_cci(df['high'], df['low'], df['close'], period=CCI_PERIOD)
        feature = normalize_cci(cci_raw)

    elif indicator == 'macd':
        macd_data = calculate_macd(
            df['close'],
            fast_period=MACD_FAST,
            slow_period=MACD_SLOW,
            signal_period=MACD_SIGNAL
        )
        feature = normalize_macd_histogram(macd_data['histogram'])

    else:
        raise ValueError(f"Indicateur inconnu: {indicator}. Choix: rsi, cci, macd")

    # Remplir NaN
    feature = pd.Series(feature).ffill().fillna(50.0).values

    return feature


def generate_octave_close_labels(df: pd.DataFrame, step: float = OCTAVE_STEP) -> np.ndarray:
    """
    Génère les labels depuis Octave(CLOSE).

    Label[t] = 1 si filtered[t-1] > filtered[t-2]
    Label[t] = 0 sinon

    Args:
        df: DataFrame avec colonne 'close'
        step: Paramètre du filtre Octave

    Returns:
        labels: np.array de 0/1
    """
    close = df['close'].values
    filtered = octave_filter(close, step)

    labels = np.zeros(len(filtered), dtype=int)
    for t in range(2, len(filtered)):
        if filtered[t-1] > filtered[t-2]:
            labels[t] = 1

    return labels


# =============================================================================
# PREPARATION D'UN ASSET
# =============================================================================

def prepare_single_asset(df: pd.DataFrame, indicator: str,
                         asset_name: str = "Asset") -> tuple:
    """
    Calcule indicateur + labels + séquences pour UN SEUL asset.

    Args:
        df: DataFrame avec OHLC pour un seul asset
        indicator: 'rsi', 'cci', ou 'macd'
        asset_name: Nom pour les logs

    Returns:
        (X, Y) où X shape=(n_sequences, 12, 1), Y shape=(n_sequences, 1)
    """
    logger.info(f"  📈 {asset_name}: Calcul {indicator.upper()} ({len(df):,} bougies)...")

    # 1. Calculer indicateur (1 seul)
    feature = calculate_single_indicator(df, indicator)
    logger.info(f"     {indicator.upper()}: min={feature.min():.1f}, max={feature.max():.1f}, "
                f"mean={feature.mean():.1f}, std={feature.std():.1f}")

    # 2. Générer labels avec filtre Octave sur CLOSE
    labels = generate_octave_close_labels(df, OCTAVE_STEP)
    buy_pct = labels.sum() / len(labels) * 100
    logger.info(f"     Labels Octave(CLOSE, {OCTAVE_STEP}): {buy_pct:.1f}% UP")

    # 3. Reshape pour create_sequences (n, 1)
    features = feature.reshape(-1, 1)
    labels_2d = labels.reshape(-1, 1)

    # 4. Créer séquences de 12 timesteps
    X, Y = create_sequences(features, labels_2d)

    logger.info(f"     → X={X.shape}, Y={Y.shape}")

    return X, Y


# =============================================================================
# PREPARATION ET SAUVEGARDE
# =============================================================================

def prepare_and_save(indicator: str,
                     assets: list = None,
                     output_path: str = None,
                     octave_step: float = OCTAVE_STEP) -> str:
    """
    Prépare les données avec filtre Octave et les sauvegarde.

    Args:
        indicator: 'rsi', 'cci', ou 'macd'
        assets: Liste des assets (défaut: DEFAULT_ASSETS)
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
    logger.info("PRÉPARATION DES DONNÉES - FILTRE OCTAVE")
    logger.info("="*80)
    logger.info(f"📊 Indicateur: {indicator.upper()}")
    logger.info(f"💰 Assets: {', '.join(assets)}")
    logger.info(f"🔧 Filtre: Octave (step={octave_step})")
    logger.info(f"🎯 Target: Direction Octave(CLOSE)")

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
    # 2. Calculer indicateur + labels pour chaque asset
    # =========================================================================
    logger.info(f"\n2. Calcul indicateur et labels par asset...")

    prepared_assets = {}
    for asset_name, df in asset_data.items():
        X, Y = prepare_single_asset(df, indicator, asset_name)
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
    logger.info(f"\n📊 Shapes des datasets:")
    logger.info(f"   Train: X={X_train.shape}, Y={Y_train.shape}")
    logger.info(f"   Val:   X={X_val.shape}, Y={Y_val.shape}")
    logger.info(f"   Test:  X={X_test.shape}, Y={Y_test.shape}")

    # =========================================================================
    # 5. Sauvegarder
    # =========================================================================
    if output_path is None:
        assets_str = '_'.join(assets).lower()
        output_path = f"data/prepared/dataset_{assets_str}_{indicator}_octave{int(octave_step*100)}.npz"

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Métadonnées
    metadata = {
        'created_at': datetime.now().isoformat(),
        'assets': assets,
        'n_assets': len(assets),
        'timeframe': '5min',
        'indicator': indicator,
        'filter_type': 'octave',
        'octave_step': octave_step,
        'target': 'CLOSE',
        'total_candles': total_candles,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'sequence_length': SEQUENCE_LENGTH,
        'n_features': 1,
        'indicator_params': {
            'rsi_period': RSI_PERIOD if indicator == 'rsi' else None,
            'cci_period': CCI_PERIOD if indicator == 'cci' else None,
            'macd_fast': MACD_FAST if indicator == 'macd' else None,
            'macd_slow': MACD_SLOW if indicator == 'macd' else None,
            'macd_signal': MACD_SIGNAL if indicator == 'macd' else None,
        },
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

    logger.info(f"\n✅ Données sauvegardées: {output_path}")
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

    parser = argparse.ArgumentParser(
        description="Prépare les datasets avec filtre Octave (1 indicateur → CLOSE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Exemples:
  # RSI seul → Octave(CLOSE)
  python src/prepare_data_octave.py --indicator rsi --assets BTC ETH BNB ADA LTC

  # CCI seul → Octave(CLOSE)
  python src/prepare_data_octave.py --indicator cci --assets BTC

  # MACD seul → Octave(CLOSE)
  python src/prepare_data_octave.py --indicator macd --assets BTC ETH

  # Avec paramètre Octave différent
  python src/prepare_data_octave.py --indicator rsi --octave-step 0.25

Assets disponibles: {', '.join(available_assets)}
        """
    )

    parser.add_argument('--indicator', '-i', type=str, required=True,
                        choices=['rsi', 'cci', 'macd'],
                        help='Indicateur à utiliser (rsi, cci, macd)')
    parser.add_argument('--assets', '-a', type=str, nargs='+',
                        default=DEFAULT_ASSETS,
                        choices=available_assets,
                        help=f'Assets à inclure (défaut: {DEFAULT_ASSETS})')
    parser.add_argument('--octave-step', type=float, default=OCTAVE_STEP,
                        help=f'Paramètre du filtre Octave (défaut: {OCTAVE_STEP})')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Chemin de sortie (défaut: auto-généré)')

    args = parser.parse_args()

    # Configurer logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    # Préparer et sauvegarder
    output_path = prepare_and_save(
        indicator=args.indicator,
        assets=args.assets,
        output_path=args.output,
        octave_step=args.octave_step
    )

    print(f"\n🎉 Terminé! Dataset prêt: {output_path}")
    print(f"\nPour entraîner:")
    print(f"  python src/train.py --data {output_path} --indicator {args.indicator}")


if __name__ == '__main__':
    main()
