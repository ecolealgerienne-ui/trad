"""
Script de préparation des données Multi-View Learning.

Pour chaque indicateur cible (RSI, CCI, MACD), les features sont calculées
avec des paramètres optimisés pour synchroniser avec cette cible.

Usage:
    python src/prepare_data_multiview.py --target rsi --assets BTC ETH BNB ADA LTC
    python src/prepare_data_multiview.py --target cci --assets BTC ETH BNB ADA LTC
    python src/prepare_data_multiview.py --target macd --assets BTC ETH BNB ADA LTC
"""

import numpy as np
import pandas as pd
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Import modules locaux
from constants import (
    AVAILABLE_ASSETS_5M, DEFAULT_ASSETS,
    TRIM_EDGES,
    PREPARED_DATA_DIR,
    SEQUENCE_LENGTH,
    KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR,
    RSI_PERIOD, CCI_PERIOD, CCI_CONSTANT,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
)
from data_utils import load_crypto_data, trim_edges, split_sequences_chronological
from indicators import (
    calculate_rsi, calculate_cci, calculate_macd,
    normalize_cci, normalize_macd_histogram,
    create_sequences
)
from filters import kalman_filter


# =============================================================================
# PARAMETRES OPTIMISES PAR CIBLE (résultats de optimize_sync_per_target.py)
# =============================================================================

# Pour prédire RSI: paramètres optimisés pour CCI et MACD
PARAMS_FOR_RSI_TARGET = {
    'RSI': {'period': RSI_PERIOD},  # Cible = défaut
    'CCI': {'period': 51},          # Optimisé pour RSI
    'MACD': {'fast': 13, 'slow': 67},  # Optimisé pour RSI
}

# Pour prédire CCI: paramètres optimisés pour RSI et MACD
PARAMS_FOR_CCI_TARGET = {
    'RSI': {'period': 18},          # Optimisé pour CCI
    'CCI': {'period': CCI_PERIOD},  # Cible = défaut
    'MACD': {'fast': 10, 'slow': 67},  # Optimisé pour CCI
}

# Pour prédire MACD: paramètres optimisés pour RSI et CCI
PARAMS_FOR_MACD_TARGET = {
    'RSI': {'period': 18},          # Optimisé pour MACD
    'CCI': {'period': 26},          # Optimisé pour MACD
    'MACD': {'fast': MACD_FAST, 'slow': MACD_SLOW},  # Cible = défaut
}

# Mapping target → params
TARGET_PARAMS = {
    'RSI': PARAMS_FOR_RSI_TARGET,
    'CCI': PARAMS_FOR_CCI_TARGET,
    'MACD': PARAMS_FOR_MACD_TARGET,
}


def calculate_indicators_multiview(df: pd.DataFrame, target: str) -> np.ndarray:
    """
    Calcule les 3 indicateurs avec paramètres optimisés pour la cible.

    Args:
        df: DataFrame avec OHLC
        target: 'RSI', 'CCI' ou 'MACD'

    Returns:
        Array (n_samples, 3) avec RSI, CCI, MACD normalisés
    """
    params = TARGET_PARAMS[target.upper()]

    # 1. RSI
    rsi = calculate_rsi(df['close'], period=params['RSI']['period'])
    logger.info(f"    RSI(period={params['RSI']['period']})")

    # 2. CCI normalisé
    cci_raw = calculate_cci(
        df['high'], df['low'], df['close'],
        period=params['CCI']['period'],
        constant=CCI_CONSTANT
    )
    cci_norm = normalize_cci(cci_raw)
    logger.info(f"    CCI(period={params['CCI']['period']})")

    # 3. MACD histogram normalisé
    macd_data = calculate_macd(
        df['close'],
        fast_period=params['MACD']['fast'],
        slow_period=params['MACD']['slow'],
        signal_period=MACD_SIGNAL
    )
    macd_hist_norm = normalize_macd_histogram(macd_data['histogram'])
    logger.info(f"    MACD(fast={params['MACD']['fast']}, slow={params['MACD']['slow']})")

    # Combiner
    indicators = np.column_stack([rsi, cci_norm, macd_hist_norm])

    # Gérer NaN
    indicators_df = pd.DataFrame(indicators, columns=['RSI', 'CCI', 'MACD'])
    indicators_df = indicators_df.ffill().fillna(50.0)

    return indicators_df.values


def generate_single_label(indicators: np.ndarray, target_idx: int) -> np.ndarray:
    """
    Génère le label binaire pour UN SEUL indicateur cible.

    Args:
        indicators: Array (n_samples, 3)
        target_idx: 0=RSI, 1=CCI, 2=MACD

    Returns:
        Array (n_samples,) avec labels binaires
    """
    # Appliquer Kalman sur la cible
    target_values = indicators[:, target_idx]
    filtered = kalman_filter(target_values, KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR)

    # Générer labels (pente)
    labels = np.zeros(len(filtered), dtype=int)
    for t in range(2, len(filtered)):
        if filtered[t-1] > filtered[t-2]:
            labels[t] = 1

    return labels


def prepare_single_asset_multiview(df: pd.DataFrame, target: str,
                                     asset_name: str = "Asset") -> tuple:
    """
    Prépare les données pour un asset avec paramètres Multi-View.

    Args:
        df: DataFrame OHLC
        target: 'RSI', 'CCI' ou 'MACD'
        asset_name: Nom pour les logs

    Returns:
        (X, Y) avec X shape=(n_seq, 12, 3), Y shape=(n_seq,)
    """
    target_upper = target.upper()
    target_idx = {'RSI': 0, 'CCI': 1, 'MACD': 2}[target_upper]

    logger.info(f"  📈 {asset_name}: Calcul indicateurs Multi-View pour {target_upper}...")

    # 1. Calculer indicateurs avec params optimisés pour cette cible
    indicators = calculate_indicators_multiview(df, target_upper)

    # 2. Générer label UNIQUE pour la cible
    labels = generate_single_label(indicators, target_idx)

    # Reshape pour create_sequences (attend 2D pour Y)
    labels_2d = labels.reshape(-1, 1)

    # 3. Créer séquences
    X, Y = create_sequences(indicators, labels_2d, sequence_length=SEQUENCE_LENGTH)

    # Y est (n_seq, 1), on le flatten
    Y = Y.flatten()

    logger.info(f"     → X={X.shape}, Y={Y.shape}")

    return X, Y


def prepare_and_save_multiview(target: str, assets: list = None,
                                 output_path: str = None) -> str:
    """
    Prépare et sauvegarde le dataset Multi-View pour une cible.

    Args:
        target: 'RSI', 'CCI' ou 'MACD'
        assets: Liste des assets
        output_path: Chemin de sortie (auto si None)

    Returns:
        Chemin du fichier sauvegardé
    """
    if assets is None:
        assets = DEFAULT_ASSETS

    target_upper = target.upper()
    params = TARGET_PARAMS[target_upper]

    logger.info("="*80)
    logger.info(f"PRÉPARATION MULTI-VIEW - Cible: {target_upper}")
    logger.info("="*80)
    logger.info(f"💰 Assets: {', '.join(assets)}")
    logger.info(f"🎯 Cible: {target_upper}")
    logger.info(f"📊 Paramètres features:")
    for ind, p in params.items():
        logger.info(f"   {ind}: {p}")

    # 1. Charger données
    logger.info(f"\n1. Chargement données...")
    asset_data = {}
    for asset_name in assets:
        file_path = AVAILABLE_ASSETS_5M[asset_name]
        df = load_crypto_data(file_path, asset_name=asset_name)
        df_trimmed = trim_edges(df, trim_start=TRIM_EDGES, trim_end=TRIM_EDGES)
        asset_data[asset_name] = df_trimmed

    # 2. Préparer par asset
    logger.info(f"\n2. Calcul indicateurs et labels Multi-View...")
    prepared_assets = {}
    for asset_name, df in asset_data.items():
        X, Y = prepare_single_asset_multiview(df, target_upper, asset_name)
        prepared_assets[asset_name] = (X, Y)

    # 3. Split chronologique par asset
    logger.info(f"\n3. Split chronologique par asset...")
    train_X, train_Y = [], []
    val_X, val_Y = [], []
    test_X, test_Y = [], []

    for asset_name, (X, Y) in prepared_assets.items():
        # Reshape Y pour split
        Y_2d = Y.reshape(-1, 1)
        (Xtr, Ytr), (Xv, Yv), (Xte, Yte) = split_sequences_chronological(X, Y_2d)

        train_X.append(Xtr)
        train_Y.append(Ytr.flatten())
        val_X.append(Xv)
        val_Y.append(Yv.flatten())
        test_X.append(Xte)
        test_Y.append(Yte.flatten())

        logger.info(f"   {asset_name}: train={len(Xtr)}, val={len(Xv)}, test={len(Xte)}")

    # 4. Merger
    X_train = np.concatenate(train_X)
    Y_train = np.concatenate(train_Y)
    X_val = np.concatenate(val_X)
    Y_val = np.concatenate(val_Y)
    X_test = np.concatenate(test_X)
    Y_test = np.concatenate(test_Y)

    logger.info(f"\n4. Datasets finaux:")
    logger.info(f"   Train: X={X_train.shape}, Y={Y_train.shape}")
    logger.info(f"   Val:   X={X_val.shape}, Y={Y_val.shape}")
    logger.info(f"   Test:  X={X_test.shape}, Y={Y_test.shape}")

    # 5. Sauvegarder
    if output_path is None:
        assets_str = '_'.join([a.lower() for a in assets])
        output_path = f"{PREPARED_DATA_DIR}/dataset_{assets_str}_multiview_{target.lower()}.npz"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        X_train=X_train, Y_train=Y_train,
        X_val=X_val, Y_val=Y_val,
        X_test=X_test, Y_test=Y_test
    )

    # Metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'target': target_upper,
        'assets': assets,
        'params': params,
        'shapes': {
            'X_train': list(X_train.shape),
            'Y_train': list(Y_train.shape),
            'X_val': list(X_val.shape),
            'Y_val': list(Y_val.shape),
            'X_test': list(X_test.shape),
            'Y_test': list(Y_test.shape),
        }
    }

    metadata_path = str(output_path).replace('.npz', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\n✅ Sauvegardé: {output_path}")
    logger.info(f"✅ Metadata: {metadata_path}")

    return str(output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Préparation données Multi-View Learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--target', '-t', type=str, required=True,
                        choices=['rsi', 'cci', 'macd'],
                        help='Indicateur cible')

    parser.add_argument('--assets', '-a', nargs='+',
                        default=['BTC', 'ETH', 'BNB', 'ADA', 'LTC'],
                        help='Assets à utiliser')

    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Chemin de sortie (auto si non spécifié)')

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    args = parse_args()

    output_path = prepare_and_save_multiview(
        target=args.target,
        assets=args.assets,
        output_path=args.output
    )

    print(f"\n📁 Dataset prêt: {output_path}")


if __name__ == '__main__':
    main()
