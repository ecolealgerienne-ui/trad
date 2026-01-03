"""
Script de préparation des données Multi-View Learning avec labels 30min.

Pour chaque indicateur cible (RSI, CCI, MACD), les features sont calculées
avec des paramètres optimisés pour synchroniser avec cette cible.

Basé sur prepare_data_30min.py mais avec paramètres Multi-View.

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
from prepare_data_30min import resample_5min_to_30min, align_30min_to_5min

# Alias
split_sequences = split_sequences_chronological


# =============================================================================
# PARAMETRES PAR DEFAUT (optimisés pour CLOSE - Multi-View abandonné)
# =============================================================================
# Note: L'approche Multi-View a été testée et abandonnée car la synchronisation
# des features réduit leur diversité et n'améliore pas la prédiction.
# Tous les indicateurs utilisent maintenant les paramètres par défaut.

DEFAULT_INDICATOR_PARAMS = {
    'RSI': {'period': RSI_PERIOD},      # 22 - optimisé pour Close
    'CCI': {'period': CCI_PERIOD},      # 32 - optimisé pour Close
    'MACD': {'fast': MACD_FAST, 'slow': MACD_SLOW},  # 8/42 - optimisé pour Close
}

# Mapping target → params (tous utilisent les mêmes paramètres par défaut)
TARGET_PARAMS = {
    'RSI': DEFAULT_INDICATOR_PARAMS,
    'CCI': DEFAULT_INDICATOR_PARAMS,
    'MACD': DEFAULT_INDICATOR_PARAMS,
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

    # 2. CCI normalisé
    cci_raw = calculate_cci(
        df['high'], df['low'], df['close'],
        period=params['CCI']['period'],
        constant=CCI_CONSTANT
    )
    cci_norm = normalize_cci(cci_raw)

    # 3. MACD histogram normalisé
    macd_data = calculate_macd(
        df['close'],
        fast_period=params['MACD']['fast'],
        slow_period=params['MACD']['slow'],
        signal_period=MACD_SIGNAL
    )
    macd_hist_norm = normalize_macd_histogram(macd_data['histogram'])

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


def prepare_single_asset_multiview_30min(df_5min: pd.DataFrame,
                                          target: str,
                                          asset_name: str = "Asset"):
    """
    Prépare les données pour un asset avec params Multi-View et labels 30min.

    Basé sur prepare_single_asset_30min mais avec:
    - Paramètres optimisés par cible pour les features
    - Label unique pour la cible

    Args:
        df_5min: DataFrame 5min avec OHLCV
        target: 'RSI', 'CCI' ou 'MACD'
        asset_name: Nom pour les logs

    Returns:
        (X, Y) avec X shape=(n_seq, 12, 7), Y shape=(n_seq,)
    """
    target_upper = target.upper()
    target_idx = {'RSI': 0, 'CCI': 1, 'MACD': 2}[target_upper]
    params = TARGET_PARAMS[target_upper]

    logger.info(f"\n{'='*60}")
    logger.info(f"📈 {asset_name}: Multi-View pour {target_upper}")
    logger.info(f"{'='*60}")
    logger.info(f"  Params: RSI({params['RSI']['period']}), "
                f"CCI({params['CCI']['period']}), "
                f"MACD({params['MACD']['fast']}/{params['MACD']['slow']})")

    # =========================================================================
    # 1. Préparer les index datetime
    # =========================================================================
    df = df_5min.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    index_5min = df.index
    logger.info(f"  Données 5min: {len(df)} bougies")

    # =========================================================================
    # 2. Resample vers 30min
    # =========================================================================
    df_30min = resample_5min_to_30min(df_5min)
    index_30min = df_30min.index

    # =========================================================================
    # 3. Calculer indicateurs 5min avec params Multi-View
    # =========================================================================
    logger.info(f"  📊 Calcul indicateurs 5min (Multi-View)...")
    df_5min_with_ts = df.reset_index()
    indicators_5min = calculate_indicators_multiview(df_5min_with_ts, target_upper)
    logger.info(f"     → Shape: {indicators_5min.shape}")

    # =========================================================================
    # 4. Calculer indicateurs 30min avec params Multi-View
    # =========================================================================
    logger.info(f"  📊 Calcul indicateurs 30min (Multi-View)...")
    df_30min_with_ts = df_30min.reset_index()
    indicators_30min = calculate_indicators_multiview(df_30min_with_ts, target_upper)
    logger.info(f"     → Shape: {indicators_30min.shape}")

    # =========================================================================
    # 5. Générer label UNIQUE pour la cible (depuis 30min)
    # =========================================================================
    logger.info(f"  🏷️ Génération label {target_upper} (depuis 30min)...")
    labels_30min = generate_single_label(indicators_30min, target_idx)

    buy_pct = labels_30min.sum() / len(labels_30min) * 100
    logger.info(f"     → {target_upper}: {buy_pct:.1f}% BUY")

    # =========================================================================
    # 6. Aligner labels 30min sur 5min (forward-fill)
    # =========================================================================
    logger.info(f"  🔄 Alignement labels 30min → 5min...")
    labels_aligned = align_30min_to_5min(labels_30min, index_30min, index_5min)

    # =========================================================================
    # 7. Aligner indicateurs 30min sur 5min
    # =========================================================================
    logger.info(f"  🔄 Alignement indicateurs 30min → 5min...")
    indicators_30min_aligned = align_30min_to_5min(indicators_30min, index_30min, index_5min)

    # Concatener: 5min + 30min = 6 features
    indicators_combined = np.hstack([indicators_5min, indicators_30min_aligned])
    logger.info(f"     → Features combinées: {indicators_combined.shape}")

    # =========================================================================
    # 8. Ajouter Step Index
    # =========================================================================
    minutes = index_5min.minute
    step_index = (minutes % 30) // 5 + 1
    step_index_normalized = (step_index - 1) / 5.0
    step_index_col = step_index_normalized.values.reshape(-1, 1)
    indicators_combined = np.hstack([indicators_combined, step_index_col])
    logger.info(f"     → Features finales: {indicators_combined.shape} (6 + step_index)")

    # =========================================================================
    # 9. Créer séquences
    # =========================================================================
    # Reshape labels pour create_sequences (attend 2D)
    labels_2d = labels_aligned.reshape(-1, 1)
    X, Y = create_sequences(indicators_combined, labels_2d, sequence_length=SEQUENCE_LENGTH)
    # Garder Y en 2D (n, 1) pour compatibilité avec train.py

    logger.info(f"  ✅ X={X.shape}, Y={Y.shape}")

    return X, Y


def prepare_and_save_multiview_30min(target: str,
                                       assets: list = None,
                                       output_path: str = None) -> str:
    """
    Prépare et sauvegarde le dataset Multi-View avec labels 30min.

    Args:
        target: 'RSI', 'CCI' ou 'MACD'
        assets: Liste des assets
        output_path: Chemin de sortie

    Returns:
        Chemin du fichier sauvegardé
    """
    if assets is None:
        assets = DEFAULT_ASSETS

    target_upper = target.upper()
    params = TARGET_PARAMS[target_upper]

    logger.info("="*80)
    logger.info(f"PRÉPARATION MULTI-VIEW 30MIN - Cible: {target_upper}")
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
    logger.info(f"\n2. Préparation Multi-View par asset...")
    prepared_assets = {}
    for asset_name, df in asset_data.items():
        X, Y = prepare_single_asset_multiview_30min(df, target_upper, asset_name)
        prepared_assets[asset_name] = (X, Y)

    # 3. Split chronologique par asset
    logger.info(f"\n3. Split chronologique...")
    train_X, train_Y = [], []
    val_X, val_Y = [], []
    test_X, test_Y = [], []

    for asset_name, (X, Y) in prepared_assets.items():
        # Y est déjà (n, 1) depuis prepare_single_asset
        (Xtr, Ytr), (Xv, Yv), (Xte, Yte) = split_sequences(X, Y)

        train_X.append(Xtr)
        train_Y.append(Ytr)  # Garder 2D (n, 1)
        val_X.append(Xv)
        val_Y.append(Yv)
        test_X.append(Xte)
        test_Y.append(Yte)

        logger.info(f"   {asset_name}: train={len(Xtr)}, val={len(Xv)}, test={len(Xte)}")

    # 4. Merger
    X_train = np.concatenate(train_X)
    Y_train = np.concatenate(train_Y)  # Shape (n_total, 1)
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
        output_path = f"data/prepared/dataset_{assets_str}_multiview30min_{target.lower()}.npz"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Metadata (doit être inclus dans le .npz pour load_prepared_data)
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'target': target_upper,
        'assets': assets,
        'n_assets': len(assets),
        'params': {k: v for k, v in params.items()},
        'n_features': X_train.shape[2],
        'feature_desc': '5min(3) + 30min(3) + step_index(1) = 7 features Multi-View',
        'label_desc': f'Pente Kalman({target_upper}) 30min',
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'sequence_length': SEQUENCE_LENGTH,
        'multiview': True,
    }

    # Sauvegarder avec metadata dans le .npz
    np.savez_compressed(
        output_path,
        X_train=X_train, Y_train=Y_train,
        X_val=X_val, Y_val=Y_val,
        X_test=X_test, Y_test=Y_test,
        metadata=json.dumps(metadata)  # Inclus dans le npz!
    )

    # Aussi sauvegarder en JSON pour référence
    metadata_path = str(output_path).replace('.npz', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\n✅ Sauvegardé: {output_path}")
    logger.info(f"✅ Metadata: {metadata_path}")

    return str(output_path)


def prepare_single_asset_multiview_5min(df_5min: pd.DataFrame,
                                         target: str,
                                         asset_name: str = "Asset"):
    """
    Prépare les données pour un asset avec params Multi-View (5min seulement).

    Args:
        df_5min: DataFrame 5min avec OHLCV
        target: 'RSI', 'CCI' ou 'MACD'
        asset_name: Nom pour les logs

    Returns:
        (X, Y) avec X shape=(n_seq, 12, 3), Y shape=(n_seq,)
    """
    target_upper = target.upper()
    target_idx = {'RSI': 0, 'CCI': 1, 'MACD': 2}[target_upper]
    params = TARGET_PARAMS[target_upper]

    logger.info(f"\n{'='*60}")
    logger.info(f"📈 {asset_name}: Multi-View 5min pour {target_upper}")
    logger.info(f"{'='*60}")
    logger.info(f"  Params: RSI({params['RSI']['period']}), "
                f"CCI({params['CCI']['period']}), "
                f"MACD({params['MACD']['fast']}/{params['MACD']['slow']})")

    # 1. Calculer indicateurs 5min avec params Multi-View
    logger.info(f"  📊 Calcul indicateurs 5min (Multi-View)...")
    indicators_5min = calculate_indicators_multiview(df_5min, target_upper)
    logger.info(f"     → Shape: {indicators_5min.shape}")

    # 2. Générer label UNIQUE pour la cible (depuis 5min)
    logger.info(f"  🏷️ Génération label {target_upper} (depuis 5min)...")
    labels = generate_single_label(indicators_5min, target_idx)

    buy_pct = labels.sum() / len(labels) * 100
    logger.info(f"     → {target_upper}: {buy_pct:.1f}% BUY")

    # 3. Créer séquences
    labels_2d = labels.reshape(-1, 1)
    X, Y = create_sequences(indicators_5min, labels_2d, sequence_length=SEQUENCE_LENGTH)
    # Garder Y en 2D (n, 1) pour compatibilité avec train.py

    logger.info(f"  ✅ X={X.shape}, Y={Y.shape}")

    return X, Y


def prepare_and_save_multiview_5min(target: str,
                                     assets: list = None,
                                     output_path: str = None) -> str:
    """
    Prépare et sauvegarde le dataset Multi-View avec labels 5min.
    """
    if assets is None:
        assets = DEFAULT_ASSETS

    target_upper = target.upper()
    params = TARGET_PARAMS[target_upper]

    logger.info("="*80)
    logger.info(f"PRÉPARATION MULTI-VIEW 5MIN - Cible: {target_upper}")
    logger.info("="*80)
    logger.info(f"💰 Assets: {', '.join(assets)}")
    logger.info(f"🎯 Cible: {target_upper}")

    # 1. Charger données
    logger.info(f"\n1. Chargement données...")
    asset_data = {}
    for asset_name in assets:
        file_path = AVAILABLE_ASSETS_5M[asset_name]
        df = load_crypto_data(file_path, asset_name=asset_name)
        df_trimmed = trim_edges(df, trim_start=TRIM_EDGES, trim_end=TRIM_EDGES)
        asset_data[asset_name] = df_trimmed

    # 2. Préparer par asset
    logger.info(f"\n2. Préparation Multi-View par asset...")
    prepared_assets = {}
    for asset_name, df in asset_data.items():
        X, Y = prepare_single_asset_multiview_5min(df, target_upper, asset_name)
        prepared_assets[asset_name] = (X, Y)

    # 3. Split chronologique
    logger.info(f"\n3. Split chronologique...")
    train_X, train_Y = [], []
    val_X, val_Y = [], []
    test_X, test_Y = [], []

    for asset_name, (X, Y) in prepared_assets.items():
        # Y est déjà (n, 1) depuis prepare_single_asset
        (Xtr, Ytr), (Xv, Yv), (Xte, Yte) = split_sequences(X, Y)

        train_X.append(Xtr)
        train_Y.append(Ytr)  # Garder 2D (n, 1)
        val_X.append(Xv)
        val_Y.append(Yv)
        test_X.append(Xte)
        test_Y.append(Yte)

        logger.info(f"   {asset_name}: train={len(Xtr)}, val={len(Xv)}, test={len(Xte)}")

    # 4. Merger
    X_train = np.concatenate(train_X)
    Y_train = np.concatenate(train_Y)  # Shape (n_total, 1)
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
        output_path = f"data/prepared/dataset_{assets_str}_multiview5min_{target.lower()}.npz"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        'timestamp': datetime.now().isoformat(),
        'target': target_upper,
        'timeframe': '5min',
        'assets': assets,
        'n_assets': len(assets),
        'params': {k: v for k, v in params.items()},
        'n_features': X_train.shape[2],
        'feature_desc': '5min(3) = 3 features Multi-View',
        'label_desc': f'Pente Kalman({target_upper}) 5min',
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'sequence_length': SEQUENCE_LENGTH,
        'multiview': True,
    }

    np.savez_compressed(
        output_path,
        X_train=X_train, Y_train=Y_train,
        X_val=X_val, Y_val=Y_val,
        X_test=X_test, Y_test=Y_test,
        metadata=json.dumps(metadata)
    )

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

    parser.add_argument('--timeframe', '-tf', type=str, default='5min',
                        choices=['5min', '30min'],
                        help='Timeframe des labels (5min ou 30min)')

    parser.add_argument('--assets', '-a', nargs='+',
                        default=['BTC', 'ETH', 'BNB', 'ADA', 'LTC'],
                        help='Assets à utiliser')

    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Chemin de sortie')

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    args = parse_args()

    if args.timeframe == '5min':
        output_path = prepare_and_save_multiview_5min(
            target=args.target,
            assets=args.assets,
            output_path=args.output
        )
    else:
        output_path = prepare_and_save_multiview_30min(
            target=args.target,
            assets=args.assets,
            output_path=args.output
        )

    print(f"\n📁 Dataset prêt: {output_path}")
    print(f"\nPour entraîner:")
    print(f"  python src/train.py --data {output_path}")


if __name__ == '__main__':
    main()
