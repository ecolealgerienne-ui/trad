"""
Script de préparation des données.

Prépare les datasets (X, Y) et les sauvegarde en format numpy (.npz).
Permet de gagner du temps en évitant de recalculer les données à chaque entraînement.

Usage:
    python src/prepare_data.py --timeframe 5 --filter kalman
    python src/prepare_data.py --timeframe 1 --filter decycler
    python src/prepare_data.py --timeframe all --filter kalman  # Combine 1min + 5min!
"""

import numpy as np
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Import modules locaux
from constants import (
    BTC_DATA_FILE_1M, ETH_DATA_FILE_1M,
    BTC_DATA_FILE_5M, ETH_DATA_FILE_5M,
    BTC_CANDLES, ETH_CANDLES,
    TRIM_EDGES,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    PREPARED_DATA_DIR, PREPARED_DATA_FILE,
    LABEL_FILTER_TYPE,
    SEQUENCE_LENGTH, NUM_INDICATORS,
    RSI_PERIOD, CCI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BOL_PERIOD, BOL_NUM_STD,
    DECYCLER_CUTOFF, KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR
)
from data_utils import load_crypto_data, trim_edges, temporal_split
from indicators import (
    prepare_datasets,
    calculate_all_indicators_for_model,
    generate_all_labels,
    create_sequences
)


def prepare_single_asset(df, filter_type: str, asset_name: str = "Asset") -> tuple:
    """
    Calcule indicateurs + labels + séquences pour UN SEUL asset.

    IMPORTANT: Les indicateurs doivent être calculés SÉPARÉMENT par asset
    avant de merger, sinon les valeurs de fin d'un asset polluent le début
    du suivant (RSI, CCI, MACD utilisent les N valeurs précédentes).

    Args:
        df: DataFrame avec OHLC pour un seul asset
        filter_type: 'kalman' ou 'decycler'
        asset_name: Nom pour les logs

    Returns:
        (X, Y) où X shape=(n_sequences, 12, 4), Y shape=(n_sequences, 4)
    """
    logger.info(f"  📈 {asset_name}: Calcul indicateurs ({len(df):,} bougies)...")

    # 1. Calculer indicateurs (RSI, CCI, BOL, MACD)
    indicators = calculate_all_indicators_for_model(df)

    # 2. Générer labels avec filtre
    labels = generate_all_labels(indicators, filter_type=filter_type)

    # 3. Créer séquences de 12 timesteps
    X, Y = create_sequences(indicators, labels)

    logger.info(f"     → X={X.shape}, Y={Y.shape}")

    return X, Y


def prepare_and_save(timeframe: str = '5',
                     filter_type: str = LABEL_FILTER_TYPE,
                     output_path: str = None,
                     btc_candles: int = None,
                     eth_candles: int = None) -> str:
    """
    Prépare les données et les sauvegarde en format numpy.

    Args:
        timeframe: '1', '5', ou 'all' (combine 1min + 5min)
        filter_type: 'decycler' ou 'kalman'
        output_path: Chemin de sortie (défaut: auto-généré)
        btc_candles: Nombre de bougies BTC par timeframe (défaut: toutes)
        eth_candles: Nombre de bougies ETH par timeframe (défaut: toutes)

    Returns:
        Chemin du fichier sauvegardé
    """
    logger.info("="*80)
    logger.info("PRÉPARATION DES DONNÉES")
    logger.info("="*80)

    timeframe = str(timeframe)  # Convertir en string
    total_btc = 0
    total_eth = 0

    # Charger selon le timeframe
    if timeframe in ['1', '5']:
        # Un seul timeframe
        if timeframe == '1':
            btc_file = BTC_DATA_FILE_1M
            eth_file = ETH_DATA_FILE_1M
            logger.info(f"📊 Timeframe: 1 minute")
        else:
            btc_file = BTC_DATA_FILE_5M
            eth_file = ETH_DATA_FILE_5M
            logger.info(f"📊 Timeframe: 5 minutes")

        logger.info(f"   → Indicateurs calculés PAR ASSET (pas de pollution entre assets)")

        btc = load_crypto_data(btc_file, n_candles=btc_candles, asset_name='BTC')
        eth = load_crypto_data(eth_file, n_candles=eth_candles, asset_name='ETH')

        btc_trimmed = trim_edges(btc, trim_start=TRIM_EDGES, trim_end=TRIM_EDGES)
        eth_trimmed = trim_edges(eth, trim_start=TRIM_EDGES, trim_end=TRIM_EDGES)

        total_btc = len(btc)
        total_eth = len(eth)

        logger.info(f"🔧 Filtre pour labels: {filter_type}")

        # =====================================================================
        # SPLIT TEMPOREL PAR ASSET (avant calcul indicateurs!)
        # =====================================================================
        logger.info(f"\n📊 Split temporel (70/15/15)...")

        btc_train, btc_val, btc_test = temporal_split(
            btc_trimmed, train_ratio=TRAIN_SPLIT, val_ratio=VAL_SPLIT,
            test_ratio=TEST_SPLIT, shuffle_train=False
        )

        eth_train, eth_val, eth_test = temporal_split(
            eth_trimmed, train_ratio=TRAIN_SPLIT, val_ratio=VAL_SPLIT,
            test_ratio=TEST_SPLIT, shuffle_train=False
        )

        # =====================================================================
        # CALCUL INDICATEURS PAR ASSET SÉPARÉMENT (évite pollution)
        # =====================================================================
        logger.info(f"\n📈 Calcul des indicateurs PAR ASSET (évite pollution)...")

        # TRAIN
        logger.info(f"\n🏋️ TRAIN SET:")
        X_btc_train, Y_btc_train = prepare_single_asset(btc_train, filter_type, "BTC-train")
        X_eth_train, Y_eth_train = prepare_single_asset(eth_train, filter_type, "ETH-train")
        X_train = np.concatenate([X_btc_train, X_eth_train], axis=0)
        Y_train = np.concatenate([Y_btc_train, Y_eth_train], axis=0)

        # VAL
        logger.info(f"\n✅ VAL SET:")
        X_btc_val, Y_btc_val = prepare_single_asset(btc_val, filter_type, "BTC-val")
        X_eth_val, Y_eth_val = prepare_single_asset(eth_val, filter_type, "ETH-val")
        X_val = np.concatenate([X_btc_val, X_eth_val], axis=0)
        Y_val = np.concatenate([Y_btc_val, Y_eth_val], axis=0)

        # TEST
        logger.info(f"\n🧪 TEST SET:")
        X_btc_test, Y_btc_test = prepare_single_asset(btc_test, filter_type, "BTC-test")
        X_eth_test, Y_eth_test = prepare_single_asset(eth_test, filter_type, "ETH-test")
        X_test = np.concatenate([X_btc_test, X_eth_test], axis=0)
        Y_test = np.concatenate([Y_btc_test, Y_eth_test], axis=0)

        # Aller directement à la sauvegarde
        logger.info(f"\n📊 Shapes des datasets:")
        logger.info(f"  Train: X={X_train.shape}, Y={Y_train.shape}")
        logger.info(f"  Val:   X={X_val.shape}, Y={Y_val.shape}")
        logger.info(f"  Test:  X={X_test.shape}, Y={Y_test.shape}")

        # Créer le répertoire de sortie
        if output_path is None:
            output_path = f"data/prepared/dataset_{timeframe}m_{filter_type}.npz"

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Métadonnées
        metadata = {
            'created_at': datetime.now().isoformat(),
            'timeframe': timeframe,
            'filter_type': filter_type,
            'btc_candles': total_btc,
            'eth_candles': total_eth,
            'total_candles': len(btc_trimmed) + len(eth_trimmed),
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'sequence_length': SEQUENCE_LENGTH,
            'num_indicators': NUM_INDICATORS,
            'indicator_params': {
                'rsi_period': RSI_PERIOD,
                'cci_period': CCI_PERIOD,
                'bol_period': BOL_PERIOD,
                'bol_num_std': BOL_NUM_STD,
                'macd_fast': MACD_FAST,
                'macd_slow': MACD_SLOW,
                'macd_signal': MACD_SIGNAL
            },
            'filter_params': {
                'decycler_cutoff': DECYCLER_CUTOFF,
                'kalman_process_var': KALMAN_PROCESS_VAR,
                'kalman_measure_var': KALMAN_MEASURE_VAR
            },
            'splits': {
                'train': TRAIN_SPLIT,
                'val': VAL_SPLIT,
                'test': TEST_SPLIT
            }
        }

        # Sauvegarder
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

    elif timeframe == 'all':
        # Train = 1min + 5min, Val/Test = 5min seulement
        logger.info(f"📊 Timeframe: ALL (1min + 5min pour TRAIN, 5min pour VAL/TEST)")
        logger.info(f"   → Plus de données train = meilleure généralisation!")
        logger.info(f"   → Val/Test sur 5min = évaluation réaliste!")
        logger.info(f"   → Indicateurs calculés PAR ASSET (pas de pollution entre assets)")

        # Charger 1min (pour train)
        logger.info(f"\n🔹 Chargement données 1 minute (train)...")
        btc_1m = load_crypto_data(BTC_DATA_FILE_1M, n_candles=btc_candles, asset_name='BTC-1m')
        eth_1m = load_crypto_data(ETH_DATA_FILE_1M, n_candles=eth_candles, asset_name='ETH-1m')

        btc_1m_trimmed = trim_edges(btc_1m, trim_start=TRIM_EDGES, trim_end=TRIM_EDGES)
        eth_1m_trimmed = trim_edges(eth_1m, trim_start=TRIM_EDGES, trim_end=TRIM_EDGES)

        # Charger 5min (pour train + val + test)
        logger.info(f"\n🔹 Chargement données 5 minutes (train + val + test)...")
        btc_5m = load_crypto_data(BTC_DATA_FILE_5M, n_candles=btc_candles, asset_name='BTC-5m')
        eth_5m = load_crypto_data(ETH_DATA_FILE_5M, n_candles=eth_candles, asset_name='ETH-5m')

        btc_5m_trimmed = trim_edges(btc_5m, trim_start=TRIM_EDGES, trim_end=TRIM_EDGES)
        eth_5m_trimmed = trim_edges(eth_5m, trim_start=TRIM_EDGES, trim_end=TRIM_EDGES)

        total_btc = len(btc_1m) + len(btc_5m)
        total_eth = len(eth_1m) + len(eth_5m)

        logger.info(f"\n📈 Données chargées:")
        logger.info(f"   1min: BTC={len(btc_1m_trimmed):,} + ETH={len(eth_1m_trimmed):,}")
        logger.info(f"   5min: BTC={len(btc_5m_trimmed):,} + ETH={len(eth_5m_trimmed):,}")

        logger.info(f"\n🔧 Filtre pour labels: {filter_type}")

        # =====================================================================
        # SPLIT TEMPOREL PAR ASSET (5min seulement pour val/test)
        # =====================================================================
        logger.info(f"\n📊 Split temporel 5min (70/15/15)...")

        # Split BTC-5m
        btc_5m_train, btc_5m_val, btc_5m_test = temporal_split(
            btc_5m_trimmed, train_ratio=TRAIN_SPLIT, val_ratio=VAL_SPLIT,
            test_ratio=TEST_SPLIT, shuffle_train=False
        )

        # Split ETH-5m
        eth_5m_train, eth_5m_val, eth_5m_test = temporal_split(
            eth_5m_trimmed, train_ratio=TRAIN_SPLIT, val_ratio=VAL_SPLIT,
            test_ratio=TEST_SPLIT, shuffle_train=False
        )

        # =====================================================================
        # CALCUL INDICATEURS PAR ASSET SÉPARÉMENT (évite pollution)
        # =====================================================================
        logger.info(f"\n📈 Calcul des indicateurs PAR ASSET (évite pollution)...")

        # --- TRAIN: 1min complet + train de 5min ---
        logger.info(f"\n🏋️ TRAIN SET:")
        X_btc_1m, Y_btc_1m = prepare_single_asset(btc_1m_trimmed, filter_type, "BTC-1m")
        X_eth_1m, Y_eth_1m = prepare_single_asset(eth_1m_trimmed, filter_type, "ETH-1m")
        X_btc_5m_train, Y_btc_5m_train = prepare_single_asset(btc_5m_train, filter_type, "BTC-5m-train")
        X_eth_5m_train, Y_eth_5m_train = prepare_single_asset(eth_5m_train, filter_type, "ETH-5m-train")

        # Merger les trains
        X_train = np.concatenate([X_btc_1m, X_eth_1m, X_btc_5m_train, X_eth_5m_train], axis=0)
        Y_train = np.concatenate([Y_btc_1m, Y_eth_1m, Y_btc_5m_train, Y_eth_5m_train], axis=0)

        # --- VAL: 5min seulement ---
        logger.info(f"\n✅ VAL SET (5min only):")
        X_btc_5m_val, Y_btc_5m_val = prepare_single_asset(btc_5m_val, filter_type, "BTC-5m-val")
        X_eth_5m_val, Y_eth_5m_val = prepare_single_asset(eth_5m_val, filter_type, "ETH-5m-val")

        X_val = np.concatenate([X_btc_5m_val, X_eth_5m_val], axis=0)
        Y_val = np.concatenate([Y_btc_5m_val, Y_eth_5m_val], axis=0)

        # --- TEST: 5min seulement ---
        logger.info(f"\n🧪 TEST SET (5min only):")
        X_btc_5m_test, Y_btc_5m_test = prepare_single_asset(btc_5m_test, filter_type, "BTC-5m-test")
        X_eth_5m_test, Y_eth_5m_test = prepare_single_asset(eth_5m_test, filter_type, "ETH-5m-test")

        X_test = np.concatenate([X_btc_5m_test, X_eth_5m_test], axis=0)
        Y_test = np.concatenate([Y_btc_5m_test, Y_eth_5m_test], axis=0)

        # Aller directement à la sauvegarde
        logger.info(f"\n📊 Shapes des datasets:")
        logger.info(f"  Train: X={X_train.shape}, Y={Y_train.shape}")
        logger.info(f"  Val:   X={X_val.shape}, Y={Y_val.shape}")
        logger.info(f"  Test:  X={X_test.shape}, Y={Y_test.shape}")

        # Créer le répertoire de sortie
        if output_path is None:
            output_path = f"data/prepared/dataset_all_{filter_type}.npz"

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Métadonnées
        total_1m = len(btc_1m_trimmed) + len(eth_1m_trimmed)
        total_5m = len(btc_5m_trimmed) + len(eth_5m_trimmed)
        train_5m_total = len(btc_5m_train) + len(eth_5m_train)

        metadata = {
            'created_at': datetime.now().isoformat(),
            'timeframe': 'all',
            'filter_type': filter_type,
            'btc_candles': total_btc,
            'eth_candles': total_eth,
            'total_candles': total_1m + total_5m,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'train_composition': {
                '1min': total_1m,
                '5min': train_5m_total
            },
            'val_test_source': '5min_only',
            'sequence_length': SEQUENCE_LENGTH,
            'num_indicators': NUM_INDICATORS,
            'indicator_params': {
                'rsi_period': RSI_PERIOD,
                'cci_period': CCI_PERIOD,
                'bol_period': BOL_PERIOD,
                'bol_num_std': BOL_NUM_STD,
                'macd_fast': MACD_FAST,
                'macd_slow': MACD_SLOW,
                'macd_signal': MACD_SIGNAL
            },
            'filter_params': {
                'decycler_cutoff': DECYCLER_CUTOFF,
                'kalman_process_var': KALMAN_PROCESS_VAR,
                'kalman_measure_var': KALMAN_MEASURE_VAR
            },
            'splits': {
                'train': TRAIN_SPLIT,
                'val': VAL_SPLIT,
                'test': TEST_SPLIT
            }
        }

        # Sauvegarder
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

    else:
        raise ValueError(f"Timeframe invalide: {timeframe}. Utilisez '1', '5', ou 'all'.")


def load_prepared_data(path: str = None) -> dict:
    """
    Charge les données préparées depuis un fichier .npz.

    Args:
        path: Chemin vers le fichier .npz (défaut: PREPARED_DATA_FILE)

    Returns:
        Dictionnaire avec:
            'train': (X_train, Y_train)
            'val': (X_val, Y_val)
            'test': (X_test, Y_test)
            'metadata': dict avec les paramètres utilisés
    """
    if path is None:
        # Chercher le fichier le plus récent
        prepared_dir = Path(PREPARED_DATA_DIR)
        if prepared_dir.exists():
            npz_files = list(prepared_dir.glob('*.npz'))
            if npz_files:
                path = max(npz_files, key=lambda p: p.stat().st_mtime)
                logger.info(f"📂 Chargement du fichier le plus récent: {path}")
            else:
                raise FileNotFoundError(f"Aucun fichier .npz trouvé dans {prepared_dir}")
        else:
            raise FileNotFoundError(f"Répertoire {prepared_dir} non trouvé")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {path}")

    logger.info(f"📂 Chargement des données: {path}")

    data = np.load(path, allow_pickle=True)

    # Parser les métadonnées
    metadata = json.loads(str(data['metadata']))

    result = {
        'train': (data['X_train'], data['Y_train']),
        'val': (data['X_val'], data['Y_val']),
        'test': (data['X_test'], data['Y_test']),
        'metadata': metadata
    }

    logger.info(f"  ✅ Données chargées:")
    logger.info(f"     Train: {data['X_train'].shape}")
    logger.info(f"     Val:   {data['X_val'].shape}")
    logger.info(f"     Test:  {data['X_test'].shape}")
    tf = metadata['timeframe']
    tf_str = f"{tf}m" if tf != 'all' else "all (1m+5m train, 5m val/test)"
    logger.info(f"     Timeframe: {tf_str}")
    logger.info(f"     Filtre: {metadata['filter_type']}")

    return result


def list_prepared_datasets():
    """Liste tous les datasets préparés disponibles."""
    prepared_dir = Path(PREPARED_DATA_DIR)

    if not prepared_dir.exists():
        print(f"❌ Répertoire {prepared_dir} non trouvé")
        return

    npz_files = list(prepared_dir.glob('*.npz'))

    if not npz_files:
        print(f"❌ Aucun dataset préparé dans {prepared_dir}")
        return

    print(f"\n📁 Datasets disponibles ({len(npz_files)}):\n")

    for f in sorted(npz_files, key=lambda p: p.stat().st_mtime, reverse=True):
        # Charger métadonnées
        metadata_path = str(f).replace('.npz', '_metadata.json')
        if Path(metadata_path).exists():
            with open(metadata_path) as mf:
                meta = json.load(mf)
            print(f"  📊 {f.name}")
            print(f"     Timeframe: {meta['timeframe']}m | Filtre: {meta['filter_type']}")
            print(f"     Train: {meta['train_size']:,} | Val: {meta['val_size']:,} | Test: {meta['test_size']:,}")
            print(f"     RSI={meta['indicator_params']['rsi_period']}, CCI={meta['indicator_params']['cci_period']}, MACD={meta['indicator_params']['macd_fast']}/{meta['indicator_params']['macd_slow']}")
            print(f"     Créé: {meta['created_at']}")
            print()
        else:
            print(f"  📊 {f.name} (pas de métadonnées)")
            print()


def main():
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Prépare et sauvegarde les datasets pour l'entraînement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python src/prepare_data.py --timeframe 5 --filter kalman
  python src/prepare_data.py --timeframe 1 --filter decycler
  python src/prepare_data.py --timeframe all --filter kalman  # 1min+5min combinés!
  python src/prepare_data.py --list
        """
    )

    parser.add_argument('--timeframe', '-t', type=str, default='5',
                        choices=['1', '5', 'all'],
                        help='Timeframe: 1, 5, ou all (1min+5min train, 5min val/test)')
    parser.add_argument('--filter', '-f', type=str, default=LABEL_FILTER_TYPE,
                        choices=['decycler', 'kalman'], help='Filtre pour les labels')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Chemin de sortie (défaut: auto-généré)')
    parser.add_argument('--btc-candles', type=int, default=None,
                        help='Nombre de bougies BTC (défaut: toutes)')
    parser.add_argument('--eth-candles', type=int, default=None,
                        help='Nombre de bougies ETH (défaut: toutes)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='Liste les datasets préparés disponibles')

    args = parser.parse_args()

    # Configurer logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    if args.list:
        list_prepared_datasets()
        return

    # Préparer et sauvegarder
    output_path = prepare_and_save(
        timeframe=args.timeframe,
        filter_type=args.filter,
        output_path=args.output,
        btc_candles=args.btc_candles,
        eth_candles=args.eth_candles
    )

    print(f"\n🎉 Terminé! Dataset prêt: {output_path}")
    print(f"\nPour entraîner:")
    print(f"  python src/train.py --data {output_path}")


if __name__ == '__main__':
    main()
