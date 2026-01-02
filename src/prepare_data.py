"""
Script de préparation des données.

Prépare les datasets (X, Y) et les sauvegarde en format numpy (.npz).
Permet de gagner du temps en évitant de recalculer les données à chaque entraînement.

Usage:
    python src/prepare_data.py --timeframe 5 --filter kalman
    python src/prepare_data.py --timeframe 1 --filter decycler --output data/prepared/dataset_1m.npz
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
from indicators import prepare_datasets


def prepare_and_save(timeframe: int = 5,
                     filter_type: str = LABEL_FILTER_TYPE,
                     output_path: str = None,
                     btc_candles: int = None,
                     eth_candles: int = None) -> str:
    """
    Prépare les données et les sauvegarde en format numpy.

    Args:
        timeframe: 1 ou 5 (minutes)
        filter_type: 'decycler' ou 'kalman'
        output_path: Chemin de sortie (défaut: auto-généré)
        btc_candles: Nombre de bougies BTC (défaut: toutes)
        eth_candles: Nombre de bougies ETH (défaut: toutes)

    Returns:
        Chemin du fichier sauvegardé
    """
    logger.info("="*80)
    logger.info("PRÉPARATION DES DONNÉES")
    logger.info("="*80)

    # Sélectionner les fichiers selon timeframe
    if timeframe == 1:
        btc_file = BTC_DATA_FILE_1M
        eth_file = ETH_DATA_FILE_1M
        logger.info(f"📊 Timeframe: 1 minute")
    elif timeframe == 5:
        btc_file = BTC_DATA_FILE_5M
        eth_file = ETH_DATA_FILE_5M
        logger.info(f"📊 Timeframe: 5 minutes")
    else:
        raise ValueError(f"Timeframe invalide: {timeframe}. Utilisez 1 ou 5.")

    logger.info(f"🔧 Filtre pour labels: {filter_type}")

    # Charger BTC
    btc = load_crypto_data(btc_file, n_candles=btc_candles, asset_name='BTC')

    # Charger ETH
    eth = load_crypto_data(eth_file, n_candles=eth_candles, asset_name='ETH')

    # Trim edges
    btc_trimmed = trim_edges(btc, trim_start=TRIM_EDGES, trim_end=TRIM_EDGES)
    eth_trimmed = trim_edges(eth, trim_start=TRIM_EDGES, trim_end=TRIM_EDGES)

    # Combiner
    import pandas as pd
    all_data = pd.concat([btc_trimmed, eth_trimmed], ignore_index=True)
    logger.info(f"🔗 Combinaison BTC + ETH : {len(all_data):,} bougies totales")

    # Split temporel
    train_df, val_df, test_df = temporal_split(
        all_data,
        train_ratio=TRAIN_SPLIT,
        val_ratio=VAL_SPLIT,
        test_ratio=TEST_SPLIT,
        shuffle_train=False  # Pas de shuffle pour la préparation
    )

    # Préparer les datasets (indicateurs + labels + séquences)
    logger.info(f"\n📈 Calcul des indicateurs et labels...")
    datasets = prepare_datasets(train_df, val_df, test_df, filter_type=filter_type)

    X_train, Y_train = datasets['train']
    X_val, Y_val = datasets['val']
    X_test, Y_test = datasets['test']

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
        'btc_candles': len(btc),
        'eth_candles': len(eth),
        'total_candles': len(all_data),
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

    # Sauvegarder en format numpy compressé
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

    # Sauvegarder métadonnées séparément (lisible)
    metadata_path = str(output_path).replace('.npz', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"   Métadonnées: {metadata_path}")

    return output_path


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
    logger.info(f"     Timeframe: {metadata['timeframe']}m")
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
  python src/prepare_data.py --list
        """
    )

    parser.add_argument('--timeframe', '-t', type=int, default=5,
                        choices=[1, 5], help='Timeframe en minutes (1 ou 5)')
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
