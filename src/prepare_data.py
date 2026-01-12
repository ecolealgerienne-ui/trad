"""
Script de préparation des données - 5min seulement.

Prépare les datasets (X, Y) et les sauvegarde en format numpy (.npz).
Permet de gagner du temps en évitant de recalculer les données à chaque entraînement.

Indicateurs: RSI, CCI, MACD (3 indicateurs, BOL retiré car non synchronisable)

Usage:
    python src/prepare_data.py --filter kalman --assets BTC ETH BNB ADA LTC
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
    AVAILABLE_ASSETS_5M, DEFAULT_ASSETS,
    TRIM_EDGES,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    PREPARED_DATA_DIR, PREPARED_DATA_FILE,
    LABEL_FILTER_TYPE,
    SEQUENCE_LENGTH, NUM_INDICATORS,
    RSI_PERIOD, CCI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    DECYCLER_CUTOFF, KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR
)
from data_utils import load_crypto_data, trim_edges, split_sequences_chronological
from utils import log_dataset_metadata
from indicators import (
    calculate_all_indicators_for_model,
    generate_all_labels,
    create_sequences
)


def prepare_single_asset(df, filter_type: str, asset_name: str = "Asset",
                         add_position_index: bool = False) -> tuple:
    """
    Calcule indicateurs + labels + séquences pour UN SEUL asset.

    IMPORTANT: Les indicateurs doivent être calculés SÉPARÉMENT par asset
    avant de merger, sinon les valeurs de fin d'un asset polluent le début
    du suivant (RSI, CCI, MACD utilisent les N valeurs précédentes).

    Note: BOL retiré car impossible à synchroniser (toujours lag +1).

    Args:
        df: DataFrame avec OHLC pour un seul asset
        filter_type: 'kalman' ou 'decycler'
        asset_name: Nom pour les logs
        add_position_index: Si True, ajoute Position Index (1/12 → 12/12)

    Returns:
        (X, Y) où X shape=(n_sequences, 12, 3 ou 4), Y shape=(n_sequences, 3)
    """
    logger.info(f"  📈 {asset_name}: Calcul indicateurs ({len(df):,} bougies)...")

    # 1. Calculer indicateurs (RSI, CCI, MACD - BOL retiré)
    indicators = calculate_all_indicators_for_model(df)

    # 2. Générer labels avec filtre
    labels = generate_all_labels(indicators, filter_type=filter_type)

    # 3. Créer séquences de 12 timesteps (avec ou sans Position Index)
    X, Y = create_sequences(indicators, labels, add_position_index=add_position_index)

    logger.info(f"     → X={X.shape}, Y={Y.shape}")

    return X, Y


# Note: split_sequences remplacée par split_sequences_chronological importée de data_utils
# La nouvelle fonction utilise un GAP entre train/val/test pour éviter le data leakage
# causé par l'overlap des séquences sliding window.
#
# Alias pour compatibilité (sera supprimé dans une version future)
split_sequences = split_sequences_chronological


def prepare_and_save(filter_type: str = LABEL_FILTER_TYPE,
                     assets: list = None,
                     output_path: str = None,
                     add_position_index: bool = False) -> str:
    """
    Prépare les données 5min et les sauvegarde en format numpy.

    Args:
        filter_type: 'decycler' ou 'kalman'
        assets: Liste des assets à utiliser (défaut: DEFAULT_ASSETS)
        output_path: Chemin de sortie (défaut: auto-généré)
        add_position_index: Si True, ajoute Position Index (4 features au lieu de 3)

    Returns:
        Chemin du fichier sauvegardé
    """
    # Utiliser les assets par défaut si non spécifiés
    if assets is None:
        assets = DEFAULT_ASSETS

    # Valider les assets demandés
    invalid_assets = [a for a in assets if a not in AVAILABLE_ASSETS_5M]
    if invalid_assets:
        raise ValueError(f"Assets invalides: {invalid_assets}. "
                        f"Disponibles: {list(AVAILABLE_ASSETS_5M.keys())}")

    n_features = 4 if add_position_index else 3

    logger.info("="*80)
    logger.info("PRÉPARATION DES DONNÉES - 5min")
    logger.info("="*80)
    logger.info(f"📊 Timeframe: 5 minutes")
    logger.info(f"💰 Assets: {', '.join(assets)}")
    logger.info(f"🔧 Filtre: {filter_type}")
    logger.info(f"📈 Features: {n_features} ({'RSI, CCI, MACD + Position Index' if add_position_index else 'RSI, CCI, MACD'})")

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
    # 2. Calculer indicateurs + labels pour chaque asset
    # =========================================================================
    logger.info(f"\n2. Calcul indicateurs et labels par asset...")

    prepared_assets = {}
    for asset_name, df in asset_data.items():
        X, Y = prepare_single_asset(df, filter_type, asset_name,
                                    add_position_index=add_position_index)
        prepared_assets[asset_name] = (X, Y)

    # =========================================================================
    # 3. Split chronologique avec GAP pour chaque asset
    # =========================================================================
    logger.info(f"\n3. Split chronologique avec GAP...")

    split_data = {}
    for asset_name, (X, Y) in prepared_assets.items():
        (X_train_a, Y_train_a), (X_val_a, Y_val_a), (X_test_a, Y_test_a) = \
            split_sequences(X, Y)
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
        pos_suffix = '_posidx' if add_position_index else ''
        output_path = f"data/prepared/dataset_{assets_str}_5min_{filter_type}{pos_suffix}.npz"

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Métadonnées
    metadata = {
        'created_at': datetime.now().isoformat(),
        'assets': assets,
        'n_assets': len(assets),
        'timeframe': '5min',
        'filter_type': filter_type,
        'total_candles': total_candles,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'sequence_length': SEQUENCE_LENGTH,
        'num_indicators': NUM_INDICATORS,
        'num_features': n_features,
        'add_position_index': add_position_index,
        'feature_description': 'RSI, CCI, MACD, Position Index (1/12→1.0)' if add_position_index else 'RSI, CCI, MACD',
        'indicator_params': {
            'rsi_period': RSI_PERIOD,
            'cci_period': CCI_PERIOD,
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

    # Parser les métadonnées (dans npz ou fichier JSON séparé)
    if 'metadata' in data.files:
        # Extraire depuis le numpy array
        metadata = data['metadata']
        if isinstance(metadata, np.ndarray):
            metadata = metadata.item()

        # Parser JSON si c'est une string, sinon utiliser directement
        if isinstance(metadata, (str, bytes)):
            metadata = json.loads(metadata)
        # Si c'est déjà un dict, on l'utilise tel quel
    else:
        # Fallback: chercher le fichier JSON à côté
        metadata_path = Path(str(path).replace('.npz', '_metadata.json'))
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            logger.info(f"  ℹ️ Metadata chargé depuis: {metadata_path}")
        else:
            # Metadata minimal par défaut
            metadata = {
                'n_features': data['X_train'].shape[2],
                'train_size': len(data['X_train']),
                'val_size': len(data['X_val']),
                'test_size': len(data['X_test']),
            }
            logger.warning(f"  ⚠️ Pas de metadata trouvé, utilisation de valeurs par défaut")

    # Vérifier si les transitions sont présentes (Phase 2.11 - Weighted Loss)
    has_transitions = 'T_train' in data.files

    # =========================================================================
    # DÉTECTION FORMAT DIRECTION-ONLY
    # =========================================================================
    # Nouveau format: X, Y et T ont 3 colonnes avec [timestamp, asset_id, data]
    # X: (n, 25, 3) avec colonne 2 = feature (c_ret)
    # Y: (n, 3) avec colonne 2 = label
    # T: (n, 3) avec colonne 2 = is_transition
    # Ancien format: Y et T ont 1-2 colonnes [label] ou [label1, label2]
    is_direction_only = False
    if data['Y_train'].ndim == 2 and data['Y_train'].shape[1] == 3:
        is_direction_only = True
        logger.info(f"  🎯 Format Direction-Only détecté (Y shape: {data['Y_train'].shape})")

        # Extraire les colonnes features (à partir de colonne 2) de X
        # NOTE: MACD/RSI ont 1 feature, CCI a 3 features (h_ret, l_ret, c_ret)
        X_train = data['X_train'][:, :, 2:]  # (n, 25, n_features+2) → (n, 25, n_features)
        X_val = data['X_val'][:, :, 2:]
        X_test = data['X_test'][:, :, 2:]

        # Extraire seulement la colonne label (colonne 2) de Y
        Y_train = data['Y_train'][:, 2:3]  # (n, 1)
        Y_val = data['Y_val'][:, 2:3]
        Y_test = data['Y_test'][:, 2:3]

        if has_transitions:
            # T a aussi 3 colonnes, extraire colonne 2
            T_train = data['T_train'][:, 2:3]  # (n, 1)
            T_val = data['T_val'][:, 2:3]
            T_test = data['T_test'][:, 2:3]

        logger.info(f"     → Extraction colonne feature (colonne 2):")
        logger.info(f"     Train: X={X_train.shape}, Y={Y_train.shape}, T={T_train.shape if has_transitions else 'N/A'}")
        logger.info(f"     Val:   X={X_val.shape}, Y={Y_val.shape}, T={T_val.shape if has_transitions else 'N/A'}")
        logger.info(f"     Test:  X={X_test.shape}, Y={Y_test.shape}, T={T_test.shape if has_transitions else 'N/A'}")
    else:
        # Ancien format - utiliser tel quel
        X_train = data['X_train']
        X_val = data['X_val']
        X_test = data['X_test']

        Y_train = data['Y_train']
        Y_val = data['Y_val']
        Y_test = data['Y_test']

        if has_transitions:
            T_train = data['T_train']
            T_val = data['T_val']
            T_test = data['T_test']

    # =========================================================================
    # CONSTRUIRE RÉSULTAT
    # =========================================================================
    if has_transitions:
        # Inclure les transitions dans les tuples
        result = {
            'train': (X_train, Y_train, T_train),
            'val': (X_val, Y_val, T_val),
            'test': (X_test, Y_test, T_test),
            'metadata': metadata
        }
        logger.info(f"  ✅ Données chargées (avec transitions):")
        logger.info(f"     Train: X={X_train.shape}, Y={Y_train.shape}, T={T_train.shape}")
        logger.info(f"     Val:   X={X_val.shape}, Y={Y_val.shape}, T={T_val.shape}")
        logger.info(f"     Test:  X={X_test.shape}, Y={Y_test.shape}, T={T_test.shape}")

        # Stats transitions (prendre colonne 0 si (n,1), sinon moyenne)
        trans_train_pct = (T_train[:, 0].mean() if T_train.shape[1] == 1 else T_train.mean()) * 100
        trans_val_pct = (T_val[:, 0].mean() if T_val.shape[1] == 1 else T_val.mean()) * 100
        trans_test_pct = (T_test[:, 0].mean() if T_test.shape[1] == 1 else T_test.mean()) * 100
        logger.info(f"     Transitions: Train {trans_train_pct:.1f}%, Val {trans_val_pct:.1f}%, Test {trans_test_pct:.1f}%")
    else:
        # Backward compatibility - sans transitions
        result = {
            'train': (X_train, Y_train),
            'val': (X_val, Y_val),
            'test': (X_test, Y_test),
            'metadata': metadata
        }
        logger.info(f"  ✅ Données chargées:")
        logger.info(f"     Train: X={X_train.shape}, Y={Y_train.shape}")
        logger.info(f"     Val:   X={X_val.shape}, Y={Y_val.shape}")
        logger.info(f"     Test:  X={X_test.shape}, Y={Y_test.shape}")

    log_dataset_metadata(metadata, logger)

    return result


def filter_by_assets(X: np.ndarray, Y: np.ndarray, T: np.ndarray, OHLCV: np.ndarray,
                     assets: list, metadata: dict) -> tuple:
    """
    Filtre les données pour ne garder que les assets spécifiés.

    Args:
        X: Features (n, seq_length, n_features+2) - colonnes 0,1 = timestamp, asset_id
        Y: Labels (n, 1) - après extraction de la colonne 2
        T: Transitions (n, 1) - après extraction de la colonne 2 (ou None)
        OHLCV: (n, 7) - colonne 1 = asset_id
        assets: Liste des assets à garder (ex: ['BTC', 'ETH'])
        metadata: Métadonnées du dataset

    Returns:
        (X_filtered, Y_filtered, T_filtered, OHLCV_filtered)
    """
    # Créer mapping asset_name → asset_id (0-indexed comme dans le dataset!)
    # L'ordre doit correspondre à celui de prepare_data_direction_only.py
    if 'assets' in metadata:
        all_assets = metadata['assets']
        asset_id_map = {name: idx for idx, name in enumerate(all_assets, start=0)}
    else:
        # Fallback: ordre par défaut (0-indexed)
        default_order = ['BTC', 'ETH', 'BNB', 'ADA', 'LTC']
        asset_id_map = {name: idx for idx, name in enumerate(default_order, start=0)}

    # Convertir assets en asset_ids
    asset_ids = []
    for asset in assets:
        if asset in asset_id_map:
            asset_ids.append(float(asset_id_map[asset]))
        else:
            logger.warning(f"⚠️  Asset {asset} non trouvé dans metadata, ignoré")

    if not asset_ids:
        raise ValueError(f"Aucun asset valide trouvé dans {assets}")

    logger.info(f"  🎯 Filtrage pour assets: {assets}")
    logger.info(f"     Asset IDs: {asset_ids}")

    # Filtrer par asset_id depuis OHLCV (colonne 1)
    # Note: X peut avoir seulement 1 feature après extraction Direction-Only,
    # donc on utilise OHLCV[:, 1] qui contient toujours l'asset_id
    asset_mask = np.isin(OHLCV[:, 1], asset_ids)

    X_filtered = X[asset_mask]
    Y_filtered = Y[asset_mask]
    T_filtered = T[asset_mask] if T is not None else None
    OHLCV_filtered = OHLCV[asset_mask]

    logger.info(f"     Avant filtrage: {len(X)} séquences")
    logger.info(f"     Après filtrage: {len(X_filtered)} séquences ({len(X_filtered)/len(X)*100:.1f}%)")

    return X_filtered, Y_filtered, T_filtered, OHLCV_filtered


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
    available_assets = list(AVAILABLE_ASSETS_5M.keys())

    parser = argparse.ArgumentParser(
        description="Prépare et sauvegarde les datasets 5min pour l'entraînement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Exemples:
  # Avec tous les assets (3 features: RSI, CCI, MACD)
  python src/prepare_data.py --filter kalman --assets BTC ETH BNB ADA LTC

  # Avec Position Index (4 features: RSI, CCI, MACD, PosIdx)
  python src/prepare_data.py --filter kalman --assets BTC ETH BNB ADA LTC --position-index

  # Avec BTC et ETH seulement (défaut)
  python src/prepare_data.py --filter kalman

  # Lister les datasets disponibles
  python src/prepare_data.py --list

Assets disponibles: {', '.join(available_assets)}

Position Index:
  Ajoute une colonne indiquant la position dans la séquence (1/12 → 12/12).
  Permet au modèle de pondérer différemment les données selon leur position.
  Hypothèse: les dernières valeurs sont plus importantes pour la prédiction.
        """
    )

    parser.add_argument('--assets', '-a', type=str, nargs='+',
                        default=DEFAULT_ASSETS,
                        choices=available_assets,
                        help=f'Assets à inclure (défaut: {DEFAULT_ASSETS})')
    parser.add_argument('--filter', '-f', type=str, default=LABEL_FILTER_TYPE,
                        choices=['decycler', 'kalman'], help='Filtre pour les labels')
    parser.add_argument('--position-index', '-p', action='store_true',
                        help='Ajoute Position Index (4 features: RSI, CCI, MACD, PosIdx)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Chemin de sortie (défaut: auto-généré)')
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
        filter_type=args.filter,
        assets=args.assets,
        output_path=args.output,
        add_position_index=args.position_index
    )

    print(f"\n🎉 Terminé! Dataset prêt: {output_path}")
    print(f"\nPour entraîner:")
    print(f"  python src/train.py --data {output_path}")


if __name__ == '__main__':
    main()
