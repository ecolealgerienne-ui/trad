"""
Préparation des données avec labels 30min.

Objectif: Prédire la pente des indicateurs 30min (moins de bruit)
à partir des indicateurs 5min (haute résolution).

Datasets générés:
    1. dataset_5min_labels30min.npz: X=5min(4 features), Y=30min slopes
    2. dataset_5min_30min_labels30min.npz: X=5min+30min(8 features), Y=30min slopes

SYNCHRONISATION CRITIQUE:
    - Pour chaque bougie 5min à temps T:
    - On utilise le label 30min de la dernière période COMPLÈTE
    - Exemple: 5min à 10:25 → label 30min de 10:00 (pas 10:30)
    - Réalisé via forward-fill (ffill)
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
    BTC_DATA_FILE_5M, ETH_DATA_FILE_5M,
    TRIM_EDGES,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    LABEL_FILTER_TYPE,
    SEQUENCE_LENGTH,
    RSI_PERIOD, CCI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BOL_PERIOD, BOL_NUM_STD,
    DECYCLER_CUTOFF, KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR
)
from data_utils import load_crypto_data, trim_edges
from indicators import (
    calculate_all_indicators_for_model,
    generate_all_labels,
    create_sequences
)
from utils import resample_to_timeframe
from prepare_data import split_sequences


def resample_5min_to_30min(df_5min: pd.DataFrame) -> pd.DataFrame:
    """
    Resample les données 5min vers 30min.

    IMPORTANT: Le timestamp doit être en index datetime pour le resampling.

    Args:
        df_5min: DataFrame 5min avec colonnes OHLCV et timestamp

    Returns:
        DataFrame 30min avec timestamp en index datetime
    """
    df = df_5min.copy()

    # S'assurer que timestamp est datetime et en index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    # Vérifier que l'index est datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("L'index doit être un DatetimeIndex pour le resampling")

    # Agrégation OHLCV
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }

    if 'volume' in df.columns:
        agg_dict['volume'] = 'sum'

    # Resample - closed='left' signifie que 10:00-10:29 → bougie 10:00
    # label='left' signifie que le timestamp de la bougie est 10:00
    df_30min = df.resample('30min', closed='left', label='left').agg(agg_dict)
    df_30min = df_30min.dropna()

    logger.info(f"  Resample 5min→30min: {len(df_5min)} → {len(df_30min)} bougies")

    return df_30min


def align_30min_to_5min(data_30min: np.ndarray,
                        index_30min: pd.DatetimeIndex,
                        index_5min: pd.DatetimeIndex) -> np.ndarray:
    """
    Aligne les données 30min sur les timestamps 5min via forward-fill.

    SYNCHRONISATION:
        - Pour chaque timestamp 5min, on prend la valeur 30min la plus récente
        - Exemple: 5min à 10:25 → valeur 30min de 10:00
        - C'est exactement ce que fait reindex avec method='ffill'

    Args:
        data_30min: Array de données 30min (peut être 1D ou 2D)
        index_30min: Index datetime des données 30min
        index_5min: Index datetime cible (5min)

    Returns:
        Array aligné sur index_5min
    """
    # Créer un DataFrame pour utiliser reindex
    if len(data_30min.shape) == 1:
        df_30min = pd.DataFrame(data_30min, index=index_30min, columns=['value'])
    else:
        df_30min = pd.DataFrame(data_30min, index=index_30min)

    # Reindex avec forward-fill (prend la dernière valeur connue)
    df_aligned = df_30min.reindex(index_5min, method='ffill')

    # Vérifier qu'il n'y a pas de NaN au début (avant la première valeur 30min)
    n_nan = df_aligned.isna().sum().sum()
    if n_nan > 0:
        logger.warning(f"  ⚠️ {n_nan} NaN après alignement (début avant première bougie 30min)")
        # Supprimer les lignes avec NaN
        df_aligned = df_aligned.dropna()

    return df_aligned.values


def prepare_single_asset_30min(df_5min: pd.DataFrame,
                                filter_type: str,
                                asset_name: str = "Asset",
                                include_30min_features: bool = False):
    """
    Prépare les données pour UN asset avec labels 30min.

    Process:
        1. Resample 5min → 30min
        2. Calculer indicateurs 5min (features)
        3. Calculer indicateurs 30min
        4. Générer labels depuis indicateurs 30min (pente)
        5. Aligner labels 30min sur timestamps 5min
        6. (Optionnel) Aligner indicateurs 30min sur 5min
        7. Créer séquences

    Args:
        df_5min: DataFrame 5min avec OHLCV
        filter_type: 'kalman' ou 'decycler'
        asset_name: Nom pour les logs
        include_30min_features: Si True, ajoute les indicateurs 30min en features (8 total)

    Returns:
        (X, Y) où:
            - X shape=(n_sequences, 12, 4 ou 8)
            - Y shape=(n_sequences, 4)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"📈 {asset_name}: Préparation avec labels 30min")
    logger.info(f"{'='*60}")

    # =========================================================================
    # 1. Préparer les index datetime
    # =========================================================================
    df = df_5min.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    index_5min = df.index
    logger.info(f"  Données 5min: {len(df)} bougies")
    logger.info(f"  Période: {index_5min[0]} → {index_5min[-1]}")

    # =========================================================================
    # 2. Resample vers 30min
    # =========================================================================
    df_30min = resample_5min_to_30min(df_5min)
    index_30min = df_30min.index
    logger.info(f"  Données 30min: {len(df_30min)} bougies")

    # =========================================================================
    # 3. Calculer indicateurs 5min (FEATURES)
    # =========================================================================
    logger.info(f"\n  📊 Calcul indicateurs 5min (features)...")
    # Besoin de remettre timestamp en colonne pour calculate_all_indicators_for_model
    df_5min_with_ts = df.reset_index()
    indicators_5min = calculate_all_indicators_for_model(df_5min_with_ts)
    logger.info(f"     → Shape: {indicators_5min.shape}")

    # =========================================================================
    # 4. Calculer indicateurs 30min
    # =========================================================================
    logger.info(f"\n  📊 Calcul indicateurs 30min...")
    df_30min_with_ts = df_30min.reset_index()
    indicators_30min = calculate_all_indicators_for_model(df_30min_with_ts)
    logger.info(f"     → Shape: {indicators_30min.shape}")

    # =========================================================================
    # 5. Générer labels depuis indicateurs 30min (PENTE)
    # =========================================================================
    logger.info(f"\n  🏷️ Génération labels (pente indicateurs 30min, filtre {filter_type})...")
    labels_30min = generate_all_labels(indicators_30min, filter_type=filter_type)
    logger.info(f"     → Shape: {labels_30min.shape}")

    # Stats des labels
    for i, name in enumerate(['RSI', 'CCI', 'BOL', 'MACD']):
        buy_pct = labels_30min[:, i].sum() / len(labels_30min) * 100
        logger.info(f"     {name}: {buy_pct:.1f}% BUY")

    # =========================================================================
    # 6. Aligner labels 30min sur timestamps 5min (FORWARD-FILL)
    # =========================================================================
    logger.info(f"\n  🔄 Alignement labels 30min → 5min (forward-fill)...")
    labels_aligned = align_30min_to_5min(labels_30min, index_30min, index_5min)
    logger.info(f"     → Shape après alignement: {labels_aligned.shape}")

    # Vérifier la synchronisation
    # Le nombre de lignes peut être réduit si les premiers timestamps 5min
    # sont avant la première bougie 30min complète
    n_valid = len(labels_aligned)
    if n_valid < len(indicators_5min):
        # Couper indicators_5min pour matcher
        start_idx = len(indicators_5min) - n_valid
        indicators_5min = indicators_5min[start_idx:]
        index_5min = index_5min[start_idx:]
        logger.info(f"     ⚠️ Coupé {start_idx} premières lignes 5min (avant première bougie 30min)")

    # =========================================================================
    # 7. (Optionnel) Aligner indicateurs 30min sur 5min pour features
    # =========================================================================
    if include_30min_features:
        logger.info(f"\n  🔄 Alignement indicateurs 30min → 5min (features)...")
        indicators_30min_aligned = align_30min_to_5min(indicators_30min, index_30min, index_5min)

        # Vérifier les dimensions
        if len(indicators_30min_aligned) != len(indicators_5min):
            raise ValueError(f"Mismatch: indicators_5min={len(indicators_5min)}, "
                           f"indicators_30min_aligned={len(indicators_30min_aligned)}")

        # Concatener: [5min_features, 30min_features] → 8 features
        indicators_combined = np.hstack([indicators_5min, indicators_30min_aligned])
        logger.info(f"     → Features combinées: {indicators_combined.shape} (5min + 30min)")
    else:
        indicators_combined = indicators_5min
        logger.info(f"     → Features: {indicators_combined.shape} (5min seulement)")

    # =========================================================================
    # 8. Vérification finale des dimensions
    # =========================================================================
    assert len(indicators_combined) == len(labels_aligned), \
        f"Mismatch: indicators={len(indicators_combined)}, labels={len(labels_aligned)}"

    logger.info(f"\n  ✅ Données alignées: {len(indicators_combined)} samples")

    # =========================================================================
    # 9. Créer séquences
    # =========================================================================
    logger.info(f"\n  📦 Création séquences (length={SEQUENCE_LENGTH})...")
    X, Y = create_sequences(indicators_combined, labels_aligned, sequence_length=SEQUENCE_LENGTH)
    logger.info(f"     → X={X.shape}, Y={Y.shape}")

    return X, Y


def prepare_and_save_30min(filter_type: str = LABEL_FILTER_TYPE,
                           include_30min_features: bool = False,
                           output_path: str = None) -> str:
    """
    Prépare les datasets avec labels 30min et les sauvegarde.

    Args:
        filter_type: 'decycler' ou 'kalman'
        include_30min_features: Si True, X a 8 features (5min+30min)
        output_path: Chemin de sortie (défaut: auto-généré)

    Returns:
        Chemin du fichier sauvegardé
    """
    logger.info("="*80)
    logger.info("PRÉPARATION DONNÉES AVEC LABELS 30MIN")
    logger.info("="*80)

    feature_type = "5min_30min" if include_30min_features else "5min"
    logger.info(f"📊 Features: {feature_type}")
    logger.info(f"🏷️ Labels: pente indicateurs 30min")
    logger.info(f"🔧 Filtre: {filter_type}")

    # =========================================================================
    # 1. Charger données 5min
    # =========================================================================
    logger.info(f"\n1. Chargement données 5min...")

    btc_5m = load_crypto_data(BTC_DATA_FILE_5M, asset_name='BTC-5m')
    eth_5m = load_crypto_data(ETH_DATA_FILE_5M, asset_name='ETH-5m')

    btc_5m_trimmed = trim_edges(btc_5m, trim_start=TRIM_EDGES, trim_end=TRIM_EDGES)
    eth_5m_trimmed = trim_edges(eth_5m, trim_start=TRIM_EDGES, trim_end=TRIM_EDGES)

    logger.info(f"   BTC: {len(btc_5m_trimmed):,} bougies")
    logger.info(f"   ETH: {len(eth_5m_trimmed):,} bougies")

    # =========================================================================
    # 2. Préparer chaque asset
    # =========================================================================
    logger.info(f"\n2. Préparation par asset...")

    X_btc, Y_btc = prepare_single_asset_30min(
        btc_5m_trimmed, filter_type, "BTC", include_30min_features
    )
    X_eth, Y_eth = prepare_single_asset_30min(
        eth_5m_trimmed, filter_type, "ETH", include_30min_features
    )

    # =========================================================================
    # 3. Split des séquences (Test=fin, Val=échantillonné)
    # =========================================================================
    logger.info(f"\n3. Split des séquences...")

    (X_btc_train, Y_btc_train), (X_btc_val, Y_btc_val), (X_btc_test, Y_btc_test) = \
        split_sequences(X_btc, Y_btc)
    (X_eth_train, Y_eth_train), (X_eth_val, Y_eth_val), (X_eth_test, Y_eth_test) = \
        split_sequences(X_eth, Y_eth)

    logger.info(f"   BTC: Train={len(X_btc_train)}, Val={len(X_btc_val)}, Test={len(X_btc_test)}")
    logger.info(f"   ETH: Train={len(X_eth_train)}, Val={len(X_eth_val)}, Test={len(X_eth_test)}")

    # Concaténer les assets
    X_train = np.concatenate([X_btc_train, X_eth_train], axis=0)
    Y_train = np.concatenate([Y_btc_train, Y_eth_train], axis=0)
    X_val = np.concatenate([X_btc_val, X_eth_val], axis=0)
    Y_val = np.concatenate([Y_btc_val, Y_eth_val], axis=0)
    X_test = np.concatenate([X_btc_test, X_eth_test], axis=0)
    Y_test = np.concatenate([Y_btc_test, Y_eth_test], axis=0)

    # =========================================================================
    # 4. Afficher stats finales
    # =========================================================================
    logger.info(f"\n📊 Shapes des datasets:")
    logger.info(f"   Train: X={X_train.shape}, Y={Y_train.shape}")
    logger.info(f"   Val:   X={X_val.shape}, Y={Y_val.shape}")
    logger.info(f"   Test:  X={X_test.shape}, Y={Y_test.shape}")

    n_features = X_train.shape[2]
    logger.info(f"\n📈 Features: {n_features} ({'5min + 30min' if n_features == 8 else '5min seulement'})")

    # =========================================================================
    # 5. Sauvegarder
    # =========================================================================
    if output_path is None:
        output_path = f"data/prepared/dataset_{feature_type}_labels30min_{filter_type}.npz"

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Métadonnées
    metadata = {
        'created_at': datetime.now().isoformat(),
        'feature_timeframe': feature_type,
        'label_timeframe': '30min',
        'filter_type': filter_type,
        'n_features': n_features,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'sequence_length': SEQUENCE_LENGTH,
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
        },
        'description': f"Features {feature_type}, Labels = pente indicateurs 30min"
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


def main():
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Prépare les datasets avec labels 30min",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Dataset 1: X=5min(4 features), Y=30min slopes
  python src/prepare_data_30min.py --filter kalman

  # Dataset 2: X=5min+30min(8 features), Y=30min slopes
  python src/prepare_data_30min.py --filter kalman --include-30min-features

Description:
  Prédire la pente des indicateurs 30min (moins de bruit) à partir
  des indicateurs 5min (haute résolution).

  Labels 30min = signal plus stable, meilleure prédictibilité.
        """
    )

    parser.add_argument('--filter', '-f', type=str, default=LABEL_FILTER_TYPE,
                        choices=['decycler', 'kalman'], help='Filtre pour les labels')
    parser.add_argument('--include-30min-features', action='store_true',
                        help='Inclure les indicateurs 30min en features (8 total)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Chemin de sortie (défaut: auto-généré)')

    args = parser.parse_args()

    # Configurer logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    # Préparer et sauvegarder
    output_path = prepare_and_save_30min(
        filter_type=args.filter,
        include_30min_features=args.include_30min_features,
        output_path=args.output
    )

    print(f"\n🎉 Terminé! Dataset prêt: {output_path}")
    print(f"\nPour entraîner:")
    print(f"  python src/train.py --data {output_path}")


if __name__ == '__main__':
    main()
