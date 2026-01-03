"""
Préparation des données avec labels 30min.

Objectif: Prédire la pente des indicateurs 30min (moins de bruit)
à partir des indicateurs 5min (haute résolution).

Indicateurs: RSI, CCI, MACD (3 indicateurs)
Note: BOL (Bollinger) retiré car impossible à synchroniser (toujours lag +1).

Datasets générés:
    1. dataset_5min_labels30min.npz: X=5min(3) + step_index(1) = 4 features, Y=30min slopes
    2. dataset_5min_30min_labels30min.npz: X=5min(3) + 30min(3) + step_index(1) = 7 features

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
    AVAILABLE_ASSETS_5M, DEFAULT_ASSETS,
    TRIM_EDGES,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    LABEL_FILTER_TYPE,
    SEQUENCE_LENGTH,
    RSI_PERIOD, CCI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    DECYCLER_CUTOFF, KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR
)
from data_utils import load_crypto_data, trim_edges, split_sequences_chronological
from indicators import (
    calculate_all_indicators_for_model,
    generate_all_labels,
    create_sequences
)
from utils import resample_to_timeframe

# Alias pour compatibilité
split_sequences = split_sequences_chronological


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
        # Utiliser bfill pour remplir les NaN au début avec la première valeur valide
        df_aligned = df_aligned.bfill()

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
        include_30min_features: Si True, ajoute les indicateurs 30min en features (7 total)

    Returns:
        (X, Y) où:
            - X shape=(n_sequences, 12, 4 ou 7)  # 3 indicators + step_index, or 6 + step_index
            - Y shape=(n_sequences, 3)  # RSI, CCI, MACD (BOL retiré)
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

    # Stats des labels (3 indicateurs: RSI, CCI, MACD - BOL retiré)
    for i, name in enumerate(['RSI', 'CCI', 'MACD']):
        buy_pct = labels_30min[:, i].sum() / len(labels_30min) * 100
        logger.info(f"     {name}: {buy_pct:.1f}% BUY")

    # =========================================================================
    # 5b. Trim des bords APRÈS Kalman (effets de bord du filtre)
    # =========================================================================
    # Le filtre Kalman forward-backward a des effets de bord au début et à la fin.
    # On supprime quelques périodes 30min pour éviter ces artifacts.
    #
    KALMAN_TRIM = 10  # 10 périodes 30min = 5 heures de chaque côté
    logger.info(f"\n  ✂️ Trim post-Kalman: {KALMAN_TRIM} périodes 30min de chaque côté...")
    labels_30min = labels_30min[KALMAN_TRIM:-KALMAN_TRIM]
    indicators_30min = indicators_30min[KALMAN_TRIM:-KALMAN_TRIM]
    index_30min = index_30min[KALMAN_TRIM:-KALMAN_TRIM]
    logger.info(f"     → Shape après trim: labels={labels_30min.shape}, indicators={indicators_30min.shape}")

    # =========================================================================
    # 6. CORRECTION: Shift des labels pour synchronisation
    # =========================================================================
    # PROBLÈME INITIAL:
    #   generate_labels() définit Label[t] = pente(t-2 → t-1)
    #   Donc labels[10:00] = pente(09:00 → 09:30) = 1h de retard!
    #
    # SOLUTION: shift(-1) pour que labels[10:00] = pente(09:30 → 10:00)
    #   Ainsi les 5min de 10:00-10:25 prédisent la pente qui vient de clore.
    #
    # NOTE: On utilise slicing au lieu de np.roll pour éviter le wrap-around
    #   np.roll ramènerait le premier élément à la fin (donnée invalide)
    #
    logger.info(f"\n  🔄 Correction du décalage labels (shift -1)...")
    labels_30min_shifted = labels_30min[1:]  # Décaler: index 0 reçoit valeur de index 1
    index_30min_for_labels = index_30min[:-1]  # Index raccourci pour labels uniquement
    logger.info(f"     → Labels décalés de -1 période 30min")
    logger.info(f"     → Shape après shift: {labels_30min_shifted.shape}")

    # =========================================================================
    # 7. Aligner labels 30min sur timestamps 5min (FORWARD-FILL)
    # =========================================================================
    logger.info(f"\n  🔄 Alignement labels 30min → 5min (forward-fill)...")
    labels_aligned = align_30min_to_5min(labels_30min_shifted, index_30min_for_labels, index_5min)
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
    # 8. (Optionnel) Aligner indicateurs 30min sur 5min pour features
    # =========================================================================
    if include_30min_features:
        logger.info(f"\n  🔄 Alignement indicateurs 30min → 5min (features)...")

        # CORRECTION: Décaler l'index de 30min pour que l'indicateur soit "disponible"
        # seulement après la clôture de la bougie 30min.
        #
        # AVANT: À 5min 10:00, on utilisait 30min de 10:00 (données 10:00-10:29 = FUTUR)
        # APRÈS: À 5min 10:00, on utilise 30min de 09:30 (dernière bougie complète)
        #
        # En live trading, l'indicateur 30min à 09:30 n'est disponible qu'à 10:00
        # (quand la bougie 09:30-09:59 se ferme).
        #
        index_30min_shifted = index_30min + pd.Timedelta('30min')
        logger.info(f"     → Index 30min décalé de +30min (causal)")

        indicators_30min_aligned = align_30min_to_5min(indicators_30min, index_30min_shifted, index_5min)

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
    # 8b. Ajouter Step Index (position dans la fenêtre 30min)
    # =========================================================================
    # Le step_index indique la position de la bougie 5min dans sa période 30min:
    #   - Minute 00 → step 1
    #   - Minute 05 → step 2
    #   - Minute 10 → step 3
    #   - Minute 15 → step 4
    #   - Minute 20 → step 5
    #   - Minute 25 → step 6
    #
    # Cela donne au modèle une "horloge interne" pour savoir où il en est
    # dans la construction de la bougie 30min.
    logger.info(f"\n  ⏱️ Ajout du Step Index (position dans fenêtre 30min)...")

    # Calculer le step_index (1-6) basé sur les minutes
    minutes = index_5min.minute
    step_index = (minutes % 30) // 5 + 1  # Résultat: 1, 2, 3, 4, 5, 6

    # Normaliser entre 0 et 1 pour cohérence avec les autres features
    step_index_normalized = (step_index - 1) / 5.0  # Résultat: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0

    # Ajouter comme colonne supplémentaire
    step_index_col = step_index_normalized.values.reshape(-1, 1)
    indicators_combined = np.hstack([indicators_combined, step_index_col])

    logger.info(f"     → Step index ajouté (normalisé 0-1)")
    logger.info(f"     → Features finales: {indicators_combined.shape}")

    # =========================================================================
    # 9. Vérification finale des dimensions
    # =========================================================================
    assert len(indicators_combined) == len(labels_aligned), \
        f"Mismatch: indicators={len(indicators_combined)}, labels={len(labels_aligned)}"

    logger.info(f"\n  ✅ Données alignées: {len(indicators_combined)} samples")

    # =========================================================================
    # 10. Créer séquences
    # =========================================================================
    logger.info(f"\n  📦 Création séquences (length={SEQUENCE_LENGTH})...")
    X, Y = create_sequences(indicators_combined, labels_aligned, sequence_length=SEQUENCE_LENGTH)
    logger.info(f"     → X={X.shape}, Y={Y.shape}")

    return X, Y


def prepare_and_save_30min(filter_type: str = LABEL_FILTER_TYPE,
                           include_30min_features: bool = False,
                           assets: list = None,
                           output_path: str = None) -> str:
    """
    Prépare les datasets avec labels 30min et les sauvegarde.

    Args:
        filter_type: 'decycler' ou 'kalman'
        include_30min_features: Si True, X a 7 features (5min+30min+step_index)
        assets: Liste des assets à utiliser (défaut: DEFAULT_ASSETS)
        output_path: Chemin de sortie (défaut: auto-généré)

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

    logger.info("="*80)
    logger.info("PRÉPARATION DONNÉES AVEC LABELS 30MIN")
    logger.info("="*80)

    feature_type = "5min_30min" if include_30min_features else "5min"
    logger.info(f"📊 Features: {feature_type}")
    logger.info(f"🏷️ Labels: pente indicateurs 30min")
    logger.info(f"🔧 Filtre: {filter_type}")
    logger.info(f"💰 Assets: {', '.join(assets)}")

    # =========================================================================
    # 1. Charger données 5min pour chaque asset
    # =========================================================================
    logger.info(f"\n1. Chargement données 5min...")

    asset_data = {}
    for asset_name in assets:
        file_path = AVAILABLE_ASSETS_5M[asset_name]
        df = load_crypto_data(file_path, asset_name=f'{asset_name}-5m')
        df_trimmed = trim_edges(df, trim_start=TRIM_EDGES, trim_end=TRIM_EDGES)
        asset_data[asset_name] = df_trimmed
        logger.info(f"   {asset_name}: {len(df_trimmed):,} bougies")

    # =========================================================================
    # 2. Préparer chaque asset
    # =========================================================================
    logger.info(f"\n2. Préparation par asset...")

    prepared_assets = {}
    for asset_name, df in asset_data.items():
        X, Y = prepare_single_asset_30min(
            df, filter_type, asset_name, include_30min_features
        )
        prepared_assets[asset_name] = (X, Y)

    # =========================================================================
    # 3. Split chronologique avec GAP (évite data leakage)
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

    n_features = X_train.shape[2]
    # Features: 3 (5min) + 3 (30min optionnel) + 1 (step_index) = 4 ou 7
    # Note: BOL retiré, maintenant 3 indicateurs (RSI, CCI, MACD)
    if n_features == 7:
        feature_desc = "5min(3) + 30min(3) + step_index(1)"
    elif n_features == 4:
        feature_desc = "5min(3) + step_index(1)"
    else:
        feature_desc = f"{n_features} features"
    logger.info(f"\n📈 Features: {n_features} ({feature_desc})")

    # =========================================================================
    # 5. Sauvegarder
    # =========================================================================
    if output_path is None:
        assets_str = '_'.join(assets).lower()
        output_path = f"data/prepared/dataset_{assets_str}_{feature_type}_labels30min_{filter_type}.npz"

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Métadonnées
    metadata = {
        'created_at': datetime.now().isoformat(),
        'assets': assets,
        'n_assets': len(assets),
        'feature_timeframe': feature_type,
        'label_timeframe': '30min',
        'filter_type': filter_type,
        'n_features': n_features,
        'feature_description': feature_desc,
        'includes_step_index': True,
        'step_index_info': 'Position dans fenêtre 30min (1-6), normalisé 0-1',
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'sequence_length': SEQUENCE_LENGTH,
        'indicator_params': {
            'rsi_period': RSI_PERIOD,
            'cci_period': CCI_PERIOD,
            'macd_fast': MACD_FAST,
            'macd_slow': MACD_SLOW,
            'macd_signal': MACD_SIGNAL
            # Note: BOL retiré (non synchronisable)
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
        'gap_size': SEQUENCE_LENGTH,
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
    available_assets = list(AVAILABLE_ASSETS_5M.keys())

    parser = argparse.ArgumentParser(
        description="Prépare les datasets avec labels 30min",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Exemples:
  # Dataset avec BTC et ETH (défaut)
  python src/prepare_data_30min.py --filter kalman --include-30min-features

  # Dataset avec tous les assets disponibles
  python src/prepare_data_30min.py --assets {' '.join(available_assets)} --include-30min-features

  # Dataset avec assets spécifiques
  python src/prepare_data_30min.py --assets BTC ETH BNB --include-30min-features

Assets disponibles: {', '.join(available_assets)}

Description:
  Prédire la pente des indicateurs 30min (moins de bruit) à partir
  des indicateurs 5min (haute résolution).

  Labels 30min = signal plus stable, meilleure prédictibilité.
        """
    )

    parser.add_argument('--assets', '-a', type=str, nargs='+',
                        default=DEFAULT_ASSETS,
                        choices=available_assets,
                        help=f'Assets à inclure (défaut: {DEFAULT_ASSETS})')
    parser.add_argument('--filter', '-f', type=str, default=LABEL_FILTER_TYPE,
                        choices=['decycler', 'kalman'], help='Filtre pour les labels')
    parser.add_argument('--include-30min-features', action='store_true',
                        help='Inclure les indicateurs 30min en features (+3 features)')
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
        assets=args.assets,
        output_path=args.output
    )

    print(f"\n🎉 Terminé! Dataset prêt: {output_path}")
    print(f"\nPour entraîner:")
    print(f"  python src/train.py --data {output_path}")


if __name__ == '__main__':
    main()
