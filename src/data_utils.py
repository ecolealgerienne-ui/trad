"""
Utilitaires de manipulation de données pour le projet IA trading.

Ce module contient les fonctions de chargement, préparation et split des données.

⚠️ RÈGLE CRITIQUE : Split TEMPOREL strict avec GAP pour éviter data leakage!

Fonctions principales:
    - load_crypto_data: Charge un fichier CSV de données crypto
    - trim_edges: Enlève les bords (warm-up + artifacts des filtres)
    - split_sequences_chronological: Split temporel avec GAP entre train/val/test
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

from constants import (
    BTC_DATA_FILE, ETH_DATA_FILE,
    BTC_CANDLES, ETH_CANDLES,
    TRIM_EDGES,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    RANDOM_SEED,
    SEQUENCE_LENGTH
)


def load_crypto_data(filepath, n_candles=None, asset_name='CRYPTO'):
    """
    Charge un fichier CSV de données crypto.

    Args:
        filepath : Chemin vers le fichier CSV
        n_candles : Nombre de bougies à charger (les dernières), None = toutes
        asset_name : Nom de l'actif (pour logs)

    Returns:
        DataFrame avec colonnes : timestamp, open, high, low, close, volume
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Fichier non trouvé : {filepath}")

    logger.info(f"📂 Chargement {asset_name} : {filepath}")

    # Charger CSV (essayer différents séparateurs)
    # D'abord essayer avec virgule (format standard)
    df = pd.read_csv(filepath)

    # Si le fichier utilise des point-virgules, on aura une seule colonne
    if len(df.columns) == 1 and ';' in df.columns[0]:
        df = pd.read_csv(filepath, sep=';')

    # Normaliser les noms de colonnes (majuscules → minuscules)
    df.columns = df.columns.str.lower()

    # Renommer colonnes si nécessaire
    column_mapping = {
        'date': 'timestamp',
        'time': 'timestamp'
    }
    df.rename(columns=column_mapping, inplace=True)

    # Vérifier colonnes requises (volume optionnel)
    required_cols = ['timestamp', 'open', 'high', 'low', 'close']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Colonnes manquantes : {missing_cols}")

    # Ajouter colonne volume si absente (avec valeur par défaut)
    if 'volume' not in df.columns:
        df['volume'] = 1.0  # Valeur par défaut (pas utilisée pour l'instant)
        logger.warning(f"  ⚠️ Colonne 'volume' absente, ajoutée avec valeur par défaut")

    # Convertir timestamp en datetime
    # Le timestamp peut être en millisecondes (epoch) ou format date
    try:
        # Essayer conversion depuis epoch (millisecondes)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    except:
        # Sinon parser comme date
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Prendre les dernières n bougies
    if n_candles is not None and len(df) > n_candles:
        df = df.tail(n_candles).reset_index(drop=True)
        logger.info(f"  → {len(df):,} dernières bougies chargées")
    else:
        logger.info(f"  → {len(df):,} bougies chargées")

    return df


def trim_edges(df, trim_start=TRIM_EDGES, trim_end=TRIM_EDGES):
    """
    Enlève les bords (warm-up + artifacts des filtres).

    Args:
        df : DataFrame
        trim_start : Nombre de bougies à enlever au début
        trim_end : Nombre de bougies à enlever à la fin

    Returns:
        DataFrame trimé
    """
    if len(df) <= trim_start + trim_end:
        raise ValueError(f"Dataset trop petit ({len(df)}) pour trim ({trim_start}+{trim_end})")

    df_trimmed = df.iloc[trim_start:-trim_end].reset_index(drop=True)

    logger.info(f"✂️ Trim edges : {len(df):,} → {len(df_trimmed):,} bougies")
    logger.info(f"  Enlevé : {trim_start} début + {trim_end} fin")

    return df_trimmed


# =============================================================================
# SPLIT TEMPOREL AVEC GAP (évite data leakage avec sliding window)
# =============================================================================

# Taille du GAP par défaut (en nombre de séquences)
# GAP = SEQUENCE_LENGTH évite tout chevauchement entre splits
DEFAULT_GAP_SIZE = SEQUENCE_LENGTH


def split_sequences_chronological(X, Y, train_ratio=TRAIN_SPLIT, val_ratio=VAL_SPLIT,
                                    test_ratio=TEST_SPLIT, gap_size=DEFAULT_GAP_SIZE,
                                    shuffle_train=True, random_seed=RANDOM_SEED):
    """
    Split temporel CHRONOLOGIQUE avec GAP entre train/val/test.

    ⚠️ CRITIQUE: Les séquences ont un overlap de (sequence_length - 1) entre elles.
    Un split aléatoire causerait du data leakage car les séquences adjacentes
    partagent des données.

    SOLUTION: Split chronologique strict avec GAP entre les splits.

    Structure:
        |<--- TRAIN --->|<- GAP ->|<--- VAL --->|<- GAP ->|<--- TEST --->|
                        ↑                       ↑
                    Pas d'overlap           Pas d'overlap

    Args:
        X: array de shape (n_sequences, seq_len, n_features)
        Y: array de shape (n_sequences, n_outputs)
        train_ratio: proportion pour train (défaut: 0.7)
        val_ratio: proportion pour validation (défaut: 0.15)
        test_ratio: proportion pour test (défaut: 0.15)
        gap_size: taille du GAP entre les splits (défaut: SEQUENCE_LENGTH)
        shuffle_train: Si True, shuffle le train APRÈS le split
        random_seed: seed pour reproductibilité du shuffle

    Returns:
        (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)

    Raises:
        ValueError: Si les ratios ne somment pas à 1.0 ou si le dataset est trop petit
    """
    # Vérifier les ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios doivent sommer à 1.0, got {total_ratio}")

    n_total = len(X)

    # Calculer les tailles avec GAP
    # Total utilisable = n_total - 2*gap_size (deux gaps entre 3 splits)
    n_usable = n_total - 2 * gap_size
    if n_usable <= 0:
        raise ValueError(f"Dataset trop petit ({n_total}) pour gap_size={gap_size}. "
                        f"Minimum requis: {2 * gap_size + 100}")

    # Calculer les tailles des splits
    n_test = int(n_usable * test_ratio)
    n_val = int(n_usable * val_ratio)
    n_train = n_usable - n_val - n_test

    # Calculer les indices de découpe
    # Structure: [TRAIN][GAP][VAL][GAP][TEST]
    train_end = n_train
    val_start = train_end + gap_size
    val_end = val_start + n_val
    test_start = val_end + gap_size
    # test_end = n_total (implicite)

    # Vérifier les indices
    if train_end <= 0:
        raise ValueError(f"train_end ({train_end}) <= 0, dataset trop petit")
    if val_start >= val_end:
        raise ValueError(f"val_start ({val_start}) >= val_end ({val_end})")
    if test_start >= n_total:
        raise ValueError(f"test_start ({test_start}) >= n_total ({n_total})")

    # Extraire les splits
    X_train, Y_train = X[:train_end], Y[:train_end]
    X_val, Y_val = X[val_start:val_end], Y[val_start:val_end]
    X_test, Y_test = X[test_start:], Y[test_start:]

    # Log les informations
    logger.info(f"📊 Split chronologique avec GAP={gap_size}:")
    logger.info(f"   Train: {len(X_train):,} séquences (indices 0:{train_end})")
    logger.info(f"   [GAP: {gap_size} séquences ignorées]")
    logger.info(f"   Val:   {len(X_val):,} séquences (indices {val_start}:{val_end})")
    logger.info(f"   [GAP: {gap_size} séquences ignorées]")
    logger.info(f"   Test:  {len(X_test):,} séquences (indices {test_start}:{n_total})")

    # Shuffle train si demandé (APRÈS le split!)
    if shuffle_train:
        np.random.seed(random_seed)
        train_indices = np.random.permutation(len(X_train))
        X_train = X_train[train_indices]
        Y_train = Y_train[train_indices]
        logger.info(f"   ✅ Train shuffled (seed={random_seed})")

    # Vérification finale
    total_used = len(X_train) + len(X_val) + len(X_test)
    total_gaps = 2 * gap_size
    assert total_used + total_gaps == n_total, \
        f"Split error: {total_used} + {total_gaps} gaps != {n_total}"

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


# =============================================================================
# FONCTIONS OBSOLÈTES (supprimées)
# =============================================================================
# Les fonctions suivantes ont été supprimées car elles causaient du data leakage:
#   - temporal_split: Utilisait un sampling aléatoire pour validation
#   - load_and_split_btc_eth: Concaténait BTC+ETH AVANT le split
#   - validate_no_leakage: Validation incorrecte
#
# Utilisez plutôt:
#   - split_sequences_chronological: Split temporel avec GAP
#   - prepare_single_asset (dans prepare_data.py): Calcul indicateurs par asset
# =============================================================================


# =============================================================================
# Exemple d'utilisation
# =============================================================================

if __name__ == '__main__':
    # Configurer logging
    logging.basicConfig(level=logging.INFO)

    # Test de la fonction split_sequences_chronological
    print("Test split_sequences_chronological:")
    print("="*60)

    # Créer des données de test
    n_samples = 1000
    seq_len = 12
    n_features = 4
    n_outputs = 4

    X = np.random.randn(n_samples, seq_len, n_features)
    Y = np.random.randint(0, 2, (n_samples, n_outputs))

    # Split
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = \
        split_sequences_chronological(X, Y, gap_size=12)

    print(f"\n📊 Résultat:")
    print(f"   Train: {X_train.shape}")
    print(f"   Val:   {X_val.shape}")
    print(f"   Test:  {X_test.shape}")

    # Vérifier pas d'overlap
    print(f"\n✅ Pas d'overlap entre les splits (GAP=12)")


