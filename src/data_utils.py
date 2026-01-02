"""
Utilitaires de manipulation de données pour le projet IA trading.

Ce module contient les fonctions de chargement, préparation et split des données.

⚠️ RÈGLE CRITIQUE : Split TEMPOREL strict pour éviter data leakage!
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
    RANDOM_SEED
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


def temporal_split(data, train_ratio=TRAIN_SPLIT, val_ratio=VAL_SPLIT, test_ratio=TEST_SPLIT,
                   shuffle_train=True, random_seed=RANDOM_SEED):
    """
    Split temporel: Test à la fin, Val échantillonné de partout.

    Stratégie:
    - TEST = toujours les données les plus récentes (fin du dataset)
    - VAL = échantillonné aléatoirement du reste (meilleure représentativité)
    - TRAIN = le reste

    Avantage: Val ne surfit pas à une période spécifique, meilleur pour
    un re-entraînement mensuel.

    Args:
        data : DataFrame de séries temporelles (ordre chronologique)
        train_ratio : Proportion pour train (défaut: 0.7)
        val_ratio : Proportion pour validation (défaut: 0.15)
        test_ratio : Proportion pour test (défaut: 0.15)
        shuffle_train : Si True, shuffle le train (APRÈS split)
        random_seed : Seed pour reproductibilité

    Returns:
        train, val, test : DataFrames splittés
    """
    # Vérifier ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Les ratios doivent sommer à 1.0 (actuellement: {total_ratio})")

    n_total = len(data)
    if n_total == 0:
        raise ValueError("Dataset vide")

    # 1. TEST = toujours à la fin (données les plus récentes)
    n_test = int(n_total * test_ratio)
    test_data = data.iloc[-n_test:].copy()
    remaining = data.iloc[:-n_test].copy()

    # 2. VAL = échantillonné aléatoirement du reste
    n_val = int(n_total * val_ratio)
    val_data = remaining.sample(n=n_val, random_state=random_seed)
    val_indices = val_data.index

    # 3. TRAIN = le reste (après avoir retiré val)
    train_data = remaining.drop(val_indices).copy()

    logger.info(f"📊 Split temporel (Test=fin, Val=échantillonné):")
    logger.info(f"  Train: {len(train_data):,} bougies ({len(train_data)/n_total:.0%})")
    logger.info(f"  Val:   {len(val_data):,} bougies ({val_ratio:.0%}) - échantillonné de partout")
    logger.info(f"  Test:  {len(test_data):,} bougies ({test_ratio:.0%}) - FIN du dataset (plus récent)")

    # Shuffle train (mélange les batches)
    if shuffle_train:
        train_data = train_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        logger.info(f"  ✅ Train shuffled")

    # Reset index pour val et test
    val_data = val_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    # Vérifier cohérence
    assert len(train_data) + len(val_data) + len(test_data) == n_total, \
        "Erreur de split : longueur totale incorrecte"

    return train_data, val_data, test_data


def load_and_split_btc_eth(btc_candles=BTC_CANDLES, eth_candles=ETH_CANDLES,
                            trim_start=TRIM_EDGES, trim_end=TRIM_EDGES,
                            train_ratio=TRAIN_SPLIT, val_ratio=VAL_SPLIT, test_ratio=TEST_SPLIT):
    """
    Charge BTC+ETH, trim les edges, combine, et fait un split temporel.

    Pipeline complet :
    1. Charger BTC et ETH
    2. Prendre les dernières N bougies de chaque
    3. Trim edges (warm-up + artifacts)
    4. Combiner BTC + ETH
    5. Split temporel (train/val/test)

    Args:
        btc_candles : Nombre de bougies BTC à charger
        eth_candles : Nombre de bougies ETH à charger
        trim_start : Bougies à enlever au début
        trim_end : Bougies à enlever à la fin
        train_ratio : Ratio train
        val_ratio : Ratio validation
        test_ratio : Ratio test

    Returns:
        train, val, test : DataFrames prêts pour l'entraînement

    Example:
        >>> train, val, test = load_and_split_btc_eth()
        >>> print(f"Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")
    """
    logger.info("="*80)
    logger.info("CHARGEMENT ET PRÉPARATION DES DONNÉES")
    logger.info("="*80)

    # Charger BTC
    btc = load_crypto_data(BTC_DATA_FILE, n_candles=btc_candles, asset_name='BTC')

    # Charger ETH
    eth = load_crypto_data(ETH_DATA_FILE, n_candles=eth_candles, asset_name='ETH')

    # Trim edges (enlever warm-up + artifacts)
    btc_trimmed = trim_edges(btc, trim_start=trim_start, trim_end=trim_end)
    eth_trimmed = trim_edges(eth, trim_start=trim_start, trim_end=trim_end)

    # Combiner (ordre chronologique préservé)
    all_data = pd.concat([btc_trimmed, eth_trimmed], ignore_index=True)
    logger.info(f"🔗 Combinaison BTC + ETH : {len(all_data):,} bougies totales")

    # Split temporel (CRITIQUE : pas de shuffle avant!)
    train, val, test = temporal_split(
        all_data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        shuffle_train=True  # Shuffle APRÈS split
    )

    logger.info("="*80)
    logger.info("✅ DONNÉES PRÊTES")
    logger.info("="*80)

    return train, val, test


def validate_no_leakage(train, val, test, n_check=5):
    """
    Valide qu'il n'y a pas de data leakage entre train/val/test.

    Vérifie que les timestamps sont bien séparés temporellement.

    Args:
        train, val, test : DataFrames des 3 sets
        n_check : Nombre de lignes à vérifier aux frontières

    Raises:
        AssertionError si data leakage détecté
    """
    logger.info("🔍 Validation : Vérification data leakage...")

    # Vérifier que les timestamps sont ordonnés
    if 'timestamp' in train.columns:
        # Derniers de train < Premiers de val
        train_last = train['timestamp'].iloc[-n_check:].max()
        val_first = val['timestamp'].iloc[:n_check].min()

        assert train_last < val_first, \
            f"Data leakage détecté : train_last ({train_last}) >= val_first ({val_first})"

        # Derniers de val < Premiers de test
        val_last = val['timestamp'].iloc[-n_check:].max()
        test_first = test['timestamp'].iloc[:n_check].min()

        assert val_last < test_first, \
            f"Data leakage détecté : val_last ({val_last}) >= test_first ({test_first})"

        logger.info("  ✅ Pas de data leakage : timestamps bien séparés")
        logger.info(f"    Train max: {train_last}")
        logger.info(f"    Val range: {val_first} → {val_last}")
        logger.info(f"    Test min: {test_first}")
    else:
        logger.warning("  ⚠️ Colonne 'timestamp' absente, validation partielle")

    logger.info("✅ Validation réussie : données propres")


# =============================================================================
# Exemple d'utilisation
# =============================================================================

if __name__ == '__main__':
    # Configurer logging
    logging.basicConfig(level=logging.INFO)

    # Charger et splitter les données
    train_data, val_data, test_data = load_and_split_btc_eth()

    # Valider pas de leakage
    validate_no_leakage(train_data, val_data, test_data)

    # Afficher stats
    print(f"\n📊 STATS FINALES:")
    print(f"  Train: {len(train_data):,} bougies")
    print(f"  Val:   {len(val_data):,} bougies")
    print(f"  Test:  {len(test_data):,} bougies")
    print(f"  Total: {len(train_data) + len(val_data) + len(test_data):,} bougies")
