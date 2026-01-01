"""
Pipeline principal de transformation des données.

MISSION: Transformer des bougies de 5min en dataset de 30min avec "Bougie Fantôme"

Concept de la Bougie Fantôme:
- Chaque 5min dans une bougie de 30min, on génère une ligne de données
- Cette ligne contient l'état actuel de la bougie 30min en formation
- À t=5min: [O,H,L,C] basé sur 1ère bougie 5m
- À t=10min: [O,H,L,C] mis à jour avec 2 premières bougies 5m
- ...
- À t=30min: Bougie 30m complète

Ce pipeline génère le dataset final prêt pour l'entraînement du modèle.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import argparse
import logging
from pathlib import Path

# Configurer le logging AVANT les imports locaux
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Imports locaux (vérifier que les fonctions existent avant de les utiliser)
from utils import (
    load_ohlcv_data,
    validate_ohlc_integrity,
    save_dataset,
    check_data_leakage
)
from advanced_features import add_all_advanced_features

# Préférer indicators_ta (bibliothèque ta) si disponible
try:
    from indicators_ta import add_all_ta_features as add_all_indicators
    logger.info("Utilisation de la bibliothèque 'ta' pour les indicateurs")
except ImportError:
    from indicators import add_all_indicators
    logger.warning("Bibliothèque 'ta' non disponible, utilisation de indicators.py")

from normalization import normalize_ohlc_ghost, normalize_features
from labeling import add_labels_to_dataframe


def create_ghost_candles(df_5m: pd.DataFrame,
                        target_timeframe: str = '30min') -> pd.DataFrame:
    """
    Crée les "bougies fantômes" - snapshots intermédiaires de la bougie cible.

    Pour chaque bougie de 30min, on génère 6 lignes (une toutes les 5min).

    Args:
        df_5m: DataFrame avec données 5min [timestamp, open, high, low, close, volume]
        target_timeframe: Timeframe cible (défaut: '30min' pour 30 minutes)

    Returns:
        DataFrame avec colonnes:
        - timestamp: Timestamp de la bougie 5min actuelle
        - ghost_open: Open de la bougie 30m
        - ghost_high: High actuel de la bougie 30m (max jusqu'à maintenant)
        - ghost_low: Low actuel de la bougie 30m (min jusqu'à maintenant)
        - ghost_close: Close de la dernière bougie 5m
        - ghost_volume: Volume cumulé
        - step: Étape dans la bougie (1-6 pour 30min)
        - candle_30m_timestamp: Timestamp de début de la bougie 30m
    """
    logger.info(f"Création des bougies fantômes ({target_timeframe})...")

    df = df_5m.copy()

    # S'assurer que timestamp est une datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df.sort_values('timestamp').reset_index(drop=True)

    # Déterminer à quelle bougie 30m appartient chaque bougie 5m
    # FIX: Décalage de -1ms pour inclure le 6ème step (ex: 14:30 dans bloc 14:00-14:30)
    # Convention: timestamp = FIN de bougie
    df['candle_30m_timestamp'] = (df['timestamp'] - pd.Timedelta('1ms')).dt.floor(target_timeframe)

    # Grouper par bougie 30m
    grouped = df.groupby('candle_30m_timestamp')

    ghost_rows = []

    for candle_30m_ts, group in grouped:
        group = group.reset_index(drop=True)
        n_candles_5m = len(group)

        # Pour chaque bougie 5m dans ce groupe, créer une ligne fantôme
        for i in range(n_candles_5m):
            # Prendre les bougies 5m jusqu'à l'index i (inclus)
            subset = group.iloc[:i+1]

            ghost_row = {
                'timestamp': group.iloc[i]['timestamp'],
                'candle_30m_timestamp': candle_30m_ts,
                'step': i + 1,  # Step 1-6
                'ghost_open': subset.iloc[0]['open'],  # Open de la 1ère bougie
                'ghost_high': subset['high'].max(),    # High max jusqu'à maintenant
                'ghost_low': subset['low'].min(),      # Low min jusqu'à maintenant
                'ghost_close': subset.iloc[-1]['close'],  # Close de la dernière bougie
                'ghost_volume': subset['volume'].sum() if 'volume' in subset.columns else 0,
                # Ajouter aussi les données de la bougie 5m actuelle pour features
                'current_5m_open': group.iloc[i]['open'],
                'current_5m_high': group.iloc[i]['high'],
                'current_5m_low': group.iloc[i]['low'],
                'current_5m_close': group.iloc[i]['close'],
            }

            ghost_rows.append(ghost_row)

    df_ghost = pd.DataFrame(ghost_rows)

    logger.info(f"Bougies fantômes créées: {len(df_ghost)} lignes")
    logger.info(f"Nombre de bougies 30m: {df_ghost['candle_30m_timestamp'].nunique()}")

    # Vérifier l'intégrité OHLC
    try:
        validate_ohlc_integrity(df_ghost, col_prefix='ghost_')
        logger.info("✅ Validation OHLC: OK")
    except ValueError as e:
        logger.error(f"❌ Validation OHLC: {e}")

    return df_ghost


def add_historical_features(df_ghost: pd.DataFrame,
                           df_5m: pd.DataFrame,
                           lookback_candles: int = 10) -> pd.DataFrame:
    """
    Ajoute des features basées sur l'historique (10 dernières bougies 5m).

    Args:
        df_ghost: DataFrame avec bougies fantômes
        df_5m: DataFrame original 5min
        lookback_candles: Nombre de bougies 5m à regarder en arrière

    Returns:
        DataFrame avec features historiques ajoutées
    """
    logger.info(f"Ajout de features historiques (lookback={lookback_candles})...")

    df = df_ghost.copy()

    # S'assurer que df_5m a un index sur timestamp
    df_5m = df_5m.set_index('timestamp')

    # Pour chaque ligne fantôme, récupérer les N dernières bougies 5m
    historical_features = []

    for idx, row in df.iterrows():
        current_timestamp = row['timestamp']

        # Prendre les N dernières bougies AVANT current_timestamp (causal)
        historical = df_5m[df_5m.index < current_timestamp].tail(lookback_candles)

        if len(historical) < lookback_candles:
            # Pas assez d'historique, remplir avec NaN
            features = {f'hist_{i}_{col}': np.nan
                       for i in range(lookback_candles)
                       for col in ['open', 'high', 'low', 'close']}
        else:
            # Créer des features pour chaque bougie historique
            features = {}
            for i, (ts, candle) in enumerate(historical.iterrows()):
                features[f'hist_{i}_open'] = candle['open']
                features[f'hist_{i}_high'] = candle['high']
                features[f'hist_{i}_low'] = candle['low']
                features[f'hist_{i}_close'] = candle['close']

        historical_features.append(features)

    # Ajouter au DataFrame
    df_hist = pd.DataFrame(historical_features)
    df = pd.concat([df, df_hist], axis=1)

    logger.info(f"Features historiques ajoutées: {len(df_hist.columns)} colonnes")

    return df


def build_dataset(input_file: str,
                 output_file: str,
                 target_timeframe: str = '30min',
                 add_indicators: bool = True,
                 add_history: bool = False,
                 lookback: int = 10,
                 label_source: str = 'rsi',
                 smoothing: float = 0.25) -> pd.DataFrame:
    """
    Pipeline complet de construction du dataset.

    Étapes:
    1. Charger données 5min
    2. Créer bougies fantômes 30min
    3. Ajouter features historiques (optionnel)
    4. Calculer indicateurs techniques
    5. Normaliser les features
    6. Créer les labels
    7. Sauvegarder le dataset

    Args:
        input_file: Chemin vers fichier CSV 5min
        output_file: Chemin de sortie
        target_timeframe: Timeframe cible
        add_indicators: Si True, calcule les indicateurs
        add_history: Si True, ajoute features historiques
        lookback: Nombre de bougies historiques
        label_source: Source pour labels ('rsi' ou 'close')
        smoothing: Paramètre de lissage du filtre

    Returns:
        DataFrame final
    """
    logger.info("="*80)
    logger.info("DÉBUT DU PIPELINE DE CONSTRUCTION DU DATASET")
    logger.info("="*80)

    # 1. Charger les données 5min
    logger.info("\n[1/7] Chargement des données 5min...")
    df_5m = load_ohlcv_data(input_file)
    logger.info(f"Données chargées: {len(df_5m)} bougies 5min")

    # 2. Créer les bougies fantômes
    logger.info("\n[2/7] Création des bougies fantômes...")
    df_ghost = create_ghost_candles(df_5m, target_timeframe=target_timeframe)

    # 2b. Ajouter les features avancées (velocity, log returns, Z-Score, etc.)
    logger.info("\n[2b/7] Ajout des features avancées...")
    df_ghost = add_all_advanced_features(
        df_ghost,
        ghost_prefix='ghost',
        open_zscore_window=50,
        max_steps=6
    )
    logger.info("✅ Features avancées ajoutées")

    # 3. Features historiques (optionnel)
    if add_history:
        logger.info("\n[3/7] Ajout des features historiques...")
        df = add_historical_features(df_ghost, df_5m, lookback_candles=lookback)
    else:
        logger.info("\n[3/7] Features historiques: IGNORÉ")
        df = df_ghost

    # 4. Indicateurs techniques sur la bougie 5m actuelle
    if add_indicators:
        logger.info("\n[4/7] Calcul des indicateurs techniques...")

        # Créer un DataFrame temporaire avec les données de la bougie 5m actuelle
        df_current_5m = df[['timestamp', 'current_5m_open', 'current_5m_high',
                           'current_5m_low', 'current_5m_close']].copy()

        df_current_5m = df_current_5m.rename(columns={
            'current_5m_open': 'open',
            'current_5m_high': 'high',
            'current_5m_low': 'low',
            'current_5m_close': 'close'
        })

        # Calculer les indicateurs
        df_with_indicators = add_all_indicators(
            df_current_5m,
            rsi_periods=[14, 21],
            cci_periods=[20],
            macd_params=[(12, 26, 9)],
            bb_periods=[20]
        )

        # Récupérer seulement les colonnes d'indicateurs
        indicator_cols = [col for col in df_with_indicators.columns
                         if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        # Ajouter au DataFrame principal
        for col in indicator_cols:
            df[col] = df_with_indicators[col].values

        logger.info(f"Indicateurs ajoutés: {len(indicator_cols)} colonnes")
    else:
        logger.info("\n[4/7] Indicateurs techniques: IGNORÉ")
        indicator_cols = []

    # 5. Normalisation
    logger.info("\n[5/7] Normalisation des features...")

    # IMPORTANT: On n'utilise PLUS normalize_ohlc_ghost avec relative_open
    # car les features avancées contiennent déjà:
    # - ghost_high_log, ghost_low_log, ghost_close_log (log returns)
    # - ghost_open_zscore (Z-Score contexte)
    # Ces features sont déjà normalisées et prêtes pour le modèle.

    # Normaliser les indicateurs (Z-Score)
    if indicator_cols:
        df = normalize_features(df, indicator_cols, method='zscore', window=50)
        logger.info("Indicateurs normalisés (Z-Score)")

    # 6. Créer les labels
    logger.info("\n[6/7] Création des labels...")

    # On utilise le close de la bougie 5m actuelle pour calculer les labels
    df = add_labels_to_dataframe(
        df,
        label_source=label_source,
        smoothing=smoothing,
        validate=True
    )

    # 7. Vérification du data leakage
    logger.info("\n[7/7] Vérification du data leakage...")

    feature_cols = [col for col in df.columns
                   if col not in ['timestamp', 'label', 'candle_30m_timestamp',
                                 'slope', 'slope_shifted', 'rsi_filtered', 'close_filtered']]

    leakage_check = check_data_leakage(df, feature_cols, label_col='label')

    if leakage_check['suspicious_features']:
        logger.error(f"❌ {len(leakage_check['suspicious_features'])} features suspectes!")
    else:
        logger.info("✅ Pas de data leakage détecté")

    # 8. Sauvegarder
    logger.info("\n[8/8] Sauvegarde du dataset...")
    save_dataset(df, output_file, compress=False)

    logger.info("="*80)
    logger.info("PIPELINE TERMINÉ AVEC SUCCÈS")
    logger.info("="*80)

    return df


def build_multiasset_dataset(input_files: Dict[str, str],
                             output_file: str,
                             target_timeframe: str = '30min',
                             add_indicators: bool = True,
                             label_source: str = 'rsi',
                             smoothing: float = 0.25) -> pd.DataFrame:
    """
    Pipeline multi-actifs avec normalisation séparée par actif.

    ⚠️ CRITIQUE: Normalise chaque actif SÉPARÉMENT pour éviter le leakage inter-actifs.

    Stratégie "Hedge Fund" (Renaissance Technologies, Two Sigma):
    - BTC et ETH sont normalisés avec LEURS propres statistiques
    - Le modèle apprend des patterns universels, pas les différences de prix
    - Permet de généraliser à de nouveaux actifs (XRP, ADA, etc.)

    Args:
        input_files: Dict {'ASSET_NAME': 'path/to/5m.csv'}
                     Ex: {'BTC': '../data_trad/BTCUSD_all_5m.csv',
                          'ETH': '../data_trad/ETHUSD_all_5m.csv'}
        output_file: Chemin de sortie pour le dataset combiné
        target_timeframe: Timeframe cible (défaut: '30min')
        add_indicators: Si True, calcule les indicateurs
        label_source: Source pour labels ('rsi' ou 'close')
        smoothing: Paramètre de lissage du filtre

    Returns:
        DataFrame combiné avec colonne 'asset' identifiant l'origine

    Example:
        >>> datasets = build_multiasset_dataset(
        ...     input_files={'BTC': 'btc_5m.csv', 'ETH': 'eth_5m.csv'},
        ...     output_file='data/processed/multi_asset_dataset.csv'
        ... )
    """
    logger.info("="*80)
    logger.info("PIPELINE MULTI-ACTIFS AVEC NORMALISATION SÉPARÉE")
    logger.info("="*80)

    all_datasets = []

    for asset_name, input_file in input_files.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAITEMENT DE L'ACTIF: {asset_name}")
        logger.info(f"{'='*60}")

        # 1. Charger les données 5min
        logger.info(f"\n[{asset_name}][1/6] Chargement des données 5min...")
        df_5m = load_ohlcv_data(input_file)
        logger.info(f"Données chargées: {len(df_5m)} bougies 5min")

        # 2. Créer les bougies fantômes
        logger.info(f"\n[{asset_name}][2/6] Création des bougies fantômes...")
        df_ghost = create_ghost_candles(df_5m, target_timeframe=target_timeframe)

        # 2b. Ajouter les features avancées
        logger.info(f"\n[{asset_name}][2b/6] Ajout des features avancées...")
        df_ghost = add_all_advanced_features(
            df_ghost,
            ghost_prefix='ghost',
            open_zscore_window=50,
            max_steps=6
        )

        # 3. Indicateurs techniques
        if add_indicators:
            logger.info(f"\n[{asset_name}][3/6] Calcul des indicateurs techniques...")

            df_current_5m = df_ghost[['timestamp', 'current_5m_open', 'current_5m_high',
                                      'current_5m_low', 'current_5m_close']].copy()

            df_current_5m = df_current_5m.rename(columns={
                'current_5m_open': 'open',
                'current_5m_high': 'high',
                'current_5m_low': 'low',
                'current_5m_close': 'close'
            })

            df_with_indicators = add_all_indicators(
                df_current_5m,
                rsi_periods=[14, 21],
                cci_periods=[20],
                macd_params=[(12, 26, 9)],
                bb_periods=[20]
            )

            indicator_cols = [col for col in df_with_indicators.columns
                             if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            for col in indicator_cols:
                df_ghost[col] = df_with_indicators[col].values

            logger.info(f"Indicateurs ajoutés: {len(indicator_cols)} colonnes")
        else:
            indicator_cols = []

        # 4. Normalisation (PAR ACTIF - CRITIQUE!)
        logger.info(f"\n[{asset_name}][4/6] Normalisation PAR ACTIF...")
        logger.info(f"⚠️  IMPORTANT: Normalisation basée sur les statistiques de {asset_name} UNIQUEMENT")

        # Les features avancées sont déjà normalisées (log returns, Z-Score)
        # On normalise seulement les indicateurs
        if indicator_cols:
            df_ghost = normalize_features(df_ghost, indicator_cols, method='zscore', window=50)
            logger.info(f"Indicateurs normalisés avec statistiques de {asset_name}")

        # 5. Créer les labels
        logger.info(f"\n[{asset_name}][5/6] Création des labels...")
        df_ghost = add_labels_to_dataframe(
            df_ghost,
            label_source=label_source,
            smoothing=smoothing,
            validate=True
        )

        # 6. Ajouter la colonne asset pour identifier l'origine
        logger.info(f"\n[{asset_name}][6/6] Ajout de l'identifiant d'actif...")
        df_ghost['asset'] = asset_name

        # Vérification du leakage pour cet actif
        feature_cols = [col for col in df_ghost.columns
                       if col not in ['timestamp', 'label', 'candle_30m_timestamp',
                                     'slope', 'slope_shifted', 'rsi_filtered',
                                     'close_filtered', 'asset']]

        leakage_check = check_data_leakage(df_ghost, feature_cols, label_col='label')

        if leakage_check['suspicious_features']:
            logger.warning(f"⚠️  {asset_name}: {len(leakage_check['suspicious_features'])} features suspectes!")
        else:
            logger.info(f"✅ {asset_name}: Pas de data leakage détecté")

        all_datasets.append(df_ghost)
        logger.info(f"\n✅ {asset_name} traité: {len(df_ghost)} lignes")

    # Combiner tous les datasets
    logger.info("\n" + "="*80)
    logger.info("COMBINAISON DES DATASETS")
    logger.info("="*80)

    df_combined = pd.concat(all_datasets, axis=0, ignore_index=True)

    # Trier par timestamp pour avoir un dataset temporel cohérent
    df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"Dataset combiné: {len(df_combined)} lignes")
    for asset_name in input_files.keys():
        count = len(df_combined[df_combined['asset'] == asset_name])
        pct = count / len(df_combined) * 100
        logger.info(f"  - {asset_name}: {count} lignes ({pct:.1f}%)")

    # Sauvegarder
    logger.info("\nSauvegarde du dataset combiné...")
    save_dataset(df_combined, output_file, compress=False)

    logger.info("="*80)
    logger.info("PIPELINE MULTI-ACTIFS TERMINÉ AVEC SUCCÈS")
    logger.info("="*80)

    return df_combined


def main():
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Pipeline de transformation 5min → 30min avec Bougie Fantôme"
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Chemin vers fichier CSV 5min (ex: ../data_trad/BTCUSD_all_5m.csv)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Chemin de sortie CSV (ex: data/processed/btc_30m_dataset.csv)'
    )

    parser.add_argument(
        '--timeframe',
        type=str,
        default='30min',
        help='Timeframe cible (défaut: 30min)'
    )

    parser.add_argument(
        '--no-indicators',
        action='store_true',
        help='Ne pas calculer les indicateurs techniques'
    )

    parser.add_argument(
        '--add-history',
        action='store_true',
        help='Ajouter features historiques (10 dernières bougies)'
    )

    parser.add_argument(
        '--lookback',
        type=int,
        default=10,
        help='Nombre de bougies historiques (défaut: 10)'
    )

    parser.add_argument(
        '--label-source',
        type=str,
        default='rsi',
        choices=['rsi', 'close'],
        help='Source pour les labels (défaut: rsi)'
    )

    parser.add_argument(
        '--smoothing',
        type=float,
        default=0.25,
        help='Paramètre de lissage du filtre (défaut: 0.25)'
    )

    args = parser.parse_args()

    # Créer le dossier de sortie si nécessaire
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Lancer le pipeline
    df = build_dataset(
        input_file=args.input,
        output_file=args.output,
        target_timeframe=args.timeframe,
        add_indicators=not args.no_indicators,
        add_history=args.add_history,
        lookback=args.lookback,
        label_source=args.label_source,
        smoothing=args.smoothing
    )

    logger.info(f"\n✅ Dataset final: {len(df)} lignes, {len(df.columns)} colonnes")
    logger.info(f"Fichier de sortie: {args.output}")


if __name__ == '__main__':
    main()
