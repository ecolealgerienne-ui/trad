#!/usr/bin/env python3
"""
Script d'exemple pour le pipeline multi-actifs.

Démontre comment utiliser build_multiasset_dataset() pour entraîner
un modèle universel sur BTC + ETH avec normalisation séparée.

Stratégie Hedge Fund (Renaissance Technologies):
- Normaliser chaque actif séparément
- Combiner les datasets normalisés
- Le modèle apprend les patterns universels
- Généralise à de nouveaux actifs (XRP, ADA, etc.)
"""

import sys
from pathlib import Path

# Ajouter src/ au path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline import build_multiasset_dataset
from utils import split_train_val_test_with_gap
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_multiasset_btc_eth():
    """
    Exemple 1: Pipeline multi-actifs BTC + ETH
    """
    logger.info("\n" + "="*80)
    logger.info("EXEMPLE 1: PIPELINE MULTI-ACTIFS BTC + ETH")
    logger.info("="*80)

    # Chemins des données 5min
    input_files = {
        'BTC': '../data_trad/BTCUSD_all_5m.csv',  # Adapter selon votre chemin
        'ETH': '../data_trad/ETHUSD_all_5m.csv',  # Adapter selon votre chemin
    }

    # Chemin de sortie
    output_file = 'data/processed/multiasset_btc_eth_30m_dataset.csv'

    # Lancer le pipeline
    df_combined = build_multiasset_dataset(
        input_files=input_files,
        output_file=output_file,
        target_timeframe='30min',
        add_indicators=True,
        label_source='rsi',
        smoothing=0.25
    )

    logger.info("\n✅ Pipeline multi-actifs terminé!")
    logger.info(f"Dataset combiné: {len(df_combined)} lignes")
    logger.info(f"Actifs: {df_combined['asset'].unique()}")
    logger.info(f"Colonnes: {len(df_combined.columns)}")
    logger.info(f"Fichier: {output_file}")

    return df_combined


def example_multiasset_with_gap_period():
    """
    Exemple 2: Pipeline multi-actifs + Split avec Gap Period
    """
    logger.info("\n" + "="*80)
    logger.info("EXEMPLE 2: MULTI-ACTIFS + GAP PERIOD SPLIT")
    logger.info("="*80)

    # Chemins des données
    input_files = {
        'BTC': '../data_trad/BTCUSD_all_5m.csv',
        'ETH': '../data_trad/ETHUSD_all_5m.csv',
    }

    output_file = 'data/processed/multiasset_with_gap.csv'

    # 1. Créer le dataset combiné
    df_combined = build_multiasset_dataset(
        input_files=input_files,
        output_file=output_file,
        target_timeframe='30min',
        add_indicators=True,
        label_source='rsi',
        smoothing=0.25
    )

    # 2. Split avec Gap Period (éviter contamination des filtres non-causaux)
    logger.info("\n" + "="*60)
    logger.info("SPLIT AVEC GAP PERIOD ANTI-CONTAMINATION")
    logger.info("="*60)

    train_df, val_df, test_df = split_train_val_test_with_gap(
        df_combined,
        train_end_date='2023-10-31',      # Fin du train
        val_start_date='2023-11-07',      # Début validation (GAP de 7 jours)
        val_end_date='2023-11-30',        # Fin validation
        test_start_date='2023-12-01',     # Début test
        timestamp_col='timestamp'
    )

    # 3. Sauvegarder les splits
    train_df.to_csv('data/processed/multiasset_train.csv', index=False)
    val_df.to_csv('data/processed/multiasset_val.csv', index=False)
    test_df.to_csv('data/processed/multiasset_test.csv', index=False)

    logger.info("\n✅ Splits sauvegardés:")
    logger.info(f"  - Train: data/processed/multiasset_train.csv ({len(train_df)} lignes)")
    logger.info(f"  - Validation: data/processed/multiasset_val.csv ({len(val_df)} lignes)")
    logger.info(f"  - Test: data/processed/multiasset_test.csv ({len(test_df)} lignes)")

    return train_df, val_df, test_df


def example_multiasset_three_assets():
    """
    Exemple 3: Pipeline avec 3 actifs (BTC + ETH + XRP)
    """
    logger.info("\n" + "="*80)
    logger.info("EXEMPLE 3: PIPELINE 3 ACTIFS (BTC + ETH + XRP)")
    logger.info("="*80)

    input_files = {
        'BTC': '../data_trad/BTCUSD_all_5m.csv',
        'ETH': '../data_trad/ETHUSD_all_5m.csv',
        'XRP': '../data_trad/XRPUSD_all_5m.csv',  # Si disponible
    }

    output_file = 'data/processed/multiasset_btc_eth_xrp.csv'

    df_combined = build_multiasset_dataset(
        input_files=input_files,
        output_file=output_file,
        target_timeframe='30min',
        add_indicators=True,
        label_source='rsi',
        smoothing=0.25
    )

    logger.info("\n✅ Pipeline 3 actifs terminé!")
    logger.info(f"Dataset: {len(df_combined)} lignes")
    logger.info(f"Actifs: {df_combined['asset'].unique()}")

    return df_combined


if __name__ == '__main__':
    """
    Point d'entrée principal.

    Décommentez l'exemple que vous souhaitez exécuter.
    """

    # Créer le dossier de sortie
    Path('data/processed').mkdir(parents=True, exist_ok=True)

    # EXEMPLE 1: Multi-actifs BTC + ETH
    # df = example_multiasset_btc_eth()

    # EXEMPLE 2: Multi-actifs + Gap Period Split
    # train, val, test = example_multiasset_with_gap_period()

    # EXEMPLE 3: 3 actifs
    # df = example_multiasset_three_assets()

    logger.info("\n" + "="*80)
    logger.info("DÉCOMMENTEZ L'EXEMPLE SOUHAITÉ DANS __main__")
    logger.info("="*80)
    logger.info("\nExemples disponibles:")
    logger.info("1. example_multiasset_btc_eth() - Pipeline BTC + ETH basique")
    logger.info("2. example_multiasset_with_gap_period() - Avec split et gap period")
    logger.info("3. example_multiasset_three_assets() - Pipeline 3 actifs")
