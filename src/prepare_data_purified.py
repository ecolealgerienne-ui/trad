"""
Preparation des donnees avec INPUTS PURIFIES par indicateur.

PRINCIPE CLE : "More Data" ≠ "Better Results"
==============================================

Chaque indicateur doit recevoir UNIQUEMENT les features causalement liees
a sa formule physique. Injecter des inputs non-causaux = bruit toxique.

Analyse des Inputs par Indicateur :
-----------------------------------

RSI (Relative Strength Index) :
  - Formule : Moyenne(Gains) / Moyenne(Pertes) sur Close
  - Inputs necessaires : Close UNIQUEMENT
  - Inputs toxiques : Open, High, Low (meches = bruit pur)
  - Features optimales : C_ret, C_momentum, C_MA

MACD (Moving Average Convergence Divergence) :
  - Formule : EMA_fast(Close) - EMA_slow(Close)
  - Inputs necessaires : Close UNIQUEMENT
  - Inputs toxiques : Open, High, Low (variance inutile)
  - Features optimales : C_ret, C_momentum, C_MA

CCI (Commodity Channel Index) :
  - Formule : (TP - MA(TP)) / (0.015 * MeanDev(TP))
  - Typical Price = (High + Low + Close) / 3
  - Inputs necessaires : High, Low, Close (volatilite des meches)
  - Inputs toxiques : Open (jamais utilise)
  - Features optimales : H_ret, L_ret, C_ret, Range_ret, ATR

Impact du Bruit :
-----------------

Scenario classique : Bougie avec meche basse mais cloture verte
  - RSI/MACD (basés sur Close) : Voient une hausse → Signal UP
  - High/Low (si injectes) : Crient "Volatilite extreme !" → Signal confus
  - Resultat : Modele hesite, accuracy baisse, micro-trades

Exemple concret (vecu) :
  Close[t-1] = 100, Close[t] = 105 → RSI/MACD voient +5%
  Mais Low[t] = 95 (meche -5%)
  Si Low_ret injecte → Modele voit (+5%, -5%) = contradiction
  Le gradient ne sait plus quoi optimiser

Solution : Purification
-----------------------

Ne donner que les inputs CAUSAUX :
  - RSI/MACD : Close-based features uniquement
  - CCI : Volatility-aware features (H/L necessaires)

Gain attendu : +2-4% accuracy (reduction massive du bruit)

Usage:
------
python src/prepare_data_purified.py \\
    --target rsi \\
    --assets BTC ETH BNB ADA LTC

python src/prepare_data_purified.py \\
    --target macd \\
    --assets BTC ETH BNB ADA LTC

python src/prepare_data_purified.py \\
    --target cci \\
    --assets BTC ETH BNB ADA LTC
"""

import numpy as np
import pandas as pd
import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict
import logging

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent))

from constants import (
    AVAILABLE_ASSETS_5M,
    RSI_PERIOD,
    CCI_PERIOD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    SEQUENCE_LENGTH,
    TRAIN_RATIO,
    VAL_RATIO,
)
from indicators_ta import calculate_rsi_ta, calculate_cci_ta, calculate_macd_ta
from filters import apply_kalman_filter
from data_utils import temporal_split_with_gap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_close_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule features PURES basees sur Close uniquement.

    Pour RSI et MACD : Pas de High/Low (bruit toxique).

    Features generees :
    - C_ret : Rendement Close-to-Close (pattern principal)
    - C_ma_5 : MA courte des rendements (tendance court terme)
    - C_ma_20 : MA longue des rendements (tendance long terme)
    - C_mom_3 : Momentum 3 periodes (acceleration courte)
    - C_mom_10 : Momentum 10 periodes (acceleration moyenne)

    Args:
        df: DataFrame avec colonne 'close'

    Returns:
        DataFrame avec 5 features Close-based
    """
    close = df['close'].values

    # Rendement Close-to-Close (pattern de base)
    c_ret = np.zeros(len(close))
    c_ret[1:] = (close[1:] - close[:-1]) / close[:-1]

    # Moyennes mobiles des rendements
    c_ret_series = pd.Series(c_ret)
    c_ma_5 = c_ret_series.rolling(5).mean().fillna(0).values
    c_ma_20 = c_ret_series.rolling(20).mean().fillna(0).values

    # Momentum (variation sur N periodes)
    c_mom_3 = np.zeros(len(close))
    c_mom_10 = np.zeros(len(close))
    c_mom_3[3:] = (close[3:] - close[:-3]) / close[:-3]
    c_mom_10[10:] = (close[10:] - close[:-10]) / close[:-10]

    features = pd.DataFrame({
        'C_ret': c_ret,
        'C_ma_5': c_ma_5,
        'C_ma_20': c_ma_20,
        'C_mom_3': c_mom_3,
        'C_mom_10': c_mom_10
    })

    return features


def calculate_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule features VOLATILITY-AWARE pour CCI.

    CCI a BESOIN de High/Low (mesure de deviation).

    Features generees :
    - C_ret : Rendement net (toujours utile)
    - H_ret : Extension haussiere intra-bougie
    - L_ret : Extension baissiere intra-bougie
    - Range_ret : Volatilite intra-bougie (High - Low)
    - ATR_norm : Average True Range normalise (compatible CCI)

    Args:
        df: DataFrame avec colonnes 'open', 'high', 'low', 'close'

    Returns:
        DataFrame avec 5 features Volatility-aware
    """
    open_price = df['open'].values
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    # Close-to-Close return (base)
    c_ret = np.zeros(len(close))
    c_ret[1:] = (close[1:] - close[:-1]) / close[:-1]

    # Extensions intra-bougie (necessaires pour CCI)
    h_ret = np.zeros(len(high))
    l_ret = np.zeros(len(low))
    h_ret[1:] = (high[1:] - close[:-1]) / close[:-1]
    l_ret[1:] = (low[1:] - close[:-1]) / close[:-1]

    # Volatilite intra-bougie (coeur du CCI)
    range_ret = np.zeros(len(high))
    range_ret[1:] = (high[1:] - low[1:]) / close[:-1]

    # ATR normalise (mesure la volatilite vraie)
    true_range = np.zeros(len(close))
    true_range[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    atr_14 = pd.Series(true_range).rolling(14).mean().fillna(0).values
    atr_norm = atr_14 / (close + 1e-8)  # Normalise par prix

    features = pd.DataFrame({
        'C_ret': c_ret,
        'H_ret': h_ret,
        'L_ret': l_ret,
        'Range_ret': range_ret,
        'ATR_norm': atr_norm
    })

    return features


def prepare_purified_data(
    target: str,
    assets: list,
    filter_type: str = 'kalman'
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Prepare donnees avec inputs purifies par indicateur.

    Args:
        target: 'rsi', 'macd', ou 'cci'
        assets: Liste des assets a inclure
        filter_type: Type de filtre pour labels ('kalman')

    Returns:
        X, Y, metadata
    """
    target = target.lower()
    if target not in ['rsi', 'macd', 'cci']:
        raise ValueError(f"Target invalide: {target}. Choisir parmi: rsi, macd, cci")

    logger.info(f"Preparation donnees PURIFIEES pour {target.upper()}")
    logger.info(f"Assets: {assets}")

    all_X = []
    all_Y = []
    asset_indices = []

    for asset_idx, asset in enumerate(assets):
        csv_path = AVAILABLE_ASSETS_5M.get(asset)
        if not csv_path:
            logger.warning(f"Asset {asset} non trouve, skip")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {asset}")
        logger.info(f"{'='*60}")

        # Charger donnees brutes
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df):,} samples")

        # Trim edges (100 debut + 100 fin)
        df = df.iloc[100:-100].copy()
        logger.info(f"After trim: {len(df):,} samples")

        # Calculer indicateur cible
        if target == 'rsi':
            indicator = calculate_rsi_ta(df['close'], window=RSI_PERIOD, fillna=False)
        elif target == 'macd':
            result = calculate_macd_ta(
                df['close'],
                window_fast=MACD_FAST,
                window_slow=MACD_SLOW,
                window_sign=MACD_SIGNAL,
                fillna=False
            )
            indicator = result['signal']  # MACD signal line
        elif target == 'cci':
            indicator = calculate_cci_ta(
                df['high'],
                df['low'],
                df['close'],
                window=CCI_PERIOD,
                fillna=False
            )

        logger.info(f"Calculated {target.upper()}")

        # Appliquer filtre pour labels
        if filter_type == 'kalman':
            filtered = apply_kalman_filter(indicator)
        else:
            raise ValueError(f"Filtre non supporte: {filter_type}")

        logger.info(f"Applied {filter_type} filter")

        # Calculer labels (pente filtree)
        labels = np.zeros(len(filtered))
        labels[3:] = (filtered[2:-1] > filtered[1:-2]).astype(int)

        # Calculer features PURIFIEES selon indicateur
        if target in ['rsi', 'macd']:
            # Close-based uniquement (pas de High/Low)
            features_df = calculate_close_based_features(df)
            logger.info(f"Features Close-based: {list(features_df.columns)}")
        else:  # cci
            # Volatility-aware (High/Low necessaires)
            features_df = calculate_volatility_features(df)
            logger.info(f"Features Volatility-aware: {list(features_df.columns)}")

        features = features_df.values
        n_features = features.shape[1]

        # Creer sequences
        n_samples = len(features) - SEQUENCE_LENGTH
        X_asset = np.zeros((n_samples, SEQUENCE_LENGTH, n_features))
        Y_asset = np.zeros(n_samples)

        for i in range(n_samples):
            X_asset[i] = features[i:i+SEQUENCE_LENGTH]
            Y_asset[i] = labels[i+SEQUENCE_LENGTH]

        # Filtrer NaN
        mask = ~np.isnan(Y_asset)
        X_asset = X_asset[mask]
        Y_asset = Y_asset[mask]

        logger.info(f"Created {len(X_asset):,} sequences (shape: {X_asset.shape})")

        all_X.append(X_asset)
        all_Y.append(Y_asset)
        asset_indices.extend([asset_idx] * len(X_asset))

    # Concatener tous les assets
    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)
    asset_indices = np.array(asset_indices)

    logger.info(f"\n{'='*60}")
    logger.info(f"TOTAL: {len(X):,} sequences")
    logger.info(f"Shape: {X.shape}")
    logger.info(f"Labels UP: {(Y==1).sum():,} ({(Y==1).mean()*100:.1f}%)")
    logger.info(f"Labels DOWN: {(Y==0).sum():,} ({(Y==0).mean()*100:.1f}%)")

    # Metadata
    metadata = {
        'target': target,
        'assets': assets,
        'filter_type': filter_type,
        'n_features': n_features,
        'feature_type': 'close_based' if target in ['rsi', 'macd'] else 'volatility_aware',
        'sequence_length': SEQUENCE_LENGTH,
        'total_samples': len(X),
    }

    return X, Y, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Preparation donnees avec inputs purifies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--target', type=str, required=True,
                        choices=['rsi', 'macd', 'cci'],
                        help='Indicateur cible (rsi, macd, cci)')
    parser.add_argument('--assets', nargs='+', default=['BTC', 'ETH'],
                        help='Liste des assets')
    parser.add_argument('--filter', type=str, default='kalman',
                        choices=['kalman'],
                        help='Type de filtre pour labels')
    parser.add_argument('--output', type=str, default=None,
                        help='Chemin de sortie .npz (auto si non specifie)')

    args = parser.parse_args()

    # Preparer donnees
    X, Y, metadata = prepare_purified_data(
        target=args.target,
        assets=args.assets,
        filter_type=args.filter
    )

    # Split temporel
    logger.info("\nSplit temporel (70/15/15 train/val/test)...")
    splits = temporal_split_with_gap(
        X, Y,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO
    )

    X_train, Y_train = splits['train']
    X_val, Y_val = splits['val']
    X_test, Y_test = splits['test']

    logger.info(f"Train: {len(X_train):,} samples")
    logger.info(f"Val:   {len(X_val):,} samples")
    logger.info(f"Test:  {len(X_test):,} samples")

    # Output path
    if args.output is None:
        assets_str = '_'.join([a.lower() for a in args.assets])
        output_path = Path('data/prepared') / f"dataset_{assets_str}_purified_{args.target}_{args.filter}.npz"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sauvegarder
    logger.info(f"\nSauvegarde: {output_path}")
    np.savez_compressed(
        output_path,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        X_test=X_test,
        Y_test=Y_test,
        metadata=metadata
    )

    logger.info("✅ Terminé!")
    logger.info(f"\nPour entrainer:")
    logger.info(f"python src/train.py --data {output_path} --indicator {args.target}")


if __name__ == '__main__':
    main()
