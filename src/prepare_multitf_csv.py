#!/usr/bin/env python3
"""
Préparation CSV multi-timeframe : indicateurs 30min et 1h à résolution 5min.

OBJECTIF:
Générer un CSV enrichi par asset avec les indicateurs 30min et 1h pré-calculés,
forward-fillés à résolution 5min. Ce fichier sert de base unique pour tous les
tests et entraînements ultérieurs.

PRINCIPE:
- Les données brutes sont en 5min
- On resample en 30min et 1h pour calculer les indicateurs
- Les indicateurs sont forward-fillés vers 5min (chaque 5min hérite de la
  dernière bougie 30min/1h COMPLÉTÉE)
- Causalité stricte : aucun look-ahead, label disponible seulement après
  complétion de la bougie

CE QUI EST INCLUS (données causales pures):
- OHLCV 5min brut
- OHLCV 30min (forward-filled depuis bougies complétées)
- OHLCV 1h (forward-filled depuis bougies complétées)
- Indicateurs MACD/RSI/CCI sur 30min et 1h
- Step index (position dans la bougie 30min et 1h)

CE QUI N'EST PAS INCLUS (calculé à la demande):
- Valeurs Kalman filtrées (non-causal, doit être appliqué après split)
- Labels de direction (dépendent du Kalman)

CAUSALITÉ:
- Bougie 30min 10:00 = données 10:00-10:29, close disponible après ~10:29
- Indicateur de cette bougie forward-fillé à partir de 10:30 (pas 10:00)
- Implémenté via shift(1) sur l'index 30min avant forward-fill

Usage:
    python src/prepare_multitf_csv.py --assets BTC ETH BNB ADA LTC
    python src/prepare_multitf_csv.py --assets BTC      # Un seul asset

Output:
    data/prepared/BTCUSD_multitf.csv
    data/prepared/ETHUSD_multitf.csv
    ...
"""

import numpy as np
import pandas as pd
import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import constants
sys.path.insert(0, str(Path(__file__).parent))
from constants import AVAILABLE_ASSETS_5M, PREPARED_DATA_DIR

# Périodes STANDARD des indicateurs (copiées de prepare_data_direction_only.py)
RSI_PERIOD = 14
CCI_PERIOD = 20
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


# =============================================================================
# CHARGEMENT DONNÉES 5min
# =============================================================================

def load_csv_5min(file_path: str, asset_name: str) -> pd.DataFrame:
    """
    Charge données 5min brutes depuis CSV.
    Copié de test_oracle_30min_pure.py → load_csv_5min.
    """
    df = pd.read_csv(file_path)

    date_col = None
    for col in ['date', 'datetime', 'time', 'timestamp', 'Date', 'Datetime']:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        raise ValueError(f"Colonne date non trouvée dans {file_path}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = 'datetime'
    df.columns = df.columns.str.lower()
    df = df.sort_index()

    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    logger.info(f"  {asset_name}: {len(df):,} bougies 5min, {df.index[0]} → {df.index[-1]}")

    return df


# =============================================================================
# RESAMPLING
# =============================================================================

def resample_ohlcv(df_5min: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
    """
    Resample 5min → tf_minutes avec agrégation OHLCV standard.

    Returns:
        DataFrame avec DatetimeIndex, colonnes open/high/low/close/volume
    """
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    df_tf = df_5min.resample(f'{tf_minutes}min').agg(agg_dict)
    df_tf = df_tf.dropna()

    return df_tf


# =============================================================================
# CALCUL INDICATEURS (copié de prepare_data_direction_only.py)
# =============================================================================

def calculate_rsi(df: pd.DataFrame) -> pd.Series:
    """Calcule RSI sur un DataFrame OHLCV."""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(span=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(span=RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_cci(df: pd.DataFrame) -> pd.Series:
    """Calcule CCI sur un DataFrame OHLCV."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(CCI_PERIOD).mean()
    mad = tp.rolling(CCI_PERIOD).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma_tp) / (0.015 * mad)


def calculate_macd(df: pd.DataFrame) -> pd.Series:
    """Calcule MACD histogram sur un DataFrame OHLCV."""
    ema_fast = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return macd_line - signal_line


# =============================================================================
# FORWARD-FILL CAUSAL
# =============================================================================

def forward_fill_causal(series_tf: pd.Series, index_5min: pd.DatetimeIndex,
                        col_name: str) -> pd.Series:
    """
    Forward-fill une série timeframe vers résolution 5min avec causalité stricte.

    CAUSALITÉ:
    - La valeur de la bougie tf à 10:00 (données 10:00-10:29) n'est disponible
      qu'après complétion (~10:29)
    - On l'assigne donc à partir de la PROCHAINE bougie tf (10:30)
    - Implémenté via shift(1) avant le forward-fill

    Args:
        series_tf: Série avec DatetimeIndex au timeframe supérieur
        index_5min: Index 5min vers lequel forward-filler
        col_name: Nom de la colonne (pour debug)

    Returns:
        Série à résolution 5min, forward-fillée causalement
    """
    # Shift de 1 période tf pour causalité
    shifted = series_tf.shift(1)

    # Forward-fill vers résolution 5min
    result = shifted.reindex(index_5min, method='ffill')

    return result


# =============================================================================
# CALCUL STEP INDEX
# =============================================================================

def compute_step_index(index_5min: pd.DatetimeIndex, tf_minutes: int) -> pd.Series:
    """
    Calcule la position (1-based) de chaque bougie 5min dans sa bougie tf.

    Exemple 30min:
        10:00 → step 1, 10:05 → step 2, ..., 10:25 → step 6
    Exemple 1h:
        10:00 → step 1, 10:05 → step 2, ..., 10:55 → step 12

    Returns:
        Série avec valeurs 1 à (tf_minutes / 5)
    """
    minutes = index_5min.minute + index_5min.hour * 60
    steps_per_candle = tf_minutes // 5
    step = (minutes % tf_minutes) // 5 + 1  # 1-based

    return pd.Series(step, index=index_5min, dtype=int)


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def generate_multitf_csv(asset_name: str, output_dir: str) -> str:
    """
    Génère le CSV multi-timeframe pour un asset.

    Pipeline:
    1. Charger 5min CSV brut
    2. Resampler en 30min et 1h
    3. Calculer indicateurs (MACD, RSI, CCI) sur 30min et 1h
    4. Forward-fill causal vers 5min
    5. Calculer step index
    6. Sauvegarder CSV

    Args:
        asset_name: 'BTC', 'ETH', etc.
        output_dir: Dossier de sortie

    Returns:
        Chemin du fichier CSV généré
    """
    file_path = AVAILABLE_ASSETS_5M[asset_name]

    logger.info(f"\n{'='*60}")
    logger.info(f"  ASSET: {asset_name}")
    logger.info(f"{'='*60}")

    # =========================================================================
    # 1. Charger 5min
    # =========================================================================
    df_5min = load_csv_5min(file_path, asset_name)
    index_5min = df_5min.index

    # Commencer le DataFrame résultat avec les données 5min brutes
    result = pd.DataFrame(index=index_5min)
    result['open'] = df_5min['open']
    result['high'] = df_5min['high']
    result['low'] = df_5min['low']
    result['close'] = df_5min['close']
    result['volume'] = df_5min['volume']

    # =========================================================================
    # 2-4. Pour chaque timeframe (30min, 1h)
    # =========================================================================
    for tf_minutes, suffix in [(30, '30m'), (60, '1h')]:
        logger.info(f"\n  --- Timeframe {tf_minutes}min ({suffix}) ---")

        # 2. Resampler
        df_tf = resample_ohlcv(df_5min, tf_minutes)
        logger.info(f"    Resample: {len(df_tf):,} bougies {suffix}")

        # Forward-fill OHLCV causal
        for col in ['open', 'high', 'low', 'close', 'volume']:
            result[f'{col}_{suffix}'] = forward_fill_causal(
                df_tf[col], index_5min, f'{col}_{suffix}'
            )

        # 3. Calculer indicateurs sur données resamplees
        macd_values = calculate_macd(df_tf)
        rsi_values = calculate_rsi(df_tf)
        cci_values = calculate_cci(df_tf)

        logger.info(f"    Indicateurs calculés (MACD, RSI, CCI)")

        # 4. Forward-fill causal des indicateurs
        result[f'macd_{suffix}'] = forward_fill_causal(macd_values, index_5min, f'macd_{suffix}')
        result[f'rsi_{suffix}'] = forward_fill_causal(rsi_values, index_5min, f'rsi_{suffix}')
        result[f'cci_{suffix}'] = forward_fill_causal(cci_values, index_5min, f'cci_{suffix}')

        # 5. Step index
        result[f'step_{suffix}'] = compute_step_index(index_5min, tf_minutes)

        # Stats
        n_nan = result[f'macd_{suffix}'].isna().sum()
        n_valid = len(result) - n_nan
        logger.info(f"    Forward-fill causal: {n_valid:,} valeurs valides, {n_nan:,} NaN (début)")

    # =========================================================================
    # Vérifications de cohérence
    # =========================================================================
    logger.info(f"\n  --- Vérifications ---")

    # Vérifier que les NaN sont uniquement au début (warm-up)
    for suffix in ['30m', '1h']:
        for col_name in [f'macd_{suffix}', f'rsi_{suffix}', f'cci_{suffix}']:
            series = result[col_name]
            first_valid = series.first_valid_index()
            last_nan_after_valid = series.loc[first_valid:].isna().sum()
            if last_nan_after_valid > 0:
                logger.warning(f"    ⚠️ {col_name}: {last_nan_after_valid} NaN APRÈS première valeur valide!")
            else:
                logger.info(f"    ✅ {col_name}: NaN uniquement au début (warm-up)")

    # Vérifier step index
    for suffix, tf in [('30m', 30), ('1h', 60)]:
        steps = result[f'step_{suffix}']
        expected_max = tf // 5
        actual_max = steps.max()
        actual_min = steps.min()
        logger.info(f"    ✅ step_{suffix}: min={actual_min}, max={actual_max} (attendu 1-{expected_max})")

    # Vérifier causalité : les données tf ne changent qu'au bon moment
    for suffix, tf in [('30m', 30), ('1h', 60)]:
        changes = result[f'close_{suffix}'].diff().abs() > 0
        change_steps = result.loc[changes, f'step_{suffix}']
        if len(change_steps) > 0:
            # Les changements devraient se produire au step 1 (début de nouvelle bougie)
            pct_step1 = (change_steps == 1).sum() / len(change_steps) * 100
            logger.info(f"    ✅ close_{suffix}: {pct_step1:.1f}% des changements au step 1 (causalité)")

    # =========================================================================
    # 6. Sauvegarder
    # =========================================================================
    os.makedirs(output_dir, exist_ok=True)

    # Nom du fichier basé sur l'asset
    asset_filename = file_path.split('/')[-1].replace('_all_5m.csv', '')
    output_path = os.path.join(output_dir, f'{asset_filename}_multitf.csv')

    # Sauvegarder avec timestamp comme colonne (pas index) pour compatibilité
    result_save = result.copy()
    result_save.index.name = 'datetime'
    result_save = result_save.reset_index()
    result_save.to_csv(output_path, index=False)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"\n  ✅ Sauvegardé: {output_path} ({file_size_mb:.1f} MB)")
    logger.info(f"     Lignes: {len(result):,}")
    logger.info(f"     Colonnes: {len(result.columns)}")
    logger.info(f"     Colonnes: {list(result.columns)}")

    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Prépare CSV multi-timeframe (30min + 1h) à résolution 5min'
    )
    parser.add_argument('--assets', nargs='+',
                        default=['BTC', 'ETH', 'BNB', 'ADA', 'LTC'],
                        help='Assets à traiter (défaut: tous)')
    parser.add_argument('--output-dir', type=str,
                        default=PREPARED_DATA_DIR,
                        help=f'Dossier de sortie (défaut: {PREPARED_DATA_DIR})')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PRÉPARATION CSV MULTI-TIMEFRAME")
    logger.info("=" * 60)
    logger.info(f"Assets: {args.assets}")
    logger.info(f"Timeframes: 5min (brut) + 30min + 1h")
    logger.info(f"Indicateurs: MACD, RSI, CCI")
    logger.info(f"Causalité: shift(1) avant forward-fill (label après complétion bougie)")
    logger.info(f"Output: {args.output_dir}/")

    generated_files = []

    for asset_name in args.assets:
        if asset_name not in AVAILABLE_ASSETS_5M:
            logger.warning(f"⚠️ Asset {asset_name} non trouvé, skip")
            continue

        output_path = generate_multitf_csv(asset_name, args.output_dir)
        generated_files.append(output_path)

    # Résumé final
    logger.info(f"\n{'='*60}")
    logger.info(f"RÉSUMÉ")
    logger.info(f"{'='*60}")
    logger.info(f"Fichiers générés: {len(generated_files)}")
    for f in generated_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        logger.info(f"  {f} ({size_mb:.1f} MB)")

    logger.info(f"\nStructure des colonnes:")
    logger.info(f"  5min brut:  open, high, low, close, volume")
    logger.info(f"  30min:      open_30m, high_30m, low_30m, close_30m, volume_30m")
    logger.info(f"              macd_30m, rsi_30m, cci_30m, step_30m")
    logger.info(f"  1h:         open_1h, high_1h, low_1h, close_1h, volume_1h")
    logger.info(f"              macd_1h, rsi_1h, cci_1h, step_1h")
    logger.info(f"\n⚠️ Kalman et labels NON inclus (appliquer après split train/test)")


if __name__ == '__main__':
    main()
