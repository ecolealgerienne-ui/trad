#!/usr/bin/env python3
"""
Test Oracle 30min Pur - Indicateurs 30min estimés toutes les 5min, SANS features 5min.

OBJECTIF:
Valider que le signal 30min pur (indicateurs calculés sur données resamplees 30min)
donne un PnL net positif grâce à la réduction naturelle du nombre de trades.

APPROCHE:
1. Charger données 5min brutes (CSV)
2. Resampler en 30min (OHLCV standard)
3. Calculer indicateur (MACD/RSI/CCI) sur 30min
4. Appliquer filtre Kalman sur indicateur 30min
5. Labels: filtered[t] > filtered[t-1] (formule Phase 2.15)
6. Forward-fill labels vers résolution 5min (chaque 5min hérite du label 30min courant)
7. Backtest Oracle sur prix 5min avec labels 30min

LOGIQUE CAUSALE:
- Indicateur 30min calculé sur bougie 30min COMPLÉTÉE
- Label assigné au premier pas 5min de la PROCHAINE bougie 30min
- Exécution à Open du pas 5min suivant le signal

COMPARAISON ATTENDUE:
- Oracle 5min (Phase 2.15): 68,924 trades, Win Rate 53.4%, PnL Net +14,359%
- Oracle 30min (ce test): ~10,000-15,000 trades, WR ~50-55%, PnL Net positif?

Usage:
    python tests/test_oracle_30min_pure.py --indicator macd --fees 0.001
    python tests/test_oracle_30min_pure.py --indicator macd --fees 0.001 --assets BTC ETH
    python tests/test_oracle_30min_pure.py --indicator macd --fees 0.001 --timeframe 60
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import argparse
from dataclasses import dataclass
from typing import List, Dict
from enum import IntEnum
import logging
from pykalman import KalmanFilter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTES (copiées de prepare_data_direction_only.py)
# =============================================================================

# Périodes STANDARD des indicateurs
RSI_PERIOD = 14
CCI_PERIOD = 20
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Kalman
KALMAN_PROCESS_VAR = 0.01
KALMAN_MEASURE_VAR = 0.1

# Assets
ASSET_FILES = {
    'BTC': 'data_trad/BTCUSD_all_5m.csv',
    'ETH': 'data_trad/ETHUSD_all_5m.csv',
    'BNB': 'data_trad/BNBUSD_all_5m.csv',
    'ADA': 'data_trad/ADAUSD_all_5m.csv',
    'LTC': 'data_trad/LTCUSD_all_5m.csv',
}

ASSET_ID_MAP = {'BTC': 0, 'ETH': 1, 'BNB': 2, 'ADA': 3, 'LTC': 4}


# =============================================================================
# TYPES (copiés de test_oracle_direction_only.py)
# =============================================================================

class Position(IntEnum):
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    duration: int
    position: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_after_fees: float
    asset_id: int = 0
    entry_timestamp: float = 0.0


@dataclass
class BacktestResult:
    n_trades: int
    n_long: int
    n_short: int
    total_pnl: float
    total_pnl_after_fees: float
    total_fees: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_duration: float
    sharpe_ratio: float
    max_drawdown: float
    trades: List[Trade]


@dataclass
class AssetResult:
    asset_id: int
    asset_name: str
    n_trades: int
    total_pnl: float
    total_pnl_after_fees: float
    win_rate: float
    avg_duration: float


@dataclass
class MonthlyResult:
    year_month: str
    n_trades: int
    total_pnl: float
    total_pnl_after_fees: float
    win_rate: float


# =============================================================================
# CHARGEMENT ET RESAMPLING
# =============================================================================

def load_csv_5min(file_path: str, asset_name: str) -> pd.DataFrame:
    """Charge données 5min brutes depuis CSV."""
    df = pd.read_csv(file_path)

    # Trouver colonne date
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


def resample_to_tf(df_5min: pd.DataFrame, tf_minutes: int = 30) -> pd.DataFrame:
    """
    Resample données 5min vers timeframe supérieur.

    Args:
        df_5min: DataFrame 5min avec DatetimeIndex
        tf_minutes: Timeframe cible en minutes (30, 60, etc.)

    Returns:
        DataFrame resampleé avec DatetimeIndex
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

    logger.info(f"    Resample {tf_minutes}min: {len(df_tf):,} bougies")

    return df_tf


# =============================================================================
# CALCUL INDICATEURS (copié de prepare_data_direction_only.py)
# =============================================================================

def calculate_indicator(df: pd.DataFrame, indicator: str) -> pd.Series:
    """Calcule un indicateur technique sur un DataFrame OHLCV."""
    if indicator == 'rsi':
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.ewm(span=RSI_PERIOD, adjust=False).mean()
        avg_loss = loss.ewm(span=RSI_PERIOD, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    elif indicator == 'cci':
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(CCI_PERIOD).mean()
        mad = tp.rolling(CCI_PERIOD).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - sma_tp) / (0.015 * mad)

    elif indicator == 'macd':
        ema_fast = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        return macd_line - signal_line

    else:
        raise ValueError(f"Indicateur inconnu: {indicator}")


# =============================================================================
# FILTRE KALMAN (copié de prepare_data_direction_only.py)
# =============================================================================

def kalman_filter_dual(data: np.ndarray,
                       process_var: float = KALMAN_PROCESS_VAR,
                       measure_var: float = KALMAN_MEASURE_VAR) -> np.ndarray:
    """Applique un filtre de Kalman CINÉMATIQUE (position + vélocité)."""
    valid_mask = ~np.isnan(data)
    if valid_mask.sum() < 10:
        return np.full((len(data), 2), np.nan)

    transition_matrix = [[1, 1], [0, 1]]
    observation_matrix = [[1, 0]]
    initial_state_mean = [data[valid_mask][0], 0.0]
    observation_covariance = measure_var
    transition_covariance = np.eye(2) * process_var

    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        initial_state_mean=initial_state_mean,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance
    )

    means, _ = kf.smooth(data[valid_mask])

    result = np.full((len(data), 2), np.nan)
    result[valid_mask] = means

    return result


# =============================================================================
# PIPELINE: CSV 5min → Labels 30min → Backtest sur prix 5min
# =============================================================================

def generate_labels_for_asset(
    asset_name: str,
    indicator: str,
    tf_minutes: int = 30,
    split_ratios: tuple = (0.70, 0.15, 0.15)
) -> Dict:
    """
    Pipeline complet pour un asset:
    1. Charger 5min CSV
    2. Resampler en tf_minutes
    3. Calculer indicateur sur tf_minutes
    4. Appliquer Kalman
    5. Calculer labels direction (filtered[t] > filtered[t-1])
    6. Forward-fill vers 5min
    7. Retourner labels + prix 5min pour backtest

    Returns:
        Dict avec 'labels_5min', 'opens_5min', 'timestamps_5min' pour chaque split
    """
    file_path = ASSET_FILES[asset_name]
    asset_id = ASSET_ID_MAP[asset_name]

    logger.info(f"\n  === {asset_name} (asset_id={asset_id}) ===")

    # 1. Charger 5min
    df_5min = load_csv_5min(file_path, asset_name)

    # 2. Resampler
    df_tf = resample_to_tf(df_5min, tf_minutes)

    # 3. Calculer indicateur sur tf_minutes
    indicator_values = calculate_indicator(df_tf, indicator)
    indicator_values = indicator_values.fillna(0)
    logger.info(f"    Indicateur {indicator.upper()} calculé sur {tf_minutes}min")

    # 4. Appliquer Kalman sur indicateur tf
    kalman_result = kalman_filter_dual(indicator_values.values)
    filtered = kalman_result[:, 0]  # Position filtrée
    logger.info(f"    Kalman appliqué")

    # 5. Labels direction: filtered[t] > filtered[t-1] (formule Phase 2.15)
    filtered_series = pd.Series(filtered, index=df_tf.index)
    labels_tf = (filtered_series > filtered_series.shift(1)).astype(int)
    labels_tf.iloc[0] = 0  # Premier label inconnu

    n_up = (labels_tf == 1).sum()
    n_down = (labels_tf == 0).sum()
    logger.info(f"    Labels {tf_minutes}min: {len(labels_tf):,} ({n_up:,} UP, {n_down:,} DOWN, {n_up/(n_up+n_down)*100:.1f}% UP)")

    # 6. Forward-fill vers 5min
    # Chaque bougie tf a un timestamp (ex: 10:00 pour bougie 10:00-10:29)
    # On assigne ce label à toutes les bougies 5min de cette période
    # CAUSALITÉ: Le label de la bougie 10:00 est connu seulement APRÈS 10:29
    # Donc on l'applique à partir de 10:30 (= shift de 1 bougie tf)
    labels_tf_shifted = labels_tf.shift(1)  # Causalité: label connu après complétion
    labels_tf_shifted.iloc[0] = 0

    # Reindexer sur l'index 5min avec forward-fill
    labels_5min = labels_tf_shifted.reindex(df_5min.index, method='ffill')
    labels_5min = labels_5min.fillna(0).astype(int)

    logger.info(f"    Forward-fill vers 5min: {len(labels_5min):,} samples")

    # Vérifier alignement
    assert len(labels_5min) == len(df_5min), f"Mismatch: {len(labels_5min)} vs {len(df_5min)}"

    # Stats labels 5min
    n_up_5m = (labels_5min == 1).sum()
    n_down_5m = (labels_5min == 0).sum()
    n_changes = (labels_5min.diff().abs() > 0).sum()
    logger.info(f"    Labels 5min: {n_up_5m:,} UP, {n_down_5m:,} DOWN, {n_changes:,} changements direction")

    # 7. Split temporel (même logique que le projet)
    n_total = len(df_5min)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])

    result = {
        'asset_name': asset_name,
        'asset_id': asset_id,
        'train': {
            'labels': labels_5min.values[:n_train],
            'opens': df_5min['open'].values[:n_train],
            'timestamps': df_5min.index[:n_train].astype(np.int64) / 1e9,
        },
        'val': {
            'labels': labels_5min.values[n_train:n_train + n_val],
            'opens': df_5min['open'].values[n_train:n_train + n_val],
            'timestamps': df_5min.index[n_train:n_train + n_val].astype(np.int64) / 1e9,
        },
        'test': {
            'labels': labels_5min.values[n_train + n_val:],
            'opens': df_5min['open'].values[n_train + n_val:],
            'timestamps': df_5min.index[n_train + n_val:].astype(np.int64) / 1e9,
        }
    }

    for split_name in ['train', 'val', 'test']:
        n = len(result[split_name]['labels'])
        logger.info(f"    Split {split_name}: {n:,} samples")

    return result


# =============================================================================
# BACKTEST (copié de test_oracle_direction_only.py)
# =============================================================================

def backtest_single_asset(labels, opens, timestamps, asset_id, fees=0.001):
    """Backtest pour UN SEUL asset. Copié de test_oracle_direction_only.py."""
    n_samples = len(labels)
    trades = []
    position = Position.FLAT
    entry_idx = 0
    entry_price = 0.0
    entry_timestamp = 0.0

    for i in range(n_samples - 1):
        direction = int(labels[i])
        target = Position.LONG if direction == 1 else Position.SHORT

        if position == Position.FLAT:
            position = target
            entry_idx = i
            entry_price = opens[i + 1]
            entry_timestamp = timestamps[i + 1]
            continue

        if position != target:
            exit_price = opens[i + 1]

            if position == Position.LONG:
                pnl = (exit_price - entry_price) / entry_price
            else:
                pnl = (entry_price - exit_price) / entry_price

            trade_fees = 2 * fees
            pnl_after_fees = pnl - trade_fees

            trades.append(Trade(
                entry_idx=entry_idx,
                exit_idx=i,
                duration=i - entry_idx,
                position='LONG' if position == Position.LONG else 'SHORT',
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                pnl_after_fees=pnl_after_fees,
                asset_id=asset_id,
                entry_timestamp=entry_timestamp
            ))

            position = target
            entry_idx = i
            entry_price = opens[i + 1]
            entry_timestamp = timestamps[i + 1]

    # Fermer position finale
    if position != Position.FLAT:
        exit_price = opens[-1]

        if position == Position.LONG:
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price

        trade_fees = 2 * fees
        pnl_after_fees = pnl - trade_fees

        trades.append(Trade(
            entry_idx=entry_idx,
            exit_idx=n_samples - 1,
            duration=n_samples - 1 - entry_idx,
            position='LONG' if position == Position.LONG else 'SHORT',
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_after_fees=pnl_after_fees,
            asset_id=asset_id,
            entry_timestamp=entry_timestamp
        ))

    return trades


def compute_stats(trades, n_long, n_short):
    """Calcule les statistiques du backtest. Copié de test_oracle_direction_only.py."""
    if len(trades) == 0:
        return BacktestResult(
            n_trades=0, n_long=0, n_short=0,
            total_pnl=0.0, total_pnl_after_fees=0.0, total_fees=0.0,
            win_rate=0.0, profit_factor=0.0,
            avg_win=0.0, avg_loss=0.0, avg_duration=0.0,
            sharpe_ratio=0.0, max_drawdown=0.0, trades=[]
        )

    pnls = np.array([t.pnl for t in trades])
    pnls_net = np.array([t.pnl_after_fees for t in trades])
    durations = np.array([t.duration for t in trades])

    total_pnl = pnls.sum()
    total_pnl_net = pnls_net.sum()
    total_fees = total_pnl - total_pnl_net

    wins = pnls_net > 0
    losses = pnls_net < 0
    win_rate = wins.mean() if len(trades) > 0 else 0.0

    sum_wins = pnls_net[wins].sum() if wins.any() else 0.0
    sum_losses = abs(pnls_net[losses].sum()) if losses.any() else 0.0
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else 0.0

    avg_win = pnls_net[wins].mean() if wins.any() else 0.0
    avg_loss = pnls_net[losses].mean() if losses.any() else 0.0
    avg_duration = durations.mean()

    if len(pnls_net) > 1 and pnls_net.std() > 0:
        sharpe = (pnls_net.mean() / pnls_net.std()) * np.sqrt(288 * 365)
    else:
        sharpe = 0.0

    cumulative = np.cumsum(pnls_net)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = drawdowns.max() if len(drawdowns) > 0 else 0.0

    return BacktestResult(
        n_trades=len(trades),
        n_long=n_long,
        n_short=n_short,
        total_pnl=total_pnl,
        total_pnl_after_fees=total_pnl_net,
        total_fees=total_fees,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_duration=avg_duration,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        trades=trades
    )


# =============================================================================
# STATS MENSUELLES (copié de test_oracle_direction_only.py)
# =============================================================================

def compute_monthly_stats(trades):
    """Calcule les statistiques par mois."""
    from datetime import datetime
    from collections import defaultdict

    monthly_data = defaultdict(list)

    for trade in trades:
        ts = trade.entry_timestamp
        if ts > 1e18:
            ts = ts / 1e9
        elif ts > 1e15:
            ts = ts / 1e6
        elif ts > 1e12:
            ts = ts / 1e3

        try:
            dt = datetime.fromtimestamp(ts)
            year_month = dt.strftime('%Y-%m')
            monthly_data[year_month].append(trade)
        except (ValueError, OSError):
            continue

    monthly_results = []
    for year_month in sorted(monthly_data.keys()):
        month_trades = monthly_data[year_month]
        n_trades = len(month_trades)
        total_pnl = sum(t.pnl for t in month_trades)
        total_pnl_net = sum(t.pnl_after_fees for t in month_trades)
        wins = sum(1 for t in month_trades if t.pnl_after_fees > 0)
        win_rate = wins / n_trades if n_trades > 0 else 0.0

        monthly_results.append(MonthlyResult(
            year_month=year_month,
            n_trades=n_trades,
            total_pnl=total_pnl,
            total_pnl_after_fees=total_pnl_net,
            win_rate=win_rate
        ))

    return monthly_results


# =============================================================================
# AFFICHAGE (copié de test_oracle_direction_only.py)
# =============================================================================

def print_results(result, indicator, tf_minutes, mode):
    """Affiche les résultats du backtest."""
    print("\n" + "=" * 70)
    print(f"RÉSULTATS {mode.upper()} - {indicator.upper()} {tf_minutes}min PUR")
    print("=" * 70)

    print(f"\nTrades:")
    print(f"  Total: {result.n_trades:,}")
    if result.n_trades > 0:
        print(f"  Long: {result.n_long:,} ({result.n_long / result.n_trades * 100:.1f}%)")
        print(f"  Short: {result.n_short:,} ({result.n_short / result.n_trades * 100:.1f}%)")
    print(f"  Durée moyenne: {result.avg_duration:.1f} périodes (~{result.avg_duration * 5:.0f} min)")

    print(f"\nPerformance:")
    print(f"  PnL Brut: {result.total_pnl * 100:+.2f}%")
    print(f"  Frais: {result.total_fees * 100:.2f}%")
    print(f"  PnL Net: {result.total_pnl_after_fees * 100:+.2f}%")

    print(f"\nMétriques:")
    print(f"  Win Rate: {result.win_rate * 100:.1f}%")
    print(f"  Profit Factor: {result.profit_factor:.2f}")
    print(f"  Avg Win: {result.avg_win * 100:+.3f}%")
    print(f"  Avg Loss: {result.avg_loss * 100:+.3f}%")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown * 100:.2f}%")

    print(f"\nVerdict:")
    if result.total_pnl_after_fees > 0:
        print(f"  ✅ PnL Net POSITIF!")
    else:
        print(f"  ❌ PnL Net NÉGATIF")


def print_asset_results(asset_results):
    """Affiche les résultats par asset."""
    print("\n" + "=" * 70)
    print("RÉSULTATS PAR ASSET")
    print("=" * 70)

    print(f"\n{'Asset':<8} {'Trades':>10} {'PnL Brut':>12} {'PnL Net':>12} {'Win Rate':>10} {'Durée Moy':>10}")
    print("-" * 70)

    for ar in asset_results:
        print(f"{ar.asset_name:<8} {ar.n_trades:>10,} {ar.total_pnl * 100:>+11.2f}% "
              f"{ar.total_pnl_after_fees * 100:>+11.2f}% {ar.win_rate * 100:>9.1f}% {ar.avg_duration:>9.1f}p")

    if asset_results:
        avg_pnl = sum(ar.total_pnl for ar in asset_results) / len(asset_results)
        avg_pnl_net = sum(ar.total_pnl_after_fees for ar in asset_results) / len(asset_results)
        avg_wr = sum(ar.win_rate for ar in asset_results) / len(asset_results)
        avg_dur = sum(ar.avg_duration for ar in asset_results) / len(asset_results)

        print("-" * 70)
        print(f"{'MOYENNE':<8} {'':>10} {avg_pnl * 100:>+11.2f}% "
              f"{avg_pnl_net * 100:>+11.2f}% {avg_wr * 100:>9.1f}% {avg_dur:>9.1f}p")


def print_monthly_results(trades):
    """Affiche les résultats par mois."""
    monthly_results = compute_monthly_stats(trades)

    print("\n" + "=" * 70)
    print("RÉSULTATS PAR MOIS")
    print("=" * 70)

    print(f"\n{'Mois':<10} {'Trades':>10} {'PnL Brut':>12} {'PnL Net':>12} {'Win Rate':>10}")
    print("-" * 60)

    for mr in monthly_results:
        print(f"{mr.year_month:<10} {mr.n_trades:>10,} {mr.total_pnl * 100:>+11.2f}% "
              f"{mr.total_pnl_after_fees * 100:>+11.2f}% {mr.win_rate * 100:>9.1f}%")

    if monthly_results:
        avg_trades = sum(mr.n_trades for mr in monthly_results) / len(monthly_results)
        avg_pnl_net = sum(mr.total_pnl_after_fees for mr in monthly_results) / len(monthly_results)
        print("-" * 60)
        print(f"{'MOYENNE':<10} {avg_trades:>10,.0f} {'':>12} {avg_pnl_net * 100:>+11.2f}%")
        print(f"\nNombre de mois: {len(monthly_results)}")


def print_comparison_5min(result, indicator, tf_minutes):
    """Affiche la comparaison avec les résultats Oracle 5min connus."""
    print("\n" + "=" * 70)
    print(f"COMPARAISON: Oracle 5min vs Oracle {tf_minutes}min")
    print("=" * 70)

    # Résultats Oracle 5min Phase 2.15 (depuis CLAUDE.md)
    ref_5min = {
        'macd': {'trades': 68924, 'wr': 53.4, 'pnl_brut': 28144, 'pnl_net': 14359, 'pf': 2.79, 'sharpe': 85.44},
        'cci':  {'trades': 82405, 'wr': 56.4, 'pnl_brut': 33816, 'pnl_net': 17335, 'pf': 3.16, 'sharpe': 87.55},
        'rsi':  {'trades': 96886, 'wr': 57.3, 'pnl_brut': 42417, 'pnl_net': 23039, 'pf': 4.02, 'sharpe': 102.67},
    }

    ref = ref_5min.get(indicator, None)
    if ref is None:
        print("  (pas de référence 5min disponible)")
        return

    print(f"\n{'Métrique':<20} {'Oracle 5min':>15} {'Oracle ' + str(tf_minutes) + 'min':>15} {'Delta':>12}")
    print("-" * 65)
    print(f"{'Trades':<20} {ref['trades']:>15,} {result.n_trades:>15,} {result.n_trades - ref['trades']:>+12,}")
    print(f"{'Win Rate':<20} {ref['wr']:>14.1f}% {result.win_rate * 100:>14.1f}% {result.win_rate * 100 - ref['wr']:>+11.1f}%")
    print(f"{'PnL Brut':<20} {ref['pnl_brut']:>+14.0f}% {result.total_pnl * 100:>+14.0f}% {'':>12}")
    print(f"{'PnL Net':<20} {ref['pnl_net']:>+14.0f}% {result.total_pnl_after_fees * 100:>+14.0f}% {'':>12}")
    print(f"{'Profit Factor':<20} {ref['pf']:>15.2f} {result.profit_factor:>15.2f} {'':>12}")
    print(f"{'Sharpe':<20} {ref['sharpe']:>15.2f} {result.sharpe_ratio:>15.2f} {'':>12}")

    # Verdict
    trade_reduction = (1 - result.n_trades / ref['trades']) * 100
    print(f"\n  Réduction trades: {trade_reduction:+.1f}%")
    if result.total_pnl_after_fees > 0:
        print(f"  ✅ PnL Net POSITIF avec {tf_minutes}min pur!")
    else:
        print(f"  ❌ PnL Net NÉGATIF avec {tf_minutes}min pur")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test Oracle 30min Pur')
    parser.add_argument('--indicator', type=str, default='macd',
                        choices=['macd', 'rsi', 'cci'],
                        help='Indicateur à tester (défaut: macd)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Split à utiliser (défaut: test)')
    parser.add_argument('--fees', type=float, default=0.001,
                        help='Frais par side (défaut: 0.1%%)')
    parser.add_argument('--timeframe', type=int, default=30,
                        help='Timeframe en minutes (défaut: 30)')
    parser.add_argument('--assets', nargs='+', default=['BTC', 'ETH', 'BNB', 'ADA', 'LTC'],
                        help='Assets à tester (défaut: tous)')

    args = parser.parse_args()

    tf = args.timeframe

    logger.info("=" * 70)
    logger.info(f"TEST ORACLE {tf}min PUR - {args.indicator.upper()}")
    logger.info("=" * 70)
    logger.info(f"Indicateur: {args.indicator.upper()} calculé sur {tf}min")
    logger.info(f"Labels: filtered[t] > filtered[t-1] (formule Phase 2.15)")
    logger.info(f"Split: {args.split}")
    logger.info(f"Frais: {args.fees * 100:.2f}% par side ({args.fees * 2 * 100:.2f}% round-trip)")
    logger.info(f"Assets: {args.assets}")
    logger.info(f"Pas de features 5min - indicateurs {tf}min UNIQUEMENT")

    # Générer labels pour chaque asset
    all_trades = []
    asset_results = []
    n_long = 0
    n_short = 0

    for asset_name in args.assets:
        if asset_name not in ASSET_FILES:
            logger.warning(f"Asset {asset_name} non trouvé, skip")
            continue

        # Pipeline: CSV 5min → indicateur tf → Kalman → labels → forward-fill 5min
        asset_data = generate_labels_for_asset(asset_name, args.indicator, tf)

        # Extraire le split demandé
        split_data = asset_data[args.split]
        labels = split_data['labels']
        opens = split_data['opens']
        timestamps = split_data['timestamps']

        logger.info(f"\n  Backtest {asset_name} ({args.split}): {len(labels):,} samples")

        # Backtest
        trades = backtest_single_asset(
            labels, opens, timestamps, asset_data['asset_id'], args.fees
        )

        # Stats par asset
        asset_pnl = sum(t.pnl for t in trades)
        asset_pnl_net = sum(t.pnl_after_fees for t in trades)
        asset_wins = sum(1 for t in trades if t.pnl_after_fees > 0)
        asset_duration = sum(t.duration for t in trades)

        for t in trades:
            if t.position == 'LONG':
                n_long += 1
            else:
                n_short += 1

        if len(trades) > 0:
            asset_results.append(AssetResult(
                asset_id=asset_data['asset_id'],
                asset_name=asset_name,
                n_trades=len(trades),
                total_pnl=asset_pnl,
                total_pnl_after_fees=asset_pnl_net,
                win_rate=asset_wins / len(trades),
                avg_duration=asset_duration / len(trades)
            ))

        all_trades.extend(trades)

    # Stats globales
    result = compute_stats(all_trades, n_long, n_short)

    # Affichage
    print_results(result, args.indicator, tf, "Oracle")
    print_asset_results(asset_results)
    print_monthly_results(result.trades)
    print_comparison_5min(result, args.indicator, tf)


if __name__ == '__main__':
    main()
