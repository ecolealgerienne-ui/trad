#!/usr/bin/env python3
"""
Test Oracle Multi-Timeframe - Kalman appliqué à résolution 5min sur indicateurs 30min/1h.

DIFFÉRENCE vs test_oracle_30min_pure.py:
- Ancien: Kalman sur ~147k points (30min) → labels 30min → forward-fill 5min
- Nouveau: Kalman sur ~880k points (5min, valeurs 30min forward-fillées) → labels 5min

AVANTAGE ATTENDU:
Le Kalman à résolution 5min crée des transitions PROGRESSIVES entre bougies 30min.
Au lieu d'un saut brutal toutes les 6 bougies, le signal filtré évolue graduellement.
Le label peut changer EN COURS de bougie 30min → meilleur timing d'entrée/sortie.

PIPELINE:
1. Charger CSV multitf pré-calculé (BTCUSD_multitf.csv, etc.)
2. Extraire colonne indicateur (macd_30m, rsi_30m, etc.)
3. Appliquer Kalman à résolution 5min sur ces valeurs
4. Labels: filtered[t] > filtered[t-1] (formule Phase 2.15)
5. Backtest sur prix 5min

Usage:
    # 30min avec Kalman à résolution 5min
    python tests/test_oracle_multitf.py --indicator macd --timeframe 30m --fees 0.001

    # 1h avec Kalman à résolution 5min
    python tests/test_oracle_multitf.py --indicator macd --timeframe 1h --fees 0.001

    # Comparer tous
    python tests/test_oracle_multitf.py --indicator macd --timeframe 30m --fees 0.001
    python tests/test_oracle_multitf.py --indicator macd --timeframe 1h --fees 0.001
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
# CONSTANTES
# =============================================================================

KALMAN_PROCESS_VAR = 0.01
KALMAN_MEASURE_VAR = 0.1

ASSET_CSV_FILES = {
    'BTC': 'data/prepared/BTCUSD_multitf.csv',
    'ETH': 'data/prepared/ETHUSD_multitf.csv',
    'BNB': 'data/prepared/BNBUSD_multitf.csv',
    'ADA': 'data/prepared/ADAUSD_multitf.csv',
    'LTC': 'data/prepared/LTCUSD_multitf.csv',
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
# CHARGEMENT CSV MULTITF
# =============================================================================

def load_multitf_csv(asset_name: str) -> pd.DataFrame:
    """Charge le CSV multi-timeframe pré-calculé."""
    file_path = ASSET_CSV_FILES[asset_name]

    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"CSV multitf introuvable: {file_path}\n"
            f"Lancez d'abord: python src/prepare_multitf_csv.py --assets {asset_name}"
        )

    df = pd.read_csv(file_path, parse_dates=['datetime'])
    df = df.set_index('datetime')
    df = df.sort_index()

    logger.info(f"  {asset_name}: {len(df):,} lignes, {df.index[0]} → {df.index[-1]}")

    return df


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
# PIPELINE: CSV multitf → Kalman 5min → Labels → Backtest
# =============================================================================

def generate_labels_from_multitf(
    asset_name: str,
    indicator: str,
    timeframe: str,
    split_ratios: tuple = (0.70, 0.15, 0.15)
) -> Dict:
    """
    Pipeline:
    1. Charger CSV multitf
    2. Extraire colonne indicateur_{timeframe} (ex: macd_30m)
    3. Appliquer Kalman à résolution 5min
    4. Labels: filtered[t] > filtered[t-1]
    5. Split et retour

    Args:
        indicator: 'macd', 'rsi', 'cci'
        timeframe: '30m' ou '1h'
    """
    asset_id = ASSET_ID_MAP[asset_name]
    col_name = f'{indicator}_{timeframe}'

    logger.info(f"\n  === {asset_name} (asset_id={asset_id}) ===")

    # 1. Charger CSV
    df = load_multitf_csv(asset_name)

    # 2. Extraire colonne indicateur
    if col_name not in df.columns:
        raise ValueError(f"Colonne '{col_name}' introuvable. Colonnes: {list(df.columns)}")

    indicator_values = df[col_name].values
    n_nan_before = np.isnan(indicator_values).sum()
    logger.info(f"    Colonne {col_name}: {len(indicator_values):,} valeurs, {n_nan_before:,} NaN (warm-up)")

    # 3. Appliquer Kalman à résolution 5min
    logger.info(f"    Application Kalman à résolution 5min ({len(indicator_values):,} points)...")
    kalman_result = kalman_filter_dual(indicator_values)
    filtered = kalman_result[:, 0]
    logger.info(f"    Kalman appliqué")

    # 4. Labels: filtered[t] > filtered[t-1]
    filtered_series = pd.Series(filtered, index=df.index)
    labels = (filtered_series > filtered_series.shift(1)).astype(float)
    labels.iloc[0] = 0

    # Remplacer NaN par 0 (début du dataset, warm-up)
    labels = labels.fillna(0).astype(int)

    # Stats
    n_valid = (~np.isnan(filtered)).sum()
    n_up = (labels == 1).sum()
    n_down = (labels == 0).sum()
    n_changes = (labels.diff().abs() > 0).sum()
    logger.info(f"    Labels 5min: {n_up:,} UP, {n_down:,} DOWN ({n_up/(n_up+n_down)*100:.1f}% UP)")
    logger.info(f"    Changements direction: {n_changes:,}")

    # Comparer avec le nombre de changements si on avait des labels 30min purs
    step_col = f'step_{timeframe}'
    if step_col in df.columns:
        n_step1 = (df[step_col] == 1).sum()
        logger.info(f"    (Référence: {n_step1:,} bougies {timeframe} complètes → max {n_step1:,} changements en mode forward-fill pur)")
        logger.info(f"    Gain résolution: {n_changes:,} changements vs ~{n_step1:,} max en mode pur → Kalman ajoute {n_changes - n_step1:+,} transitions intra-bougie)")

    # 5. Split temporel
    opens = df['open'].values
    timestamps = df.index.astype(np.int64) / 1e9

    n_total = len(df)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])

    result = {
        'asset_name': asset_name,
        'asset_id': asset_id,
    }

    for split_name, start, end in [
        ('train', 0, n_train),
        ('val', n_train, n_train + n_val),
        ('test', n_train + n_val, n_total)
    ]:
        result[split_name] = {
            'labels': labels.values[start:end],
            'opens': opens[start:end],
            'timestamps': timestamps[start:end],
        }
        logger.info(f"    Split {split_name}: {end - start:,} samples")

    return result


# =============================================================================
# BACKTEST (copié de test_oracle_30min_pure.py)
# =============================================================================

def backtest_single_asset(labels, opens, timestamps, asset_id, fees=0.001):
    """Backtest pour UN SEUL asset."""
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
                entry_idx=entry_idx, exit_idx=i, duration=i - entry_idx,
                position='LONG' if position == Position.LONG else 'SHORT',
                entry_price=entry_price, exit_price=exit_price,
                pnl=pnl, pnl_after_fees=pnl_after_fees,
                asset_id=asset_id, entry_timestamp=entry_timestamp
            ))

            position = target
            entry_idx = i
            entry_price = opens[i + 1]
            entry_timestamp = timestamps[i + 1]

    if position != Position.FLAT:
        exit_price = opens[-1]
        if position == Position.LONG:
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price
        trade_fees = 2 * fees
        pnl_after_fees = pnl - trade_fees
        trades.append(Trade(
            entry_idx=entry_idx, exit_idx=n_samples - 1, duration=n_samples - 1 - entry_idx,
            position='LONG' if position == Position.LONG else 'SHORT',
            entry_price=entry_price, exit_price=exit_price,
            pnl=pnl, pnl_after_fees=pnl_after_fees,
            asset_id=asset_id, entry_timestamp=entry_timestamp
        ))

    return trades


def compute_stats(trades, n_long, n_short):
    """Calcule les statistiques du backtest."""
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
        n_trades=len(trades), n_long=n_long, n_short=n_short,
        total_pnl=total_pnl, total_pnl_after_fees=total_pnl_net, total_fees=total_fees,
        win_rate=win_rate, profit_factor=profit_factor,
        avg_win=avg_win, avg_loss=avg_loss, avg_duration=avg_duration,
        sharpe_ratio=sharpe, max_drawdown=max_drawdown, trades=trades
    )


# =============================================================================
# STATS MENSUELLES
# =============================================================================

def compute_monthly_stats(trades):
    from datetime import datetime
    from collections import defaultdict

    monthly_data = defaultdict(list)
    for trade in trades:
        ts = trade.entry_timestamp
        if ts > 1e18: ts = ts / 1e9
        elif ts > 1e15: ts = ts / 1e6
        elif ts > 1e12: ts = ts / 1e3
        try:
            dt = datetime.fromtimestamp(ts)
            monthly_data[dt.strftime('%Y-%m')].append(trade)
        except (ValueError, OSError):
            continue

    results = []
    for ym in sorted(monthly_data.keys()):
        mt = monthly_data[ym]
        n = len(mt)
        pnl = sum(t.pnl for t in mt)
        pnl_net = sum(t.pnl_after_fees for t in mt)
        wins = sum(1 for t in mt if t.pnl_after_fees > 0)
        results.append(MonthlyResult(ym, n, pnl, pnl_net, wins / n if n > 0 else 0))
    return results


# =============================================================================
# AFFICHAGE
# =============================================================================

def print_results(result, indicator, timeframe, mode):
    print("\n" + "=" * 70)
    print(f"RÉSULTATS {mode.upper()} - {indicator.upper()} {timeframe} (Kalman 5min)")
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


def print_comparison(result, indicator, timeframe):
    """Compare avec les résultats précédents (5min Oracle et 30min/1h pur)."""
    print("\n" + "=" * 70)
    print(f"COMPARAISON: 3 APPROCHES")
    print("=" * 70)

    # Références connues (CLAUDE.md + résultats précédents)
    refs = {
        'macd': {
            '5min Oracle':       {'trades': 68924,  'wr': 53.4, 'pnl_net': 14359, 'pf': 2.79,  'sharpe': 85.44},
            '30min pur (Kalman 30min)': {'trades': 11158, 'wr': 64.2, 'pnl_net': 8316,  'pf': 4.76,  'sharpe': 133.62},
            '1h pur (Kalman 1h)':      {'trades': 5422,  'wr': 70.8, 'pnl_net': 7083,  'pf': 6.72,  'sharpe': 161.38},
        }
    }

    ref_data = refs.get(indicator, {})

    print(f"\n{'Approche':<30} {'Trades':>10} {'Win Rate':>10} {'PnL Net':>12} {'PF':>8} {'Sharpe':>10}")
    print("-" * 82)

    for name, ref in ref_data.items():
        print(f"{name:<30} {ref['trades']:>10,} {ref['wr']:>9.1f}% {ref['pnl_net']:>+11.0f}% {ref['pf']:>8.2f} {ref['sharpe']:>10.2f}")

    # Résultat actuel
    current_name = f"{timeframe} (Kalman 5min)"
    print(f"{current_name:<30} {result.n_trades:>10,} {result.win_rate * 100:>9.1f}% "
          f"{result.total_pnl_after_fees * 100:>+11.0f}% {result.profit_factor:>8.2f} {result.sharpe_ratio:>10.2f}")
    print("-" * 82)
    print(f"  ← NOUVEAU")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test Oracle Multi-Timeframe (Kalman 5min)')
    parser.add_argument('--indicator', type=str, default='macd',
                        choices=['macd', 'rsi', 'cci'],
                        help='Indicateur (défaut: macd)')
    parser.add_argument('--timeframe', type=str, default='30m',
                        choices=['30m', '1h'],
                        help='Timeframe des indicateurs (défaut: 30m)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Split (défaut: test)')
    parser.add_argument('--fees', type=float, default=0.001,
                        help='Frais par side (défaut: 0.1%%)')
    parser.add_argument('--assets', nargs='+',
                        default=['BTC', 'ETH', 'BNB', 'ADA', 'LTC'],
                        help='Assets (défaut: tous)')

    args = parser.parse_args()

    col_name = f'{args.indicator}_{args.timeframe}'

    logger.info("=" * 70)
    logger.info(f"TEST ORACLE MULTITF - {args.indicator.upper()} {args.timeframe} (Kalman à résolution 5min)")
    logger.info("=" * 70)
    logger.info(f"Colonne source: {col_name}")
    logger.info(f"Kalman appliqué sur {col_name} à résolution 5min (~880k points/asset)")
    logger.info(f"Labels: filtered[t] > filtered[t-1] à résolution 5min")
    logger.info(f"Split: {args.split}")
    logger.info(f"Frais: {args.fees * 100:.2f}% par side ({args.fees * 2 * 100:.2f}% round-trip)")
    logger.info(f"Assets: {args.assets}")

    all_trades = []
    asset_results = []
    n_long = 0
    n_short = 0

    for asset_name in args.assets:
        if asset_name not in ASSET_CSV_FILES:
            logger.warning(f"Asset {asset_name} non trouvé, skip")
            continue

        # Pipeline: CSV multitf → Kalman 5min → labels → backtest
        asset_data = generate_labels_from_multitf(
            asset_name, args.indicator, args.timeframe
        )

        split_data = asset_data[args.split]
        labels = split_data['labels']
        opens = split_data['opens']
        timestamps = split_data['timestamps']

        logger.info(f"\n  Backtest {asset_name} ({args.split}): {len(labels):,} samples")

        trades = backtest_single_asset(
            labels, opens, timestamps, asset_data['asset_id'], args.fees
        )

        asset_pnl = sum(t.pnl for t in trades)
        asset_pnl_net = sum(t.pnl_after_fees for t in trades)
        asset_wins = sum(1 for t in trades if t.pnl_after_fees > 0)
        asset_duration = sum(t.duration for t in trades)

        for t in trades:
            if t.position == 'LONG': n_long += 1
            else: n_short += 1

        if len(trades) > 0:
            asset_results.append(AssetResult(
                asset_id=asset_data['asset_id'], asset_name=asset_name,
                n_trades=len(trades), total_pnl=asset_pnl,
                total_pnl_after_fees=asset_pnl_net,
                win_rate=asset_wins / len(trades),
                avg_duration=asset_duration / len(trades)
            ))

        all_trades.extend(trades)

    result = compute_stats(all_trades, n_long, n_short)

    print_results(result, args.indicator, args.timeframe, "Oracle")
    print_asset_results(asset_results)
    print_monthly_results(result.trades)
    print_comparison(result, args.indicator, args.timeframe)


if __name__ == '__main__':
    main()
