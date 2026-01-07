#!/usr/bin/env python3
"""
Test de comparaison des filtres Octave 0.2 vs 0.25.

OBJECTIF:
    Comparer 2 stratégies de trading basées sur le filtre Octave :
    1. Stratégie Classique : Utilise filt_02[t-2] > filt_02[t-3]
    2. Stratégie Différence : Utilise diff[t-2] > diff[t-3]
       où diff = filt_02 - filt_025

PIPELINE:
    1. Charger 10,000 valeurs BTC
    2. Trim ±200 bougies
    3. Calculer indicateurs (RSI, CCI, MACD)
    4. Appliquer 2 filtres Octave :
       - Octave(step=0.2) → filt_02
       - Octave(step=0.25) → filt_025
    5. Calculer diff = filt_02 - filt_025
    6. Backtest 2 stratégies (6 combinaisons au total)
    7. Comparer Win Rate, PnL, Profit Factor

STRATÉGIES TESTÉES (6 combinaisons):
    Pour chaque indicateur (RSI, CCI, MACD):
    - Stratégie 1 (Classique) : filt_02[t-2] > filt_02[t-3] → BUY/SELL
    - Stratégie 2 (Différence) : diff > 0 → BUY, diff < 0 → SELL
      (où diff = filt_02 - filt_025)

DONNÉES:
    Source: data_trad/BTCUSD_all_5m.csv
    Taille: 10,000 bougies (après trim: 9,600)
    Trim: ±200 bougies (warmup + cooldown)

Usage:
    python tests/test_octave_filter_comparison.py

    # Avec paramètres personnalisés
    python tests/test_octave_filter_comparison.py --n-samples 20000 --fees 0.1
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import argparse
import logging
from typing import Tuple, Dict
from enum import Enum
from dataclasses import dataclass
import scipy.signal as signal

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTES
# =============================================================================

BTC_CSV_PATH = 'data_trad/BTCUSD_all_5m.csv'
TRIM_EDGES = 200

# Périodes des indicateurs
RSI_PERIOD = 14
CCI_PERIOD = 20
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


# =============================================================================
# CHARGEMENT DONNÉES
# =============================================================================

def load_btc_data(n_samples: int = 10000) -> pd.DataFrame:
    """
    Charge les données BTC et applique le trim.

    Args:
        n_samples: Nombre de bougies à charger (avant trim)

    Returns:
        DataFrame avec colonnes: open, high, low, close
    """
    logger.info(f"📁 Chargement BTC ({n_samples} bougies)...")

    df = pd.read_csv(BTC_CSV_PATH, nrows=n_samples)

    # Normaliser les noms de colonnes
    df.columns = df.columns.str.lower()

    # Vérifier colonnes requises
    required = ['open', 'high', 'low', 'close']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    logger.info(f"   Chargé: {len(df)} lignes")

    # Trim edges
    df = df.iloc[TRIM_EDGES:-TRIM_EDGES].copy()
    logger.info(f"   Après trim ±{TRIM_EDGES}: {len(df)} lignes")

    return df


# =============================================================================
# CALCUL INDICATEURS
# =============================================================================

def calculate_rsi(close: pd.Series, period: int = RSI_PERIOD) -> np.ndarray:
    """Calcule le RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.values


def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = CCI_PERIOD) -> np.ndarray:
    """Calcule le CCI."""
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())

    cci = (tp - sma_tp) / (0.015 * mad)

    return cci.values


def calculate_macd(close: pd.Series,
                   fast: int = MACD_FAST,
                   slow: int = MACD_SLOW,
                   signal_period: int = MACD_SIGNAL) -> np.ndarray:
    """Calcule le MACD histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    histogram = macd_line - signal_line

    return histogram.values


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute RSI, CCI, MACD au DataFrame.

    Args:
        df: DataFrame avec OHLC

    Returns:
        DataFrame avec colonnes: rsi, cci, macd
    """
    logger.info("📊 Calcul des indicateurs...")

    df = df.copy()

    df['rsi'] = calculate_rsi(df['close'])
    df['cci'] = calculate_cci(df['high'], df['low'], df['close'])
    df['macd'] = calculate_macd(df['close'])

    logger.info("   ✅ RSI, CCI, MACD calculés")

    return df


# =============================================================================
# FILTRE OCTAVE
# =============================================================================

def octave_filter(data: np.ndarray, step: float = 0.20, order: int = 3) -> np.ndarray:
    """
    Applique le filtre Octave (Butterworth + filtfilt).

    Args:
        data: Signal à filtrer
        step: Paramètre du filtre (défaut: 0.20)
        order: Ordre du filtre (défaut: 3)

    Returns:
        Signal filtré
    """
    valid_mask = ~np.isnan(data)
    if valid_mask.sum() < 10:
        return np.full_like(data, np.nan)

    # Créer le filtre Butterworth
    B, A = signal.butter(order, step, output='ba')

    # Appliquer filtfilt sur les données valides
    valid_data = data[valid_mask]
    filtered_valid = signal.filtfilt(B, A, valid_data)

    # Reconstruire le tableau complet
    filtered = np.full_like(data, np.nan, dtype=float)
    filtered[valid_mask] = filtered_valid

    return filtered


def apply_dual_octave_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique les 2 filtres Octave (0.2 et 0.25) sur chaque indicateur.

    Ajoute les colonnes:
        - {indicator}_filt_02 : Octave step=0.2
        - {indicator}_filt_025 : Octave step=0.25
        - {indicator}_diff : filt_02 - filt_025

    Args:
        df: DataFrame avec indicateurs (rsi, cci, macd)

    Returns:
        DataFrame avec filtres et différences
    """
    logger.info("🔧 Application des filtres Octave...")

    df = df.copy()

    indicators = ['rsi', 'cci', 'macd']

    for ind in indicators:
        # Octave 0.2
        df[f'{ind}_filt_02'] = octave_filter(df[ind].values, step=0.20)

        # Octave 0.25
        df[f'{ind}_filt_025'] = octave_filter(df[ind].values, step=0.25)

        # Différence
        df[f'{ind}_diff'] = df[f'{ind}_filt_02'] - df[f'{ind}_filt_025']

        logger.info(f"   ✅ {ind.upper()}: filt_02, filt_025, diff")

    return df


# =============================================================================
# CALCUL DES SIGNAUX
# =============================================================================

def calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les signaux de trading pour les 2 stratégies.

    Stratégie 1 (Classique): filt_02[t-2] > filt_02[t-3]
    Stratégie 2 (Différence): diff > 0 (filt_02 > filt_025)

    Ajoute les colonnes:
        - {indicator}_signal_classic : Stratégie 1 (1=BUY, 0=SELL)
        - {indicator}_signal_diff : Stratégie 2 (1=BUY si diff>0, 0=SELL si diff<0)

    Args:
        df: DataFrame avec filtres

    Returns:
        DataFrame avec signaux
    """
    logger.info("📈 Calcul des signaux de trading...")

    df = df.copy()

    indicators = ['rsi', 'cci', 'macd']

    for ind in indicators:
        # Stratégie 1 : Filtre classique (t-2 > t-3)
        filt_02_t2 = df[f'{ind}_filt_02'].shift(2)
        filt_02_t3 = df[f'{ind}_filt_02'].shift(3)
        df[f'{ind}_signal_classic'] = (filt_02_t2 > filt_02_t3).astype(int)

        # Stratégie 2 : Signe de la différence (avec shift t-2 pour alignement)
        # BUY si filt_02 > filt_025 (différence positive)
        # SELL si filt_02 < filt_025 (différence négative)
        diff_t2 = df[f'{ind}_diff'].shift(2)
        df[f'{ind}_signal_diff'] = (diff_t2 > 0).astype(int)

    logger.info("   ✅ Signaux calculés (Classique + Différence)")

    return df


# =============================================================================
# BACKTEST
# =============================================================================

def backtest_strategy(
    signals: pd.Series,
    returns: pd.Series,
    fees: float = 0.0
) -> Dict:
    """
    Backtest simple : LONG si signal=1, SHORT si signal=0.

    Args:
        signals: Signaux de trading (1=BUY, 0=SELL)
        returns: Rendements close-to-close
        fees: Frais par trade (défaut: 0.0)

    Returns:
        Dict avec métriques : trades, win_rate, pnl_gross, pnl_net, profit_factor
    """
    # Nettoyer les NaN
    valid = ~(signals.isna() | returns.isna())
    signals = signals[valid]
    returns = returns[valid]

    if len(signals) == 0:
        return {
            'trades': 0,
            'win_rate': 0.0,
            'pnl_gross': 0.0,
            'pnl_net': 0.0,
            'profit_factor': 0.0
        }

    # Calculer les rendements de stratégie
    # LONG (signal=1): PnL = +returns
    # SHORT (signal=0): PnL = -returns
    strategy_returns = np.where(signals == 1, returns, -returns)

    # Détecter les trades (changements de signal)
    signal_changes = signals.diff().fillna(0) != 0
    n_trades = signal_changes.sum()

    # Frais par trade
    total_fees = n_trades * fees / 100.0  # fees en %

    # PnL
    pnl_gross = strategy_returns.sum() * 100  # en %
    pnl_net = pnl_gross - (total_fees * 100)

    # Win Rate
    wins = (strategy_returns > 0).sum()
    win_rate = wins / len(strategy_returns) * 100 if len(strategy_returns) > 0 else 0

    # Profit Factor
    total_wins = strategy_returns[strategy_returns > 0].sum()
    total_losses = abs(strategy_returns[strategy_returns < 0].sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

    return {
        'trades': n_trades,
        'win_rate': win_rate,
        'pnl_gross': pnl_gross,
        'pnl_net': pnl_net,
        'profit_factor': profit_factor
    }


def run_all_backtests(df: pd.DataFrame, fees: float = 0.0) -> pd.DataFrame:
    """
    Exécute les backtests pour toutes les combinaisons.

    Args:
        df: DataFrame avec signaux et returns
        fees: Frais par trade (défaut: 0.0)

    Returns:
        DataFrame avec résultats
    """
    logger.info("\n" + "="*80)
    logger.info("🎯 BACKTESTS - Comparaison Stratégies")
    logger.info("="*80)

    # Calculer les returns
    df['c_ret'] = df['close'].pct_change()

    results = []

    indicators = ['rsi', 'cci', 'macd']
    strategies = ['classic', 'diff']

    for ind in indicators:
        for strat in strategies:
            signal_col = f'{ind}_signal_{strat}'

            if signal_col not in df.columns:
                continue

            metrics = backtest_strategy(
                signals=df[signal_col],
                returns=df['c_ret'],
                fees=fees
            )

            results.append({
                'Indicateur': ind.upper(),
                'Stratégie': 'Classique (filt_02)' if strat == 'classic' else 'Différence (filt_02 - filt_025)',
                'Trades': metrics['trades'],
                'Win Rate (%)': f"{metrics['win_rate']:.2f}",
                'PnL Brut (%)': f"{metrics['pnl_gross']:.2f}",
                'PnL Net (%)': f"{metrics['pnl_net']:.2f}",
                'Profit Factor': f"{metrics['profit_factor']:.2f}"
            })

    results_df = pd.DataFrame(results)

    return results_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Test de comparaison filtres Octave 0.2 vs 0.25",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--n-samples', type=int, default=10000,
                        help='Nombre de bougies à charger (défaut: 10000)')
    parser.add_argument('--fees', type=float, default=0.15,
                        help='Frais par trade en %% (défaut: 0.15)')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("TEST - COMPARAISON FILTRES OCTAVE 0.2 vs 0.25")
    logger.info("="*80)
    logger.info(f"Samples: {args.n_samples}")
    logger.info(f"Frais: {args.fees}%")
    logger.info("")

    # 1. Charger données
    df = load_btc_data(n_samples=args.n_samples)

    # 2. Calculer indicateurs
    df = add_indicators(df)

    # 3. Appliquer filtres Octave
    df = apply_dual_octave_filters(df)

    # 4. Calculer signaux
    df = calculate_signals(df)

    # 5. Backtests
    results_df = run_all_backtests(df, fees=args.fees)

    # 6. Afficher résultats
    logger.info("\n" + "="*80)
    logger.info("📊 RÉSULTATS")
    logger.info("="*80)
    logger.info("\n" + results_df.to_string(index=False))
    logger.info("\n" + "="*80)

    # 7. Analyse comparative
    logger.info("\n🔍 ANALYSE COMPARATIVE:")
    logger.info("-"*80)

    for ind in ['RSI', 'CCI', 'MACD']:
        ind_results = results_df[results_df['Indicateur'] == ind]
        if len(ind_results) == 2:
            classic = ind_results[ind_results['Stratégie'].str.contains('Classique')].iloc[0]
            diff = ind_results[ind_results['Stratégie'].str.contains('Différence')].iloc[0]

            logger.info(f"\n{ind}:")
            logger.info(f"  Classique (filt_02)   : Win Rate={classic['Win Rate (%)']:>6}%, PnL Net={classic['PnL Net (%)']:>8}%, PF={classic['Profit Factor']:>5}")
            logger.info(f"  Différence (02 - 025) : Win Rate={diff['Win Rate (%)']:>6}%, PnL Net={diff['PnL Net (%)']:>8}%, PF={diff['Profit Factor']:>5}")

            # Delta
            wr_classic = float(classic['Win Rate (%)'])
            wr_diff = float(diff['Win Rate (%)'])
            delta_wr = wr_diff - wr_classic

            pnl_classic = float(classic['PnL Net (%)'])
            pnl_diff = float(diff['PnL Net (%)'])
            delta_pnl = pnl_diff - pnl_classic

            emoji_wr = "✅" if delta_wr > 0 else "❌" if delta_wr < 0 else "⚪"
            emoji_pnl = "✅" if delta_pnl > 0 else "❌" if delta_pnl < 0 else "⚪"

            logger.info(f"  → Delta Win Rate : {delta_wr:+.2f}% {emoji_wr}")
            logger.info(f"  → Delta PnL Net  : {delta_pnl:+.2f}% {emoji_pnl}")

    logger.info("\n" + "="*80)
    logger.info("✅ Test terminé!")


if __name__ == '__main__':
    main()
