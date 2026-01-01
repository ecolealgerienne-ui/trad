#!/usr/bin/env python3
"""
Test : Filtre Decycler appliqué sur Indicateurs Techniques

Méthode:
1. Calculer indicateurs techniques (RSI, CCI, Bollinger, MACD)
2. Appliquer Decycler PARFAIT (forward-backward) sur chaque indicateur
3. Générer signal : filtered_indicator[t-1] > filtered_indicator[t-2] → BUY, sinon SELL
4. Trader à open[t+1]
5. Comparer les résultats

Objectif: Trouver quel indicateur + Decycler donne le meilleur Profit Factor
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Imports
from adaptive_filters import ehlers_decycler

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


def calculate_rsi(close, period=14):
    """Calcule le RSI."""
    delta = np.diff(close, prepend=close[0])
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)

    avg_gain = np.zeros(len(close))
    avg_loss = np.zeros(len(close))

    # Premier calcul
    avg_gain[period] = np.mean(gains[1:period+1])
    avg_loss[period] = np.mean(losses[1:period+1])

    # Lissage exponentiel
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period

    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))

    # Remplir les NaN initiaux
    rsi[:period] = 50  # Neutre

    return rsi


def calculate_cci(high, low, close, period=20):
    """Calcule le CCI (Commodity Channel Index)."""
    typical_price = (high + low + close) / 3

    cci = np.zeros(len(close))

    for i in range(period - 1, len(close)):
        window = typical_price[i - period + 1:i + 1]
        sma = np.mean(window)
        mad = np.mean(np.abs(window - sma))

        if mad > 0:
            cci[i] = (typical_price[i] - sma) / (0.015 * mad)
        else:
            cci[i] = 0

    # Remplir les NaN initiaux
    cci[:period-1] = 0  # Neutre

    return cci


def calculate_bollinger_position(close, period=20, num_std=2):
    """
    Calcule la position du prix dans les Bollinger Bands.

    Returns:
        %B : (close - lower_band) / (upper_band - lower_band)
        Valeurs entre 0 (sur bande basse) et 1 (sur bande haute)
    """
    bb_position = np.zeros(len(close))

    for i in range(period - 1, len(close)):
        window = close[i - period + 1:i + 1]
        sma = np.mean(window)
        std = np.std(window, ddof=1)

        upper_band = sma + num_std * std
        lower_band = sma - num_std * std

        band_width = upper_band - lower_band

        if band_width > 0:
            bb_position[i] = (close[i] - lower_band) / band_width
        else:
            bb_position[i] = 0.5  # Neutre

    # Remplir les NaN initiaux
    bb_position[:period-1] = 0.5  # Neutre

    # Clamp entre 0 et 1
    bb_position = np.clip(bb_position, 0, 1)

    return bb_position * 100  # Convertir en 0-100 pour cohérence avec RSI


def calculate_macd(close, fast=12, slow=26, signal=9):
    """
    Calcule MACD histogram.

    Returns:
        MACD histogram (MACD line - Signal line)
    """
    # EMA rapide et lente
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values

    # MACD line
    macd_line = ema_fast - ema_slow

    # Signal line
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values

    # Histogram
    histogram = macd_line - signal_line

    return histogram


def apply_decycler_perfect(signal):
    """Applique Decycler en mode PARFAIT (forward-backward)."""
    forward = ehlers_decycler(signal)
    backward = ehlers_decycler(forward[::-1])
    return backward[::-1]


def load_btc_data_or_simulate(n=10000):
    """Charger données BTC ou simuler."""
    btc_path = Path('data/raw/BTCUSD_all_5m.csv')

    if btc_path.exists():
        print(f"📂 Chargement données BTC réelles: {btc_path}")
        df = pd.read_csv(btc_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        if len(df) > n:
            df = df.tail(n).reset_index(drop=True)

        print(f"✅ {len(df)} bougies BTC chargées")
        return df

    # Simuler GBM
    print(f"⚠️ Simulation de données (GBM)")
    np.random.seed(42)

    timestamps = pd.date_range('2024-12-01', periods=n, freq='5min')
    base_price = 95000
    dt = 5 / (60 * 24)
    drift = 0.0001
    volatility = 0.02

    log_returns = np.random.normal(drift * dt, volatility * np.sqrt(dt), n)
    prices = base_price * np.exp(np.cumsum(log_returns))

    # Ajouter cycles
    t = np.linspace(0, 4*np.pi, n)
    cycles = prices * 0.005 * np.sin(t)
    prices = prices + cycles

    # Créer OHLC
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.0005)
            open_price = prices[i-1] + gap

        intrabar_vol = np.random.uniform(0.001, 0.003)
        high = max(open_price, close) * (1 + intrabar_vol)
        low = min(open_price, close) * (1 - intrabar_vol)
        volume = np.random.lognormal(2, 1)

        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    return pd.DataFrame(data)


def backtest_indicator_filtered(df, indicator_name, indicator_values, trim_edges=100):
    """
    Backtest d'un indicateur filtré avec Decycler parfait.

    Args:
        df: DataFrame OHLCV
        indicator_name: Nom de l'indicateur
        indicator_values: Valeurs de l'indicateur (array)
        trim_edges: Edges à trim

    Returns:
        dict résultats
    """
    print(f"\n{'='*80}")
    print(f"BACKTEST: Decycler({indicator_name})")
    print(f"{'='*80}")

    # Appliquer Decycler parfait sur l'indicateur
    filtered_indicator = apply_decycler_perfect(indicator_values)

    # Trim
    df_trade = df.iloc[trim_edges:-trim_edges].copy()
    filtered_trade = filtered_indicator[trim_edges:-trim_edges]

    print(f"Dataset total: {len(df)} valeurs ({len(df) * 5 / 60 / 24:.1f} jours)")
    print(f"Zone de trading: {len(df_trade)} valeurs")

    # Calculer signaux
    signals = []
    positions = []
    position = 0

    for i in range(2, len(filtered_trade)):
        t_minus_1 = filtered_trade[i-1]
        t_minus_2 = filtered_trade[i-2]

        if t_minus_1 > t_minus_2:
            signal = 'BUY'
            new_position = 1
        elif t_minus_1 < t_minus_2:
            signal = 'SELL'
            new_position = -1
        else:
            signal = 'HOLD'
            new_position = position

        signals.append(signal)
        positions.append(new_position)
        position = new_position

    df_trade = df_trade.iloc[2:].copy()
    df_trade['signal'] = signals
    df_trade['position'] = positions
    df_trade['filtered'] = filtered_trade[2:]

    # Calculer rendements
    trades_list = []
    entry_price = None
    entry_position = None
    entry_idx = None

    for idx in range(len(df_trade) - 1):
        current_pos = df_trade.iloc[idx]['position']
        next_open = df_trade.iloc[idx + 1]['open']

        if idx == 0:
            if current_pos != 0:
                entry_price = next_open
                entry_position = current_pos
                entry_idx = idx
        else:
            prev_pos = df_trade.iloc[idx-1]['position']

            if current_pos != prev_pos:
                if entry_price is not None:
                    exit_price = next_open

                    if entry_position == 1:
                        trade_return = (exit_price - entry_price) / entry_price
                    elif entry_position == -1:
                        trade_return = (entry_price - exit_price) / entry_price
                    else:
                        trade_return = 0.0

                    trades_list.append({
                        'entry_idx': entry_idx,
                        'exit_idx': idx,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_type': 'LONG' if entry_position == 1 else 'SHORT',
                        'return': trade_return,
                        'return_pct': trade_return * 100
                    })

                if current_pos != 0:
                    entry_price = next_open
                    entry_position = current_pos
                    entry_idx = idx
                else:
                    entry_price = None
                    entry_position = None
                    entry_idx = None

    # Fermer dernière position
    if entry_price is not None:
        exit_price = df_trade.iloc[-1]['open']

        if entry_position == 1:
            trade_return = (exit_price - entry_price) / entry_price
        elif entry_position == -1:
            trade_return = (entry_price - exit_price) / entry_price
        else:
            trade_return = 0.0

        trades_list.append({
            'entry_idx': entry_idx,
            'exit_idx': len(df_trade) - 1,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_type': 'LONG' if entry_position == 1 else 'SHORT',
            'return': trade_return,
            'return_pct': trade_return * 100
        })

    df_trades = pd.DataFrame(trades_list)

    # Métriques
    if len(df_trades) > 0:
        total_return = (1 + df_trades['return']).prod() - 1
        total_return_pct = total_return * 100

        wins = (df_trades['return'] > 0).sum()
        losses = (df_trades['return'] < 0).sum()
        win_rate = (wins / len(df_trades) * 100) if len(df_trades) > 0 else 0

        gross_profit = df_trades[df_trades['return'] > 0]['return'].sum()
        gross_loss = abs(df_trades[df_trades['return'] < 0]['return'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        returns_mean = df_trades['return'].mean()
        returns_std = df_trades['return'].std()
        n_days = len(df_trade) * 5 / 60 / 24
        trades_per_year = (len(df_trades) / n_days) * 365 if n_days > 0 else 0
        sharpe = (returns_mean / returns_std) * np.sqrt(trades_per_year) if returns_std > 0 else 0

        df_trades['cumulative'] = (1 + df_trades['return']).cumprod()
        running_max = df_trades['cumulative'].expanding().max()
        df_trades['drawdown'] = (df_trades['cumulative'] - running_max) / running_max
        max_drawdown = df_trades['drawdown'].min() * 100
    else:
        total_return_pct = 0
        win_rate = 0
        profit_factor = 0
        sharpe = 0
        max_drawdown = 0

    buy_hold_return = (df_trade['close'].iloc[-1] / df_trade['close'].iloc[0] - 1) * 100

    results = {
        'indicator_name': indicator_name,
        'total_return_pct': total_return_pct,
        'buy_hold_return_pct': buy_hold_return,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_drawdown,
        'win_rate_pct': win_rate,
        'total_trades': len(df_trades),
        'df_trade': df_trade,
        'df_trades': df_trades
    }

    print(f"\n📊 RÉSULTATS:")
    print(f"  Rendement stratégie: {total_return_pct:+.2f}%")
    print(f"  Rendement Buy & Hold: {buy_hold_return:+.2f}%")
    print(f"  💰 Profit Factor: {profit_factor:.2f}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_drawdown:.2f}%")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Total trades: {len(df_trades)}")

    return results


def compare_all_indicators_with_decycler():
    """Compare tous les indicateurs avec Decycler parfait."""
    print("\n" + "="*80)
    print("COMPARAISON: Decycler sur Différents Indicateurs")
    print("="*80)

    # Charger données
    df = load_btc_data_or_simulate(n=10000)

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    # Calculer tous les indicateurs
    print("\n📈 Calcul des indicateurs techniques...")
    rsi = calculate_rsi(close, period=14)
    cci = calculate_cci(high, low, close, period=20)
    bol = calculate_bollinger_position(close, period=20)
    macd = calculate_macd(close)

    # Normaliser MACD pour être comparable (0-100 comme RSI)
    macd_normalized = ((macd - macd.min()) / (macd.max() - macd.min())) * 100

    print("✅ Indicateurs calculés: RSI, CCI, Bollinger %B, MACD")

    # Tester chaque indicateur
    indicators = [
        ('Close (prix brut)', close),
        ('RSI(14)', rsi),
        ('CCI(20)', cci),
        ('Bollinger %B(20)', bol),
        ('MACD Histogram', macd_normalized)
    ]

    all_results = []

    for indicator_name, indicator_values in indicators:
        try:
            results = backtest_indicator_filtered(df, indicator_name, indicator_values)
            all_results.append(results)
        except Exception as e:
            print(f"\n❌ Erreur avec {indicator_name}: {e}")
            import traceback
            traceback.print_exc()

    # Tableau comparatif
    print("\n" + "="*80)
    print("TABLEAU COMPARATIF - Decycler sur Indicateurs")
    print("="*80)

    comparison = pd.DataFrame([
        {
            'Indicateur': r['indicator_name'],
            'Rendement (%)': f"{r['total_return_pct']:+.2f}",
            'Profit Factor': f"{r['profit_factor']:.2f}",
            'Sharpe': f"{r['sharpe_ratio']:.2f}",
            'Max DD (%)': f"{r['max_drawdown_pct']:.2f}",
            'Win Rate (%)': f"{r['win_rate_pct']:.1f}",
            'Trades': r['total_trades']
        }
        for r in all_results
    ])

    print("\n" + comparison.to_string(index=False))

    # Meilleur indicateur
    best = max(all_results, key=lambda x: x['profit_factor'])
    print(f"\n🏆 MEILLEUR INDICATEUR: {best['indicator_name']}")
    print(f"   Profit Factor: {best['profit_factor']:.2f}")
    print(f"   Rendement: {best['total_return_pct']:+.2f}%")
    print(f"   Win Rate: {best['win_rate_pct']:.1f}%")

    print("\n" + "="*80)
    print("INTERPRÉTATION:")
    print("="*80)
    print("\n✅ Ce test compare Decycler appliqué sur différentes sources:")
    print("   → Prix brut (close)")
    print("   → Indicateurs techniques (RSI, CCI, Bollinger, MACD)")
    print("\n💡 Le meilleur indicateur montre quel signal est le plus adapté")
    print("   pour la stratégie de pente filtered[t-1] > filtered[t-2]")

    return all_results, comparison


if __name__ == '__main__':
    print("\n" + "="*80)
    print("TEST: Decycler sur Indicateurs Techniques")
    print("="*80)
    print("\n📌 OBJECTIF:")
    print("  Comparer Decycler appliqué sur:")
    print("  - Prix brut (close)")
    print("  - RSI(14)")
    print("  - CCI(20)")
    print("  - Bollinger %B(20)")
    print("  - MACD Histogram")
    print("\n📊 MÉTHODE:")
    print("  1. Calculer indicateur")
    print("  2. Appliquer Decycler PARFAIT (forward-backward)")
    print("  3. Signal: filtered[t-1] > filtered[t-2] → BUY, sinon SELL")
    print("  4. Trade à open[t+1]")

    # Comparer tous les indicateurs
    all_results, comparison = compare_all_indicators_with_decycler()

    print("\n" + "="*80)
    print("✅ TEST TERMINÉ")
    print("="*80)
