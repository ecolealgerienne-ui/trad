#!/usr/bin/env python3
"""
Backtesting de la stratégie de trading basée sur filtres adaptatifs.

STRATÉGIE:
1. Charger 1000 valeurs
2. Appliquer filtre sur fermeture
3. Prendre 500 valeurs au milieu (250-750)
4. Signal ACHAT: filtre[t-1] > filtre[t-2] → Acheter à Open[t+1]
5. Signal VENTE: filtre[t-1] < filtre[t-2] → Vendre à Open[t+1]

Teste tous les filtres: KAMA, HMA, SuperSmoother, Decycler, Ensemble
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Imports des filtres
from adaptive_filters import (
    kama_filter,
    hma_filter,
    ehlers_supersmoother,
    ehlers_decycler,
    adaptive_filter_ensemble,
    kalman_filter_causal,
    butterworth_causal
)

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


def load_btc_data_or_simulate(n=1000):
    """
    Charge les données BTC réelles si disponibles, sinon simule des données réalistes.

    Returns:
        DataFrame avec OHLCV
    """
    # Essayer de charger les données réelles
    btc_path = Path('data/raw/BTCUSD_all_5m.csv')

    if btc_path.exists():
        logger.info(f"📂 Chargement données BTC réelles: {btc_path}")
        df = pd.read_csv(btc_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Prendre les n dernières valeurs
        if len(df) > n:
            df = df.tail(n).reset_index(drop=True)
            logger.info(f"✅ {len(df)} bougies BTC chargées (dernières {n})")
        else:
            logger.info(f"✅ {len(df)} bougies BTC chargées (toutes)")

        return df

    # Sinon, simuler des données réalistes inspirées du BTC
    logger.warning(f"⚠️ Données BTC non trouvées, simulation de données réalistes")

    np.random.seed(42)

    # Timestamps
    timestamps = pd.date_range('2024-12-01', periods=n, freq='5min')

    # Prix BTC typique: 40000-100000 USD avec tendance et volatilité
    base_price = 95000  # Prix BTC approximatif fin 2024

    # Marche aléatoire géométrique (modèle GBM simplifié)
    # Drift (tendance) et volatilité inspirés du BTC
    dt = 5 / (60 * 24)  # 5 minutes en jours
    drift = 0.0001  # Légère tendance haussière
    volatility = 0.02  # 2% volatilité (typique BTC intraday)

    # Générer les returns log
    log_returns = np.random.normal(drift * dt, volatility * np.sqrt(dt), n)

    # Convertir en prix
    prices = base_price * np.exp(np.cumsum(log_returns))

    # Ajouter cycles et micro-tendances
    t = np.linspace(0, 4*np.pi, n)
    cycles = prices * 0.005 * np.sin(t)  # 0.5% oscillations
    prices = prices + cycles

    # Créer OHLC réaliste
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        if i == 0:
            open_price = close
        else:
            # Gap entre open et close précédent
            gap = np.random.normal(0, close * 0.0005)  # 0.05% gap moyen
            open_price = prices[i-1] + gap

        # Volatilité intra-bougie (high-low range)
        intrabar_vol = np.random.uniform(0.001, 0.003)  # 0.1-0.3%
        high = max(open_price, close) * (1 + intrabar_vol)
        low = min(open_price, close) * (1 - intrabar_vol)

        # Volume typique BTC (en BTC)
        volume = np.random.lognormal(2, 1)  # Distribution log-normale

        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    df = pd.DataFrame(data)

    logger.info(f"✅ {len(df)} bougies simulées (style BTC réaliste)")
    logger.info(f"   Prix min: ${df['close'].min():.2f}")
    logger.info(f"   Prix max: ${df['close'].max():.2f}")
    logger.info(f"   Prix moyen: ${df['close'].mean():.2f}")

    return df


def backtest_filter_strategy(df, filter_func, filter_name, start_idx=250, end_idx=750):
    """
    Backtest de la stratégie sur un filtre donné.

    Args:
        df: DataFrame avec OHLC (minimum 1000 lignes)
        filter_func: Fonction de filtre à appliquer
        filter_name: Nom du filtre pour affichage
        start_idx: Index de début (défaut 250, milieu des 1000)
        end_idx: Index de fin (défaut 750, milieu des 1000)

    Returns:
        dict avec résultats du backtest
    """
    print(f"\n{'='*80}")
    print(f"BACKTEST: {filter_name}")
    print(f"{'='*80}")

    # Appliquer le filtre sur toutes les données
    close = df['close'].values
    filtered = filter_func(close)

    # Extraire la zone du milieu (500 valeurs)
    df_trade = df.iloc[start_idx:end_idx].copy()
    filtered_trade = filtered[start_idx:end_idx]

    print(f"Dataset total: {len(df)} valeurs")
    print(f"Zone de trading: {len(df_trade)} valeurs (index {start_idx}-{end_idx})")

    # Calculer les signaux
    signals = []
    positions = []  # 1 = LONG, 0 = OUT, -1 = SHORT

    position = 0  # Position initiale: OUT

    for i in range(2, len(filtered_trade)):
        # Indice dans la zone de trading
        t_minus_2 = filtered_trade[i-2]
        t_minus_1 = filtered_trade[i-1]

        # Signal
        if t_minus_1 > t_minus_2:
            # Tendance haussière → ACHAT
            signal = 'BUY'
            new_position = 1
        elif t_minus_1 < t_minus_2:
            # Tendance baissière → VENTE
            signal = 'SELL'
            new_position = -1
        else:
            # Pas de changement
            signal = 'HOLD'
            new_position = position

        signals.append(signal)
        positions.append(new_position)
        position = new_position

    # Ajouter les signaux au dataframe
    df_trade = df_trade.iloc[2:].copy()  # Enlever les 2 premières (pas de signal)
    df_trade['signal'] = signals
    df_trade['position'] = positions
    df_trade['filtered'] = filtered_trade[2:]

    # Calculer les rendements
    # À t, on a le signal, on exécute à l'ouverture de t+1
    df_trade['next_open'] = df_trade['open'].shift(-1)
    df_trade['return'] = df_trade['next_open'].pct_change()

    # Rendement de la stratégie = rendement du marché * position
    df_trade['strategy_return'] = df_trade['return'] * df_trade['position'].shift(1)

    # Calculer les métriques
    total_return = (1 + df_trade['strategy_return'].fillna(0)).cumprod().iloc[-1] - 1
    total_return_pct = total_return * 100

    buy_signals = (df_trade['signal'] == 'BUY').sum()
    sell_signals = (df_trade['signal'] == 'SELL').sum()
    hold_signals = (df_trade['signal'] == 'HOLD').sum()

    # Buy & Hold pour comparaison
    buy_hold_return = (df_trade['close'].iloc[-1] / df_trade['close'].iloc[0] - 1) * 100

    # Sharpe ratio simplifié (annualisé)
    # Assuming 5min candles, 12 per hour, 24h trading = 288 candles/day
    returns_mean = df_trade['strategy_return'].mean()
    returns_std = df_trade['strategy_return'].std()
    sharpe = (returns_mean / returns_std) * np.sqrt(288 * 365) if returns_std > 0 else 0

    # Drawdown maximum
    cumulative = (1 + df_trade['strategy_return'].fillna(0)).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    # Win rate (sur les trades fermés)
    wins = (df_trade['strategy_return'] > 0).sum()
    losses = (df_trade['strategy_return'] < 0).sum()
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    results = {
        'filter_name': filter_name,
        'total_return_pct': total_return_pct,
        'buy_hold_return_pct': buy_hold_return,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_drawdown,
        'win_rate_pct': win_rate,
        'total_trades': total_trades,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'hold_signals': hold_signals,
        'df_trade': df_trade
    }

    # Afficher résultats
    print(f"\n📊 RÉSULTATS:")
    print(f"  Rendement stratégie: {total_return_pct:+.2f}%")
    print(f"  Rendement Buy & Hold: {buy_hold_return:+.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_drawdown:.2f}%")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Total trades: {total_trades}")
    print(f"  Signaux: {buy_signals} BUY, {sell_signals} SELL, {hold_signals} HOLD")

    return results


def compare_all_filters():
    """
    Compare tous les filtres sur la même stratégie.
    """
    print("\n" + "="*80)
    print("COMPARAISON DE TOUS LES FILTRES - STRATÉGIE DE TRADING")
    print("="*80)

    # Charger données BTC réelles ou simuler
    df = load_btc_data_or_simulate(n=1000)

    # Définir les filtres à tester
    filters = [
        (kama_filter, 'KAMA'),
        (hma_filter, 'HMA'),
        (ehlers_supersmoother, 'SuperSmoother'),
        (ehlers_decycler, 'Decycler'),
        (kalman_filter_causal, 'Kalman'),
        (butterworth_causal, 'Butterworth'),
        (adaptive_filter_ensemble, 'Ensemble')
    ]

    # Backtester chaque filtre
    all_results = []

    for filter_func, filter_name in filters:
        results = backtest_filter_strategy(df, filter_func, filter_name)
        all_results.append(results)

    # Créer tableau comparatif
    print("\n" + "="*80)
    print("TABLEAU COMPARATIF")
    print("="*80)

    df_comparison = pd.DataFrame([
        {
            'Filtre': r['filter_name'],
            'Rendement (%)': f"{r['total_return_pct']:+.2f}",
            'Buy&Hold (%)': f"{r['buy_hold_return_pct']:+.2f}",
            'Sharpe': f"{r['sharpe_ratio']:.2f}",
            'Max DD (%)': f"{r['max_drawdown_pct']:.2f}",
            'Win Rate (%)': f"{r['win_rate_pct']:.1f}",
            'Trades': r['total_trades']
        }
        for r in all_results
    ])

    print("\n" + df_comparison.to_string(index=False))

    # Visualiser
    visualize_comparison(all_results)

    # Déterminer le meilleur filtre
    best_filter = max(all_results, key=lambda x: x['total_return_pct'])
    print(f"\n🏆 MEILLEUR FILTRE: {best_filter['filter_name']}")
    print(f"   Rendement: {best_filter['total_return_pct']:+.2f}%")

    return all_results, df_comparison


def visualize_comparison(all_results):
    """
    Visualise les résultats de comparaison.
    """
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    # Extraire données
    filter_names = [r['filter_name'] for r in all_results]
    returns = [r['total_return_pct'] for r in all_results]
    sharpes = [r['sharpe_ratio'] for r in all_results]
    drawdowns = [r['max_drawdown_pct'] for r in all_results]
    win_rates = [r['win_rate_pct'] for r in all_results]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    # 1. Rendements
    axes[0, 0].bar(filter_names, returns, color=colors, alpha=0.8)
    axes[0, 0].set_title('Rendement Total (%)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Rendement (%)')
    axes[0, 0].set_xticklabels(filter_names, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    for i, v in enumerate(returns):
        axes[0, 0].text(i, v + 0.5, f'{v:+.1f}%', ha='center', fontweight='bold')

    # 2. Sharpe Ratio
    axes[0, 1].bar(filter_names, sharpes, color=colors, alpha=0.8)
    axes[0, 1].set_title('Sharpe Ratio', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Sharpe')
    axes[0, 1].set_xticklabels(filter_names, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    for i, v in enumerate(sharpes):
        axes[0, 1].text(i, v + 0.05, f'{v:.2f}', ha='center', fontweight='bold')

    # 3. Max Drawdown
    axes[0, 2].bar(filter_names, drawdowns, color=colors, alpha=0.8)
    axes[0, 2].set_title('Maximum Drawdown (%)', fontsize=14, fontweight='bold')
    axes[0, 2].set_ylabel('Drawdown (%)')
    axes[0, 2].set_xticklabels(filter_names, rotation=45, ha='right')
    axes[0, 2].grid(True, alpha=0.3)
    for i, v in enumerate(drawdowns):
        axes[0, 2].text(i, v - 0.5, f'{v:.1f}%', ha='center', fontweight='bold')

    # 4. Win Rate
    axes[1, 0].bar(filter_names, win_rates, color=colors, alpha=0.8)
    axes[1, 0].set_title('Win Rate (%)', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Win Rate (%)')
    axes[1, 0].set_xticklabels(filter_names, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=50, color='red', linestyle='--', linewidth=1, label='50% (hasard)')
    axes[1, 0].legend()
    for i, v in enumerate(win_rates):
        axes[1, 0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

    # 5. Courbe de capital (cumulatif)
    for result in all_results:
        df_trade = result['df_trade']
        cumulative = (1 + df_trade['strategy_return'].fillna(0)).cumprod()
        axes[1, 1].plot(cumulative.values, label=result['filter_name'], linewidth=2)

    axes[1, 1].set_title('Courbe de Capital (cumulatif)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Trade #')
    axes[1, 1].set_ylabel('Capital (base 1.0)')
    axes[1, 1].legend(loc='best')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=1, color='black', linestyle='--', linewidth=1)

    # 6. Comparaison Rendement vs Buy & Hold
    returns_bh = [r['buy_hold_return_pct'] for r in all_results]

    x = np.arange(len(filter_names))
    width = 0.35

    axes[1, 2].bar(x - width/2, returns, width, label='Stratégie', color=colors, alpha=0.8)
    axes[1, 2].bar(x + width/2, returns_bh, width, label='Buy & Hold', color='gray', alpha=0.6)
    axes[1, 2].set_title('Stratégie vs Buy & Hold', fontsize=14, fontweight='bold')
    axes[1, 2].set_ylabel('Rendement (%)')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(filter_names, rotation=45, ha='right')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axhline(y=0, color='black', linestyle='--', linewidth=1)

    plt.tight_layout()
    output_path = Path('tests/validation_output/05_trading_strategy_comparison.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualisation sauvegardée: {output_path}")
    plt.close()


if __name__ == '__main__':
    print("\n" + "="*80)
    print("BACKTESTING STRATÉGIE DE TRADING - FILTRES ADAPTATIFS")
    print("="*80)
    print("\nStratégie:")
    print("  1. Charger 1000 valeurs")
    print("  2. Appliquer filtre sur fermeture")
    print("  3. Prendre 500 valeurs au milieu (250-750)")
    print("  4. Signal ACHAT: filtre[t-1] > filtre[t-2] → Acheter Open[t+1]")
    print("  5. Signal VENTE: filtre[t-1] < filtre[t-2] → Vendre Open[t+1]")

    # Comparer tous les filtres
    all_results, df_comparison = compare_all_filters()

    print("\n" + "="*80)
    print("✅ BACKTESTING TERMINÉ")
    print("="*80)
