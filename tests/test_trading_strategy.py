#!/usr/bin/env python3
"""
Backtesting de la stratégie de trading basée sur filtres adaptatifs.

STRATÉGIE DÉTAILLÉE:
1. Charger 10000 bougies 5min (34.7 jours de données BTC)
2. Appliquer filtre causal sur prix de fermeture
3. Trim des bords: enlever 100 premières et 100 dernières (warm-up + artifacts)
4. Calculer signaux de trading:
   - À l'instant t, on compare filtre[t-1] vs filtre[t-2]
   - Si filtre[t-1] > filtre[t-2] → Pente positive → ACHAT (position LONG = +1)
   - Si filtre[t-1] < filtre[t-2] → Pente négative → VENTE (position SHORT = -1)
5. Exécution:
   - Signal calculé à t
   - Trade exécuté à Open[t+1] (bougie suivante)
   - Rendement = (Open[t+1] - Open[t]) / Open[t] × position

CALCUL DU RENDEMENT (LOGIQUE CORRECTE DE TRADING):
On entre en position au changement de signal.
On sort (et calcule le rendement) au prochain changement de signal.

Exemple LONG:
  t=10: Signal BUY → Entre LONG à entry_price = open[11] = $95,000
  t=11-13: Garde position LONG (pas de calcul)
  t=14: Signal SELL → Sort LONG à exit_price = open[15] = $96,000
        Rendement LONG = (exit_price - entry_price) / entry_price
                       = (96,000 - 95,000) / 95,000 = +1.05%

Exemple SHORT:
  t=14: Signal SELL → Entre SHORT à entry_price = open[15] = $96,000
  t=15-19: Garde position SHORT (pas de calcul)
  t=20: Signal BUY → Sort SHORT à exit_price = open[21] = $95,000
        Rendement SHORT = (entry_price - exit_price) / entry_price
                        = (96,000 - 95,000) / 96,000 = +1.04%

Rendement total = Produit composé de tous les trades fermés

FILTRES TESTÉS:
- KAMA (Kaufman Adaptive MA)
- HMA (Hull MA)
- SuperSmoother (Ehlers)
- Decycler (Ehlers)
- Kalman (filtre causal)
- Butterworth (filtre causal)
- Ensemble (moyenne des 4 adaptatifs)

GPU-FRIENDLY:
- Utilise numpy vectorisé (compatible CUDA via CuPy si besoin)
- Pas de boucles Python inutiles
- Calculs optimisés pour parallélisation
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


def backtest_filter_strategy(df, filter_func, filter_name, trim_edges=100):
    """
    Backtest de la stratégie sur un filtre donné.

    Args:
        df: DataFrame avec OHLC
        filter_func: Fonction de filtre à appliquer
        filter_name: Nom du filtre pour affichage
        trim_edges: Nombre de valeurs à enlever au début et fin (warm-up + artifacts)

    Returns:
        dict avec résultats du backtest
    """
    print(f"\n{'='*80}")
    print(f"BACKTEST: {filter_name}")
    print(f"{'='*80}")

    # Appliquer le filtre sur toutes les données
    close = df['close'].values
    filtered = filter_func(close)

    # Enlever les bords (warm-up + artifacts)
    # Règle critique du projet: trim avant de trader!
    df_trade = df.iloc[trim_edges:-trim_edges].copy()
    filtered_trade = filtered[trim_edges:-trim_edges]

    print(f"Dataset total: {len(df)} valeurs ({len(df) * 5 / 60 / 24:.1f} jours)")
    print(f"Zone de trading: {len(df_trade)} valeurs (après trim de {trim_edges} début+fin)")
    print(f"Durée trading: {len(df_trade) * 5 / 60 / 24:.1f} jours")

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

    # =====================================================================
    # CALCUL DES RENDEMENTS - LOGIQUE CORRECTE DE TRADING
    # =====================================================================
    # On entre en position au changement de signal
    # On sort (et calcule le rendement) au prochain changement de signal
    #
    # Exemple LONG:
    #   t=10: BUY → Entre LONG à open[11] = $95k
    #   t=11-13: Garde position LONG
    #   t=14: SELL → Sort LONG à open[15] = $96k
    #         Rendement = (96k - 95k) / 95k = +1.05%
    #
    # Exemple SHORT:
    #   t=14: SELL → Entre SHORT à open[15] = $96k
    #   t=15-19: Garde position SHORT
    #   t=20: BUY → Sort SHORT à open[21] = $95k
    #         Rendement = (96k - 95k) / 96k = +1.04%
    # =====================================================================

    trades_list = []
    entry_price = None
    entry_position = None
    entry_idx = None

    for idx in range(len(df_trade)):
        current_pos = df_trade.iloc[idx]['position']
        current_open = df_trade.iloc[idx]['open']

        # Détecter changement de position
        if idx == 0:
            # Premier signal
            if current_pos != 0:
                entry_price = current_open
                entry_position = current_pos
                entry_idx = idx
        else:
            prev_pos = df_trade.iloc[idx-1]['position']

            # Changement de position détecté
            if current_pos != prev_pos:
                # Fermer la position précédente si elle existait
                if entry_price is not None:
                    exit_price = current_open

                    # Calculer le rendement selon le type de position
                    if entry_position == 1:  # LONG
                        trade_return = (exit_price - entry_price) / entry_price
                    elif entry_position == -1:  # SHORT
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

                # Ouvrir nouvelle position
                if current_pos != 0:
                    entry_price = current_open
                    entry_position = current_pos
                    entry_idx = idx
                else:
                    entry_price = None
                    entry_position = None
                    entry_idx = None

    # Fermer la dernière position si elle est ouverte
    if entry_price is not None:
        exit_price = df_trade.iloc[-1]['open']

        if entry_position == 1:  # LONG
            trade_return = (exit_price - entry_price) / entry_price
        elif entry_position == -1:  # SHORT
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

    # Créer DataFrame des trades
    df_trades = pd.DataFrame(trades_list)

    # Calculer le rendement total (produit composé)
    if len(df_trades) > 0:
        total_return = (1 + df_trades['return']).prod() - 1
        total_return_pct = total_return * 100
    else:
        total_return = 0
        total_return_pct = 0

    buy_signals = (df_trade['signal'] == 'BUY').sum()
    sell_signals = (df_trade['signal'] == 'SELL').sum()
    hold_signals = (df_trade['signal'] == 'HOLD').sum()

    # Buy & Hold pour comparaison
    buy_hold_return = (df_trade['close'].iloc[-1] / df_trade['close'].iloc[0] - 1) * 100

    # Sharpe ratio (annualisé)
    # Basé sur les rendements des trades individuels
    if len(df_trades) > 0:
        returns_mean = df_trades['return'].mean()
        returns_std = df_trades['return'].std()

        # Nombre de trades par an (approximatif)
        n_days = len(df_trade) * 5 / 60 / 24  # Durée en jours
        trades_per_year = (len(df_trades) / n_days) * 365 if n_days > 0 else 0

        sharpe = (returns_mean / returns_std) * np.sqrt(trades_per_year) if returns_std > 0 else 0
    else:
        sharpe = 0

    # Drawdown maximum (basé sur le capital cumulatif)
    if len(df_trades) > 0:
        df_trades['cumulative'] = (1 + df_trades['return']).cumprod()
        running_max = df_trades['cumulative'].expanding().max()
        df_trades['drawdown'] = (df_trades['cumulative'] - running_max) / running_max
        max_drawdown = df_trades['drawdown'].min() * 100
    else:
        max_drawdown = 0

    # Win rate (sur les trades fermés)
    if len(df_trades) > 0:
        wins = (df_trades['return'] > 0).sum()
        losses = (df_trades['return'] < 0).sum()
        total_trades = len(df_trades)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    else:
        total_trades = 0
        win_rate = 0

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
        'df_trade': df_trade,
        'df_trades': df_trades  # DataFrame des trades individuels
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
    # 10000 bougies × 5min = 50000 min = 34.7 jours
    df = load_btc_data_or_simulate(n=10000)

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
        df_trades = result['df_trades']
        if len(df_trades) > 0:
            cumulative = (1 + df_trades['return']).cumprod()
            axes[1, 1].plot(cumulative.values, label=result['filter_name'], linewidth=2)
        else:
            axes[1, 1].plot([1.0], label=result['filter_name'], linewidth=2)

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
    print("\n📊 STRATÉGIE:")
    print("  1. Charger 10000 bougies 5min (34.7 jours)")
    print("  2. Appliquer filtre CAUSAL sur fermeture")
    print("  3. Trim: enlever 100 début + 100 fin (warm-up + artifacts)")
    print("  4. Signal ACHAT: filtre[t-1] > filtre[t-2] → LONG à Open[t+1]")
    print("  5. Signal VENTE: filtre[t-1] < filtre[t-2] → SHORT à Open[t+1]")
    print("  6. Rendement: (Open[t+1] - Open[t]) / Open[t] × position")

    # Comparer tous les filtres
    all_results, df_comparison = compare_all_filters()

    print("\n" + "="*80)
    print("✅ BACKTESTING TERMINÉ")
    print("="*80)

    # Résumé final
    print("\n📈 RÉSUMÉ FINAL:")
    best = max(all_results, key=lambda x: x['total_return_pct'])
    best_sharpe = max(all_results, key=lambda x: x['sharpe_ratio'])
    best_dd = max(all_results, key=lambda x: -x['max_drawdown_pct'])  # Max = min drawdown

    print(f"  🏆 Meilleur rendement: {best['filter_name']} ({best['total_return_pct']:+.2f}%)")
    print(f"  📊 Meilleur Sharpe: {best_sharpe['filter_name']} ({best_sharpe['sharpe_ratio']:.2f})")
    print(f"  🛡️ Drawdown minimal: {best_dd['filter_name']} ({best_dd['max_drawdown_pct']:.2f}%)")

    n_days = len(all_results[0]['df_trade']) * 5 / 60 / 24
    print(f"\n  Durée backtest: {n_days:.1f} jours")
    print(f"  Total trades fermés: {all_results[0]['total_trades']}")
    if n_days > 0 and all_results[0]['total_trades'] > 0:
        print(f"  Fréquence: {all_results[0]['total_trades'] / n_days:.1f} trades/jour")
        print(f"  Durée moyenne par trade: {n_days / all_results[0]['total_trades']:.2f} jours")
