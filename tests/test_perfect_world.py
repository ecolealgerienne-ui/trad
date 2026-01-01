#!/usr/bin/env python3
"""
VALIDATION DE LA MÉTHODE EN "MONDE PARFAIT"

Ce test valide la règle de trading dans des conditions idéales:
- Filtre de Kalman EXACT (non-causal, connaissant le futur)
- Signal: filter[t-1] > filter[t-2]
- Exécution: open[t+1]

Objectif: Prouver que SI on a le filtre parfait, ALORS la méthode fonctionne.

Référence: ResultatsDesTests0.docx
- Profit Factor: 7.44
- Gain cumulé: 982% sur 10 mois
- Delta 0: Pente immédiate

Ce n'est PAS un backtest réaliste, c'est un PROOF OF CONCEPT théorique.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import filtfilt, butter

# Imports
from filters import kalman_filter  # ← NON-CAUSAL (smooth)
from adaptive_filters import kalman_filter_causal  # ← CAUSAL (filter)

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


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


def backtest_perfect_world(df, filter_name='Kalman Perfect', use_noncausal=True, trim_edges=100):
    """
    Backtest en "monde parfait" avec filtre EXACT.

    Args:
        df: DataFrame OHLCV
        filter_name: Nom du filtre
        use_noncausal: True = smooth() (parfait), False = filter() (causal)
        trim_edges: Edges à enlever

    Returns:
        dict avec résultats
    """
    print(f"\n{'='*80}")
    print(f"BACKTEST MONDE PARFAIT: {filter_name}")
    print(f"{'='*80}")

    close = df['close'].values

    # Appliquer le filtre
    if use_noncausal:
        print("🔮 Utilisation du filtre NON-CAUSAL (connaît le futur)")
        filtered = kalman_filter(close, process_variance=0.01, measurement_variance=0.1)
    else:
        print("⚙️ Utilisation du filtre CAUSAL (réalité imparfaite)")
        filtered = kalman_filter_causal(close, process_variance=0.01, measurement_variance=0.1)

    # Trim des bords
    df_trade = df.iloc[trim_edges:-trim_edges].copy()
    filtered_trade = filtered[trim_edges:-trim_edges]

    print(f"Dataset total: {len(df)} valeurs ({len(df) * 5 / 60 / 24:.1f} jours)")
    print(f"Zone de trading: {len(df_trade)} valeurs")

    # Calculer les signaux
    # RÈGLE: À t, comparer filter[t-1] > filter[t-2]
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

    # Association
    df_trade = df_trade.iloc[2:].copy()
    df_trade['signal'] = signals
    df_trade['position'] = positions
    df_trade['filtered'] = filtered_trade[2:]

    # Calculer les rendements
    # Trade à open[t+1]
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

        # Profit Factor
        gross_profit = df_trades[df_trades['return'] > 0]['return'].sum()
        gross_loss = abs(df_trades[df_trades['return'] < 0]['return'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        # Sharpe
        returns_mean = df_trades['return'].mean()
        returns_std = df_trades['return'].std()
        n_days = len(df_trade) * 5 / 60 / 24
        trades_per_year = (len(df_trades) / n_days) * 365 if n_days > 0 else 0
        sharpe = (returns_mean / returns_std) * np.sqrt(trades_per_year) if returns_std > 0 else 0

        # Drawdown
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
        'filter_name': filter_name,
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

    # Afficher
    print(f"\n📊 RÉSULTATS:")
    print(f"  Rendement stratégie: {total_return_pct:+.2f}%")
    print(f"  Rendement Buy & Hold: {buy_hold_return:+.2f}%")
    print(f"  💰 Profit Factor: {profit_factor:.2f}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_drawdown:.2f}%")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Total trades: {len(df_trades)}")

    return results


def compare_perfect_vs_reality():
    """Compare monde parfait (non-causal) vs réalité (causal)."""
    print("\n" + "="*80)
    print("VALIDATION MÉTHODE: MONDE PARFAIT vs RÉALITÉ")
    print("="*80)

    # Charger données
    df = load_btc_data_or_simulate(n=10000)

    # Test 1: Monde parfait (smooth - connaît le futur)
    results_perfect = backtest_perfect_world(
        df,
        filter_name='Kalman Perfect (smooth)',
        use_noncausal=True
    )

    # Test 2: Réalité (filter - causal)
    results_reality = backtest_perfect_world(
        df,
        filter_name='Kalman Reality (filter)',
        use_noncausal=False
    )

    # Comparaison
    print("\n" + "="*80)
    print("COMPARAISON FINALE")
    print("="*80)

    comparison = pd.DataFrame([
        {
            'Scénario': 'MONDE PARFAIT (smooth)',
            'Rendement (%)': f"{results_perfect['total_return_pct']:+.2f}",
            'Profit Factor': f"{results_perfect['profit_factor']:.2f}",
            'Sharpe': f"{results_perfect['sharpe_ratio']:.2f}",
            'Win Rate (%)': f"{results_perfect['win_rate_pct']:.1f}",
            'Trades': results_perfect['total_trades']
        },
        {
            'Scénario': 'RÉALITÉ (filter)',
            'Rendement (%)': f"{results_reality['total_return_pct']:+.2f}",
            'Profit Factor': f"{results_reality['profit_factor']:.2f}",
            'Sharpe': f"{results_reality['sharpe_ratio']:.2f}",
            'Win Rate (%)': f"{results_reality['win_rate_pct']:.1f}",
            'Trades': results_reality['total_trades']
        }
    ])

    print("\n" + comparison.to_string(index=False))

    print("\n" + "="*80)
    print("INTERPRÉTATION:")
    print("="*80)
    print("\n🔮 MONDE PARFAIT (filtre exact connaissant le futur):")
    print(f"   → Devrait avoir Profit Factor élevé (cible: 7.44 comme dans tes tests)")
    print(f"   → Valide que la MÉTHODE fonctionne SI on a le filtre parfait")
    print("\n⚙️ RÉALITÉ (filtre causal avec retard):")
    print(f"   → Performance plus faible (filtre imparfait)")
    print(f"   → Montre l'écart entre théorie et pratique")

    print("\n✅ Si Monde Parfait >> Réalité:")
    print("   → La méthode est validée théoriquement")
    print("   → Le challenge est d'avoir un filtre prédictif précis")

    return results_perfect, results_reality


if __name__ == '__main__':
    print("\n" + "="*80)
    print("VALIDATION DE LA MÉTHODE - PROOF OF CONCEPT")
    print("="*80)
    print("\n📌 OBJECTIF:")
    print("  Valider que la règle 'filter[t-1] > filter[t-2] → trade open[t+1]'")
    print("  fonctionne dans des conditions idéales (filtre exact)")
    print("\n📚 RÉFÉRENCE:")
    print("  ResultatsDesTests0.docx: Profit Factor 7.44, +982% sur 10 mois")
    print("\n⚠️ IMPORTANT:")
    print("  Ceci n'est PAS un backtest réaliste, c'est un PROOF OF CONCEPT")

    results_perfect, results_reality = compare_perfect_vs_reality()

    print("\n" + "="*80)
    print("✅ VALIDATION TERMINÉE")
    print("="*80)
