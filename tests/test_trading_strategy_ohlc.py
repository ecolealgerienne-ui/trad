#!/usr/bin/env python3
"""
Backtesting de la stratégie de trading avec données OHLC préparées.

ENTRÉE:
    Fichier .npz généré par prepare_data_ohlc_v2.py
    Exemple: dataset_btc_eth_bnb_ada_ltc_ohlcv2_close_octave20.npz

DONNÉES DISPONIBLES:
    X: Features OHLC (O_ret, H_ret, L_ret, C_ret, Range_ret) - séquences de 12 timesteps
    Y: Labels (0/1) basés sur filtered[i-2] > filtered[i-3-delta]

STRATÉGIE:
    Signal[i] = Y[i] (label ou prédiction)
    - Signal = 1 → LONG
    - Signal = 0 → SHORT

    Exécution:
    - Entrer en position au changement de signal
    - Rester tant que le signal est stable
    - Sortir (et enregistrer le rendement) au prochain changement

    Rendement calculé avec C_ret (rendement clôture-à-clôture)

Usage:
    python tests/test_trading_strategy_ohlc.py --data data/prepared/dataset_xxx.npz
    python tests/test_trading_strategy_ohlc.py --data data/prepared/dataset_xxx.npz --model models/best_model.pth
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
import logging

logger = logging.getLogger(__name__)

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Index des features dans X
FEATURE_INDICES = {
    'o_ret': 0,
    'h_ret': 1,
    'l_ret': 2,
    'c_ret': 3,
    'range_ret': 4
}


def load_dataset(npz_path: str) -> dict:
    """
    Charge le dataset .npz et extrait les métadonnées.

    Returns:
        dict avec X_train, Y_train, X_val, Y_val, X_test, Y_test, metadata
    """
    logger.info(f"📂 Chargement: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    result = {
        'X_train': data['X_train'],
        'Y_train': data['Y_train'],
        'X_val': data['X_val'],
        'Y_val': data['Y_val'],
        'X_test': data['X_test'],
        'Y_test': data['Y_test'],
    }

    # Charger metadata
    if 'metadata' in data:
        metadata_str = str(data['metadata'])
        result['metadata'] = json.loads(metadata_str)
    else:
        result['metadata'] = {}

    logger.info(f"  X_test shape: {result['X_test'].shape}")
    logger.info(f"  Y_test shape: {result['Y_test'].shape}")

    if result['metadata']:
        logger.info(f"  Target: {result['metadata'].get('target', 'N/A')}")
        logger.info(f"  Label: {result['metadata'].get('label_formula', 'N/A')}")

    return result


def load_model_predictions(model_path: str, X: np.ndarray) -> np.ndarray:
    """
    Charge un modèle et génère les prédictions.

    Returns:
        np.array de prédictions (0 ou 1)
    """
    import torch
    from model import MultiOutputCNNLSTM

    logger.info(f"🤖 Chargement modèle: {model_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Charger le modèle
    checkpoint = torch.load(model_path, map_location=device)

    # Créer le modèle
    n_features = X.shape[2]
    model = MultiOutputCNNLSTM(
        input_size=n_features,
        num_outputs=1  # Single output pour OHLC
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Prédictions
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = (torch.sigmoid(outputs) > 0.5).cpu().numpy().flatten()

    logger.info(f"  Prédictions générées: {len(predictions)}")
    logger.info(f"  Distribution: {predictions.mean()*100:.1f}% LONG, {(1-predictions.mean())*100:.1f}% SHORT")

    return predictions.astype(int)


def backtest_strategy(X: np.ndarray, signals: np.ndarray, name: str = "Strategy") -> dict:
    """
    Backtest de la stratégie basée sur les signaux.

    Args:
        X: Features OHLC (n_samples, seq_length, n_features)
        signals: Signaux de trading (n_samples,) - 1=LONG, 0=SHORT
        name: Nom de la stratégie pour affichage

    Returns:
        dict avec résultats du backtest
    """
    print(f"\n{'='*80}")
    print(f"BACKTEST: {name}")
    print(f"{'='*80}")

    n_samples = len(signals)
    signals = signals.flatten()

    # Extraire C_ret de la dernière feature de chaque séquence
    # X[i, -1, 3] = C_ret à l'instant correspondant à la séquence i
    c_ret = X[:, -1, FEATURE_INDICES['c_ret']]

    print(f"Samples: {n_samples}")
    print(f"C_ret range: [{c_ret.min()*100:.2f}%, {c_ret.max()*100:.2f}%]")
    print(f"Signals: {signals.sum()} LONG ({signals.mean()*100:.1f}%), {n_samples - signals.sum()} SHORT")

    # =========================================================================
    # LOGIQUE DE TRADING
    # =========================================================================
    # - Entrer en position au changement de signal
    # - Rester tant que le signal est stable
    # - Sortir au prochain changement et enregistrer le rendement
    #
    # Signal = 1 → LONG (on gagne si le prix monte)
    # Signal = 0 → SHORT (on gagne si le prix baisse)
    #
    # Rendement LONG: produit des (1 + C_ret) sur la période
    # Rendement SHORT: produit des (1 - C_ret) sur la période (ou inverse)
    # =========================================================================

    trades_list = []
    entry_idx = None
    entry_position = None  # 1 = LONG, -1 = SHORT
    cumulative_return = 1.0  # Pour accumuler le rendement pendant la position

    for i in range(1, n_samples):
        current_signal = signals[i]
        prev_signal = signals[i-1]

        # Première entrée
        if i == 1:
            entry_idx = 0
            entry_position = 1 if signals[0] == 1 else -1
            cumulative_return = 1.0

        # Accumuler le rendement de la bougie actuelle
        if entry_position == 1:  # LONG
            cumulative_return *= (1 + c_ret[i])
        else:  # SHORT
            cumulative_return *= (1 - c_ret[i])

        # Détecter changement de signal
        if current_signal != prev_signal:
            # Fermer la position précédente
            trade_return = cumulative_return - 1.0

            trades_list.append({
                'entry_idx': entry_idx,
                'exit_idx': i,
                'duration': i - entry_idx,
                'position_type': 'LONG' if entry_position == 1 else 'SHORT',
                'return': trade_return,
                'return_pct': trade_return * 100
            })

            # Ouvrir nouvelle position
            entry_idx = i
            entry_position = 1 if current_signal == 1 else -1
            cumulative_return = 1.0

    # Fermer la dernière position
    if entry_idx is not None and entry_idx < n_samples - 1:
        trade_return = cumulative_return - 1.0

        trades_list.append({
            'entry_idx': entry_idx,
            'exit_idx': n_samples - 1,
            'duration': n_samples - 1 - entry_idx,
            'position_type': 'LONG' if entry_position == 1 else 'SHORT',
            'return': trade_return,
            'return_pct': trade_return * 100
        })

    # Créer DataFrame des trades
    df_trades = pd.DataFrame(trades_list)

    # Calculer métriques
    if len(df_trades) > 0:
        # Rendement total (produit composé)
        total_return = (1 + df_trades['return']).prod() - 1
        total_return_pct = total_return * 100

        # Win rate
        wins = (df_trades['return'] > 0).sum()
        total_trades = len(df_trades)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        # Rendement moyen par trade
        avg_return = df_trades['return'].mean() * 100

        # Durée moyenne des trades (en périodes et en jours)
        avg_duration = df_trades['duration'].mean()
        avg_duration_days = avg_duration * 5 / 60 / 24  # 5 min par période

        # Sharpe ratio (simplifié)
        returns_std = df_trades['return'].std()
        sharpe = (df_trades['return'].mean() / returns_std) * np.sqrt(252) if returns_std > 0 else 0

        # Max drawdown
        df_trades['cumulative'] = (1 + df_trades['return']).cumprod()
        running_max = df_trades['cumulative'].expanding().max()
        df_trades['drawdown'] = (df_trades['cumulative'] - running_max) / running_max
        max_drawdown = df_trades['drawdown'].min() * 100

        # Trades LONG vs SHORT
        long_trades = df_trades[df_trades['position_type'] == 'LONG']
        short_trades = df_trades[df_trades['position_type'] == 'SHORT']

        long_return = (1 + long_trades['return']).prod() - 1 if len(long_trades) > 0 else 0
        short_return = (1 + short_trades['return']).prod() - 1 if len(short_trades) > 0 else 0

    else:
        total_return_pct = 0
        win_rate = 0
        avg_return = 0
        avg_duration = 0
        avg_duration_days = 0
        sharpe = 0
        max_drawdown = 0
        total_trades = 0
        long_return = 0
        short_return = 0

    # Buy & Hold pour comparaison
    buy_hold_return = (1 + c_ret).prod() - 1
    buy_hold_return_pct = buy_hold_return * 100

    # Durée totale du backtest en jours
    total_duration_days = n_samples * 5 / 60 / 24

    results = {
        'name': name,
        'total_return_pct': total_return_pct,
        'buy_hold_return_pct': buy_hold_return_pct,
        'win_rate_pct': win_rate,
        'avg_return_pct': avg_return,
        'avg_duration': avg_duration,
        'avg_duration_days': avg_duration_days,
        'total_duration_days': total_duration_days,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_drawdown,
        'total_trades': total_trades,
        'long_return_pct': long_return * 100,
        'short_return_pct': short_return * 100,
        'df_trades': df_trades
    }

    # Afficher résultats
    print(f"\n📊 RÉSULTATS:")
    print(f"  Durée totale backtest: {total_duration_days:.1f} jours")
    print(f"  Rendement stratégie: {total_return_pct:+.2f}%")
    print(f"  Rendement Buy & Hold: {buy_hold_return_pct:+.2f}%")
    print(f"  Surperformance: {total_return_pct - buy_hold_return_pct:+.2f}%")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_drawdown:.2f}%")
    print(f"  Total trades: {total_trades}")
    print(f"  Durée moyenne trade: {avg_duration:.1f} périodes ({avg_duration_days:.2f} jours)")
    print(f"  Rendement LONG: {long_return*100:+.2f}%")
    print(f"  Rendement SHORT: {short_return*100:+.2f}%")

    return results


def visualize_results(results: dict, output_path: str = None):
    """
    Visualise les résultats du backtest.
    """
    df_trades = results['df_trades']

    if len(df_trades) == 0:
        print("Pas de trades à visualiser")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Courbe de capital
    cumulative = (1 + df_trades['return']).cumprod()
    axes[0, 0].plot(cumulative.values, 'b-', linewidth=2)
    axes[0, 0].axhline(y=1, color='black', linestyle='--', linewidth=1)
    axes[0, 0].set_title(f"Courbe de Capital - {results['name']}", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Trade #')
    axes[0, 0].set_ylabel('Capital (base 1.0)')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Distribution des rendements
    axes[0, 1].hist(df_trades['return_pct'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].axvline(x=df_trades['return_pct'].mean(), color='green', linestyle='-', linewidth=2, label=f"Moyenne: {df_trades['return_pct'].mean():.2f}%")
    axes[0, 1].set_title('Distribution des Rendements par Trade', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Rendement (%)')
    axes[0, 1].set_ylabel('Fréquence')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Rendements LONG vs SHORT
    long_returns = df_trades[df_trades['position_type'] == 'LONG']['return_pct']
    short_returns = df_trades[df_trades['position_type'] == 'SHORT']['return_pct']

    data_to_plot = []
    labels = []
    if len(long_returns) > 0:
        data_to_plot.append(long_returns)
        labels.append(f'LONG (n={len(long_returns)})')
    if len(short_returns) > 0:
        data_to_plot.append(short_returns)
        labels.append(f'SHORT (n={len(short_returns)})')

    if data_to_plot:
        bp = axes[1, 0].boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = ['lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)

    axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].set_title('Rendements LONG vs SHORT', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Rendement (%)')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Durée des trades
    axes[1, 1].hist(df_trades['duration'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].axvline(x=df_trades['duration'].mean(), color='red', linestyle='-', linewidth=2, label=f"Moyenne: {df_trades['duration'].mean():.1f}")
    axes[1, 1].set_title('Distribution de la Durée des Trades', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Durée (périodes)')
    axes[1, 1].set_ylabel('Fréquence')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Backtest: {results['name']}\nRendement: {results['total_return_pct']:+.2f}% | Win Rate: {results['win_rate_pct']:.1f}% | Sharpe: {results['sharpe_ratio']:.2f}",
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path is None:
        output_path = Path('tests/validation_output/trading_strategy_ohlc.png')
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualisation sauvegardée: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Backtest stratégie de trading avec données OHLC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Mode Oracle (utilise les labels réels Y)
  python tests/test_trading_strategy_ohlc.py --data data/prepared/dataset_xxx.npz

  # Limiter à 20000 samples (~69 jours)
  python tests/test_trading_strategy_ohlc.py --data data/prepared/dataset_xxx.npz --limit 20000

  # Mode Modèle (utilise les prédictions du modèle)
  python tests/test_trading_strategy_ohlc.py --data data/prepared/dataset_xxx.npz --model models/best_model.pth

  # Spécifier le split à utiliser
  python tests/test_trading_strategy_ohlc.py --data data/prepared/dataset_xxx.npz --split train
        """
    )

    parser.add_argument('--data', '-d', type=str, required=True,
                        help='Chemin vers le dataset .npz')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Chemin vers le modèle .pth (optionnel, sinon utilise les labels)')
    parser.add_argument('--split', '-s', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Split à utiliser (défaut: test)')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Limiter le nombre de samples (ex: 20000)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Chemin de sortie pour la visualisation')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "="*80)
    print("BACKTESTING STRATÉGIE DE TRADING - OHLC")
    print("="*80)

    # Charger le dataset
    dataset = load_dataset(args.data)

    # Sélectionner le split
    split_map = {
        'train': ('X_train', 'Y_train'),
        'val': ('X_val', 'Y_val'),
        'test': ('X_test', 'Y_test')
    }
    X_key, Y_key = split_map[args.split]
    X = dataset[X_key]
    Y = dataset[Y_key]

    # Limiter le nombre de samples si demandé
    if args.limit and args.limit < len(X):
        X = X[:args.limit]
        Y = Y[:args.limit]
        print(f"\n📊 Split utilisé: {args.split} (limité à {args.limit} samples)")
    else:
        print(f"\n📊 Split utilisé: {args.split}")
    print(f"   Samples: {len(X)}")

    # Calculer durée totale en jours (5 min par période)
    total_periods = len(X)
    total_days = total_periods * 5 / 60 / 24
    print(f"   Durée totale: {total_days:.1f} jours ({total_periods} périodes × 5min)")

    # Déterminer les signaux
    if args.model:
        # Mode Modèle : utiliser les prédictions
        signals = load_model_predictions(args.model, X)
        strategy_name = f"Modèle ({Path(args.model).stem})"
    else:
        # Mode Oracle : utiliser les labels réels
        signals = Y.flatten()
        strategy_name = "Oracle (Labels Réels)"

    # Backtest
    results = backtest_strategy(X, signals, name=strategy_name)

    # Visualisation
    visualize_results(results, args.output)

    # Résumé final
    print("\n" + "="*80)
    print("✅ BACKTEST TERMINÉ")
    print("="*80)

    metadata = dataset.get('metadata', {})
    print(f"\n📈 RÉSUMÉ:")
    print(f"  Dataset: {Path(args.data).name}")
    print(f"  Target: {metadata.get('target', 'N/A')}")
    print(f"  Label: {metadata.get('label_formula', 'N/A')}")
    print(f"  Split: {args.split}")
    print(f"  Samples: {len(X)}")
    print(f"  Durée: {results['total_duration_days']:.1f} jours")
    print(f"  Mode: {'Modèle' if args.model else 'Oracle'}")
    print(f"\n  Rendement: {results['total_return_pct']:+.2f}%")
    print(f"  vs Buy&Hold: {results['buy_hold_return_pct']:+.2f}%")
    print(f"  Surperformance: {results['total_return_pct'] - results['buy_hold_return_pct']:+.2f}%")
    print(f"  Trades: {results['total_trades']} (durée moy: {results['avg_duration_days']:.2f} jours)")


if __name__ == '__main__':
    main()
