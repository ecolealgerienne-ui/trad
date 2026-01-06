#!/usr/bin/env python3
"""
Test de Validation: Profitability-Based Relabeling (Proposition B)

Approche: Au lieu de deviner ce qui est un piège via des proxies (Durée/Vol),
on regarde directement le PnL futur pour chaque signal STRONG.

Algorithme:
  Pour chaque signal STRONG à t:
    1. Simuler le trade (entrer si STRONG prédit)
    2. Calculer Max Return sur k prochaines bougies
    3. Si Max Return < Frais (0.2%) → Relabeler Force=WEAK
    4. Sinon → Garder Force=STRONG

Avantage: Zéro hypothèse, nettoyage parfait, apprentissage optimal.
"""

import numpy as np
import argparse
from pathlib import Path
import sys

# Ajouter src/ au path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_utils import load_prepared_data


def profitability_relabeling(X, Y, prices, fees=0.002, horizon=12, min_return_threshold=None):
    """
    Relabeling basé sur la profitabilité RÉELLE.

    Pour chaque signal STRONG:
      - Simuler le trade
      - Calculer Max Return sur `horizon` prochaines bougies
      - Si Max Return < fees → Relabeler WEAK (faux positif)

    Args:
        X: Features (n, seq_len, n_features)
        Y: Labels (n, 2) - [direction, force]
        prices: Prix Close (n,)
        fees: Frais totaux (entrée + sortie) en % (default: 0.002 = 0.2%)
        horizon: Nombre de bougies à regarder dans le futur (default: 12 = 1h)
        min_return_threshold: Seuil minimum de return (default: fees)

    Returns:
        Y_relabeled: Labels modifiés
        stats: Statistiques de relabeling
    """
    if min_return_threshold is None:
        min_return_threshold = fees

    Y_relabeled = Y.copy()

    n_strong_total = np.sum(Y[:, 1] == 1)
    n_relabeled = 0
    relabeled_indices = []

    # Stats des trades relabelés
    relabeled_max_returns = []

    for i in range(len(Y) - horizon):
        # Si ce sample est STRONG
        if Y[i, 1] == 1:
            direction = Y[i, 0]  # Direction (UP=1, DOWN=0)

            # Calculer les returns futurs sur l'horizon
            future_returns = []
            for j in range(1, horizon + 1):
                ret = (prices[i+j] - prices[i]) / prices[i]
                if direction == 0:  # SHORT
                    ret = -ret  # Inverser si SHORT
                future_returns.append(ret)

            # Max return sur l'horizon (meilleur exit possible)
            max_return = max(future_returns)

            # Si le max return ne couvre même pas les frais → FAUX POSITIF
            if max_return < min_return_threshold:
                Y_relabeled[i, 1] = 0  # Relabeler Force=WEAK
                n_relabeled += 1
                relabeled_indices.append(i)
                relabeled_max_returns.append(max_return)

    # Stats
    stats = {
        'n_strong_total': n_strong_total,
        'n_relabeled': n_relabeled,
        'pct_relabeled': 100.0 * n_relabeled / n_strong_total if n_strong_total > 0 else 0,
        'relabeled_indices': relabeled_indices,
        'relabeled_max_returns': relabeled_max_returns,
        'avg_max_return_relabeled': np.mean(relabeled_max_returns) if relabeled_max_returns else 0,
    }

    return Y_relabeled, stats


def simulate_trading(Y, prices, fees=0.002):
    """
    Simule une stratégie de trading Oracle basée sur les labels.

    Args:
        Y: Labels (n, 2) - [direction, force]
        prices: Prix Close (n,)
        fees: Frais par trade (entrée + sortie)

    Returns:
        Dict avec métriques de trading
    """
    position = 0  # 0=FLAT, 1=LONG, -1=SHORT
    entry_price = 0
    pnl_total = 0
    trades = []

    for i in range(len(Y)):
        direction = Y[i, 0]  # 1=UP, 0=DOWN
        force = Y[i, 1]      # 1=STRONG, 0=WEAK

        # Logique Oracle: Agir uniquement si STRONG
        if force == 1:
            target_position = 1 if direction == 1 else -1
        else:
            target_position = 0

        # Gestion transitions
        if target_position != position:
            # Sortir position actuelle si existante
            if position != 0:
                exit_price = prices[i]
                ret = (exit_price - entry_price) / entry_price
                if position == -1:
                    ret = -ret
                pnl = ret - fees
                pnl_total += pnl
                trades.append(pnl)

            # Entrer nouvelle position si target != FLAT
            if target_position != 0:
                entry_price = prices[i]

            position = target_position

    # Clore position finale si existante
    if position != 0 and len(prices) > 0:
        exit_price = prices[-1]
        ret = (exit_price - entry_price) / entry_price
        if position == -1:
            ret = -ret
        pnl = ret - fees
        pnl_total += pnl
        trades.append(pnl)

    # Métriques
    if trades:
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        win_rate = 100.0 * len(wins) / len(trades)
        avg_win = np.mean(wins) * 100 if wins else 0
        avg_loss = np.mean(losses) * 100 if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0

    return {
        'trades': trades,
        'n_trades': len(trades),
        'win_rate': win_rate,
        'pnl_total': pnl_total * 100,  # En %
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
    }


def calculate_predictivity(Y, prices):
    """
    Calcule la corrélation entre labels STRONG et returns futurs.

    Prédictivité STRONG = Corr(Force=1, Future_Return > 0)
    """
    strong_mask = (Y[:, 1] == 1)
    if np.sum(strong_mask) == 0:
        return 0

    # Returns 1 période ahead
    returns = np.diff(prices) / prices[:-1]
    returns = np.append(returns, 0)  # Pad dernier

    # Direction des returns
    direction_mask = (Y[:, 0] == 1)
    returns_adjusted = np.where(direction_mask, returns, -returns)

    # Corrélation
    strong_returns = returns_adjusted[strong_mask]
    predictivity = np.mean(strong_returns > 0) if len(strong_returns) > 0 else 0

    return predictivity


def main():
    parser = argparse.ArgumentParser(description='Test Profitability-Based Relabeling')
    parser.add_argument(
        '--indicator',
        type=str,
        default='macd',
        choices=['macd', 'rsi', 'cci'],
        help='Indicateur à tester'
    )
    parser.add_argument(
        '--fees',
        type=float,
        default=0.002,
        help='Frais totaux (entrée + sortie) en décimal (default: 0.002 = 0.2%%)'
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=12,
        help='Horizon de recherche du Max Return en bougies (default: 12 = 1h)'
    )
    parser.add_argument(
        '--threshold-multiplier',
        type=float,
        default=1.0,
        help='Multiplicateur du seuil (default: 1.0 = fees exactement). Ex: 1.5 = fees × 1.5'
    )

    args = parser.parse_args()

    indicator = args.indicator.lower()
    fees = args.fees
    horizon = args.horizon
    min_return_threshold = fees * args.threshold_multiplier

    print("=" * 80)
    print("TEST DE VALIDATION: Profitability-Based Relabeling (Proposition B)")
    print("=" * 80)
    print()
    print(f"Indicateur: {indicator.upper()}")
    print(f"Frais: {fees*100:.2f}%")
    print(f"Horizon: {horizon} bougies (~{horizon*5} min)")
    print(f"Seuil minimum return: {min_return_threshold*100:.2f}%")
    print()

    # Charger données test
    print("📁 Chargement données test...")
    dataset_pattern = f"dataset_*_{indicator}_dual_binary_kalman.npz"
    dataset_files = list(Path('data/prepared').glob(dataset_pattern))

    if not dataset_files:
        print(f"❌ Aucun dataset trouvé: {dataset_pattern}")
        return 1

    dataset_path = dataset_files[0]
    print(f"   Dataset: {dataset_path.name}")

    splits = load_prepared_data(str(dataset_path))
    X_test = splits['X_test']
    Y_test = splits['Y_test']

    print(f"   Samples test: {len(X_test)}")

    # Reconstruire prix à partir des returns (comme test_relabeling_impact.py)
    print("   Reconstruction prix à partir de c_ret...")
    idx_ret = 2 if indicator == 'cci' else 0  # CCI: c_ret est index 2, autres: index 0
    returns = X_test[:, -1, idx_ret]  # Dernier timestep de chaque séquence

    # Reconstruire série de prix (prix initial arbitraire = 100)
    prices_test = 100 * np.cumprod(1 + returns)
    print(f"   Prix reconstruits: {len(prices_test)} samples")
    print()

    # Application Profitability Relabeling
    print("🔄 Application Profitability Relabeling...")
    Y_relabeled, stats = profitability_relabeling(
        X_test, Y_test, prices_test,
        fees=fees,
        horizon=horizon,
        min_return_threshold=min_return_threshold
    )

    print(f"   Relabeling: {stats['n_relabeled']} labels Force 1→0")
    print(f"   % STRONG relabelés: {stats['pct_relabeled']:.1f}%")
    print(f"   Avg Max Return (relabelés): {stats['avg_max_return_relabeled']*100:.3f}%")
    print()

    # Comparaison Oracle AVANT vs APRÈS
    print("=" * 80)
    print("📊 COMPARAISON ORACLE AVANT vs APRÈS")
    print("=" * 80)
    print()

    # Oracle AVANT
    print("Oracle AVANT (baseline):")
    trading_before = simulate_trading(Y_test, prices_test, fees=fees)
    pred_before = calculate_predictivity(Y_test, prices_test)

    print(f"   Prédictivité STRONG:        {pred_before:.4f}")
    print()
    print("Oracle AVANT - Trading Simulé:")
    print(f"   Trades totaux:    {trading_before['n_trades']}")
    print(f"   Win Rate:         {trading_before['win_rate']:.2f}%")
    print(f"   PnL Total:        {trading_before['pnl_total']:+.2f}%")
    print(f"   Avg Win:          {trading_before['avg_win']:+.3f}%")
    print(f"   Avg Loss:         {trading_before['avg_loss']:+.3f}%")
    print(f"   Profit Factor:    {trading_before['profit_factor']:.2f}")
    print()

    # Oracle APRÈS
    print("Oracle APRÈS (relabeled):")
    trading_after = simulate_trading(Y_relabeled, prices_test, fees=fees)
    pred_after = calculate_predictivity(Y_relabeled, prices_test)

    print(f"   Prédictivité STRONG:        {pred_after:.4f}")
    print()
    print("Oracle APRÈS - Trading Simulé:")
    print(f"   Trades totaux:    {trading_after['n_trades']}")
    print(f"   Win Rate:         {trading_after['win_rate']:.2f}%")
    print(f"   PnL Total:        {trading_after['pnl_total']:+.2f}%")
    print(f"   Avg Win:          {trading_after['avg_win']:+.3f}%")
    print(f"   Avg Loss:         {trading_after['avg_loss']:+.3f}%")
    print(f"   Profit Factor:    {trading_after['profit_factor']:.2f}")
    print()

    # Synthèse
    print("=" * 80)
    print("🎯 SYNTHÈSE")
    print("=" * 80)
    print()
    print("Impact Relabeling:")
    delta_wr = trading_after['win_rate'] - trading_before['win_rate']
    delta_pnl = trading_after['pnl_total'] - trading_before['pnl_total']
    delta_pred = pred_after - pred_before
    delta_pf = trading_after['profit_factor'] - trading_before['profit_factor']
    delta_trades = trading_after['n_trades'] - trading_before['n_trades']

    print(f"   ΔWin Rate:        {delta_wr:+.2f}%")
    print(f"   ΔPnL Total:       {delta_pnl:+.2f}%")
    print(f"   ΔPrédictivité:    {delta_pred:+.4f} ({delta_pred/pred_before*100:+.1f}%)")
    print(f"   ΔProfit Factor:   {delta_pf:+.2f} ({delta_pf/trading_before['profit_factor']*100:+.1f}%)")
    print(f"   ΔTrades:          {delta_trades:+d} ({delta_trades/trading_before['n_trades']*100:+.1f}%)")
    print()

    # Verdict
    success_criteria = (
        delta_wr > 1.0 and  # Win Rate +1%+
        delta_pnl > -3000 and  # PnL pas catastrophique
        delta_pred > 0.05  # Prédictivité +5%+
    )

    if success_criteria:
        print("✅ VALIDATION RÉUSSIE: Profitability Relabeling améliore la qualité!")
        print("   → Procéder au relabeling complet et réentraînement")
    elif delta_wr > 0 and delta_pred > 0:
        print("⚠️  VALIDATION MITIGÉE: Amélioration modérée")
        print("   → Tester avec d'autres paramètres (horizon, threshold)")
    else:
        print("❌ VALIDATION ÉCHOUÉE: Profitability Relabeling n'améliore pas")
        print("   → Revoir approche ou paramètres")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
