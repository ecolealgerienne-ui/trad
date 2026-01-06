#!/usr/bin/env python3
"""
Test de Validation: Smart Hybrid Relabeling (Proposition A)

Approche: Heuristique affinée qui différencie le bruit pur des structures naissantes.

Règles:
  - Durée 3 (15 min):    SUPPRIMER TOUT (bruit pur, jamais suffisant pour payer frais)
  - Durée 4-5 (20-25 min): SUPPRIMER SI Vol Q4 uniquement (structures naissantes,
                           garder si volatilité saine, jeter si "pump" nerveux)

Impact attendu: Entre Config 3 (trop agressif) et Config 4 (trop conservateur).
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def smart_hybrid_relabeling(X, Y, duration, vol_rolling):
    """
    Relabeling avec heuristique affinée (Proposition A).

    Règles:
      - Durée 3: TOUT relabeler (bruit pur)
      - Durée 4-5: Relabeler SI Vol Q4 uniquement

    Args:
        X: Features (n, seq_len, n_features)
        Y: Labels (n, 2) - [direction, force]
        duration: Durées STRONG (n,)
        vol_rolling: Volatilité rolling (n,)

    Returns:
        Y_relabeled: Labels modifiés
        stats: Statistiques
    """
    Y_relabeled = Y.copy()

    # Calculer Q4 volatilité
    q4_threshold = np.percentile(vol_rolling, 75)

    # Masques
    mask_duration_3 = (duration == 3)
    mask_duration_4_5 = np.isin(duration, [4, 5])
    mask_vol_q4 = (vol_rolling > q4_threshold)

    # Règle Durée 3: TOUT relabeler
    mask_trap_d3 = mask_duration_3

    # Règle Durée 4-5: Relabeler SI Vol Q4 uniquement
    mask_trap_d45 = mask_duration_4_5 & mask_vol_q4

    # Combinaison
    mask_trap = mask_trap_d3 | mask_trap_d45

    # Appliquer relabeling
    n_relabeled = 0
    for i in np.where(mask_trap)[0]:
        if Y_relabeled[i, 1] == 1:  # Si STRONG
            Y_relabeled[i, 1] = 0  # → WEAK
            n_relabeled += 1

    # Stats
    n_d3 = np.sum(mask_trap_d3 & (Y[:, 1] == 1))
    n_d45_vol = np.sum(mask_trap_d45 & (Y[:, 1] == 1))

    stats = {
        'n_relabeled': n_relabeled,
        'n_duration_3': n_d3,
        'n_duration_4_5_vol_q4': n_d45_vol,
        'q4_threshold': q4_threshold,
    }

    return Y_relabeled, stats


def simulate_trading(Y, prices, fees=0.002):
    """Simule trading Oracle."""
    position = 0
    entry_price = 0
    pnl_total = 0
    trades = []

    for i in range(len(Y)):
        direction = Y[i, 0]
        force = Y[i, 1]

        if force == 1:
            target_position = 1 if direction == 1 else -1
        else:
            target_position = 0

        if target_position != position:
            if position != 0:
                exit_price = prices[i]
                ret = (exit_price - entry_price) / entry_price
                if position == -1:
                    ret = -ret
                pnl = ret - fees
                pnl_total += pnl
                trades.append(pnl)

            if target_position != 0:
                entry_price = prices[i]

            position = target_position

    if position != 0 and len(prices) > 0:
        exit_price = prices[-1]
        ret = (exit_price - entry_price) / entry_price
        if position == -1:
            ret = -ret
        pnl = ret - fees
        pnl_total += pnl
        trades.append(pnl)

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
        'pnl_total': pnl_total * 100,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
    }


def calculate_predictivity(Y, prices):
    """Calcule prédictivité STRONG."""
    strong_mask = (Y[:, 1] == 1)
    if np.sum(strong_mask) == 0:
        return 0

    returns = np.diff(prices) / prices[:-1]
    returns = np.append(returns, 0)

    direction_mask = (Y[:, 0] == 1)
    returns_adjusted = np.where(direction_mask, returns, -returns)

    strong_returns = returns_adjusted[strong_mask]
    predictivity = np.mean(strong_returns > 0) if len(strong_returns) > 0 else 0

    return predictivity


def main():
    parser = argparse.ArgumentParser(description='Test Smart Hybrid Relabeling (Proposition A)')
    parser.add_argument(
        '--indicator',
        type=str,
        default='macd',
        choices=['macd', 'rsi', 'cci'],
        help='Indicateur à tester'
    )

    args = parser.parse_args()
    indicator = args.indicator.lower()

    print("=" * 80)
    print("TEST DE VALIDATION: Smart Hybrid Relabeling (Proposition A)")
    print("=" * 80)
    print()
    print(f"Indicateur: {indicator.upper()}")
    print()
    print("Règles:")
    print("  - Durée 3:    SUPPRIMER TOUT (bruit pur)")
    print("  - Durée 4-5:  SUPPRIMER SI Vol Q4 uniquement")
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

    data = np.load(str(dataset_path), allow_pickle=True)
    X_test = data['X_test']
    Y_test = data['Y_test']

    print(f"   Samples test: {len(X_test)}")

    # Recalculer métadonnées à partir de X (comme test_relabeling_impact.py)
    print("   Recalcul métadonnées (duration, vol_rolling, prices)...")
    idx_ret = 2 if indicator == 'cci' else 0  # CCI: c_ret est index 2, autres: index 0
    returns = X_test[:, -1, idx_ret]

    # Duration
    force_labels = Y_test[:, 1]
    duration_test = np.zeros_like(force_labels, dtype=int)
    count = 0
    for i in range(len(force_labels)):
        if force_labels[i] == 1:
            count += 1
        else:
            count = 0
        duration_test[i] = count

    # Vol rolling
    vol_rolling_test = pd.Series(returns).abs().rolling(window=20).mean().fillna(0).values

    # Prix (pour affichage uniquement)
    prices_test = 100 * np.cumprod(1 + returns)

    print(f"   Métadonnées recalculées: duration, vol_rolling, prices")
    print()

    # Application Smart Hybrid Relabeling
    print("🔄 Application Smart Hybrid Relabeling...")
    Y_relabeled, stats = smart_hybrid_relabeling(X_test, Y_test, duration_test, vol_rolling_test)

    print(f"   Relabeling: {stats['n_relabeled']} labels Force 1→0")
    print(f"     - Durée 3:           {stats['n_duration_3']}")
    print(f"     - Durée 4-5 ET Vol Q4: {stats['n_duration_4_5_vol_q4']}")
    print(f"   Vol Q4 threshold: {stats['q4_threshold']:.6f}")
    print()

    # Comparaison Oracle AVANT vs APRÈS
    print("=" * 80)
    print("📊 COMPARAISON ORACLE AVANT vs APRÈS")
    print("=" * 80)
    print()

    # Oracle AVANT
    print("Oracle AVANT (baseline):")
    trading_before = simulate_trading(Y_test, prices_test, fees=0.002)
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
    trading_after = simulate_trading(Y_relabeled, prices_test, fees=0.002)
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
        delta_wr > 1.0 and
        delta_pnl > -5000 and
        delta_pred > 0.03
    )

    if success_criteria:
        print("✅ VALIDATION RÉUSSIE: Smart Hybrid améliore la qualité!")
        print("   → Procéder au relabeling complet")
    elif delta_wr > 0 and delta_pred > 0:
        print("⚠️  VALIDATION MITIGÉE: Amélioration modérée")
        print("   → Comparer avec Proposition B")
    else:
        print("❌ VALIDATION ÉCHOUÉE: Smart Hybrid n'améliore pas")
        print("   → Essayer Proposition B")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
