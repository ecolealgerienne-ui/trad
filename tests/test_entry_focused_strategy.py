#!/usr/bin/env python3
"""
Test Entry-Focused Strategy - Use ML for Entries, Ignore ML for Exits

Principe:
- ENTRÉES: Utiliser ML predictions (92.5% accuracy validé)
- SORTIES: Ignorer ML, utiliser Time-based + Return-based (TP/SL)
- ANTI-FLIP: Si sortie LONG → bloquer re-LONG jusqu'à signal SHORT (et vice-versa)

Règle Anti-Flip Stricte:
  Exit LONG  → Bloquer LONG jusqu'à prise SHORT
  Exit SHORT → Bloquer SHORT jusqu'à prise LONG
  Force alternance réelle LONG/SHORT

Objectif:
- Réduire les micro-sorties (98% flickering)
- Maintenir la qualité des entrées (92.5%)
- Forcer vraie alternance directionnelle

Usage:
    python tests/test_entry_focused_strategy.py --indicator macd --split test
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass
from enum import IntEnum


class Position(IntEnum):
    """Position types."""
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass
class ExitConfig:
    """Configuration for hybrid exit strategy."""
    holding_min: int         # Time-based: minimum holding periods
    take_profit: float       # Return-based: TP threshold (e.g. 0.02 = +2%)
    stop_loss: float         # Return-based: SL threshold (e.g. -0.01 = -1%)


@dataclass
class TradeResult:
    """Results for a single configuration."""
    config: ExitConfig
    n_trades: int
    n_long: int
    n_short: int
    win_rate: float
    pnl_gross: float
    pnl_net: float
    avg_duration: float

    # Breakdown by exit reason
    exits_time: int
    exits_tp: int
    exits_sl: int
    exits_flip: int  # Model changed direction (rare in this strategy)

    # Anti-flip stats
    entries_blocked: int

    # Return distribution
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float


def load_data(indicator: str, filter_type: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load predictions and returns.

    Args:
        indicator: 'macd', 'rsi', 'cci'
        filter_type: 'kalman' ou 'octave20'
        split: 'train', 'val', 'test'

    Returns:
        (predictions, returns)
        - predictions: (n_samples,) - probabilities [0,1]
        - returns: (n_samples,) - period returns
    """
    dataset_pattern = f"dataset_btc_eth_bnb_ada_ltc_{indicator}_direction_only_{filter_type}.npz"
    dataset_path = Path("data/prepared") / dataset_pattern

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {dataset_path}")

    print(f"📂 Chargement: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)

    # Load predictions
    Y_pred = data.get(f'Y_{split}_pred', None)
    if Y_pred is None:
        raise ValueError(f"Y_{split}_pred introuvable dans {dataset_path}")

    predictions = Y_pred.flatten()  # (n_samples,) - probabilities [0,1]

    # Load returns
    X = data[f'X_{split}']
    # X shape: (n_samples, 25, 1) for Direction-Only
    # Use last timestep close return as proxy for period return
    returns = X[:, -1, 0]  # (n_samples,) - last close return of each sequence

    print(f"✅ Chargé: {len(predictions):,} samples")
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Returns shape: {returns.shape}")
    print()

    return predictions, returns


def run_entry_focused_strategy(
    predictions: np.ndarray,
    returns: np.ndarray,
    config: ExitConfig,
    fees: float = 0.003
) -> TradeResult:
    """
    Run Entry-Focused strategy with hybrid exit.

    Strategy:
    1. ENTRY: Use ML predictions (UP → LONG, DOWN → SHORT)
    2. EXIT: First of (Time-based, TP reached, SL reached)
    3. ANTI-FLIP: Skip entry if last exit same direction (wait for opposite signal)

    Args:
        predictions: ML predictions (0-1 probabilities)
        returns: Period returns
        config: Exit configuration
        fees: Round-trip fees (default 0.3%)

    Returns:
        TradeResult with detailed metrics
    """
    n_samples = len(predictions)
    position = Position.FLAT
    entry_time = -1
    entry_price = 0.0
    current_pnl = 0.0

    # Tracking
    trades = []
    total_pnl = 0.0

    # Anti-flip tracking (directional)
    last_exit_direction = None  # Track last exit direction
    entries_blocked = 0

    # Exit reason counters
    exits_time = 0
    exits_tp = 0
    exits_sl = 0
    exits_flip = 0

    for i in range(n_samples):
        pred_prob = predictions[i]
        pred_dir = 1 if pred_prob > 0.5 else 0  # UP=1, DOWN=0
        current_return = returns[i]

        # Convert prediction to target position
        if pred_dir == 1:
            target = Position.LONG
        else:
            target = Position.SHORT

        # --- ENTRY LOGIC ---
        if position == Position.FLAT:
            # Anti-flip rule check (directional)
            if last_exit_direction == target:
                # SKIP: Wait for opposite signal
                # Exit LONG → Block LONG until SHORT taken
                # Exit SHORT → Block SHORT until LONG taken
                entries_blocked += 1
                continue

            # Enter position based on ML prediction
            position = target
            entry_time = i
            entry_price = 1.0  # Normalized
            current_pnl = 0.0

            # Clear last_exit_direction on successful entry (opposite direction)
            # This allows re-entry same direction after taking opposite
            last_exit_direction = None
            continue

        # --- EXIT LOGIC (Hybrid: Time + TP + SL) ---
        if position != Position.FLAT:
            # Update PnL
            if position == Position.LONG:
                current_pnl += current_return
            else:  # SHORT
                current_pnl -= current_return

            trade_duration = i - entry_time
            should_exit = False
            exit_reason = None

            # Check 1: Time-based exit
            if trade_duration >= config.holding_min:
                should_exit = True
                exit_reason = "TIME"
                exits_time += 1

            # Check 2: Take Profit
            if current_pnl >= config.take_profit:
                should_exit = True
                exit_reason = "TP"
                exits_tp += 1

            # Check 3: Stop Loss
            if current_pnl <= -config.stop_loss:
                should_exit = True
                exit_reason = "SL"
                exits_sl += 1

            # Check 4: Direction flip - DISABLED (Entry-Focused pure)
            # We IGNORE ML during trade, only use Time/TP/SL
            # if target != position:
            #     should_exit = True
            #     exit_reason = "FLIP"
            #     exits_flip += 1

            if should_exit:
                # Record trade
                pnl_with_fees = current_pnl - fees
                trades.append({
                    'direction': position,
                    'duration': trade_duration,
                    'pnl_gross': current_pnl,
                    'pnl_net': pnl_with_fees,
                    'exit_reason': exit_reason
                })
                total_pnl += pnl_with_fees

                # Update anti-flip tracking (directional)
                last_exit_direction = position

                # Reset position
                position = Position.FLAT

    # --- COMPUTE METRICS ---
    if len(trades) == 0:
        return TradeResult(
            config=config,
            n_trades=0,
            n_long=0,
            n_short=0,
            win_rate=0.0,
            pnl_gross=0.0,
            pnl_net=0.0,
            avg_duration=0.0,
            exits_time=0,
            exits_tp=0,
            exits_sl=0,
            exits_flip=0,
            entries_blocked=entries_blocked,
            avg_win=0.0,
            avg_loss=0.0,
            max_win=0.0,
            max_loss=0.0
        )

    # Extract metrics
    trades_array = np.array([t['pnl_net'] for t in trades])
    wins = trades_array > 0
    losses = trades_array <= 0

    win_rate = wins.sum() / len(trades) * 100
    pnl_gross = sum(t['pnl_gross'] for t in trades) * 100  # Percentage
    pnl_net = total_pnl * 100  # Percentage
    avg_duration = np.mean([t['duration'] for t in trades])

    n_long = sum(1 for t in trades if t['direction'] == Position.LONG)
    n_short = sum(1 for t in trades if t['direction'] == Position.SHORT)

    # Return distribution
    wins_array = trades_array[wins]
    losses_array = trades_array[losses]

    avg_win = wins_array.mean() * 100 if len(wins_array) > 0 else 0.0
    avg_loss = losses_array.mean() * 100 if len(losses_array) > 0 else 0.0
    max_win = wins_array.max() * 100 if len(wins_array) > 0 else 0.0
    max_loss = losses_array.min() * 100 if len(losses_array) > 0 else 0.0

    return TradeResult(
        config=config,
        n_trades=len(trades),
        n_long=n_long,
        n_short=n_short,
        win_rate=win_rate,
        pnl_gross=pnl_gross,
        pnl_net=pnl_net,
        avg_duration=avg_duration,
        exits_time=exits_time,
        exits_tp=exits_tp,
        exits_sl=exits_sl,
        exits_flip=exits_flip,
        entries_blocked=entries_blocked,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_win=max_win,
        max_loss=max_loss
    )


def test_multiple_configs(
    predictions: np.ndarray,
    returns: np.ndarray,
    fees: float = 0.003
) -> Dict[str, TradeResult]:
    """
    Test multiple exit configurations.

    Configurations tested:
    - Holding: 10, 15, 20, 30 periods
    - TP: 1%, 1.5%, 2%
    - SL: 0.5%, 1%, 1.5%
    - Anti-flip: Directional (automatic, no cooldown parameter)

    Returns:
        Dict mapping config name to results
    """
    configs = []

    # Grid search (no cooldown needed - directional blocking)
    holdings = [10, 15, 20, 30]
    tps = [0.01, 0.015, 0.02]  # 1%, 1.5%, 2%
    sls = [0.005, 0.01, 0.015]  # 0.5%, 1%, 1.5%

    print("=" * 80)
    print("GÉNÉRATION CONFIGURATIONS")
    print("=" * 80)
    print(f"Holding periods: {holdings}")
    print(f"Take Profits: {[f'{tp*100:.1f}%' for tp in tps]}")
    print(f"Stop Losses: {[f'{sl*100:.1f}%' for sl in sls]}")
    print(f"Anti-flip: Directional (block same direction until opposite taken)")
    print()

    # Create all combinations
    for holding in holdings:
        for tp in tps:
            for sl in sls:
                config = ExitConfig(
                    holding_min=holding,
                    take_profit=tp,
                    stop_loss=sl
                )
                configs.append(config)

    print(f"Total configurations: {len(configs)}")
    print()

    # Test all configs
    results = {}
    for idx, config in enumerate(configs, 1):
        config_name = f"H{config.holding_min}_TP{config.take_profit*100:.1f}_SL{config.stop_loss*100:.1f}"

        if idx % 10 == 0:
            print(f"Testing config {idx}/{len(configs)}: {config_name}")

        result = run_entry_focused_strategy(predictions, returns, config, fees)
        results[config_name] = result

    print(f"✅ Testé {len(results)} configurations")
    print()

    return results


def print_results_table(results: Dict[str, TradeResult]):
    """Print results in formatted table."""
    print("=" * 120)
    print("RÉSULTATS ENTRY-FOCUSED STRATEGY")
    print("=" * 120)

    # Sort by PnL Net descending
    sorted_results = sorted(results.items(), key=lambda x: x[1].pnl_net, reverse=True)

    # Print top 20
    print("\n🏆 TOP 20 CONFIGURATIONS (triées par PnL Net):\n")
    print(f"{'Config':<30} {'Trades':<8} {'WR':<8} {'PnL Net':<10} {'PnL Gross':<10} {'Avg Dur':<8} {'Exits (T/TP/SL/F)':<20} {'Blocked':<8}")
    print("-" * 120)

    for idx, (config_name, result) in enumerate(sorted_results[:20], 1):
        exit_breakdown = f"{result.exits_time}/{result.exits_tp}/{result.exits_sl}/{result.exits_flip}"

        print(f"{config_name:<30} "
              f"{result.n_trades:<8} "
              f"{result.win_rate:>6.2f}% "
              f"{result.pnl_net:>9.2f}% "
              f"{result.pnl_gross:>9.2f}% "
              f"{result.avg_duration:>7.1f}p "
              f"{exit_breakdown:<20} "
              f"{result.entries_blocked:<8}")

    print("\n" + "=" * 120)

    # Best by different criteria
    print("\n📊 MEILLEURS PAR CRITÈRE:\n")

    best_pnl = max(sorted_results, key=lambda x: x[1].pnl_net)
    best_wr = max(sorted_results, key=lambda x: x[1].win_rate)
    min_trades = min(sorted_results, key=lambda x: x[1].n_trades)

    print(f"Meilleur PnL Net: {best_pnl[0]} → {best_pnl[1].pnl_net:+.2f}% ({best_pnl[1].n_trades} trades)")
    print(f"Meilleur Win Rate: {best_wr[0]} → {best_wr[1].win_rate:.2f}% ({best_wr[1].n_trades} trades)")
    print(f"Moins de trades: {min_trades[0]} → {min_trades[1].n_trades} trades (WR {min_trades[1].win_rate:.2f}%)")
    print()

    # Detailed best config
    print("=" * 80)
    print("DÉTAILS MEILLEURE CONFIGURATION")
    print("=" * 80)
    best_name, best_result = best_pnl
    print(f"Config: {best_name}")
    print(f"  Holding Min: {best_result.config.holding_min} périodes")
    print(f"  Take Profit: {best_result.config.take_profit*100:.1f}%")
    print(f"  Stop Loss: {best_result.config.stop_loss*100:.1f}%")
    print(f"  Anti-Flip: Directional (block same direction until opposite)")
    print()
    print(f"Trades: {best_result.n_trades}")
    print(f"  LONG: {best_result.n_long} ({best_result.n_long/best_result.n_trades*100:.1f}%)")
    print(f"  SHORT: {best_result.n_short} ({best_result.n_short/best_result.n_trades*100:.1f}%)")
    print()
    print(f"Performance:")
    print(f"  Win Rate: {best_result.win_rate:.2f}%")
    print(f"  PnL Brut: {best_result.pnl_gross:+.2f}%")
    print(f"  PnL Net: {best_result.pnl_net:+.2f}%")
    print(f"  Avg Duration: {best_result.avg_duration:.1f} périodes")
    print()
    print(f"Exits:")
    print(f"  Time-based: {best_result.exits_time} ({best_result.exits_time/best_result.n_trades*100:.1f}%)")
    print(f"  Take Profit: {best_result.exits_tp} ({best_result.exits_tp/best_result.n_trades*100:.1f}%)")
    print(f"  Stop Loss: {best_result.exits_sl} ({best_result.exits_sl/best_result.n_trades*100:.1f}%)")
    print(f"  Direction Flip: {best_result.exits_flip} ({best_result.exits_flip/best_result.n_trades*100:.1f}%)")
    print()
    print(f"Anti-Flip:")
    print(f"  Entries Blocked: {best_result.entries_blocked}")
    print()
    print(f"Return Distribution:")
    print(f"  Avg Win: {best_result.avg_win:+.2f}%")
    print(f"  Avg Loss: {best_result.avg_loss:+.2f}%")
    print(f"  Max Win: {best_result.max_win:+.2f}%")
    print(f"  Max Loss: {best_result.max_loss:+.2f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="Test Entry-Focused Strategy")
    parser.add_argument('--indicator', type=str, default='macd', choices=['macd', 'rsi', 'cci'],
                        help="Indicateur à tester (défaut: macd)")
    parser.add_argument('--filter-type', type=str, default='kalman', choices=['kalman', 'octave20'],
                        help="Filtre à utiliser (défaut: kalman)")
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help="Split à tester (défaut: test)")
    parser.add_argument('--fees', type=float, default=0.003,
                        help="Frais round-trip (défaut: 0.003 = 0.3%%)")

    args = parser.parse_args()

    print("=" * 80)
    print("TEST ENTRY-FOCUSED STRATEGY")
    print("=" * 80)
    print(f"Indicateur: {args.indicator.upper()}")
    print(f"Filtre: {args.filter_type}")
    print(f"Split: {args.split}")
    print(f"Frais: {args.fees*100:.2f}%")
    print()

    # Load data
    predictions, returns = load_data(args.indicator, args.filter_type, args.split)

    # Test multiple configs
    results = test_multiple_configs(predictions, returns, args.fees)

    # Print results
    print_results_table(results)


if __name__ == '__main__':
    main()
