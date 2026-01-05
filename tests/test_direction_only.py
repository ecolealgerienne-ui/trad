#!/usr/bin/env python3
"""
Test stratégie Direction SEULE (ignorer Force).

But: Vérifier si Direction est bien prédite en contexte trading,
ou si le problème vient de la combinaison Direction+Force.
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Ajouter src/ au path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from test_dual_binary_trading import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Position(Enum):
    """Positions possibles."""
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Trade:
    """Représente un trade fermé."""
    start: int
    end: int = 0
    duration: int = 0
    position: str = ""
    pnl: float = 0.0
    pnl_after_fees: float = 0.0
    direction_at_entry: int = 0


@dataclass
class Context:
    """État du contexte de trading."""
    position: Position = Position.FLAT
    entry_time: int = 0
    trades: List[Trade] = field(default_factory=list)
    current_pnl: float = 0.0
    direction_at_entry: int = 0


def run_direction_only_strategy(
    Y: np.ndarray,
    returns: np.ndarray,
    fees: float = 0.0,
    use_predictions: bool = False,
    Y_pred: np.ndarray = None
) -> Tuple[np.ndarray, Dict]:
    """
    Stratégie Direction SEULE (ignorer Force).

    Args:
        Y: Labels (n_samples, 2) → [direction, force]
        returns: Rendements c_ret (n_samples,)
        fees: Frais par trade
        use_predictions: Si True, utiliser Y_pred
        Y_pred: Prédictions (si use_predictions=True)

    Returns:
        positions: Array des positions
        stats: Statistiques
    """
    if use_predictions:
        if Y_pred is None:
            raise ValueError("use_predictions=True mais Y_pred est None")
        signals = Y_pred
        logger.info("🎯 Mode: Prédictions modèle")
    else:
        signals = Y
        logger.info("🎯 Mode: Labels Oracle")

    # Convertir probabilités → binaire
    if signals.max() > 1.0 or signals.min() < 0.0:
        pass  # Déjà binaire
    else:
        unique_vals = np.unique(signals)
        if len(unique_vals) > 2:
            direction = (signals[:, 0] > 0.5).astype(int)
            signals = np.column_stack([direction, signals[:, 1]])  # Garder Force mais ne pas l'utiliser
            logger.info("   📊 Prédictions converties (Direction seuil=0.5)")

    # Extraire Direction uniquement
    direction_signals = signals[:, 0]
    dir_up = (direction_signals == 1).sum()
    logger.info(f"   📊 Distribution Direction: UP={dir_up/len(direction_signals)*100:.1f}%")

    n_samples = len(signals)
    positions = np.zeros(n_samples, dtype=int)
    ctx = Context()

    # Stats
    stats = {
        'n_trades': 0,
        'n_long': 0,
        'n_short': 0,
        'total_pnl': 0.0,
        'total_pnl_after_fees': 0.0,
        'total_fees': 0.0,
    }

    # Loop
    for i in range(n_samples):
        direction = int(direction_signals[i])

        # Calculer P&L si en position
        if ctx.position == Position.LONG:
            ctx.current_pnl += returns[i] * 100  # Rendement en %
        elif ctx.position == Position.SHORT:
            ctx.current_pnl += -returns[i] * 100

        # Decision Matrix SIMPLIFIÉE (Direction seule)
        if direction == 1:
            # UP → LONG
            target_position = Position.LONG
        else:
            # DOWN (0) → SHORT
            target_position = Position.SHORT

        # Logique de trading
        if ctx.position == Position.FLAT:
            # Entrer en position
            ctx.position = target_position
            ctx.entry_time = i
            ctx.current_pnl = 0.0
            ctx.direction_at_entry = direction

            if target_position == Position.LONG:
                stats['n_long'] += 1
            else:
                stats['n_short'] += 1

        elif ctx.position != target_position:
            # Changement de direction = sortie + nouvelle entrée
            trade_fees = 2 * fees  # Sortie ancienne position
            pnl_after_fees = ctx.current_pnl - trade_fees

            trade = Trade(
                start=ctx.entry_time,
                end=i,
                duration=i - ctx.entry_time,
                position=ctx.position.value,
                pnl=ctx.current_pnl,
                pnl_after_fees=pnl_after_fees,
                direction_at_entry=ctx.direction_at_entry
            )
            ctx.trades.append(trade)

            stats['n_trades'] += 1
            stats['total_pnl'] += ctx.current_pnl
            stats['total_pnl_after_fees'] += pnl_after_fees
            stats['total_fees'] += trade_fees

            # Nouvelle entrée
            ctx.position = target_position
            ctx.entry_time = i
            ctx.current_pnl = 0.0
            ctx.direction_at_entry = direction

            if target_position == Position.LONG:
                stats['n_long'] += 1
            else:
                stats['n_short'] += 1

            trade_fees_entry = 0  # Pas de frais d'entrée car on est déjà sorti
            stats['total_fees'] += trade_fees_entry

        # Enregistrer position courante
        if ctx.position == Position.LONG:
            positions[i] = 1
        elif ctx.position == Position.SHORT:
            positions[i] = -1

    # Fermer position finale
    if ctx.position != Position.FLAT:
        trade_fees = 2 * fees
        pnl_after_fees = ctx.current_pnl - trade_fees

        trade = Trade(
            start=ctx.entry_time,
            end=n_samples - 1,
            duration=n_samples - 1 - ctx.entry_time,
            position=ctx.position.value,
            pnl=ctx.current_pnl,
            pnl_after_fees=pnl_after_fees,
            direction_at_entry=ctx.direction_at_entry
        )
        ctx.trades.append(trade)

        stats['n_trades'] += 1
        stats['total_pnl'] += ctx.current_pnl
        stats['total_pnl_after_fees'] += pnl_after_fees
        stats['total_fees'] += trade_fees

    # Calculer métriques
    if stats['n_trades'] > 0:
        wins = [t for t in ctx.trades if t.pnl_after_fees > 0]
        losses = [t for t in ctx.trades if t.pnl_after_fees <= 0]

        stats['n_wins'] = len(wins)
        stats['n_losses'] = len(losses)
        stats['win_rate'] = len(wins) / stats['n_trades']

        if len(wins) > 0:
            stats['avg_win'] = np.mean([t.pnl for t in wins])
        else:
            stats['avg_win'] = 0.0

        if len(losses) > 0:
            stats['avg_loss'] = np.mean([t.pnl for t in losses])
        else:
            stats['avg_loss'] = 0.0

        sum_wins = sum(t.pnl for t in wins)
        sum_losses = abs(sum(t.pnl for t in losses))
        stats['profit_factor'] = sum_wins / sum_losses if sum_losses > 0 else 0.0

        stats['avg_duration'] = np.mean([t.duration for t in ctx.trades])
    else:
        stats['n_wins'] = 0
        stats['n_losses'] = 0
        stats['win_rate'] = 0.0
        stats['avg_win'] = 0.0
        stats['avg_loss'] = 0.0
        stats['profit_factor'] = 0.0
        stats['avg_duration'] = 0.0

    return positions, stats


def print_results(stats: Dict, indicator: str, split: str, use_predictions: bool):
    """Affiche résultats."""
    mode = "Prédictions" if use_predictions else "Oracle"

    logger.info("\n" + "="*70)
    logger.info(f"📊 RÉSULTATS - {indicator.upper()} {mode} (Direction SEULE)")
    logger.info("="*70)

    logger.info("\n📈 Trades:")
    logger.info(f"   Total Trades:     {stats['n_trades']:,}")
    logger.info(f"   LONG:             {stats['n_long']:,}")
    logger.info(f"   SHORT:            {stats['n_short']:,}")
    logger.info(f"   Avg Duration:     {stats['avg_duration']:.1f} périodes")

    logger.info("\n💰 Performance:")
    logger.info(f"   Win Rate:         {stats['win_rate']*100:.2f}%")
    logger.info(f"   Profit Factor:    {stats['profit_factor']:.3f}")
    logger.info(f"   Avg Win:          +{stats['avg_win']:.3f}%")
    logger.info(f"   Avg Loss:         {stats['avg_loss']:.3f}%")

    logger.info("\n💵 PnL:")
    logger.info(f"   PnL Brut:         {stats['total_pnl']:+.2f}%")
    logger.info(f"   Frais Totaux:     {stats['total_fees']:.2f}%")
    logger.info(f"   PnL Net:          {stats['total_pnl_after_fees']:+.2f}%")

    logger.info("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Test stratégie Direction SEULE (sans Force)"
    )
    parser.add_argument(
        '--indicator',
        required=True,
        choices=['rsi', 'macd', 'cci'],
        help="Indicateur"
    )
    parser.add_argument(
        '--split',
        default='test',
        choices=['train', 'val', 'test'],
        help="Split (défaut: test)"
    )
    parser.add_argument(
        '--fees',
        type=float,
        default=0.1,
        help="Frais par trade en %% (défaut: 0.1%%)"
    )
    parser.add_argument(
        '--use-predictions',
        action='store_true',
        help="Utiliser prédictions modèle"
    )

    args = parser.parse_args()

    # Convertir fees
    fees_decimal = args.fees / 100.0

    # Charger données
    data = load_dataset(args.indicator, args.split)

    # Extraire returns
    X = data['X']
    if args.indicator in ['rsi', 'macd']:
        returns = X[:, -1, 0]  # c_ret à index 0
    else:  # cci
        returns = X[:, -1, 2]  # c_ret à index 2

    logger.info(f"\n🚀 Test Direction SEULE: {args.indicator.upper()} ({args.split})")
    logger.info(f"   Samples: {len(data['Y']):,}")
    logger.info(f"   Frais: {args.fees}% par trade")

    # Run backtest
    positions, stats = run_direction_only_strategy(
        Y=data['Y'],
        returns=returns,
        fees=fees_decimal,
        use_predictions=args.use_predictions,
        Y_pred=data['Y_pred']
    )

    # Afficher résultats
    print_results(stats, args.indicator, args.split, args.use_predictions)

    logger.info("\n✅ Test terminé")


if __name__ == '__main__':
    main()
