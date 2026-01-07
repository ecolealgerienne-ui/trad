#!/usr/bin/env python3
"""
Test Stratégie Holding Minimum - Forcer durée minimale des trades.

HYPOTHÈSE:
Les erreurs MACD (Accuracy 92%, Win Rate 14%) viennent de SORTIES TROP PRÉCOCES.
Forcer une durée minimale de trade pourrait améliorer Win Rate en capturant le mouvement réel.

PRINCIPE:
- Entrée: MACD Direction=UP & Force=STRONG
- Sortie NORMALE: Force=WEAK
- Sortie FORCÉE: Uniquement si trade_duration >= MIN_HOLDING

TESTS:
- Baseline: Pas de holding minimum (sortie immédiate si Force=WEAK)
- MIN_HOLDING = 10 périodes (~50 min)
- MIN_HOLDING = 15 périodes (~75 min)
- MIN_HOLDING = 20 périodes (~100 min)
- MIN_HOLDING = 30 périodes (~150 min)

Usage:
    python tests/test_holding_strategy.py --indicator macd --split test
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import argparse
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTES
# =============================================================================

MIN_HOLDING_VALUES = [0, 10, 15, 20, 30]  # Périodes à tester


# =============================================================================
# DATACLASSES
# =============================================================================

class Position(Enum):
    """Positions possibles."""
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Trade:
    """Enregistrement d'un trade."""
    start: int
    end: int
    duration: int
    position: str
    pnl: float
    pnl_after_fees: float
    exit_reason: str  # "FORCE_WEAK", "HOLDING_MIN", "DIRECTION_FLIP"


@dataclass
class StrategyResult:
    """Résultats d'une stratégie."""
    name: str
    min_holding: int
    n_trades: int
    n_long: int
    n_short: int
    total_pnl: float
    total_pnl_after_fees: float
    total_fees: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_duration: float
    sharpe_ratio: float
    trades: List[Trade]
    # Nouvelles métriques
    n_exits_force_weak: int
    n_exits_holding_min: int
    n_exits_direction_flip: int


# =============================================================================
# CHARGEMENT DONNÉES
# =============================================================================

def load_dataset(indicator: str, split: str = 'test') -> Dict:
    """Charge dataset Kalman."""
    path = f'data/prepared/dataset_btc_eth_bnb_ada_ltc_{indicator}_dual_binary_kalman.npz'

    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset introuvable: {path}")

    logger.info(f"📂 Chargement: {path}")
    data = np.load(path, allow_pickle=True)

    return {
        'X': data[f'X_{split}'],
        'Y': data[f'Y_{split}'],
        'Y_pred': data.get(f'Y_{split}_pred', None),
    }


def extract_c_ret(X: np.ndarray, indicator: str) -> np.ndarray:
    """Extrait c_ret des features."""
    if indicator in ['rsi', 'macd']:
        c_ret = X[:, :, 0]
        return c_ret[:, -1]
    elif indicator == 'cci':
        c_ret = X[:, :, 2]
        return c_ret[:, -1]
    else:
        raise ValueError(f"Indicateur inconnu: {indicator}")


# =============================================================================
# BACKTEST AVEC HOLDING MINIMUM
# =============================================================================

def backtest_holding_strategy(
    pred: np.ndarray,
    returns: np.ndarray,
    fees: float,
    min_holding: int = 0
) -> StrategyResult:
    """
    Backtest avec holding minimum.

    Args:
        pred: Prédictions (n_samples, 2) - [Direction, Force]
        returns: Returns (n_samples,)
        fees: Frais par side
        min_holding: Durée minimale du trade (0 = baseline)

    Returns:
        StrategyResult
    """
    # Convertir en binaire
    pred_bin = (pred > 0.5).astype(int)

    n_samples = len(pred_bin)
    trades = []

    position = Position.FLAT
    entry_time = 0
    current_pnl = 0.0

    n_long = 0
    n_short = 0
    n_exits_force_weak = 0
    n_exits_holding_min = 0
    n_exits_direction_flip = 0

    for i in range(n_samples):
        direction = int(pred_bin[i, 0])  # 0=DOWN, 1=UP
        force = int(pred_bin[i, 1])      # 0=WEAK, 1=STRONG
        ret = returns[i]

        # Accumuler PnL
        if position != Position.FLAT:
            if position == Position.LONG:
                current_pnl += ret
            else:  # SHORT
                current_pnl -= ret

        # Decision Matrix
        if direction == 1 and force == 1:
            target = Position.LONG
        elif direction == 0 and force == 1:
            target = Position.SHORT
        else:
            target = Position.FLAT

        # Durée trade actuel
        trade_duration = i - entry_time if position != Position.FLAT else 0

        # LOGIQUE SORTIE avec HOLDING MINIMUM
        exit_signal = False
        exit_reason = None

        if position != Position.FLAT:
            # Cas 1: Force=WEAK et holding minimum atteint
            if target == Position.FLAT and trade_duration >= min_holding:
                exit_signal = True
                exit_reason = "FORCE_WEAK"
                n_exits_force_weak += 1

            # Cas 2: Direction flip (toujours prioritaire, même si < min_holding)
            elif target != Position.FLAT and target != position:
                exit_signal = True
                exit_reason = "DIRECTION_FLIP"
                n_exits_direction_flip += 1

            # Cas 3: Force=WEAK mais holding minimum PAS atteint
            elif target == Position.FLAT and trade_duration < min_holding:
                # IGNORER signal sortie, continuer trade
                exit_signal = False
                exit_reason = "HOLDING_MIN_BLOCK"
                n_exits_holding_min += 1

        # Exécuter sortie si nécessaire
        if exit_signal:
            trade_fees = 2 * fees
            pnl_after_fees = current_pnl - trade_fees

            trades.append(Trade(
                start=entry_time,
                end=i,
                duration=i - entry_time,
                position=position.value,
                pnl=current_pnl,
                pnl_after_fees=pnl_after_fees,
                exit_reason=exit_reason
            ))

            # Reset si sortie complète (FORCE_WEAK)
            if exit_reason == "FORCE_WEAK":
                position = Position.FLAT
                current_pnl = 0.0

            # Flip immédiat si DIRECTION_FLIP
            elif exit_reason == "DIRECTION_FLIP":
                position = target
                entry_time = i
                current_pnl = 0.0
                if target == Position.LONG:
                    n_long += 1
                else:
                    n_short += 1

        # Nouvelle entrée si FLAT
        elif position == Position.FLAT and target != Position.FLAT:
            position = target
            entry_time = i
            current_pnl = 0.0
            if target == Position.LONG:
                n_long += 1
            else:
                n_short += 1

    # Fermer position finale
    if position != Position.FLAT:
        trade_fees = 2 * fees
        pnl_after_fees = current_pnl - trade_fees

        trades.append(Trade(
            start=entry_time,
            end=n_samples - 1,
            duration=n_samples - 1 - entry_time,
            position=position.value,
            pnl=current_pnl,
            pnl_after_fees=pnl_after_fees,
            exit_reason="END_OF_DATA"
        ))

    return compute_stats(
        trades, n_long, n_short, min_holding,
        n_exits_force_weak, n_exits_holding_min, n_exits_direction_flip
    )


def compute_stats(
    trades: List[Trade],
    n_long: int,
    n_short: int,
    min_holding: int,
    n_exits_force_weak: int,
    n_exits_holding_min: int,
    n_exits_direction_flip: int
) -> StrategyResult:
    """Calcule statistiques."""
    if len(trades) == 0:
        return StrategyResult(
            name=f"Holding {min_holding}p",
            min_holding=min_holding,
            n_trades=0, n_long=0, n_short=0,
            total_pnl=0.0, total_pnl_after_fees=0.0, total_fees=0.0,
            win_rate=0.0, profit_factor=0.0,
            avg_win=0.0, avg_loss=0.0, avg_duration=0.0,
            sharpe_ratio=0.0, trades=[],
            n_exits_force_weak=0, n_exits_holding_min=0, n_exits_direction_flip=0
        )

    pnls = np.array([t.pnl for t in trades])
    pnls_after_fees = np.array([t.pnl_after_fees for t in trades])
    durations = np.array([t.duration for t in trades])

    total_pnl = pnls.sum()
    total_pnl_after_fees = pnls_after_fees.sum()
    total_fees = total_pnl - total_pnl_after_fees

    wins = pnls_after_fees > 0
    losses = pnls_after_fees < 0

    win_rate = wins.mean() if len(trades) > 0 else 0.0

    sum_wins = pnls_after_fees[wins].sum() if wins.any() else 0.0
    sum_losses = abs(pnls_after_fees[losses].sum()) if losses.any() else 0.0
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else 0.0

    avg_win = pnls_after_fees[wins].mean() if wins.any() else 0.0
    avg_loss = pnls_after_fees[losses].mean() if losses.any() else 0.0

    avg_duration = durations.mean()

    # Sharpe Ratio (annualisé, 5min = 288 périodes/jour)
    if len(pnls_after_fees) > 1:
        returns_mean = pnls_after_fees.mean()
        returns_std = pnls_after_fees.std()
        if returns_std > 0:
            sharpe = (returns_mean / returns_std) * np.sqrt(288 * 365)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    return StrategyResult(
        name=f"Holding {min_holding}p" if min_holding > 0 else "Baseline (0p)",
        min_holding=min_holding,
        n_trades=len(trades),
        n_long=n_long,
        n_short=n_short,
        total_pnl=total_pnl,
        total_pnl_after_fees=total_pnl_after_fees,
        total_fees=total_fees,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_duration=avg_duration,
        sharpe_ratio=sharpe,
        trades=trades,
        n_exits_force_weak=n_exits_force_weak,
        n_exits_holding_min=n_exits_holding_min,
        n_exits_direction_flip=n_exits_direction_flip
    )


# =============================================================================
# AFFICHAGE
# =============================================================================

def print_results(results: List[StrategyResult]):
    """Affiche résultats comparatifs."""
    logger.info("\n" + "="*100)
    logger.info("COMPARAISON STRATÉGIES HOLDING MINIMUM")
    logger.info("="*100)
    logger.info(f"{'Stratégie':<20} {'Trades':>8} {'Win Rate':>9} {'PnL Net':>10} {'Sharpe':>8} {'Avg Dur':>9} {'Exits FW':>10} {'Exits HM':>10}")
    logger.info("-"*100)

    baseline = results[0]

    for r in results:
        delta_trades = r.n_trades - baseline.n_trades
        delta_pct = (delta_trades / baseline.n_trades * 100) if baseline.n_trades > 0 else 0

        logger.info(
            f"{r.name:<20} {r.n_trades:>8,} {r.win_rate*100:>8.2f}% {r.total_pnl_after_fees*100:>9.2f}% "
            f"{r.sharpe_ratio:>8.3f} {r.avg_duration:>8.1f}p {r.n_exits_force_weak:>10,} {r.n_exits_holding_min:>10,}"
        )

        if r != baseline:
            logger.info(
                f"{'  └─ vs Baseline':<20} {delta_trades:>+8,} ({delta_pct:>+6.1f}%)"
            )

    logger.info("\n📊 DÉTAILS PAR STRATÉGIE:")

    for r in results:
        logger.info(f"\n{r.name}")
        logger.info(f"  Trades: {r.n_trades:,} (LONG: {r.n_long:,}, SHORT: {r.n_short:,})")
        logger.info(f"  Win Rate: {r.win_rate*100:.2f}%")
        logger.info(f"  Profit Factor: {r.profit_factor:.3f}")
        logger.info(f"  PnL Brut: {r.total_pnl*100:+.2f}%")
        logger.info(f"  PnL Net: {r.total_pnl_after_fees*100:+.2f}%")
        logger.info(f"  Frais Total: {r.total_fees*100:.2f}%")
        logger.info(f"  Avg Win: {r.avg_win*100:+.3f}%")
        logger.info(f"  Avg Loss: {r.avg_loss*100:+.3f}%")
        logger.info(f"  Avg Duration: {r.avg_duration:.1f} périodes")
        logger.info(f"  Sharpe Ratio: {r.sharpe_ratio:.3f}")
        logger.info(f"  Exits:")
        logger.info(f"    Force=WEAK: {r.n_exits_force_weak:,}")
        logger.info(f"    Holding Min (bloqués): {r.n_exits_holding_min:,}")
        logger.info(f"    Direction Flip: {r.n_exits_direction_flip:,}")

    # Meilleure stratégie
    best = max(results, key=lambda r: r.sharpe_ratio)

    logger.info(f"\n✅ MEILLEURE STRATÉGIE: {best.name}")
    logger.info(f"   Sharpe Ratio: {best.sharpe_ratio:.3f}")
    logger.info(f"   PnL Net: {best.total_pnl_after_fees*100:+.2f}%")
    logger.info(f"   Win Rate: {best.win_rate*100:.2f}%")
    logger.info(f"   Avg Duration: {best.avg_duration:.1f} périodes")

    if best.total_pnl_after_fees > 0:
        logger.info("\n🎉 STRATÉGIE RENTABLE! Le holding minimum fonctionne.")
    else:
        logger.info("\n⚠️  Toujours négatif. Le holding minimum n'a pas résolu le problème.")

    logger.info("\n" + "="*100 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Test stratégie holding minimum',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--indicator', type=str, default='macd',
                        help='Indicateur (défaut: macd)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Split (défaut: test)')
    parser.add_argument('--fees', type=float, default=0.0015,
                        help='Frais par side (défaut: 0.0015)')

    args = parser.parse_args()

    logger.info("="*100)
    logger.info(f"TEST HOLDING MINIMUM - {args.indicator.upper()}")
    logger.info("="*100)
    logger.info(f"Indicateur: {args.indicator}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Fees: {args.fees*100:.2f}% par side ({args.fees*2*100:.2f}% round-trip)")
    logger.info(f"Holding values testés: {MIN_HOLDING_VALUES}")
    logger.info("="*100 + "\n")

    # Charger données
    data = load_dataset(args.indicator, args.split)
    returns = extract_c_ret(data['X'], args.indicator)

    logger.info(f"  Samples: {len(data['Y']):,}")
    logger.info(f"  Prédictions disponibles: {'✅' if data['Y_pred'] is not None else '❌'}\n")

    if data['Y_pred'] is None:
        logger.error("❌ Pas de prédictions disponibles!")
        return

    # Tester chaque holding value
    results = []

    for min_holding in MIN_HOLDING_VALUES:
        logger.info(f"🔧 Test Holding {min_holding} périodes...")
        result = backtest_holding_strategy(
            data['Y_pred'],
            returns,
            args.fees,
            min_holding
        )
        results.append(result)

    # Afficher résultats
    print_results(results)


if __name__ == '__main__':
    main()
