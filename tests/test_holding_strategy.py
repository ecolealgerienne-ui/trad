#!/usr/bin/env python3
"""
Test Stratégie Holding Minimum - Forcer durée minimale des trades.

VERSION 2.0 - Corrigé avec prix Open réels (pas c_ret!)

HYPOTHÈSE:
Les micro-sorties (direction flips fréquents) détruisent le PnL.
Forcer une durée minimale de trade devrait améliorer le Win Rate.

PRINCIPE:
- Toujours en position (LONG ou SHORT basé sur direction)
- Sur direction flip: flip UNIQUEMENT si trade_duration >= MIN_HOLDING
- Si < MIN_HOLDING: IGNORER le flip, continuer le trade actuel

CORRECTIONS vs v1.0:
- Utilise prix Open réels (OHLCV[:, 2]) au lieu de c_ret
- Backtest PAR ASSET (pas global sur dataset concaténé)
- Calcul PnL: (exit_price - entry_price) / entry_price
- Utilise dataset direction-only (pas dual-binary)

Usage:
    python tests/test_holding_strategy.py --indicator macd --split test --fees 0.001
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import argparse
from dataclasses import dataclass
from typing import List, Dict
from enum import IntEnum
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTES ET TYPES
# =============================================================================

class Position(IntEnum):
    """Position de trading."""
    FLAT = 0
    LONG = 1
    SHORT = -1


MIN_HOLDING_VALUES = [0, 10, 15, 20, 30]  # Périodes à tester


@dataclass
class Trade:
    """Enregistrement d'un trade."""
    entry_idx: int
    exit_idx: int
    duration: int
    position: str  # 'LONG' ou 'SHORT'
    entry_price: float
    exit_price: float
    pnl: float  # PnL brut (%)
    pnl_after_fees: float  # PnL net (%)
    asset_id: int = 0
    exit_reason: str = ""  # "DIRECTION_FLIP", "HOLDING_BLOCK", "END"


@dataclass
class AssetResult:
    """Résultats par asset."""
    asset_id: int
    n_trades: int
    total_pnl: float
    total_pnl_after_fees: float
    win_rate: float
    avg_duration: float


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
    asset_results: List[AssetResult]
    # Métriques sorties
    n_flips_executed: int
    n_flips_blocked: int


# =============================================================================
# CHARGEMENT DONNÉES
# =============================================================================

def load_dataset(indicator: str, split: str = 'test') -> Dict:
    """
    Charge le dataset direction-only avec OHLCV.

    RÉUTILISE: Structure de test_oracle_direction_only.py
    """
    path = Path(f'data/prepared/dataset_btc_eth_bnb_ada_ltc_{indicator}_direction_only_kalman.npz')

    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {path}")

    logger.info(f"Chargement: {path}")
    data = np.load(path, allow_pickle=True)

    # Extraire données du split
    Y = data[f'Y_{split}']
    OHLCV = data[f'OHLCV_{split}']
    Y_pred = data.get(f'Y_{split}_pred', None)

    logger.info(f"  Y shape: {Y.shape} - [timestamp, asset_id, direction]")
    logger.info(f"  OHLCV shape: {OHLCV.shape} - [timestamp, asset_id, O, H, L, C, V]")

    return {
        'Y': Y,
        'OHLCV': OHLCV,
        'Y_pred': Y_pred
    }


# =============================================================================
# BACKTEST PAR ASSET AVEC HOLDING MINIMUM
# =============================================================================

def backtest_single_asset(
    labels: np.ndarray,
    opens: np.ndarray,
    timestamps: np.ndarray,
    asset_id: int,
    fees: float = 0.001,
    min_holding: int = 0
) -> tuple:
    """
    Backtest pour UN SEUL asset avec holding minimum.

    LOGIQUE CAUSALE:
    - Signal à index i → Exécution à Open[i+1]
    - Toujours en position (LONG ou SHORT, jamais FLAT)
    - Direction: 1=UP→LONG, 0=DOWN→SHORT
    - Flip autorisé UNIQUEMENT si trade_duration >= min_holding

    Args:
        labels: (n,) Direction labels
        opens: (n,) Prix Open
        timestamps: (n,) Timestamps
        asset_id: ID de l'asset
        fees: Frais par side
        min_holding: Durée minimale avant flip autorisé

    Returns:
        (trades, n_flips_executed, n_flips_blocked)
    """
    n_samples = len(labels)
    trades = []

    position = Position.FLAT
    entry_idx = 0
    entry_price = 0.0
    entry_timestamp = 0.0

    n_flips_executed = 0
    n_flips_blocked = 0

    for i in range(n_samples - 1):
        direction = int(labels[i])
        target = Position.LONG if direction == 1 else Position.SHORT

        # Première entrée
        if position == Position.FLAT:
            position = target
            entry_idx = i
            entry_price = opens[i + 1]
            entry_timestamp = timestamps[i + 1]
            continue

        # Gestion position existante
        trade_duration = i - entry_idx

        # Direction flip?
        if target != position:
            # Vérifier holding minimum
            if trade_duration >= min_holding:
                # Flip autorisé
                exit_price = opens[i + 1]

                # Calcul PnL (CORRECT: prix réels!)
                if position == Position.LONG:
                    pnl = (exit_price - entry_price) / entry_price
                else:  # SHORT
                    pnl = (entry_price - exit_price) / entry_price

                trade_fees = 2 * fees
                pnl_after_fees = pnl - trade_fees

                trades.append(Trade(
                    entry_idx=entry_idx,
                    exit_idx=i,
                    duration=trade_duration,
                    position='LONG' if position == Position.LONG else 'SHORT',
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl=pnl,
                    pnl_after_fees=pnl_after_fees,
                    asset_id=asset_id,
                    exit_reason="DIRECTION_FLIP"
                ))

                # Flip immédiat
                position = target
                entry_idx = i
                entry_price = opens[i + 1]
                entry_timestamp = timestamps[i + 1]
                n_flips_executed += 1
            else:
                # Flip bloqué - holding minimum pas atteint
                n_flips_blocked += 1
                # Continuer le trade actuel (ignorer le signal)

    # Fermer position finale
    if position != Position.FLAT:
        exit_price = opens[-1]
        trade_duration = n_samples - 1 - entry_idx

        if position == Position.LONG:
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price

        trade_fees = 2 * fees
        pnl_after_fees = pnl - trade_fees

        trades.append(Trade(
            entry_idx=entry_idx,
            exit_idx=n_samples - 1,
            duration=trade_duration,
            position='LONG' if position == Position.LONG else 'SHORT',
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_after_fees=pnl_after_fees,
            asset_id=asset_id,
            exit_reason="END"
        ))

    return trades, n_flips_executed, n_flips_blocked


def backtest_with_holding(
    labels: np.ndarray,
    ohlcv: np.ndarray,
    fees: float = 0.001,
    min_holding: int = 0
) -> StrategyResult:
    """
    Backtest avec holding minimum, PAR ASSET.

    Args:
        labels: (n,) Direction labels
        ohlcv: (n, 7) Prix [timestamp, asset_id, O, H, L, C, V]
        fees: Frais par side
        min_holding: Durée minimale avant flip

    Returns:
        StrategyResult
    """
    # Extraire colonnes OHLCV
    timestamps = ohlcv[:, 0]
    asset_ids = ohlcv[:, 1].astype(int)
    opens = ohlcv[:, 2]

    unique_assets = np.unique(asset_ids)

    all_trades = []
    asset_results = []
    n_long = 0
    n_short = 0
    total_flips_executed = 0
    total_flips_blocked = 0

    # Backtest PAR ASSET
    for asset_id in unique_assets:
        mask = asset_ids == asset_id
        asset_labels = labels[mask]
        asset_opens = opens[mask]
        asset_timestamps = timestamps[mask]

        trades, flips_exec, flips_block = backtest_single_asset(
            asset_labels, asset_opens, asset_timestamps,
            int(asset_id), fees, min_holding
        )

        total_flips_executed += flips_exec
        total_flips_blocked += flips_block

        # Stats par asset
        asset_pnl = 0.0
        asset_pnl_net = 0.0
        asset_wins = 0
        asset_duration = 0

        for t in trades:
            if t.position == 'LONG':
                n_long += 1
            else:
                n_short += 1
            asset_pnl += t.pnl
            asset_pnl_net += t.pnl_after_fees
            if t.pnl_after_fees > 0:
                asset_wins += 1
            asset_duration += t.duration

        if len(trades) > 0:
            asset_results.append(AssetResult(
                asset_id=int(asset_id),
                n_trades=len(trades),
                total_pnl=asset_pnl,
                total_pnl_after_fees=asset_pnl_net,
                win_rate=asset_wins / len(trades),
                avg_duration=asset_duration / len(trades)
            ))

        all_trades.extend(trades)

    # Calculer stats globales
    return compute_stats(
        all_trades, n_long, n_short, min_holding,
        total_flips_executed, total_flips_blocked, asset_results
    )


def compute_stats(
    trades: List[Trade],
    n_long: int,
    n_short: int,
    min_holding: int,
    n_flips_executed: int,
    n_flips_blocked: int,
    asset_results: List[AssetResult]
) -> StrategyResult:
    """Calcule les statistiques du backtest."""
    name = f"Holding {min_holding}p" if min_holding > 0 else "Baseline (0p)"

    if len(trades) == 0:
        return StrategyResult(
            name=name, min_holding=min_holding,
            n_trades=0, n_long=0, n_short=0,
            total_pnl=0.0, total_pnl_after_fees=0.0, total_fees=0.0,
            win_rate=0.0, profit_factor=0.0,
            avg_win=0.0, avg_loss=0.0, avg_duration=0.0,
            sharpe_ratio=0.0, trades=[], asset_results=[],
            n_flips_executed=0, n_flips_blocked=0
        )

    pnls = np.array([t.pnl for t in trades])
    pnls_net = np.array([t.pnl_after_fees for t in trades])
    durations = np.array([t.duration for t in trades])

    total_pnl = pnls.sum()
    total_pnl_net = pnls_net.sum()
    total_fees = total_pnl - total_pnl_net

    # Win rate
    wins = pnls_net > 0
    losses = pnls_net < 0
    win_rate = wins.mean()

    # Profit factor
    sum_wins = pnls_net[wins].sum() if wins.any() else 0.0
    sum_losses = abs(pnls_net[losses].sum()) if losses.any() else 0.0
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else 0.0

    # Moyennes
    avg_win = pnls_net[wins].mean() if wins.any() else 0.0
    avg_loss = pnls_net[losses].mean() if losses.any() else 0.0
    avg_duration = durations.mean()

    # Sharpe (annualisé, 288 périodes/jour)
    if len(pnls_net) > 1 and pnls_net.std() > 0:
        sharpe = (pnls_net.mean() / pnls_net.std()) * np.sqrt(288 * 365)
    else:
        sharpe = 0.0

    return StrategyResult(
        name=name,
        min_holding=min_holding,
        n_trades=len(trades),
        n_long=n_long,
        n_short=n_short,
        total_pnl=total_pnl,
        total_pnl_after_fees=total_pnl_net,
        total_fees=total_fees,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_duration=avg_duration,
        sharpe_ratio=sharpe,
        trades=trades,
        asset_results=asset_results,
        n_flips_executed=n_flips_executed,
        n_flips_blocked=n_flips_blocked
    )


# =============================================================================
# AFFICHAGE RÉSULTATS
# =============================================================================

def print_comparison_table(results: List[StrategyResult]):
    """Affiche tableau comparatif."""
    print(f"\n{'='*120}")
    print("COMPARAISON STRATÉGIES HOLDING MINIMUM")
    print("="*120)

    baseline = results[0]

    header = f"{'Stratégie':<18} | {'Trades':>10} | {'Réduction':>10} | {'Win Rate':>10} | {'PnL Brut':>12} | {'PnL Net':>12} | {'Durée Moy':>10} | {'Flips Bloqués':>14}"
    print(header)
    print("-"*120)

    for res in results:
        if res == baseline:
            reduction = "-"
        else:
            reduction = f"{(1 - res.n_trades/baseline.n_trades)*100:+.1f}%"

        print(f"{res.name:<18} | {res.n_trades:>10,} | {reduction:>10} | {res.win_rate*100:>9.1f}% | {res.total_pnl*100:>+11.2f}% | {res.total_pnl_after_fees*100:>+11.2f}% | {res.avg_duration:>9.1f}p | {res.n_flips_blocked:>14,}")

    print("="*120)


def print_results(result: StrategyResult, indicator: str):
    """Affiche les résultats détaillés."""
    print(f"\n{'='*70}")
    print(f"RÉSULTATS {result.name} - {indicator.upper()}")
    print("="*70)

    print(f"\nTrades:")
    print(f"  Total: {result.n_trades:,}")
    print(f"  Long: {result.n_long:,}")
    print(f"  Short: {result.n_short:,}")
    print(f"  Durée moyenne: {result.avg_duration:.1f} périodes")

    print(f"\nPerformance:")
    print(f"  PnL Brut: {result.total_pnl*100:+.2f}%")
    print(f"  Frais: {result.total_fees*100:.2f}%")
    print(f"  PnL Net: {result.total_pnl_after_fees*100:+.2f}%")

    print(f"\nMétriques:")
    print(f"  Win Rate: {result.win_rate*100:.1f}%")
    print(f"  Profit Factor: {result.profit_factor:.2f}")
    print(f"  Avg Win: {result.avg_win*100:+.3f}%")
    print(f"  Avg Loss: {result.avg_loss*100:+.3f}%")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")

    print(f"\nFlips:")
    print(f"  Exécutés: {result.n_flips_executed:,}")
    print(f"  Bloqués (holding min): {result.n_flips_blocked:,}")


def print_asset_results(asset_results: List[AssetResult]):
    """Affiche les résultats par asset."""
    asset_names = {0: 'BTC', 1: 'ETH', 2: 'BNB', 3: 'ADA', 4: 'LTC'}

    print(f"\n{'='*70}")
    print("RÉSULTATS PAR ASSET")
    print("="*70)

    print(f"\n{'Asset':<8} {'Trades':>10} {'PnL Brut':>12} {'PnL Net':>12} {'Win Rate':>10} {'Durée':>10}")
    print("-"*70)

    for ar in asset_results:
        name = asset_names.get(ar.asset_id, f'Asset{ar.asset_id}')
        print(f"{name:<8} {ar.n_trades:>10,} {ar.total_pnl*100:>+11.2f}% {ar.total_pnl_after_fees*100:>+11.2f}% {ar.win_rate*100:>9.1f}% {ar.avg_duration:>9.1f}p")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test Holding Strategy v2.0')
    parser.add_argument('--indicator', type=str, default='macd',
                        choices=['macd', 'rsi', 'cci'],
                        help='Indicateur à tester')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Split à utiliser')
    parser.add_argument('--fees', type=float, default=0.001,
                        help='Frais par side (défaut: 0.1%%)')
    parser.add_argument('--use-predictions', action='store_true',
                        help='Utiliser prédictions ML au lieu de labels Oracle')

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("TEST HOLDING STRATEGY v2.0")
    logger.info("="*70)
    logger.info(f"Indicateur: {args.indicator.upper()}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Frais: {args.fees*100:.2f}% par side")
    logger.info(f"Holding values: {MIN_HOLDING_VALUES}")

    # Charger données
    data = load_dataset(args.indicator, args.split)

    Y = data['Y']
    OHLCV = data['OHLCV']
    Y_pred = data['Y_pred']

    # Labels: colonne 2 = direction
    labels = Y[:, 2].astype(int)

    logger.info(f"\nDonnées:")
    logger.info(f"  Samples: {len(labels):,}")
    logger.info(f"  Labels UP: {(labels == 1).sum():,} ({(labels == 1).mean()*100:.1f}%)")

    # Mode Oracle ou ML
    if args.use_predictions and Y_pred is not None:
        mode = "ML Predictions"
        pred = Y_pred[:, 0] if Y_pred.ndim > 1 else Y_pred
        labels_to_use = (pred > 0.5).astype(int)
        logger.info(f"  Mode: Prédictions ML")
    else:
        mode = "Oracle"
        labels_to_use = labels
        logger.info(f"  Mode: Oracle (labels parfaits)")

    # Tester chaque holding value
    all_results = []

    for min_holding in MIN_HOLDING_VALUES:
        logger.info(f"\n{'='*60}")
        logger.info(f"Test: Holding {min_holding} périodes")
        logger.info("="*60)

        result = backtest_with_holding(
            labels_to_use, OHLCV,
            fees=args.fees,
            min_holding=min_holding
        )

        all_results.append(result)

        logger.info(f"  Trades: {result.n_trades:,}")
        logger.info(f"  Win Rate: {result.win_rate*100:.1f}%")
        logger.info(f"  PnL Net: {result.total_pnl_after_fees*100:+.2f}%")
        logger.info(f"  Flips bloqués: {result.n_flips_blocked:,}")

    # Afficher résultats
    print_comparison_table(all_results)

    # Afficher détails du baseline
    print_results(all_results[0], args.indicator)
    print_asset_results(all_results[0].asset_results)

    # Afficher meilleur holding
    non_baseline = [r for r in all_results if r.min_holding > 0]
    if non_baseline:
        best = max(non_baseline, key=lambda r: r.total_pnl_after_fees)
        print(f"\n🏆 MEILLEUR HOLDING: {best.name}")
        print_results(best, args.indicator)
        print_asset_results(best.asset_results)

        # Amélioration vs baseline
        baseline = all_results[0]
        print(f"\n📊 AMÉLIORATION vs Baseline:")
        print(f"  Trades: {baseline.n_trades:,} → {best.n_trades:,} ({(1-best.n_trades/baseline.n_trades)*100:+.1f}%)")
        print(f"  Win Rate: {baseline.win_rate*100:.1f}% → {best.win_rate*100:.1f}% ({(best.win_rate-baseline.win_rate)*100:+.1f}%)")
        print(f"  PnL Net: {baseline.total_pnl_after_fees*100:+.2f}% → {best.total_pnl_after_fees*100:+.2f}%")


if __name__ == '__main__':
    main()
