#!/usr/bin/env python3
"""
Trading Utilities - Fonctions Communes Validées (Phases 2.6 & 2.7)

Module partagé contenant toute la logique de trading validée et prouvée:
- PnL accumulation (commit 8ec2610)
- Direction flip immédiat (commit e51a691)
- Holding minimum (Phase 2.6)
- Calcul métriques standardisé

⚠️ RÈGLE D'OR: Ne JAMAIS modifier cette logique sans tests complets!
Toute modification doit être validée sur test_holding_strategy.py d'abord.

Sources validées:
- tests/test_holding_strategy.py (Phase 2.6 référence)
- tests/test_confidence_veto.py (Phase 2.7 corrigé)
"""

import numpy as np
from pathlib import Path
from enum import IntEnum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS ET DATACLASSES
# =============================================================================

class Position(IntEnum):
    """Position de trading (standardisé)."""
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass
class Trade:
    """
    Enregistrement d'un trade.

    Attributes:
        entry: Index d'entrée
        exit: Index de sortie
        duration: Durée en périodes
        position: 'LONG' ou 'SHORT'
        pnl: PnL brut (sans frais)
        pnl_after_fees: PnL net (après frais)
        exit_reason: Raison de sortie ('FORCE_WEAK', 'DIRECTION_FLIP', 'END_OF_DATA', etc.)
    """
    entry: int
    exit: int
    duration: int
    position: str
    pnl: float
    pnl_after_fees: float
    exit_reason: str


@dataclass
class StrategyResult:
    """
    Résultats d'une stratégie de trading.

    Métriques standards pour comparer différentes configurations.
    """
    name: str
    n_trades: int
    n_long: int
    n_short: int

    # PnL
    total_pnl: float
    total_pnl_after_fees: float
    total_fees: float

    # Performance
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_duration: float
    sharpe_ratio: float
    max_drawdown: float

    # Détails
    trades: List[Trade]

    # Métriques optionnelles (selon stratégie)
    extra_metrics: Dict = None


# =============================================================================
# CHARGEMENT DONNÉES
# =============================================================================

def load_dataset(indicator: str, split: str = 'test') -> Dict:
    """
    Charge dataset dual-binary Kalman.

    Args:
        indicator: 'rsi', 'macd', ou 'cci'
        split: 'train', 'val', ou 'test'

    Returns:
        Dict avec:
        - X: Features (n, sequence_length, n_features)
        - Y: Labels (n, 2) [Direction, Force]
        - Y_pred: Prédictions (n, 2) si disponibles

    Raises:
        FileNotFoundError: Si dataset introuvable
    """
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
    """
    Extrait c_ret (Close return) des features.

    ⚠️ IMPORTANT: Utilise la dernière valeur de la séquence (t=-1).

    Args:
        X: Features (n_samples, sequence_length, n_features)
        indicator: 'rsi', 'macd', ou 'cci'

    Returns:
        c_ret pour chaque sample (n_samples,)

    Architecture Pure Signal:
    - RSI/MACD: 1 feature (c_ret) → index 0
    - CCI: 3 features (h_ret, l_ret, c_ret) → c_ret à index 2
    """
    if indicator in ['rsi', 'macd']:
        # 1 feature: c_ret uniquement
        return X[:, -1, 0]
    elif indicator == 'cci':
        # 3 features: h_ret, l_ret, c_ret
        return X[:, -1, 2]
    else:
        raise ValueError(f"Indicateur inconnu: {indicator}")


# =============================================================================
# LOGIQUE CORE BACKTEST (✅ VALIDÉE PHASES 2.6 & 2.7)
# =============================================================================

def compute_pnl_step(
    position: Position,
    return_t: float,
    current_pnl: float
) -> float:
    """
    Calcule le PnL accumulé pour un timestep.

    ✅ VALIDÉ: commit 8ec2610 (test_holding_strategy.py)

    Args:
        position: Position actuelle (FLAT, LONG, SHORT)
        return_t: Return à l'instant t
        current_pnl: PnL accumulé jusqu'à t-1

    Returns:
        PnL accumulé jusqu'à t

    Logique:
    - FLAT: pas de changement
    - LONG: current_pnl += return_t
    - SHORT: current_pnl -= return_t
    """
    if position == Position.FLAT:
        return current_pnl
    elif position == Position.LONG:
        return current_pnl + return_t
    else:  # SHORT
        return current_pnl - return_t


def detect_exit_signal(
    position: Position,
    target: Position,
    force: int,
    trade_duration: int,
    holding_min: int
) -> Tuple[bool, Optional[str]]:
    """
    Détecte si on doit sortir de la position actuelle.

    ✅ VALIDÉ: test_holding_strategy.py (Phase 2.6) + fix commit e51a691

    Args:
        position: Position actuelle
        target: Position cible (selon signal)
        force: Force du signal (0=WEAK, 1=STRONG)
        trade_duration: Durée du trade actuel (périodes)
        holding_min: Durée minimale requise (périodes)

    Returns:
        (exit_signal, exit_reason)
        - exit_signal: True si sortie nécessaire
        - exit_reason: 'FORCE_WEAK', 'DIRECTION_FLIP', ou None

    Priorités (ordre important!):
    1. Direction flip: TOUJOURS sortir (même si < holding_min)
    2. Force=WEAK: Sortir UNIQUEMENT SI holding_min atteint
    """
    if position == Position.FLAT:
        return False, None

    # Cas 1: Direction flip (priorité absolue)
    if target != Position.FLAT and target != position:
        return True, "DIRECTION_FLIP"

    # Cas 2: Force=WEAK ET holding minimum atteint
    if force == 0 and trade_duration >= holding_min:
        return True, "FORCE_WEAK"

    # Cas 3: Force=WEAK MAIS holding minimum PAS atteint
    # → IGNORER signal sortie (holding minimum bloque)
    return False, None


def execute_exit(
    trades: List[Trade],
    entry_time: int,
    exit_time: int,
    position: Position,
    current_pnl: float,
    exit_reason: str,
    fees: float
) -> None:
    """
    Enregistre un trade fermé.

    Args:
        trades: Liste des trades (modifiée in-place)
        entry_time: Timestamp d'entrée
        exit_time: Timestamp de sortie
        position: Position fermée (LONG ou SHORT)
        current_pnl: PnL brut accumulé
        exit_reason: Raison de sortie
        fees: Frais par side (ex: 0.0015 = 0.15%)
    """
    trade_fees = 2 * fees  # Round-trip
    pnl_after_fees = current_pnl - trade_fees

    trades.append(Trade(
        entry=entry_time,
        exit=exit_time,
        duration=exit_time - entry_time,
        position=position.name,  # 'LONG' ou 'SHORT'
        pnl=current_pnl,
        pnl_after_fees=pnl_after_fees,
        exit_reason=exit_reason
    ))


def handle_direction_flip(
    target: Position
) -> Tuple[Position, int, float]:
    """
    Gère le flip immédiat de position (LONG→SHORT ou SHORT→LONG).

    ✅ VALIDÉ: commit e51a691 (bug fix critique)

    ⚠️ CRITIQUE: Ne JAMAIS passer par FLAT!
    - INCORRECT: LONG → FLAT → SHORT (2 trades)
    - CORRECT: LONG → SHORT (1 trade)

    Args:
        target: Nouvelle position cible

    Returns:
        (new_position, reset_time, reset_pnl)
        - new_position: Position après flip (= target)
        - reset_time: Temps à reset (sera = current_time)
        - reset_pnl: PnL à reset (0.0)
    """
    return target, 0, 0.0  # Entry time et current_pnl seront assignés par caller


# =============================================================================
# BACKTEST STANDARD
# =============================================================================

def backtest_strategy(
    predictions: np.ndarray,
    returns: np.ndarray,
    fees: float = 0.0015,
    holding_min: int = 0
) -> StrategyResult:
    """
    Backtest standard avec holding minimum.

    ✅ VALIDÉ: test_holding_strategy.py (Phase 2.6 référence)

    Args:
        predictions: Prédictions (n, 2) [Direction, Force] (probabilités ou binaires)
        returns: Returns (n,)
        fees: Frais par side (défaut: 0.15%)
        holding_min: Durée minimale de trade (0 = baseline)

    Returns:
        StrategyResult avec toutes les métriques

    Decision Matrix:
    - Direction=UP & Force=STRONG → LONG
    - Direction=DOWN & Force=STRONG → SHORT
    - Sinon → FLAT
    """
    # Convertir en binaire si probabilités
    pred_bin = (predictions > 0.5).astype(int) if predictions.dtype == np.float64 else predictions

    n_samples = len(pred_bin)
    trades = []

    # État trading
    position = Position.FLAT
    entry_time = 0
    current_pnl = 0.0

    # Compteurs
    n_long = 0
    n_short = 0

    for i in range(n_samples):
        direction = int(pred_bin[i, 0])  # 0=DOWN, 1=UP
        force = int(pred_bin[i, 1])      # 0=WEAK, 1=STRONG
        ret = returns[i]

        # Accumuler PnL
        current_pnl = compute_pnl_step(position, ret, current_pnl)

        # Decision Matrix
        if direction == 1 and force == 1:
            target = Position.LONG
        elif direction == 0 and force == 1:
            target = Position.SHORT
        else:
            target = Position.FLAT

        # Durée trade actuel
        trade_duration = i - entry_time if position != Position.FLAT else 0

        # Détecter sortie
        exit_signal, exit_reason = detect_exit_signal(
            position, target, force, trade_duration, holding_min
        )

        # Exécuter sortie
        if exit_signal:
            execute_exit(
                trades, entry_time, i, position,
                current_pnl, exit_reason, fees
            )

            # Gérer sortie selon raison
            if exit_reason == "FORCE_WEAK":
                # Sortie complète → FLAT
                position = Position.FLAT
                current_pnl = 0.0

            elif exit_reason == "DIRECTION_FLIP":
                # Flip immédiat SANS passer par FLAT
                position = target
                entry_time = i
                current_pnl = 0.0

                # Compter
                if target == Position.LONG:
                    n_long += 1
                else:
                    n_short += 1

        # Nouvelle entrée si FLAT
        elif position == Position.FLAT and target != Position.FLAT:
            position = target
            entry_time = i
            current_pnl = 0.0

            # Compter
            if target == Position.LONG:
                n_long += 1
            else:
                n_short += 1

    # Fermer position finale
    if position != Position.FLAT:
        execute_exit(
            trades, entry_time, n_samples - 1, position,
            current_pnl, "END_OF_DATA", fees
        )

    # Calculer statistiques
    return compute_trading_stats(
        trades=trades,
        n_long=n_long,
        n_short=n_short,
        strategy_name=f"Holding {holding_min}p" if holding_min > 0 else "Baseline",
        holding_min=holding_min
    )


# =============================================================================
# CALCUL STATISTIQUES
# =============================================================================

def compute_trading_stats(
    trades: List[Trade],
    n_long: int,
    n_short: int,
    strategy_name: str = "Strategy",
    holding_min: int = 0,
    extra_metrics: Dict = None
) -> StrategyResult:
    """
    Calcule statistiques de trading standardisées.

    Args:
        trades: Liste des trades
        n_long: Nombre de positions LONG
        n_short: Nombre de positions SHORT
        strategy_name: Nom de la stratégie
        holding_min: Holding minimum utilisé
        extra_metrics: Métriques additionnelles optionnelles

    Returns:
        StrategyResult complet
    """
    if len(trades) == 0:
        return StrategyResult(
            name=strategy_name,
            n_trades=0, n_long=0, n_short=0,
            total_pnl=0.0, total_pnl_after_fees=0.0, total_fees=0.0,
            win_rate=0.0, profit_factor=0.0,
            avg_win=0.0, avg_loss=0.0, avg_duration=0.0,
            sharpe_ratio=0.0, max_drawdown=0.0,
            trades=[], extra_metrics=extra_metrics
        )

    # Extraire PnLs et durées
    pnls = np.array([t.pnl for t in trades])
    pnls_after_fees = np.array([t.pnl_after_fees for t in trades])
    durations = np.array([t.duration for t in trades])

    # PnL total
    total_pnl = pnls.sum()
    total_pnl_after_fees = pnls_after_fees.sum()
    total_fees = total_pnl - total_pnl_after_fees

    # Win Rate
    wins = pnls_after_fees > 0
    losses = pnls_after_fees < 0
    win_rate = wins.mean() if len(trades) > 0 else 0.0

    # Profit Factor
    sum_wins = pnls_after_fees[wins].sum() if wins.any() else 0.0
    sum_losses = abs(pnls_after_fees[losses].sum()) if losses.any() else 0.0
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else 0.0

    # Moyennes
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

    # Max Drawdown
    cumulative = np.cumsum(pnls_after_fees)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = drawdown.max() if len(drawdown) > 0 else 0.0

    return StrategyResult(
        name=strategy_name,
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
        max_drawdown=max_drawdown,
        trades=trades,
        extra_metrics=extra_metrics
    )


# =============================================================================
# AFFICHAGE
# =============================================================================

def print_comparison_table(
    results: List[StrategyResult],
    baseline_idx: int = 0
):
    """
    Affiche tableau comparatif des stratégies.

    Args:
        results: Liste des résultats à comparer
        baseline_idx: Index de la stratégie baseline (défaut: 0)
    """
    logger.info("\n" + "="*100)
    logger.info("COMPARAISON STRATÉGIES")
    logger.info("="*100)
    logger.info(
        f"{'Stratégie':<25} {'Trades':>10} {'Réduction':>10} "
        f"{'Win Rate':>10} {'PnL Brut':>12} {'PnL Net':>12} {'Sharpe':>8}"
    )
    logger.info("-"*100)

    baseline = results[baseline_idx]

    for r in results:
        # Calcul réduction vs baseline
        if baseline.n_trades > 0:
            delta_trades = r.n_trades - baseline.n_trades
            delta_pct = (delta_trades / baseline.n_trades * 100)
        else:
            delta_trades = r.n_trades
            delta_pct = 0.0

        logger.info(
            f"{r.name:<25} {r.n_trades:>10,} {delta_pct:>+9.1f}% "
            f"{r.win_rate*100:>9.2f}% {r.total_pnl*100:>+11.2f}% "
            f"{r.total_pnl_after_fees*100:>+11.2f}% {r.sharpe_ratio:>8.3f}"
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
        logger.info(f"  Avg Duration: {r.avg_duration:.1f} périodes (~{r.avg_duration*5:.0f} min)")
        logger.info(f"  Sharpe Ratio: {r.sharpe_ratio:.3f}")
        logger.info(f"  Max Drawdown: {r.max_drawdown*100:.2f}%")

        # Métriques extra si présentes
        if r.extra_metrics:
            logger.info("  Métriques additionnelles:")
            for key, value in r.extra_metrics.items():
                if isinstance(value, float):
                    logger.info(f"    {key}: {value:.3f}")
                else:
                    logger.info(f"    {key}: {value}")

    # Meilleure stratégie (Sharpe Ratio)
    best = max(results, key=lambda r: r.sharpe_ratio)

    logger.info(f"\n✅ MEILLEURE STRATÉGIE: {best.name}")
    logger.info(f"   Sharpe Ratio: {best.sharpe_ratio:.3f}")
    logger.info(f"   PnL Net: {best.total_pnl_after_fees*100:+.2f}%")
    logger.info(f"   Win Rate: {best.win_rate*100:.2f}%")
    logger.info(f"   Trades: {best.n_trades:,} ({best.n_trades - baseline.n_trades:+,} vs baseline)")

    if best.total_pnl_after_fees > 0:
        logger.info("\n🎉 STRATÉGIE RENTABLE TROUVÉE!")
    else:
        logger.info("\n⚠️  Aucune stratégie rentable. Approche à revoir.")

    logger.info("\n" + "="*100 + "\n")
