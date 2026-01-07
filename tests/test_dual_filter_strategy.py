#!/usr/bin/env python3
"""
Test de la stratégie de filtrage dual-filter pour éliminer les micro-sorties.

CONTEXTE:
Le modèle ML a ~90% accuracy (très bon!) mais génère beaucoup de micro-sorties
qui détruisent le PnL avec les frais. Le problème = 10% d'erreurs qui causent
des entrées/sorties rapides.

STRATÉGIE DE FILTRAGE:
Utiliser 2 filtres (Octave + Kalman) pour avoir 2 estimations indépendantes
et trader SEULEMENT quand les signaux sont FORTS et COHÉRENTS.

RÈGLES DE FILTRAGE TESTÉES:
1. Baseline: Prédictions d'un seul filtre (attendu: -14,000%)
2. Filtrage Direction: Octave ET Kalman d'accord sur Direction
3. Filtrage Direction+Force: + Au moins 1 filtre dit Force STRONG
4. Filtrage Direction+Force+Confirmation: + Signal stable 2+ périodes

OBJECTIF:
Réduire les trades de 70-80% en éliminant les micro-sorties, tout en gardant
les signaux forts qui ont un bon Win Rate.

Usage:
    # Tester MACD avec filtrage
    python tests/test_dual_filter_strategy.py --indicator macd --split test

    # Tester RSI
    python tests/test_dual_filter_strategy.py --indicator rsi --split test

    # Comparer tous les indicateurs
    for ind in rsi cci macd; do
        python tests/test_dual_filter_strategy.py --indicator $ind --split test
    done
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
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTES
# =============================================================================

FEATURE_INDICES = {
    'rsi': {'c_ret': 0},
    'macd': {'c_ret': 0},
    'cci': {'h_ret': 0, 'l_ret': 1, 'c_ret': 2},
}


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


@dataclass
class StrategyResult:
    """Résultats d'une stratégie."""
    name: str
    n_trades: int
    n_long: int
    n_short: int
    n_filtered: int  # Trades bloqués par le filtrage
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


# =============================================================================
# FONCTIONS DE CHARGEMENT
# =============================================================================

def get_dataset_path(indicator: str, filter_type: str) -> str:
    """Construit le chemin du dataset."""
    filter_suffix = 'octave20' if filter_type == 'octave' else 'kalman'
    return f'data/prepared/dataset_btc_eth_bnb_ada_ltc_{indicator}_dual_binary_{filter_suffix}.npz'


def load_dual_datasets(indicator: str, split: str) -> Tuple[Dict, Dict]:
    """
    Charge les 2 datasets (Octave + Kalman) pour un indicateur.

    Returns:
        data_octave, data_kalman
    """
    path_octave = get_dataset_path(indicator, 'octave')
    path_kalman = get_dataset_path(indicator, 'kalman')

    if not Path(path_octave).exists() or not Path(path_kalman).exists():
        raise FileNotFoundError(
            f"Datasets introuvables:\n"
            f"  Octave: {path_octave}\n"
            f"  Kalman: {path_kalman}"
        )

    logger.info(f"📂 Chargement Octave: {path_octave}")
    data_octave_raw = np.load(path_octave, allow_pickle=True)

    logger.info(f"📂 Chargement Kalman: {path_kalman}")
    data_kalman_raw = np.load(path_kalman, allow_pickle=True)

    data_octave = {
        'X': data_octave_raw[f'X_{split}'],
        'Y': data_octave_raw[f'Y_{split}'],
        'Y_pred': data_octave_raw.get(f'Y_{split}_pred', None),
    }

    data_kalman = {
        'X': data_kalman_raw[f'X_{split}'],
        'Y': data_kalman_raw[f'Y_{split}'],
        'Y_pred': data_kalman_raw.get(f'Y_{split}_pred', None),
    }

    logger.info(f"  ✅ Octave - X: {data_octave['X'].shape}, Y: {data_octave['Y'].shape}")
    if data_octave['Y_pred'] is not None:
        logger.info(f"  ✅ Octave - Y_pred: {data_octave['Y_pred'].shape}")
    else:
        raise ValueError("Prédictions Octave non disponibles! Entraîner le modèle d'abord.")

    logger.info(f"  ✅ Kalman - X: {data_kalman['X'].shape}, Y: {data_kalman['Y'].shape}")
    if data_kalman['Y_pred'] is not None:
        logger.info(f"  ✅ Kalman - Y_pred: {data_kalman['Y_pred'].shape}")
    else:
        raise ValueError("Prédictions Kalman non disponibles! Entraîner le modèle d'abord.")

    # Vérifier que les features sont identiques
    if not np.allclose(data_octave['X'], data_kalman['X'], rtol=1e-10, atol=1e-10):
        logger.warning("⚠️  ATTENTION: Features X différentes entre Octave et Kalman!")

    return data_octave, data_kalman


def extract_c_ret(X: np.ndarray, indicator: str) -> np.ndarray:
    """Extrait c_ret (Close return) de X."""
    if indicator in ['rsi', 'macd']:
        c_ret = X[:, -1, 0]
    elif indicator == 'cci':
        c_ret = X[:, -1, 2]
    else:
        raise ValueError(f"Indicateur inconnu: {indicator}")
    return c_ret


def convert_to_binary(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convertit probabilités en labels binaires."""
    if predictions.max() <= 1.0 and predictions.min() >= 0.0:
        unique_vals = np.unique(predictions)
        if len(unique_vals) > 2:  # Probabilités continues
            direction = (predictions[:, 0] > threshold).astype(int)
            force = (predictions[:, 1] > threshold).astype(int)
            return np.column_stack([direction, force])
    return predictions


# =============================================================================
# STRATÉGIES DE FILTRAGE
# =============================================================================

def strategy_baseline(
    pred_octave: np.ndarray,
    pred_kalman: np.ndarray,
    returns: np.ndarray,
    fees: float
) -> StrategyResult:
    """
    Stratégie 1: BASELINE - Utiliser prédictions d'UN SEUL filtre (Kalman).

    Attendu: ~-14,000% PnL (comme on a vu dans Vigilance #2)
    Problème: Beaucoup de micro-sorties non filtrées
    """
    # Convertir en binaire
    pred_kalman_bin = convert_to_binary(pred_kalman)

    # Decision Matrix simple
    n_samples = len(pred_kalman_bin)
    positions = np.zeros(n_samples, dtype=int)
    trades = []

    position = Position.FLAT
    entry_time = 0
    current_pnl = 0.0

    n_long = 0
    n_short = 0

    for i in range(n_samples):
        direction = int(pred_kalman_bin[i, 0])  # 0=DOWN, 1=UP
        force = int(pred_kalman_bin[i, 1])      # 0=WEAK, 1=STRONG
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

        # Sortie si FLAT
        if target == Position.FLAT and position != Position.FLAT:
            trade_fees = 2 * fees
            pnl_after_fees = current_pnl - trade_fees

            trades.append(Trade(
                start=entry_time,
                end=i,
                duration=i - entry_time,
                position=position.value,
                pnl=current_pnl,
                pnl_after_fees=pnl_after_fees
            ))

            position = Position.FLAT
            current_pnl = 0.0

        # Entrée ou changement
        elif target != Position.FLAT:
            if position == Position.FLAT:
                # Nouvelle entrée
                position = target
                entry_time = i
                current_pnl = 0.0
                if target == Position.LONG:
                    n_long += 1
                else:
                    n_short += 1

            elif position != target:
                # Changement de direction
                trade_fees = 2 * fees
                pnl_after_fees = current_pnl - trade_fees

                trades.append(Trade(
                    start=entry_time,
                    end=i,
                    duration=i - entry_time,
                    position=position.value,
                    pnl=current_pnl,
                    pnl_after_fees=pnl_after_fees
                ))

                # Nouvelle position
                position = target
                entry_time = i
                current_pnl = 0.0
                if target == Position.LONG:
                    n_long += 1
                else:
                    n_short += 1

        # Enregistrer position
        if position == Position.LONG:
            positions[i] = 1
        elif position == Position.SHORT:
            positions[i] = -1

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
            pnl_after_fees=pnl_after_fees
        ))

    return compute_stats(trades, n_long, n_short, 0, "Baseline (Kalman seul)")


def strategy_direction_filter(
    pred_octave: np.ndarray,
    pred_kalman: np.ndarray,
    returns: np.ndarray,
    fees: float
) -> StrategyResult:
    """
    Stratégie 2: FILTRAGE DIRECTION

    Règle: Trader SEULEMENT si Octave ET Kalman d'accord sur Direction
    Objectif: Éliminer les zones d'incertitude (désaccords)
    """
    pred_octave_bin = convert_to_binary(pred_octave)
    pred_kalman_bin = convert_to_binary(pred_kalman)

    n_samples = len(pred_kalman_bin)
    positions = np.zeros(n_samples, dtype=int)
    trades = []

    position = Position.FLAT
    entry_time = 0
    current_pnl = 0.0

    n_long = 0
    n_short = 0
    n_filtered = 0  # Compteur trades bloqués

    for i in range(n_samples):
        dir_octave = int(pred_octave_bin[i, 0])
        dir_kalman = int(pred_kalman_bin[i, 0])
        force_kalman = int(pred_kalman_bin[i, 1])
        ret = returns[i]

        # Accumuler PnL
        if position != Position.FLAT:
            if position == Position.LONG:
                current_pnl += ret
            else:
                current_pnl -= ret

        # FILTRAGE: Les 2 filtres doivent être d'accord sur Direction
        if dir_octave != dir_kalman:
            # Désaccord → NE PAS TRADER (micro-sortie probable)
            target = Position.FLAT
            if position == Position.FLAT:
                n_filtered += 1  # Signal bloqué
        else:
            # Accord → Appliquer Decision Matrix
            if dir_kalman == 1 and force_kalman == 1:
                target = Position.LONG
            elif dir_kalman == 0 and force_kalman == 1:
                target = Position.SHORT
            else:
                target = Position.FLAT

        # Logique trading (identique baseline)
        if target == Position.FLAT and position != Position.FLAT:
            trade_fees = 2 * fees
            pnl_after_fees = current_pnl - trade_fees

            trades.append(Trade(
                start=entry_time,
                end=i,
                duration=i - entry_time,
                position=position.value,
                pnl=current_pnl,
                pnl_after_fees=pnl_after_fees
            ))

            position = Position.FLAT
            current_pnl = 0.0

        elif target != Position.FLAT:
            if position == Position.FLAT:
                position = target
                entry_time = i
                current_pnl = 0.0
                if target == Position.LONG:
                    n_long += 1
                else:
                    n_short += 1

            elif position != target:
                trade_fees = 2 * fees
                pnl_after_fees = current_pnl - trade_fees

                trades.append(Trade(
                    start=entry_time,
                    end=i,
                    duration=i - entry_time,
                    position=position.value,
                    pnl=current_pnl,
                    pnl_after_fees=pnl_after_fees
                ))

                position = target
                entry_time = i
                current_pnl = 0.0
                if target == Position.LONG:
                    n_long += 1
                else:
                    n_short += 1

        if position == Position.LONG:
            positions[i] = 1
        elif position == Position.SHORT:
            positions[i] = -1

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
            pnl_after_fees=pnl_after_fees
        ))

    return compute_stats(trades, n_long, n_short, n_filtered, "Filtrage Direction (Accord Octave+Kalman)")


def strategy_direction_force_filter(
    pred_octave: np.ndarray,
    pred_kalman: np.ndarray,
    returns: np.ndarray,
    fees: float
) -> StrategyResult:
    """
    Stratégie 3: FILTRAGE DIRECTION + FORCE

    Règles:
    1. Octave ET Kalman d'accord sur Direction
    2. AU MOINS 1 filtre dit Force STRONG

    Objectif: Filtrer encore plus les signaux faibles
    """
    pred_octave_bin = convert_to_binary(pred_octave)
    pred_kalman_bin = convert_to_binary(pred_kalman)

    n_samples = len(pred_kalman_bin)
    positions = np.zeros(n_samples, dtype=int)
    trades = []

    position = Position.FLAT
    entry_time = 0
    current_pnl = 0.0

    n_long = 0
    n_short = 0
    n_filtered = 0

    for i in range(n_samples):
        dir_octave = int(pred_octave_bin[i, 0])
        dir_kalman = int(pred_kalman_bin[i, 0])
        force_octave = int(pred_octave_bin[i, 1])
        force_kalman = int(pred_kalman_bin[i, 1])
        ret = returns[i]

        # Accumuler PnL
        if position != Position.FLAT:
            if position == Position.LONG:
                current_pnl += ret
            else:
                current_pnl -= ret

        # FILTRAGE 1: Accord sur Direction
        if dir_octave != dir_kalman:
            target = Position.FLAT
            if position == Position.FLAT:
                n_filtered += 1
        else:
            # FILTRAGE 2: Au moins 1 filtre dit STRONG
            if force_octave == 0 and force_kalman == 0:
                # Les 2 disent WEAK → NE PAS TRADER
                target = Position.FLAT
                if position == Position.FLAT:
                    n_filtered += 1
            else:
                # Au moins 1 dit STRONG → OK trader
                if dir_kalman == 1:
                    target = Position.LONG
                else:
                    target = Position.SHORT

        # Logique trading
        if target == Position.FLAT and position != Position.FLAT:
            trade_fees = 2 * fees
            pnl_after_fees = current_pnl - trade_fees

            trades.append(Trade(
                start=entry_time,
                end=i,
                duration=i - entry_time,
                position=position.value,
                pnl=current_pnl,
                pnl_after_fees=pnl_after_fees
            ))

            position = Position.FLAT
            current_pnl = 0.0

        elif target != Position.FLAT:
            if position == Position.FLAT:
                position = target
                entry_time = i
                current_pnl = 0.0
                if target == Position.LONG:
                    n_long += 1
                else:
                    n_short += 1

            elif position != target:
                trade_fees = 2 * fees
                pnl_after_fees = current_pnl - trade_fees

                trades.append(Trade(
                    start=entry_time,
                    end=i,
                    duration=i - entry_time,
                    position=position.value,
                    pnl=current_pnl,
                    pnl_after_fees=pnl_after_fees
                ))

                position = target
                entry_time = i
                current_pnl = 0.0
                if target == Position.LONG:
                    n_long += 1
                else:
                    n_short += 1

        if position == Position.LONG:
            positions[i] = 1
        elif position == Position.SHORT:
            positions[i] = -1

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
            pnl_after_fees=pnl_after_fees
        ))

    return compute_stats(trades, n_long, n_short, n_filtered, "Filtrage Direction+Force (Accord + 1 STRONG)")


def strategy_full_filter(
    pred_octave: np.ndarray,
    pred_kalman: np.ndarray,
    returns: np.ndarray,
    fees: float,
    min_confirmation: int = 2
) -> StrategyResult:
    """
    Stratégie 4: FILTRAGE COMPLET

    Règles:
    1. Octave ET Kalman d'accord sur Direction
    2. AU MOINS 1 filtre dit Force STRONG
    3. Signal stable pendant min_confirmation périodes

    Objectif: Filtrage maximum pour éliminer micro-sorties
    """
    pred_octave_bin = convert_to_binary(pred_octave)
    pred_kalman_bin = convert_to_binary(pred_kalman)

    n_samples = len(pred_kalman_bin)
    positions = np.zeros(n_samples, dtype=int)
    trades = []

    position = Position.FLAT
    entry_time = 0
    current_pnl = 0.0

    n_long = 0
    n_short = 0
    n_filtered = 0

    # Variables confirmation
    prev_target = Position.FLAT
    confirmation_count = 0

    for i in range(n_samples):
        dir_octave = int(pred_octave_bin[i, 0])
        dir_kalman = int(pred_kalman_bin[i, 0])
        force_octave = int(pred_octave_bin[i, 1])
        force_kalman = int(pred_kalman_bin[i, 1])
        ret = returns[i]

        # Accumuler PnL
        if position != Position.FLAT:
            if position == Position.LONG:
                current_pnl += ret
            else:
                current_pnl -= ret

        # FILTRAGE 1: Accord Direction
        if dir_octave != dir_kalman:
            target = Position.FLAT
        else:
            # FILTRAGE 2: Force
            if force_octave == 0 and force_kalman == 0:
                target = Position.FLAT
            else:
                if dir_kalman == 1:
                    target = Position.LONG
                else:
                    target = Position.SHORT

        # FILTRAGE 3: Confirmation
        if target == prev_target:
            confirmation_count += 1
        else:
            prev_target = target
            confirmation_count = 1

        confirmed = (confirmation_count >= min_confirmation)

        if not confirmed and target != Position.FLAT and position == Position.FLAT:
            n_filtered += 1  # Signal bloqué (pas assez confirmé)

        # Logique trading (avec confirmation)
        if confirmed and target == Position.FLAT and position != Position.FLAT:
            trade_fees = 2 * fees
            pnl_after_fees = current_pnl - trade_fees

            trades.append(Trade(
                start=entry_time,
                end=i,
                duration=i - entry_time,
                position=position.value,
                pnl=current_pnl,
                pnl_after_fees=pnl_after_fees
            ))

            position = Position.FLAT
            current_pnl = 0.0

        elif confirmed and target != Position.FLAT:
            if position == Position.FLAT:
                position = target
                entry_time = i
                current_pnl = 0.0
                if target == Position.LONG:
                    n_long += 1
                else:
                    n_short += 1

            elif position != target:
                trade_fees = 2 * fees
                pnl_after_fees = current_pnl - trade_fees

                trades.append(Trade(
                    start=entry_time,
                    end=i,
                    duration=i - entry_time,
                    position=position.value,
                    pnl=current_pnl,
                    pnl_after_fees=pnl_after_fees
                ))

                position = target
                entry_time = i
                current_pnl = 0.0
                if target == Position.LONG:
                    n_long += 1
                else:
                    n_short += 1

        if position == Position.LONG:
            positions[i] = 1
        elif position == Position.SHORT:
            positions[i] = -1

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
            pnl_after_fees=pnl_after_fees
        ))

    return compute_stats(trades, n_long, n_short, n_filtered, f"Filtrage Complet (Dir+Force+Conf {min_confirmation}p)")


# =============================================================================
# CALCUL STATISTIQUES
# =============================================================================

def compute_stats(
    trades: List[Trade],
    n_long: int,
    n_short: int,
    n_filtered: int,
    name: str
) -> StrategyResult:
    """Calcule statistiques à partir des trades."""

    if len(trades) == 0:
        return StrategyResult(
            name=name,
            n_trades=0,
            n_long=n_long,
            n_short=n_short,
            n_filtered=n_filtered,
            total_pnl=0.0,
            total_pnl_after_fees=0.0,
            total_fees=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_duration=0.0,
            sharpe_ratio=0.0,
            trades=trades
        )

    n_trades = len(trades)
    total_pnl = sum(t.pnl for t in trades)
    total_pnl_after_fees = sum(t.pnl_after_fees for t in trades)
    total_fees = total_pnl - total_pnl_after_fees

    wins = [t for t in trades if t.pnl_after_fees > 0]
    losses = [t for t in trades if t.pnl_after_fees < 0]
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0.0

    gross_profit = sum(t.pnl_after_fees for t in wins)
    gross_loss = abs(sum(t.pnl_after_fees for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    avg_win = gross_profit / len(wins) if len(wins) > 0 else 0.0
    avg_loss = -gross_loss / len(losses) if len(losses) > 0 else 0.0
    avg_duration = sum(t.duration for t in trades) / n_trades

    # Sharpe Ratio
    trade_returns = [t.pnl_after_fees for t in trades]
    returns_mean = np.mean(trade_returns)
    returns_std = np.std(trade_returns)
    sharpe_ratio = (returns_mean / returns_std) * np.sqrt(len(trades)) if returns_std > 0 else 0.0

    return StrategyResult(
        name=name,
        n_trades=n_trades,
        n_long=n_long,
        n_short=n_short,
        n_filtered=n_filtered,
        total_pnl=total_pnl,
        total_pnl_after_fees=total_pnl_after_fees,
        total_fees=total_fees,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_duration=avg_duration,
        sharpe_ratio=sharpe_ratio,
        trades=trades
    )


# =============================================================================
# RAPPORT COMPARATIF
# =============================================================================

def print_comparison_report(results: List[StrategyResult], indicator: str):
    """Imprime rapport comparatif des stratégies."""

    logger.info("\n" + "="*80)
    logger.info(f"RAPPORT STRATÉGIES DE FILTRAGE - {indicator.upper()}")
    logger.info("="*80)

    logger.info("\n📊 TABLEAU COMPARATIF")
    logger.info("-" * 80)

    # Header
    header = f"{'Stratégie':<45} {'Trades':>10} {'Filtrés':>10} {'WR %':>8} {'PnL Net %':>12} {'Sharpe':>8}"
    logger.info(header)
    logger.info("-" * 80)

    # Baseline
    baseline = results[0]

    # Rows
    for result in results:
        reduction_pct = ((baseline.n_trades - result.n_trades) / baseline.n_trades * 100) if baseline.n_trades > 0 else 0

        logger.info(
            f"{result.name:<45} "
            f"{result.n_trades:>10} "
            f"{result.n_filtered:>10} "
            f"{result.win_rate:>8.2f} "
            f"{result.total_pnl_after_fees*100:>+12.2f} "
            f"{result.sharpe_ratio:>8.3f}"
        )

        if result != baseline:
            logger.info(
                f"{'  └─ vs Baseline':<45} "
                f"{result.n_trades - baseline.n_trades:>+10} "
                f"({reduction_pct:>+6.1f}%) "
                f"{result.win_rate - baseline.win_rate:>+8.2f} "
                f"{(result.total_pnl_after_fees - baseline.total_pnl_after_fees)*100:>+12.2f} "
                f"{result.sharpe_ratio - baseline.sharpe_ratio:>+8.3f}"
            )

    logger.info("\n📈 DÉTAILS PAR STRATÉGIE")
    logger.info("-" * 80)

    for result in results:
        logger.info(f"\n🔹 {result.name}")
        logger.info(f"   Trades: {result.n_trades} (LONG: {result.n_long}, SHORT: {result.n_short})")
        logger.info(f"   Signaux filtrés: {result.n_filtered}")
        logger.info(f"   Win Rate: {result.win_rate:.2f}%")
        logger.info(f"   Profit Factor: {result.profit_factor:.3f}")
        logger.info(f"   PnL Brut: {result.total_pnl*100:+.2f}%")
        logger.info(f"   PnL Net: {result.total_pnl_after_fees*100:+.2f}%")
        logger.info(f"   Frais Total: {result.total_fees*100:.2f}%")
        logger.info(f"   Avg Win: {result.avg_win*100:+.3f}%")
        logger.info(f"   Avg Loss: {result.avg_loss*100:+.3f}%")
        logger.info(f"   Avg Duration: {result.avg_duration:.1f} périodes")
        logger.info(f"   Sharpe Ratio: {result.sharpe_ratio:.3f}")

    logger.info("\n" + "="*80)
    logger.info("RECOMMANDATIONS")
    logger.info("="*80)

    # Trouver meilleure stratégie (Sharpe Ratio)
    best = max(results, key=lambda r: r.sharpe_ratio)

    logger.info(f"\n✅ MEILLEURE STRATÉGIE: {best.name}")
    logger.info(f"   Sharpe Ratio: {best.sharpe_ratio:.3f}")
    logger.info(f"   PnL Net: {best.total_pnl_after_fees*100:+.2f}%")
    logger.info(f"   Win Rate: {best.win_rate:.2f}%")
    logger.info(f"   Trades: {best.n_trades} (réduction {((baseline.n_trades - best.n_trades) / baseline.n_trades * 100):+.1f}% vs baseline)")

    if best.total_pnl_after_fees > 0:
        logger.info("\n🎉 STRATÉGIE RENTABLE! Le filtrage fonctionne.")
    else:
        logger.info("\n⚠️  Toujours négatif, mais amélioration vs baseline.")

    logger.info("\n" + "="*80 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Test stratégies de filtrage dual-filter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--indicator', type=str, required=True, choices=['rsi', 'macd', 'cci'],
                        help='Indicateur à tester')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Split à utiliser (défaut: test)')
    parser.add_argument('--fees', type=float, default=0.0015,
                        help='Frais par trade side (défaut: 0.0015 = 0.15%%)')
    parser.add_argument('--min-confirmation', type=int, default=2,
                        help='Périodes de confirmation (défaut: 2)')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info(f"TEST STRATÉGIES DE FILTRAGE - {args.indicator.upper()}")
    logger.info("="*80)
    logger.info(f"Indicateur: {args.indicator}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Fees: {args.fees*100:.2f}% par side ({args.fees*2*100:.2f}% round-trip)")
    logger.info(f"Confirmation min: {args.min_confirmation} périodes")
    logger.info("="*80 + "\n")

    # Charger les 2 datasets
    data_octave, data_kalman = load_dual_datasets(args.indicator, args.split)

    # Extraire returns
    returns = extract_c_ret(data_kalman['X'], args.indicator)

    logger.info("\n🔧 EXÉCUTION DES 4 STRATÉGIES...\n")

    # Stratégie 1: Baseline
    logger.info("1️⃣  Baseline (Kalman seul)...")
    result_baseline = strategy_baseline(
        data_octave['Y_pred'],
        data_kalman['Y_pred'],
        returns,
        args.fees
    )

    # Stratégie 2: Filtrage Direction
    logger.info("2️⃣  Filtrage Direction (Accord Octave+Kalman)...")
    result_direction = strategy_direction_filter(
        data_octave['Y_pred'],
        data_kalman['Y_pred'],
        returns,
        args.fees
    )

    # Stratégie 3: Filtrage Direction + Force
    logger.info("3️⃣  Filtrage Direction+Force (Accord + 1 STRONG)...")
    result_dir_force = strategy_direction_force_filter(
        data_octave['Y_pred'],
        data_kalman['Y_pred'],
        returns,
        args.fees
    )

    # Stratégie 4: Filtrage Complet
    logger.info(f"4️⃣  Filtrage Complet (Dir+Force+Conf {args.min_confirmation}p)...")
    result_full = strategy_full_filter(
        data_octave['Y_pred'],
        data_kalman['Y_pred'],
        returns,
        args.fees,
        args.min_confirmation
    )

    # Rapport
    results = [result_baseline, result_direction, result_dir_force, result_full]
    print_comparison_report(results, args.indicator)


if __name__ == '__main__':
    main()
