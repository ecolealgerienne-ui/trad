#!/usr/bin/env python3
"""
Comparaison complète PnL Octave vs Kalman (Vigilance #2 - Expert 2).

Objectifs:
1. Comparer PnL, Win Rate, Profit Factor entre Octave et Kalman
2. Calculer Sharpe Ratio et Sortino Ratio
3. Analyser distribution des gains (histograms, fat tails)
4. Calculer MAE/MFE (Maximum Adverse/Favorable Excursion)
5. Analyser zones de désaccord (isolés vs blocs structurels)
6. Générer rapport complet pour validation Vigilance #2

Vigilance #2 (Expert 2):
> "Tester en PnL, pas seulement en WR. Certaines zones évitées peuvent être
> peu fréquentes mais très rentables."

Usage:
    # Comparer RSI Octave vs Kalman
    python tests/compare_dual_filter_pnl.py --indicator rsi --split test

    # Comparer MACD avec prédictions modèle
    python tests/compare_dual_filter_pnl.py --indicator macd --split test --use-predictions

    # Comparer tous les indicateurs
    for ind in rsi cci macd; do
        python tests/compare_dual_filter_pnl.py --indicator $ind --split test
    done
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import argparse
import json
from typing import Dict, Tuple, List
from dataclasses import dataclass
import logging

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

@dataclass
class Trade:
    """Enregistrement d'un trade."""
    start: int
    end: int
    duration: int
    position: str  # LONG ou SHORT
    pnl: float
    pnl_after_fees: float
    returns: List[float]  # Returns step by step (pour MAE/MFE)


@dataclass
class BacktestResult:
    """Résultats d'un backtest."""
    filter_name: str
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
    sortino_ratio: float
    max_drawdown: float
    trades: List[Trade]
    positions: np.ndarray
    returns: np.ndarray


# =============================================================================
# FONCTIONS DE CHARGEMENT
# =============================================================================

def get_dataset_path(indicator: str, filter_type: str) -> str:
    """Construit le chemin du dataset."""
    filter_suffix = 'octave20' if filter_type == 'octave' else 'kalman'
    return f'data/prepared/dataset_btc_eth_bnb_ada_ltc_{indicator}_dual_binary_{filter_suffix}.npz'


def load_dataset(indicator: str, split: str, filter_type: str) -> Dict:
    """
    Charge le dataset .npz pour l'indicateur et le filtre spécifiés.

    Returns:
        dict avec X, Y, Y_pred (si disponible), metadata
    """
    path = get_dataset_path(indicator, filter_type)

    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset introuvable: {path}")

    logger.info(f"📂 Chargement {filter_type.upper()}: {path}")
    data = np.load(path, allow_pickle=True)

    result = {
        'X': data[f'X_{split}'],
        'Y': data[f'Y_{split}'],
        'Y_pred': data.get(f'Y_{split}_pred', None),
        'indicator': indicator,
        'filter': filter_type,
    }

    # Charger metadata
    if 'metadata' in data:
        metadata_str = str(data['metadata'])
        result['metadata'] = json.loads(metadata_str)
    else:
        result['metadata'] = {}

    logger.info(f"  ✅ X shape: {result['X'].shape}, Y shape: {result['Y'].shape}")
    if result['Y_pred'] is not None:
        logger.info(f"  ✅ Y_pred shape: {result['Y_pred'].shape}")

    return result


def extract_c_ret(X: np.ndarray, indicator: str) -> np.ndarray:
    """Extrait c_ret (Close return) de X."""
    if indicator in ['rsi', 'macd']:
        c_ret = X[:, -1, 0]  # (n_samples,)
    elif indicator == 'cci':
        c_ret = X[:, -1, 2]  # c_ret à index 2
    else:
        raise ValueError(f"Indicateur inconnu: {indicator}")
    return c_ret


# =============================================================================
# BACKTEST AVEC TRACKING MAE/MFE
# =============================================================================

def run_backtest_with_tracking(
    Y: np.ndarray,
    returns: np.ndarray,
    fees: float = 0.0015,
    use_predictions: bool = False,
    Y_pred: np.ndarray = None
) -> BacktestResult:
    """
    Backtest simple avec tracking détaillé pour MAE/MFE.

    Stratégie: Decision Matrix simple
    - Direction UP + Force STRONG → LONG
    - Direction DOWN + Force STRONG → SHORT
    - Sinon → FLAT (sortir)
    """
    if use_predictions:
        if Y_pred is None:
            raise ValueError("use_predictions=True mais Y_pred est None")
        signals = Y_pred
    else:
        signals = Y

    # Convertir probabilités en labels binaires si nécessaire
    if signals.max() <= 1.0 and signals.min() >= 0.0:
        unique_vals = np.unique(signals)
        if len(unique_vals) > 2:  # Probabilités continues
            direction = (signals[:, 0] > 0.5).astype(int)
            force = (signals[:, 1] > 0.5).astype(int)
            signals = np.column_stack([direction, force])

    n_samples = len(signals)
    positions = np.zeros(n_samples, dtype=int)
    trades = []

    # État
    position = 'FLAT'  # FLAT, LONG, SHORT
    entry_time = 0
    entry_position = 'FLAT'
    current_pnl = 0.0
    trade_returns = []  # Returns step by step

    for i in range(n_samples):
        direction = int(signals[i, 0])  # 0=DOWN, 1=UP
        force = int(signals[i, 1])      # 0=WEAK, 1=STRONG
        ret = returns[i]

        # Accumuler PnL si en position
        if position != 'FLAT':
            if position == 'LONG':
                step_pnl = ret
            else:  # SHORT
                step_pnl = -ret

            current_pnl += step_pnl
            trade_returns.append(step_pnl)

        # Decision Matrix
        if direction == 1 and force == 1:
            target_position = 'LONG'
        elif direction == 0 and force == 1:
            target_position = 'SHORT'
        else:
            target_position = 'FLAT'

        # Sortie si FLAT demandé
        if target_position == 'FLAT' and position != 'FLAT':
            trade_fees = 2 * fees
            pnl_after_fees = current_pnl - trade_fees

            trade = Trade(
                start=entry_time,
                end=i,
                duration=i - entry_time,
                position=entry_position,
                pnl=current_pnl,
                pnl_after_fees=pnl_after_fees,
                returns=trade_returns.copy()
            )
            trades.append(trade)

            position = 'FLAT'
            current_pnl = 0.0
            trade_returns = []

        # Entrée ou changement
        elif target_position != 'FLAT':
            if position == 'FLAT':
                # Nouvelle entrée
                position = target_position
                entry_position = position
                entry_time = i
                current_pnl = 0.0
                trade_returns = []

            elif position != target_position:
                # Changement de direction = fermer + ouvrir
                trade_fees = 2 * fees
                pnl_after_fees = current_pnl - trade_fees

                trade = Trade(
                    start=entry_time,
                    end=i,
                    duration=i - entry_time,
                    position=entry_position,
                    pnl=current_pnl,
                    pnl_after_fees=pnl_after_fees,
                    returns=trade_returns.copy()
                )
                trades.append(trade)

                # Nouvelle position
                position = target_position
                entry_position = position
                entry_time = i
                current_pnl = 0.0
                trade_returns = []

        # Enregistrer position
        if position == 'LONG':
            positions[i] = 1
        elif position == 'SHORT':
            positions[i] = -1
        else:
            positions[i] = 0

    # Fermer position finale
    if position != 'FLAT':
        trade_fees = 2 * fees
        pnl_after_fees = current_pnl - trade_fees

        trade = Trade(
            start=entry_time,
            end=n_samples - 1,
            duration=n_samples - 1 - entry_time,
            position=entry_position,
            pnl=current_pnl,
            pnl_after_fees=pnl_after_fees,
            returns=trade_returns.copy()
        )
        trades.append(trade)

    return trades, positions, returns


def compute_statistics(
    trades: List[Trade],
    positions: np.ndarray,
    returns: np.ndarray,
    filter_name: str
) -> BacktestResult:
    """Calcule toutes les statistiques à partir des trades."""

    if len(trades) == 0:
        return BacktestResult(
            filter_name=filter_name,
            n_trades=0,
            n_long=0,
            n_short=0,
            total_pnl=0.0,
            total_pnl_after_fees=0.0,
            total_fees=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_duration=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            trades=trades,
            positions=positions,
            returns=returns
        )

    # Basic stats
    n_trades = len(trades)
    n_long = sum(1 for t in trades if t.position == 'LONG')
    n_short = sum(1 for t in trades if t.position == 'SHORT')

    total_pnl = sum(t.pnl for t in trades)
    total_pnl_after_fees = sum(t.pnl_after_fees for t in trades)
    total_fees = total_pnl - total_pnl_after_fees

    # Win Rate
    wins = [t for t in trades if t.pnl_after_fees > 0]
    losses = [t for t in trades if t.pnl_after_fees < 0]
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0.0

    # Profit Factor
    gross_profit = sum(t.pnl_after_fees for t in wins)
    gross_loss = abs(sum(t.pnl_after_fees for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Avg Win/Loss
    avg_win = gross_profit / len(wins) if len(wins) > 0 else 0.0
    avg_loss = -gross_loss / len(losses) if len(losses) > 0 else 0.0

    # Avg Duration
    avg_duration = sum(t.duration for t in trades) / n_trades

    # Sharpe Ratio (annualisé)
    trade_returns = [t.pnl_after_fees for t in trades]
    returns_mean = np.mean(trade_returns)
    returns_std = np.std(trade_returns)

    # Estimer nombre de trades par an (données 5min)
    n_samples = len(positions)
    n_days = n_samples * 5 / 60 / 24
    trades_per_year = (n_trades / n_days) * 365 if n_days > 0 else 0

    sharpe_ratio = (returns_mean / returns_std) * np.sqrt(trades_per_year) if returns_std > 0 else 0.0

    # Sortino Ratio (downside deviation only)
    negative_returns = [r for r in trade_returns if r < 0]
    if len(negative_returns) > 0:
        downside_std = np.std(negative_returns)
        sortino_ratio = (returns_mean / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0.0
    else:
        sortino_ratio = np.inf  # Pas de pertes

    # Max Drawdown
    cumulative = [1.0]
    for t in trades:
        cumulative.append(cumulative[-1] * (1 + t.pnl_after_fees))
    cumulative = np.array(cumulative)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(drawdown.min()) * 100

    return BacktestResult(
        filter_name=filter_name,
        n_trades=n_trades,
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
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        trades=trades,
        positions=positions,
        returns=returns
    )


# =============================================================================
# ANALYSE MAE/MFE
# =============================================================================

def compute_mae_mfe(trades: List[Trade]) -> Dict:
    """
    Calcule MAE (Maximum Adverse Excursion) et MFE (Maximum Favorable Excursion).

    MAE: Plus grosse perte intra-trade (drawdown max dans le trade)
    MFE: Plus gros gain intra-trade (runup max dans le trade)
    """
    if len(trades) == 0:
        return {
            'avg_mae': 0.0,
            'avg_mfe': 0.0,
            'avg_mae_pct': 0.0,
            'avg_mfe_pct': 0.0,
        }

    maes = []
    mfes = []

    for trade in trades:
        if len(trade.returns) == 0:
            continue

        cumulative = np.cumsum(trade.returns)

        # MAE: Plus bas point (perte max)
        mae = float(cumulative.min())
        maes.append(mae)

        # MFE: Plus haut point (gain max)
        mfe = float(cumulative.max())
        mfes.append(mfe)

    avg_mae = np.mean(maes) if len(maes) > 0 else 0.0
    avg_mfe = np.mean(mfes) if len(mfes) > 0 else 0.0

    # Pourcentage (relatif au PnL final)
    avg_mae_pct = np.mean([mae / abs(t.pnl) * 100 if t.pnl != 0 else 0 for mae, t in zip(maes, trades)])
    avg_mfe_pct = np.mean([mfe / abs(t.pnl) * 100 if t.pnl != 0 else 0 for mfe, t in zip(mfes, trades)])

    return {
        'avg_mae': avg_mae,
        'avg_mfe': avg_mfe,
        'avg_mae_pct': avg_mae_pct,
        'avg_mfe_pct': avg_mfe_pct,
    }


# =============================================================================
# ANALYSE DISTRIBUTION GAINS
# =============================================================================

def analyze_distribution(trades: List[Trade]) -> Dict:
    """
    Analyse la distribution des gains.

    Objectif Vigilance #2: Détecter fat tails (queues épaisses)
    qui indiqueraient des gains rares mais très rentables.
    """
    if len(trades) == 0:
        return {}

    returns = [t.pnl_after_fees for t in trades]

    # Statistiques descriptives
    stats = {
        'mean': np.mean(returns),
        'median': np.median(returns),
        'std': np.std(returns),
        'skewness': float(np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 3)) if np.std(returns) > 0 else 0.0,
        'kurtosis': float(np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 4)) if np.std(returns) > 0 else 0.0,
    }

    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        stats[f'p{p}'] = float(np.percentile(returns, p))

    # Fat tails detection
    # Kurtosis > 3 = leptokurtic (fat tails)
    # Kurtosis < 3 = platykurtic (thin tails)
    stats['has_fat_tails'] = stats['kurtosis'] > 3.0

    return stats


# =============================================================================
# ANALYSE ZONES DE DÉSACCORD
# =============================================================================

def analyze_disagreement_zones(
    Y_octave: np.ndarray,
    Y_kalman: np.ndarray,
    trades_octave: List[Trade],
    trades_kalman: List[Trade]
) -> Dict:
    """
    Analyse les zones de désaccord entre Octave et Kalman.

    Types:
    - Isolés (1 sample): Microstructure noise (78-89% des désaccords)
    - Blocs (2+ samples): Transitions structurelles (11-22% des désaccords)

    Objectif: Vérifier si trades évités (zones filtrées) sont rentables ou non.
    """
    n_samples = len(Y_octave)

    # Direction agreement
    dir_agree = (Y_octave[:, 0] == Y_kalman[:, 0])

    # Force agreement
    force_agree = (Y_octave[:, 1] == Y_kalman[:, 1])

    # Total agreement
    total_agree = dir_agree & force_agree

    # Compter les types de désaccord
    n_agree = int(total_agree.sum())
    n_disagree = n_samples - n_agree
    pct_disagree = n_disagree / n_samples * 100

    # Identifier désaccords isolés vs blocs
    disagree_mask = ~total_agree
    disagree_indices = np.where(disagree_mask)[0]

    isolated = 0
    blocks = 0

    if len(disagree_indices) > 0:
        # Un désaccord est isolé si ses voisins sont d'accord
        for idx in disagree_indices:
            prev_agree = (idx == 0) or total_agree[idx - 1]
            next_agree = (idx == n_samples - 1) or total_agree[idx + 1]

            if prev_agree and next_agree:
                isolated += 1
            else:
                blocks += 1

    pct_isolated = isolated / n_disagree * 100 if n_disagree > 0 else 0.0
    pct_blocks = blocks / n_disagree * 100 if n_disagree > 0 else 0.0

    # PnL dans zones d'accord vs désaccord
    # TODO: Calculer PnL moyen par trade dans chaque zone

    return {
        'n_samples': n_samples,
        'n_agree': n_agree,
        'n_disagree': n_disagree,
        'pct_disagree': pct_disagree,
        'n_isolated': isolated,
        'n_blocks': blocks,
        'pct_isolated': pct_isolated,
        'pct_blocks': pct_blocks,
    }


# =============================================================================
# RAPPORT COMPARATIF
# =============================================================================

def print_comparison_report(
    result_octave: BacktestResult,
    result_kalman: BacktestResult,
    mae_mfe_octave: Dict,
    mae_mfe_kalman: Dict,
    dist_octave: Dict,
    dist_kalman: Dict,
    disagree_stats: Dict,
    indicator: str
):
    """Imprime le rapport comparatif complet."""

    logger.info("\n" + "="*80)
    logger.info(f"RAPPORT COMPARATIF - {indicator.upper()} (VIGILANCE #2)")
    logger.info("="*80)

    # Section 1: Métriques de base
    logger.info("\n📊 SECTION 1: MÉTRIQUES DE BASE")
    logger.info("-" * 80)
    logger.info(f"{'Métrique':<30} {'Octave':>15} {'Kalman':>15} {'Delta':>15}")
    logger.info("-" * 80)

    metrics = [
        ('Total Trades', result_octave.n_trades, result_kalman.n_trades),
        ('  - LONG', result_octave.n_long, result_kalman.n_long),
        ('  - SHORT', result_octave.n_short, result_kalman.n_short),
        ('PnL Net (%)', result_octave.total_pnl_after_fees * 100, result_kalman.total_pnl_after_fees * 100),
        ('Total Fees (%)', result_octave.total_fees * 100, result_kalman.total_fees * 100),
        ('Win Rate (%)', result_octave.win_rate, result_kalman.win_rate),
        ('Profit Factor', result_octave.profit_factor, result_kalman.profit_factor),
        ('Avg Win (%)', result_octave.avg_win * 100, result_kalman.avg_win * 100),
        ('Avg Loss (%)', result_octave.avg_loss * 100, result_kalman.avg_loss * 100),
        ('Avg Duration (periods)', result_octave.avg_duration, result_kalman.avg_duration),
    ]

    for name, val_o, val_k in metrics:
        if 'Trades' in name or 'LONG' in name or 'SHORT' in name:
            delta = val_o - val_k
            logger.info(f"{name:<30} {val_o:>15.0f} {val_k:>15.0f} {delta:>+15.0f}")
        else:
            delta = val_o - val_k
            logger.info(f"{name:<30} {val_o:>15.2f} {val_k:>15.2f} {delta:>+15.2f}")

    # Section 2: Sharpe et Sortino (VIGILANCE #2 CRITIQUE)
    logger.info("\n📈 SECTION 2: RATIOS RISQUE/RENDEMENT (VIGILANCE #2)")
    logger.info("-" * 80)
    logger.info(f"{'Métrique':<30} {'Octave':>15} {'Kalman':>15} {'Delta':>15}")
    logger.info("-" * 80)

    logger.info(f"{'Sharpe Ratio':<30} {result_octave.sharpe_ratio:>15.3f} {result_kalman.sharpe_ratio:>15.3f} {result_octave.sharpe_ratio - result_kalman.sharpe_ratio:>+15.3f}")
    logger.info(f"{'Sortino Ratio':<30} {result_octave.sortino_ratio:>15.3f} {result_kalman.sortino_ratio:>15.3f} {result_octave.sortino_ratio - result_kalman.sortino_ratio:>+15.3f}")
    logger.info(f"{'Max Drawdown (%)':<30} {result_octave.max_drawdown:>15.2f} {result_kalman.max_drawdown:>15.2f} {result_octave.max_drawdown - result_kalman.max_drawdown:>+15.2f}")

    # Section 3: MAE/MFE
    logger.info("\n🎯 SECTION 3: MAE/MFE (Maximum Adverse/Favorable Excursion)")
    logger.info("-" * 80)
    logger.info(f"{'Métrique':<30} {'Octave':>15} {'Kalman':>15} {'Delta':>15}")
    logger.info("-" * 80)

    logger.info(f"{'Avg MAE (%)':<30} {mae_mfe_octave['avg_mae']*100:>15.3f} {mae_mfe_kalman['avg_mae']*100:>15.3f} {(mae_mfe_octave['avg_mae'] - mae_mfe_kalman['avg_mae'])*100:>+15.3f}")
    logger.info(f"{'Avg MFE (%)':<30} {mae_mfe_octave['avg_mfe']*100:>15.3f} {mae_mfe_kalman['avg_mfe']*100:>15.3f} {(mae_mfe_octave['avg_mfe'] - mae_mfe_kalman['avg_mfe'])*100:>+15.3f}")
    logger.info(f"{'MAE % of final PnL':<30} {mae_mfe_octave['avg_mae_pct']:>15.1f} {mae_mfe_kalman['avg_mae_pct']:>15.1f} {mae_mfe_octave['avg_mae_pct'] - mae_mfe_kalman['avg_mae_pct']:>+15.1f}")
    logger.info(f"{'MFE % of final PnL':<30} {mae_mfe_octave['avg_mfe_pct']:>15.1f} {mae_mfe_kalman['avg_mfe_pct']:>15.1f} {mae_mfe_octave['avg_mfe_pct'] - mae_mfe_kalman['avg_mfe_pct']:>+15.1f}")

    # Section 4: Distribution (FAT TAILS - VIGILANCE #2)
    logger.info("\n📉 SECTION 4: DISTRIBUTION DES GAINS (FAT TAILS - VIGILANCE #2)")
    logger.info("-" * 80)
    logger.info(f"{'Métrique':<30} {'Octave':>15} {'Kalman':>15} {'Delta':>15}")
    logger.info("-" * 80)

    if dist_octave and dist_kalman:
        logger.info(f"{'Mean (%)':<30} {dist_octave['mean']*100:>15.3f} {dist_kalman['mean']*100:>15.3f} {(dist_octave['mean'] - dist_kalman['mean'])*100:>+15.3f}")
        logger.info(f"{'Median (%)':<30} {dist_octave['median']*100:>15.3f} {dist_kalman['median']*100:>15.3f} {(dist_octave['median'] - dist_kalman['median'])*100:>+15.3f}")
        logger.info(f"{'Std (%)':<30} {dist_octave['std']*100:>15.3f} {dist_kalman['std']*100:>15.3f} {(dist_octave['std'] - dist_kalman['std'])*100:>+15.3f}")
        logger.info(f"{'Skewness':<30} {dist_octave['skewness']:>15.3f} {dist_kalman['skewness']:>15.3f} {dist_octave['skewness'] - dist_kalman['skewness']:>+15.3f}")
        logger.info(f"{'Kurtosis':<30} {dist_octave['kurtosis']:>15.3f} {dist_kalman['kurtosis']:>15.3f} {dist_octave['kurtosis'] - dist_kalman['kurtosis']:>+15.3f}")
        logger.info(f"{'Fat Tails?':<30} {'Yes' if dist_octave['has_fat_tails'] else 'No':>15} {'Yes' if dist_kalman['has_fat_tails'] else 'No':>15} {'-':>15}")

        logger.info("\nPercentiles:")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            logger.info(f"{'P' + str(p) + ' (%)':<30} {dist_octave[f'p{p}']*100:>15.3f} {dist_kalman[f'p{p}']*100:>15.3f} {(dist_octave[f'p{p}'] - dist_kalman[f'p{p}'])*100:>+15.3f}")

    # Section 5: Zones de désaccord
    logger.info("\n🔍 SECTION 5: ZONES DE DÉSACCORD (ISOLÉS vs BLOCS)")
    logger.info("-" * 80)

    logger.info(f"Total samples:          {disagree_stats['n_samples']:>10}")
    logger.info(f"Accord total:           {disagree_stats['n_agree']:>10} ({(100 - disagree_stats['pct_disagree']):>6.2f}%)")
    logger.info(f"Désaccords:             {disagree_stats['n_disagree']:>10} ({disagree_stats['pct_disagree']:>6.2f}%)")
    logger.info(f"  - Isolés (1 sample):  {disagree_stats['n_isolated']:>10} ({disagree_stats['pct_isolated']:>6.2f}% des désaccords)")
    logger.info(f"  - Blocs (2+ samples): {disagree_stats['n_blocks']:>10} ({disagree_stats['pct_blocks']:>6.2f}% des désaccords)")

    # Section 6: Recommandations
    logger.info("\n✅ SECTION 6: RECOMMANDATIONS (VIGILANCE #2)")
    logger.info("-" * 80)

    # Sharpe Ratio comparison
    if result_octave.sharpe_ratio > result_kalman.sharpe_ratio:
        logger.info(f"✅ Octave a un meilleur Sharpe Ratio (+{result_octave.sharpe_ratio - result_kalman.sharpe_ratio:.3f})")
    elif result_kalman.sharpe_ratio > result_octave.sharpe_ratio:
        logger.info(f"⚠️  Kalman a un meilleur Sharpe Ratio (+{result_kalman.sharpe_ratio - result_octave.sharpe_ratio:.3f})")
    else:
        logger.info("⚪ Sharpe Ratio identique")

    # Sortino Ratio comparison
    if result_octave.sortino_ratio > result_kalman.sortino_ratio:
        logger.info(f"✅ Octave a un meilleur Sortino Ratio (+{result_octave.sortino_ratio - result_kalman.sortino_ratio:.3f})")
    elif result_kalman.sortino_ratio > result_octave.sortino_ratio:
        logger.info(f"⚠️  Kalman a un meilleur Sortino Ratio (+{result_kalman.sortino_ratio - result_octave.sortino_ratio:.3f})")
    else:
        logger.info("⚪ Sortino Ratio identique")

    # Fat tails
    if dist_octave and dist_kalman:
        if dist_octave['has_fat_tails'] and not dist_kalman['has_fat_tails']:
            logger.info("⚠️  Octave a des fat tails (gains rares mais très rentables possibles)")
        elif dist_kalman['has_fat_tails'] and not dist_octave['has_fat_tails']:
            logger.info("⚠️  Kalman a des fat tails (gains rares mais très rentables possibles)")
        elif dist_octave['has_fat_tails'] and dist_kalman['has_fat_tails']:
            logger.info("⚠️  Les deux ont des fat tails (distribution non-normale)")
        else:
            logger.info("✅ Pas de fat tails détectées (distribution proche de normale)")

    # Désaccords isolés
    if disagree_stats['pct_isolated'] > 65:
        logger.info(f"✅ {disagree_stats['pct_isolated']:.1f}% des désaccords sont isolés (bruit microstructure)")
        logger.info("   → Confirme recommandation Expert 1: Confirmation 2+ périodes élimine ce bruit")

    logger.info("\n" + "="*80)
    logger.info(f"VALIDATION VIGILANCE #2 - {indicator.upper()}")
    logger.info("="*80)

    # Verdict final
    if result_octave.sharpe_ratio > result_kalman.sharpe_ratio and result_octave.sortino_ratio > result_kalman.sortino_ratio:
        logger.info("✅ OCTAVE SUPÉRIEUR sur tous les critères risque/rendement")
    elif result_kalman.sharpe_ratio > result_octave.sharpe_ratio and result_kalman.sortino_ratio > result_octave.sortino_ratio:
        logger.info("⚠️  KALMAN SUPÉRIEUR sur tous les critères risque/rendement")
    else:
        logger.info("⚪ RÉSULTATS MIXTES - Analyse détaillée requise")

    logger.info("="*80 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comparaison PnL Octave vs Kalman (Vigilance #2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--indicator', type=str, required=True, choices=['rsi', 'macd', 'cci'],
                        help='Indicateur à tester')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Split à utiliser (défaut: test)')
    parser.add_argument('--fees', type=float, default=0.0015,
                        help='Frais par trade side (défaut: 0.0015 = 0.15%%)')
    parser.add_argument('--use-predictions', action='store_true',
                        help='Utiliser prédictions modèle au lieu de labels Oracle')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info(f"COMPARAISON OCTAVE VS KALMAN - {args.indicator.upper()}")
    logger.info("="*80)
    logger.info(f"Indicateur: {args.indicator}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Fees: {args.fees*100:.2f}% par side ({args.fees*2*100:.2f}% round-trip)")
    logger.info(f"Mode: {'Prédictions' if args.use_predictions else 'Labels Oracle'}")
    logger.info("="*80 + "\n")

    # Charger les deux datasets
    data_octave = load_dataset(args.indicator, args.split, 'octave')
    data_kalman = load_dataset(args.indicator, args.split, 'kalman')

    # Extraire c_ret
    returns_octave = extract_c_ret(data_octave['X'], args.indicator)
    returns_kalman = extract_c_ret(data_kalman['X'], args.indicator)

    # Vérifier alignement
    if not np.allclose(returns_octave, returns_kalman, rtol=1e-10, atol=1e-10):
        logger.warning("⚠️  ATTENTION: returns (c_ret) ne sont pas identiques entre Octave et Kalman!")
        logger.warning(f"   Max diff: {np.max(np.abs(returns_octave - returns_kalman)):.6e}")
    else:
        logger.info("✅ Returns (c_ret) identiques entre Octave et Kalman\n")

    # Backtest Octave
    logger.info("🔵 BACKTEST OCTAVE")
    logger.info("-" * 80)
    trades_octave, positions_octave, _ = run_backtest_with_tracking(
        Y=data_octave['Y'],
        returns=returns_octave,
        fees=args.fees,
        use_predictions=args.use_predictions,
        Y_pred=data_octave['Y_pred']
    )
    result_octave = compute_statistics(trades_octave, positions_octave, returns_octave, 'Octave')
    logger.info(f"Trades: {result_octave.n_trades}, PnL Net: {result_octave.total_pnl_after_fees*100:+.2f}%\n")

    # Backtest Kalman
    logger.info("🟢 BACKTEST KALMAN")
    logger.info("-" * 80)
    trades_kalman, positions_kalman, _ = run_backtest_with_tracking(
        Y=data_kalman['Y'],
        returns=returns_kalman,
        fees=args.fees,
        use_predictions=args.use_predictions,
        Y_pred=data_kalman['Y_pred']
    )
    result_kalman = compute_statistics(trades_kalman, positions_kalman, returns_kalman, 'Kalman')
    logger.info(f"Trades: {result_kalman.n_trades}, PnL Net: {result_kalman.total_pnl_after_fees*100:+.2f}%\n")

    # MAE/MFE
    logger.info("🎯 CALCUL MAE/MFE...")
    mae_mfe_octave = compute_mae_mfe(trades_octave)
    mae_mfe_kalman = compute_mae_mfe(trades_kalman)

    # Distribution
    logger.info("📉 ANALYSE DISTRIBUTION...")
    dist_octave = analyze_distribution(trades_octave)
    dist_kalman = analyze_distribution(trades_kalman)

    # Désaccords
    logger.info("🔍 ANALYSE DÉSACCORDS...")
    disagree_stats = analyze_disagreement_zones(
        data_octave['Y'],
        data_kalman['Y'],
        trades_octave,
        trades_kalman
    )

    # Rapport final
    print_comparison_report(
        result_octave,
        result_kalman,
        mae_mfe_octave,
        mae_mfe_kalman,
        dist_octave,
        dist_kalman,
        disagree_stats,
        args.indicator
    )


if __name__ == '__main__':
    main()
