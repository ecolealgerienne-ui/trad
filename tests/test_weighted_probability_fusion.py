#!/usr/bin/env python3
"""
Test Weighted Probability Fusion - Consensus Multi-Indicateurs

Implémente la fusion probabiliste pondérée (Weighted Probabilistic Fusion)
basée sur la littérature (López de Prado, Ryu & Kim 2022).

Méthode:
1. Normaliser les probabilités avec z-score par indicateur
   p_norm = (prob - mean) / std
2. Fusionner avec poids pondérés
   score = w_baseline * p_baseline_norm + w_other1 * p_other1_norm + w_other2 * p_other2_norm
3. Décision basée sur seuil
   score > threshold → LONG
   score < -threshold → SHORT
   sinon → HOLD

Poids par défaut (basés sur performance empirique):
- Baseline: 0.56 (indicateur principal)
- Other1:   0.28 (support)
- Other2:   0.16 (modulateur faible)

Usage:
    python tests/test_weighted_probability_fusion.py --split test --baseline macd
    python tests/test_weighted_probability_fusion.py --split test --baseline rsi
    python tests/test_weighted_probability_fusion.py --split test --baseline cci --threshold 0.5
"""

import argparse
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Tuple, Optional
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Position(IntEnum):
    """Position de trading."""
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass
class Trade:
    """Un trade individuel."""
    asset_id: int
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    direction: Position
    pnl_gross: float
    pnl_net: float
    duration: int
    score_at_entry: float


@dataclass
class IndicatorStats:
    """Statistiques de calibration pour un indicateur."""
    name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    n_samples: int

    def normalize(self, prob: float) -> float:
        """Normalise une probabilité avec z-score."""
        if self.std > 0:
            return (prob - self.mean) / self.std
        return 0.0


@dataclass
class StrategyResult:
    """Résultats d'une stratégie."""
    name: str
    total_trades: int
    win_rate: float
    pnl_gross: float
    pnl_net: float
    avg_duration: float
    sharpe: float
    fees_paid: float
    long_trades: int
    short_trades: int
    hold_periods: int

    # Per-asset results
    per_asset_pnl: Dict[int, float] = field(default_factory=dict)


def load_indicator_data(
    indicator: str,
    split: str,
    filter_type: str = 'kalman'
) -> Optional[Dict]:
    """Charge les données d'un indicateur."""
    base_path = Path('data/prepared')
    pattern = f'dataset_*_{indicator}_direction_only_{filter_type}.npz'
    files = list(base_path.glob(pattern))

    if not files:
        return None

    data = np.load(files[0], allow_pickle=True)

    result = {
        'Y': data[f'Y_{split}'],          # (n, 3) [timestamp, asset_id, direction]
        'OHLCV': data[f'OHLCV_{split}'],  # (n, 7) [timestamp, asset_id, O, H, L, C, V]
        'filename': files[0].name
    }

    # Charger probabilités si disponibles
    pred_key = f'Y_{split}_pred'
    if pred_key in data:
        Y_pred = data[pred_key]
        if Y_pred.ndim == 2:
            Y_pred = Y_pred[:, 0]
        result['Y_pred'] = Y_pred
    else:
        result['Y_pred'] = None

    # Charger aussi train pour calibration si on est sur test/val
    if split != 'train':
        train_pred_key = 'Y_train_pred'
        if train_pred_key in data:
            train_pred = data[train_pred_key]
            if train_pred.ndim == 2:
                train_pred = train_pred[:, 0]
            result['Y_train_pred'] = train_pred
        else:
            result['Y_train_pred'] = None

    return result


def compute_indicator_stats(probs: np.ndarray, name: str) -> IndicatorStats:
    """Calcule les statistiques de calibration pour un indicateur."""
    return IndicatorStats(
        name=name,
        mean=float(np.mean(probs)),
        std=float(np.std(probs)),
        min_val=float(np.min(probs)),
        max_val=float(np.max(probs)),
        n_samples=len(probs)
    )


def backtest_single_asset(
    labels: np.ndarray,
    opens: np.ndarray,
    fusion_scores: np.ndarray,
    asset_id: int,
    threshold: float = 0.5,
    fees: float = 0.001
) -> Tuple[List[Trade], Dict]:
    """
    Backtest sur un seul asset avec fusion probabiliste.

    Logique:
    - score > threshold → LONG
    - score < -threshold → SHORT
    - sinon → HOLD (pas de position)
    """
    n_samples = len(labels)

    if n_samples < 2:
        return [], {'long': 0, 'short': 0, 'hold': 0}

    position = Position.FLAT
    entry_idx = 0
    entry_price = 0.0
    entry_score = 0.0

    trades = []
    n_long = 0
    n_short = 0
    n_hold = 0

    for i in range(n_samples - 1):
        score = fusion_scores[i]

        # Déterminer le signal
        if score > threshold:
            target = Position.LONG
            n_long += 1
        elif score < -threshold:
            target = Position.SHORT
            n_short += 1
        else:
            target = Position.FLAT
            n_hold += 1

        # Gestion de position
        if position != Position.FLAT:
            # Sortie si signal opposé ou FLAT
            if target != position:
                exit_price = opens[i + 1]
                pnl_gross = (exit_price - entry_price) / entry_price
                if position == Position.SHORT:
                    pnl_gross = -pnl_gross

                fee = fees * 2
                pnl_net = pnl_gross - fee

                trades.append(Trade(
                    asset_id=asset_id,
                    entry_idx=entry_idx,
                    exit_idx=i,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    direction=position,
                    pnl_gross=pnl_gross,
                    pnl_net=pnl_net,
                    duration=i - entry_idx,
                    score_at_entry=entry_score
                ))

                # Nouvelle entrée si target != FLAT
                if target != Position.FLAT:
                    position = target
                    entry_idx = i
                    entry_price = opens[i + 1]
                    entry_score = score
                else:
                    position = Position.FLAT

        elif target != Position.FLAT:
            # Nouvelle entrée
            position = target
            entry_idx = i
            entry_price = opens[i + 1]
            entry_score = score

    # Fermer position finale
    if position != Position.FLAT:
        exit_idx = n_samples - 1
        exit_price = opens[exit_idx]

        pnl_gross = (exit_price - entry_price) / entry_price
        if position == Position.SHORT:
            pnl_gross = -pnl_gross

        fee = fees * 2
        pnl_net = pnl_gross - fee

        trades.append(Trade(
            asset_id=asset_id,
            entry_idx=entry_idx,
            exit_idx=exit_idx,
            entry_price=entry_price,
            exit_price=exit_price,
            direction=position,
            pnl_gross=pnl_gross,
            pnl_net=pnl_net,
            duration=exit_idx - entry_idx,
            score_at_entry=entry_score
        ))

    stats = {'long': n_long, 'short': n_short, 'hold': n_hold}
    return trades, stats


def run_weighted_fusion_backtest(
    datasets: Dict,
    baseline: str,
    weights: Dict[str, float],
    threshold: float,
    fees: float = 0.001,
    use_train_calibration: bool = True,
    raw_probs: bool = False,
    bias: float = 0.5
) -> StrategyResult:
    """
    Exécute le backtest avec fusion pondérée.

    Args:
        datasets: Données des 3 indicateurs
        baseline: Indicateur de référence ('macd', 'rsi', 'cci')
        weights: Dict des poids {'macd': 0.56, 'cci': 0.28, 'rsi': 0.16}
        threshold: Seuil de décision
        fees: Frais par trade
        use_train_calibration: Utiliser stats train pour normaliser
        raw_probs: Si True, utilise score = w1*p1 + w2*p2 + w3*p3 - bias
        bias: Biais pour raw_probs (défaut: 0.5)
    """
    indicators = ['macd', 'rsi', 'cci']

    # Extraire données baseline (référence pour Y et OHLCV)
    baseline_data = datasets[baseline]
    Y_baseline = baseline_data['Y']
    OHLCV = baseline_data['OHLCV']

    # Extraire probabilités pour chaque indicateur
    probs = {}
    for ind in indicators:
        if ind in datasets and datasets[ind].get('Y_pred') is not None:
            probs[ind] = datasets[ind]['Y_pred']
        elif ind == baseline:
            # Utiliser labels si pas de prédictions pour baseline
            logger.warning(f"⚠️ Pas de prédictions {baseline.upper()} - utilisation des labels")
            probs[ind] = Y_baseline[:, 2].astype(float)
        else:
            probs[ind] = None

    # === CALIBRATION (stats sur train) - seulement si z-score ===
    stats = {}
    if not raw_probs:
        for ind in indicators:
            if probs[ind] is None:
                stats[ind] = None
                continue

            if use_train_calibration:
                train_pred = datasets[ind].get('Y_train_pred') if ind in datasets else None
                if train_pred is not None:
                    stats[ind] = compute_indicator_stats(train_pred, ind.upper())
                else:
                    stats[ind] = compute_indicator_stats(probs[ind], ind.upper())
            else:
                stats[ind] = compute_indicator_stats(probs[ind], ind.upper())

        # Afficher stats calibration
        logger.info(f"\n📊 CALIBRATION (z-score normalization):")
        for ind in indicators:
            if stats[ind]:
                marker = "⭐" if ind == baseline else "  "
                logger.info(f"  {marker} {ind.upper()}: mean={stats[ind].mean:.4f}, std={stats[ind].std:.4f}, weight={weights.get(ind, 0):.2f}")
    else:
        # Mode raw probs - afficher stats basiques
        logger.info(f"\n📊 MODE RAW PROBS (score = w1*p1 + w2*p2 + w3*p3 - {bias}):")
        for ind in indicators:
            if probs[ind] is not None:
                marker = "⭐" if ind == baseline else "  "
                p_mean = np.mean(probs[ind])
                p_std = np.std(probs[ind])
                logger.info(f"  {marker} {ind.upper()}: mean={p_mean:.4f}, std={p_std:.4f}, weight={weights.get(ind, 0):.2f}")

    # === CALCUL SCORES FUSIONNÉS ===
    n_samples = len(probs[baseline])
    fusion_scores = np.zeros(n_samples)

    for i in range(n_samples):
        score = 0.0
        for ind in indicators:
            if probs[ind] is not None:
                if raw_probs:
                    # Raw probs: score = w1*p1 + w2*p2 + w3*p3 - bias
                    score += weights.get(ind, 0) * probs[ind][i]
                else:
                    # Z-score normalization
                    if stats[ind] is not None:
                        p_norm = stats[ind].normalize(probs[ind][i])
                        score += weights.get(ind, 0) * p_norm

        if raw_probs:
            score -= bias  # Soustraire le biais pour centrer autour de 0

        fusion_scores[i] = score

    # Afficher distribution des scores
    logger.info(f"\n📈 DISTRIBUTION SCORES FUSIONNÉS:")
    logger.info(f"  min={fusion_scores.min():.3f}, max={fusion_scores.max():.3f}")
    logger.info(f"  mean={fusion_scores.mean():.3f}, std={fusion_scores.std():.3f}")
    logger.info(f"  >+{threshold}: {(fusion_scores > threshold).sum():,} ({(fusion_scores > threshold).mean()*100:.1f}%)")
    logger.info(f"  <-{threshold}: {(fusion_scores < -threshold).sum():,} ({(fusion_scores < -threshold).mean()*100:.1f}%)")
    logger.info(f"  HOLD zone: {((fusion_scores >= -threshold) & (fusion_scores <= threshold)).sum():,}")

    # === BACKTEST PER-ASSET ===
    asset_ids = np.unique(OHLCV[:, 1].astype(int))
    all_trades = []
    per_asset_pnl = {}
    total_long = 0
    total_short = 0
    total_hold = 0

    for asset_id in asset_ids:
        mask = OHLCV[:, 1].astype(int) == asset_id

        asset_labels = Y_baseline[mask, 2]
        asset_opens = OHLCV[mask, 2]
        asset_scores = fusion_scores[mask]

        trades, stats = backtest_single_asset(
            labels=asset_labels,
            opens=asset_opens,
            fusion_scores=asset_scores,
            asset_id=asset_id,
            threshold=threshold,
            fees=fees
        )

        all_trades.extend(trades)
        total_long += stats['long']
        total_short += stats['short']
        total_hold += stats['hold']

        if trades:
            per_asset_pnl[asset_id] = sum(t.pnl_net for t in trades) * 100

    # === MÉTRIQUES ===
    if not all_trades:
        return StrategyResult(
            name=f"Fusion(t={threshold})",
            total_trades=0,
            win_rate=0.0,
            pnl_gross=0.0,
            pnl_net=0.0,
            avg_duration=0.0,
            sharpe=0.0,
            fees_paid=0.0,
            long_trades=0,
            short_trades=0,
            hold_periods=total_hold,
            per_asset_pnl={}
        )

    n_trades = len(all_trades)
    wins = sum(1 for t in all_trades if t.pnl_net > 0)
    win_rate = wins / n_trades * 100

    pnl_gross = sum(t.pnl_gross for t in all_trades) * 100
    pnl_net = sum(t.pnl_net for t in all_trades) * 100
    fees_paid = n_trades * fees * 2 * 100

    avg_duration = np.mean([t.duration for t in all_trades])

    # Sharpe
    returns = np.array([t.pnl_net for t in all_trades])
    if len(returns) > 1 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 288)
    else:
        sharpe = 0.0

    # Compter long/short trades
    long_trades = sum(1 for t in all_trades if t.direction == Position.LONG)
    short_trades = sum(1 for t in all_trades if t.direction == Position.SHORT)

    return StrategyResult(
        name=f"Fusion(t={threshold})",
        total_trades=n_trades,
        win_rate=win_rate,
        pnl_gross=pnl_gross,
        pnl_net=pnl_net,
        avg_duration=avg_duration,
        sharpe=sharpe,
        fees_paid=fees_paid,
        long_trades=long_trades,
        short_trades=short_trades,
        hold_periods=total_hold,
        per_asset_pnl=per_asset_pnl
    )


def run_baseline_only(
    datasets: Dict,
    baseline: str,
    fees: float = 0.001
) -> StrategyResult:
    """Baseline: indicateur seul (direction binaire, toujours en position)."""

    baseline_data = datasets[baseline]
    Y_baseline = baseline_data['Y']
    OHLCV = baseline_data['OHLCV']

    asset_ids = np.unique(OHLCV[:, 1].astype(int))
    all_trades = []
    per_asset_pnl = {}

    for asset_id in asset_ids:
        mask = OHLCV[:, 1].astype(int) == asset_id

        labels = Y_baseline[mask, 2]
        opens = OHLCV[mask, 2]

        n_samples = len(labels)
        if n_samples < 2:
            continue

        position = Position.FLAT
        entry_idx = 0
        entry_price = 0.0
        trades = []

        for i in range(n_samples - 1):
            direction = int(labels[i])
            target = Position.LONG if direction == 1 else Position.SHORT

            if position != Position.FLAT and target != position:
                # Sortie + flip
                exit_price = opens[i + 1]
                pnl_gross = (exit_price - entry_price) / entry_price
                if position == Position.SHORT:
                    pnl_gross = -pnl_gross

                pnl_net = pnl_gross - fees * 2

                trades.append(Trade(
                    asset_id=asset_id,
                    entry_idx=entry_idx,
                    exit_idx=i,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    direction=position,
                    pnl_gross=pnl_gross,
                    pnl_net=pnl_net,
                    duration=i - entry_idx,
                    score_at_entry=0.0
                ))

                position = target
                entry_idx = i
                entry_price = opens[i + 1]

            elif position == Position.FLAT:
                position = target
                entry_idx = i
                entry_price = opens[i + 1]

        # Fermer position finale
        if position != Position.FLAT:
            exit_price = opens[n_samples - 1]
            pnl_gross = (exit_price - entry_price) / entry_price
            if position == Position.SHORT:
                pnl_gross = -pnl_gross

            pnl_net = pnl_gross - fees * 2

            trades.append(Trade(
                asset_id=asset_id,
                entry_idx=entry_idx,
                exit_idx=n_samples - 1,
                entry_price=entry_price,
                exit_price=exit_price,
                direction=position,
                pnl_gross=pnl_gross,
                pnl_net=pnl_net,
                duration=n_samples - 1 - entry_idx,
                score_at_entry=0.0
            ))

        all_trades.extend(trades)
        if trades:
            per_asset_pnl[asset_id] = sum(t.pnl_net for t in trades) * 100

    # Métriques
    if not all_trades:
        return StrategyResult(
            name=f"{baseline.upper()} Baseline",
            total_trades=0,
            win_rate=0.0,
            pnl_gross=0.0,
            pnl_net=0.0,
            avg_duration=0.0,
            sharpe=0.0,
            fees_paid=0.0,
            long_trades=0,
            short_trades=0,
            hold_periods=0,
            per_asset_pnl={}
        )

    n_trades = len(all_trades)
    wins = sum(1 for t in all_trades if t.pnl_net > 0)
    win_rate = wins / n_trades * 100

    pnl_gross = sum(t.pnl_gross for t in all_trades) * 100
    pnl_net = sum(t.pnl_net for t in all_trades) * 100
    fees_paid = n_trades * fees * 2 * 100

    avg_duration = np.mean([t.duration for t in all_trades])

    returns = np.array([t.pnl_net for t in all_trades])
    if len(returns) > 1 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 288)
    else:
        sharpe = 0.0

    long_trades = sum(1 for t in all_trades if t.direction == Position.LONG)
    short_trades = sum(1 for t in all_trades if t.direction == Position.SHORT)

    return StrategyResult(
        name=f"{baseline.upper()} Baseline",
        total_trades=n_trades,
        win_rate=win_rate,
        pnl_gross=pnl_gross,
        pnl_net=pnl_net,
        avg_duration=avg_duration,
        sharpe=sharpe,
        fees_paid=fees_paid,
        long_trades=long_trades,
        short_trades=short_trades,
        hold_periods=0,
        per_asset_pnl=per_asset_pnl
    )


def print_results(results: List[StrategyResult], baseline: StrategyResult):
    """Affiche les résultats comparatifs."""

    logger.info("\n" + "="*140)
    logger.info("RÉSULTATS - WEIGHTED PROBABILITY FUSION")
    logger.info("="*140)

    # Header
    logger.info(f"\n{'Stratégie':<20} {'Trades':>8} {'Réduc':>7} {'WR':>7} {'Δ WR':>7} "
                f"{'PnL Brut':>12} {'PnL Net':>12} {'Frais':>10} {'Sharpe':>8} {'Long':>6} {'Short':>6} {'HOLD':>8}")
    logger.info("-"*140)

    # Baseline
    logger.info(f"{baseline.name:<20} {baseline.total_trades:>8,} {'-':>7} "
                f"{baseline.win_rate:>6.2f}% {'-':>7} "
                f"{baseline.pnl_gross:>+11.2f}% {baseline.pnl_net:>+11.2f}% "
                f"{baseline.fees_paid:>9.2f}% {baseline.sharpe:>8.2f} "
                f"{baseline.long_trades:>6,} {baseline.short_trades:>6,} {baseline.hold_periods:>8,}")

    # Autres stratégies
    for result in results:
        if baseline.total_trades > 0:
            reduction = (baseline.total_trades - result.total_trades) / baseline.total_trades * 100
            delta_wr = result.win_rate - baseline.win_rate
        else:
            reduction = 0
            delta_wr = 0

        logger.info(f"{result.name:<20} {result.total_trades:>8,} {reduction:>6.1f}% "
                    f"{result.win_rate:>6.2f}% {delta_wr:>+6.2f}% "
                    f"{result.pnl_gross:>+11.2f}% {result.pnl_net:>+11.2f}% "
                    f"{result.fees_paid:>9.2f}% {result.sharpe:>8.2f} "
                    f"{result.long_trades:>6,} {result.short_trades:>6,} {result.hold_periods:>8,}")

    logger.info("="*140)

    # Per-asset breakdown pour la meilleure stratégie
    best_result = max(results, key=lambda r: r.pnl_net) if results else baseline

    logger.info(f"\n📊 RÉSULTATS PAR ASSET ({best_result.name}):")
    logger.info("-"*80)

    for asset_id in sorted(best_result.per_asset_pnl.keys()):
        pnl = best_result.per_asset_pnl[asset_id]
        baseline_pnl = baseline.per_asset_pnl.get(asset_id, 0)
        delta = pnl - baseline_pnl
        status = "✅" if pnl > 0 else "❌"
        logger.info(f"  {status} Asset {asset_id}: {pnl:>+10.2f}% (vs baseline: {delta:>+8.2f}%)")

    # Analyse
    logger.info("\n" + "="*140)
    logger.info("ANALYSE")
    logger.info("="*140)

    if results:
        best = max(results, key=lambda r: r.pnl_net)
        if best.pnl_net > baseline.pnl_net:
            improvement = best.pnl_net - baseline.pnl_net
            logger.info(f"\n✅ MEILLEURE STRATÉGIE: {best.name}")
            logger.info(f"   PnL Net: {baseline.pnl_net:+.2f}% → {best.pnl_net:+.2f}% ({improvement:+.2f}%)")
            logger.info(f"   Trades: {baseline.total_trades:,} → {best.total_trades:,} "
                        f"({(best.total_trades - baseline.total_trades) / baseline.total_trades * 100:+.1f}%)")
        else:
            logger.info(f"\n⚠️ Aucune amélioration vs baseline {baseline.name}")
            logger.info(f"   Baseline reste meilleur: {baseline.pnl_net:+.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Test Weighted Probability Fusion',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Split de données (défaut: test)')
    parser.add_argument('--filter', type=str, default='kalman',
                        choices=['kalman', 'octave20'],
                        help='Type de filtre (défaut: kalman)')
    parser.add_argument('--fees', type=float, default=0.001,
                        help='Frais par trade (défaut: 0.001 = 0.1%%)')

    # Baseline indicator
    parser.add_argument('--baseline', type=str, default='macd',
                        choices=['macd', 'rsi', 'cci'],
                        help='Indicateur baseline (défaut: macd)')

    # Poids (baseline=0.56, other1=0.28, other2=0.16 par défaut)
    parser.add_argument('--w-macd', type=float, default=None,
                        help='Poids MACD (auto si non spécifié)')
    parser.add_argument('--w-cci', type=float, default=None,
                        help='Poids CCI (auto si non spécifié)')
    parser.add_argument('--w-rsi', type=float, default=None,
                        help='Poids RSI (auto si non spécifié)')

    # Seuils à tester
    parser.add_argument('--threshold', type=float, default=None,
                        help='Seuil unique à tester (défaut: teste plusieurs)')
    parser.add_argument('--thresholds', type=str, default='0.3,0.5,0.7,1.0',
                        help='Seuils à tester (défaut: 0.3,0.5,0.7,1.0)')

    # Mode raw probs vs z-score
    parser.add_argument('--raw-probs', action='store_true',
                        help='Utiliser score = w1*p1 + w2*p2 + w3*p3 - bias (au lieu de z-score)')
    parser.add_argument('--bias', type=float, default=0.5,
                        help='Biais pour raw-probs (défaut: 0.5)')

    args = parser.parse_args()

    # Déterminer les poids automatiquement si non spécifiés
    # Baseline = 0.56, premier autre = 0.28, second autre = 0.16
    default_weights = {'macd': 0.0, 'rsi': 0.0, 'cci': 0.0}
    other_indicators = [ind for ind in ['macd', 'rsi', 'cci'] if ind != args.baseline]

    if args.w_macd is None and args.w_rsi is None and args.w_cci is None:
        # Auto-assign based on baseline
        default_weights[args.baseline] = 0.56
        default_weights[other_indicators[0]] = 0.28
        default_weights[other_indicators[1]] = 0.16
    else:
        # Use provided values
        default_weights['macd'] = args.w_macd if args.w_macd is not None else 0.0
        default_weights['rsi'] = args.w_rsi if args.w_rsi is not None else 0.0
        default_weights['cci'] = args.w_cci if args.w_cci is not None else 0.0

    weights = default_weights

    logger.info("="*140)
    logger.info("TEST WEIGHTED PROBABILITY FUSION")
    logger.info("="*140)
    logger.info(f"\n⚙️  CONFIGURATION:")
    logger.info(f"  Split: {args.split}")
    logger.info(f"  Filter: {args.filter}")
    logger.info(f"  Frais: {args.fees*100:.2f}%")
    logger.info(f"  Baseline: {args.baseline.upper()} ⭐")
    if args.raw_probs:
        logger.info(f"  Mode: RAW PROBS (score = w1*p1 + w2*p2 + w3*p3 - {args.bias})")
    else:
        logger.info(f"  Mode: Z-SCORE (score = w1*z1 + w2*z2 + w3*z3)")
    logger.info(f"\n⚖️  POIDS:")
    for ind in ['macd', 'rsi', 'cci']:
        marker = "⭐" if ind == args.baseline else "  "
        logger.info(f"  {marker} {ind.upper()}: {weights[ind]:.2f}")

    # Charger données
    logger.info(f"\n📂 CHARGEMENT DES DONNÉES...")

    datasets = {}
    for indicator in ['macd', 'rsi', 'cci']:
        data = load_indicator_data(indicator, args.split, args.filter)
        if data:
            n_samples = len(data['Y'])
            has_preds = data.get('Y_pred') is not None
            logger.info(f"  ✅ {indicator.upper()}: {n_samples:,} samples, prédictions: {'✅' if has_preds else '❌'}")
            datasets[indicator] = data
        else:
            logger.warning(f"  ❌ {indicator.upper()}: Non trouvé")

    if args.baseline not in datasets:
        logger.error(f"❌ {args.baseline.upper()} requis!")
        return 1

    # Baseline (indicateur seul)
    logger.info(f"\n🔄 Test Baseline ({args.baseline.upper()} seul)...")
    baseline_result = run_baseline_only(datasets, baseline=args.baseline, fees=args.fees)

    # Tests avec différents seuils
    if args.threshold is not None:
        thresholds = [args.threshold]
    else:
        thresholds = [float(t) for t in args.thresholds.split(',')]

    results = []
    for threshold in thresholds:
        logger.info(f"\n🔄 Test Fusion (threshold={threshold})...")
        result = run_weighted_fusion_backtest(
            datasets=datasets,
            baseline=args.baseline,
            weights=weights,
            threshold=threshold,
            fees=args.fees,
            use_train_calibration=True,
            raw_probs=args.raw_probs,
            bias=args.bias
        )
        results.append(result)

    # Afficher résultats
    print_results(results, baseline_result)

    logger.info(f"\n✅ Test terminé!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
