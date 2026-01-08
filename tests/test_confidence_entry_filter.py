#!/usr/bin/env python3
"""
Test Stratégie Confidence Entry Filter - Filtrer ENTRÉES par confiance du modèle.

PHILOSOPHIE (Entry-Focused):
- ENTRÉE = Responsabilité ML (filtrer signaux de haute qualité)
- SORTIE = Portfolio Management externe (stop loss, trailing stop, temps)

HYPOTHÈSE:
Le modèle Direction-Only génère trop d'entrées de mauvaise qualité (88k trades, WR 10%).
Filtrer par CONFIDENCE (pred_proba > threshold) devrait améliorer qualité ENTRÉES.

PRINCIPE:
- ENTRÉE: Direction=UP/DOWN ET pred_proba > threshold
- SORTIE: Temps fixe (20 périodes ~100 min) OU Stop Loss (-2%)
- Test thresholds: 0.50 (baseline), 0.60, 0.65, 0.70, 0.75

OBJECTIF:
- Baseline: 88,113 trades, WR 9.90%, PnL Net -523%
- Cible: <30,000 entrées (-66%), WR >30%, PnL Net POSITIF

Usage:
    python tests/test_confidence_entry_filter.py --split test --fees 0.003 --exit-mode time
    python tests/test_confidence_entry_filter.py --split test --fees 0.003 --exit-mode stop

Référence:
    López de Prado (2018) - "Meta-Labeling" (Ch. 3)
    Signal Quality Filtering for High-Frequency Trading
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

# Thresholds de confidence à tester
CONFIDENCE_THRESHOLDS = [0.50, 0.60, 0.65, 0.70, 0.75]

# Paramètres sortie (Portfolio Management)
EXIT_TIME_DEFAULT = 20  # 20 périodes ~100 min
EXIT_STOP_LOSS = 0.02   # 2% stop loss


# =============================================================================
# DATACLASSES (RÉUTILISÉES)
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
    exit_reason: str  # "TIME", "STOP_LOSS", "DIRECTION_FLIP", "END"
    entry_confidence: float  # Confidence à l'entrée


@dataclass
class StrategyResult:
    """Résultats d'une stratégie."""
    name: str
    threshold: float
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
    # Métriques confidence
    avg_entry_confidence: float
    min_entry_confidence: float
    max_entry_confidence: float
    # Métriques sorties
    n_exits_time: int
    n_exits_stop_loss: int
    n_exits_direction_flip: int


# =============================================================================
# CHARGEMENT DONNÉES
# =============================================================================

def load_dataset_direction_only(split: str = 'test') -> Dict:
    """
    Charge dataset MACD Direction-Only Kalman.

    RÉUTILISE: Structure de test_atr_structural_filter.py.
    """
    path = f'data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_direction_only_kalman.npz'

    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset introuvable: {path}")

    logger.info(f"📂 Chargement: {path}")
    data = np.load(path, allow_pickle=True)

    # Vérifier shape Y (doit être (n,1) pour Direction-Only)
    Y_split = data[f'Y_{split}']
    if Y_split.ndim == 2 and Y_split.shape[1] == 1:
        logger.info(f"✅ Direction-Only détecté: Y shape = {Y_split.shape}")
    else:
        raise ValueError(f"Y shape incorrect (attendu (n,1)): {Y_split.shape}")

    return {
        'X': data[f'X_{split}'],
        'Y': data[f'Y_{split}'],
        'Y_pred': data.get(f'Y_{split}_pred', None),
    }


def extract_c_ret(X: np.ndarray) -> np.ndarray:
    """
    Extrait c_ret des features.

    RÉUTILISE: Logique exacte de test_atr_structural_filter.py.
    """
    # MACD: c_ret est feature 0
    c_ret = X[:, :, 0]
    return c_ret[:, -1]  # Dernier timestep


# =============================================================================
# BACKTEST AVEC FILTRE CONFIDENCE
# =============================================================================

def backtest_confidence_filter(
    pred: np.ndarray,
    returns: np.ndarray,
    fees: float,
    confidence_threshold: float = 0.5,
    exit_mode: str = 'time',
    exit_time: int = 20,
    exit_stop_loss: float = 0.02
) -> StrategyResult:
    """
    Backtest avec filtre confidence ENTRÉE + sortie simple.

    RÉUTILISE: Calcul PnL de test_atr_structural_filter.py (PROUVÉE!).
    FOCUS: Qualité ENTRÉES (confidence), sortie simple (portfolio management).

    Args:
        pred: Prédictions (n_samples, 1) - [Direction probabilities]
        returns: Returns (n_samples,)
        fees: Frais par side (ex: 0.003 = 0.3%)
        confidence_threshold: Seuil confidence ENTRÉE (0.5-1.0)
        exit_mode: 'time' (sortie temps fixe) ou 'stop' (stop loss)
        exit_time: Durée max trade en périodes (défaut: 20)
        exit_stop_loss: Stop loss % (défaut: 0.02 = 2%)

    Returns:
        StrategyResult
    """
    n_samples = len(pred)
    trades = []

    position = Position.FLAT
    entry_time = 0
    entry_confidence = 0.0
    current_pnl = 0.0
    best_pnl = 0.0  # Pour trailing stop (si besoin futur)

    n_long = 0
    n_short = 0
    n_exits_time = 0
    n_exits_stop_loss = 0
    n_exits_direction_flip = 0

    for i in range(n_samples):
        # Probabilité brute (Shape: (n,1) → extraire scalaire)
        prob = pred[i, 0] if pred.ndim == 2 else pred[i]
        ret = returns[i]

        # Direction (0=DOWN, 1=UP) avec confidence
        direction = 1 if prob > 0.5 else 0
        confidence = abs(prob - 0.5) * 2  # Normaliser [0.5,1.0] → [0,1]

        # Accumuler PnL (RÉUTILISE logique PROUVÉE!)
        if position != Position.FLAT:
            if position == Position.LONG:
                current_pnl += ret
            else:  # SHORT
                current_pnl -= ret

            # Tracker meilleur PnL
            best_pnl = max(best_pnl, current_pnl)

        # Decision Matrix ENTRÉE (confidence filter)
        if confidence >= confidence_threshold:
            if direction == 1:
                target = Position.LONG
            else:
                target = Position.SHORT
        else:
            target = Position.FLAT  # Confidence trop faible

        # Durée trade actuel
        trade_duration = i - entry_time if position != Position.FLAT else 0

        # LOGIQUE SORTIE (Portfolio Management Simple)
        exit_signal = False
        exit_reason = None

        if position != Position.FLAT:
            # Sortie 1: Temps max (exit_mode='time')
            if exit_mode == 'time' and trade_duration >= exit_time:
                exit_signal = True
                exit_reason = "TIME"
                n_exits_time += 1

            # Sortie 2: Stop loss (exit_mode='stop')
            elif exit_mode == 'stop' and current_pnl < -exit_stop_loss:
                exit_signal = True
                exit_reason = "STOP_LOSS"
                n_exits_stop_loss += 1

            # Sortie 3: Direction flip (TOUJOURS prioritaire)
            elif target != Position.FLAT and target != position:
                exit_signal = True
                exit_reason = "DIRECTION_FLIP"
                n_exits_direction_flip += 1

        # Exécuter sortie si nécessaire
        if exit_signal:
            trade_fees = 2 * fees  # Round-trip
            pnl_after_fees = current_pnl - trade_fees

            trades.append(Trade(
                start=entry_time,
                end=i,
                duration=trade_duration,
                position=position.value,
                pnl=current_pnl,
                pnl_after_fees=pnl_after_fees,
                exit_reason=exit_reason,
                entry_confidence=entry_confidence
            ))

            # Flip immédiat si DIRECTION_FLIP (RÉUTILISE logique commit e51a691!)
            if exit_reason == "DIRECTION_FLIP":
                position = target
                entry_time = i
                entry_confidence = confidence
                current_pnl = 0.0
                best_pnl = 0.0
                if target == Position.LONG:
                    n_long += 1
                else:
                    n_short += 1
            else:
                # Sortie complète (TIME ou STOP_LOSS)
                position = Position.FLAT
                current_pnl = 0.0
                best_pnl = 0.0

        # Nouvelle entrée si FLAT (confidence filter)
        elif position == Position.FLAT and target != Position.FLAT:
            position = target
            entry_time = i
            entry_confidence = confidence
            current_pnl = 0.0
            best_pnl = 0.0
            if target == Position.LONG:
                n_long += 1
            else:
                n_short += 1

    # Clôturer position finale si ouverte
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
            exit_reason="END",
            entry_confidence=entry_confidence
        ))

    # Calculer métriques (RÉUTILISE formules de test_atr_structural_filter.py)
    if len(trades) == 0:
        return StrategyResult(
            name=f"Conf_{confidence_threshold:.2f}",
            threshold=confidence_threshold,
            n_trades=0, n_long=0, n_short=0,
            total_pnl=0.0, total_pnl_after_fees=0.0, total_fees=0.0,
            win_rate=0.0, profit_factor=0.0,
            avg_win=0.0, avg_loss=0.0, avg_duration=0.0,
            sharpe_ratio=0.0, trades=[],
            avg_entry_confidence=0.0,
            min_entry_confidence=0.0,
            max_entry_confidence=0.0,
            n_exits_time=0,
            n_exits_stop_loss=0,
            n_exits_direction_flip=0
        )

    pnl_brut = sum(t.pnl for t in trades)
    pnl_net = sum(t.pnl_after_fees for t in trades)
    total_fees = pnl_brut - pnl_net

    wins = [t for t in trades if t.pnl_after_fees > 0]
    losses = [t for t in trades if t.pnl_after_fees <= 0]

    win_rate = len(wins) / len(trades) if trades else 0.0

    total_wins = sum(t.pnl_after_fees for t in wins)
    total_losses = abs(sum(t.pnl_after_fees for t in losses))
    profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

    avg_win = np.mean([t.pnl_after_fees for t in wins]) if wins else 0.0
    avg_loss = np.mean([t.pnl_after_fees for t in losses]) if losses else 0.0
    avg_duration = np.mean([t.duration for t in trades])

    # Sharpe Ratio
    trade_returns = np.array([t.pnl_after_fees for t in trades])
    sharpe = (trade_returns.mean() / trade_returns.std()) if trade_returns.std() > 0 else 0.0

    # Métriques confidence
    entry_confidences = [t.entry_confidence for t in trades]
    avg_entry_confidence = np.mean(entry_confidences)
    min_entry_confidence = np.min(entry_confidences)
    max_entry_confidence = np.max(entry_confidences)

    return StrategyResult(
        name=f"Conf_{confidence_threshold:.2f}",
        threshold=confidence_threshold,
        n_trades=len(trades),
        n_long=n_long,
        n_short=n_short,
        total_pnl=pnl_brut,
        total_pnl_after_fees=pnl_net,
        total_fees=total_fees,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_duration=avg_duration,
        sharpe_ratio=sharpe,
        trades=trades,
        avg_entry_confidence=avg_entry_confidence,
        min_entry_confidence=min_entry_confidence,
        max_entry_confidence=max_entry_confidence,
        n_exits_time=n_exits_time,
        n_exits_stop_loss=n_exits_stop_loss,
        n_exits_direction_flip=n_exits_direction_flip
    )


# =============================================================================
# AFFICHAGE RÉSULTATS
# =============================================================================

def print_comparison_table(results: List[StrategyResult], baseline: StrategyResult):
    """
    Affiche tableau comparatif (RÉUTILISE style de test_atr_structural_filter.py).
    """
    print("\n" + "="*120)
    print("RÉSULTATS CONFIDENCE ENTRY FILTER (Entry-Focused Strategy)".center(120))
    print("="*120)

    # Header
    header = (
        f"{'Config':<12} | "
        f"{'Trades':>8} | {'Réduc':>7} | "
        f"{'WR':>6} | {'Δ WR':>7} | "
        f"{'PnL Brut':>10} | {'PnL Net':>10} | "
        f"{'Avg Conf':>9} | {'Avg Dur':>8} | {'Verdict':<10}"
    )
    print(header)
    print("-"*120)

    # Baseline
    print(
        f"{'Baseline':<12} | "
        f"{baseline.n_trades:>8,} | {'-':>7} | "
        f"{baseline.win_rate*100:>5.2f}% | {'-':>7} | "
        f"{baseline.total_pnl:>+9.2f}% | {baseline.total_pnl_after_fees:>+9.2f}% | "
        f"{baseline.avg_entry_confidence:>8.2f} | {baseline.avg_duration:>7.1f}p | {'Référence':<10}"
    )
    print("-"*120)

    # Autres configs
    for res in results:
        if res.name == "Baseline":
            continue

        # Calculer réduction (avec protection division par zéro)
        if baseline.n_trades > 0:
            reduction = (1 - res.n_trades / baseline.n_trades) * 100
        else:
            reduction = 0.0

        delta_wr = (res.win_rate - baseline.win_rate) * 100

        # Verdict
        if res.total_pnl_after_fees > 0:
            verdict = "✅ POSITIF"
        elif res.win_rate > 0.30 and res.n_trades < 30000:
            verdict = "🎯 Objectif"
        elif res.total_pnl_after_fees > baseline.total_pnl_after_fees:
            verdict = "⚠️ Mieux"
        else:
            verdict = "❌ Pire"

        print(
            f"{res.name:<12} | "
            f"{res.n_trades:>8,} | {reduction:>+6.1f}% | "
            f"{res.win_rate*100:>5.2f}% | {delta_wr:>+6.2f}% | "
            f"{res.total_pnl:>+9.2f}% | {res.total_pnl_after_fees:>+9.2f}% | "
            f"{res.avg_entry_confidence:>8.2f} | {res.avg_duration:>7.1f}p | {verdict:<10}"
        )

    print("="*120)

    # Métriques détaillées du meilleur
    non_baseline_results = [r for r in results if r.name != "Baseline"]

    if non_baseline_results:
        best = max(non_baseline_results, key=lambda r: r.total_pnl_after_fees)
        print(f"\n🏆 MEILLEURE CONFIG: {best.name}")
        print(f"   Trades: {best.n_trades:,} ({best.n_long:,} LONG + {best.n_short:,} SHORT)")
        print(f"   Win Rate: {best.win_rate*100:.2f}%")
        print(f"   PnL Brut: {best.total_pnl:+.2f}%")
        print(f"   PnL Net: {best.total_pnl_after_fees:+.2f}%")
        print(f"   Frais: {best.total_fees:.2f}%")
        print(f"   Profit Factor: {best.profit_factor:.2f}")
        print(f"   Sharpe Ratio: {best.sharpe_ratio:.2f}")
        print(f"   Avg Duration: {best.avg_duration:.1f} périodes")
        print(f"   Entry Confidence: {best.avg_entry_confidence:.2f} (min={best.min_entry_confidence:.2f}, max={best.max_entry_confidence:.2f})")
        print(f"   Sorties: TIME={best.n_exits_time}, STOP={best.n_exits_stop_loss}, FLIP={best.n_exits_direction_flip}")
    else:
        print(f"\n⚠️ Aucune config testée (baseline uniquement)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test Confidence Entry Filter")
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Split à tester (défaut: test)')
    parser.add_argument('--fees', type=float, default=0.003,
                        help='Frais par side (défaut: 0.003 = 0.3%%)')
    parser.add_argument('--exit-mode', type=str, default='time', choices=['time', 'stop'],
                        help='Mode sortie: time (durée fixe) ou stop (stop loss)')
    parser.add_argument('--exit-time', type=int, default=20,
                        help='Durée max trade en périodes (défaut: 20 ~100min)')
    parser.add_argument('--exit-stop-loss', type=float, default=0.02,
                        help='Stop loss %% (défaut: 0.02 = 2%%)')

    args = parser.parse_args()

    logger.info(f"🚀 Test Confidence Entry Filter (Entry-Focused)")
    logger.info(f"   Split: {args.split}")
    logger.info(f"   Frais: {args.fees*100:.2f}% par side ({args.fees*2*100:.2f}% round-trip)")
    logger.info(f"   Exit Mode: {args.exit_mode.upper()}")
    if args.exit_mode == 'time':
        logger.info(f"   Exit Time: {args.exit_time} périodes (~{args.exit_time*5} min)")
    else:
        logger.info(f"   Stop Loss: {args.exit_stop_loss*100:.1f}%")

    # Charger dataset Direction-Only
    data = load_dataset_direction_only(args.split)

    X = data['X']
    Y = data['Y']
    Y_pred = data['Y_pred']

    if Y_pred is None:
        raise ValueError("Prédictions manquantes! Utiliser src/train.py avec --save-predictions")

    logger.info(f"✅ Dataset: X={X.shape}, Y={Y.shape}, Y_pred={Y_pred.shape}")

    # Extraire returns (RÉUTILISE logique validée!)
    returns = extract_c_ret(X)
    logger.info(f"✅ Returns extraits: {returns.shape}")

    # Tester configs confidence
    results = []

    # Baseline (threshold 0.50 = tous les signaux)
    logger.info(f"\n{'='*60}\nBASELINE (confidence >= 0.50, tous signaux)\n{'='*60}")

    baseline = backtest_confidence_filter(
        pred=Y_pred,
        returns=returns,
        fees=args.fees,
        confidence_threshold=0.50,
        exit_mode=args.exit_mode,
        exit_time=args.exit_time,
        exit_stop_loss=args.exit_stop_loss
    )
    baseline.name = "Baseline"
    results.append(baseline)

    logger.info(f"   Trades: {baseline.n_trades:,}")
    logger.info(f"   Win Rate: {baseline.win_rate*100:.2f}%")
    logger.info(f"   PnL Brut: {baseline.total_pnl:+.2f}%")
    logger.info(f"   PnL Net: {baseline.total_pnl_after_fees:+.2f}%")
    logger.info(f"   Avg Confidence: {baseline.avg_entry_confidence:.2f}")

    # Tester confidence thresholds
    for threshold in CONFIDENCE_THRESHOLDS[1:]:  # Skip 0.50 (baseline)
        logger.info(f"\n{'='*60}\nConfidence >= {threshold:.2f}\n{'='*60}")

        res = backtest_confidence_filter(
            pred=Y_pred,
            returns=returns,
            fees=args.fees,
            confidence_threshold=threshold,
            exit_mode=args.exit_mode,
            exit_time=args.exit_time,
            exit_stop_loss=args.exit_stop_loss
        )

        results.append(res)

        # Calculer réduction (avec protection division par zéro)
        if baseline.n_trades > 0:
            reduction_pct = (1 - res.n_trades / baseline.n_trades) * 100
            logger.info(f"   Trades: {res.n_trades:,} ({reduction_pct:+.1f}%)")
        else:
            logger.info(f"   Trades: {res.n_trades:,} (N/A)")

        logger.info(f"   Win Rate: {res.win_rate*100:.2f}% ({(res.win_rate-baseline.win_rate)*100:+.2f}%)")
        logger.info(f"   PnL Net: {res.total_pnl_after_fees:+.2f}%")
        logger.info(f"   Avg Confidence: {res.avg_entry_confidence:.2f}")

    # Afficher tableau comparatif
    print_comparison_table(results, baseline)

    logger.info(f"\n✅ Tests terminés!")


if __name__ == "__main__":
    main()
