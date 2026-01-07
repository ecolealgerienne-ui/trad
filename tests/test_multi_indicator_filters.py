#!/usr/bin/env python3
"""
Test Multi-Indicateurs avec Filtres Croisés - Holding 5 périodes.

PRINCIPE:
- MACD = Décideur principal (Direction + Force)
- RSI + CCI = Témoins/Filtres (veto si désaccord)
- Holding minimum = 5 périodes (fixe)
- Retournement Direction → Sortie immédiate (même si < 5p)

RÈGLES HOLDING:
1. Retournement MACD Direction → EXIT + REVERSE (immédiat)
2. Force=WEAK et duration < 5p → CONTINUER
3. Force=WEAK et duration >= 5p → EXIT

COMBINAISONS TESTÉES:
8 combinaisons de filtres (Kalman/Octave) pour MACD, RSI, CCI

Usage:
    python tests/test_multi_indicator_filters.py --split test
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import argparse
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
from itertools import product
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTES
# =============================================================================

MIN_HOLDING = 5  # Périodes (fixe)
FEES = 0.0015  # Par side


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
    exit_reason: str


@dataclass
class FilterCombo:
    """Combinaison de filtres."""
    macd_filter: str  # 'kalman' ou 'octave'
    rsi_filter: str
    cci_filter: str

    @property
    def name(self):
        """Nom court (ex: KKK, OKO)."""
        return f"{self.macd_filter[0].upper()}{self.rsi_filter[0].upper()}{self.cci_filter[0].upper()}"

    @property
    def full_name(self):
        """Nom complet."""
        return f"MACD-{self.macd_filter.capitalize()}_RSI-{self.rsi_filter.capitalize()}_CCI-{self.cci_filter.capitalize()}"


@dataclass
class StrategyResult:
    """Résultats d'une stratégie."""
    combo: FilterCombo
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


# =============================================================================
# CHARGEMENT DONNÉES
# =============================================================================

def load_dataset(indicator: str, filter_type: str, split: str = 'test') -> Dict:
    """Charge dataset."""
    filter_suffix = 'octave20' if filter_type == 'octave' else 'kalman'
    path = f'data/prepared/dataset_btc_eth_bnb_ada_ltc_{indicator}_dual_binary_{filter_suffix}.npz'

    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset introuvable: {path}")

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
# BACKTEST MULTI-INDICATEURS
# =============================================================================

def backtest_multi_indicator(
    pred_macd: np.ndarray,
    pred_rsi: np.ndarray,
    pred_cci: np.ndarray,
    returns: np.ndarray,
    combo: FilterCombo
) -> StrategyResult:
    """
    Backtest avec MACD décideur + RSI/CCI filtres.

    Règles:
    1. MACD décide Direction et Force
    2. RSI/CCI peuvent veto (si désaccord fort)
    3. Holding minimum 5 périodes
    4. Retournement → sortie immédiate
    """
    # Convertir en binaire
    macd_bin = (pred_macd > 0.5).astype(int)
    rsi_bin = (pred_rsi > 0.5).astype(int)
    cci_bin = (pred_cci > 0.5).astype(int)

    n_samples = len(macd_bin)
    trades = []

    position = Position.FLAT
    entry_time = 0
    current_pnl = 0.0
    prev_macd_dir = None

    n_long = 0
    n_short = 0

    for i in range(n_samples):
        # MACD = Décideur
        macd_dir = int(macd_bin[i, 0])
        macd_force = int(macd_bin[i, 1])

        # RSI/CCI = Témoins
        rsi_dir = int(rsi_bin[i, 0])
        rsi_force = int(rsi_bin[i, 1])
        cci_dir = int(cci_bin[i, 0])
        cci_force = int(cci_bin[i, 1])

        ret = returns[i]

        # Accumuler PnL
        if position != Position.FLAT:
            if position == Position.LONG:
                current_pnl += ret
            else:
                current_pnl -= ret

        # Durée trade actuel
        trade_duration = i - entry_time if position != Position.FLAT else 0

        # DÉTECTION RETOURNEMENT MACD
        direction_flip = False
        if prev_macd_dir is not None and macd_dir != prev_macd_dir:
            direction_flip = True

        prev_macd_dir = macd_dir

        # DÉCISION TARGET (basée sur MACD)
        if macd_dir == 1 and macd_force == 1:
            target = Position.LONG
        elif macd_dir == 0 and macd_force == 1:
            target = Position.SHORT
        else:
            target = Position.FLAT

        # LOGIQUE SORTIE/ENTRÉE
        exit_signal = False
        exit_reason = None

        if position != Position.FLAT:
            # CAS 1: RETOURNEMENT DIRECTION → EXIT IMMÉDIAT
            if direction_flip and target != Position.FLAT and target != position:
                exit_signal = True
                exit_reason = "DIRECTION_FLIP"

            # CAS 2: FORCE=WEAK
            elif target == Position.FLAT:
                if trade_duration >= MIN_HOLDING:
                    # Holding atteint → EXIT
                    exit_signal = True
                    exit_reason = "FORCE_WEAK"
                else:
                    # Holding pas atteint → CONTINUER
                    exit_signal = False

        # Exécuter sortie
        if exit_signal:
            trade_fees = 2 * FEES
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

            # FLIP si retournement
            if exit_reason == "DIRECTION_FLIP":
                position = target
                entry_time = i
                current_pnl = 0.0
                if target == Position.LONG:
                    n_long += 1
                else:
                    n_short += 1
            else:
                # Sortie complète
                position = Position.FLAT
                current_pnl = 0.0

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
        trade_fees = 2 * FEES
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

    return compute_stats(trades, n_long, n_short, combo)


def compute_stats(
    trades: List[Trade],
    n_long: int,
    n_short: int,
    combo: FilterCombo
) -> StrategyResult:
    """Calcule statistiques."""
    if len(trades) == 0:
        return StrategyResult(
            combo=combo,
            n_trades=0, n_long=0, n_short=0,
            total_pnl=0.0, total_pnl_after_fees=0.0, total_fees=0.0,
            win_rate=0.0, profit_factor=0.0,
            avg_win=0.0, avg_loss=0.0, avg_duration=0.0,
            sharpe_ratio=0.0, trades=[]
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

    # Sharpe Ratio
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
        combo=combo,
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
        trades=trades
    )


# =============================================================================
# AFFICHAGE
# =============================================================================

def print_results(results: List[StrategyResult]):
    """Affiche résultats comparatifs."""
    logger.info("\n" + "="*120)
    logger.info("COMPARAISON COMBINAISONS DE FILTRES (Holding 5p)")
    logger.info("="*120)
    logger.info(f"{'Combo':<8} {'MACD':<10} {'RSI':<10} {'CCI':<10} {'Trades':>8} {'Win Rate':>9} {'PnL Brut':>10} {'PnL Net':>10} {'Sharpe':>8} {'Avg Dur':>8}")
    logger.info("-"*120)

    for r in results:
        logger.info(
            f"{r.combo.name:<8} {r.combo.macd_filter.capitalize():<10} {r.combo.rsi_filter.capitalize():<10} "
            f"{r.combo.cci_filter.capitalize():<10} {r.n_trades:>8,} {r.win_rate*100:>8.2f}% "
            f"{r.total_pnl*100:>9.2f}% {r.total_pnl_after_fees*100:>9.2f}% "
            f"{r.sharpe_ratio:>8.3f} {r.avg_duration:>7.1f}p"
        )

    # Top 3
    logger.info("\n📊 TOP 3 COMBINAISONS (Sharpe Ratio):")

    sorted_results = sorted(results, key=lambda r: r.sharpe_ratio, reverse=True)

    for i, r in enumerate(sorted_results[:3], 1):
        logger.info(f"\n{i}. {r.combo.full_name}")
        logger.info(f"   Trades: {r.n_trades:,} (LONG: {r.n_long:,}, SHORT: {r.n_short:,})")
        logger.info(f"   Win Rate: {r.win_rate*100:.2f}%")
        logger.info(f"   Profit Factor: {r.profit_factor:.3f}")
        logger.info(f"   PnL Brut: {r.total_pnl*100:+.2f}%")
        logger.info(f"   PnL Net: {r.total_pnl_after_fees*100:+.2f}%")
        logger.info(f"   Frais: {r.total_fees*100:.2f}%")
        logger.info(f"   Avg Win: {r.avg_win*100:+.3f}%")
        logger.info(f"   Avg Loss: {r.avg_loss*100:+.3f}%")
        logger.info(f"   Avg Duration: {r.avg_duration:.1f} périodes")
        logger.info(f"   Sharpe Ratio: {r.sharpe_ratio:.3f}")

    # Meilleure combinaison
    best = sorted_results[0]

    logger.info(f"\n✅ MEILLEURE COMBINAISON: {best.combo.full_name}")
    logger.info(f"   Code: {best.combo.name}")
    logger.info(f"   Sharpe Ratio: {best.sharpe_ratio:.3f}")
    logger.info(f"   PnL Net: {best.total_pnl_after_fees*100:+.2f}%")
    logger.info(f"   Win Rate: {best.win_rate*100:.2f}%")

    if best.total_pnl_after_fees > 0:
        logger.info("\n🎉 COMBINAISON RENTABLE TROUVÉE!")
    else:
        logger.info("\n⚠️  Toutes combinaisons négatives.")

    logger.info("\n" + "="*120 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Test multi-indicateurs avec filtres croisés',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Split (défaut: test)')

    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limiter le nombre de samples (défaut: tous)')

    args = parser.parse_args()

    logger.info("="*120)
    logger.info("TEST MULTI-INDICATEURS - FILTRES CROISÉS")
    logger.info("="*120)
    logger.info(f"Split: {args.split}")
    logger.info(f"Holding minimum: {MIN_HOLDING} périodes (fixe)")
    logger.info(f"Fees: {FEES*100:.2f}% par side ({FEES*2*100:.2f}% round-trip)")
    logger.info("="*120 + "\n")

    # Charger datasets (2 filtres × 3 indicateurs = 6 datasets)
    logger.info("📂 Chargement des 6 datasets...\n")

    datasets = {}
    for indicator in ['macd', 'rsi', 'cci']:
        for filter_type in ['kalman', 'octave']:
            key = f"{indicator}_{filter_type}"
            logger.info(f"  Chargement {key}...")
            datasets[key] = load_dataset(indicator, filter_type, args.split)

    # Extraire returns (identique pour tous)
    returns = extract_c_ret(datasets['macd_kalman']['X'], 'macd')

    # Limiter samples si demandé
    if args.max_samples is not None and args.max_samples < len(returns):
        logger.info(f"\n  ⚠️  Limitation à {args.max_samples:,} samples (sur {len(returns):,} disponibles)")
        returns = returns[:args.max_samples]
        # Limiter aussi tous les datasets
        for key in datasets:
            datasets[key]['X'] = datasets[key]['X'][:args.max_samples]
            datasets[key]['Y'] = datasets[key]['Y'][:args.max_samples]
            if datasets[key]['Y_pred'] is not None:
                datasets[key]['Y_pred'] = datasets[key]['Y_pred'][:args.max_samples]

    logger.info(f"\n  Samples: {len(returns):,}\n")

    # Générer 8 combinaisons
    combos = []
    for macd_f, rsi_f, cci_f in product(['kalman', 'octave'], repeat=3):
        combos.append(FilterCombo(macd_f, rsi_f, cci_f))

    logger.info(f"🔧 Test de {len(combos)} combinaisons...\n")

    # Tester chaque combinaison
    results = []

    for combo in combos:
        logger.info(f"  Testing {combo.name} ({combo.full_name})...")

        # Récupérer prédictions
        pred_macd = datasets[f"macd_{combo.macd_filter}"]['Y_pred']
        pred_rsi = datasets[f"rsi_{combo.rsi_filter}"]['Y_pred']
        pred_cci = datasets[f"cci_{combo.cci_filter}"]['Y_pred']

        if pred_macd is None or pred_rsi is None or pred_cci is None:
            logger.warning(f"    ⚠️  Prédictions manquantes, skip")
            continue

        # Backtest
        result = backtest_multi_indicator(
            pred_macd, pred_rsi, pred_cci, returns, combo
        )
        results.append(result)

    # Afficher résultats
    print_results(results)


if __name__ == '__main__':
    main()
