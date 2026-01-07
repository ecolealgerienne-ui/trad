#!/usr/bin/env python3
"""
Test Confidence-Based Veto Rules - Phase 2.7

Implémente et teste les 3 règles de veto basées sur les scores de confiance:
1. Zone Grise: Bloquer si décideur conf <0.20
2. Veto Ultra-Fort: Bloquer si témoin conf >0.70 ET désaccord
3. Confirmation: Exiger témoin conf >0.50 si décideur conf 0.20-0.40

Architecture:
- MACD: Décideur principal (7.46% erreurs, 30.3% zone grise)
- RSI + CCI: Témoins avec veto (60% détection combinée)

Usage:
    python tests/test_confidence_veto.py --split test --max-samples 20000 --enable-rule1 --enable-rule2 --enable-rule3
"""

import argparse
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple
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
class StrategyResult:
    """Résultats d'une stratégie."""
    name: str
    trades: int
    win_rate: float
    pnl_brut: float
    pnl_net: float
    avg_duration: float
    sharpe: float
    max_dd: float

    # Détails règles
    rule1_blocks: int = 0  # Zone grise
    rule2_blocks: int = 0  # Veto ultra-fort
    rule3_blocks: int = 0  # Confirmation requise


def compute_confidence(prob: float) -> float:
    """
    Calcule le score de confiance d'une probabilité.

    Score = 0: très incertain (prob = 0.5)
    Score = 1: très confiant (prob = 0.0 ou 1.0)

    Args:
        prob: Probabilité brute [0.0, 1.0]

    Returns:
        Score de confiance [0.0, 1.0]

    Examples:
        prob=0.51 → confidence = 0.02 (zone grise)
        prob=0.70 → confidence = 0.40 (moyen)
        prob=0.85 → confidence = 0.70 (fort)
        prob=0.95 → confidence = 0.90 (très fort)
    """
    return abs(prob - 0.5) * 2.0


def load_multi_indicator_data(
    split: str = 'test',
    filter_type: str = 'kalman',
    max_samples: int = None
) -> Dict:
    """
    Charge les données des 3 indicateurs avec probabilités brutes.

    Args:
        split: 'train', 'val', ou 'test'
        filter_type: 'kalman' ou 'octave20'
        max_samples: Limiter le nombre de samples (None = tous)

    Returns:
        Dict avec:
        - returns: Rendements close
        - macd: {Y_oracle, Y_pred_probs}
        - rsi: {Y_oracle, Y_pred_probs}
        - cci: {Y_oracle, Y_pred_probs}
    """
    base_path = Path('data/prepared')

    # Trouver les fichiers
    pattern = f'dataset_*_dual_binary_{filter_type}.npz'
    files = list(base_path.glob(pattern))

    if not files:
        raise FileNotFoundError(f"Aucun fichier trouvé: {base_path}/{pattern}")

    # Charger chaque indicateur
    datasets = {}
    returns = None

    for indicator in ['macd', 'rsi', 'cci']:
        # Trouver fichier pour cet indicateur
        indicator_file = None
        for f in files:
            if f'_{indicator}_' in f.name:
                indicator_file = f
                break

        if not indicator_file:
            raise FileNotFoundError(f"Fichier {indicator} non trouvé")

        logger.info(f"Chargement {indicator}: {indicator_file.name}")
        data = np.load(indicator_file, allow_pickle=True)

        # Extraire données du split
        Y_oracle = data[f'Y_{split}']  # (n, 2) [direction, force]
        Y_pred = data.get(f'Y_{split}_pred', None)  # (n, 2) probabilités brutes

        if Y_pred is None:
            raise ValueError(f"Prédictions manquantes pour {indicator} ({split})")

        # Limiter samples si demandé
        if max_samples is not None and max_samples < len(Y_oracle):
            Y_oracle = Y_oracle[:max_samples]
            Y_pred = Y_pred[:max_samples]

        datasets[indicator] = {
            'Y_oracle': Y_oracle,
            'Y_pred_probs': Y_pred  # PROBABILITÉS BRUTES [0.0, 1.0]
        }

        # Charger returns (une seule fois)
        if returns is None:
            returns = data[f'returns_{split}']
            if max_samples is not None and max_samples < len(returns):
                returns = returns[:max_samples]

    n_samples = len(returns)
    logger.info(f"\n📊 Données chargées:")
    logger.info(f"  Split: {split}")
    logger.info(f"  Samples: {n_samples:,}")
    logger.info(f"  Indicateurs: MACD, RSI, CCI")

    return {
        'returns': returns,
        'macd': datasets['macd'],
        'rsi': datasets['rsi'],
        'cci': datasets['cci']
    }


def backtest_with_confidence_veto(
    returns: np.ndarray,
    macd_data: Dict,
    rsi_data: Dict,
    cci_data: Dict,
    enable_rule1: bool = False,
    enable_rule2: bool = False,
    enable_rule3: bool = False,
    holding_min: int = 5,
    fees: float = 0.1
) -> StrategyResult:
    """
    Backtest avec règles de veto basées sur confiance.

    Architecture:
    - MACD: Décideur principal
    - RSI + CCI: Témoins avec veto

    Règles:
    1. Zone Grise: Bloquer si MACD conf <0.20
    2. Veto Ultra-Fort: Bloquer si témoin conf >0.70 ET désaccord direction
    3. Confirmation: Exiger témoin conf >0.50 si MACD conf 0.20-0.40

    Args:
        returns: Rendements close
        macd_data: {Y_oracle, Y_pred_probs}
        rsi_data: {Y_oracle, Y_pred_probs}
        cci_data: {Y_oracle, Y_pred_probs}
        enable_rule1: Activer règle zone grise
        enable_rule2: Activer règle veto ultra-fort
        enable_rule3: Activer règle confirmation
        holding_min: Durée minimale de holding (périodes)
        fees: Frais par trade (%)

    Returns:
        StrategyResult avec métriques
    """
    n_samples = len(returns)

    # Extraire probabilités brutes
    macd_prob = macd_data['Y_pred_probs']  # (n, 2) [dir_prob, force_prob]
    rsi_prob = rsi_data['Y_pred_probs']
    cci_prob = cci_data['Y_pred_probs']

    # Variables trading
    position = Position.FLAT
    entry_price = 0.0
    entry_step = 0
    trades = []
    equity_curve = [1.0]

    # Compteurs blocages
    rule1_blocks = 0
    rule2_blocks = 0
    rule3_blocks = 0

    for i in range(1, n_samples):
        current_price = 1.0 + returns[i]

        # === Calculer confiances et directions ===

        # MACD (décideur)
        macd_prob_dir = macd_prob[i, 0]
        macd_prob_force = macd_prob[i, 1]
        macd_conf_dir = compute_confidence(macd_prob_dir)
        macd_conf_force = compute_confidence(macd_prob_force)
        macd_dir = 1 if macd_prob_dir > 0.5 else 0
        macd_force = 1 if macd_prob_force > 0.5 else 0

        # RSI (témoin)
        rsi_prob_dir = rsi_prob[i, 0]
        rsi_conf_dir = compute_confidence(rsi_prob_dir)
        rsi_dir = 1 if rsi_prob_dir > 0.5 else 0

        # CCI (témoin)
        cci_prob_dir = cci_prob[i, 0]
        cci_conf_dir = compute_confidence(cci_prob_dir)
        cci_dir = 1 if cci_prob_dir > 0.5 else 0

        # === Logique de décision MACD avec veto ===

        # Signal MACD baseline
        if macd_dir == 1 and macd_force == 1:
            target = Position.LONG
        elif macd_dir == 0 and macd_force == 1:
            target = Position.SHORT
        else:
            target = Position.FLAT

        # === Appliquer règles de veto ===

        veto = False

        # Règle #1: Zone Grise MACD
        if enable_rule1 and macd_conf_dir < 0.20:
            veto = True
            rule1_blocks += 1

        # Règle #2: Veto Ultra-Fort (si pas déjà veto par règle 1)
        if not veto and enable_rule2 and macd_conf_dir < 0.20:
            # RSI ultra-confiant contredit MACD faible
            if rsi_conf_dir > 0.70 and rsi_dir != macd_dir:
                veto = True
                rule2_blocks += 1
            # CCI ultra-confiant contredit MACD faible
            elif cci_conf_dir > 0.70 and cci_dir != macd_dir:
                veto = True
                rule2_blocks += 1

        # Règle #3: Confirmation Requise (si pas déjà veto)
        if not veto and enable_rule3 and 0.20 <= macd_conf_dir < 0.40:
            # Au moins un témoin doit confirmer avec conf >0.50
            has_confirmation = (
                (rsi_conf_dir > 0.50 and rsi_dir == macd_dir) or
                (cci_conf_dir > 0.50 and cci_dir == macd_dir)
            )
            if not has_confirmation:
                veto = True
                rule3_blocks += 1

        # Appliquer veto
        if veto:
            target = Position.FLAT

        # === Gestion position ===

        # Sortie sur retournement direction (bypass holding)
        if position != Position.FLAT:
            if (position == Position.LONG and macd_dir == 0) or \
               (position == Position.SHORT and macd_dir == 1):
                # Retournement direction → EXIT immédiat
                pnl = (current_price / entry_price - 1.0) if position == Position.LONG else (entry_price / current_price - 1.0)
                pnl -= fees / 100.0

                trades.append({
                    'entry': entry_step,
                    'exit': i,
                    'duration': i - entry_step,
                    'pnl': pnl,
                    'win': pnl > 0
                })

                equity_curve.append(equity_curve[-1] * (1 + pnl))
                position = Position.FLAT

        # Sortie sur Force=WEAK après holding minimum
        if position != Position.FLAT and macd_force == 0:
            duration = i - entry_step
            if duration >= holding_min:
                # Exit OK
                pnl = (current_price / entry_price - 1.0) if position == Position.LONG else (entry_price / current_price - 1.0)
                pnl -= fees / 100.0

                trades.append({
                    'entry': entry_step,
                    'exit': i,
                    'duration': duration,
                    'pnl': pnl,
                    'win': pnl > 0
                })

                equity_curve.append(equity_curve[-1] * (1 + pnl))
                position = Position.FLAT

        # Entrée si FLAT et signal valide (pas de veto)
        if position == Position.FLAT and target != Position.FLAT:
            position = target
            entry_price = current_price
            entry_step = i

    # Fermer position finale si ouverte
    if position != Position.FLAT:
        current_price = 1.0 + returns[-1]
        pnl = (current_price / entry_price - 1.0) if position == Position.LONG else (entry_price / current_price - 1.0)
        pnl -= fees / 100.0

        trades.append({
            'entry': entry_step,
            'exit': n_samples - 1,
            'duration': n_samples - 1 - entry_step,
            'pnl': pnl,
            'win': pnl > 0
        })
        equity_curve.append(equity_curve[-1] * (1 + pnl))

    # === Calcul métriques ===

    if len(trades) == 0:
        return StrategyResult(
            name="No Trades",
            trades=0,
            win_rate=0.0,
            pnl_brut=0.0,
            pnl_net=0.0,
            avg_duration=0.0,
            sharpe=0.0,
            max_dd=0.0,
            rule1_blocks=rule1_blocks,
            rule2_blocks=rule2_blocks,
            rule3_blocks=rule3_blocks
        )

    n_trades = len(trades)
    wins = [t for t in trades if t['win']]
    win_rate = len(wins) / n_trades * 100

    pnl_brut = sum(t['pnl'] + fees / 100.0 for t in trades) * 100
    pnl_net = sum(t['pnl'] for t in trades) * 100

    avg_duration = np.mean([t['duration'] for t in trades])

    # Sharpe
    returns_trades = np.array([t['pnl'] for t in trades])
    if len(returns_trades) > 1 and returns_trades.std() > 0:
        sharpe = returns_trades.mean() / returns_trades.std() * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max Drawdown
    equity = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_dd = abs(drawdown.min()) * 100

    # Nom stratégie
    rules_active = []
    if enable_rule1:
        rules_active.append("R1")
    if enable_rule2:
        rules_active.append("R2")
    if enable_rule3:
        rules_active.append("R3")

    name = "Baseline" if not rules_active else "+".join(rules_active)

    return StrategyResult(
        name=name,
        trades=n_trades,
        win_rate=win_rate,
        pnl_brut=pnl_brut,
        pnl_net=pnl_net,
        avg_duration=avg_duration,
        sharpe=sharpe,
        max_dd=max_dd,
        rule1_blocks=rule1_blocks,
        rule2_blocks=rule2_blocks,
        rule3_blocks=rule3_blocks
    )


def print_comparison(results: List[StrategyResult]):
    """Affiche comparaison des résultats."""

    logger.info("\n" + "="*120)
    logger.info("COMPARAISON STRATÉGIES - Veto Confiance")
    logger.info("="*120)

    # Header
    logger.info(f"{'Stratégie':<15} {'Trades':>8} {'Réduc':>7} {'WR':>7} {'Δ WR':>7} "
                f"{'PnL Brut':>10} {'PnL Net':>10} {'Sharpe':>7} {'Avg Dur':>8} {'Blocages (R1/R2/R3)':>20}")
    logger.info("-"*120)

    # Baseline
    baseline = results[0]
    logger.info(f"{baseline.name:<15} {baseline.trades:>8,} {'-':>7} "
                f"{baseline.win_rate:>6.2f}% {'-':>7} "
                f"{baseline.pnl_brut:>9.2f}% {baseline.pnl_net:>9.2f}% "
                f"{baseline.sharpe:>7.2f} {baseline.avg_duration:>7.1f}p {'-':>20}")

    # Autres stratégies
    for result in results[1:]:
        reduction = (baseline.trades - result.trades) / baseline.trades * 100
        delta_wr = result.win_rate - baseline.win_rate
        blocages = f"{result.rule1_blocks}/{result.rule2_blocks}/{result.rule3_blocks}"

        logger.info(f"{result.name:<15} {result.trades:>8,} {reduction:>6.1f}% "
                    f"{result.win_rate:>6.2f}% {delta_wr:>+6.2f}% "
                    f"{result.pnl_brut:>9.2f}% {result.pnl_net:>9.2f}% "
                    f"{result.sharpe:>7.2f} {result.avg_duration:>7.1f}p {blocages:>20}")

    logger.info("="*120)

    # Légende
    logger.info("\n📖 LÉGENDE:")
    logger.info("  Réduc: Réduction trades vs Baseline")
    logger.info("  Δ WR: Changement Win Rate vs Baseline")
    logger.info("  Blocages: R1 = Zone Grise | R2 = Veto Ultra-Fort | R3 = Confirmation")
    logger.info("  Avg Dur: Durée moyenne trade (périodes)")

    # Analyse
    best = max(results[1:], key=lambda r: r.pnl_net, default=None)

    if best and best.pnl_net > baseline.pnl_net:
        improvement = best.pnl_net - baseline.pnl_net
        logger.info(f"\n✅ MEILLEURE STRATÉGIE: {best.name}")
        logger.info(f"  PnL Net: {baseline.pnl_net:.2f}% → {best.pnl_net:.2f}% (+{improvement:.2f}%)")
        logger.info(f"  Trades: {baseline.trades:,} → {best.trades:,} ({(best.trades - baseline.trades) / baseline.trades * 100:+.1f}%)")
        logger.info(f"  Win Rate: {baseline.win_rate:.2f}% → {best.win_rate:.2f}% ({best.win_rate - baseline.win_rate:+.2f}%)")

        if best.pnl_net > 0:
            logger.info(f"\n🎉 PnL NET POSITIF! (+{best.pnl_net:.2f}%)")
        else:
            logger.info(f"\n⚠️  PnL Net encore négatif ({best.pnl_net:.2f}%), mais amélioration de +{improvement:.2f}%")
    else:
        logger.info(f"\n❌ Aucune amélioration vs Baseline")


def main():
    parser = argparse.ArgumentParser(description='Test Confidence-Based Veto Rules')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Split de données (défaut: test)')
    parser.add_argument('--filter', type=str, default='kalman', choices=['kalman', 'octave20'],
                        help='Type de filtre (défaut: kalman)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limiter le nombre de samples (défaut: tous)')
    parser.add_argument('--holding-min', type=int, default=5,
                        help='Holding minimum (périodes, défaut: 5)')
    parser.add_argument('--fees', type=float, default=0.1,
                        help='Frais par trade % (défaut: 0.1)')

    # Règles
    parser.add_argument('--enable-rule1', action='store_true',
                        help='Activer Règle #1: Zone Grise (conf <0.20)')
    parser.add_argument('--enable-rule2', action='store_true',
                        help='Activer Règle #2: Veto Ultra-Fort (témoin >0.70)')
    parser.add_argument('--enable-rule3', action='store_true',
                        help='Activer Règle #3: Confirmation (témoin >0.50)')
    parser.add_argument('--enable-all', action='store_true',
                        help='Activer toutes les règles')

    args = parser.parse_args()

    # Activer toutes les règles si demandé
    if args.enable_all:
        args.enable_rule1 = True
        args.enable_rule2 = True
        args.enable_rule3 = True

    logger.info("="*120)
    logger.info("TEST CONFIDENCE-BASED VETO - Phase 2.7")
    logger.info("="*120)
    logger.info(f"\n⚙️  CONFIGURATION:")
    logger.info(f"  Split: {args.split}")
    logger.add_argument('--filter', type=str, default='kalman')
    logger.info(f"  Holding Min: {args.holding_min}p")
    logger.info(f"  Frais: {args.fees}%")
    logger.info(f"\n🎯 RÈGLES ACTIVÉES:")
    logger.info(f"  Règle #1 (Zone Grise): {'✅' if args.enable_rule1 else '❌'}")
    logger.info(f"  Règle #2 (Veto Ultra-Fort): {'✅' if args.enable_rule2 else '❌'}")
    logger.info(f"  Règle #3 (Confirmation): {'✅' if args.enable_rule3 else '❌'}")

    # Charger données
    try:
        data = load_multi_indicator_data(
            split=args.split,
            filter_type=args.filter,
            max_samples=args.max_samples
        )
    except Exception as e:
        logger.error(f"❌ Erreur chargement données: {e}")
        return 1

    returns = data['returns']
    macd_data = data['macd']
    rsi_data = data['rsi']
    cci_data = data['cci']

    # Test configurations
    results = []

    # Baseline (aucune règle)
    logger.info("\n🔄 Test Baseline (aucune règle)...")
    baseline = backtest_with_confidence_veto(
        returns, macd_data, rsi_data, cci_data,
        enable_rule1=False,
        enable_rule2=False,
        enable_rule3=False,
        holding_min=args.holding_min,
        fees=args.fees
    )
    results.append(baseline)

    # Configuration demandée
    if args.enable_rule1 or args.enable_rule2 or args.enable_rule3:
        logger.info(f"\n🔄 Test avec règles activées...")
        custom = backtest_with_confidence_veto(
            returns, macd_data, rsi_data, cci_data,
            enable_rule1=args.enable_rule1,
            enable_rule2=args.enable_rule2,
            enable_rule3=args.enable_rule3,
            holding_min=args.holding_min,
            fees=args.fees
        )
        results.append(custom)

    # Tester aussi les règles individuellement (si --enable-all)
    if args.enable_all:
        logger.info(f"\n🔄 Test Règle #1 seule...")
        r1_only = backtest_with_confidence_veto(
            returns, macd_data, rsi_data, cci_data,
            enable_rule1=True,
            enable_rule2=False,
            enable_rule3=False,
            holding_min=args.holding_min,
            fees=args.fees
        )
        results.append(r1_only)

        logger.info(f"\n🔄 Test Règles #1+#2...")
        r1_r2 = backtest_with_confidence_veto(
            returns, macd_data, rsi_data, cci_data,
            enable_rule1=True,
            enable_rule2=True,
            enable_rule3=False,
            holding_min=args.holding_min,
            fees=args.fees
        )
        results.append(r1_r2)

    # Afficher résultats
    print_comparison(results)

    logger.info(f"\n✅ Test terminé!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
