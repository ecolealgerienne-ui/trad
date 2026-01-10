#!/usr/bin/env python3
"""
Test Confidence-Based Veto Rules - Phase 2.7 v2.0

REWRITTEN: Utilise la bonne méthode de trading (Open prices, per-asset backtest)

Implémente et teste les 3 règles de veto basées sur les scores de confiance:
1. Zone Grise: Bloquer si décideur conf <0.20
2. Veto Ultra-Fort: Bloquer si témoin conf >0.70 ET désaccord
3. Confirmation: Exiger témoin conf >0.50 si décideur conf 0.20-0.40

Architecture:
- MACD: Décideur principal
- RSI + CCI: Témoins avec veto

Corrections v2.0:
- Utilise Open prices (OHLCV[:, 2]) au lieu de c_ret
- PnL = (exit_price - entry_price) / entry_price
- Backtest per-asset (évite erreurs cross-asset)
- Dataset direction_only au lieu de dual_binary

Usage:
    python tests/test_confidence_veto.py --split test --enable-all
"""

import argparse
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Tuple, Optional
from datetime import datetime
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
    exit_reason: str


@dataclass
class AssetResult:
    """Résultats pour un asset."""
    asset_id: int
    trades: List[Trade]
    pnl_gross: float
    pnl_net: float
    win_rate: float
    n_trades: int
    fees_paid: float


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
    max_dd: float
    fees_paid: float

    # Résultats per-asset
    per_asset: Dict[int, AssetResult] = field(default_factory=dict)

    # Détails règles
    rule1_blocks: int = 0  # Zone grise
    rule2_blocks: int = 0  # Veto ultra-fort
    rule3_blocks: int = 0  # Confirmation requise


def compute_confidence(prob: float) -> float:
    """
    Calcule le score de confiance d'une probabilité.

    Score = 0: très incertain (prob = 0.5)
    Score = 1: très confiant (prob = 0.0 ou 1.0)
    """
    return abs(prob - 0.5) * 2.0


def load_direction_only_data(
    indicator: str,
    split: str = 'test',
    filter_type: str = 'kalman'
) -> Dict:
    """
    Charge les données direction-only pour un indicateur.

    Returns:
        Dict avec:
        - Y: labels direction (n, 3) = [timestamp, asset_id, direction]
        - OHLCV: (n, 7) = [timestamp, asset_id, O, H, L, C, V]
        - Y_pred: probabilités prédites (n,)
    """
    base_path = Path('data/prepared')

    # Trouver le fichier direction-only
    pattern = f'dataset_*_{indicator}_direction_only_{filter_type}.npz'
    files = list(base_path.glob(pattern))

    if not files:
        raise FileNotFoundError(f"Fichier non trouvé: {base_path}/{pattern}")

    data_file = files[0]
    logger.info(f"  Chargement {indicator}: {data_file.name}")

    data = np.load(data_file, allow_pickle=True)

    Y = data[f'Y_{split}']        # (n, 3) [timestamp, asset_id, direction]
    OHLCV = data[f'OHLCV_{split}']  # (n, 7) [timestamp, asset_id, O, H, L, C, V]

    # Charger prédictions si disponibles
    Y_pred = data.get(f'Y_{split}_pred', None)
    if Y_pred is not None:
        # Y_pred peut être (n,) ou (n, 1)
        if Y_pred.ndim == 2:
            Y_pred = Y_pred[:, 0]

    return {
        'Y': Y,
        'OHLCV': OHLCV,
        'Y_pred': Y_pred
    }


def load_multi_indicator_data(
    split: str = 'test',
    filter_type: str = 'kalman'
) -> Dict:
    """
    Charge les données des 3 indicateurs (direction-only).

    Returns:
        Dict avec données pour macd, rsi, cci
    """
    logger.info(f"\n📊 Chargement données {split}...")

    datasets = {}

    for indicator in ['macd', 'rsi', 'cci']:
        try:
            datasets[indicator] = load_direction_only_data(
                indicator=indicator,
                split=split,
                filter_type=filter_type
            )
        except FileNotFoundError as e:
            logger.warning(f"  ⚠️ {indicator}: {e}")
            datasets[indicator] = None

    # Vérifier qu'on a au moins MACD
    if datasets['macd'] is None:
        raise FileNotFoundError("MACD requis comme décideur principal")

    n_samples = len(datasets['macd']['Y'])
    logger.info(f"  Total samples: {n_samples:,}")

    return datasets


def backtest_single_asset(
    # Labels et prix pour tous les indicateurs
    macd_labels: np.ndarray,
    macd_preds: Optional[np.ndarray],
    rsi_preds: Optional[np.ndarray],
    cci_preds: Optional[np.ndarray],
    opens: np.ndarray,
    timestamps: np.ndarray,
    asset_id: int,
    # Paramètres stratégie
    enable_rule1: bool = False,
    enable_rule2: bool = False,
    enable_rule3: bool = False,
    holding_min: int = 0,
    fees: float = 0.001
) -> Tuple[List[Trade], Dict]:
    """
    Backtest sur un seul asset avec règles de veto.

    Logique causale:
    - Signal à l'index i → Exécution à Open[i+1]
    - Direction: 1=UP→LONG, 0=DOWN→SHORT

    Règles de veto (appliquées SEULEMENT sur les entrées):
    1. Zone Grise: Bloquer si MACD conf < 0.20
    2. Veto Ultra-Fort: Bloquer si témoin conf > 0.70 ET désaccord
    3. Confirmation: Exiger témoin conf > 0.50 si MACD conf 0.20-0.40

    Returns:
        (trades, stats_dict)
    """
    n_samples = len(macd_labels)

    if n_samples < 2:
        return [], {'rule1': 0, 'rule2': 0, 'rule3': 0}

    # Variables de trading
    position = Position.FLAT
    entry_idx = 0
    entry_price = 0.0

    trades = []
    rule1_blocks = 0
    rule2_blocks = 0
    rule3_blocks = 0

    for i in range(n_samples - 1):
        # Direction MACD (labels)
        macd_dir = int(macd_labels[i])  # 1=UP, 0=DOWN
        target = Position.LONG if macd_dir == 1 else Position.SHORT

        # === Calculer confiances (si prédictions disponibles) ===
        macd_conf = 0.5  # défaut: incertain
        rsi_conf = 0.5
        cci_conf = 0.5
        rsi_dir = macd_dir  # défaut: même direction
        cci_dir = macd_dir

        if macd_preds is not None:
            macd_conf = compute_confidence(macd_preds[i])

        if rsi_preds is not None:
            rsi_conf = compute_confidence(rsi_preds[i])
            rsi_dir = 1 if rsi_preds[i] > 0.5 else 0

        if cci_preds is not None:
            cci_conf = compute_confidence(cci_preds[i])
            cci_dir = 1 if cci_preds[i] > 0.5 else 0

        # === Appliquer règles de veto (SEULEMENT sur entrées) ===
        veto = False

        if position == Position.FLAT:
            # Règle #1: Zone Grise MACD
            if enable_rule1 and macd_conf < 0.20:
                veto = True
                rule1_blocks += 1

            # Règle #2: Veto Ultra-Fort (si pas déjà bloqué)
            if not veto and enable_rule2 and macd_conf < 0.20:
                # RSI ultra-confiant contredit MACD
                if rsi_conf > 0.70 and rsi_dir != macd_dir:
                    veto = True
                    rule2_blocks += 1
                # CCI ultra-confiant contredit MACD
                elif cci_conf > 0.70 and cci_dir != macd_dir:
                    veto = True
                    rule2_blocks += 1

            # Règle #3: Confirmation requise (si pas déjà bloqué)
            if not veto and enable_rule3 and 0.20 <= macd_conf < 0.40:
                # Au moins un témoin doit confirmer avec conf > 0.50
                has_confirmation = (
                    (rsi_conf > 0.50 and rsi_dir == macd_dir) or
                    (cci_conf > 0.50 and cci_dir == macd_dir)
                )
                if not has_confirmation:
                    veto = True
                    rule3_blocks += 1

        # === Gestion de position ===

        trade_duration = i - entry_idx if position != Position.FLAT else 0

        # Cas 1: Sortie sur DIRECTION_FLIP
        if position != Position.FLAT and target != position:
            # Vérifier holding minimum
            if trade_duration >= holding_min:
                exit_price = opens[i + 1]
                pnl_gross = (exit_price - entry_price) / entry_price
                if position == Position.SHORT:
                    pnl_gross = -pnl_gross

                fee = fees * 2  # Entrée + sortie
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
                    duration=trade_duration,
                    exit_reason="DIRECTION_FLIP"
                ))

                # Flip immédiat (pas de veto sur les flips!)
                position = target
                entry_idx = i
                entry_price = opens[i + 1]

        # Cas 2: Entrée depuis FLAT (avec veto possible)
        elif position == Position.FLAT:
            if not veto:
                position = target
                entry_idx = i
                entry_price = opens[i + 1]

    # Fermer position finale
    if position != Position.FLAT:
        exit_idx = n_samples - 1
        exit_price = opens[exit_idx]
        trade_duration = exit_idx - entry_idx

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
            duration=trade_duration,
            exit_reason="END_OF_DATA"
        ))

    stats = {
        'rule1': rule1_blocks,
        'rule2': rule2_blocks,
        'rule3': rule3_blocks
    }

    return trades, stats


def backtest_with_confidence_veto(
    datasets: Dict,
    enable_rule1: bool = False,
    enable_rule2: bool = False,
    enable_rule3: bool = False,
    holding_min: int = 0,
    fees: float = 0.001
) -> StrategyResult:
    """
    Backtest multi-indicateurs avec règles de veto.

    Effectue un backtest PAR ASSET puis agrège les résultats.
    """
    macd_data = datasets['macd']
    rsi_data = datasets.get('rsi')
    cci_data = datasets.get('cci')

    # Extraire données MACD (référence)
    Y_macd = macd_data['Y']  # (n, 3) [timestamp, asset_id, direction]
    OHLCV = macd_data['OHLCV']  # (n, 7) [timestamp, asset_id, O, H, L, C, V]
    macd_preds = macd_data.get('Y_pred')

    # Extraire prédictions RSI/CCI si disponibles
    rsi_preds = rsi_data['Y_pred'] if rsi_data and rsi_data.get('Y_pred') is not None else None
    cci_preds = cci_data['Y_pred'] if cci_data and cci_data.get('Y_pred') is not None else None

    # Identifier les assets uniques
    asset_ids = np.unique(OHLCV[:, 1].astype(int))
    logger.info(f"  Assets trouvés: {len(asset_ids)} ({asset_ids.tolist()})")

    # Backtest per-asset
    all_trades = []
    per_asset_results = {}
    total_rule1 = 0
    total_rule2 = 0
    total_rule3 = 0

    for asset_id in asset_ids:
        # Masque pour cet asset
        mask = OHLCV[:, 1].astype(int) == asset_id

        # Extraire données asset
        asset_labels = Y_macd[mask, 2]  # direction
        asset_opens = OHLCV[mask, 2]    # Open prices
        asset_timestamps = OHLCV[mask, 0]

        # Prédictions pour cet asset
        asset_macd_preds = macd_preds[mask] if macd_preds is not None else None
        asset_rsi_preds = rsi_preds[mask] if rsi_preds is not None else None
        asset_cci_preds = cci_preds[mask] if cci_preds is not None else None

        # Backtest
        trades, stats = backtest_single_asset(
            macd_labels=asset_labels,
            macd_preds=asset_macd_preds,
            rsi_preds=asset_rsi_preds,
            cci_preds=asset_cci_preds,
            opens=asset_opens,
            timestamps=asset_timestamps,
            asset_id=asset_id,
            enable_rule1=enable_rule1,
            enable_rule2=enable_rule2,
            enable_rule3=enable_rule3,
            holding_min=holding_min,
            fees=fees
        )

        all_trades.extend(trades)
        total_rule1 += stats['rule1']
        total_rule2 += stats['rule2']
        total_rule3 += stats['rule3']

        # Résultats per-asset
        if trades:
            pnl_gross = sum(t.pnl_gross for t in trades) * 100
            pnl_net = sum(t.pnl_net for t in trades) * 100
            fees_paid = len(trades) * fees * 2 * 100
            wins = sum(1 for t in trades if t.pnl_net > 0)
            win_rate = wins / len(trades) * 100

            per_asset_results[asset_id] = AssetResult(
                asset_id=asset_id,
                trades=trades,
                pnl_gross=pnl_gross,
                pnl_net=pnl_net,
                win_rate=win_rate,
                n_trades=len(trades),
                fees_paid=fees_paid
            )

    # === Calcul métriques globales ===

    if not all_trades:
        return StrategyResult(
            name="No Trades",
            total_trades=0,
            win_rate=0.0,
            pnl_gross=0.0,
            pnl_net=0.0,
            avg_duration=0.0,
            sharpe=0.0,
            max_dd=0.0,
            fees_paid=0.0,
            per_asset={},
            rule1_blocks=total_rule1,
            rule2_blocks=total_rule2,
            rule3_blocks=total_rule3
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

    # Max Drawdown
    equity = [1.0]
    for t in all_trades:
        equity.append(equity[-1] * (1 + t.pnl_net))
    equity = np.array(equity)
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
        total_trades=n_trades,
        win_rate=win_rate,
        pnl_gross=pnl_gross,
        pnl_net=pnl_net,
        avg_duration=avg_duration,
        sharpe=sharpe,
        max_dd=max_dd,
        fees_paid=fees_paid,
        per_asset=per_asset_results,
        rule1_blocks=total_rule1,
        rule2_blocks=total_rule2,
        rule3_blocks=total_rule3
    )


def print_comparison(results: List[StrategyResult]):
    """Affiche comparaison des résultats."""

    logger.info("\n" + "="*130)
    logger.info("COMPARAISON STRATÉGIES - Veto Confiance v2.0")
    logger.info("="*130)

    # Header
    logger.info(f"{'Stratégie':<15} {'Trades':>8} {'Réduc':>7} {'WR':>7} {'Δ WR':>7} "
                f"{'PnL Brut':>12} {'PnL Net':>12} {'Frais':>10} {'Sharpe':>8} {'Blocages (R1/R2/R3)':>22}")
    logger.info("-"*130)

    baseline = results[0]

    for i, result in enumerate(results):
        if i == 0:
            reduction = "-"
            delta_wr = "-"
        else:
            reduction = f"{(baseline.total_trades - result.total_trades) / baseline.total_trades * 100:.1f}%"
            delta_wr = f"{result.win_rate - baseline.win_rate:+.2f}%"

        blocages = f"{result.rule1_blocks}/{result.rule2_blocks}/{result.rule3_blocks}"

        logger.info(f"{result.name:<15} {result.total_trades:>8,} {reduction:>7} "
                    f"{result.win_rate:>6.2f}% {delta_wr:>7} "
                    f"{result.pnl_gross:>+11.2f}% {result.pnl_net:>+11.2f}% "
                    f"{result.fees_paid:>9.2f}% {result.sharpe:>8.2f} {blocages:>22}")

    logger.info("="*130)

    # Per-asset breakdown
    logger.info("\n📊 RÉSULTATS PAR ASSET:")
    logger.info("-"*100)
    logger.info(f"{'Asset':<8} {'Trades':>8} {'Win Rate':>10} {'PnL Brut':>12} {'PnL Net':>12} {'Frais':>10}")
    logger.info("-"*100)

    # Afficher pour la meilleure stratégie (dernière)
    best_result = results[-1] if len(results) > 1 else results[0]

    for asset_id, asset_result in sorted(best_result.per_asset.items()):
        status = "✅" if asset_result.pnl_net > 0 else "❌"
        logger.info(f"  {status} {asset_id:<5} {asset_result.n_trades:>8,} "
                    f"{asset_result.win_rate:>9.2f}% "
                    f"{asset_result.pnl_gross:>+11.2f}% "
                    f"{asset_result.pnl_net:>+11.2f}% "
                    f"{asset_result.fees_paid:>9.2f}%")

    logger.info("-"*100)

    # Analyse
    logger.info("\n📖 LÉGENDE:")
    logger.info("  Réduc: Réduction trades vs Baseline")
    logger.info("  Δ WR: Changement Win Rate vs Baseline")
    logger.info("  Blocages: R1 = Zone Grise | R2 = Veto Ultra-Fort | R3 = Confirmation")

    if len(results) > 1:
        best = results[-1]
        improvement = best.pnl_net - baseline.pnl_net

        if improvement > 0:
            logger.info(f"\n✅ AMÉLIORATION avec {best.name}:")
            logger.info(f"  PnL Net: {baseline.pnl_net:+.2f}% → {best.pnl_net:+.2f}% ({improvement:+.2f}%)")
        else:
            logger.info(f"\n⚠️ Pas d'amélioration significative avec les règles de veto")


def main():
    parser = argparse.ArgumentParser(
        description='Test Confidence-Based Veto Rules v2.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Baseline (aucune règle)
  python tests/test_confidence_veto.py --split test

  # Avec toutes les règles
  python tests/test_confidence_veto.py --split test --enable-all

  # Avec holding minimum
  python tests/test_confidence_veto.py --split test --enable-all --holding-min 10
        """
    )

    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Split de données (défaut: test)')
    parser.add_argument('--filter', type=str, default='kalman',
                        choices=['kalman', 'octave20'],
                        help='Type de filtre (défaut: kalman)')
    parser.add_argument('--holding-min', type=int, default=0,
                        help='Holding minimum (périodes, défaut: 0)')
    parser.add_argument('--fees', type=float, default=0.001,
                        help='Frais par trade (défaut: 0.001 = 0.1%%)')

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

    logger.info("="*130)
    logger.info("TEST CONFIDENCE-BASED VETO v2.0 - Phase 2.7")
    logger.info("="*130)
    logger.info(f"\n⚙️  CONFIGURATION:")
    logger.info(f"  Split: {args.split}")
    logger.info(f"  Filter: {args.filter}")
    logger.info(f"  Holding Min: {args.holding_min}p")
    logger.info(f"  Frais: {args.fees*100:.2f}% par trade")
    logger.info(f"\n🎯 RÈGLES ACTIVÉES:")
    logger.info(f"  Règle #1 (Zone Grise): {'✅' if args.enable_rule1 else '❌'}")
    logger.info(f"  Règle #2 (Veto Ultra-Fort): {'✅' if args.enable_rule2 else '❌'}")
    logger.info(f"  Règle #3 (Confirmation): {'✅' if args.enable_rule3 else '❌'}")

    # Charger données
    try:
        datasets = load_multi_indicator_data(
            split=args.split,
            filter_type=args.filter
        )
    except Exception as e:
        logger.error(f"❌ Erreur chargement données: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test configurations
    results = []

    # Baseline (aucune règle)
    logger.info("\n🔄 Test Baseline (aucune règle)...")
    baseline = backtest_with_confidence_veto(
        datasets,
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
            datasets,
            enable_rule1=args.enable_rule1,
            enable_rule2=args.enable_rule2,
            enable_rule3=args.enable_rule3,
            holding_min=args.holding_min,
            fees=args.fees
        )
        results.append(custom)

    # Afficher résultats
    print_comparison(results)

    logger.info(f"\n✅ Test terminé!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
