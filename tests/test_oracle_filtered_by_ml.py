#!/usr/bin/env python3
"""
Test Diagnostique: Oracle Filtré par Consensus ML

OBJECTIF:
Mesurer où le modèle ML se trompe en testant Oracle sur 2 zones distinctes:

Test 1: Oracle SI Prédictions ML CONSENSUS (tous 6 d'accord)
        → Trade avec labels Oracle UNIQUEMENT si ML a consensus
        → Mesure: Oracle est-il bon sur les zones où ML est confiant?

Test 2: Oracle SI Prédictions ML DÉSACCORD (pas tous d'accord)
        → Trade avec labels Oracle UNIQUEMENT si ML n'a PAS consensus
        → Mesure: Oracle est-il bon sur les zones où ML est incertain?

HYPOTHÈSES À VALIDER:
H1: Consensus ML = bonnes zones → Test 1 devrait être très performant
H2: Désaccord ML = mauvaises zones → Test 2 devrait être moins performant

Si H1 fausse: Le consensus ML filtre MAL (filtre les bonnes zones!)

Usage:
    python tests/test_oracle_filtered_by_ml.py --split test --fees 0.001
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import argparse
from typing import Dict, List
import logging

# Réutiliser trading_utils validé
from trading_utils import (
    Position, Trade, StrategyResult,
    load_dataset, extract_c_ret,
    compute_pnl_step, execute_exit,
    compute_trading_stats
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CHARGEMENT MULTI-INDICATEURS MULTI-FILTRES
# =============================================================================

def load_all_datasets(split: str = 'test') -> Dict:
    """
    Charge les 3 indicateurs × 2 filtres = 6 datasets.
    (Identique au script précédent)
    """
    base_path = Path('data/prepared')
    results = {}

    logger.info("📂 Chargement datasets...")

    # Liste des combinaisons
    combinations = [
        ('macd', 'kalman'),
        ('macd', 'octave20'),
        ('rsi', 'kalman'),
        ('rsi', 'octave20'),
        ('cci', 'kalman'),
        ('cci', 'octave20'),
    ]

    for indicator, filter_type in combinations:
        filename = f'dataset_btc_eth_bnb_ada_ltc_{indicator}_dual_binary_{filter_type}.npz'
        path = base_path / filename

        if not path.exists():
            logger.warning(f"⚠️  Dataset non trouvé: {filename}")
            results[f'{indicator}_{filter_type}'] = None
            continue

        logger.info(f"   Loading {indicator}_{filter_type}...")
        data = np.load(path, allow_pickle=True)

        key = f'{indicator}_{filter_type}'
        results[key] = {
            'Y': data[f'Y_{split}'],
            'Y_pred': data.get(f'Y_{split}_pred', None),
            'X': data[f'X_{split}']
        }

    # Vérifier qu'au moins un dataset existe
    valid_keys = [k for k, v in results.items() if v is not None]
    if not valid_keys:
        raise FileNotFoundError("Aucun dataset trouvé!")

    # Extraire returns
    if results.get('macd_kalman') is not None:
        X = results['macd_kalman']['X']
        indicator = 'macd'
    elif results.get('rsi_kalman') is not None:
        X = results['rsi_kalman']['X']
        indicator = 'rsi'
    else:
        first_key = valid_keys[0]
        X = results[first_key]['X']
        indicator = first_key.split('_')[0]

    results['returns'] = extract_c_ret(X, indicator)

    logger.info(f"✅ {len(valid_keys)}/6 datasets chargés")
    logger.info(f"   Samples: {len(results['returns']):,}")

    return results


# =============================================================================
# BACKTEST ORACLE FILTRÉ PAR ML
# =============================================================================

def backtest_oracle_filtered_by_ml(
    data: Dict,
    fees: float = 0.001,
    filter_mode: str = 'consensus',
    min_agreement: int = 6,
    signal_type: str = 'direction',
    indicator: str = None,
    use_force_filter: bool = False
) -> StrategyResult:
    """
    Backtest Oracle filtré par consensus ML paramétrable.

    Args:
        data: Dict retourné par load_all_datasets()
        fees: Frais par side
        filter_mode: 'consensus' ou 'disagreement'
            - 'consensus': Trade UNIQUEMENT si >= min_agreement signaux d'accord
            - 'disagreement': Trade UNIQUEMENT si < min_agreement signaux d'accord
        min_agreement: Nombre minimum de signaux devant être d'accord
            - Si indicator=None: 1-6 (6 signaux = 3 indicateurs × 2 filtres)
            - Si indicator='macd': 1-2 (2 signaux = 1 indicateur × 2 filtres)
        signal_type: Type de signal à tester ('direction' ou 'force')
            - 'direction': Colonne 0 (pente UP/DOWN)
            - 'force': Colonne 1 (vélocité WEAK/STRONG)
        indicator: Indicateur unique à tester (None, 'macd', 'rsi', 'cci')
            - None: Tous les indicateurs (6 signaux)
            - 'macd'/'rsi'/'cci': Un seul indicateur (2 signaux)
        use_force_filter: Si True, ajoute filtre Force STRONG (nécessite signal_type='direction')
            - Trade UNIQUEMENT si Direction consensus ET Force STRONG

    Returns:
        StrategyResult

    Logique:
        1. Charger signaux selon indicator (None=tous, ou indicateur unique)
        2. Compter consensus Direction
        3. Si use_force_filter: Vérifier Force STRONG aussi
        4. Trader avec Oracle labels si autorisé
    """
    returns = data['returns']
    n_samples = len(returns)

    # Extraire prédictions ET labels (Oracle)
    predictions_dir = {}  # Direction
    predictions_force = {}  # Force (si use_force_filter)
    labels_dir = {}  # Direction Oracle
    available_signals = []

    # Déterminer quels indicateurs charger
    if indicator is None:
        indicators_to_load = ['macd', 'rsi', 'cci']
    else:
        indicators_to_load = [indicator]

    # Charger Direction (toujours)
    for ind in indicators_to_load:
        for filter_type in ['kalman', 'octave20']:
            key = f'{ind}_{filter_type}'
            if data.get(key) is not None and data[key]['Y_pred'] is not None:
                # Direction: Prédictions ML (pour filtrer)
                pred = data[key]['Y_pred']
                predictions_dir[key] = (pred[:, 0] > 0.5).astype(int)

                # Direction: Labels Oracle (pour trader)
                labels_dir[key] = data[key]['Y'][:, 0].astype(int)

                # Force: Charger si demandé
                if use_force_filter:
                    predictions_force[key] = (pred[:, 1] > 0.5).astype(int)

                available_signals.append(key)

    logger.info(f"Signaux disponibles: {len(available_signals)}/6")
    logger.info(f"   {', '.join(available_signals)}")

    if len(available_signals) == 0:
        raise ValueError("Aucun signal avec prédictions disponible!")

    # État trading
    position = Position.FLAT
    entry_time = 0
    current_pnl = 0.0
    trades = []
    n_long = 0
    n_short = 0

    # Compteurs zones
    n_consensus_zones = 0
    n_disagreement_zones = 0
    n_trades_in_zone = 0

    for i in range(n_samples):
        ret = returns[i]

        # Accumuler PnL
        current_pnl = compute_pnl_step(position, ret, current_pnl)

        # === ÉTAPE 1: Vérifier consensus Direction ML ===
        pred_directions = [predictions_dir[key][i] for key in available_signals]

        # Compter signaux UP et DOWN
        n_up = sum(d == 1 for d in pred_directions)
        n_down = sum(d == 0 for d in pred_directions)
        n_total = len(pred_directions)

        # Consensus Direction ML selon seuil min_agreement
        ml_has_consensus = (n_up >= min_agreement) or (n_down >= min_agreement)

        if ml_has_consensus:
            n_consensus_zones += 1
        else:
            n_disagreement_zones += 1

        # === ÉTAPE 2: Vérifier Force si demandé ===
        force_ok = True
        if use_force_filter:
            # Force: UNIQUEMENT si consensus STRONG (1)
            pred_forces = [predictions_force[key][i] for key in available_signals]
            n_strong = sum(f == 1 for f in pred_forces)
            # Force OK si au moins min_agreement signaux STRONG
            force_ok = (n_strong >= min_agreement)

        # === ÉTAPE 3: Appliquer filtre selon mode ===
        if filter_mode == 'consensus':
            # Trade si Direction consensus ET Force OK
            trade_allowed = ml_has_consensus and force_ok
        elif filter_mode == 'disagreement':
            # Trade si Direction désaccord ET Force OK
            trade_allowed = (not ml_has_consensus) and force_ok
        else:
            raise ValueError(f"Mode inconnu: {filter_mode}")

        # === ÉTAPE 4: Décider direction avec ORACLE si autorisé ===
        if trade_allowed:
            n_trades_in_zone += 1

            # Vérifier consensus ORACLE Direction (labels)
            oracle_directions = [labels_dir[key][i] for key in available_signals]

            if all(d == 1 for d in oracle_directions):
                target = Position.LONG
            elif all(d == 0 for d in oracle_directions):
                target = Position.SHORT
            else:
                target = Position.FLAT
        else:
            # Pas autorisé à trader dans cette zone
            target = Position.FLAT

        # === ÉTAPE 4: Gestion position (identique) ===
        exit_signal = False
        exit_reason = None

        if position != Position.FLAT and target != position:
            exit_signal = True
            if target == Position.FLAT:
                exit_reason = "ORACLE_FLAT"
            else:
                exit_reason = "DIRECTION_FLIP"

        # Exécuter sortie
        if exit_signal:
            execute_exit(
                trades, entry_time, i, position,
                current_pnl, exit_reason, fees
            )

            # Gérer sortie
            if exit_reason == "ORACLE_FLAT":
                position = Position.FLAT
                current_pnl = 0.0
            elif exit_reason == "DIRECTION_FLIP":
                position = target
                entry_time = i
                current_pnl = 0.0

                if target == Position.LONG:
                    n_long += 1
                else:
                    n_short += 1

        # ENTRÉE si FLAT et signal valide
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
        execute_exit(
            trades, entry_time, n_samples - 1, position,
            current_pnl, "END_OF_DATA", fees
        )

    # Statistiques zones
    logger.info(f"\n📊 Distribution Zones:")
    logger.info(f"   Consensus ML: {n_consensus_zones:,} ({n_consensus_zones/n_samples*100:.1f}%)")
    logger.info(f"   Désaccord ML: {n_disagreement_zones:,} ({n_disagreement_zones/n_samples*100:.1f}%)")
    logger.info(f"   Samples tradés: {n_trades_in_zone:,} ({n_trades_in_zone/n_samples*100:.1f}%)")

    # Calculer stats
    mode_name = "Consensus ML" if filter_mode == 'consensus' else "Désaccord ML"
    return compute_trading_stats(
        trades=trades,
        n_long=n_long,
        n_short=n_short,
        strategy_name=f"Oracle filtré par {mode_name}",
        extra_metrics={
            'n_signaux': len(available_signals),
            'filter_mode': filter_mode,
            'n_consensus_zones': n_consensus_zones,
            'n_disagreement_zones': n_disagreement_zones,
            'n_samples_traded': n_trades_in_zone,
            'zone_coverage': n_trades_in_zone / n_samples * 100
        }
    )


# =============================================================================
# AFFICHAGE COMPARATIF
# =============================================================================

def print_diagnostic_results(
    test1: StrategyResult,
    test2: StrategyResult,
    baseline_oracle: StrategyResult
):
    """Affiche résultats diagnostiques."""
    logger.info("\n" + "="*100)
    logger.info("DIAGNOSTIC: Où le Modèle ML Se Trompe-t-il?")
    logger.info("="*100)
    logger.info(f"{'Métrique':<30} {'Oracle Baseline':>20} {'Test 1 (Consensus)':>20} {'Test 2 (Désaccord)':>20}")
    logger.info("-"*100)

    # Métriques principales
    metrics = [
        ('Samples tradés', f"{baseline_oracle.extra_metrics.get('n_samples_traded', 0):,}",
         f"{test1.extra_metrics['n_samples_traded']:,}", f"{test2.extra_metrics['n_samples_traded']:,}"),
        ('Coverage', f"-", f"{test1.extra_metrics['zone_coverage']:.1f}%", f"{test2.extra_metrics['zone_coverage']:.1f}%"),
        ('Trades', baseline_oracle.n_trades, test1.n_trades, test2.n_trades),
        ('Win Rate', f"{baseline_oracle.win_rate*100:.2f}%", f"{test1.win_rate*100:.2f}%", f"{test2.win_rate*100:.2f}%"),
        ('PnL Brut', f"{baseline_oracle.total_pnl*100:+.2f}%", f"{test1.total_pnl*100:+.2f}%", f"{test2.total_pnl*100:+.2f}%"),
        ('PnL Net', f"{baseline_oracle.total_pnl_after_fees*100:+.2f}%",
         f"{test1.total_pnl_after_fees*100:+.2f}%", f"{test2.total_pnl_after_fees*100:+.2f}%"),
        ('Sharpe Ratio', f"{baseline_oracle.sharpe_ratio:.2f}", f"{test1.sharpe_ratio:.2f}", f"{test2.sharpe_ratio:.2f}"),
    ]

    for row in metrics:
        name = row[0]
        if isinstance(row[1], str):
            logger.info(f"{name:<30} {row[1]:>20} {row[2]:>20} {row[3]:>20}")
        else:
            logger.info(f"{name:<30} {row[1]:>20,} {row[2]:>20,} {row[3]:>20,}")

    # Analyse
    logger.info("\n" + "="*100)
    logger.info("🔍 ANALYSE:")
    logger.info("="*100)

    # Comparaison Test 1 vs Test 2
    if test1.total_pnl_after_fees > test2.total_pnl_after_fees:
        winner = "Test 1 (Consensus ML)"
        gap = test1.total_pnl_after_fees - test2.total_pnl_after_fees
        logger.info(f"✅ {winner} est MEILLEUR (+{gap*100:.2f}%)")
        logger.info(f"   → Le consensus ML identifie bien les bonnes zones ✅")
    else:
        winner = "Test 2 (Désaccord ML)"
        gap = test2.total_pnl_after_fees - test1.total_pnl_after_fees
        logger.info(f"⚠️  {winner} est MEILLEUR (+{gap*100:.2f}%)")
        logger.info(f"   → Le consensus ML filtre MAL (filtre les bonnes zones!) ❌")

    # Comparaison vs Baseline
    logger.info(f"\n📊 Performance vs Baseline Oracle:")
    logger.info(f"   Baseline:         {baseline_oracle.total_pnl_after_fees*100:+.2f}%")
    logger.info(f"   Test 1 (Consensus): {test1.total_pnl_after_fees*100:+.2f}%")
    logger.info(f"   Test 2 (Désaccord): {test2.total_pnl_after_fees*100:+.2f}%")

    # Somme Test 1 + Test 2 devrait ≈ Baseline
    sum_tests = test1.total_pnl_after_fees + test2.total_pnl_after_fees
    logger.info(f"   Test 1 + Test 2:  {sum_tests*100:+.2f}% (devrait ≈ Baseline)")

    logger.info("\n" + "="*100 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Test Diagnostique: Oracle filtré par consensus ML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Split à tester (défaut: test)')
    parser.add_argument('--fees', type=float, default=0.001,
                        help='Frais par side en décimal (défaut: 0.001 = 0.1%)')
    parser.add_argument('--min-agreement', type=int, default=None,
                        help='Nombre minimum de signaux d\'accord (défaut: auto selon indicateur)')
    parser.add_argument('--signal-type', type=str, default='direction',
                        choices=['direction', 'force'],
                        help='Type de signal à tester (défaut: direction = pente, force = vélocité)')
    parser.add_argument('--indicator', type=str, default=None,
                        choices=['macd', 'rsi', 'cci'],
                        help='Indicateur unique à tester (défaut: None = tous les 6 signaux)')
    parser.add_argument('--use-force-filter', action='store_true',
                        help='Ajouter filtre Force STRONG (nécessite --signal-type direction)')

    args = parser.parse_args()

    # Ajuster min_agreement selon mode
    if args.min_agreement is None:
        if args.indicator is None:
            args.min_agreement = 4  # 4/6 (sweet spot découvert)
        else:
            args.min_agreement = 2  # 2/2 (consensus total pour indicateur unique)

    logger.info("="*100)
    logger.info("TEST DIAGNOSTIQUE - Oracle Filtré par Consensus ML")
    # Déterminer nombre total de signaux
    n_signals = 2 if args.indicator else 6
    indicator_str = args.indicator.upper() if args.indicator else "TOUS (MACD+RSI+CCI)"
    force_str = " + Force STRONG filter" if args.use_force_filter else ""

    logger.info("="*100)
    logger.info(f"Split: {args.split}")
    logger.info(f"Frais: {args.fees*100:.2f}% par side ({args.fees*2*100:.2f}% round-trip)")
    logger.info(f"Indicateur: {indicator_str} ({n_signals} signaux)")
    logger.info(f"Seuil consensus: {args.min_agreement}/{n_signals} signaux minimum")
    logger.info(f"Signal type: Direction{force_str}")
    logger.info("\nObjectif: Mesurer où le modèle ML se trompe")
    logger.info(f"   Test 1: Oracle SI >= {args.min_agreement}/{n_signals} signaux ML d'accord (consensus)")
    logger.info("   Test 2: Oracle SI prédictions ML désaccord")
    logger.info("="*100 + "\n")

    # Charger données
    data = load_all_datasets(args.split)

    # Baseline: Oracle sans filtre (résultat précédent pour référence)
    logger.info("🔮 BASELINE: Oracle Consensus Total (pour référence)")
    logger.info("-"*100)
    # On pourrait le recalculer, mais on garde juste les métriques pour comparaison

    # Test 1: Oracle filtré par CONSENSUS ML
    logger.info(f"\n🧪 TEST 1: Oracle UNIQUEMENT sur zones CONSENSUS ML (>= {args.min_agreement}/{n_signals})")
    logger.info("-"*100)
    test1_result = backtest_oracle_filtered_by_ml(
        data, args.fees, filter_mode='consensus', min_agreement=args.min_agreement,
        signal_type='direction', indicator=args.indicator, use_force_filter=args.use_force_filter
    )

    # Test 2: Oracle filtré par DÉSACCORD ML
    logger.info(f"\n🧪 TEST 2: Oracle UNIQUEMENT sur zones DÉSACCORD ML (< {args.min_agreement}/{n_signals})")
    logger.info("-"*100)
    test2_result = backtest_oracle_filtered_by_ml(
        data, args.fees, filter_mode='disagreement', min_agreement=args.min_agreement,
        signal_type='direction', indicator=args.indicator, use_force_filter=args.use_force_filter
    )

    # Créer baseline fictif pour comparaison (utiliser métriques du test précédent)
    # Pour l'instant, on compare juste Test 1 vs Test 2
    baseline = StrategyResult(
        name="Oracle Baseline",
        n_trades=75968,  # Du test précédent
        n_long=0, n_short=0,
        total_pnl=242.001,
        total_pnl_after_fees=90.065,
        total_fees=151.936,
        win_rate=0.4685,
        profit_factor=0,
        avg_win=0, avg_loss=0, avg_duration=6.0,
        sharpe_ratio=62.574,
        max_drawdown=0,
        trades=[],
        extra_metrics={'n_samples_traded': 457229}
    )

    # Comparaison
    print_diagnostic_results(test1_result, test2_result, baseline)


if __name__ == '__main__':
    main()
