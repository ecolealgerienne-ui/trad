#!/usr/bin/env python3
"""
Test Consensus Total - Tous Indicateurs ET Tous Filtres d'Accord

PRINCIPE:
Ne trader QUE si TOUS les indicateurs ET filtres sont d'accord sur la DIRECTION.
Pas de vitesse (Force), seulement la pente (Direction).

6 Signaux de Direction:
1. MACD Direction (Kalman)
2. MACD Direction (Octave)
3. RSI Direction (Kalman)
4. RSI Direction (Octave)
5. CCI Direction (Kalman)
6. CCI Direction (Octave)

Logique:
- LONG: Si TOUS les 6 signaux = UP
- SHORT: Si TOUS les 6 signaux = DOWN
- FLAT: Sinon (attendre consensus)

Tests:
1. Oracle (labels parfaits) → Potentiel maximum
2. Prédictions ML → Performance réelle

Objectif:
- Réduire trades drastiquement (30k → 3-5k?)
- Win Rate élevé (signaux ultra-confirmés)
- PnL Net POSITIF avec frais 0.1% par side (0.2% round-trip)

Usage:
    python tests/test_consensus_all_filters.py --split test --fees 0.1
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
    compute_trading_stats, print_comparison_table
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CHARGEMENT MULTI-INDICATEURS MULTI-FILTRES
# =============================================================================

def load_all_datasets(
    split: str = 'test'
) -> Dict:
    """
    Charge les 3 indicateurs × 2 filtres = 6 datasets.

    Args:
        split: 'train', 'val', ou 'test'

    Returns:
        Dict avec:
        - returns: Returns extraits (MACD comme référence)
        - macd_kalman: {Y, Y_pred}
        - macd_octave: {Y, Y_pred}
        - rsi_kalman: {Y, Y_pred}
        - rsi_octave: {Y, Y_pred}
        - cci_kalman: {Y, Y_pred}
        - cci_octave: {Y, Y_pred}

    Raises:
        FileNotFoundError: Si un dataset manque
    """
    base_path = Path('data/prepared')
    results = {}

    # Extraire returns depuis MACD Kalman (référence)
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
            logger.warning(f"   Essai de continuer sans {indicator}_{filter_type}...")
            results[f'{indicator}_{filter_type}'] = None
            continue

        logger.info(f"   Loading {indicator}_{filter_type}...")
        data = np.load(path, allow_pickle=True)

        key = f'{indicator}_{filter_type}'
        results[key] = {
            'Y': data[f'Y_{split}'],           # Labels (n, 2) [Direction, Force]
            'Y_pred': data.get(f'Y_{split}_pred', None),  # Predictions si disponibles
            'X': data[f'X_{split}']  # Pour extraire returns
        }

    # Vérifier qu'au moins un dataset existe
    valid_keys = [k for k, v in results.items() if v is not None]
    if not valid_keys:
        raise FileNotFoundError("Aucun dataset trouvé!")

    # Extraire returns depuis le premier dataset valide (MACD Kalman de préférence)
    if results.get('macd_kalman') is not None:
        X = results['macd_kalman']['X']
        indicator = 'macd'
    elif results.get('rsi_kalman') is not None:
        X = results['rsi_kalman']['X']
        indicator = 'rsi'
    else:
        # Prendre le premier disponible
        first_key = valid_keys[0]
        X = results[first_key]['X']
        indicator = first_key.split('_')[0]

    results['returns'] = extract_c_ret(X, indicator)

    logger.info(f"✅ {len(valid_keys)}/6 datasets chargés")
    logger.info(f"   Samples: {len(results['returns']):,}")

    return results


# =============================================================================
# BACKTEST CONSENSUS TOTAL
# =============================================================================

def backtest_consensus(
    data: Dict,
    fees: float = 0.001,
    use_predictions: bool = False
) -> StrategyResult:
    """
    Backtest consensus total: LONG/SHORT seulement si TOUS d'accord.

    Args:
        data: Dict retourné par load_all_datasets()
        fees: Frais par side (défaut: 0.1%)
        use_predictions: Si True, utilise Y_pred, sinon Y (Oracle)

    Returns:
        StrategyResult

    Logique:
        Position = LONG si ALL 6 signaux Direction = UP
        Position = SHORT si ALL 6 signaux Direction = DOWN
        Position = FLAT sinon
    """
    returns = data['returns']
    n_samples = len(returns)

    # Extraire les 6 signaux de direction
    signals = {}
    available_signals = []

    for indicator in ['macd', 'rsi', 'cci']:
        for filter_type in ['kalman', 'octave20']:
            key = f'{indicator}_{filter_type}'
            if data.get(key) is not None:
                if use_predictions and data[key]['Y_pred'] is not None:
                    # Utiliser prédictions
                    pred = data[key]['Y_pred']
                    signals[key] = (pred[:, 0] > 0.5).astype(int)  # Direction seulement
                else:
                    # Utiliser labels (Oracle)
                    labels = data[key]['Y']
                    signals[key] = labels[:, 0].astype(int)  # Direction seulement

                available_signals.append(key)

    logger.info(f"Signaux disponibles: {len(available_signals)}/6")
    logger.info(f"   {', '.join(available_signals)}")

    if len(available_signals) == 0:
        raise ValueError("Aucun signal disponible!")

    # État trading
    position = Position.FLAT
    entry_time = 0
    current_pnl = 0.0
    trades = []
    n_long = 0
    n_short = 0

    # Compteurs consensus
    n_long_signals = 0
    n_short_signals = 0
    n_flat_signals = 0

    for i in range(n_samples):
        ret = returns[i]

        # Accumuler PnL
        current_pnl = compute_pnl_step(position, ret, current_pnl)

        # Vérifier consensus sur la direction
        directions = [signals[key][i] for key in available_signals]

        # Consensus: TOUS doivent être identiques
        if all(d == 1 for d in directions):
            # Tous UP
            target = Position.LONG
            n_long_signals += 1
        elif all(d == 0 for d in directions):
            # Tous DOWN
            target = Position.SHORT
            n_short_signals += 1
        else:
            # Désaccord
            target = Position.FLAT
            n_flat_signals += 1

        # SORTIE si position change
        exit_signal = False
        exit_reason = None

        if position != Position.FLAT and target != position:
            exit_signal = True
            if target == Position.FLAT:
                exit_reason = "CONSENSUS_LOST"
            else:
                exit_reason = "DIRECTION_FLIP"

        # Exécuter sortie
        if exit_signal:
            execute_exit(
                trades, entry_time, i, position,
                current_pnl, exit_reason, fees
            )

            # Gérer sortie
            if exit_reason == "CONSENSUS_LOST":
                # Retour FLAT
                position = Position.FLAT
                current_pnl = 0.0

            elif exit_reason == "DIRECTION_FLIP":
                # Flip immédiat
                position = target
                entry_time = i
                current_pnl = 0.0

                # Compter
                if target == Position.LONG:
                    n_long += 1
                else:
                    n_short += 1

        # ENTRÉE si FLAT et signal consensus
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

    # Statistiques consensus
    logger.info(f"\n📊 Distribution Signaux Consensus:")
    logger.info(f"   LONG signals: {n_long_signals:,} ({n_long_signals/n_samples*100:.1f}%)")
    logger.info(f"   SHORT signals: {n_short_signals:,} ({n_short_signals/n_samples*100:.1f}%)")
    logger.info(f"   FLAT (désaccord): {n_flat_signals:,} ({n_flat_signals/n_samples*100:.1f}%)")

    # Calculer stats
    mode = "Predictions ML" if use_predictions else "Oracle"
    return compute_trading_stats(
        trades=trades,
        n_long=n_long,
        n_short=n_short,
        strategy_name=f"Consensus Total ({mode})",
        extra_metrics={
            'n_signaux': len(available_signals),
            'long_signals': n_long_signals,
            'short_signals': n_short_signals,
            'flat_signals': n_flat_signals,
            'consensus_rate': (n_long_signals + n_short_signals) / n_samples * 100
        }
    )


# =============================================================================
# AFFICHAGE SPÉCIALISÉ
# =============================================================================

def print_consensus_results(oracle: StrategyResult, predictions: StrategyResult):
    """Affiche comparaison Oracle vs Predictions."""
    logger.info("\n" + "="*100)
    logger.info("RÉSULTATS CONSENSUS TOTAL - Oracle vs Prédictions ML")
    logger.info("="*100)
    logger.info(f"{'Métrique':<30} {'Oracle':>20} {'Predictions ML':>20} {'Delta':>15}")
    logger.info("-"*100)

    # Métriques principales
    metrics = [
        ('Trades', oracle.n_trades, predictions.n_trades),
        ('Win Rate', f"{oracle.win_rate*100:.2f}%", f"{predictions.win_rate*100:.2f}%"),
        ('PnL Brut', f"{oracle.total_pnl*100:+.2f}%", f"{predictions.total_pnl*100:+.2f}%"),
        ('PnL Net', f"{oracle.total_pnl_after_fees*100:+.2f}%", f"{predictions.total_pnl_after_fees*100:+.2f}%"),
        ('Frais Total', f"{oracle.total_fees*100:.2f}%", f"{predictions.total_fees*100:.2f}%"),
        ('Sharpe Ratio', f"{oracle.sharpe_ratio:.3f}", f"{predictions.sharpe_ratio:.3f}"),
        ('Avg Duration', f"{oracle.avg_duration:.1f}p", f"{predictions.avg_duration:.1f}p"),
    ]

    for name, oracle_val, pred_val in metrics:
        if isinstance(oracle_val, str):
            logger.info(f"{name:<30} {oracle_val:>20} {pred_val:>20} {'-':>15}")
        else:
            delta = pred_val - oracle_val
            logger.info(f"{name:<30} {oracle_val:>20,} {pred_val:>20,} {delta:>+15,}")

    # Métriques extra
    if oracle.extra_metrics:
        logger.info("\n📊 Métriques Consensus:")
        logger.info(f"   Signaux utilisés: {oracle.extra_metrics['n_signaux']}/6")
        logger.info(f"   Consensus LONG: {oracle.extra_metrics['long_signals']:,} ({oracle.extra_metrics['consensus_rate']/2:.1f}%)")
        logger.info(f"   Consensus SHORT: {oracle.extra_metrics['short_signals']:,} ({oracle.extra_metrics['consensus_rate']/2:.1f}%)")
        logger.info(f"   FLAT (désaccord): {oracle.extra_metrics['flat_signals']:,} ({100-oracle.extra_metrics['consensus_rate']:.1f}%)")

    # Verdict
    logger.info("\n" + "="*100)

    if predictions.total_pnl_after_fees > 0:
        logger.info("✅ SUCCÈS: PnL Net POSITIF avec prédictions ML!")
        logger.info(f"   Le consensus total fonctionne: {predictions.total_pnl_after_fees*100:+.2f}%")
    else:
        logger.info("❌ ÉCHEC: PnL Net encore négatif")
        logger.info(f"   Perte nette: {predictions.total_pnl_after_fees*100:+.2f}%")

        # Diagnostic
        gap_oracle_pred = oracle.total_pnl_after_fees - predictions.total_pnl_after_fees
        logger.info(f"\n🔍 Diagnostic:")
        logger.info(f"   Gap Oracle vs ML: {gap_oracle_pred*100:.2f}%")
        logger.info(f"   Oracle PnL Net: {oracle.total_pnl_after_fees*100:+.2f}%")

        if oracle.total_pnl_after_fees > 0:
            logger.info(f"   → Signal existe (Oracle positif) mais ML pas assez précis")
        else:
            logger.info(f"   → Signal insuffisant même avec Oracle parfait")

    logger.info("="*100 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Test Consensus Total - Tous indicateurs ET filtres d\'accord',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Split à tester (défaut: test)')
    parser.add_argument('--fees', type=float, default=0.001,
                        help='Frais par side en % (défaut: 0.1%)')

    args = parser.parse_args()

    logger.info("="*100)
    logger.info("TEST CONSENSUS TOTAL - Tous Indicateurs ET Filtres d'Accord")
    logger.info("="*100)
    logger.info(f"Split: {args.split}")
    logger.info(f"Frais: {args.fees*100:.2f}% par side ({args.fees*2*100:.2f}% round-trip)")
    logger.info("\nStratégie: LONG/SHORT seulement si TOUS les 6 signaux d'accord")
    logger.info("   - MACD (Kalman + Octave)")
    logger.info("   - RSI (Kalman + Octave)")
    logger.info("   - CCI (Kalman + Octave)")
    logger.info("="*100 + "\n")

    # Charger données
    data = load_all_datasets(args.split)

    # Test 1: Oracle (labels parfaits)
    logger.info("🔮 TEST 1: ORACLE (Labels Parfaits)")
    logger.info("-"*100)
    oracle_result = backtest_consensus(data, args.fees, use_predictions=False)

    # Test 2: Prédictions ML
    logger.info("\n🤖 TEST 2: PRÉDICTIONS ML (Modèle 92% Accuracy)")
    logger.info("-"*100)
    pred_result = backtest_consensus(data, args.fees, use_predictions=True)

    # Comparaison
    print_consensus_results(oracle_result, pred_result)


if __name__ == '__main__':
    main()
