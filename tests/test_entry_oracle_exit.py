#!/usr/bin/env python3
"""
Test Entry Strategy with Oracle Exit

Stratégie:
- ENTRÉE: Score pondéré avec seuils variables
  score_IN = (w1*MACD_prob + w2*CCI_prob + w3*RSI_prob) / sum(weights)
  LONG si score_IN > threshold_long
  SHORT si score_IN < threshold_short

- SORTIE: Oracle (labels parfaits) - changement de direction
  Exit quand oracle_label[i] != oracle_label[i-1]

Grid Search:
- Poids: [0.2, 0.4, 0.6, 0.8]^3 = 64 combinaisons
- threshold_long: [0.2, 0.4, 0.6, 0.8] = 4 valeurs
- threshold_short: [0.2, 0.4, 0.6, 0.8] = 4 valeurs
- Oracle: [MACD, RSI, CCI] = 3 options

Total: 64 × 4 × 4 × 3 = 3,072 combinaisons

Usage:
    python tests/test_entry_oracle_exit.py --asset BTC --split test
"""

import argparse
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from itertools import product
import time

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# === CONSTANTES ===
WEIGHT_VALUES = [0.2, 0.4, 0.6, 0.8]
THRESHOLD_VALUES = [0.2, 0.4, 0.6, 0.8]
INDICATORS = ['macd', 'cci', 'rsi']


@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    direction: str  # 'LONG' or 'SHORT'
    pnl_gross: float
    pnl_net: float
    duration: int
    score_in_at_entry: float


@dataclass
class StrategyResult:
    weights: Tuple[float, float, float]  # (w_macd, w_cci, w_rsi)
    threshold_long: float
    threshold_short: float
    oracle_indicator: str
    total_trades: int
    win_rate: float
    pnl_gross: float
    pnl_net: float
    avg_duration: float
    n_long: int
    n_short: int


def load_dataset(indicator: str, filter_type: str = 'kalman') -> Optional[Dict]:
    """Charge un dataset direction-only."""
    data_dir = Path('data/prepared')
    pattern = f'dataset_btc_eth_bnb_ada_ltc_{indicator}_direction_only_{filter_type}.npz'
    filepath = data_dir / pattern

    if not filepath.exists():
        logger.error(f"Dataset non trouvé: {filepath}")
        return None

    data = dict(np.load(filepath, allow_pickle=True))
    return data


def get_split_data(data: Dict, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extrait les données pour un split donné."""
    X = data[f'X_{split}']
    Y = data[f'Y_{split}']
    OHLCV = data[f'OHLCV_{split}']

    # Prédictions (probabilités)
    pred_key = f'Y_{split}_pred'
    if pred_key in data:
        probs = data[pred_key].flatten()
    else:
        logger.warning(f"Pas de prédictions pour {split}, utilisation des labels")
        probs = Y[:, 2].astype(float) if Y.ndim > 1 else Y.astype(float)

    return X, Y, OHLCV, probs


def filter_by_asset(Y: np.ndarray, OHLCV: np.ndarray, probs: np.ndarray,
                    asset_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filtre les données pour un asset spécifique."""
    mask = OHLCV[:, 1].astype(int) == asset_id
    return Y[mask], OHLCV[mask], probs[mask]


def calculate_score(probs: Dict[str, np.ndarray], weights: Tuple[float, float, float],
                    idx: int) -> float:
    """Calcule le score normalisé."""
    w_macd, w_cci, w_rsi = weights

    p_macd = probs['macd'][idx]
    p_cci = probs['cci'][idx]
    p_rsi = probs['rsi'][idx]

    numerator = w_macd * p_macd + w_cci * p_cci + w_rsi * p_rsi
    denominator = w_macd + w_cci + w_rsi

    return numerator / denominator


def backtest_strategy(
    probs: Dict[str, np.ndarray],
    oracle_labels: np.ndarray,
    opens: np.ndarray,
    weights: Tuple[float, float, float],
    threshold_long: float,
    threshold_short: float,
    fees: float = 0.001
) -> List[Trade]:
    """
    Exécute le backtest avec entrée pondérée et sortie Oracle.

    Entrée:
    - LONG si score_IN > threshold_long
    - SHORT si score_IN < threshold_short

    Sortie:
    - EXIT quand oracle_label change de direction
    """
    n_samples = len(opens)
    trades = []

    position = 'FLAT'
    entry_idx = 0
    entry_price = 0.0
    entry_score_in = 0.0

    for i in range(1, n_samples - 1):  # Start at 1 to check previous label
        score_in = calculate_score(probs, weights, i)

        # Détection changement Oracle (pour sortie)
        oracle_changed = oracle_labels[i] != oracle_labels[i - 1]

        if position == 'FLAT':
            # Chercher une entrée
            if score_in > threshold_long:
                position = 'LONG'
                entry_idx = i
                entry_price = opens[i + 1]  # Exécution au prochain Open
                entry_score_in = score_in
            elif score_in < threshold_short:
                position = 'SHORT'
                entry_idx = i
                entry_price = opens[i + 1]
                entry_score_in = score_in

        elif position == 'LONG':
            # Sortie Oracle: changement de direction
            if oracle_changed:
                exit_price = opens[i + 1]
                pnl_gross = (exit_price - entry_price) / entry_price
                pnl_net = pnl_gross - 2 * fees

                trades.append(Trade(
                    entry_idx=entry_idx,
                    exit_idx=i,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    direction='LONG',
                    pnl_gross=pnl_gross,
                    pnl_net=pnl_net,
                    duration=i - entry_idx,
                    score_in_at_entry=entry_score_in
                ))
                position = 'FLAT'

        elif position == 'SHORT':
            # Sortie Oracle: changement de direction
            if oracle_changed:
                exit_price = opens[i + 1]
                pnl_gross = (entry_price - exit_price) / entry_price
                pnl_net = pnl_gross - 2 * fees

                trades.append(Trade(
                    entry_idx=entry_idx,
                    exit_idx=i,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    direction='SHORT',
                    pnl_gross=pnl_gross,
                    pnl_net=pnl_net,
                    duration=i - entry_idx,
                    score_in_at_entry=entry_score_in
                ))
                position = 'FLAT'

    return trades


def evaluate_trades(trades: List[Trade]) -> Dict:
    """Calcule les métriques à partir des trades."""
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'pnl_gross': 0.0,
            'pnl_net': 0.0,
            'avg_duration': 0.0,
            'n_long': 0,
            'n_short': 0
        }

    n_wins = sum(1 for t in trades if t.pnl_net > 0)
    pnl_gross = sum(t.pnl_gross for t in trades) * 100
    pnl_net = sum(t.pnl_net for t in trades) * 100
    avg_duration = np.mean([t.duration for t in trades])
    n_long = sum(1 for t in trades if t.direction == 'LONG')
    n_short = sum(1 for t in trades if t.direction == 'SHORT')

    return {
        'total_trades': len(trades),
        'win_rate': n_wins / len(trades) * 100,
        'pnl_gross': pnl_gross,
        'pnl_net': pnl_net,
        'avg_duration': avg_duration,
        'n_long': n_long,
        'n_short': n_short
    }


def run_grid_search(
    probs: Dict[str, np.ndarray],
    oracle_labels: Dict[str, np.ndarray],
    opens: np.ndarray,
    fees: float = 0.001,
    top_n: int = 20
) -> List[StrategyResult]:
    """
    Exécute le grid search sur toutes les combinaisons.

    64 (poids) × 4 (seuil long) × 4 (seuil short) × 3 (oracle) = 3,072 combinaisons
    """
    all_results = []

    # Générer toutes les combinaisons
    weight_combos = list(product(WEIGHT_VALUES, repeat=3))

    total_combos = len(weight_combos) * len(THRESHOLD_VALUES) * len(THRESHOLD_VALUES) * len(INDICATORS)
    logger.info(f"🔍 Grid search: {total_combos:,} combinaisons")

    start_time = time.time()
    combo_idx = 0

    for weights in weight_combos:
        for threshold_long in THRESHOLD_VALUES:
            for threshold_short in THRESHOLD_VALUES:
                for oracle_ind in INDICATORS:
                    combo_idx += 1

                    # Progress tous les 500
                    if combo_idx % 500 == 0:
                        elapsed = time.time() - start_time
                        rate = combo_idx / elapsed
                        remaining = (total_combos - combo_idx) / rate
                        logger.info(f"  Progress: {combo_idx:,}/{total_combos:,} "
                                    f"({combo_idx/total_combos*100:.1f}%) - ETA: {remaining:.0f}s")

                    trades = backtest_strategy(
                        probs, oracle_labels[oracle_ind], opens,
                        weights, threshold_long, threshold_short, fees
                    )
                    metrics = evaluate_trades(trades)

                    result = StrategyResult(
                        weights=weights,
                        threshold_long=threshold_long,
                        threshold_short=threshold_short,
                        oracle_indicator=oracle_ind,
                        total_trades=metrics['total_trades'],
                        win_rate=metrics['win_rate'],
                        pnl_gross=metrics['pnl_gross'],
                        pnl_net=metrics['pnl_net'],
                        avg_duration=metrics['avg_duration'],
                        n_long=metrics['n_long'],
                        n_short=metrics['n_short']
                    )
                    all_results.append(result)

    elapsed = time.time() - start_time
    logger.info(f"✅ Grid search terminé en {elapsed:.1f}s")

    # Trier par PnL Net décroissant
    all_results.sort(key=lambda x: x.pnl_net, reverse=True)

    return all_results[:top_n]


def main():
    parser = argparse.ArgumentParser(description='Test Entry Strategy with Oracle Exit')
    parser.add_argument('--asset', type=str, default='BTC',
                        choices=['BTC', 'ETH', 'BNB', 'ADA', 'LTC'],
                        help='Asset à tester (défaut: BTC)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Split à utiliser (défaut: test)')
    parser.add_argument('--fees', type=float, default=0.001,
                        help='Frais par trade (défaut: 0.001 = 0.1%%)')
    parser.add_argument('--top-n', type=int, default=20,
                        help='Nombre de meilleurs résultats à afficher (défaut: 20)')
    parser.add_argument('--filter', type=str, default='kalman',
                        choices=['kalman', 'octave20'],
                        help='Type de filtre (défaut: kalman)')

    args = parser.parse_args()

    # Mapping asset name to ID
    asset_map = {'BTC': 0, 'ETH': 1, 'BNB': 2, 'ADA': 3, 'LTC': 4}
    asset_id = asset_map[args.asset]

    logger.info("=" * 120)
    logger.info("🎯 TEST ENTRY STRATEGY WITH ORACLE EXIT")
    logger.info("=" * 120)
    logger.info(f"Asset: {args.asset} (ID={asset_id})")
    logger.info(f"Split: {args.split}")
    logger.info(f"Fees: {args.fees*100:.2f}%")
    logger.info(f"Filter: {args.filter}")
    logger.info(f"Poids testés: {WEIGHT_VALUES}")
    logger.info(f"Seuils testés: {THRESHOLD_VALUES}")
    logger.info(f"Oracle indicators: {INDICATORS}")

    # === CHARGEMENT DES DONNÉES ===
    logger.info("\n📂 Chargement des datasets...")

    datasets = {}
    probs_all = {}
    oracle_labels = {}

    for ind in INDICATORS:
        data = load_dataset(ind, args.filter)
        if data is None:
            logger.error(f"❌ Impossible de charger {ind}")
            return
        datasets[ind] = data

        # Extraire les données du split
        X, Y, OHLCV, probs = get_split_data(data, args.split)

        # Filtrer par asset
        Y_asset, OHLCV_asset, probs_asset = filter_by_asset(Y, OHLCV, probs, asset_id)

        probs_all[ind] = probs_asset

        # Labels Oracle (colonne 2 = direction)
        oracle_labels[ind] = Y_asset[:, 2].astype(int)

        logger.info(f"  {ind.upper()}: {len(probs_asset):,} samples, "
                    f"Oracle labels: {oracle_labels[ind].sum():,} UP / {len(oracle_labels[ind]) - oracle_labels[ind].sum():,} DOWN")

    # Vérifier alignement
    n_samples = [len(probs_all[ind]) for ind in INDICATORS]
    if len(set(n_samples)) > 1:
        min_n = min(n_samples)
        logger.warning(f"⚠️ Truncation à {min_n:,} samples")
        for ind in INDICATORS:
            probs_all[ind] = probs_all[ind][:min_n]
            oracle_labels[ind] = oracle_labels[ind][:min_n]

    # Récupérer OHLCV pour les prix
    _, _, OHLCV_ref, _ = get_split_data(datasets['macd'], args.split)
    _, OHLCV_asset, _ = filter_by_asset(
        datasets['macd'][f'Y_{args.split}'],
        OHLCV_ref,
        np.zeros(len(OHLCV_ref)),
        asset_id
    )
    opens = OHLCV_asset[:len(probs_all['macd']), 2]

    logger.info(f"\n📊 Données chargées: {len(opens):,} samples")

    # === GRID SEARCH ===
    logger.info("\n" + "=" * 120)
    top_results = run_grid_search(probs_all, oracle_labels, opens, args.fees, args.top_n)

    # === AFFICHAGE DES RÉSULTATS ===
    logger.info("\n" + "=" * 120)
    logger.info(f"🏆 TOP {args.top_n} MEILLEURES COMBINAISONS (par PnL Net)")
    logger.info("=" * 120)

    logger.info(f"\n{'Rank':<5} {'Weights (M,C,R)':<18} {'ThLong':<8} {'ThShort':<8} {'Oracle':<8} "
                f"{'Trades':<8} {'WinRate':<8} {'PnL Gross':<12} {'PnL Net':<12} {'AvgDur':<8} {'L/S':<10}")
    logger.info("-" * 130)

    for rank, res in enumerate(top_results, 1):
        w_str = f"({res.weights[0]:.1f},{res.weights[1]:.1f},{res.weights[2]:.1f})"
        ls_str = f"{res.n_long}/{res.n_short}"

        logger.info(f"{rank:<5} {w_str:<18} {res.threshold_long:<8.1f} {res.threshold_short:<8.1f} "
                    f"{res.oracle_indicator.upper():<8} {res.total_trades:<8} {res.win_rate:<7.1f}% "
                    f"{res.pnl_gross:>+10.2f}% {res.pnl_net:>+10.2f}% "
                    f"{res.avg_duration:<8.1f} {ls_str:<10}")

    # === ANALYSE DU MEILLEUR ===
    if top_results:
        best = top_results[0]
        logger.info("\n" + "=" * 120)
        logger.info("🥇 MEILLEURE COMBINAISON")
        logger.info("=" * 120)
        logger.info(f"Weights: MACD={best.weights[0]:.1f}, CCI={best.weights[1]:.1f}, RSI={best.weights[2]:.1f}")
        logger.info(f"Threshold LONG: > {best.threshold_long}")
        logger.info(f"Threshold SHORT: < {best.threshold_short}")
        logger.info(f"Oracle Exit: {best.oracle_indicator.upper()}")
        logger.info(f"Trades: {best.total_trades} (LONG: {best.n_long}, SHORT: {best.n_short})")
        logger.info(f"Win Rate: {best.win_rate:.1f}%")
        logger.info(f"PnL Gross: {best.pnl_gross:+.2f}%")
        logger.info(f"PnL Net: {best.pnl_net:+.2f}%")
        logger.info(f"Durée moyenne: {best.avg_duration:.1f} périodes")

        # Stats par Oracle
        logger.info("\n📊 MEILLEUR RÉSULTAT PAR ORACLE:")
        for oracle_ind in INDICATORS:
            oracle_results = [r for r in top_results if r.oracle_indicator == oracle_ind]
            if oracle_results:
                best_oracle = oracle_results[0]
                logger.info(f"  {oracle_ind.upper()}: PnL Net={best_oracle.pnl_net:+.2f}%, "
                            f"WR={best_oracle.win_rate:.1f}%, Trades={best_oracle.total_trades}")

    # === STATS DISTRIBUTION ===
    logger.info("\n📈 DISTRIBUTION DES RÉSULTATS (top 20):")
    all_pnl_net = [r.pnl_net for r in top_results]
    all_trades = [r.total_trades for r in top_results]

    logger.info(f"  PnL Net - min: {min(all_pnl_net):.2f}%, max: {max(all_pnl_net):.2f}%, mean: {np.mean(all_pnl_net):.2f}%")
    logger.info(f"  Trades - min: {min(all_trades)}, max: {max(all_trades)}, mean: {np.mean(all_trades):.0f}")


if __name__ == '__main__':
    main()
