#!/usr/bin/env python3
"""
Test ML Predictions Direction-Only - Validation des prédictions du modèle ML.

OBJECTIF:
Tester les prédictions ML (Y_pred) sur le split test pour mesurer la performance réelle
en trading (Win Rate, PnL Net) au lieu de seulement l'accuracy.

DIFFÉRENCE AVEC ORACLE:
- Oracle: Utilise les labels parfaits (Y) → Performance maximale théorique
- ML: Utilise les prédictions du modèle (Y_pred) → Performance réelle attendue

LOGIQUE CAUSALE:
- Signal à temps t (basé sur prediction[t])
- Exécution à Open[t+1] (prochaine bougie)
- PnL calculé sur prix réels (pas returns)

STRUCTURE DATASET:
- Y: (n, 3) = [timestamp, asset_id, direction] (labels Oracle)
- Y_pred: (n,) = probabilités ML (0-1) → converti en direction (>0.5 = UP)
- OHLCV: (n, 7) = [timestamp, asset_id, O, H, L, C, V]

Usage:
    python tests/test_ml_predictions_direction_only.py --indicator macd --split test
    python tests/test_ml_predictions_direction_only.py --indicator rsi --split test --fees 0.001
    python tests/test_ml_predictions_direction_only.py --indicator cci --split test --compare-oracle
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import argparse
from dataclasses import dataclass
from typing import List, Dict
from enum import IntEnum
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTES ET TYPES
# =============================================================================

class Position(IntEnum):
    """Position de trading."""
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass
class Trade:
    """Enregistrement d'un trade."""
    entry_idx: int
    exit_idx: int
    duration: int
    position: str  # 'LONG' ou 'SHORT'
    entry_price: float
    exit_price: float
    pnl: float  # PnL brut (%)
    pnl_after_fees: float  # PnL net (%)
    asset_id: int = 0  # ID de l'asset
    entry_timestamp: float = 0.0  # Timestamp d'entrée


@dataclass
class BacktestResult:
    """Résultats du backtest."""
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
    max_drawdown: float
    trades: List[Trade]


@dataclass
class AssetResult:
    """Résultats par asset."""
    asset_id: int
    n_trades: int
    total_pnl: float
    total_pnl_after_fees: float
    win_rate: float
    avg_duration: float


@dataclass
class MonthlyResult:
    """Résultats par mois."""
    year_month: str
    n_trades: int
    total_pnl: float
    total_pnl_after_fees: float
    win_rate: float


# =============================================================================
# CHARGEMENT DONNÉES
# =============================================================================

def load_dataset(indicator: str, split: str = 'test') -> Dict:
    """
    Charge le dataset direction-only avec prédictions ML.

    Args:
        indicator: 'macd', 'rsi', ou 'cci'
        split: 'train', 'val', ou 'test'

    Returns:
        Dict avec Y (labels Oracle), Y_pred (prédictions ML), OHLCV (prix)
    """
    path = Path(f'data/prepared/dataset_btc_eth_bnb_ada_ltc_{indicator}_direction_only_kalman.npz')

    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {path}")

    logger.info(f"Chargement: {path}")
    data = np.load(path, allow_pickle=True)

    # Extraire les données du split
    Y = data[f'Y_{split}']
    OHLCV = data[f'OHLCV_{split}']

    # Prédictions ML (CRITIQUE: doivent exister!)
    pred_key = f'Y_{split}_pred'
    if pred_key not in data:
        raise KeyError(
            f"Prédictions ML introuvables dans le dataset!\n"
            f"Clé attendue: {pred_key}\n"
            f"Clés disponibles: {list(data.keys())}\n\n"
            f"SOLUTION: Régénérer les prédictions avec:\n"
            f"  python src/regenerate_predictions.py --data {path} --indicator {indicator}"
        )

    Y_pred = data[pred_key]

    logger.info(f"  Y shape: {Y.shape} - [timestamp, asset_id, direction]")
    logger.info(f"  Y_pred shape: {Y_pred.shape} - probabilités ML")
    logger.info(f"  OHLCV shape: {OHLCV.shape} - [timestamp, asset_id, O, H, L, C, V]")

    return {
        'Y': Y,
        'Y_pred': Y_pred,
        'OHLCV': OHLCV
    }


# =============================================================================
# BACKTEST (PAR ASSET)
# =============================================================================

def backtest_single_asset(
    labels: np.ndarray,
    opens: np.ndarray,
    timestamps: np.ndarray,
    asset_id: int,
    fees: float = 0.001
) -> List[Trade]:
    """
    Backtest pour UN SEUL asset.

    LOGIQUE CAUSALE:
    - Signal à index i → Exécution à Open[i+1]
    - Toujours en position (LONG ou SHORT, jamais FLAT)
    - Direction: 1=UP→LONG, 0=DOWN→SHORT

    Args:
        labels: (n,) Direction labels/predictions pour cet asset
        opens: (n,) Prix Open pour cet asset
        timestamps: (n,) Timestamps pour cet asset
        asset_id: ID de l'asset
        fees: Frais par side

    Returns:
        Liste des trades
    """
    n_samples = len(labels)
    trades = []
    position = Position.FLAT
    entry_idx = 0
    entry_price = 0.0
    entry_timestamp = 0.0

    # Boucle principale
    for i in range(n_samples - 1):  # -1 car besoin de Open[i+1]
        direction = int(labels[i])
        target = Position.LONG if direction == 1 else Position.SHORT

        # Première entrée
        if position == Position.FLAT:
            position = target
            entry_idx = i
            entry_price = opens[i + 1]  # Entrée à Open[i+1]
            entry_timestamp = timestamps[i + 1]
            continue

        # Changement de position (reversal)
        if position != target:
            # Sortie à Open[i+1]
            exit_price = opens[i + 1]

            # Calcul PnL
            if position == Position.LONG:
                pnl = (exit_price - entry_price) / entry_price
            else:  # SHORT
                pnl = (entry_price - exit_price) / entry_price

            # Frais: entrée + sortie
            trade_fees = 2 * fees
            pnl_after_fees = pnl - trade_fees

            # Enregistrer trade
            trades.append(Trade(
                entry_idx=entry_idx,
                exit_idx=i,
                duration=i - entry_idx,
                position='LONG' if position == Position.LONG else 'SHORT',
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                pnl_after_fees=pnl_after_fees,
                asset_id=asset_id,
                entry_timestamp=entry_timestamp
            ))

            # Nouvelle position (reversal immédiat)
            position = target
            entry_idx = i
            entry_price = opens[i + 1]
            entry_timestamp = timestamps[i + 1]

    # Fermer position finale
    if position != Position.FLAT:
        exit_price = opens[-1]

        if position == Position.LONG:
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price

        trade_fees = 2 * fees
        pnl_after_fees = pnl - trade_fees

        trades.append(Trade(
            entry_idx=entry_idx,
            exit_idx=n_samples - 1,
            duration=n_samples - 1 - entry_idx,
            position='LONG' if position == Position.LONG else 'SHORT',
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_after_fees=pnl_after_fees,
            asset_id=asset_id,
            entry_timestamp=entry_timestamp
        ))

    return trades


def backtest(
    labels: np.ndarray,
    ohlcv: np.ndarray,
    fees: float = 0.001
) -> tuple:
    """
    Backtest avec labels/predictions.

    CRITIQUE: Le dataset contient PLUSIEURS assets concaténés!
    On doit faire le backtest PAR ASSET pour éviter de calculer
    des PnL entre prix BTC et ETH.

    Args:
        labels: (n,) Direction labels/predictions (0=DOWN, 1=UP)
        ohlcv: (n, 7) Prix [timestamp, asset_id, O, H, L, C, V]
        fees: Frais par side (défaut: 0.1%)

    Returns:
        (BacktestResult, List[AssetResult])
    """
    # Extraire colonnes
    timestamps = ohlcv[:, 0]  # Colonne 0 = timestamp
    asset_ids = ohlcv[:, 1].astype(int)  # Colonne 1 = asset_id
    opens = ohlcv[:, 2]  # Colonne 2 = Open

    # Identifier les assets uniques
    unique_assets = np.unique(asset_ids)
    logger.info(f"  Assets détectés: {len(unique_assets)} ({unique_assets})")

    all_trades = []
    asset_results = []
    n_long = 0
    n_short = 0

    # Backtest PAR ASSET
    for asset_id in unique_assets:
        # Masque pour cet asset
        mask = asset_ids == asset_id
        asset_labels = labels[mask]
        asset_opens = opens[mask]
        asset_timestamps = timestamps[mask]

        logger.info(f"    Asset {int(asset_id)}: {len(asset_labels):,} samples")

        # Backtest cet asset
        trades = backtest_single_asset(
            asset_labels, asset_opens, asset_timestamps, int(asset_id), fees
        )

        # Compter LONG/SHORT et stats par asset
        asset_pnl = 0.0
        asset_pnl_net = 0.0
        asset_wins = 0
        asset_duration = 0
        for t in trades:
            if t.position == 'LONG':
                n_long += 1
            else:
                n_short += 1
            asset_pnl += t.pnl
            asset_pnl_net += t.pnl_after_fees
            if t.pnl_after_fees > 0:
                asset_wins += 1
            asset_duration += t.duration

        # Résultat pour cet asset
        if len(trades) > 0:
            asset_results.append(AssetResult(
                asset_id=int(asset_id),
                n_trades=len(trades),
                total_pnl=asset_pnl,
                total_pnl_after_fees=asset_pnl_net,
                win_rate=asset_wins / len(trades) if len(trades) > 0 else 0.0,
                avg_duration=asset_duration / len(trades) if len(trades) > 0 else 0.0
            ))

        all_trades.extend(trades)

    logger.info(f"  Total trades: {len(all_trades):,}")

    # Calculer statistiques globales
    result = compute_stats(all_trades, n_long, n_short)
    return result, asset_results


def compute_stats(trades: List[Trade], n_long: int, n_short: int) -> BacktestResult:
    """Calcule les statistiques du backtest."""
    if len(trades) == 0:
        return BacktestResult(
            n_trades=0, n_long=0, n_short=0,
            total_pnl=0.0, total_pnl_after_fees=0.0, total_fees=0.0,
            win_rate=0.0, profit_factor=0.0,
            avg_win=0.0, avg_loss=0.0, avg_duration=0.0,
            sharpe_ratio=0.0, max_drawdown=0.0, trades=[]
        )

    pnls = np.array([t.pnl for t in trades])
    pnls_net = np.array([t.pnl_after_fees for t in trades])
    durations = np.array([t.duration for t in trades])

    total_pnl = pnls.sum()
    total_pnl_net = pnls_net.sum()
    total_fees = total_pnl - total_pnl_net

    # Win rate
    wins = pnls_net > 0
    losses = pnls_net < 0
    win_rate = wins.mean() if len(trades) > 0 else 0.0

    # Profit factor
    sum_wins = pnls_net[wins].sum() if wins.any() else 0.0
    sum_losses = abs(pnls_net[losses].sum()) if losses.any() else 0.0
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else 0.0

    # Moyennes
    avg_win = pnls_net[wins].mean() if wins.any() else 0.0
    avg_loss = pnls_net[losses].mean() if losses.any() else 0.0
    avg_duration = durations.mean()

    # Sharpe (annualisé, 5min = 288 périodes/jour)
    if len(pnls_net) > 1 and pnls_net.std() > 0:
        sharpe = (pnls_net.mean() / pnls_net.std()) * np.sqrt(288 * 365)
    else:
        sharpe = 0.0

    # Max Drawdown
    cumulative = np.cumsum(pnls_net)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = drawdowns.max() if len(drawdowns) > 0 else 0.0

    return BacktestResult(
        n_trades=len(trades),
        n_long=n_long,
        n_short=n_short,
        total_pnl=total_pnl,
        total_pnl_after_fees=total_pnl_net,
        total_fees=total_fees,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_duration=avg_duration,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        trades=trades
    )


# =============================================================================
# CALCUL STATS MENSUELLES
# =============================================================================

def compute_monthly_stats(trades: List[Trade]) -> List[MonthlyResult]:
    """
    Calcule les statistiques par mois.

    Args:
        trades: Liste des trades avec timestamps

    Returns:
        Liste des résultats mensuels
    """
    from datetime import datetime
    from collections import defaultdict

    # Grouper par mois
    monthly_data = defaultdict(list)

    for trade in trades:
        # Convertir timestamp en datetime
        # Détection automatique de l'unité du timestamp
        ts = trade.entry_timestamp
        if ts > 1e18:  # Nanosecondes
            ts = ts / 1e9
        elif ts > 1e15:  # Microsecondes
            ts = ts / 1e6
        elif ts > 1e12:  # Millisecondes
            ts = ts / 1e3
        # Sinon c'est déjà en secondes

        try:
            dt = datetime.fromtimestamp(ts)
            year_month = dt.strftime('%Y-%m')
            monthly_data[year_month].append(trade)
        except (ValueError, OSError):
            # Timestamp invalide, ignorer ce trade pour les stats mensuelles
            continue

    # Calculer stats par mois
    monthly_results = []
    for year_month in sorted(monthly_data.keys()):
        month_trades = monthly_data[year_month]
        n_trades = len(month_trades)
        total_pnl = sum(t.pnl for t in month_trades)
        total_pnl_net = sum(t.pnl_after_fees for t in month_trades)
        wins = sum(1 for t in month_trades if t.pnl_after_fees > 0)
        win_rate = wins / n_trades if n_trades > 0 else 0.0

        monthly_results.append(MonthlyResult(
            year_month=year_month,
            n_trades=n_trades,
            total_pnl=total_pnl,
            total_pnl_after_fees=total_pnl_net,
            win_rate=win_rate
        ))

    return monthly_results


# =============================================================================
# AFFICHAGE RÉSULTATS
# =============================================================================

def print_asset_results(asset_results: List[AssetResult]):
    """Affiche les résultats par asset."""
    # Mapping asset_id → nom
    asset_names = {0: 'BTC', 1: 'ETH', 2: 'BNB', 3: 'ADA', 4: 'LTC'}

    print("\n" + "="*70)
    print("RÉSULTATS PAR ASSET")
    print("="*70)

    print(f"\n{'Asset':<8} {'Trades':>10} {'PnL Brut':>12} {'PnL Net':>12} {'Win Rate':>10} {'Durée Moy':>10}")
    print("-"*70)

    for ar in asset_results:
        name = asset_names.get(ar.asset_id, f'Asset{ar.asset_id}')
        print(f"{name:<8} {ar.n_trades:>10,} {ar.total_pnl*100:>+11.2f}% {ar.total_pnl_after_fees*100:>+11.2f}% {ar.win_rate*100:>9.1f}% {ar.avg_duration:>9.1f}p")

    # Moyennes
    if asset_results:
        avg_pnl = sum(ar.total_pnl for ar in asset_results) / len(asset_results)
        avg_pnl_net = sum(ar.total_pnl_after_fees for ar in asset_results) / len(asset_results)
        avg_wr = sum(ar.win_rate for ar in asset_results) / len(asset_results)
        avg_dur = sum(ar.avg_duration for ar in asset_results) / len(asset_results)
        avg_trades = sum(ar.n_trades for ar in asset_results) / len(asset_results)

        print("-"*70)
        print(f"{'MOYENNE':<8} {avg_trades:>10,.0f} {avg_pnl*100:>+11.2f}% {avg_pnl_net*100:>+11.2f}% {avg_wr*100:>9.1f}% {avg_dur:>9.1f}p")


def print_monthly_results(trades: List[Trade]):
    """Affiche les résultats par mois."""
    monthly_results = compute_monthly_stats(trades)

    print("\n" + "="*70)
    print("RÉSULTATS PAR MOIS")
    print("="*70)

    print(f"\n{'Mois':<10} {'Trades':>10} {'PnL Brut':>12} {'PnL Net':>12} {'Win Rate':>10}")
    print("-"*60)

    for mr in monthly_results:
        print(f"{mr.year_month:<10} {mr.n_trades:>10,} {mr.total_pnl*100:>+11.2f}% {mr.total_pnl_after_fees*100:>+11.2f}% {mr.win_rate*100:>9.1f}%")

    # Moyennes
    if monthly_results:
        avg_trades = sum(mr.n_trades for mr in monthly_results) / len(monthly_results)
        avg_pnl = sum(mr.total_pnl for mr in monthly_results) / len(monthly_results)
        avg_pnl_net = sum(mr.total_pnl_after_fees for mr in monthly_results) / len(monthly_results)
        avg_wr = sum(mr.win_rate for mr in monthly_results) / len(monthly_results)

        print("-"*60)
        print(f"{'MOYENNE':<10} {avg_trades:>10,.0f} {avg_pnl*100:>+11.2f}% {avg_pnl_net*100:>+11.2f}% {avg_wr*100:>9.1f}%")
        print(f"\nNombre de mois: {len(monthly_results)}")


def print_results(result: BacktestResult, indicator: str):
    """Affiche les résultats du backtest ML."""
    print("\n" + "="*70)
    print(f"RÉSULTATS ML PREDICTIONS - {indicator.upper()}")
    print("="*70)

    print(f"\nTrades:")
    print(f"  Total: {result.n_trades:,}")
    print(f"  Long: {result.n_long:,} ({result.n_long/result.n_trades*100:.1f}%)" if result.n_trades > 0 else "  Long: 0")
    print(f"  Short: {result.n_short:,} ({result.n_short/result.n_trades*100:.1f}%)" if result.n_trades > 0 else "  Short: 0")
    print(f"  Durée moyenne: {result.avg_duration:.1f} périodes")

    print(f"\nPerformance:")
    print(f"  PnL Brut: {result.total_pnl*100:+.2f}%")
    print(f"  Frais: {result.total_fees*100:.2f}%")
    print(f"  PnL Net: {result.total_pnl_after_fees*100:+.2f}%")

    print(f"\nMétriques:")
    print(f"  Win Rate: {result.win_rate*100:.1f}%")
    print(f"  Profit Factor: {result.profit_factor:.2f}")
    print(f"  Avg Win: {result.avg_win*100:+.3f}%")
    print(f"  Avg Loss: {result.avg_loss*100:+.3f}%")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown*100:.2f}%")

    # Verdict
    print(f"\nVerdict:")
    if result.total_pnl_after_fees > 0:
        print(f"  ✅ PnL Net POSITIF - Modèle ML profitable!")
    else:
        print(f"  ❌ PnL Net NÉGATIF - Modèle ML non profitable")

    # Critères de succès
    print(f"\nCritères Phase 2.15:")
    print(f"  Win Rate ≥ 45%: {'✅' if result.win_rate >= 0.45 else '❌'} ({result.win_rate*100:.1f}%)")
    print(f"  PnL Net > 0: {'✅' if result.total_pnl_after_fees > 0 else '❌'}")
    print(f"  Profit Factor ≥ 1.5: {'✅' if result.profit_factor >= 1.5 else '❌'} ({result.profit_factor:.2f})")


def print_comparison(ml_result: BacktestResult, oracle_result: BacktestResult, indicator: str):
    """Affiche la comparaison ML vs Oracle."""
    print("\n" + "="*70)
    print(f"COMPARAISON ML vs ORACLE - {indicator.upper()}")
    print("="*70)

    print(f"\n{'Métrique':<20} {'ML':>15} {'Oracle':>15} {'Delta':>15}")
    print("-"*70)

    # Trades
    print(f"{'Trades':<20} {ml_result.n_trades:>15,} {oracle_result.n_trades:>15,} {ml_result.n_trades - oracle_result.n_trades:>+15,}")

    # Win Rate
    delta_wr = (ml_result.win_rate - oracle_result.win_rate) * 100
    print(f"{'Win Rate':<20} {ml_result.win_rate*100:>14.1f}% {oracle_result.win_rate*100:>14.1f}% {delta_wr:>+14.1f}%")

    # PnL Net
    delta_pnl = (ml_result.total_pnl_after_fees - oracle_result.total_pnl_after_fees) * 100
    print(f"{'PnL Net':<20} {ml_result.total_pnl_after_fees*100:>+14.2f}% {oracle_result.total_pnl_after_fees*100:>+14.2f}% {delta_pnl:>+14.2f}%")

    # Profit Factor
    delta_pf = ml_result.profit_factor - oracle_result.profit_factor
    print(f"{'Profit Factor':<20} {ml_result.profit_factor:>15.2f} {oracle_result.profit_factor:>15.2f} {delta_pf:>+15.2f}")

    # Sharpe
    delta_sharpe = ml_result.sharpe_ratio - oracle_result.sharpe_ratio
    print(f"{'Sharpe Ratio':<20} {ml_result.sharpe_ratio:>15.2f} {oracle_result.sharpe_ratio:>15.2f} {delta_sharpe:>+15.2f}")

    # Analyse
    print(f"\nAnalyse:")
    pct_of_oracle_wr = (ml_result.win_rate / oracle_result.win_rate * 100) if oracle_result.win_rate > 0 else 0
    print(f"  ML capture {pct_of_oracle_wr:.1f}% du Win Rate Oracle")

    if ml_result.total_pnl_after_fees > 0:
        pct_of_oracle_pnl = (ml_result.total_pnl_after_fees / oracle_result.total_pnl_after_fees * 100) if oracle_result.total_pnl_after_fees > 0 else 0
        print(f"  ML capture {pct_of_oracle_pnl:.1f}% du PnL Net Oracle")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test ML Predictions Direction-Only')
    parser.add_argument('--indicator', type=str, default='macd',
                        choices=['macd', 'rsi', 'cci'],
                        help='Indicateur à tester')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Split à utiliser')
    parser.add_argument('--fees', type=float, default=0.001,
                        help='Frais par side (défaut: 0.1%%)')
    parser.add_argument('--compare-oracle', action='store_true',
                        help='Comparer avec Oracle')

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("TEST ML PREDICTIONS DIRECTION-ONLY")
    logger.info("="*70)
    logger.info(f"Indicateur: {args.indicator.upper()}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Frais: {args.fees*100:.2f}% par side")

    # Charger données
    data = load_dataset(args.indicator, args.split)

    # Extraire prédictions ML et OHLCV
    Y_pred = data['Y_pred']
    OHLCV = data['OHLCV']

    # Convertir probabilités en labels binaires (>0.5 = UP)
    if Y_pred.ndim > 1:
        # Si Y_pred = (n, 1) ou (n, k), prendre première colonne
        ml_labels = (Y_pred[:, 0] > 0.5).astype(int)
    else:
        # Si Y_pred = (n,)
        ml_labels = (Y_pred > 0.5).astype(int)

    logger.info(f"\nDonnées:")
    logger.info(f"  Samples: {len(ml_labels):,}")
    logger.info(f"  ML Prédictions UP: {(ml_labels == 1).sum():,} ({(ml_labels == 1).mean()*100:.1f}%)")
    logger.info(f"  ML Prédictions DOWN: {(ml_labels == 0).sum():,} ({(ml_labels == 0).mean()*100:.1f}%)")

    # Backtest ML
    ml_result, ml_asset_results = backtest(ml_labels, OHLCV, fees=args.fees)

    # Afficher résultats ML
    print_results(ml_result, args.indicator)

    # Afficher résultats par asset
    print_asset_results(ml_asset_results)

    # Afficher résultats par mois
    print_monthly_results(ml_result.trades)

    # Comparaison avec Oracle si demandé
    if args.compare_oracle:
        Y = data['Y']
        oracle_labels = Y[:, 2].astype(int)
        oracle_result, _ = backtest(oracle_labels, OHLCV, fees=args.fees)
        print_comparison(ml_result, oracle_result, args.indicator)


if __name__ == '__main__':
    main()
