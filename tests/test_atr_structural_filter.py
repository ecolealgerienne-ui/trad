#!/usr/bin/env python3
"""
Test Stratégie ATR Structural Filter - Filtrer trades par régime de volatilité.

VERSION 2.0 - Corrigé avec prix Open réels (pas c_ret!)

HYPOTHÈSE (López de Prado 2018):
Ne trader QUE dans les régimes de volatilité "sains" (ni trop basse, ni trop haute).
- ATR < Q20: Marché trop calme (ranging, signaux faibles)
- Q20 < ATR < Q80: Volatilité saine (conditions optimales)
- ATR > Q80: Volatilité extrême (gaps, slippage élevé)

CORRECTIONS vs v1.0:
- Utilise prix Open réels (OHLCV[:, 2]) au lieu de c_ret
- Backtest PAR ASSET (pas global sur dataset concaténé)
- Calcul PnL: (exit_price - entry_price) / entry_price
- Intègre structure de test_oracle_direction_only.py

Usage:
    python tests/test_atr_structural_filter.py --indicator macd --split test --fees 0.001
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import argparse
from dataclasses import dataclass
from typing import List, Dict
from enum import IntEnum
from datetime import datetime
from collections import defaultdict
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


# Percentiles à tester (low, high)
PERCENTILE_CONFIGS = [
    (0, 100, "Baseline"),   # Tout autoriser (référence)
    (10, 90, "Large"),      # Large (garde 80%)
    (20, 80, "Standard"),   # Standard (garde 60%)
    (30, 70, "Strict"),     # Strict (garde 40%)
]


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
    asset_id: int = 0
    entry_timestamp: float = 0.0
    exit_reason: str = ""  # "DIRECTION_FLIP", "ATR_OUT", "END"


@dataclass
class BacktestResult:
    """Résultats du backtest."""
    name: str
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
    atr_coverage: float  # % samples autorisés par ATR
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


# =============================================================================
# CHARGEMENT DONNÉES
# =============================================================================

def load_dataset(indicator: str, split: str = 'test') -> Dict:
    """
    Charge le dataset direction-only avec OHLCV.

    RÉUTILISE: Structure de test_oracle_direction_only.py
    """
    path = Path(f'data/prepared/dataset_btc_eth_bnb_ada_ltc_{indicator}_direction_only_kalman.npz')

    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {path}")

    logger.info(f"Chargement: {path}")
    data = np.load(path, allow_pickle=True)

    # Extraire données du split
    Y = data[f'Y_{split}']
    OHLCV = data[f'OHLCV_{split}']
    Y_pred = data.get(f'Y_{split}_pred', None)

    logger.info(f"  Y shape: {Y.shape} - [timestamp, asset_id, direction]")
    logger.info(f"  OHLCV shape: {OHLCV.shape} - [timestamp, asset_id, O, H, L, C, V]")

    return {
        'Y': Y,
        'OHLCV': OHLCV,
        'Y_pred': Y_pred
    }


# =============================================================================
# CALCUL ATR
# =============================================================================

def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Calcule l'ATR normalisé (ATR / Close).

    Args:
        high: Prix High
        low: Prix Low
        close: Prix Close
        window: Période ATR (défaut: 14)

    Returns:
        ATR normalisé (%)
    """
    n = len(close)
    tr = np.zeros(n)

    # True Range
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)

    # ATR (EMA du True Range)
    atr = np.zeros(n)
    atr[:window] = np.nan
    atr[window-1] = np.mean(tr[:window])

    multiplier = 2 / (window + 1)
    for i in range(window, n):
        atr[i] = tr[i] * multiplier + atr[i-1] * (1 - multiplier)

    # Normaliser par Close
    atr_norm = atr / close

    return atr_norm


def compute_atr_per_asset(ohlcv: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Calcule l'ATR normalisé PAR ASSET.

    IMPORTANT: Calculer ATR par asset pour éviter les discontinuités
    entre assets concaténés dans le dataset.

    Args:
        ohlcv: (n, 7) - [timestamp, asset_id, O, H, L, C, V]
        window: Période ATR

    Returns:
        ATR array (n,)
    """
    asset_ids = ohlcv[:, 1].astype(int)
    unique_assets = np.unique(asset_ids)

    highs = ohlcv[:, 3]
    lows = ohlcv[:, 4]
    closes = ohlcv[:, 5]

    atr = np.zeros(len(ohlcv))

    for asset_id in unique_assets:
        mask = asset_ids == asset_id
        indices = np.where(mask)[0]

        asset_high = highs[mask]
        asset_low = lows[mask]
        asset_close = closes[mask]

        asset_atr = calculate_atr(asset_high, asset_low, asset_close, window)
        atr[indices] = asset_atr

    # Remplacer NaN par médiane (warmup period)
    valid_atr = atr[~np.isnan(atr)]
    if len(valid_atr) > 0:
        atr[np.isnan(atr)] = np.median(valid_atr)

    return atr


# =============================================================================
# BACKTEST PAR ASSET (RÉUTILISE test_oracle_direction_only.py)
# =============================================================================

def backtest_single_asset(
    labels: np.ndarray,
    opens: np.ndarray,
    timestamps: np.ndarray,
    atr_values: np.ndarray,
    atr_mask: np.ndarray,
    asset_id: int,
    fees: float = 0.001
) -> List[Trade]:
    """
    Backtest pour UN SEUL asset avec filtre ATR.

    LOGIQUE CAUSALE (RÉUTILISE test_oracle_direction_only.py):
    - Signal à index i → Exécution à Open[i+1]
    - Toujours en position (LONG ou SHORT, jamais FLAT sauf si ATR bloque)
    - Direction: 1=UP→LONG, 0=DOWN→SHORT
    - ATR: Trade autorisé seulement si atr_mask[i] == True

    Args:
        labels: (n,) Direction labels
        opens: (n,) Prix Open
        timestamps: (n,) Timestamps
        atr_values: (n,) ATR normalisé
        atr_mask: (n,) True si ATR dans range acceptable
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

    for i in range(n_samples - 1):
        direction = int(labels[i])
        atr_ok = atr_mask[i]

        # Target position basée sur direction ET ATR
        if atr_ok:
            target = Position.LONG if direction == 1 else Position.SHORT
        else:
            # ATR hors range → ne pas entrer ou sortir si déjà en position
            target = Position.FLAT

        # Première entrée
        if position == Position.FLAT:
            if target != Position.FLAT:
                position = target
                entry_idx = i
                entry_price = opens[i + 1]
                entry_timestamp = timestamps[i + 1]
            continue

        # Gestion position existante
        exit_signal = False
        exit_reason = ""

        # Cas 1: Direction flip (même si ATR OK)
        if target != Position.FLAT and target != position:
            exit_signal = True
            exit_reason = "DIRECTION_FLIP"

        # Cas 2: ATR sort du range → sortie forcée
        elif not atr_ok:
            exit_signal = True
            exit_reason = "ATR_OUT"

        if exit_signal:
            exit_price = opens[i + 1]

            # Calcul PnL (CORRECT: prix réels!)
            if position == Position.LONG:
                pnl = (exit_price - entry_price) / entry_price
            else:  # SHORT
                pnl = (entry_price - exit_price) / entry_price

            trade_fees = 2 * fees
            pnl_after_fees = pnl - trade_fees

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
                entry_timestamp=entry_timestamp,
                exit_reason=exit_reason
            ))

            # Flip immédiat si DIRECTION_FLIP
            if exit_reason == "DIRECTION_FLIP":
                position = target
                entry_idx = i
                entry_price = opens[i + 1]
                entry_timestamp = timestamps[i + 1]
            else:
                # ATR_OUT → retour FLAT
                position = Position.FLAT
                entry_price = 0.0

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
            entry_timestamp=entry_timestamp,
            exit_reason="END"
        ))

    return trades


def backtest_with_atr_filter(
    labels: np.ndarray,
    ohlcv: np.ndarray,
    atr: np.ndarray,
    fees: float = 0.001,
    atr_q_low: float = 0.0,
    atr_q_high: float = 100.0
) -> tuple:
    """
    Backtest avec filtre ATR, PAR ASSET.

    RÉUTILISE: Structure de test_oracle_direction_only.py

    Args:
        labels: (n,) Direction labels
        ohlcv: (n, 7) Prix [timestamp, asset_id, O, H, L, C, V]
        atr: (n,) ATR normalisé
        fees: Frais par side
        atr_q_low: Percentile bas ATR
        atr_q_high: Percentile haut ATR

    Returns:
        (BacktestResult, List[AssetResult])
    """
    # Calculer percentiles ATR
    valid_atr = atr[~np.isnan(atr)]
    if len(valid_atr) == 0:
        raise ValueError("Aucune valeur ATR valide!")

    q_low_val = np.percentile(valid_atr, atr_q_low)
    q_high_val = np.percentile(valid_atr, atr_q_high)

    # Masque ATR
    if atr_q_low == 0.0 and atr_q_high == 100.0:
        atr_mask = np.ones(len(atr), dtype=bool)
    else:
        atr_mask = (atr >= q_low_val) & (atr <= q_high_val)

    atr_coverage = atr_mask.mean()

    logger.info(f"  ATR Q{atr_q_low:.0f}={q_low_val:.4f}, Q{atr_q_high:.0f}={q_high_val:.4f}")
    logger.info(f"  ATR Coverage: {atr_coverage*100:.1f}%")

    # Extraire colonnes OHLCV
    timestamps = ohlcv[:, 0]
    asset_ids = ohlcv[:, 1].astype(int)
    opens = ohlcv[:, 2]

    unique_assets = np.unique(asset_ids)
    logger.info(f"  Assets détectés: {len(unique_assets)}")

    all_trades = []
    asset_results = []
    n_long = 0
    n_short = 0

    # Backtest PAR ASSET
    for asset_id in unique_assets:
        mask = asset_ids == asset_id
        asset_labels = labels[mask]
        asset_opens = opens[mask]
        asset_timestamps = timestamps[mask]
        asset_atr = atr[mask]
        asset_atr_mask = atr_mask[mask]

        trades = backtest_single_asset(
            asset_labels, asset_opens, asset_timestamps,
            asset_atr, asset_atr_mask,
            int(asset_id), fees
        )

        # Stats par asset
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

        if len(trades) > 0:
            asset_results.append(AssetResult(
                asset_id=int(asset_id),
                n_trades=len(trades),
                total_pnl=asset_pnl,
                total_pnl_after_fees=asset_pnl_net,
                win_rate=asset_wins / len(trades),
                avg_duration=asset_duration / len(trades)
            ))

        all_trades.extend(trades)

    # Calculer stats globales
    result = compute_stats(all_trades, n_long, n_short, atr_coverage, atr_q_low, atr_q_high)
    return result, asset_results


def compute_stats(
    trades: List[Trade],
    n_long: int,
    n_short: int,
    atr_coverage: float,
    atr_q_low: float,
    atr_q_high: float
) -> BacktestResult:
    """Calcule les statistiques du backtest."""
    name = f"ATR_Q{int(atr_q_low)}-Q{int(atr_q_high)}"

    if len(trades) == 0:
        return BacktestResult(
            name=name,
            n_trades=0, n_long=0, n_short=0,
            total_pnl=0.0, total_pnl_after_fees=0.0, total_fees=0.0,
            win_rate=0.0, profit_factor=0.0,
            avg_win=0.0, avg_loss=0.0, avg_duration=0.0,
            sharpe_ratio=0.0, atr_coverage=atr_coverage, trades=[]
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
    win_rate = wins.mean()

    # Profit factor
    sum_wins = pnls_net[wins].sum() if wins.any() else 0.0
    sum_losses = abs(pnls_net[losses].sum()) if losses.any() else 0.0
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else 0.0

    # Moyennes
    avg_win = pnls_net[wins].mean() if wins.any() else 0.0
    avg_loss = pnls_net[losses].mean() if losses.any() else 0.0
    avg_duration = durations.mean()

    # Sharpe
    if len(pnls_net) > 1 and pnls_net.std() > 0:
        sharpe = (pnls_net.mean() / pnls_net.std()) * np.sqrt(288 * 365)
    else:
        sharpe = 0.0

    return BacktestResult(
        name=name,
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
        atr_coverage=atr_coverage,
        trades=trades
    )


# =============================================================================
# AFFICHAGE RÉSULTATS
# =============================================================================

def print_results(result: BacktestResult, indicator: str):
    """Affiche les résultats détaillés."""
    print(f"\n{'='*70}")
    print(f"RÉSULTATS {result.name} - {indicator.upper()}")
    print("="*70)

    print(f"\nTrades:")
    print(f"  Total: {result.n_trades:,}")
    print(f"  Long: {result.n_long:,}")
    print(f"  Short: {result.n_short:,}")
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
    print(f"  ATR Coverage: {result.atr_coverage*100:.1f}%")


def print_asset_results(asset_results: List[AssetResult]):
    """Affiche les résultats par asset."""
    asset_names = {0: 'BTC', 1: 'ETH', 2: 'BNB', 3: 'ADA', 4: 'LTC'}

    print(f"\n{'='*70}")
    print("RÉSULTATS PAR ASSET")
    print("="*70)

    print(f"\n{'Asset':<8} {'Trades':>10} {'PnL Brut':>12} {'PnL Net':>12} {'Win Rate':>10} {'Durée':>10}")
    print("-"*70)

    for ar in asset_results:
        name = asset_names.get(ar.asset_id, f'Asset{ar.asset_id}')
        print(f"{name:<8} {ar.n_trades:>10,} {ar.total_pnl*100:>+11.2f}% {ar.total_pnl_after_fees*100:>+11.2f}% {ar.win_rate*100:>9.1f}% {ar.avg_duration:>9.1f}p")


def print_comparison_table(results: List[BacktestResult]):
    """Affiche tableau comparatif."""
    print(f"\n{'='*100}")
    print("COMPARAISON ATR FILTERS")
    print("="*100)

    baseline = results[0]

    header = f"{'Config':<18} | {'Trades':>10} | {'Réduction':>10} | {'Win Rate':>10} | {'PnL Brut':>12} | {'PnL Net':>12} | {'Coverage':>10}"
    print(header)
    print("-"*100)

    for res in results:
        if res == baseline:
            reduction = "-"
        else:
            reduction = f"{(1 - res.n_trades/baseline.n_trades)*100:+.1f}%"

        print(f"{res.name:<18} | {res.n_trades:>10,} | {reduction:>10} | {res.win_rate*100:>9.1f}% | {res.total_pnl*100:>+11.2f}% | {res.total_pnl_after_fees*100:>+11.2f}% | {res.atr_coverage*100:>9.1f}%")

    print("="*100)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test ATR Structural Filter v2.0')
    parser.add_argument('--indicator', type=str, default='macd',
                        choices=['macd', 'rsi', 'cci'],
                        help='Indicateur à tester')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Split à utiliser')
    parser.add_argument('--fees', type=float, default=0.001,
                        help='Frais par side (défaut: 0.1%%)')
    parser.add_argument('--use-predictions', action='store_true',
                        help='Utiliser prédictions ML au lieu de labels Oracle')
    parser.add_argument('--atr-window', type=int, default=14,
                        help='Période ATR (défaut: 14)')

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("TEST ATR STRUCTURAL FILTER v2.0")
    logger.info("="*70)
    logger.info(f"Indicateur: {args.indicator.upper()}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Frais: {args.fees*100:.2f}% par side")
    logger.info(f"ATR Window: {args.atr_window}")

    # Charger données
    data = load_dataset(args.indicator, args.split)

    Y = data['Y']
    OHLCV = data['OHLCV']
    Y_pred = data['Y_pred']

    # Labels: colonne 2 = direction
    labels = Y[:, 2].astype(int)

    logger.info(f"\nDonnées:")
    logger.info(f"  Samples: {len(labels):,}")
    logger.info(f"  Labels UP: {(labels == 1).sum():,} ({(labels == 1).mean()*100:.1f}%)")

    # Mode Oracle ou ML
    if args.use_predictions and Y_pred is not None:
        mode = "ML Predictions"
        pred = Y_pred[:, 0] if Y_pred.ndim > 1 else Y_pred
        labels_to_use = (pred > 0.5).astype(int)
        logger.info(f"  Mode: Prédictions ML")
    else:
        mode = "Oracle"
        labels_to_use = labels
        logger.info(f"  Mode: Oracle (labels parfaits)")

    # Calculer ATR depuis OHLCV
    logger.info(f"\nCalcul ATR...")
    atr = compute_atr_per_asset(OHLCV, window=args.atr_window)
    logger.info(f"  ATR mean: {np.nanmean(atr):.4f}, median: {np.nanmedian(atr):.4f}")

    # Tester toutes les configs ATR
    all_results = []
    all_asset_results = []

    for q_low, q_high, name in PERCENTILE_CONFIGS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Test: {name} (Q{q_low}-Q{q_high})")
        logger.info("="*60)

        result, asset_results = backtest_with_atr_filter(
            labels_to_use, OHLCV, atr,
            fees=args.fees,
            atr_q_low=q_low,
            atr_q_high=q_high
        )
        result.name = name

        all_results.append(result)
        all_asset_results.append(asset_results)

        logger.info(f"  Trades: {result.n_trades:,}")
        logger.info(f"  Win Rate: {result.win_rate*100:.1f}%")
        logger.info(f"  PnL Net: {result.total_pnl_after_fees*100:+.2f}%")

    # Afficher résultats
    print_comparison_table(all_results)

    # Afficher détails du baseline
    print_results(all_results[0], args.indicator)
    print_asset_results(all_asset_results[0])

    # Afficher meilleur non-baseline
    non_baseline = [r for r in all_results if r.name != "Baseline"]
    if non_baseline:
        best = max(non_baseline, key=lambda r: r.total_pnl_after_fees)
        print(f"\n🏆 MEILLEURE CONFIG: {best.name}")
        print_results(best, args.indicator)

        # Trouver asset results correspondant
        best_idx = all_results.index(best)
        print_asset_results(all_asset_results[best_idx])


if __name__ == '__main__':
    main()
