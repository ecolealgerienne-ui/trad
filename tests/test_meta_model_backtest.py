#!/usr/bin/env python3
"""
Test Meta-Model Backtest - Phase 2.18

Backtest avec filtrage meta-model pour valider l'impact du filtrage
sur les trades (réduction trades, amélioration Win Rate, PnL Net positif).

Pipeline SIMPLIFIÉ:
1. Charger meta_labels_*.npz (contient TOUT: predictions, OHLCV, meta_labels)
2. Charger meta-model (Logistic Regression)
3. Construire features meta (6: probs + confidence + ATR)
4. Backt avec filtrage par threshold
5. Comparer stratégies (baseline vs thresholds 0.5, 0.6, 0.7)

Usage:
    # Comparer toutes les stratégies
    python tests/test_meta_model_backtest.py --indicator macd --split test --compare-thresholds

    # Test un seul threshold
    python tests/test_meta_model_backtest.py --indicator macd --split test --threshold 0.6
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import IntEnum
import logging
import joblib

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
    asset_id: int = 0
    entry_timestamp: float = 0.0
    meta_prob: float = 0.0  # Probabilité meta-model


@dataclass
class BacktestResult:
    """Résultats du backtest."""
    strategy_name: str
    n_trades: int
    n_long: int
    n_short: int
    n_filtered: int  # Trades bloqués par meta-filter
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


# =============================================================================
# CHARGEMENT DONNÉES
# =============================================================================

def load_meta_labels_data(indicator: str, filter_type: str = 'kalman', split: str = 'test', aligned: bool = False) -> Dict:
    """
    Charge TOUTES les données depuis meta_labels_*.npz.

    Ce fichier contient TOUT ce dont on a besoin:
    - predictions (MACD, RSI, CCI)
    - OHLCV (pour backtest)
    - meta_labels (pour validation)

    Args:
        indicator: 'macd', 'rsi', ou 'cci'
        filter_type: 'kalman' ou 'octave'
        split: 'train', 'val', ou 'test'
        aligned: Si True, charge labels aligned (signal reversal)

    Returns:
        Dict avec predictions, OHLCV, meta_labels
    """
    suffix = '_aligned' if aligned else ''
    path = Path(f'data/prepared/meta_labels_{indicator}_{filter_type}_{split}{suffix}.npz')

    if not path.exists():
        raise FileNotFoundError(
            f"Meta-labels introuvables: {path}\n\n"
            f"SOLUTION: Générer avec:\n"
            f"  python src/create_meta_labels_phase215.py \\\n"
            f"    --indicator {indicator} --filter {filter_type} --split {split}"
        )

    logger.info(f"Chargement données: {path}")
    data = np.load(path, allow_pickle=True)

    # Extraire predictions (3 indicateurs)
    predictions_macd = data['predictions_macd']
    predictions_rsi = data['predictions_rsi']
    predictions_cci = data['predictions_cci']

    # OHLCV pour backtest
    ohlcv = data['OHLCV']

    # Meta-labels (pour validation)
    meta_labels = data['meta_labels']

    logger.info(f"  Predictions MACD: {predictions_macd.shape}")
    logger.info(f"  Predictions RSI: {predictions_rsi.shape}")
    logger.info(f"  Predictions CCI: {predictions_cci.shape}")
    logger.info(f"  OHLCV: {ohlcv.shape}")
    logger.info(f"  Meta-labels: {meta_labels.shape}")
    logger.info(f"  Positive: {np.sum(meta_labels == 1)}")
    logger.info(f"  Negative: {np.sum(meta_labels == 0)}")

    return {
        'predictions_macd': predictions_macd,
        'predictions_rsi': predictions_rsi,
        'predictions_cci': predictions_cci,
        'ohlcv': ohlcv,
        'meta_labels': meta_labels
    }


def load_meta_model(filter_type: str = 'kalman', aligned: bool = False, model_type: str = 'logistic') -> object:
    """Charge le meta-model entraîné."""
    suffix = '_aligned' if aligned else ''
    # Note: train script sauve avec model_name = args.model directement
    model_name = model_type  # 'logistic', 'xgboost', 'random_forest'
    path = Path(f'models/meta_model/meta_model_{model_name}_{filter_type}{suffix}.pkl')

    if not path.exists():
        aligned_flag = ' --aligned' if aligned else ''
        model_flag = f' --model {model_type}' if model_type != 'logistic' else ''
        raise FileNotFoundError(
            f"Meta-model introuvable: {path}\n\n"
            f"SOLUTION: Entraîner avec:\n"
            f"  python src/train_meta_model_phase217.py --filter {filter_type}{aligned_flag}{model_flag}"
        )

    logger.info(f"Chargement meta-model: {path}")
    model = joblib.load(path)
    logger.info(f"  Type: {type(model).__name__}")
    logger.info(f"  Classes: {model.classes_}")

    return model


def calculate_atr(ohlcv: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calcule l'ATR (Average True Range) normalisé.

    Args:
        ohlcv: Array (n, 7) [timestamp, asset_id, O, H, L, C, V]
        period: Période ATR (défaut 14)

    Returns:
        ATR normalisé (n,) - ATR / Close
    """
    highs = ohlcv[:, 3]  # H
    lows = ohlcv[:, 4]   # L
    closes = ohlcv[:, 5] # C

    # True Range = max(H-L, |H-C_prev|, |L-C_prev|)
    tr1 = highs - lows
    tr2 = np.abs(highs - np.roll(closes, 1))
    tr3 = np.abs(lows - np.roll(closes, 1))
    tr2[0] = tr1[0]  # Pas de précédent pour le premier
    tr3[0] = tr1[0]

    true_range = np.maximum.reduce([tr1, tr2, tr3])

    # ATR = EMA du True Range
    atr = np.zeros_like(true_range)
    atr[0] = true_range[0]

    alpha = 1.0 / period
    for i in range(1, len(true_range)):
        atr[i] = alpha * true_range[i] + (1 - alpha) * atr[i-1]

    # Normaliser par le prix
    atr_normalized = atr / closes

    return atr_normalized


def build_meta_features(
    predictions_macd: np.ndarray,
    predictions_rsi: np.ndarray,
    predictions_cci: np.ndarray,
    ohlcv: np.ndarray
) -> np.ndarray:
    """
    Construit les 6 features meta depuis predictions + OHLCV.

    Args:
        predictions_*: Probabilités des modèles primaires (n,)
        ohlcv: Données OHLCV (n, 7)

    Returns:
        Features (n, 6):
            [macd_prob, rsi_prob, cci_prob,
             confidence_spread, confidence_mean, volatility_atr]
    """
    logger.info("Construction features meta (6)...")

    # 1. Probabilités primaires
    macd_prob = predictions_macd
    rsi_prob = predictions_rsi
    cci_prob = predictions_cci

    # 2. Confidence metrics
    probs = np.stack([macd_prob, rsi_prob, cci_prob], axis=1)  # (n, 3)
    confidence_spread = np.max(probs, axis=1) - np.min(probs, axis=1)  # (n,)
    confidence_mean = np.mean(probs, axis=1)  # (n,)

    # 3. Volatilité ATR
    volatility_atr = calculate_atr(ohlcv, period=14)

    # 4. Stack features
    X_meta = np.stack([
        macd_prob,
        rsi_prob,
        cci_prob,
        confidence_spread,
        confidence_mean,
        volatility_atr
    ], axis=1)  # (n, 6)

    logger.info(f"  Features shape: {X_meta.shape}")
    logger.info(f"  MACD prob: [{macd_prob.min():.3f}, {macd_prob.max():.3f}]")
    logger.info(f"  Confidence spread: [{confidence_spread.min():.3f}, {confidence_spread.max():.3f}]")
    logger.info(f"  ATR: [{volatility_atr.min():.6f}, {volatility_atr.max():.6f}]")

    return X_meta


# =============================================================================
# BACKTEST
# =============================================================================

def backtest_with_meta_filter(
    primary_indicator: str,
    predictions_primary: np.ndarray,
    meta_features: np.ndarray,
    meta_model: object,
    ohlcv: np.ndarray,
    threshold: float = 0.5,
    fees: float = 0.001
) -> BacktestResult:
    """
    Backtest avec filtrage meta-model.

    LOGIC:
    1. Primary prediction at index i (MACD → UP or DOWN)
    2. Meta-prediction: Will this trade be profitable?
    3. If meta_prob > threshold → Execute at Open[i+1]
    4. Else → HOLD (no trade, no fees)

    Args:
        primary_indicator: 'macd', 'rsi', ou 'cci'
        predictions_primary: Prédictions primaires (n,) - probas [0,1]
        meta_features: Features meta (n, 6)
        meta_model: Meta-model entraîné
        ohlcv: Données OHLCV (n, 7) - [timestamp, asset_id, O, H, L, C, V]
        threshold: Seuil meta-prob (défaut 0.5)
        fees: Frais par trade (défaut 0.001 = 0.1%)

    Returns:
        BacktestResult avec tous les trades et métriques
    """
    strategy_name = f"Meta-Filter (threshold={threshold:.1f})" if threshold > 0 else "Baseline (no filter)"
    logger.info(f"\n{'='*80}")
    logger.info(f"BACKTEST: {strategy_name}")
    logger.info(f"{'='*80}")

    # Générer meta-probabilities pour tous les timesteps
    meta_probs = meta_model.predict_proba(meta_features)[:, 1]
    logger.info(f"Meta-probs: [{meta_probs.min():.3f}, {meta_probs.max():.3f}]")

    # Convertir primary predictions en directions
    primary_directions = (predictions_primary > 0.5).astype(int)

    # Extraire colonnes OHLCV
    timestamps = ohlcv[:, 0]
    asset_ids = ohlcv[:, 1]
    opens = ohlcv[:, 2]

    # Backtest per asset (évite cross-contamination)
    all_trades = []
    n_filtered = 0

    unique_assets = np.unique(asset_ids)
    logger.info(f"Assets: {len(unique_assets)} ({unique_assets})")

    for asset_id in unique_assets:
        mask = asset_ids == asset_id
        asset_preds = primary_directions[mask]
        asset_meta_probs = meta_probs[mask]
        asset_opens = opens[mask]
        asset_timestamps = timestamps[mask]

        n_asset = len(asset_preds)
        position = Position.FLAT
        entry_idx = -1
        entry_price = 0.0
        entry_meta_prob = 0.0

        for i in range(n_asset - 1):
            direction = int(asset_preds[i])
            meta_prob = asset_meta_probs[i]
            target = Position.LONG if direction == 1 else Position.SHORT

            # CAS 1: FLAT - décider si entrer
            if position == Position.FLAT:
                if meta_prob > threshold:
                    # Entrer en position
                    position = target
                    entry_idx = i
                    entry_price = asset_opens[i + 1]
                    entry_meta_prob = meta_prob
                else:
                    # Rester FLAT (pas de trade, filtré)
                    n_filtered += 1
                continue

            # CAS 2: EN POSITION - vérifier si sortir
            if position != target:
                # TOUJOURS sortir (signal a changé, protéger capital)
                exit_idx = i
                exit_price = asset_opens[i + 1]
                duration = exit_idx - entry_idx

                # Calculate PnL avec entry_price SAUVEGARDÉ
                if position == Position.LONG:
                    pnl = (exit_price - entry_price) / entry_price
                else:  # SHORT
                    pnl = (entry_price - exit_price) / entry_price

                # Frais: entrée + sortie
                trade_fees = 2 * fees
                pnl_after_fees = pnl - trade_fees

                trade = Trade(
                    entry_idx=entry_idx,
                    exit_idx=exit_idx,
                    duration=duration,
                    position='LONG' if position == Position.LONG else 'SHORT',
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl=pnl,
                    pnl_after_fees=pnl_after_fees,
                    asset_id=int(asset_id),
                    entry_timestamp=asset_timestamps[entry_idx],
                    meta_prob=entry_meta_prob
                )
                all_trades.append(trade)

                # RETOUR À FLAT
                position = Position.FLAT

                # Décider si nouvelle entrée immédiate
                if meta_prob > threshold:
                    position = target
                    entry_idx = i
                    entry_price = asset_opens[i + 1]
                    entry_meta_prob = meta_prob
                else:
                    # Rester FLAT (filtré)
                    n_filtered += 1

        # Close final position (if any)
        if position != Position.FLAT:
            exit_idx = n_asset - 1
            exit_price = asset_opens[-1]
            duration = exit_idx - entry_idx

            if position == Position.LONG:
                pnl = (exit_price - entry_price) / entry_price
            else:
                pnl = (entry_price - exit_price) / entry_price

            trade_fees = 2 * fees
            pnl_after_fees = pnl - trade_fees

            trade = Trade(
                entry_idx=entry_idx,
                exit_idx=exit_idx,
                duration=duration,
                position='LONG' if position == Position.LONG else 'SHORT',
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                pnl_after_fees=pnl_after_fees,
                asset_id=int(asset_id),
                entry_timestamp=asset_timestamps[entry_idx],
                meta_prob=entry_meta_prob
            )
            all_trades.append(trade)

    # Calculate metrics
    return calculate_metrics(strategy_name, all_trades, n_filtered)


def calculate_metrics(strategy_name: str, trades: List[Trade], n_filtered: int) -> BacktestResult:
    """Calcule les métriques du backtest."""
    if not trades:
        return BacktestResult(
            strategy_name=strategy_name,
            n_trades=0,
            n_long=0,
            n_short=0,
            n_filtered=n_filtered,
            total_pnl=0.0,
            total_pnl_after_fees=0.0,
            total_fees=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_duration=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            trades=[]
        )

    n_trades = len(trades)
    n_long = sum(1 for t in trades if t.position == 'LONG')
    n_short = sum(1 for t in trades if t.position == 'SHORT')

    pnls = np.array([t.pnl_after_fees for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    total_pnl = sum(t.pnl for t in trades)
    total_pnl_after_fees = sum(t.pnl_after_fees for t in trades)
    total_fees = total_pnl - total_pnl_after_fees

    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0.0

    # Profit Factor
    sum_wins = wins.sum() if len(wins) > 0 else 0.0
    sum_losses = -losses.sum() if len(losses) > 0 else 0.0
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else 0.0

    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    avg_duration = np.mean([t.duration for t in trades])

    # Sharpe Ratio
    sharpe_ratio = pnls.mean() / pnls.std() if pnls.std() > 0 else 0.0

    # Max Drawdown
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min()

    return BacktestResult(
        strategy_name=strategy_name,
        n_trades=n_trades,
        n_long=n_long,
        n_short=n_short,
        n_filtered=n_filtered,
        total_pnl=total_pnl,
        total_pnl_after_fees=total_pnl_after_fees,
        total_fees=total_fees,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_duration=avg_duration,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        trades=trades
    )


def print_results(result: BacktestResult):
    """Affiche les résultats du backtest."""
    print(f"\n{result.strategy_name}")
    print(f"{'='*80}")
    print(f"Trades: {result.n_trades:,} (LONG: {result.n_long:,}, SHORT: {result.n_short:,})")
    print(f"Filtrés: {result.n_filtered:,}")
    print(f"Win Rate: {result.win_rate:.2f}%")
    print(f"PnL Brut: {result.total_pnl * 100:+.2f}%")
    print(f"PnL Net: {result.total_pnl_after_fees * 100:+.2f}%")
    print(f"Frais: {result.total_fees * 100:.2f}%")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Avg Win: {result.avg_win * 100:+.3f}%")
    print(f"Avg Loss: {result.avg_loss * 100:+.3f}%")
    print(f"Avg Duration: {result.avg_duration:.1f} périodes")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown * 100:.2f}%")


def compare_strategies(
    primary_indicator: str,
    data: Dict,
    meta_model: object,
    thresholds: List[float],
    fees: float
):
    """Compare baseline vs multiple thresholds."""
    print(f"\n{'='*80}")
    print(f"COMPARAISON DES STRATÉGIES")
    print(f"{'='*80}")

    # Sélectionner primary predictions
    predictions_primary = data[f'predictions_{primary_indicator}']

    # Build meta-features
    meta_features = build_meta_features(
        data['predictions_macd'],
        data['predictions_rsi'],
        data['predictions_cci'],
        data['ohlcv']
    )

    results = []

    # Baseline (no filtering)
    result = backtest_with_meta_filter(
        primary_indicator=primary_indicator,
        predictions_primary=predictions_primary,
        meta_features=meta_features,
        meta_model=meta_model,
        ohlcv=data['ohlcv'],
        threshold=0.0,  # Accept all trades
        fees=fees
    )
    results.append(result)
    print_results(result)

    # Test each threshold
    for threshold in thresholds:
        result = backtest_with_meta_filter(
            primary_indicator=primary_indicator,
            predictions_primary=predictions_primary,
            meta_features=meta_features,
            meta_model=meta_model,
            ohlcv=data['ohlcv'],
            threshold=threshold,
            fees=fees
        )
        results.append(result)
        print_results(result)

    # Summary table
    print(f"\n{'='*80}")
    print(f"RÉSUMÉ COMPARATIF")
    print(f"{'='*80}")
    print(f"{'Stratégie':<30} {'Trades':>10} {'Filtrés':>10} {'WR%':>8} {'PnL Net%':>12}")
    print(f"{'-'*80}")

    for r in results:
        print(f"{r.strategy_name:<30} {r.n_trades:>10,} {r.n_filtered:>10,} {r.win_rate:>8.2f} {r.total_pnl_after_fees * 100:>12.2f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Backtest meta-model filtering')
    parser.add_argument('--indicator', type=str, default='macd', choices=['macd', 'rsi', 'cci'],
                        help='Indicateur primaire (défaut: macd)')
    parser.add_argument('--filter', type=str, default='kalman', choices=['kalman', 'octave'],
                        help='Type de filtre (défaut: kalman)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Split à tester (défaut: test)')
    parser.add_argument('--aligned', action='store_true',
                        help='Use aligned meta-model (signal reversal labels)')
    parser.add_argument('--model', type=str, default='logistic',
                        choices=['logistic', 'xgboost', 'random_forest'],
                        help='Model type: logistic (baseline), xgboost, or random_forest (défaut: logistic)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Seuil unique à tester (défaut: None)')
    parser.add_argument('--compare-thresholds', action='store_true',
                        help='Comparer plusieurs thresholds (0.5, 0.6, 0.7)')
    parser.add_argument('--fees', type=float, default=0.001,
                        help='Frais par trade (défaut: 0.001 = 0.1%%)')

    args = parser.parse_args()

    print(f"{'='*80}")
    print(f"META-MODEL BACKTEST - Phase 2.18")
    print(f"{'='*80}")
    print(f"Indicateur primaire: {args.indicator}")
    print(f"Filter: {args.filter}")
    print(f"Split: {args.split}")
    print(f"Labels: {'aligned (signal reversal)' if args.aligned else 'Triple Barrier'}")
    print(f"Model: {args.model}")
    print(f"Frais: {args.fees * 100:.2f}%")

    # Load data (TOUT depuis meta_labels_*.npz)
    data = load_meta_labels_data(args.indicator, args.filter, args.split, aligned=args.aligned)

    # Load meta-model
    meta_model = load_meta_model(args.filter, aligned=args.aligned, model_type=args.model)

    if args.compare_thresholds:
        # Compare multiple thresholds
        thresholds = [0.5, 0.6, 0.7]
        compare_strategies(
            primary_indicator=args.indicator,
            data=data,
            meta_model=meta_model,
            thresholds=thresholds,
            fees=args.fees
        )

    elif args.threshold is not None:
        # Test single threshold
        predictions_primary = data[f'predictions_{args.indicator}']

        meta_features = build_meta_features(
            data['predictions_macd'],
            data['predictions_rsi'],
            data['predictions_cci'],
            data['ohlcv']
        )

        result = backtest_with_meta_filter(
            primary_indicator=args.indicator,
            predictions_primary=predictions_primary,
            meta_features=meta_features,
            meta_model=meta_model,
            ohlcv=data['ohlcv'],
            threshold=args.threshold,
            fees=args.fees
        )
        print_results(result)

    else:
        print("ERROR: Spécifier --threshold ou --compare-thresholds")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"✅ BACKTEST TERMINÉ")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
