#!/usr/bin/env python3
"""
Test Meta-Model Backtest - Validation du filtrage meta-labeling.

OBJECTIF:
Tester l'impact du meta-model pour filtrer les trades non-profitables.
Comparer plusieurs seuils de probabilité (0.5, 0.6, 0.7) vs baseline.

ARCHITECTURE:
- Modèles primaires (MACD, RSI, CCI) → Prédisent direction (UP/DOWN)
- Meta-model (Logistic Regression) → Prédit si trade sera profitable (OUI/NON)
- Stratégie: N'agir QUE si meta-prob > threshold

LOGIQUE CAUSALE:
- Signal primaire à temps t (MACD prediction[t])
- Meta-prediction à temps t (profitable?)
- Si meta-prob > threshold → Exécution à Open[t+1]
- Sinon → HOLD (pas de trade)

STRUCTURE DONNÉES:
- dataset_*_direction_only_kalman.npz: Y_pred (prédictions MACD primaires)
- meta_labels_*_kalman_test.npz: Features meta (6) + predictions (3 indicateurs)
- models/meta_model/meta_model_baseline_kalman.pkl: Meta-model entraîné

Usage:
    # Baseline (sans filtrage)
    python tests/test_meta_model_backtest.py --indicator macd --split test

    # Avec threshold 0.6 (recommandé)
    python tests/test_meta_model_backtest.py --indicator macd --split test --threshold 0.6

    # Comparer plusieurs thresholds
    python tests/test_meta_model_backtest.py --indicator macd --split test --compare-thresholds
"""

import sys
from pathlib import Path
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

def load_primary_predictions(indicator: str, split: str = 'test') -> Dict:
    """
    Charge les prédictions du modèle primaire (MACD, RSI, ou CCI).

    Args:
        indicator: 'macd', 'rsi', ou 'cci'
        split: 'train', 'val', ou 'test'

    Returns:
        Dict avec Y_pred (prédictions primaires) et OHLCV
    """
    path = Path(f'data/prepared/dataset_btc_eth_bnb_ada_ltc_{indicator}_direction_only_kalman.npz')

    if not path.exists():
        raise FileNotFoundError(f"Dataset primaire introuvable: {path}")

    logger.info(f"Chargement prédictions primaires: {path}")
    data = np.load(path, allow_pickle=True)

    # Prédictions primaires (probabilités 0-1)
    pred_key = f'Y_{split}_pred'
    if pred_key not in data:
        raise KeyError(
            f"Prédictions primaires introuvables!\n"
            f"Clé attendue: {pred_key}\n\n"
            f"SOLUTION: Régénérer avec:\n"
            f"  python src/regenerate_predictions.py --data {path} --indicator {indicator}"
        )

    Y_pred = data[pred_key]
    Y = data[f'Y_{split}']  # Labels Oracle (pour comparaison)
    OHLCV = data[f'OHLCV_{split}']

    logger.info(f"  Y_pred shape: {Y_pred.shape} - Probabilités primaires")
    logger.info(f"  Y shape: {Y.shape} - [timestamp, asset_id, direction]")
    logger.info(f"  OHLCV shape: {OHLCV.shape}")

    return {
        'Y_pred': Y_pred,
        'Y': Y,
        'OHLCV': OHLCV
    }


def load_meta_features(indicator: str, filter_type: str = 'kalman', split: str = 'test') -> Dict:
    """
    Charge les features meta-model.

    Args:
        indicator: 'macd', 'rsi', ou 'cci'
        filter_type: 'kalman' ou 'octave'
        split: 'train', 'val', ou 'test'

    Returns:
        Dict avec features meta (6), predictions (3), meta_labels
    """
    path = Path(f'data/prepared/meta_labels_{indicator}_{filter_type}_{split}.npz')

    if not path.exists():
        raise FileNotFoundError(
            f"Meta-labels introuvables: {path}\n\n"
            f"SOLUTION: Générer avec:\n"
            f"  python src/create_meta_labels_phase215.py \\\n"
            f"    --indicator {indicator} --filter {filter_type} --split {split}"
        )

    logger.info(f"Chargement features meta: {path}")
    data = np.load(path, allow_pickle=True)

    # Extraire features meta (6 features)
    features = data['features']
    meta_labels = data['meta_labels']

    # Predictions des 3 modèles primaires
    predictions = data['predictions']

    logger.info(f"  Features shape: {features.shape} - 6 features meta")
    logger.info(f"  Meta-labels shape: {meta_labels.shape}")
    logger.info(f"  Predictions shape: {predictions.shape} - 3 indicateurs")

    return {
        'features': features,
        'meta_labels': meta_labels,
        'predictions': predictions
    }


def load_meta_model(filter_type: str = 'kalman') -> object:
    """
    Charge le meta-model entraîné.

    Args:
        filter_type: 'kalman' ou 'octave'

    Returns:
        Meta-model (Logistic Regression)
    """
    path = Path(f'models/meta_model/meta_model_baseline_{filter_type}.pkl')

    if not path.exists():
        raise FileNotFoundError(
            f"Meta-model introuvable: {path}\n\n"
            f"SOLUTION: Entraîner avec:\n"
            f"  python src/train_meta_model_phase217.py --filter {filter_type}"
        )

    logger.info(f"Chargement meta-model: {path}")
    meta_model = joblib.load(path)
    logger.info(f"  Type: {type(meta_model).__name__}")

    return meta_model


# =============================================================================
# BACKTEST AVEC META-FILTRAGE
# =============================================================================

def backtest_with_meta_filter(
    primary_preds: np.ndarray,
    meta_features: np.ndarray,
    meta_model: object,
    opens: np.ndarray,
    timestamps: np.ndarray,
    asset_ids: np.ndarray,
    threshold: float = 0.5,
    fees: float = 0.001
) -> BacktestResult:
    """
    Backtest avec filtrage meta-model.

    LOGIQUE:
    1. Prédiction primaire à index i (MACD → UP ou DOWN)
    2. Meta-prediction: Est-ce que ce trade sera profitable?
    3. Si meta_prob > threshold → Exécution à Open[i+1]
    4. Sinon → HOLD (pas de trade, pas de frais)

    Args:
        primary_preds: (n,) Probabilités primaires (0-1)
        meta_features: (n, 6) Features meta-model
        meta_model: Meta-model entraîné
        opens: (n,) Prix Open
        timestamps: (n,) Timestamps
        asset_ids: (n,) Asset IDs
        threshold: Seuil meta-prob pour agir
        fees: Frais par side

    Returns:
        BacktestResult avec métriques complètes
    """
    n_samples = len(primary_preds)

    # Prédire probabilités meta pour tous les timesteps
    meta_probs = meta_model.predict_proba(meta_features)[:, 1]  # Classe 1 = profitable

    # Convertir prédictions primaires en directions (>0.5 = UP)
    primary_directions = (primary_preds > 0.5).astype(int)

    trades = []
    position = Position.FLAT
    entry_idx = 0
    entry_price = 0.0
    entry_timestamp = 0.0
    entry_meta_prob = 0.0
    n_filtered = 0  # Compteur trades bloqués

    # Backtest par asset (pour éviter pollution entre assets)
    unique_assets = np.unique(asset_ids)
    all_trades = []

    for asset_id in unique_assets:
        # Mask pour cet asset
        mask = asset_ids == asset_id
        asset_preds = primary_directions[mask]
        asset_meta_probs = meta_probs[mask]
        asset_opens = opens[mask]
        asset_timestamps = timestamps[mask]
        n_asset = len(asset_preds)

        # Reset position pour chaque asset
        position = Position.FLAT

        for i in range(n_asset - 1):
            direction = int(asset_preds[i])
            meta_prob = asset_meta_probs[i]
            target = Position.LONG if direction == 1 else Position.SHORT

            # Première entrée
            if position == Position.FLAT:
                # Filtrage meta: n'entrer QUE si meta_prob > threshold
                if meta_prob <= threshold:
                    n_filtered += 1
                    continue

                position = target
                entry_idx = i
                entry_price = asset_opens[i + 1]
                entry_timestamp = asset_timestamps[i + 1]
                entry_meta_prob = meta_prob
                continue

            # Changement de position (reversal)
            if position != target:
                # Filtrage meta: ne reverser QUE si nouveau signal > threshold
                if meta_prob <= threshold:
                    n_filtered += 1
                    continue

                # Sortie
                exit_price = asset_opens[i + 1]

                if position == Position.LONG:
                    pnl = (exit_price - entry_price) / entry_price
                else:
                    pnl = (entry_price - exit_price) / entry_price

                trade_fees = 2 * fees
                pnl_after_fees = pnl - trade_fees

                all_trades.append(Trade(
                    entry_idx=entry_idx,
                    exit_idx=i,
                    duration=i - entry_idx,
                    position='LONG' if position == Position.LONG else 'SHORT',
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl=pnl,
                    pnl_after_fees=pnl_after_fees,
                    asset_id=int(asset_id),
                    entry_timestamp=entry_timestamp,
                    meta_prob=entry_meta_prob
                ))

                # Nouvelle position
                position = target
                entry_idx = i
                entry_price = asset_opens[i + 1]
                entry_timestamp = asset_timestamps[i + 1]
                entry_meta_prob = meta_prob

        # Fermer position finale
        if position != Position.FLAT:
            exit_price = asset_opens[-1]

            if position == Position.LONG:
                pnl = (exit_price - entry_price) / entry_price
            else:
                pnl = (entry_price - exit_price) / entry_price

            trade_fees = 2 * fees
            pnl_after_fees = pnl - trade_fees

            all_trades.append(Trade(
                entry_idx=entry_idx,
                exit_idx=n_asset - 1,
                duration=n_asset - 1 - entry_idx,
                position='LONG' if position == Position.LONG else 'SHORT',
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                pnl_after_fees=pnl_after_fees,
                asset_id=int(asset_id),
                entry_timestamp=entry_timestamp,
                meta_prob=entry_meta_prob
            ))

    # Calculer métriques
    return calculate_metrics(all_trades, n_filtered, threshold)


def calculate_metrics(trades: List[Trade], n_filtered: int, threshold: float) -> BacktestResult:
    """Calcule les métriques de backtest."""
    if not trades:
        return BacktestResult(
            strategy_name=f'Meta-Filter (threshold={threshold:.2f})',
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

    # PnL
    total_pnl = sum(t.pnl for t in trades)
    total_pnl_after_fees = sum(t.pnl_after_fees for t in trades)
    total_fees = total_pnl - total_pnl_after_fees

    # Win Rate
    winners = [t for t in trades if t.pnl_after_fees > 0]
    losers = [t for t in trades if t.pnl_after_fees <= 0]
    win_rate = len(winners) / n_trades if n_trades > 0 else 0.0

    # Profit Factor
    total_wins = sum(t.pnl_after_fees for t in winners) if winners else 0.0
    total_losses = abs(sum(t.pnl_after_fees for t in losers)) if losers else 0.0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

    # Moyennes
    avg_win = np.mean([t.pnl_after_fees for t in winners]) if winners else 0.0
    avg_loss = np.mean([t.pnl_after_fees for t in losers]) if losers else 0.0
    avg_duration = np.mean([t.duration for t in trades])

    # Sharpe Ratio (simplifié)
    pnls = np.array([t.pnl_after_fees for t in trades])
    sharpe = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0.0
    sharpe_annualized = sharpe * np.sqrt(252 * 24 * 12)  # 5min bars

    # Max Drawdown
    cumulative = np.cumsum([t.pnl_after_fees for t in trades])
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

    return BacktestResult(
        strategy_name=f'Meta-Filter (threshold={threshold:.2f})',
        n_trades=n_trades,
        n_long=n_long,
        n_short=n_short,
        n_filtered=n_filtered,
        total_pnl=total_pnl * 100,  # En %
        total_pnl_after_fees=total_pnl_after_fees * 100,
        total_fees=total_fees * 100,
        win_rate=win_rate * 100,
        profit_factor=profit_factor,
        avg_win=avg_win * 100,
        avg_loss=avg_loss * 100,
        avg_duration=avg_duration,
        sharpe_ratio=sharpe_annualized,
        max_drawdown=max_drawdown * 100,
        trades=trades
    )


def print_results(result: BacktestResult):
    """Affiche les résultats."""
    print(f"\n{'='*80}")
    print(f"RÉSULTATS: {result.strategy_name}")
    print(f"{'='*80}")
    print(f"Trades totaux:        {result.n_trades:,}")
    print(f"  - LONG:             {result.n_long:,}")
    print(f"  - SHORT:            {result.n_short:,}")
    print(f"  - Filtrés:          {result.n_filtered:,}")
    print(f"\nPnL Brut:             {result.total_pnl:+.2f}%")
    print(f"Frais totaux:         {result.total_fees:.2f}%")
    print(f"PnL Net:              {result.total_pnl_after_fees:+.2f}%")
    print(f"\nWin Rate:             {result.win_rate:.2f}%")
    print(f"Profit Factor:        {result.profit_factor:.2f}")
    print(f"Avg Win:              {result.avg_win:+.3f}%")
    print(f"Avg Loss:             {result.avg_loss:+.3f}%")
    print(f"Avg Duration:         {result.avg_duration:.1f} périodes (~{result.avg_duration*5:.0f}min)")
    print(f"\nSharpe Ratio:         {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown:         {result.max_drawdown:.2f}%")
    print(f"{'='*80}\n")


def compare_strategies(
    primary_data: Dict,
    meta_data: Dict,
    meta_model: object,
    thresholds: List[float],
    fees: float
):
    """Compare plusieurs stratégies avec différents thresholds."""
    opens = primary_data['OHLCV'][:, 2]  # Colonne Open
    timestamps = primary_data['OHLCV'][:, 0]
    asset_ids = primary_data['OHLCV'][:, 1]

    results = []

    # Baseline: Pas de filtrage meta (threshold=0.0)
    logger.info("Test Baseline (pas de filtrage meta)...")
    result = backtest_with_meta_filter(
        primary_preds=primary_data['Y_pred'],
        meta_features=meta_data['features'],
        meta_model=meta_model,
        opens=opens,
        timestamps=timestamps,
        asset_ids=asset_ids,
        threshold=0.0,
        fees=fees
    )
    result.strategy_name = 'Baseline (no filter)'
    results.append(result)
    print_results(result)

    # Tests avec différents thresholds
    for threshold in thresholds:
        logger.info(f"Test avec threshold={threshold:.2f}...")
        result = backtest_with_meta_filter(
            primary_preds=primary_data['Y_pred'],
            meta_features=meta_data['features'],
            meta_model=meta_model,
            opens=opens,
            timestamps=timestamps,
            asset_ids=asset_ids,
            threshold=threshold,
            fees=fees
        )
        results.append(result)
        print_results(result)

    # Tableau comparatif
    print(f"\n{'='*120}")
    print(f"{'COMPARAISON DES STRATÉGIES':^120}")
    print(f"{'='*120}")
    print(f"{'Stratégie':<30} {'Trades':>10} {'Filtrés':>10} {'WR%':>8} {'PnL Brut%':>12} {'PnL Net%':>12} {'PF':>8} {'Sharpe':>10}")
    print(f"{'-'*120}")

    for r in results:
        print(f"{r.strategy_name:<30} {r.n_trades:>10,} {r.n_filtered:>10,} "
              f"{r.win_rate:>8.2f} {r.total_pnl:>12.2f} {r.total_pnl_after_fees:>12.2f} "
              f"{r.profit_factor:>8.2f} {r.sharpe_ratio:>10.2f}")

    print(f"{'='*120}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test Meta-Model Backtest')
    parser.add_argument('--indicator', type=str, default='macd', choices=['macd', 'rsi', 'cci'],
                        help='Indicateur primaire')
    parser.add_argument('--filter', type=str, default='kalman', choices=['kalman', 'octave'],
                        help='Type de filtre')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Split à tester')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Seuil meta-prob (None = compare plusieurs)')
    parser.add_argument('--compare-thresholds', action='store_true',
                        help='Comparer 0.5, 0.6, 0.7')
    parser.add_argument('--fees', type=float, default=0.001,
                        help='Frais par side (0.001 = 0.1%)')
    args = parser.parse_args()

    print(f"{'='*80}")
    print(f"META-MODEL BACKTEST - Phase 2.18")
    print(f"{'='*80}")
    print(f"Indicateur: {args.indicator.upper()}")
    print(f"Filtre: {args.filter}")
    print(f"Split: {args.split}")
    print(f"Frais: {args.fees*100:.2f}% par side")
    print(f"{'='*80}\n")

    # Charger données
    logger.info("Chargement des données...")
    primary_data = load_primary_predictions(args.indicator, args.split)
    meta_data = load_meta_features(args.indicator, args.filter, args.split)
    meta_model = load_meta_model(args.filter)

    # Vérifier alignement
    assert len(primary_data['Y_pred']) == len(meta_data['features']), \
        "Désalignement primary vs meta features!"

    # Comparer stratégies
    if args.compare_thresholds or args.threshold is None:
        thresholds = [0.5, 0.6, 0.7]
        compare_strategies(
            primary_data=primary_data,
            meta_data=meta_data,
            meta_model=meta_model,
            thresholds=thresholds,
            fees=args.fees
        )
    else:
        # Test un seul threshold
        opens = primary_data['OHLCV'][:, 2]
        timestamps = primary_data['OHLCV'][:, 0]
        asset_ids = primary_data['OHLCV'][:, 1]

        result = backtest_with_meta_filter(
            primary_preds=primary_data['Y_pred'],
            meta_features=meta_data['features'],
            meta_model=meta_model,
            opens=opens,
            timestamps=timestamps,
            asset_ids=asset_ids,
            threshold=args.threshold,
            fees=args.fees
        )
        print_results(result)

    logger.info("✅ Backtest terminé!")


if __name__ == '__main__':
    main()
