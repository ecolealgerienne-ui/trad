#!/usr/bin/env python3
"""
Création des meta-labels pour Model B avec filtrage régime (Model A).

Pipeline:
    1. Charger dataset enrichi avec prédictions régime (Y[:, 5-9])
    2. Charger prédictions MACD direction (modèle primaire)
    3. Simuler backtest avec filtrage régime:
       - Entrée: MACD prédit UP → LONG, DOWN → SHORT
       - Filtre: Seulement si régime autorisé (ex: RANGE=0/1 ou TREND=2/3)
       - Sortie: Changement direction MACD
    4. Pour chaque trade: label_meta = 1 si PnL > 0 else 0
    5. Sauvegarder meta-labels alignés pour training Model B

Usage:
    python src/create_meta_labels_regime.py \
        --regime-filter range \
        --split train \
        --fees 0.001
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from enum import Enum
import json


class Position(Enum):
    """Position types."""
    FLAT = 0
    LONG = 1
    SHORT = 2


def load_regime_dataset(split: str) -> Dict:
    """
    Charge le dataset régime enrichi.

    Args:
        split: 'train', 'val', ou 'test'

    Returns:
        Dict avec X, Y (enrichi), OHLCV, metadata
    """
    dataset_path = Path('data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz')

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Loading regime dataset: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)

    # Clés selon le split
    X_key = f'X_{split}'
    Y_key = f'Y_{split}'
    OHLCV_key = f'OHLCV_{split}'

    result = {
        'X': data[X_key],               # (n, 12, ~22)
        'Y': data[Y_key],               # (n, 10) - enrichi avec prédictions
        'OHLCV': data[OHLCV_key],       # (n, 7)
        'metadata': data['metadata'].item() if 'metadata' in data else {}
    }

    # Extraire les colonnes importantes de Y
    result['timestamps'] = result['Y'][:, 0]
    result['asset_ids'] = result['Y'][:, 1].astype(int)
    result['regime_labels'] = result['Y'][:, 2].astype(int)  # Ground truth
    result['regime_preds'] = result['Y'][:, 5].astype(int)   # Model A predictions
    result['regime_probs'] = result['Y'][:, 6:10]            # (n, 4) probabilities

    print(f"  Split: {split}")
    print(f"  Samples: {len(result['Y']):,}")
    print(f"  X shape: {result['X'].shape}")
    print(f"  Y shape: {result['Y'].shape}")
    print(f"  OHLCV shape: {result['OHLCV'].shape}")

    return result


def load_macd_predictions(split: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge les prédictions MACD direction depuis meta_labels_aligned.

    Args:
        split: 'train', 'val', ou 'test'

    Returns:
        predictions: (n,) - prédictions MACD
        probabilities: (n,) - probabilités MACD
    """
    # Essayer de charger depuis meta_labels_aligned
    meta_path = Path(f'data/prepared/meta_labels_macd_kalman_{split}_aligned.npz')

    if not meta_path.exists():
        raise FileNotFoundError(
            f"MACD predictions not found: {meta_path}\n"
            f"Run create_meta_labels_aligned.py first to generate MACD predictions."
        )

    print(f"\nLoading MACD predictions: {meta_path}")
    data = np.load(meta_path, allow_pickle=True)

    predictions = data['predictions_macd']  # (n,) - probabilités [0,1]

    print(f"  MACD predictions: {predictions.shape}")
    print(f"  Mean prob: {predictions.mean():.4f}")

    return predictions, predictions  # Retourner 2x pour compatibilité


def backtest_single_asset(
    asset_id: int,
    asset_mask: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    ohlcv: np.ndarray,
    regime_preds: np.ndarray,
    allowed_regimes: set,
    fees: float
) -> Tuple[List[dict], int, int]:
    """
    Backtest pour un seul asset avec filtrage régime (Model A).

    Args:
        asset_id: ID de l'asset
        asset_mask: Masque booléen pour cet asset
        predictions: (n,) Prédictions MACD direction
        labels: (n, 3) Labels
        ohlcv: (n, 7) OHLCV
        regime_preds: (n,) Prédictions régime de Model A
        allowed_regimes: Set de régimes autorisés (ex: {0, 1} pour RANGE)
        fees: Frais par side

    Returns:
        (trades, trades_executed, trades_blocked)
    """
    # Filtrer samples pour cet asset
    asset_indices = np.where(asset_mask)[0]
    asset_preds = predictions[asset_mask]
    asset_regime_preds = regime_preds[asset_mask]
    asset_opens = ohlcv[asset_mask, 2]  # Colonne 2 = Open
    asset_timestamps = labels[asset_mask, 0]  # Colonne 0 = timestamp

    n_asset = len(asset_preds)
    trades = []
    trades_executed = 0
    trades_blocked = 0

    # Variables de tracking
    position = Position.FLAT
    entry_idx = 0
    entry_price = 0.0

    for i in range(n_asset - 1):
        direction = int(asset_preds[i])
        target = Position.LONG if direction == 1 else Position.SHORT

        # CAS 1: FLAT - entrer (avec filtrage régime)
        if position == Position.FLAT:
            # ✨ FILTRAGE RÉGIME: Vérifier si régime autorisé
            current_regime = int(asset_regime_preds[i])
            if current_regime not in allowed_regimes:
                trades_blocked += 1
                continue  # Bloquer l'entrée

            # Régime autorisé → Entrer
            trades_executed += 1
            position = target
            entry_idx = asset_indices[i]  # Global index
            entry_price = asset_opens[i + 1]
            continue

        # CAS 2: EN POSITION - vérifier si sortir
        if position != target:
            # Sortie (signal a changé)
            exit_idx = asset_indices[i]  # Global index
            exit_price = asset_opens[i + 1]
            duration = i - (entry_idx - asset_indices[0])  # Local duration

            # Calculate PnL
            if position == Position.LONG:
                pnl = (exit_price - entry_price) / entry_price
            else:  # SHORT
                pnl = (entry_price - exit_price) / entry_price

            # Frais
            trade_fees = 2 * fees
            pnl_after_fees = pnl - trade_fees

            trade = {
                'entry_idx': entry_idx,
                'exit_idx': exit_idx,
                'duration': duration,
                'position': 'LONG' if position == Position.LONG else 'SHORT',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_after_fees': pnl_after_fees,
                'asset_id': int(asset_id),
                'entry_timestamp': int(asset_timestamps[i]),
            }
            trades.append(trade)

            # Nouvelle position (reversal)
            position = target
            entry_idx = asset_indices[i]
            entry_price = asset_opens[i + 1]

    # Close final position (if any)
    if position != Position.FLAT:
        exit_idx = asset_indices[n_asset - 1]
        exit_price = asset_opens[-1]
        duration = (n_asset - 1) - (entry_idx - asset_indices[0])

        if position == Position.LONG:
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price

        trade_fees = 2 * fees
        pnl_after_fees = pnl - trade_fees

        trade = {
            'entry_idx': entry_idx,
            'exit_idx': exit_idx,
            'duration': duration,
            'position': 'LONG' if position == Position.LONG else 'SHORT',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_after_fees': pnl_after_fees,
            'asset_id': int(asset_id),
            'entry_timestamp': int(asset_timestamps[-1]),
        }
        trades.append(trade)

    return trades, trades_executed, trades_blocked


def simulate_oracle_backtest(
    predictions: np.ndarray,
    labels: np.ndarray,
    ohlcv: np.ndarray,
    regime_preds: np.ndarray,
    regime_filter: str = 'all',
    fees: float = 0.001
) -> Tuple[List[dict], int, int, int]:
    """
    Simule backtest avec filtrage régime (Model A).

    Args:
        predictions: (n,) Prédictions MACD direction
        labels: (n, 3) Labels
        ohlcv: (n, 7) OHLCV
        regime_preds: (n,) Prédictions régime de Model A
        regime_filter: 'range', 'trend', 'range_low', 'trend_low', 'all'
        fees: Frais par side

    Returns:
        (all_trades, n_samples, total_executed, total_blocked)
    """
    print(f"\n=== Simulating Regime-Filtered Backtest (Filter: {regime_filter}) ===")

    # Définir les régimes autorisés selon le filtre
    if regime_filter == 'range':
        allowed_regimes = {0, 1}  # RANGE LOW VOL + HIGH VOL
    elif regime_filter == 'trend':
        allowed_regimes = {2, 3}  # TREND LOW VOL + HIGH VOL
    elif regime_filter == 'range_low':
        allowed_regimes = {0}  # RANGE LOW VOL uniquement
    elif regime_filter == 'trend_low':
        allowed_regimes = {2}  # TREND LOW VOL uniquement
    elif regime_filter == 'all':
        allowed_regimes = {0, 1, 2, 3}  # Tous (pas de filtrage)
    else:
        raise ValueError(f"Unknown regime_filter: {regime_filter}")

    print(f"Allowed regimes: {sorted(allowed_regimes)}")

    all_trades = []
    n_samples = len(predictions)
    total_executed = 0
    total_blocked = 0

    # Extraire asset_ids uniques
    asset_ids = np.unique(labels[:, 1])  # Colonne 1 = asset_id
    print(f"Assets found: {asset_ids}")

    for asset_id in asset_ids:
        # Filtrer samples pour cet asset
        asset_mask = (labels[:, 1] == asset_id)

        # Backtest pour cet asset avec filtrage régime
        asset_trades, executed, blocked = backtest_single_asset(
            asset_id=asset_id,
            asset_mask=asset_mask,
            predictions=predictions,
            labels=labels,
            ohlcv=ohlcv,
            regime_preds=regime_preds,
            allowed_regimes=allowed_regimes,
            fees=fees
        )

        all_trades.extend(asset_trades)
        total_executed += executed
        total_blocked += blocked
        print(f"  Asset {int(asset_id)}: {len(asset_trades)} trades (executed: {executed}, blocked: {blocked})")

    print(f"\nTotal trades: {len(all_trades)}")
    print(f"Total executed: {total_executed}, blocked: {total_blocked}")
    if total_executed + total_blocked > 0:
        block_rate = 100 * total_blocked / (total_executed + total_blocked)
        print(f"Block rate: {block_rate:.2f}%")

    return all_trades, n_samples, total_executed, total_blocked


def create_meta_labels_aligned(
    trades: List[dict],
    pnl_threshold: float = 0.0
) -> np.ndarray:
    """
    Crée les meta-labels ALIGNÉS (pas de contrainte durée).

    DIFFÉRENCE vs Triple Barrier:
    - Triple Barrier: label = 1 si pnl > threshold ET duration >= min_duration
    - Aligned: label = 1 si pnl > threshold (PAS de contrainte durée)

    Args:
        trades: Liste de trades
        pnl_threshold: Seuil PnL (default: 0.0)

    Returns:
        meta_labels: (n_trades,) array de labels 0/1
    """
    meta_labels = []

    for trade in trades:
        pnl_net = trade['pnl_after_fees']

        # RÈGLE ALIGNED: Profitable (PnL NET > threshold), PAS de contrainte durée
        if pnl_net > pnl_threshold:
            label = 1  # Accepter le trade
        else:
            label = 0  # Rejeter le trade

        meta_labels.append(label)

    meta_labels = np.array(meta_labels, dtype=np.int32)

    # Stats
    n_positive = np.sum(meta_labels == 1)
    n_negative = np.sum(meta_labels == 0)
    total = len(meta_labels)

    print(f"\n=== Meta-Labels Statistics (ALIGNED) ===")
    print(f"Total trades: {total}")
    print(f"Positive (1): {n_positive} ({100*n_positive/total:.1f}%)")
    print(f"Negative (0): {n_negative} ({100*n_negative/total:.1f}%)")
    print(f"\nNote: Pas de contrainte de durée minimale (ALIGNED)")

    return meta_labels


def map_trade_labels_to_timesteps(trades: List[dict], meta_labels: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Mappe les meta-labels des trades aux timesteps individuels.

    Args:
        trades: Liste de trades
        meta_labels: (n_trades,) Labels par trade
        n_samples: Nombre total de samples

    Returns:
        timestep_labels: (n_samples,) Labels par timestep (-1 si hors trade)
    """
    print("\n=== Mapping Trade Labels to Timesteps ===")

    # Initialiser à -1 (hors trade)
    timestep_labels = np.full(n_samples, -1, dtype=np.int32)

    # Pour chaque trade, assigner le label à TOUS les timesteps du trade
    for i, trade in enumerate(trades):
        entry_idx = trade['entry_idx']
        exit_idx = trade['exit_idx']
        label = meta_labels[i]

        # Label ALL timesteps from entry to exit (inclusive)
        timestep_labels[entry_idx:exit_idx+1] = label

    # Distribution
    n_positive = np.sum(timestep_labels == 1)
    n_negative = np.sum(timestep_labels == 0)
    n_ignored = np.sum(timestep_labels == -1)

    print(f"Timestep labels distribution:")
    print(f"  Positive (1): {n_positive} ({100*n_positive/n_samples:.1f}%)")
    print(f"  Negative (0): {n_negative} ({100*n_negative/n_samples:.1f}%)")
    print(f"  Ignored (-1): {n_ignored} ({100*n_ignored/n_samples:.1f}%)")

    return timestep_labels


def save_meta_dataset(
    output_path: Path,
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    OHLCV: np.ndarray,
    predictions_macd: np.ndarray,
    predictions_rsi: np.ndarray,
    predictions_cci: np.ndarray,
    meta_labels: np.ndarray,
    split: str,
    metadata: dict
):
    """
    Sauvegarde dataset avec meta-labels.

    Args:
        output_path: Chemin de sortie .npz
        X: Sequences
        Y: Labels
        T: Timestamps
        OHLCV: OHLCV data
        predictions_macd: Prédictions MACD
        predictions_rsi: Prédictions RSI
        predictions_cci: Prédictions CCI
        meta_labels: Meta-labels
        split: Split name
        metadata: Metadata dict
    """
    print(f"\n=== Saving Meta-Dataset ===")
    print(f"Output: {output_path}")

    # Construire le dict de sauvegarde
    save_dict = {
        # Prédictions des 3 indicateurs (clés SANS suffix)
        'predictions_macd': predictions_macd,
        'predictions_rsi': predictions_rsi,
        'predictions_cci': predictions_cci,

        # Meta-labels (clé SANS suffix)
        'meta_labels': meta_labels,

        # OHLCV (clé SANS suffix pour train_meta_model)
        'OHLCV': OHLCV,

        # Données originales (AVEC suffix pour archive)
        f'X_{split}': X,
        f'Y_{split}': Y,
        f'T_{split}': T,

        # Metadata enrichie
        'metadata': json.dumps({
            **metadata,
            'meta_labeling': {
                'method': 'aligned',
                'description': 'Meta-labels aligned with backtest strategy (signal reversal)',
                'split': split,
                'n_samples': len(meta_labels),
                'n_positive': int(np.sum(meta_labels == 1)),
                'n_negative': int(np.sum(meta_labels == 0)),
                'n_ignored': int(np.sum(meta_labels == -1)),
            }
        })
    }

    # Sauvegarder
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **save_dict)

    print(f"  Saved successfully")
    print(f"  Size: {output_path.stat().st_size / (1024**2):.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='Create regime-filtered meta-labels for Model B')
    parser.add_argument('--regime-filter', type=str, default='range',
                        choices=['range', 'trend', 'range_low', 'trend_low', 'all'],
                        help='Regime filter (default: range)')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'],
                        help='Dataset split to process')
    parser.add_argument('--fees', type=float, default=0.001,
                        help='Fees per side (default: 0.001 = 0.1%)')
    parser.add_argument('--pnl-threshold', type=float, default=0.0,
                        help='PnL threshold for positive label (default: 0.0)')
    parser.add_argument('--output-dir', type=str, default='data/prepared',
                        help='Output directory')

    args = parser.parse_args()

    print("=" * 80)
    print("CREATE REGIME-FILTERED META-LABELS FOR MODEL B")
    print("=" * 80)
    print(f"Regime Filter: {args.regime_filter}")
    print(f"Split: {args.split}")
    print(f"Fees: {args.fees} ({args.fees * 100:.2f}%)")
    print(f"PnL Threshold: {args.pnl_threshold}")

    # 1. Charger dataset régime enrichi
    data = load_regime_dataset(args.split)

    # 2. Charger prédictions MACD direction
    predictions_macd, _ = load_macd_predictions(args.split)

    # 3. Extraire prédictions régime (Model A) depuis Y[:, 5]
    regime_preds = data['regime_preds']
    print(f"\nRegime predictions distribution:")
    for regime_id in range(4):
        count = np.sum(regime_preds == regime_id)
        pct = 100 * count / len(regime_preds)
        regime_name = ['R0 (RANGE LOW)', 'R1 (RANGE HIGH)', 'R2 (TREND LOW)', 'R3 (TREND HIGH)'][regime_id]
        print(f"  {regime_name}: {count} ({pct:.1f}%)")

    # 4. Simuler backtest avec filtrage régime
    trades, n_samples, total_executed, total_blocked = simulate_oracle_backtest(
        predictions=predictions_macd,
        labels=data['Y'],
        ohlcv=data['OHLCV'],
        regime_preds=regime_preds,
        regime_filter=args.regime_filter,
        fees=args.fees
    )

    # 5. Créer meta-labels (profitable = 1, unprofitable = 0)
    meta_labels_trades = create_meta_labels_aligned(
        trades=trades,
        pnl_threshold=args.pnl_threshold
    )

    # 6. Mapper aux timesteps
    meta_labels = map_trade_labels_to_timesteps(trades, meta_labels_trades, n_samples)

    # 7. Sauvegarder meta-labels régime
    output_path = Path(args.output_dir) / f'meta_labels_regime_{args.regime_filter}_{args.split}.npz'

    # Note: On sauvegarde UNIQUEMENT MACD car c'est le signal primaire utilisé
    save_meta_dataset(
        output_path=output_path,
        X=data['X'],
        Y=data['Y'],
        T=data['timestamps'],
        OHLCV=data['OHLCV'],
        predictions_macd=predictions_macd,
        predictions_rsi=predictions_macd,  # Placeholder (pas utilisé)
        predictions_cci=predictions_macd,  # Placeholder (pas utilisé)
        meta_labels=meta_labels,
        split=args.split,
        metadata=data['metadata']
    )

    print("\n" + "=" * 80)
    print("REGIME-FILTERED META-LABELS CREATED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Create meta-labels for other splits:")
    print(f"   python src/create_meta_labels_regime.py --regime-filter {args.regime_filter} --split train --fees {args.fees}")
    print(f"   python src/create_meta_labels_regime.py --regime-filter {args.regime_filter} --split val --fees {args.fees}")
    print(f"2. Train Model B (Logistic Binary):")
    print(f"   python src/train_meta_model_regime.py --regime-filter {args.regime_filter}")
    print(f"3. Backtest regime strategy:")
    print(f"   python tests/test_regime_strategy.py --regime-filter {args.regime_filter} --split test")


if __name__ == '__main__':
    main()
