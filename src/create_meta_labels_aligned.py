#!/usr/bin/env python3
"""
Script de création des meta-labels ALIGNÉS avec la stratégie backtest réelle.

Basé sur create_meta_labels_phase215.py avec labeling modifié.

Différence vs Triple Barrier (create_meta_labels_phase215.py):
- Triple Barrier: label = 1 si pnl > 0 ET duration >= min_duration
- Aligned: label = 1 si pnl > 0 (PAS de contrainte durée)

Objectif:
- Charger datasets direction-only existants (.npz)
- Charger prédictions existantes (déjà dans .npz)
- Simuler backtest EXACTEMENT comme test_meta_model_backtest.py
- Pour chaque trade: label_meta = 1 si pnl_after_fees > 0 else 0
- Mapper labels aux timesteps individuels
- Sauvegarder avec MÊME structure + meta_labels

CRITIQUE: Les labels correspondent EXACTEMENT au calcul PnL du backtest réel.
Pas de contrainte de durée minimale.

Usage:
    python src/create_meta_labels_aligned.py --indicator macd --filter kalman --split train --fees 0.001
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


def load_dataset(indicator: str, filter_type: str, split: str) -> Dict:
    """
    Charge dataset direction-only existant.

    Args:
        indicator: 'macd', 'rsi', ou 'cci'
        filter_type: 'kalman' ou 'octave'
        split: 'train', 'val', ou 'test'

    Returns:
        Dict avec sequences, labels, timestamps, ohlcv, metadata
    """
    dataset_path = Path(f'data/prepared/dataset_btc_eth_bnb_ada_ltc_{indicator}_direction_only_{filter_type}.npz')

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Loading dataset: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)

    # Clés selon le split
    X_key = f'X_{split}'
    Y_key = f'Y_{split}'
    T_key = f'T_{split}'
    OHLCV_key = f'OHLCV_{split}'
    pred_key = f'Y_{split}_pred'

    # Vérifier que les prédictions existent
    if pred_key not in data:
        raise KeyError(f"Predictions not found in dataset. Expected key: {pred_key}")

    # Extraire les données du split demandé
    result = {
        'sequences': data[X_key],           # (n, 25, features)
        'labels': data[Y_key],              # (n, 3) - [timestamp, asset_id, direction]
        'timestamps': data[T_key],          # (n, 3) - [timestamp, asset_id, is_transition]
        'ohlcv': data[OHLCV_key],          # (n, 7) - [timestamp, asset_id, O, H, L, C, V]
        'predictions': data[pred_key],      # (n,) - Prédictions existantes!
        'metadata': json.loads(data['metadata'].item()) if 'metadata' in data else {}
    }

    print(f"  Split: {split}")
    print(f"  Sequences: {result['sequences'].shape}")
    print(f"  Labels: {result['labels'].shape}")
    print(f"  Timestamps: {result['timestamps'].shape}")
    print(f"  OHLCV: {result['ohlcv'].shape}")
    print(f"  Predictions: {result['predictions'].shape}")

    return result


def load_predictions(indicator: str, filter_type: str, split: str) -> np.ndarray:
    """
    Charge les prédictions existantes depuis le dataset.

    Args:
        indicator: 'macd', 'rsi', ou 'cci'
        filter_type: 'kalman' ou 'octave'
        split: 'train', 'val', ou 'test'

    Returns:
        predictions: (n,) array de prédictions 0/1
    """
    dataset_path = Path(f'data/prepared/dataset_btc_eth_bnb_ada_ltc_{indicator}_direction_only_{filter_type}.npz')

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"\nLoading predictions from: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)

    pred_key = f'Y_{split}_pred'
    if pred_key not in data:
        raise KeyError(f"Predictions not found. Key '{pred_key}' not in dataset.")

    predictions = data[pred_key]
    print(f"  Loaded {len(predictions)} predictions")
    print(f"  Distribution: UP={np.sum(predictions == 1)}, DOWN={np.sum(predictions == 0)}")

    return predictions


def backtest_single_asset(
    asset_id: int,
    asset_mask: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    ohlcv: np.ndarray,
    fees: float
) -> List[dict]:
    """
    Backtest pour un seul asset avec logique ALIGNÉE.

    Args:
        asset_id: ID de l'asset
        asset_mask: Masque booléen pour cet asset
        predictions: (n,) Prédictions
        labels: (n, 3) Labels
        ohlcv: (n, 7) OHLCV
        fees: Frais par side

    Returns:
        Liste de trades
    """
    # Filtrer samples pour cet asset
    asset_indices = np.where(asset_mask)[0]
    asset_preds = predictions[asset_mask]
    asset_opens = ohlcv[asset_mask, 2]  # Colonne 2 = Open
    asset_timestamps = labels[asset_mask, 0]  # Colonne 0 = timestamp

    n_asset = len(asset_preds)
    trades = []

    # Variables de tracking
    position = Position.FLAT
    entry_idx = 0
    entry_price = 0.0

    for i in range(n_asset - 1):
        direction = int(asset_preds[i])
        target = Position.LONG if direction == 1 else Position.SHORT

        # CAS 1: FLAT - entrer
        if position == Position.FLAT:
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

    return trades


def simulate_oracle_backtest(
    predictions: np.ndarray,
    labels: np.ndarray,
    ohlcv: np.ndarray,
    fees: float = 0.001
) -> Tuple[List[dict], int]:
    """
    Simule backtest Oracle pour générer trades.

    Args:
        predictions: (n,) Prédictions
        labels: (n, 3) Labels
        ohlcv: (n, 7) OHLCV
        fees: Frais par side

    Returns:
        (all_trades, n_samples)
    """
    print("\n=== Simulating Oracle Backtest (Aligned Strategy) ===")

    all_trades = []
    n_samples = len(predictions)

    # Extraire asset_ids uniques
    asset_ids = np.unique(labels[:, 1])  # Colonne 1 = asset_id
    print(f"Assets found: {asset_ids}")

    for asset_id in asset_ids:
        # Filtrer samples pour cet asset
        asset_mask = (labels[:, 1] == asset_id)

        # Backtest pour cet asset
        asset_trades = backtest_single_asset(
            asset_id=asset_id,
            asset_mask=asset_mask,
            predictions=predictions,
            labels=labels,
            ohlcv=ohlcv,
            fees=fees
        )

        all_trades.extend(asset_trades)
        print(f"  Asset {int(asset_id)}: {len(asset_trades)} trades")

    print(f"\nTotal trades: {len(all_trades)}")

    return all_trades, n_samples


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
    predictions: np.ndarray,
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
        predictions: Prédictions modèle primaire
        meta_labels: Meta-labels
        split: Split name
        metadata: Metadata dict
    """
    print(f"\n=== Saving Meta-Dataset ===")
    print(f"Output: {output_path}")

    # Construire le dict de sauvegarde
    save_dict = {
        # Données originales (préservées)
        f'X_{split}': X,
        f'Y_{split}': Y,
        f'T_{split}': T,
        f'OHLCV_{split}': OHLCV,

        # Prédictions (préservées)
        f'predictions_{split}': predictions,

        # NOUVEAU: meta-labels
        f'meta_labels_{split}': meta_labels,

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
    parser = argparse.ArgumentParser(description='Create aligned meta-labels')
    parser.add_argument('--indicator', type=str, required=True, choices=['macd', 'rsi', 'cci'],
                        help='Indicator to use')
    parser.add_argument('--filter', type=str, default='kalman', choices=['kalman', 'octave'],
                        help='Filter type')
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
    print("CREATE ALIGNED META-LABELS (Phase 2.18)")
    print("=" * 80)
    print(f"Indicator: {args.indicator}")
    print(f"Filter: {args.filter}")
    print(f"Split: {args.split}")
    print(f"Fees: {args.fees} ({args.fees * 100:.2f}%)")
    print(f"PnL Threshold: {args.pnl_threshold}")

    # 1. Charger dataset
    data = load_dataset(args.indicator, args.filter, args.split)

    # 2. Charger prédictions (depuis .npz, PAS de chargement modèle)
    predictions = load_predictions(args.indicator, args.filter, args.split)

    # 3. Simuler backtest Oracle (stratégie aligned)
    trades, n_samples = simulate_oracle_backtest(
        predictions=predictions,
        labels=data['labels'],
        ohlcv=data['ohlcv'],
        fees=args.fees
    )

    # 4. Créer meta-labels ALIGNED (pas de contrainte durée)
    meta_labels_trades = create_meta_labels_aligned(
        trades=trades,
        pnl_threshold=args.pnl_threshold
    )

    # 5. Mapper aux timesteps
    meta_labels = map_trade_labels_to_timesteps(trades, meta_labels_trades, n_samples)

    # 6. Sauvegarder
    output_path = Path(args.output_dir) / f'meta_labels_{args.indicator}_{args.filter}_{args.split}_aligned.npz'
    save_meta_dataset(
        output_path=output_path,
        X=data['sequences'],
        Y=data['labels'],
        T=data['timestamps'],
        OHLCV=data['ohlcv'],
        predictions=predictions,
        meta_labels=meta_labels,
        split=args.split,
        metadata=data['metadata']
    )

    print("\n" + "=" * 80)
    print("ALIGNED META-LABELS CREATED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Create meta-labels for other splits:")
    print(f"   python src/create_meta_labels_aligned.py --indicator {args.indicator} --filter {args.filter} --split train --fees {args.fees}")
    print(f"   python src/create_meta_labels_aligned.py --indicator {args.indicator} --filter {args.filter} --split val --fees {args.fees}")
    print(f"2. Train aligned meta-model:")
    print(f"   python src/train_meta_model_phase217.py --filter {args.filter} --aligned")
    print(f"3. Re-backtest:")
    print(f"   python tests/test_meta_model_backtest.py --indicator {args.indicator} --split test --aligned")


if __name__ == '__main__':
    main()
