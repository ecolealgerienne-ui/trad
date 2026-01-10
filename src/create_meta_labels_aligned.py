#!/usr/bin/env python3
"""
Script de création des meta-labels ALIGNÉS avec la stratégie backtest réelle.

Différence vs Triple Barrier (create_meta_labels_phase215.py):
- Triple Barrier: utilise barrières prix + contraintes durée (min_duration=5)
- Aligned: utilise changement de signal (direction flip) + PnL exact du backtest

Objectif:
- Charger datasets direction-only existants (.npz)
- Charger modèle primaire et générer prédictions (ou charger preds existants)
- Simuler backtest EXACTEMENT comme test_meta_model_backtest.py (threshold=0.0)
- Pour chaque trade: label_meta = 1 si pnl_after_fees > 0 else 0
- Mapper labels aux timesteps individuels
- Sauvegarder avec MÊME structure + meta_labels

CRITIQUE: Les labels correspondent EXACTEMENT au calcul PnL du backtest réel.

Usage:
    python src/create_meta_labels_aligned.py --indicator macd --filter kalman --split train --fees 0.001
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, List, Dict
from enum import Enum
import json

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from model import CNNLSTMModel
from constants import SEQUENCE_LENGTH


class Position(Enum):
    """Position types."""
    FLAT = 0
    LONG = 1
    SHORT = 2


def load_dataset(indicator: str, filter_type: str, split: str) -> dict:
    """
    Charge dataset direction-only existant avec timestamps.

    Args:
        indicator: 'macd', 'rsi', ou 'cci'
        filter_type: 'kalman' ou 'octave'
        split: 'train', 'val', ou 'test'

    Returns:
        dict avec sequences, labels, timestamps, ohlcv, metadata
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

    # Extraire les données du split demandé
    result = {
        'sequences': data[X_key],           # (n, 25, features)
        'labels': data[Y_key],              # (n, 3) - [timestamp, asset_id, direction]
        'timestamps': data[T_key],          # (n, 3) - [timestamp, asset_id, is_transition]
        'ohlcv': data[OHLCV_key],          # (n, 7) - [timestamp, asset_id, O, H, L, C, V]
        'metadata': json.loads(data['metadata'].item()) if 'metadata' in data else {}
    }

    print(f"  Split: {split}")
    print(f"  Sequences: {result['sequences'].shape}")
    print(f"  Labels: {result['labels'].shape}")
    print(f"  Timestamps: {result['timestamps'].shape}")
    print(f"  OHLCV: {result['ohlcv'].shape}")

    return result


def load_model_and_predict(indicator: str, filter_type: str, X: np.ndarray, device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge le modèle primaire entraîné et génère les prédictions.

    Args:
        indicator: 'macd', 'rsi', ou 'cci'
        filter_type: 'kalman' ou 'octave'
        X: Sequences (n, seq_len, features)
        device: 'cuda' ou 'cpu'

    Returns:
        (predictions, probabilities)
        - predictions: (n,) int 0/1
        - probabilities: (n,) float [0,1]
    """
    model_path = Path(f'models/best_model_{indicator}_{filter_type}_dual_binary.pth')

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"\nLoading model: {model_path}")

    # Déterminer features selon indicateur
    if indicator in ['macd', 'rsi']:
        n_features = 1
    elif indicator == 'cci':
        n_features = 3
    else:
        raise ValueError(f"Unknown indicator: {indicator}")

    # Charger modèle
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    model = CNNLSTMModel(
        n_features=n_features,
        sequence_length=SEQUENCE_LENGTH,
        n_outputs=1  # Direction-only
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    print("Generating predictions...")

    # Générer prédictions
    predictions = []
    probabilities = []

    batch_size = 2048
    n_samples = len(X)

    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_X = torch.tensor(X[start_idx:end_idx], dtype=torch.float32, device=device)

            # Forward pass
            outputs = model(batch_X)  # (batch, 1) - déjà sigmoid

            # Direction output (first column)
            dir_probs = outputs[:, 0].cpu().numpy()  # (batch,)
            dir_preds = (dir_probs > 0.5).astype(int)

            predictions.extend(dir_preds)
            probabilities.extend(dir_probs)

            if (start_idx // batch_size) % 10 == 0:
                print(f"  Progress: {start_idx}/{n_samples}")

    predictions = np.array(predictions, dtype=int)
    probabilities = np.array(probabilities, dtype=float)

    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Distribution: UP={np.sum(predictions == 1)}, DOWN={np.sum(predictions == 0)}")

    return predictions, probabilities


def backtest_aligned(
    predictions: np.ndarray,
    labels: np.ndarray,
    ohlcv: np.ndarray,
    fees: float = 0.001
) -> Tuple[List[dict], int]:
    """
    Backtest ALIGNÉ avec test_meta_model_backtest.py (threshold=0.0).

    LOGIQUE EXACTE:
    - Par asset
    - Signal à index i → Exécution à Open[i+1]
    - Position: FLAT, LONG, SHORT
    - Sortie: Quand direction change (pas de barrières)
    - PnL: Calcul exact comme dans backtest
    - Pas de filtrage meta-model (threshold=0.0)

    Args:
        predictions: (n,) Prédictions modèle primaire (0/1)
        labels: (n, 3) - [timestamp, asset_id, direction]
        ohlcv: (n, 7) - [timestamp, asset_id, O, H, L, C, V]
        fees: Frais par side

    Returns:
        (all_trades, n_samples_processed)
        - all_trades: Liste de dicts avec entry_idx, exit_idx, pnl_after_fees, ...
        - n_samples_processed: Nombre total de samples traités
    """
    print("\n=== Backtest Aligned (Exact strategy replication) ===")

    all_trades = []
    n_samples = len(predictions)

    # Extraire asset_ids uniques
    asset_ids = np.unique(labels[:, 1])  # Colonne 1 = asset_id
    print(f"Assets found: {asset_ids}")

    for asset_id in asset_ids:
        # Filtrer samples pour cet asset
        asset_mask = (labels[:, 1] == asset_id)
        asset_indices = np.where(asset_mask)[0]

        asset_preds = predictions[asset_mask]
        asset_opens = ohlcv[asset_mask, 2]  # Colonne 2 = Open
        asset_timestamps = labels[asset_mask, 0]  # Colonne 0 = timestamp

        n_asset = len(asset_preds)
        print(f"\nAsset {int(asset_id)}: {n_asset} samples")

        # Variables de tracking
        position = Position.FLAT
        entry_idx = 0
        entry_price = 0.0

        for i in range(n_asset - 1):
            direction = int(asset_preds[i])
            target = Position.LONG if direction == 1 else Position.SHORT

            # CAS 1: FLAT - décider si entrer
            if position == Position.FLAT:
                # Pas de filtrage meta-model (threshold=0.0)
                # Entrer directement
                position = target
                entry_idx = asset_indices[i]  # Global index
                entry_price = asset_opens[i + 1]
                continue

            # CAS 2: EN POSITION - vérifier si sortir
            if position != target:
                # TOUJOURS sortir (signal a changé)
                exit_idx = asset_indices[i]  # Global index
                exit_price = asset_opens[i + 1]
                duration = i - (entry_idx - asset_indices[0])  # Local duration

                # Calculate PnL EXACTEMENT comme dans backtest
                if position == Position.LONG:
                    pnl = (exit_price - entry_price) / entry_price
                else:  # SHORT
                    pnl = (entry_price - exit_price) / entry_price

                # Frais: entrée + sortie
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
                    'entry_timestamp': asset_timestamps[i],
                    # Label meta ALIGNÉ: 1 si profitable, 0 sinon
                    'label_meta': 1 if pnl_after_fees > 0 else 0
                }
                all_trades.append(trade)

                # Nouvelle position (reversal immédiat)
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
                'entry_timestamp': asset_timestamps[-1],
                'label_meta': 1 if pnl_after_fees > 0 else 0
            }
            all_trades.append(trade)

    print(f"\n=== Backtest Results ===")
    print(f"Total trades: {len(all_trades)}")

    # Distribution labels
    n_profitable = sum(1 for t in all_trades if t['label_meta'] == 1)
    n_losing = len(all_trades) - n_profitable
    print(f"Label distribution:")
    print(f"  Profitable (1): {n_profitable} ({100*n_profitable/len(all_trades):.1f}%)")
    print(f"  Losing (0): {n_losing} ({100*n_losing/len(all_trades):.1f}%)")

    # Métriques de validation
    total_pnl = sum(t['pnl'] for t in all_trades)
    total_pnl_net = sum(t['pnl_after_fees'] for t in all_trades)
    win_rate = 100 * n_profitable / len(all_trades)

    print(f"\nValidation metrics:")
    print(f"  Win Rate: {win_rate:.2f}%")
    print(f"  PnL Brut: {total_pnl * 100:+.2f}%")
    print(f"  PnL Net: {total_pnl_net * 100:+.2f}%")

    return all_trades, n_samples


def map_trades_to_timesteps(trades: List[dict], n_samples: int) -> np.ndarray:
    """
    Mappe les labels de trades aux timesteps individuels.

    Pour chaque timestep, le label_meta correspond au trade qui sera exécuté
    si on entre à ce timestep (ou -1 si pas de trade associé).

    Args:
        trades: Liste de trades avec entry_idx, exit_idx, label_meta
        n_samples: Nombre total de samples

    Returns:
        meta_labels: (n_samples,) int - label meta pour chaque timestep
                     1 = trade profitable, 0 = trade perdant, -1 = hors trade
    """
    print("\nMapping trades to timesteps...")

    # Initialiser à -1 (hors trade)
    meta_labels = np.full(n_samples, -1, dtype=int)

    # Pour chaque trade, assigner le label à TOUS les timesteps du trade
    for trade in trades:
        entry_idx = trade['entry_idx']
        exit_idx = trade['exit_idx']
        label = trade['label_meta']

        # Label ALL timesteps from entry to exit (inclusive)
        meta_labels[entry_idx:exit_idx+1] = label

    # Distribution
    n_positive = np.sum(meta_labels == 1)
    n_negative = np.sum(meta_labels == 0)
    n_ignored = np.sum(meta_labels == -1)

    print(f"Meta-labels distribution:")
    print(f"  Positive (1): {n_positive} ({100*n_positive/n_samples:.1f}%)")
    print(f"  Negative (0): {n_negative} ({100*n_negative/n_samples:.1f}%)")
    print(f"  Ignored (-1): {n_ignored} ({100*n_ignored/n_samples:.1f}%)")

    return meta_labels


def save_aligned_dataset(
    output_path: Path,
    data: dict,
    meta_labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    split: str
):
    """
    Sauvegarde dataset avec meta-labels alignés.

    Structure IDENTIQUE au dataset original + meta_labels + predictions.

    Args:
        output_path: Chemin de sortie .npz
        data: Dict original (sequences, labels, timestamps, ohlcv, metadata)
        meta_labels: (n,) Meta-labels alignés
        predictions: (n,) Prédictions modèle primaire
        probabilities: (n,) Probabilités modèle primaire
        split: 'train', 'val', ou 'test'
    """
    print(f"\nSaving aligned dataset: {output_path}")

    # Construire le dict de sauvegarde
    save_dict = {
        # Données originales
        f'X_{split}': data['sequences'],
        f'Y_{split}': data['labels'],
        f'T_{split}': data['timestamps'],
        f'OHLCV_{split}': data['ohlcv'],

        # NOUVEAUX: meta-labels + predictions
        f'meta_labels_{split}': meta_labels,
        f'predictions_{split}': predictions,
        f'probabilities_{split}': probabilities,

        # Metadata enrichie
        'metadata': json.dumps({
            **data['metadata'],
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

    print(f"  Saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / (1024**2):.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='Create aligned meta-labels for meta-labeling')
    parser.add_argument('--indicator', type=str, required=True, choices=['macd', 'rsi', 'cci'],
                        help='Indicator to use')
    parser.add_argument('--filter', type=str, default='kalman', choices=['kalman', 'octave'],
                        help='Filter type')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'],
                        help='Dataset split to process')
    parser.add_argument('--fees', type=float, default=0.001,
                        help='Fees per side (default: 0.001 = 0.1%%)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use for predictions')
    parser.add_argument('--output-dir', type=str, default='data/prepared',
                        help='Output directory')

    args = parser.parse_args()

    print("=" * 80)
    print("CREATE ALIGNED META-LABELS")
    print("=" * 80)
    print(f"Indicator: {args.indicator}")
    print(f"Filter: {args.filter}")
    print(f"Split: {args.split}")
    print(f"Fees: {args.fees} ({args.fees * 100:.2f}%)")
    print(f"Device: {args.device}")

    # 1. Charger dataset
    data = load_dataset(args.indicator, args.filter, args.split)

    # 2. Charger modèle et générer prédictions
    predictions, probabilities = load_model_and_predict(
        args.indicator,
        args.filter,
        data['sequences'],
        device=args.device
    )

    # 3. Backtest aligné (threshold=0.0, pas de filtrage)
    trades, n_samples = backtest_aligned(
        predictions=predictions,
        labels=data['labels'],
        ohlcv=data['ohlcv'],
        fees=args.fees
    )

    # 4. Mapper trades aux timesteps
    meta_labels = map_trades_to_timesteps(trades, n_samples)

    # 5. Sauvegarder dataset avec meta-labels
    output_path = Path(args.output_dir) / f'meta_labels_{args.indicator}_{args.filter}_{args.split}_aligned.npz'
    save_aligned_dataset(
        output_path=output_path,
        data=data,
        meta_labels=meta_labels,
        predictions=predictions,
        probabilities=probabilities,
        split=args.split
    )

    print("\n" + "=" * 80)
    print("ALIGNED META-LABELS CREATED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Create meta-labels for other splits (train/val/test)")
    print(f"2. Train meta-model: python src/train_meta_model_phase217.py --filter {args.filter} --aligned")
    print(f"3. Re-backtest: python tests/test_meta_model_backtest.py --indicator {args.indicator} --split test --aligned")


if __name__ == '__main__':
    main()
