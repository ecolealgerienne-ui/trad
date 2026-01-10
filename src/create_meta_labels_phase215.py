#!/usr/bin/env python3
"""
Script de création des meta-labels pour Phase 2.15 (Meta-Labeling).

Objectif:
- Charger les datasets Kalman direction-only existants (.npz)
- Simuler backtest Oracle pour obtenir les points d'entrée/sortie
- Appliquer Triple Barrier Method pour créer meta-labels
- Label = 1 si trade profitable ET duration >= min_duration
- Label = 0 si trade perdant OU duration < min_duration (micro-sortie)
- Sauvegarder nouveau fichier avec MÊME structure + meta-labels

CRITIQUE: Synchronisation timestamps préservée pour éviter data leakage.

Usage:
    python src/create_meta_labels_phase215.py --indicator macd --filter kalman --split test
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, List
import json


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


def backtest_single_asset(
    asset_labels: np.ndarray,
    asset_opens: np.ndarray,
    start_idx: int,
    fees: float = 0.001
) -> Tuple[List[dict], float, float]:
    """
    Backtest pour UN SEUL asset.

    LOGIQUE CAUSALE:
    - Signal à index i → Exécution à Open[i+1]
    - Toujours en position (LONG ou SHORT, jamais FLAT)
    - Direction: 1=UP→LONG, 0=DOWN→SHORT

    Args:
        asset_labels: (n,) Direction labels pour cet asset
        asset_opens: (n,) Prix Open pour cet asset
        start_idx: Index de départ global (pour entry_idx/exit_idx)
        fees: Frais par side

    Returns:
        (trades, pnl_gross, pnl_net)
    """
    n_samples = len(asset_labels)
    trades = []
    position = 'FLAT'
    entry_idx = 0
    entry_price = 0.0
    pnl_gross = 0.0
    pnl_net = 0.0

    # Boucle principale
    for i in range(n_samples - 1):  # -1 car besoin de Open[i+1]
        direction = int(asset_labels[i])
        target = 'LONG' if direction == 1 else 'SHORT'

        # Première entrée
        if position == 'FLAT':
            position = target
            entry_idx = i
            entry_price = asset_opens[i + 1]  # Entrée à Open[i+1]
            continue

        # Changement de position (reversal)
        if position != target:
            # Sortie à Open[i+1]
            exit_price = asset_opens[i + 1]

            # Calcul PnL
            if position == 'LONG':
                pnl = (exit_price - entry_price) / entry_price
            else:  # SHORT
                pnl = (entry_price - exit_price) / entry_price

            # Frais: entrée + sortie
            trade_fees = 2 * fees
            pnl_after_fees = pnl - trade_fees

            pnl_gross += pnl
            pnl_net += pnl_after_fees

            # Enregistrer trade (avec indices globaux)
            trades.append({
                'entry_idx': start_idx + entry_idx,
                'exit_idx': start_idx + i,
                'direction': position,
                'pnl': pnl,
                'duration': i - entry_idx,
                'exit_reason': 'DIRECTION_FLIP'
            })

            # Nouvelle position (reversal immédiat)
            position = target
            entry_idx = i
            entry_price = asset_opens[i + 1]

    # Fermer position finale
    if position != 'FLAT':
        exit_price = asset_opens[-1]

        if position == 'LONG':
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price

        trade_fees = 2 * fees
        pnl_after_fees = pnl - trade_fees

        pnl_gross += pnl
        pnl_net += pnl_after_fees

        trades.append({
            'entry_idx': start_idx + entry_idx,
            'exit_idx': start_idx + n_samples - 1,
            'direction': position,
            'pnl': pnl,
            'duration': n_samples - 1 - entry_idx,
            'exit_reason': 'END_OF_ASSET'
        })

    return trades, pnl_gross, pnl_net


def simulate_oracle_backtest(
    labels: np.ndarray,
    ohlcv: np.ndarray,
    timestamps: np.ndarray,
    fees: float = 0.001
) -> Tuple[List[dict], float, float]:
    """
    Simule backtest Oracle (labels parfaits) pour obtenir les trades.

    CRITIQUE: Le dataset contient PLUSIEURS assets concaténés!
    On doit faire le backtest PAR ASSET pour éviter de calculer
    des PnL entre prix BTC et ETH.

    Args:
        labels: Labels Oracle (n, 3) - [timestamp, asset_id, direction]
        ohlcv: Prix OHLCV (n, 7) - [timestamp, asset_id, O, H, L, C, V]
        timestamps: Timestamps (n, 3) - [timestamp, asset_id, is_transition]
        fees: Frais par trade (0.001 = 0.1%)

    Returns:
        (trades, pnl_gross, pnl_net)
        trades: Liste de dict avec {entry_idx, exit_idx, direction, pnl, duration}
        pnl_gross: PnL brut cumulé
        pnl_net: PnL net après frais
    """
    print("Simulating Oracle backtest (per-asset)...")

    # Extraire colonnes
    asset_ids = ohlcv[:, 1].astype(int)  # Colonne 1 = asset_id
    opens = ohlcv[:, 2]  # Colonne 2 = Open
    direction_labels = labels[:, 2].astype(int)  # Colonne 2 = direction

    # Identifier les assets uniques
    unique_assets = np.unique(asset_ids)
    print(f"  Assets detected: {len(unique_assets)} ({unique_assets})")

    all_trades = []
    total_pnl_gross = 0.0
    total_pnl_net = 0.0

    # Backtest PAR ASSET
    for asset_id in unique_assets:
        # Masque pour cet asset
        mask = asset_ids == asset_id
        asset_labels = direction_labels[mask]
        asset_opens = opens[mask]

        # Index de départ global (pour entry_idx/exit_idx)
        start_idx = np.where(mask)[0][0]

        print(f"    Asset {int(asset_id)}: {len(asset_labels):,} samples (start_idx={start_idx})")

        # Backtest cet asset
        trades, pnl_gross, pnl_net = backtest_single_asset(
            asset_labels, asset_opens, start_idx, fees
        )

        all_trades.extend(trades)
        total_pnl_gross += pnl_gross
        total_pnl_net += pnl_net

    print(f"\n  Oracle backtest completed:")
    print(f"    Total trades: {len(all_trades)}")
    print(f"    PnL Gross: {total_pnl_gross*100:.2f}%")
    print(f"    PnL Net: {total_pnl_net*100:.2f}%")
    if len(all_trades) > 0:
        print(f"    Avg duration: {np.mean([t['duration'] for t in all_trades]):.1f} periods")

    return all_trades, total_pnl_gross, total_pnl_net


def create_meta_labels_triple_barrier(
    trades: List[dict],
    min_duration: int = 5,
    pnl_threshold: float = 0.0
) -> np.ndarray:
    """
    Crée meta-labels à partir des trades Oracle avec Triple Barrier.

    Règle Phase 2.15:
    - Label = 1 SI: trade profitable (PnL > threshold) ET duration >= min_duration
    - Label = 0 SI: trade perdant (PnL <= threshold) OU duration < min_duration

    Objectif: Filtrer micro-sorties (duration < 5 périodes) qui détruisent le PnL.

    Args:
        trades: Liste de trades Oracle
        min_duration: Durée minimale (périodes) pour accepter le trade
        pnl_threshold: Seuil PnL pour considérer profitable (0.0 = break-even)

    Returns:
        meta_labels (n_trades,) - 0 ou 1
    """
    print(f"Creating meta-labels (min_duration={min_duration}, pnl_threshold={pnl_threshold})...")

    meta_labels = []

    for trade in trades:
        pnl = trade['pnl']
        duration = trade['duration']

        # Règle: Profitable ET pas micro-sortie
        if pnl > pnl_threshold and duration >= min_duration:
            label = 1  # Accepter le trade
        else:
            label = 0  # Rejeter le trade

        meta_labels.append(label)

    meta_labels = np.array(meta_labels, dtype=np.int32)

    # Statistiques
    n_positive = np.sum(meta_labels == 1)
    n_negative = np.sum(meta_labels == 0)

    print(f"  Meta-labels distribution:")
    print(f"    Positive (1): {n_positive} ({100*n_positive/len(meta_labels):.1f}%)")
    print(f"    Negative (0): {n_negative} ({100*n_negative/len(meta_labels):.1f}%)")

    # Analyse des rejetés
    rejected_profitable = sum(1 for t in trades if t['pnl'] > 0 and t['duration'] < min_duration)
    rejected_losing = sum(1 for t in trades if t['pnl'] <= 0)

    print(f"  Rejection reasons:")
    print(f"    Micro-exits (profitable but < {min_duration}p): {rejected_profitable}")
    print(f"    Losing trades: {rejected_losing}")

    return meta_labels


def map_trade_labels_to_timesteps(
    trades: List[dict],
    meta_labels: np.ndarray,
    n_timesteps: int
) -> np.ndarray:
    """
    Mappe les meta-labels des trades vers les timesteps individuels.

    Chaque timestep reçoit le label du trade auquel il appartient.
    Si un timestep n'appartient à aucun trade (flat), label = -1 (ignore).

    IMPORTANT: Pour les reversals (entry_idx == last_exit), le timestep de
    transition appartient au trade sortant. Le nouveau trade commence après.

    Args:
        trades: Liste de trades avec entry_idx et exit_idx
        meta_labels: Labels des trades (n_trades,)
        n_timesteps: Nombre total de timesteps

    Returns:
        timestep_labels (n_timesteps,) - 0, 1, ou -1 (ignore)
    """
    print("Mapping trade labels to timesteps...")

    timestep_labels = np.full(n_timesteps, -1, dtype=np.int32)
    last_exit = -1
    n_reversals = 0

    for trade, label in zip(trades, meta_labels):
        entry_idx = trade['entry_idx']
        exit_idx = trade['exit_idx']

        # Si c'est un reversal (entry == last_exit), le timestep de transition
        # appartient au trade précédent. On commence après.
        if entry_idx == last_exit:
            entry_idx = entry_idx + 1
            n_reversals += 1

        # Assigner label à tous les timesteps du trade
        if entry_idx <= exit_idx:  # Vérifier que le range est valide
            timestep_labels[entry_idx:exit_idx+1] = label

        last_exit = exit_idx

    n_labeled = np.sum(timestep_labels != -1)
    n_ignored = np.sum(timestep_labels == -1)

    print(f"  Labeled timesteps: {n_labeled}/{n_timesteps} ({100*n_labeled/n_timesteps:.1f}%)")
    print(f"  Ignored timesteps: {n_ignored} ({100*n_ignored/n_timesteps:.1f}%)")
    print(f"  Reversals detected: {n_reversals}")

    return timestep_labels


def save_meta_dataset(
    output_path: Path,
    sequences: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    ohlcv: np.ndarray,
    meta_labels: np.ndarray,
    trades: List[dict],
    metadata: dict,
    args: argparse.Namespace
):
    """
    Sauvegarde nouveau dataset avec meta-labels.

    CRITIQUE: Préserve la structure originale + ajoute nouvelles données.
    """
    print(f"Saving meta-dataset to: {output_path}")

    # Métadonnées enrichies
    meta_config = {
        'min_duration': args.min_duration,
        'pnl_threshold': args.pnl_threshold,
        'n_trades': len(trades),
        'n_positive': int(np.sum(meta_labels == 1)),
        'n_negative': int(np.sum(meta_labels == 0)),
        'fees': args.fees
    }

    metadata_enriched = {
        **metadata,
        'meta_labeling': meta_config,
        'indicator': args.indicator,
        'filter_type': args.filter
    }

    # Sauvegarder avec MÊME structure + nouvelles données
    np.savez_compressed(
        output_path,
        # Données originales (préservées)
        X=sequences,
        Y=labels,
        T=timestamps,
        OHLCV=ohlcv,
        # Nouvelles données
        meta_labels=meta_labels,      # (n,) - 0, 1, ou -1
        # Métadonnées
        metadata=json.dumps(metadata_enriched),
        trades=trades  # Liste de dict (sauvegardé comme objet)
    )

    print(f"  Dataset saved successfully")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Create meta-labels for Phase 2.15')
    parser.add_argument('--indicator', type=str, required=True, choices=['macd', 'rsi', 'cci'],
                        help='Indicator to process')
    parser.add_argument('--filter', type=str, default='kalman', choices=['kalman', 'octave'],
                        help='Filter type (default: kalman)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to process (default: test)')
    parser.add_argument('--min-duration', type=int, default=5,
                        help='Minimum trade duration to accept (default: 5 periods)')
    parser.add_argument('--pnl-threshold', type=float, default=0.0,
                        help='PnL threshold for profitable trades (default: 0.0)')
    parser.add_argument('--fees', type=float, default=0.001,
                        help='Trading fees per trade (default: 0.001 = 0.1%)')
    parser.add_argument('--output-dir', type=str, default='data/prepared',
                        help='Output directory for meta-datasets')

    args = parser.parse_args()

    print("="*80)
    print("CREATE META-LABELS - Phase 2.15")
    print("="*80)
    print(f"Indicator: {args.indicator}")
    print(f"Filter: {args.filter}")
    print(f"Split: {args.split}")
    print(f"Min duration: {args.min_duration} periods")
    print(f"PnL threshold: {args.pnl_threshold}")
    print(f"Fees: {args.fees*100:.2f}%")
    print()

    # 1. Charger dataset avec timestamps
    data = load_dataset(args.indicator, args.filter, args.split)
    sequences = data['sequences']
    labels = data['labels']
    timestamps = data['timestamps']
    ohlcv = data['ohlcv']
    metadata = data['metadata']

    # 2. Simuler backtest Oracle
    trades, pnl_gross, pnl_net = simulate_oracle_backtest(
        labels=labels,
        ohlcv=ohlcv,
        timestamps=timestamps,
        fees=args.fees
    )
    print()

    # 5. Créer meta-labels avec Triple Barrier
    trade_meta_labels = create_meta_labels_triple_barrier(
        trades=trades,
        min_duration=args.min_duration,
        pnl_threshold=args.pnl_threshold
    )
    print()

    # 6. Mapper labels vers timesteps
    timestep_meta_labels = map_trade_labels_to_timesteps(
        trades=trades,
        meta_labels=trade_meta_labels,
        n_timesteps=len(sequences)
    )
    print()

    # 6. Sauvegarder nouveau dataset
    output_path = Path(args.output_dir) / f'meta_labels_{args.indicator}_{args.filter}_{args.split}.npz'
    save_meta_dataset(
        output_path=output_path,
        sequences=sequences,
        labels=labels,
        timestamps=timestamps,
        ohlcv=ohlcv,
        meta_labels=timestep_meta_labels,
        trades=trades,
        metadata=metadata,
        args=args
    )
    print()

    # 8. Statistiques finales
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Dataset: {args.indicator} {args.filter} {args.split}")
    print(f"Total timesteps: {len(sequences)}")
    print(f"Total trades (Oracle): {len(trades)}")
    print(f"Meta-labels positive: {np.sum(timestep_meta_labels == 1)} ({100*np.sum(timestep_meta_labels == 1)/len(sequences):.1f}%)")
    print(f"Meta-labels negative: {np.sum(timestep_meta_labels == 0)} ({100*np.sum(timestep_meta_labels == 0)/len(sequences):.1f}%)")
    print(f"Meta-labels ignored: {np.sum(timestep_meta_labels == -1)} ({100*np.sum(timestep_meta_labels == -1)/len(sequences):.1f}%)")
    print(f"Oracle PnL Gross: {pnl_gross*100:.2f}%")
    print(f"Oracle PnL Net: {pnl_net*100:.2f}%")
    print(f"Output: {output_path}")
    print()
    print("✅ Meta-labels created successfully!")


if __name__ == '__main__':
    main()
