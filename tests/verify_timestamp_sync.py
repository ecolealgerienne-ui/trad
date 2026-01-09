#!/usr/bin/env python3
"""
Vérifie la synchronisation des timestamps entre assets.

CRITIQUE pour le vote multi-indicateurs: les timestamps doivent être alignés
pour que le vote soit cohérent.

Usage:
    python tests/verify_timestamp_sync.py
"""

import numpy as np
from pathlib import Path
from datetime import datetime
import sys


def timestamp_to_datetime(ts):
    """Convertit timestamp en datetime."""
    # Détection auto du format (s, ms, us, ns)
    if ts > 1e18:  # nanoseconds
        return datetime.fromtimestamp(ts / 1e9)
    elif ts > 1e15:  # microseconds
        return datetime.fromtimestamp(ts / 1e6)
    elif ts > 1e12:  # milliseconds
        return datetime.fromtimestamp(ts / 1e3)
    else:  # seconds
        return datetime.fromtimestamp(ts)


def check_dataset_sync(data_file: Path, split: str = 'test'):
    """Vérifie la synchronisation des timestamps dans un dataset."""

    print(f"\n{'='*80}")
    print(f"📂 Fichier: {data_file.name}")
    print(f"   Split: {split}")
    print('='*80)

    data = np.load(data_file, allow_pickle=True)

    # Vérifier clés disponibles
    print(f"\n📋 Clés disponibles: {list(data.keys())}")

    # Charger OHLCV
    ohlcv_key = f'OHLCV_{split}'
    if ohlcv_key not in data:
        print(f"❌ Clé {ohlcv_key} non trouvée!")
        return False

    OHLCV = data[ohlcv_key]  # (n, 7) [timestamp, asset_id, O, H, L, C, V]
    print(f"\n📊 OHLCV shape: {OHLCV.shape}")

    # Identifier les assets
    asset_ids = np.unique(OHLCV[:, 1].astype(int))
    print(f"   Assets trouvés: {len(asset_ids)} → {asset_ids.tolist()}")

    # === ANALYSE PAR ASSET ===
    print(f"\n{'='*80}")
    print("TIMESTAMPS PAR ASSET")
    print('='*80)

    asset_data = {}
    for asset_id in asset_ids:
        mask = OHLCV[:, 1].astype(int) == asset_id
        timestamps = OHLCV[mask, 0]

        ts_min = timestamps.min()
        ts_max = timestamps.max()
        n_samples = len(timestamps)

        # Convertir en datetime
        dt_min = timestamp_to_datetime(ts_min)
        dt_max = timestamp_to_datetime(ts_max)

        asset_data[asset_id] = {
            'timestamps': timestamps,
            'n_samples': n_samples,
            'dt_min': dt_min,
            'dt_max': dt_max
        }

        print(f"\n  Asset {asset_id}:")
        print(f"    Samples: {n_samples:,}")
        print(f"    Début:   {dt_min}")
        print(f"    Fin:     {dt_max}")

    # === VÉRIFICATION ALIGNEMENT ===
    print(f"\n{'='*80}")
    print("VÉRIFICATION ALIGNEMENT TIMESTAMPS")
    print('='*80)

    ref_asset = asset_ids[0]
    ref_timestamps = asset_data[ref_asset]['timestamps']
    ref_n = len(ref_timestamps)

    all_aligned = True
    alignment_issues = []

    for asset_id in asset_ids[1:]:
        other_timestamps = asset_data[asset_id]['timestamps']
        other_n = len(other_timestamps)

        # Comparer longueurs
        if ref_n != other_n:
            print(f"\n❌ Asset {asset_id}: Longueur différente!")
            print(f"   Référence (asset {ref_asset}): {ref_n:,} samples")
            print(f"   Asset {asset_id}: {other_n:,} samples")
            all_aligned = False
            alignment_issues.append(f"Longueur: {ref_asset}={ref_n} vs {asset_id}={other_n}")
            continue

        # Comparer timestamps
        matches = (ref_timestamps == other_timestamps).sum()
        pct = matches / ref_n * 100

        if pct == 100:
            print(f"\n✅ Asset {asset_id}: 100% aligné avec asset {ref_asset}")
        else:
            print(f"\n❌ Asset {asset_id}: {pct:.2f}% aligné ({matches:,}/{ref_n:,})")
            all_aligned = False

            # Trouver désalignements
            mismatches = np.where(ref_timestamps != other_timestamps)[0]
            print(f"   Désalignements: {len(mismatches):,} timestamps")

            # Premiers exemples
            for idx in mismatches[:5]:
                ref_dt = timestamp_to_datetime(ref_timestamps[idx])
                other_dt = timestamp_to_datetime(other_timestamps[idx])
                print(f"   idx {idx}: asset{ref_asset}={ref_dt} vs asset{asset_id}={other_dt}")

            alignment_issues.append(f"Timestamps: {asset_id} désaligné {100-pct:.2f}%")

    # === VÉRIFICATION ORDRE (GROUPÉ vs INTERCALÉ) ===
    print(f"\n{'='*80}")
    print("ORDRE DES DONNÉES (GROUPÉ vs INTERCALÉ)")
    print('='*80)

    asset_changes = np.diff(OHLCV[:, 1].astype(int))
    change_indices = np.where(asset_changes != 0)[0]
    n_changes = len(change_indices)

    print(f"\n  Nombre de changements d'asset: {n_changes:,}")

    if n_changes < len(asset_ids):
        print(f"  → Données GROUPÉES par asset (séquentielles)")
        print(f"     ⚠️  Pour le vote, il faudrait INTERCALER les données!")
    else:
        # Vérifier si vraiment intercalé (changement tous les 1 sample)
        expected_changes = len(OHLCV) - len(OHLCV) // len(asset_ids)
        if n_changes > expected_changes * 0.9:
            print(f"  → Données INTERCALÉES (timestamps synchronisés)")
            print(f"     ✅ Bon pour le vote!")
        else:
            print(f"  → Données PARTIELLEMENT groupées")

    # Afficher structure
    print(f"\n  Structure (premiers 30 samples):")
    for i in range(min(30, len(OHLCV))):
        asset_id = int(OHLCV[i, 1])
        ts = OHLCV[i, 0]
        dt = timestamp_to_datetime(ts)
        print(f"    idx {i:3d}: asset={asset_id}, timestamp={dt}")

    # === CONCLUSION ===
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print('='*80)

    if all_aligned:
        print("\n✅ TOUS LES ASSETS SONT SYNCHRONISÉS!")
        print("   Le vote entre indicateurs/assets est possible.")
    else:
        print("\n❌ PROBLÈME DE SYNCHRONISATION DÉTECTÉ!")
        print("   Le vote entre assets n'est PAS fiable!")
        print("\n   Problèmes trouvés:")
        for issue in alignment_issues:
            print(f"     - {issue}")

    return all_aligned


def main():
    base_path = Path('data/prepared')

    print("="*80)
    print("VÉRIFICATION SYNCHRONISATION TIMESTAMPS ENTRE ASSETS")
    print("="*80)

    # Trouver les fichiers direction-only
    patterns = [
        'dataset_*_macd_direction_only_kalman.npz',
        'dataset_*_rsi_direction_only_kalman.npz',
        'dataset_*_cci_direction_only_kalman.npz'
    ]

    all_ok = True

    for pattern in patterns:
        files = list(base_path.glob(pattern))
        if files:
            data_file = files[0]
            if not check_dataset_sync(data_file, split='test'):
                all_ok = False
            break  # Un seul fichier suffit pour vérifier

    if not files:
        print(f"\n❌ Aucun fichier direction-only trouvé dans {base_path}")
        return 1

    print(f"\n{'='*80}")
    if all_ok:
        print("🎉 VÉRIFICATION TERMINÉE - TOUT EST OK")
    else:
        print("⚠️  VÉRIFICATION TERMINÉE - PROBLÈMES DÉTECTÉS")
    print("="*80)

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
