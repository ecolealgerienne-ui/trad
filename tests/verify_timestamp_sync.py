#!/usr/bin/env python3
"""
Vérifie la synchronisation des timestamps entre indicateurs pour le vote.

CRITIQUE pour le vote multi-indicateurs: les timestamps doivent être alignés
entre MACD, RSI et CCI pour que le vote soit cohérent.

Usage:
    python tests/verify_timestamp_sync.py --split test
"""

import argparse
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


def load_indicator_data(indicator: str, split: str, filter_type: str = 'kalman'):
    """Charge les données d'un indicateur."""
    base_path = Path('data/prepared')
    pattern = f'dataset_*_{indicator}_direction_only_{filter_type}.npz'
    files = list(base_path.glob(pattern))

    if not files:
        return None

    data = np.load(files[0], allow_pickle=True)

    Y = data[f'Y_{split}']        # (n, 3) [timestamp, asset_id, direction]
    OHLCV = data[f'OHLCV_{split}']  # (n, 7) [timestamp, asset_id, O, H, L, C, V]

    return {
        'Y': Y,
        'OHLCV': OHLCV,
        'filename': files[0].name
    }


def check_indicator_sync(split: str = 'test', filter_type: str = 'kalman'):
    """Vérifie la synchronisation entre les 3 indicateurs."""

    print("="*100)
    print(f"VÉRIFICATION SYNCHRONISATION TIMESTAMPS ENTRE INDICATEURS")
    print(f"Split: {split} | Filter: {filter_type}")
    print("="*100)

    # Charger les 3 indicateurs
    indicators = ['macd', 'rsi', 'cci']
    data = {}

    print("\n📂 CHARGEMENT DES DATASETS:")
    for ind in indicators:
        data[ind] = load_indicator_data(ind, split, filter_type)
        if data[ind]:
            n_samples = len(data[ind]['Y'])
            print(f"  ✅ {ind.upper()}: {n_samples:,} samples - {data[ind]['filename']}")
        else:
            print(f"  ❌ {ind.upper()}: Non trouvé!")

    # Vérifier qu'on a au moins MACD
    if data['macd'] is None:
        print("\n❌ MACD requis!")
        return False

    # === COMPARAISON GLOBALE ===
    print("\n" + "="*100)
    print("COMPARAISON GLOBALE (tous assets confondus)")
    print("="*100)

    ref_ind = 'macd'
    ref_Y = data[ref_ind]['Y']
    ref_OHLCV = data[ref_ind]['OHLCV']
    ref_n = len(ref_Y)

    all_ok = True

    for ind in ['rsi', 'cci']:
        if data[ind] is None:
            continue

        other_Y = data[ind]['Y']
        other_n = len(other_Y)

        print(f"\n  MACD vs {ind.upper()}:")

        # Comparer longueurs
        if ref_n != other_n:
            print(f"    ❌ Longueurs différentes: MACD={ref_n:,} vs {ind.upper()}={other_n:,}")
            all_ok = False
            continue

        # Comparer timestamps
        ts_match = (ref_Y[:, 0] == other_Y[:, 0]).sum()
        ts_pct = ts_match / ref_n * 100

        # Comparer asset_ids
        asset_match = (ref_Y[:, 1] == other_Y[:, 1]).sum()
        asset_pct = asset_match / ref_n * 100

        print(f"    Timestamps alignés: {ts_match:,}/{ref_n:,} ({ts_pct:.2f}%)")
        print(f"    Asset IDs alignés:  {asset_match:,}/{ref_n:,} ({asset_pct:.2f}%)")

        if ts_pct == 100 and asset_pct == 100:
            print(f"    ✅ Parfaitement synchronisé!")
        else:
            print(f"    ❌ Désalignement détecté!")
            all_ok = False

            # Montrer premiers désalignements
            if ts_pct < 100:
                mismatches = np.where(ref_Y[:, 0] != other_Y[:, 0])[0][:5]
                print(f"    Premiers désalignements timestamps:")
                for idx in mismatches:
                    ref_ts = timestamp_to_datetime(ref_Y[idx, 0])
                    other_ts = timestamp_to_datetime(other_Y[idx, 0])
                    print(f"      idx {idx}: MACD={ref_ts} vs {ind.upper()}={other_ts}")

    # === COMPARAISON PAR ASSET ===
    print("\n" + "="*100)
    print("COMPARAISON PAR ASSET")
    print("="*100)

    asset_ids = np.unique(ref_OHLCV[:, 1].astype(int))
    print(f"\nAssets trouvés: {len(asset_ids)} → {asset_ids.tolist()}")

    for asset_id in asset_ids:
        print(f"\n  Asset {asset_id}:")

        # Masque MACD pour cet asset
        ref_mask = ref_OHLCV[:, 1].astype(int) == asset_id
        ref_asset_ts = ref_Y[ref_mask, 0]
        ref_asset_n = len(ref_asset_ts)

        ref_ts_min = timestamp_to_datetime(ref_asset_ts.min())
        ref_ts_max = timestamp_to_datetime(ref_asset_ts.max())

        print(f"    MACD: {ref_asset_n:,} samples | {ref_ts_min} → {ref_ts_max}")

        for ind in ['rsi', 'cci']:
            if data[ind] is None:
                continue

            other_OHLCV = data[ind]['OHLCV']
            other_Y = data[ind]['Y']

            other_mask = other_OHLCV[:, 1].astype(int) == asset_id
            other_asset_ts = other_Y[other_mask, 0]
            other_asset_n = len(other_asset_ts)

            if other_asset_n == 0:
                print(f"    {ind.upper()}: ❌ Aucune donnée!")
                all_ok = False
                continue

            other_ts_min = timestamp_to_datetime(other_asset_ts.min())
            other_ts_max = timestamp_to_datetime(other_asset_ts.max())

            # Comparer
            if ref_asset_n != other_asset_n:
                status = "❌ Longueur diff"
                all_ok = False
            elif (ref_asset_ts == other_asset_ts).all():
                status = "✅ Synchronisé"
            else:
                match_pct = (ref_asset_ts == other_asset_ts).sum() / ref_asset_n * 100
                status = f"❌ {match_pct:.1f}% aligné"
                all_ok = False

            print(f"    {ind.upper()}: {other_asset_n:,} samples | {other_ts_min} → {other_ts_max} | {status}")

    # === CONCLUSION ===
    print("\n" + "="*100)
    print("CONCLUSION")
    print("="*100)

    if all_ok:
        print("\n✅ TOUS LES INDICATEURS SONT SYNCHRONISÉS!")
        print("   Le vote MACD + RSI + CCI est possible.")
    else:
        print("\n❌ PROBLÈME DE SYNCHRONISATION DÉTECTÉ!")
        print("   Le vote entre indicateurs n'est PAS fiable!")
        print("\n   Solution: Régénérer les datasets avec le même timestamp range.")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description='Vérifier synchronisation timestamps')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Split à vérifier (défaut: test)')
    parser.add_argument('--filter', type=str, default='kalman',
                        choices=['kalman', 'octave20'],
                        help='Type de filtre (défaut: kalman)')

    args = parser.parse_args()

    success = check_indicator_sync(split=args.split, filter_type=args.filter)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
