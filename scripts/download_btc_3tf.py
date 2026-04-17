#!/usr/bin/env python3
"""
Télécharge 3 timeframes BTC (5m, 30m, 1h) sur 3 mois depuis Binance API.

Les 3 fichiers couvrent exactement la même période, pour permettre de valider
le resample 5m → 30m et 5m → 1h contre les données "source" téléchargées.

Sortie:
    data/raw/BTCUSD_3months_5m.csv
    data/raw/BTCUSD_3months_30m.csv
    data/raw/BTCUSD_3months_1h.csv

Usage:
    python scripts/download_btc_3tf.py
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = 'https://api.binance.com/api/v3/klines'
SYMBOL = 'BTCUSDT'
DAYS = 90  # 3 mois
TIMEFRAMES = ['5m', '30m', '1h']


def download_range(symbol, interval, start_ms, end_ms):
    """
    Télécharge OHLCV sur un range [start_ms, end_ms] pour un intervalle donné.
    Binance limite à 1000 candles par requête — on boucle en avançant start_ms.
    """
    all_data = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_ms,
            'limit': 1000,
        }
        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if not data:
                break
            all_data.extend(data)
            current_start = data[-1][0] + 1
            logger.info(f"    [{interval}] +{len(data)} candles (total: {len(all_data)})")
            time.sleep(0.1)  # throttle pour respecter la limite de taux
        except Exception as e:
            logger.error(f"  Erreur {interval}: {e}")
            break

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore',
    ])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def main():
    print("=" * 80)
    print(f"TÉLÉCHARGEMENT {SYMBOL} — 3 timeframes sur {DAYS} jours")
    print("=" * 80)

    # Alignement sur des bordures d'heure entière (xx:00) pour les 3 TFs.
    # - end_ts = dernière heure entière COMPLÈTEMENT passée (bougie 1h close)
    # - start_ts = end_ts - DAYS jours (déjà aligné xx:00)
    # Les 3 CSV auront le même premier et dernier timestamp.
    now = datetime.now()
    end_time = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    start_time = end_time - timedelta(days=DAYS)
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    print(f"Période alignée (xx:00): {start_time} → {end_time}")
    print(f"  (start et end sont des timestamps de 1ère/dernière bougie, pour les 3 TFs)")
    print()

    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for tf in TIMEFRAMES:
        print(f"[{tf}] Téléchargement …")
        df = download_range(SYMBOL, tf, start_ms, end_ms)
        # Filtrer pour garantir start == start_time et end == end_time (bornes incluses)
        df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)].reset_index(drop=True)
        out_path = output_dir / f'BTCUSD_3months_{tf}.csv'
        df.to_csv(out_path, index=False)
        results[tf] = (df, out_path)
        print(f"  ✅ Sauvé: {out_path}")
        print(f"     {len(df):,} candles | {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
        print()

    # Récap final
    print("=" * 80)
    print("RÉSUMÉ")
    print("=" * 80)
    print(f"{'Timeframe':<12} {'Candles':>10} {'Début':<22} {'Fin':<22}")
    print("-" * 80)
    for tf in TIMEFRAMES:
        df, _ = results[tf]
        print(f"{tf:<12} {len(df):>10,} {str(df['timestamp'].iloc[0]):<22} {str(df['timestamp'].iloc[-1]):<22}")
    print("=" * 80)

    # Vérifier alignement : les 3 fichiers commencent et finissent au même timestamp
    starts = {tf: results[tf][0]['timestamp'].iloc[0] for tf in TIMEFRAMES}
    ends = {tf: results[tf][0]['timestamp'].iloc[-1] for tf in TIMEFRAMES}
    all_starts_equal = len(set(starts.values())) == 1
    all_ends_equal = len(set(ends.values())) == 1
    print(f"\nAlignement timestamps :")
    print(f"  Starts identiques : {'✅' if all_starts_equal else '❌'}  ({dict(starts)})")
    print(f"  Ends identiques   : {'✅' if all_ends_equal else '❌'}  ({dict(ends)})")

    # Sanity: les 3 fichiers devraient avoir un ratio cohérent
    n5m = len(results['5m'][0])
    n30m = len(results['30m'][0])
    n1h = len(results['1h'][0])
    # Avec bornes incluses et alignement xx:00, on attend:
    #   n1h  = DAYS*24 + 1
    #   n30m = DAYS*48 + 1
    #   n5m  = DAYS*288 + 1
    expected_1h = DAYS * 24 + 1
    expected_30m = DAYS * 48 + 1
    expected_5m = DAYS * 288 + 1
    print(f"\nSanity check (bornes incluses, alignement xx:00) :")
    print(f"  1h  : {n1h:>6} (attendu {expected_1h})")
    print(f"  30m : {n30m:>6} (attendu {expected_30m})")
    print(f"  5m  : {n5m:>6} (attendu {expected_5m})")
    print()
    print("Prochaine étape: valider resample(5m → 30m) ≈ 30m téléchargé,")
    print("                 valider resample(5m → 1h) ≈ 1h téléchargé.")


if __name__ == '__main__':
    main()
