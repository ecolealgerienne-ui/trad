#!/usr/bin/env python3
"""
Télécharge les données BTCUSD 5min depuis Binance API.

Les données sont sauvegardées dans data/raw/BTCUSD_all_5m.csv
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_binance_klines(symbol='BTCUSDT', interval='5m', days=30):
    """
    Télécharge les données OHLCV depuis Binance.

    Args:
        symbol: Paire de trading (ex: BTCUSDT)
        interval: Intervalle (1m, 5m, 15m, 1h, etc.)
        days: Nombre de jours à télécharger

    Returns:
        DataFrame avec colonnes: timestamp, open, high, low, close, volume
    """
    base_url = 'https://api.binance.com/api/v3/klines'

    # Calculer les timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    all_data = []

    logger.info(f"Téléchargement {symbol} {interval} depuis {start_time.date()} jusqu'à {end_time.date()}")

    # Binance limite à 1000 candles par requête
    current_start = start_ms

    while current_start < end_ms:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_ms,
            'limit': 1000
        }

        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            all_data.extend(data)

            # Mettre à jour le timestamp de départ pour la prochaine requête
            current_start = data[-1][0] + 1

            logger.info(f"  Téléchargé {len(data)} candles, total: {len(all_data)}")

            # Pause pour respecter les limites de taux
            time.sleep(0.1)

        except Exception as e:
            logger.error(f"Erreur lors du téléchargement: {e}")
            break

    # Convertir en DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    # Ne garder que les colonnes nécessaires
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # Convertir les types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)

    # Trier par timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"✅ Téléchargement terminé: {len(df)} candles")
    logger.info(f"   Période: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

    return df


def main():
    """
    Télécharge les données BTC et les sauvegarde.
    """
    print("="*80)
    print("TÉLÉCHARGEMENT DONNÉES BTCUSD 5MIN DEPUIS BINANCE")
    print("="*80)

    # Télécharger 30 jours de données (environ 8640 candles 5min)
    df = download_binance_klines(symbol='BTCUSDT', interval='5m', days=30)

    # Créer le dossier si nécessaire
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarder
    output_path = output_dir / 'BTCUSD_all_5m.csv'
    df.to_csv(output_path, index=False)

    print(f"\n✅ Données sauvegardées: {output_path}")
    print(f"   {len(df)} candles de 5 minutes")
    print(f"   Période: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    print(f"   Prix min: ${df['close'].min():.2f}")
    print(f"   Prix max: ${df['close'].max():.2f}")
    print(f"   Prix moyen: ${df['close'].mean():.2f}")

    # Stats
    print(f"\n📊 STATISTIQUES:")
    print(f"   Nombre de lignes: {len(df)}")
    print(f"   Première bougie: {df['timestamp'].iloc[0]}")
    print(f"   Dernière bougie: {df['timestamp'].iloc[-1]}")
    print(f"   Durée totale: {(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days} jours")

    return df


if __name__ == '__main__':
    main()
