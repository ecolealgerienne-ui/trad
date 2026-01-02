#!/usr/bin/env python3
"""
Analyse la profondeur des données disponibles sur Binance et télécharge l'historique complet.

Usage:
    # Voir la profondeur disponible pour chaque actif
    python scripts/check_binance_depth.py --check

    # Télécharger tout l'historique pour BTC et ETH
    python scripts/check_binance_depth.py --download BTCUSDT ETHUSDT

    # Télécharger tout l'historique pour les top 10 cryptos
    python scripts/check_binance_depth.py --download-top 10
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Ajouter le dossier src au path pour importer les constantes
SCRIPT_DIR = Path(__file__).parent.absolute()
SRC_DIR = SCRIPT_DIR.parent / 'src'
sys.path.insert(0, str(SRC_DIR))

from constants import DATA_TRAD_DIR

# Configuration
OUTPUT_DIR = Path(SRC_DIR / DATA_TRAD_DIR).resolve()

# Top cryptos par capitalisation (avec dates de listing approximatives sur Binance)
TOP_CRYPTOS = {
    'BTCUSDT': '2017-08-17',   # Plus ancien
    'ETHUSDT': '2017-08-17',   # Plus ancien
    'BNBUSDT': '2017-11-06',
    'XRPUSDT': '2018-05-04',
    'ADAUSDT': '2018-04-17',
    'DOGEUSDT': '2019-07-05',
    'SOLUSDT': '2020-08-11',
    'MATICUSDT': '2019-04-26',
    'DOTUSDT': '2020-08-18',
    'LTCUSDT': '2017-12-13',
    'AVAXUSDT': '2020-09-22',
    'LINKUSDT': '2019-01-16',
    'ATOMUSDT': '2019-04-22',
    'UNIUSDT': '2020-09-17',
    'XLMUSDT': '2018-05-04',
}

# Mapping pour noms de fichiers
SYMBOL_MAP = {
    'BTCUSDT': 'BTCUSD',
    'ETHUSDT': 'ETHUSD',
    'BNBUSDT': 'BNBUSD',
    'XRPUSDT': 'XRPUSD',
    'ADAUSDT': 'ADAUSD',
    'DOGEUSDT': 'DOGEUSD',
    'SOLUSDT': 'SOLUSD',
    'MATICUSDT': 'MATICUSD',
    'DOTUSDT': 'DOTUSD',
    'LTCUSDT': 'LTCUSD',
    'AVAXUSDT': 'AVAXUSD',
    'LINKUSDT': 'LINKUSD',
    'ATOMUSDT': 'ATOMUSD',
    'UNIUSDT': 'UNIUSD',
    'XLMUSDT': 'XLMUSD',
}


def get_binance_client():
    """Crée le client Binance."""
    try:
        from binance.client import Client
    except ImportError:
        print("❌ python-binance non installé. Exécutez:")
        print("   pip install python-binance")
        raise

    api_key = os.environ.get('BINANCE_API_KEY', '')
    api_secret = os.environ.get('BINANCE_API_SECRET', '')

    if not api_key or not api_secret:
        print("⚠️  Clés API non trouvées. Définissez:")
        print("   export BINANCE_API_KEY='...'")
        print("   export BINANCE_API_SECRET='...'")

    return Client(api_key, api_secret)


def get_earliest_timestamp(client, symbol: str, interval: str = '5m') -> datetime:
    """Trouve la date la plus ancienne disponible pour un symbole."""
    from binance.client import Client as BinanceClient

    interval_map = {
        '1m': BinanceClient.KLINE_INTERVAL_1MINUTE,
        '5m': BinanceClient.KLINE_INTERVAL_5MINUTE,
    }

    try:
        # Récupérer les premières klines disponibles
        klines = client.get_historical_klines(
            symbol,
            interval_map.get(interval, BinanceClient.KLINE_INTERVAL_5MINUTE),
            "1 Jan, 2017",  # Date très ancienne
            limit=1
        )

        if klines:
            return datetime.fromtimestamp(klines[0][0] / 1000)
        return None

    except Exception as e:
        print(f"  Erreur pour {symbol}: {e}")
        return None


def check_depth(client, symbols: list, interval: str = '5m'):
    """Vérifie la profondeur des données pour chaque symbole."""
    print("\n" + "=" * 80)
    print("PROFONDEUR DES DONNÉES BINANCE")
    print("=" * 80)
    print(f"{'Symbole':<12} {'Première donnée':<20} {'Jours dispo':<12} {'Bougies 5min estimées'}")
    print("-" * 80)

    results = []

    for symbol in symbols:
        earliest = get_earliest_timestamp(client, symbol, interval)

        if earliest:
            days = (datetime.now() - earliest).days
            candles_5m = days * 24 * 12  # 12 bougies 5min par heure

            print(f"{symbol:<12} {earliest.strftime('%Y-%m-%d'):<20} {days:>6} jours   ~{candles_5m:>10,} bougies")

            results.append({
                'symbol': symbol,
                'earliest': earliest,
                'days': days,
                'candles_5m': candles_5m
            })
        else:
            print(f"{symbol:<12} {'N/A':<20} {'N/A':<12}")

        time.sleep(0.5)  # Rate limiting

    print("-" * 80)

    # Trier par nombre de jours (plus ancien en premier)
    results.sort(key=lambda x: x['days'], reverse=True)

    print("\n📊 TOP ACTIFS PAR PROFONDEUR:")
    for i, r in enumerate(results[:10], 1):
        print(f"  {i}. {r['symbol']}: {r['days']} jours ({r['candles_5m']:,} bougies 5min)")

    return results


def download_full_history(client, symbol: str, interval: str = '5m', output_dir: Path = OUTPUT_DIR):
    """Télécharge tout l'historique disponible pour un symbole."""
    from binance.client import Client as BinanceClient

    interval_map = {
        '1m': BinanceClient.KLINE_INTERVAL_1MINUTE,
        '5m': BinanceClient.KLINE_INTERVAL_5MINUTE,
        '15m': BinanceClient.KLINE_INTERVAL_15MINUTE,
        '30m': BinanceClient.KLINE_INTERVAL_30MINUTE,
    }

    # Date de listing approximative ou très ancienne
    start_date = TOP_CRYPTOS.get(symbol, "1 Jan, 2017")

    print(f"\n📥 Téléchargement {symbol} {interval} depuis {start_date}...")

    try:
        klines = client.get_historical_klines(
            symbol,
            interval_map.get(interval, BinanceClient.KLINE_INTERVAL_5MINUTE),
            start_date
        )

        if not klines:
            print(f"  ❌ Aucune donnée pour {symbol}")
            return None

        # Nettoyer les données
        for line in klines:
            del line[6:]

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df = df.sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)

        # Sauvegarder
        output_dir.mkdir(parents=True, exist_ok=True)
        symbol_name = SYMBOL_MAP.get(symbol, symbol.replace('USDT', 'USD'))
        filename = f"{symbol_name}_all_{interval}.csv"
        filepath = output_dir / filename

        df.to_csv(filepath, index=False)

        size_mb = filepath.stat().st_size / (1024 * 1024)
        first_date = df['timestamp'].iloc[0]
        last_date = df['timestamp'].iloc[-1]
        days = (last_date - first_date).days

        print(f"  ✅ {len(df):,} bougies téléchargées")
        print(f"     Période: {first_date.date()} → {last_date.date()} ({days} jours)")
        print(f"     Fichier: {filepath} ({size_mb:.2f} MB)")

        return filepath

    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyse et télécharge l\'historique complet depuis Binance',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--check', '-c',
        action='store_true',
        help='Vérifier la profondeur disponible pour les top cryptos'
    )

    parser.add_argument(
        '--download', '-d',
        nargs='+',
        help='Télécharger l\'historique complet pour ces symboles'
    )

    parser.add_argument(
        '--download-top',
        type=int,
        metavar='N',
        help='Télécharger l\'historique des N cryptos avec le plus de profondeur'
    )

    parser.add_argument(
        '--interval', '-i',
        default='5m',
        choices=['1m', '5m', '15m', '30m'],
        help='Intervalle (défaut: 5m)'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=OUTPUT_DIR,
        help=f'Dossier de sortie (défaut: {OUTPUT_DIR})'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Résoudre chemin
    output_dir = args.output
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent.parent / args.output.name

    client = get_binance_client()

    if args.check:
        check_depth(client, list(TOP_CRYPTOS.keys()), args.interval)

    elif args.download:
        print("=" * 70)
        print("TÉLÉCHARGEMENT HISTORIQUE COMPLET")
        print("=" * 70)

        for symbol in args.download:
            download_full_history(client, symbol, args.interval, output_dir)
            time.sleep(2)  # Pause entre les téléchargements

    elif args.download_top:
        print("=" * 70)
        print(f"TÉLÉCHARGEMENT TOP {args.download_top} CRYPTOS")
        print("=" * 70)

        # Vérifier la profondeur d'abord
        results = check_depth(client, list(TOP_CRYPTOS.keys()), args.interval)

        # Télécharger les N premiers
        print(f"\n📥 Téléchargement des {args.download_top} actifs avec le plus de données...")

        for r in results[:args.download_top]:
            download_full_history(client, r['symbol'], args.interval, output_dir)
            time.sleep(2)

    else:
        print("Usage:")
        print("  --check              Vérifier la profondeur disponible")
        print("  --download SYMBOLS   Télécharger l'historique complet")
        print("  --download-top N     Télécharger les N cryptos avec le plus de données")
        print()
        print("Exemple:")
        print("  python scripts/check_binance_depth.py --check")
        print("  python scripts/check_binance_depth.py --download BTCUSDT ETHUSDT")
        print("  python scripts/check_binance_depth.py --download-top 5")


if __name__ == '__main__':
    main()
