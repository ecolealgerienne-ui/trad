#!/usr/bin/env python3
"""
Téléchargement des données OHLCV depuis Binance API.

Usage:
    # Télécharger BTC et ETH en 5min (6 mois)
    python scripts/download_binance_data.py

    # Télécharger seulement BTC en 1min (30 jours)
    python scripts/download_binance_data.py --symbols BTCUSDT --intervals 1m --days 30

    # Télécharger tous les timeframes pour augmentation
    python scripts/download_binance_data.py --intervals 1m 5m 15m 30m --days 180

    # Lister les fichiers existants
    python scripts/download_binance_data.py --list
"""

import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration par défaut
DEFAULT_SYMBOLS = ['BTCUSDT', 'ETHUSDT']
DEFAULT_INTERVALS = ['5m']
DEFAULT_DAYS = 180
OUTPUT_DIR = Path('../data_trad')

# Mapping intervalle -> nom fichier
INTERVAL_MAP = {
    '1m': '1m',
    '5m': '5m',
    '15m': '15m',
    '30m': '30m',
    '1h': '1h',
    '4h': '4h',
    '1d': '1d',
}

# Mapping symbole Binance -> nom fichier
SYMBOL_MAP = {
    'BTCUSDT': 'BTCUSD',
    'ETHUSDT': 'ETHUSD',
    'MATICUSDT': 'MATICUSD',
    'SOLUSDT': 'SOLUSD',
    'BNBUSDT': 'BNBUSD',
}


def download_klines(
    symbol: str = 'BTCUSDT',
    interval: str = '5m',
    days: int = 180,
    retries: int = 3
) -> pd.DataFrame:
    """
    Télécharge les données OHLCV depuis Binance.

    Args:
        symbol: Paire de trading (ex: BTCUSDT)
        interval: Intervalle (1m, 5m, 15m, 30m, 1h, 4h, 1d)
        days: Nombre de jours à télécharger
        retries: Nombre de tentatives en cas d'erreur

    Returns:
        DataFrame avec colonnes: timestamp, open, high, low, close, volume
    """
    base_url = 'https://api.binance.com/api/v3/klines'

    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    all_data = []
    current_start = start_ms

    logger.info(f"Téléchargement {symbol} {interval} ({days} jours)...")
    logger.info(f"  Période: {start_time.strftime('%Y-%m-%d')} → {end_time.strftime('%Y-%m-%d')}")

    while current_start < end_ms:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_ms,
            'limit': 1000
        }

        for attempt in range(retries):
            try:
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if not data:
                    break

                all_data.extend(data)
                current_start = data[-1][0] + 1

                if len(all_data) % 10000 == 0:
                    logger.info(f"  {len(all_data):,} bougies...")

                time.sleep(0.1)  # Rate limiting
                break

            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"  Erreur, retry dans {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"  Échec après {retries} tentatives: {e}")
                    break

        else:
            continue
        break

    if not all_data:
        logger.warning(f"  Aucune donnée récupérée pour {symbol} {interval}")
        return pd.DataFrame()

    # Convertir en DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    # Garder colonnes utiles
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # Convertir types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # Trier et dédupliquer
    df = df.sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)

    logger.info(f"  ✅ {len(df):,} bougies téléchargées")

    return df


def save_data(df: pd.DataFrame, symbol: str, interval: str, output_dir: Path) -> Path:
    """Sauvegarde les données en CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Nom du fichier
    symbol_name = SYMBOL_MAP.get(symbol, symbol.replace('USDT', 'USD'))
    filename = f"{symbol_name}_all_{interval}.csv"
    filepath = output_dir / filename

    # Sauvegarder (format compatible avec notre pipeline)
    df.to_csv(filepath, index=False)

    logger.info(f"  💾 Sauvegardé: {filepath}")
    logger.info(f"     Période: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    logger.info(f"     Prix: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    return filepath


def list_existing_files(output_dir: Path):
    """Liste les fichiers de données existants."""
    if not output_dir.exists():
        logger.info("Aucun fichier trouvé.")
        return

    files = sorted(output_dir.glob("*.csv"))
    if not files:
        logger.info("Aucun fichier CSV trouvé.")
        return

    logger.info(f"\nFichiers existants dans {output_dir}:")
    logger.info("-" * 60)

    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        try:
            df = pd.read_csv(f, nrows=1)
            first_date = pd.to_datetime(df['timestamp'].iloc[0])

            df_last = pd.read_csv(f, skiprows=lambda x: x > 0 and x < sum(1 for _ in open(f)) - 1)
            last_date = pd.to_datetime(df_last['timestamp'].iloc[-1])

            logger.info(f"  {f.name:30s} {size_mb:6.2f} MB  {first_date.date()} → {last_date.date()}")
        except Exception:
            logger.info(f"  {f.name:30s} {size_mb:6.2f} MB")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Télécharge les données OHLCV depuis Binance API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Télécharger BTC et ETH en 5min (6 mois par défaut)
  python scripts/download_binance_data.py

  # Télécharger tous les timeframes pour augmentation de données
  python scripts/download_binance_data.py --intervals 1m 5m 15m 30m

  # Télécharger 1 an de données
  python scripts/download_binance_data.py --days 365

  # Ajouter MATIC
  python scripts/download_binance_data.py --symbols BTCUSDT ETHUSDT MATICUSDT
        """
    )

    parser.add_argument(
        '--symbols', '-s',
        nargs='+',
        default=DEFAULT_SYMBOLS,
        help=f'Symboles à télécharger (défaut: {DEFAULT_SYMBOLS})'
    )

    parser.add_argument(
        '--intervals', '-i',
        nargs='+',
        default=DEFAULT_INTERVALS,
        choices=list(INTERVAL_MAP.keys()),
        help=f'Intervalles à télécharger (défaut: {DEFAULT_INTERVALS})'
    )

    parser.add_argument(
        '--days', '-d',
        type=int,
        default=DEFAULT_DAYS,
        help=f'Nombre de jours à télécharger (défaut: {DEFAULT_DAYS})'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=OUTPUT_DIR,
        help=f'Dossier de sortie (défaut: {OUTPUT_DIR})'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='Lister les fichiers existants'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Résoudre le chemin de sortie
    output_dir = args.output
    if not output_dir.is_absolute():
        # Relatif au dossier du script
        output_dir = Path(__file__).parent.parent / args.output.name

    if args.list:
        list_existing_files(output_dir)
        return

    print("=" * 70)
    print("TÉLÉCHARGEMENT DONNÉES BINANCE")
    print("=" * 70)
    print(f"Symboles: {args.symbols}")
    print(f"Intervalles: {args.intervals}")
    print(f"Période: {args.days} jours")
    print(f"Sortie: {output_dir}")
    print("=" * 70)

    downloaded = []

    for symbol in args.symbols:
        for interval in args.intervals:
            print()
            df = download_klines(
                symbol=symbol,
                interval=interval,
                days=args.days
            )

            if not df.empty:
                filepath = save_data(df, symbol, interval, output_dir)
                downloaded.append(filepath)

    print()
    print("=" * 70)
    print(f"✅ TERMINÉ - {len(downloaded)} fichiers téléchargés")
    print("=" * 70)

    for f in downloaded:
        print(f"  - {f}")


if __name__ == '__main__':
    main()
