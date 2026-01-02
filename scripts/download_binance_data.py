#!/usr/bin/env python3
"""
Téléchargement des données OHLCV depuis Binance API.

Configuration:
    Créer un fichier .env ou exporter les variables:
    export BINANCE_API_KEY='votre_api_key'
    export BINANCE_API_SECRET='votre_api_secret'

Usage:
    # Télécharger BTC et ETH en 5min (6 mois)
    python scripts/download_binance_data.py

    # Télécharger plus de cryptos
    python scripts/download_binance_data.py --symbols BTCUSDT ETHUSDT SOLUSDT MATICUSDT

    # Télécharger 1 an de données
    python scripts/download_binance_data.py --days 365

    # Lister les fichiers existants
    python scripts/download_binance_data.py --list
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Ajouter le dossier src au path pour importer les constantes
SCRIPT_DIR = Path(__file__).parent.absolute()
SRC_DIR = SCRIPT_DIR.parent / 'src'
sys.path.insert(0, str(SRC_DIR))

from constants import DATA_TRAD_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration par défaut
DEFAULT_SYMBOLS = ['BTCUSDT', 'ETHUSDT']
DEFAULT_INTERVAL = '5m'
DEFAULT_DAYS = 365  # 1 an par défaut
OUTPUT_DIR = Path(SRC_DIR / DATA_TRAD_DIR).resolve()

# Mapping symbole Binance -> nom fichier
SYMBOL_MAP = {
    'BTCUSDT': 'BTCUSD',
    'ETHUSDT': 'ETHUSD',
    'MATICUSDT': 'MATICUSD',
    'SOLUSDT': 'SOLUSD',
    'BNBUSDT': 'BNBUSD',
    'XRPUSDT': 'XRPUSD',
    'ADAUSDT': 'ADAUSD',
    'DOGEUSDT': 'DOGEUSD',
    'AVAXUSDT': 'AVAXUSD',
    'LINKUSDT': 'LINKUSD',
}


def get_binance_client():
    """Crée et retourne le client Binance avec les clés API."""
    try:
        from binance.client import Client
    except ImportError:
        logger.error("Bibliothèque binance non installée. Exécutez:")
        logger.error("  pip install python-binance")
        raise

    # Récupérer les clés depuis les variables d'environnement
    api_key = os.environ.get('BINANCE_API_KEY', '')
    api_secret = os.environ.get('BINANCE_API_SECRET', '')

    if not api_key or not api_secret:
        logger.warning("⚠️  Clés API non trouvées dans l'environnement.")
        logger.warning("   Définissez BINANCE_API_KEY et BINANCE_API_SECRET")
        logger.warning("   Tentative de connexion sans authentification...")

    client = Client(api_key, api_secret)

    # Utiliser l'API US si nécessaire (décommenter si besoin)
    # client.API_URL = 'https://api.binance.us/api'

    return client


def download_klines(
    client,
    symbol: str = 'BTCUSDT',
    interval: str = '5m',
    start_date: str = None,
    days: int = 365
) -> pd.DataFrame:
    """
    Télécharge les données OHLCV depuis Binance.

    Args:
        client: Client Binance
        symbol: Paire de trading (ex: BTCUSDT)
        interval: Intervalle (1m, 5m, 15m, 30m, 1h, 4h, 1d)
        start_date: Date de début (format: "1 Jan, 2023")
        days: Nombre de jours si start_date non spécifié

    Returns:
        DataFrame avec colonnes: timestamp, open, high, low, close, volume
    """
    from binance.client import Client as BinanceClient

    # Mapper l'intervalle
    interval_map = {
        '1m': BinanceClient.KLINE_INTERVAL_1MINUTE,
        '5m': BinanceClient.KLINE_INTERVAL_5MINUTE,
        '15m': BinanceClient.KLINE_INTERVAL_15MINUTE,
        '30m': BinanceClient.KLINE_INTERVAL_30MINUTE,
        '1h': BinanceClient.KLINE_INTERVAL_1HOUR,
        '4h': BinanceClient.KLINE_INTERVAL_4HOUR,
        '1d': BinanceClient.KLINE_INTERVAL_1DAY,
    }

    kline_interval = interval_map.get(interval, BinanceClient.KLINE_INTERVAL_5MINUTE)

    # Date de début
    if start_date is None:
        start = datetime.now() - timedelta(days=days)
        start_date = start.strftime("%d %b, %Y")

    logger.info(f"Téléchargement {symbol} {interval} depuis {start_date}...")

    try:
        # Télécharger les données
        klines = client.get_historical_klines(
            symbol,
            kline_interval,
            start_date
        )

        if not klines:
            logger.warning(f"  Aucune donnée pour {symbol}")
            return pd.DataFrame()

        # Garder seulement OHLCV (les 6 premières colonnes)
        for line in klines:
            del line[6:]

        # Créer DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        ])

        # Convertir types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        # Trier et dédupliquer
        df = df.sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)

        logger.info(f"  ✅ {len(df):,} bougies téléchargées")
        logger.info(f"     Période: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

        return df

    except Exception as e:
        logger.error(f"  Erreur: {e}")
        return pd.DataFrame()


def save_data(df: pd.DataFrame, symbol: str, interval: str, output_dir: Path) -> Path:
    """Sauvegarde les données en CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Nom du fichier
    symbol_name = SYMBOL_MAP.get(symbol, symbol.replace('USDT', 'USD'))
    filename = f"{symbol_name}_all_{interval}.csv"
    filepath = output_dir / filename

    # Sauvegarder (format compatible avec notre pipeline)
    df.to_csv(filepath, index=False)

    size_mb = filepath.stat().st_size / (1024 * 1024)
    logger.info(f"  💾 Sauvegardé: {filepath} ({size_mb:.2f} MB)")

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

    print(f"\nFichiers existants dans {output_dir}:")
    print("-" * 70)

    total_size = 0
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += size_mb
        try:
            # Lire première et dernière ligne
            df = pd.read_csv(f)
            first_date = pd.to_datetime(df['timestamp'].iloc[0])
            last_date = pd.to_datetime(df['timestamp'].iloc[-1])
            rows = len(df)
            print(f"  {f.name:30s} {size_mb:6.2f} MB  {rows:>8,} rows  {first_date.date()} → {last_date.date()}")
        except Exception:
            print(f"  {f.name:30s} {size_mb:6.2f} MB")

    print("-" * 70)
    print(f"  Total: {total_size:.2f} MB")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Télécharge les données OHLCV depuis Binance API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Télécharger BTC et ETH en 5min (1 an)
  python scripts/download_binance_data.py

  # Télécharger plus de cryptos
  python scripts/download_binance_data.py --symbols BTCUSDT ETHUSDT SOLUSDT MATICUSDT

  # Télécharger 2 ans de données
  python scripts/download_binance_data.py --days 730

  # Depuis une date spécifique
  python scripts/download_binance_data.py --start "1 Jan, 2022"

Configuration des clés API:
  export BINANCE_API_KEY='votre_api_key'
  export BINANCE_API_SECRET='votre_api_secret'
        """
    )

    parser.add_argument(
        '--symbols', '-s',
        nargs='+',
        default=DEFAULT_SYMBOLS,
        help=f'Symboles à télécharger (défaut: {DEFAULT_SYMBOLS})'
    )

    parser.add_argument(
        '--interval', '-i',
        default=DEFAULT_INTERVAL,
        choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
        help=f'Intervalle (défaut: {DEFAULT_INTERVAL})'
    )

    parser.add_argument(
        '--days', '-d',
        type=int,
        default=DEFAULT_DAYS,
        help=f'Nombre de jours à télécharger (défaut: {DEFAULT_DAYS})'
    )

    parser.add_argument(
        '--start',
        type=str,
        default=None,
        help='Date de début (ex: "1 Jan, 2022")'
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
        output_dir = Path(__file__).parent.parent / args.output.name

    if args.list:
        list_existing_files(output_dir)
        return

    print("=" * 70)
    print("TÉLÉCHARGEMENT DONNÉES BINANCE")
    print("=" * 70)
    print(f"Symboles: {args.symbols}")
    print(f"Intervalle: {args.interval}")
    print(f"Période: {args.days} jours" if not args.start else f"Depuis: {args.start}")
    print(f"Sortie: {output_dir}")
    print("=" * 70)

    # Créer client Binance
    client = get_binance_client()

    downloaded = []

    for symbol in args.symbols:
        print()
        df = download_klines(
            client=client,
            symbol=symbol,
            interval=args.interval,
            start_date=args.start,
            days=args.days
        )

        if not df.empty:
            filepath = save_data(df, symbol, args.interval, output_dir)
            downloaded.append((filepath, len(df)))

        # Pause entre les requêtes
        time.sleep(1)

    print()
    print("=" * 70)
    print(f"✅ TERMINÉ - {len(downloaded)} fichiers téléchargés")
    print("=" * 70)

    for filepath, rows in downloaded:
        print(f"  - {filepath.name}: {rows:,} bougies")


if __name__ == '__main__':
    main()
