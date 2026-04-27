"""
make_30min_csv.py — Convert a 5min OHLCV CSV to 30min via resampling.

Reuses the resample logic from src/signal_processing/core.py so the conversion
is identical to other parts of the codebase. Produces a 30min OHLCV CSV that
can be fed directly to feature_builder.py (same schema as the 5min input).

Aggregation rules:
  open   = first
  high   = max
  low    = min
  close  = last
  volume = sum

Usage:
    python -m experiments.patchtst_v5.make_30min_csv \\
        --input data_trad/BTCUSD_all_5m.csv \\
        --output data_trad/BTCUSD_all_30m.csv \\
        --tf-minutes 30
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger("patchtst_v5.make_30min_csv")


def resample_ohlcv(df: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
    """Identical aggregation to src/signal_processing/core.py:resample_ohlcv."""
    return df.resample(f"{tf_minutes}min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path,
                   default=Path("data_trad/BTCUSD_all_5m.csv"),
                   help="Input 5min OHLCV CSV")
    p.add_argument("--output", type=Path,
                   default=Path("data_trad/BTCUSD_all_30m.csv"),
                   help="Output resampled CSV")
    p.add_argument("--tf-minutes", type=int, default=30,
                   help="Target timeframe in minutes (default: 30)")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    logger.info("Loading: %s", args.input)
    df = pd.read_csv(args.input)
    if len(df.columns) == 1 and ";" in df.columns[0]:
        df = pd.read_csv(args.input, sep=";")
    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={"date": "timestamp", "time": "timestamp"}, inplace=True)

    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Columns missing in CSV: {missing}")
    if "volume" not in df.columns:
        logger.warning("No 'volume' column — defaulting to 1.0")
        df["volume"] = 1.0

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df = df.dropna(subset=["open", "high", "low", "close"])

    n_in = len(df)
    logger.info("Input: %d bars (5min), span: %s → %s",
                n_in, df.index.min(), df.index.max())

    df_resampled = resample_ohlcv(df, args.tf_minutes)
    df_resampled = df_resampled.reset_index()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_resampled.to_csv(args.output, index=False)

    n_out = len(df_resampled)
    logger.info("Output: %d bars (%dmin) saved to %s",
                n_out, args.tf_minutes, args.output)
    logger.info("Reduction factor: %.2fx (%d → %d)",
                n_in / n_out, n_in, n_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
