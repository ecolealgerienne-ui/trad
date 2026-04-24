"""
Data loader for slope improvement experiments.

Loads BTC 5min OHLCV, filters to 2022+, computes RSI, splits 50/25/25
chronologically (no shuffle).

REUSES:
    - src.indicators.calculate_rsi        (Wilder's RSI)
    - src.constants.RSI_PERIOD            (project default: 22)

Does NOT modify any existing file; adds the src/ directory to sys.path.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Reuse existing project code without modifying it.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from indicators import calculate_rsi  # noqa: E402
from constants import RSI_PERIOD  # noqa: E402


@dataclass
class SplitData:
    """Container for a single temporal split."""
    name: str
    rsi: np.ndarray           # shape (n,)
    close: np.ndarray         # shape (n,) — kept for downstream sanity / secondary GT
    timestamps: np.ndarray    # shape (n,) datetime64
    idx_start: int            # global index in the full series (before split)
    idx_end: int              # global index in the full series (exclusive)

    @property
    def n(self) -> int:
        return len(self.rsi)


def load_btc_5min(
    csv_path: str | Path = "data_trad/BTCUSD_all_5m.csv",
    start_date: str = "2022-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Load BTC 5min OHLCV, filter chronologically from start_date.

    Column-detection logic mirrors src/prepare_multitf_csv_aqkf.py:load_csv_5min
    (without importing it to avoid pulling its heavy dependencies).
    """
    path = Path(csv_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"CSV introuvable: {path}")

    df = pd.read_csv(path)

    date_col = None
    for col in ["date", "datetime", "time", "timestamp", "Date", "Datetime"]:
        if col in df.columns:
            date_col = col
            break
    if date_col is None:
        raise ValueError(f"Aucune colonne date trouvée dans {path}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df.columns = df.columns.str.lower()

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    df = df.loc[start_date:]
    if end_date is not None:
        df = df.loc[:end_date]
    df = df.dropna(subset=["close"])
    df.index.name = "datetime"
    return df


def make_splits(
    csv_path: str | Path = "data_trad/BTCUSD_all_5m.csv",
    start_date: str = "2022-01-01",
    end_date: str | None = None,
    rsi_period: int = RSI_PERIOD,
    train_frac: float = 0.50,
    val_frac: float = 0.25,
    test_frac: float = 0.25,
) -> Tuple[SplitData, SplitData, SplitData, dict]:
    """
    Load BTC 5min, compute RSI (via src.indicators.calculate_rsi), split
    chronologically.

    Returns (train, val, test, meta). Splits are disjoint and contiguous.
    RSI is computed on the FULL filtered series before splitting to avoid
    edge NaNs at boundaries. RSI is causal: RSI[t] depends only on close[<=t].
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"

    df = load_btc_5min(csv_path, start_date, end_date)
    close = df["close"].to_numpy(dtype=np.float64)
    timestamps = df.index.to_numpy()

    rsi = calculate_rsi(close, period=rsi_period)

    # Drop initial NaNs from RSI warmup
    finite_mask = np.isfinite(rsi)
    if not finite_mask.any():
        raise RuntimeError("RSI entièrement NaN — vérifier la période/le dataset")
    first_valid = int(np.argmax(finite_mask))
    rsi = rsi[first_valid:]
    close = close[first_valid:]
    timestamps = timestamps[first_valid:]

    n = len(rsi)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    def _slice(name: str, a: int, b: int) -> SplitData:
        return SplitData(
            name=name,
            rsi=rsi[a:b].copy(),
            close=close[a:b].copy(),
            timestamps=timestamps[a:b].copy(),
            idx_start=a,
            idx_end=b,
        )

    train = _slice("train", 0, n_train)
    val = _slice("val", n_train, n_train + n_val)
    test = _slice("test", n_train + n_val, n)

    meta = {
        "csv_path": str(csv_path),
        "start_date": start_date,
        "end_date": end_date,
        "rsi_period": rsi_period,
        "n_total": n,
        "n_train": train.n,
        "n_val": val.n,
        "n_test": test.n,
        "train_start": str(train.timestamps[0]),
        "train_end": str(train.timestamps[-1]),
        "val_start": str(val.timestamps[0]),
        "val_end": str(val.timestamps[-1]),
        "test_start": str(test.timestamps[0]),
        "test_end": str(test.timestamps[-1]),
        "rsi_warmup_dropped": first_valid,
    }
    return train, val, test, meta


if __name__ == "__main__":
    import json
    train, val, test, meta = make_splits()
    print(json.dumps(meta, indent=2, default=str))
    print(f"train RSI[first5] = {train.rsi[:5]}")
    print(f"train RSI[last5]  = {train.rsi[-5:]}")
