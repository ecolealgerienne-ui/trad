"""
Feature Engineering Module for LLM-assisted Crypto Trading.

Produces a JSON-serializable dict of features for each 15-min decision cycle.
Anti-look-ahead is enforced via strict candle filtering:
    bougie utilisable <=> bougie.timestamp + tf_duration <= as_of

All timestamps are UTC naive. Indicators are precomputed at load time
(all are causal) for backtest performance (~40k calls).
"""

import json
import math
import numpy as np
import pandas as pd
import talib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"]
TIMEFRAMES = {"15m": 15, "1h": 60, "4h": 240}

_SYMBOL_TO_FILE = {
    "BTCUSDT": "BTCUSD",
    "ETHUSDT": "ETHUSD",
    "SOLUSDT": "SOLUSD",
    "XRPUSDT": "XRPUSD",
    "BNBUSDT": "BNBUSD",
}

PSYCHO_LEVELS = {
    "BTCUSDT": 1000,
    "ETHUSDT": 100,
    "SOLUSDT": 10,
    "XRPUSDT": 0.05,
    "BNBUSDT": 50,
}

SESSION_START_HOUR = 8  # UTC
_EPOCH = pd.Timestamp("2025-02-01")  # Reference for cycle_index

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_all_data(data_dir: str) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Load all CSV files into memory. Called once at backtest start.

    Returns dict keyed by (symbol, tf_label), e.g. ("BTCUSDT", "15m").
    DataFrames have DatetimeIndex (UTC naive), columns: open, high, low, close, volume.
    Indicators are precomputed on the full dataset for performance.
    """
    data_dir = Path(data_dir)
    data: Dict[Tuple[str, str], pd.DataFrame] = {}

    for symbol in SYMBOLS:
        file_stem = _SYMBOL_TO_FILE[symbol]
        for tf_label in TIMEFRAMES:
            filename = f"{file_stem}_all_{tf_label}.csv"
            filepath = data_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Missing data file: {filepath}")

            df = _load_csv(filepath)
            df = _precompute_indicators(df, tf_label)
            data[(symbol, tf_label)] = df

    return data


def _load_csv(filepath: Path) -> pd.DataFrame:
    """Load a single CSV, normalize columns, set DatetimeIndex (UTC naive)."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip()

    # Find timestamp column
    ts_col = None
    for candidate in ["timestamp", "date", "datetime", "time"]:
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        raise ValueError(f"No timestamp column found in {filepath}")

    df[ts_col] = pd.to_datetime(df[ts_col], utc=False)
    if df[ts_col].dt.tz is not None:
        df[ts_col] = df[ts_col].dt.tz_localize(None)

    df = df.set_index(ts_col).sort_index()
    df.index.name = "timestamp"

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {filepath}")

    return df[["open", "high", "low", "close", "volume"]].copy()


# ---------------------------------------------------------------------------
# Anti-Look-Ahead Filtering  (PRIORITY #1)
# ---------------------------------------------------------------------------


def filter_closed_candles(
    df: pd.DataFrame, tf_minutes: int, as_of: pd.Timestamp
) -> pd.DataFrame:
    """Return only candles whose closing time <= as_of.

    Binance convention: candle timestamp = OPEN time.
    Closing time = timestamp + tf_duration.
    Usable iff timestamp + tf_duration <= as_of  =>  timestamp <= as_of - tf_duration.

    The boundary is INCLUSIVE: a candle whose close equals as_of exactly IS included.
    """
    if tf_minutes not in TIMEFRAMES.values():
        raise ValueError(f"Unsupported timeframe: {tf_minutes}m")

    cutoff = as_of - pd.Timedelta(minutes=tf_minutes)
    result = df[df.index <= cutoff]

    if result.empty:
        first_ts = df.index[0] if len(df) > 0 else "N/A"
        first_close = (
            first_ts + pd.Timedelta(minutes=tf_minutes) if first_ts != "N/A" else "N/A"
        )
        raise ValueError(
            f"No closed candles at as_of={as_of} (tf={tf_minutes}m). "
            f"First candle opens at {first_ts}, closes at {first_close}."
        )

    return result


# ---------------------------------------------------------------------------
# Indicator Precomputation (all causal — safe to run on full dataset)
# ---------------------------------------------------------------------------


def _precompute_indicators(df: pd.DataFrame, tf_label: str) -> pd.DataFrame:
    """Add indicator columns to DataFrame. Called once per (symbol, tf) at load."""
    df = df.copy()
    if tf_label == "15m":
        _precompute_15m(df)
    elif tf_label == "1h":
        _precompute_1h(df)
    elif tf_label == "4h":
        _precompute_4h(df)
    return df


def _precompute_15m(df: pd.DataFrame) -> None:
    """15m: EMA20, RSI14, ATR14 (+ratio), BB(20,2), VWAP session, vol_rel."""
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)

    # EMA 20
    ema20 = pd.Series(talib.EMA(close, timeperiod=20), index=df.index)
    df["ema20"] = ema20
    df["ema20_slope_pct"] = ema20.pct_change() * 100
    df["ema20_dist_pct"] = (df["close"] - ema20) / ema20 * 100

    # RSI 14
    df["rsi14"] = talib.RSI(close, timeperiod=14)

    # ATR 14
    atr = pd.Series(talib.ATR(high, low, close, timeperiod=14), index=df.index)
    df["atr14"] = atr
    df["atr14_pct"] = atr / df["close"] * 100
    atr_avg50 = atr.rolling(50, min_periods=1).mean()
    df["atr14_avg50"] = atr_avg50
    df["atr14_ratio"] = atr / atr_avg50

    # Bollinger Bands 20/2 — position in band (0-1)
    bb_upper, bb_mid, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    bb_range = bb_upper - bb_lower
    df["bb_position"] = np.where(bb_range > 0, (close - bb_lower) / bb_range, 0.5)

    # Volume relative vs 20-period mean
    vol_s = df["volume"]
    vol_ma20 = vol_s.rolling(20, min_periods=1).mean()
    df["vol_rel"] = vol_s / vol_ma20
    df["vol_ma20"] = vol_ma20

    # Session VWAP (resets at SESSION_START_HOUR UTC each day)
    _precompute_session_vwap(df)


def _precompute_session_vwap(df: pd.DataFrame) -> None:
    """VWAP that resets at session start hour each day."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical_price * df["volume"]

    # Session ID = date shifted by session start hour
    shifted = df.index - pd.Timedelta(hours=SESSION_START_HOUR)
    session_id = shifted.date

    cum_tp_vol = tp_vol.groupby(session_id).cumsum()
    cum_vol = df["volume"].groupby(session_id).cumsum()

    df["vwap"] = np.where(cum_vol > 0, cum_tp_vol / cum_vol, np.nan)
    df["vwap_dist_pct"] = np.where(
        df["vwap"] > 0, (df["close"] - df["vwap"]) / df["vwap"] * 100, np.nan
    )


def _precompute_1h(df: pd.DataFrame) -> None:
    """1h: EMA50, RSI14, ADX14, vol_rel."""
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)

    ema50 = pd.Series(talib.EMA(close, timeperiod=50), index=df.index)
    df["ema50"] = ema50
    df["ema50_slope_pct"] = ema50.pct_change() * 100
    df["ema50_dist_pct"] = (df["close"] - ema50) / ema50 * 100

    df["rsi14"] = talib.RSI(close, timeperiod=14)
    df["adx14"] = talib.ADX(high, low, close, timeperiod=14)

    vol_s = df["volume"]
    vol_ma20 = vol_s.rolling(20, min_periods=1).mean()
    df["vol_rel"] = vol_s / vol_ma20


def _precompute_4h(df: pd.DataFrame) -> None:
    """4h: EMA200 only."""
    close = df["close"].values.astype(float)
    ema200 = pd.Series(talib.EMA(close, timeperiod=200), index=df.index)
    df["ema200"] = ema200
    df["ema200_slope_pct"] = ema200.pct_change() * 100
    df["ema200_dist_pct"] = (df["close"] - ema200) / ema200 * 100


# ---------------------------------------------------------------------------
# Indicator Extraction (read precomputed values from last closed candle)
# ---------------------------------------------------------------------------


def compute_indicators(df_closed: pd.DataFrame, tf_label: str) -> Dict[str, Any]:
    """Read precomputed indicator values from the last closed candle.

    Returns None for indicators with insufficient history (NaN in source).
    """
    if df_closed.empty:
        return {}

    last = df_closed.iloc[-1]

    def _safe(col: str) -> Any:
        val = last.get(col)
        if val is None or (isinstance(val, float) and (pd.isna(val) or np.isinf(val))):
            return None
        return round(float(val), 6)

    if tf_label == "15m":
        return {
            "ema20_slope_pct": _safe("ema20_slope_pct"),
            "ema20_dist_pct": _safe("ema20_dist_pct"),
            "rsi14": _safe("rsi14"),
            "atr14_abs": _safe("atr14"),
            "atr14_pct": _safe("atr14_pct"),
            "atr14_ratio_vs_avg50": _safe("atr14_ratio"),
            "bb_position": _safe("bb_position"),
            "vwap_dist_pct": _safe("vwap_dist_pct"),
            "vol_rel": _safe("vol_rel"),
        }
    elif tf_label == "1h":
        return {
            "ema50_slope_pct": _safe("ema50_slope_pct"),
            "ema50_dist_pct": _safe("ema50_dist_pct"),
            "rsi14": _safe("rsi14"),
            "adx14": _safe("adx14"),
            "vol_rel": _safe("vol_rel"),
        }
    elif tf_label == "4h":
        return {
            "ema200_slope_pct": _safe("ema200_slope_pct"),
            "ema200_dist_pct": _safe("ema200_dist_pct"),
        }
    return {}


# ---------------------------------------------------------------------------
# Current Bar Reconstruction
# ---------------------------------------------------------------------------


def build_current_bar(
    df_15m_closed: pd.DataFrame, tf_minutes: int, as_of: pd.Timestamp
) -> Optional[Dict[str, float]]:
    """Reconstruct the in-progress higher-TF candle from closed 15m candles.

    Only for tf > 15m (1h, 4h). Returns None if no closed 15m bars fall
    within the current higher-TF bar window, or if tf <= 15m.
    The result is ISOLATED — never injected into indicator DataFrames.
    """
    if tf_minutes <= 15:
        return None

    # Opening time of the current higher-TF bar
    tf_td = pd.Timedelta(minutes=tf_minutes)
    bar_open = as_of.floor(tf_td)

    # Select closed 15m candles within this bar's window
    mask = (df_15m_closed.index >= bar_open) & (df_15m_closed.index < bar_open + tf_td)
    bars = df_15m_closed[mask]

    if bars.empty:
        return None

    total_expected = tf_minutes // 15
    return {
        "o": float(bars.iloc[0]["open"]),
        "h": float(bars["high"].max()),
        "l": float(bars["low"].min()),
        "c": float(bars.iloc[-1]["close"]),
        "v": float(bars["volume"].sum()),
        "progress_pct": round(len(bars) / total_expected * 100, 1),
    }


def build_current_bar_features(
    current_bar: Optional[Dict[str, float]],
    atr_15m: Optional[float],
    avg_volume_15m: Optional[float],
    tf_minutes: int,
) -> Optional[Dict[str, Any]]:
    """Derive features from a reconstructed current bar."""
    if current_bar is None:
        return None

    o, h, l, c = current_bar["o"], current_bar["h"], current_bar["l"], current_bar["c"]
    v, progress = current_bar["v"], current_bar["progress_pct"]

    move_pct = (c - o) / o * 100 if o != 0 else 0.0

    range_vs_atr = None
    if atr_15m and atr_15m > 0:
        range_vs_atr = round((h - l) / atr_15m, 4)

    vol_vs_expected = None
    if avg_volume_15m and avg_volume_15m > 0 and progress > 0:
        n_bars = progress / 100 * (tf_minutes / 15)
        expected = n_bars * avg_volume_15m
        if expected > 0:
            vol_vs_expected = round(v / expected, 4)

    return {
        "progress_pct": round(progress, 1),
        "move_pct": round(move_pct, 4),
        "range_vs_atr": range_vs_atr,
        "vol_vs_expected": vol_vs_expected,
    }


# ---------------------------------------------------------------------------
# Support / Resistance Detection
# ---------------------------------------------------------------------------


def detect_levels(
    df_15m_closed: pd.DataFrame, current_price: float, symbol: str
) -> Dict[str, Any]:
    """Nearest support & resistance from swing pivots + psychological levels.

    Swing pivots: N=5 on last 50 closed 15m candles.
    Psycho levels: fixed step per symbol.
    Returns the closest level in each direction with type and dist_pct.
    """
    swing_lows, swing_highs = _find_swing_pivots(df_15m_closed, n=5, lookback=50)

    # Nearest swing below / above current price
    swing_support = None
    for p in sorted(swing_lows, reverse=True):
        if p < current_price:
            swing_support = p
            break

    swing_resistance = None
    for p in sorted(swing_highs):
        if p > current_price:
            swing_resistance = p
            break

    # Psychological levels
    step = PSYCHO_LEVELS.get(symbol, 100)
    n_below = math.floor(current_price / step)
    psycho_support = n_below * step
    psycho_resistance = (n_below + 1) * step
    if psycho_support >= current_price:
        psycho_support = (n_below - 1) * step

    support = _pick_closest(
        current_price, swing_support, "swing", psycho_support, "psycho", "below"
    )
    resistance = _pick_closest(
        current_price, swing_resistance, "swing", psycho_resistance, "psycho", "above"
    )

    return {"support": support, "resistance": resistance}


def _find_swing_pivots(
    df: pd.DataFrame, n: int = 5, lookback: int = 50
) -> Tuple[List[float], List[float]]:
    """Find swing low/high prices in the last `lookback` candles."""
    window = df.tail(lookback)
    if len(window) < 2 * n + 1:
        return [], []

    highs = window["high"].values
    lows = window["low"].values
    swing_highs: List[float] = []
    swing_lows: List[float] = []

    for i in range(n, len(window) - n):
        if highs[i] == max(highs[i - n : i + n + 1]):
            swing_highs.append(float(highs[i]))
        if lows[i] == min(lows[i - n : i + n + 1]):
            swing_lows.append(float(lows[i]))

    return swing_lows, swing_highs


def _pick_closest(
    current_price: float,
    swing_level: Optional[float],
    swing_type: str,
    psycho_level: float,
    psycho_type: str,
    direction: str,
) -> Dict[str, Any]:
    """Pick the closest level between swing and psycho in a given direction."""
    candidates: List[Tuple[float, float, str]] = []

    if swing_level is not None:
        candidates.append((abs(swing_level - current_price), swing_level, swing_type))

    if direction == "below" and psycho_level < current_price:
        candidates.append((abs(psycho_level - current_price), psycho_level, psycho_type))
    elif direction == "above" and psycho_level > current_price:
        candidates.append((abs(psycho_level - current_price), psycho_level, psycho_type))

    if not candidates:
        return {"price": None, "dist_pct": None, "type": None}

    candidates.sort(key=lambda x: x[0])
    _, price, level_type = candidates[0]
    dist_pct = (price - current_price) / current_price * 100

    return {
        "price": round(price, 6),
        "dist_pct": round(dist_pct, 4),
        "type": level_type,
    }


# ---------------------------------------------------------------------------
# Session Context
# ---------------------------------------------------------------------------


def compute_session_context(
    df_15m_closed: pd.DataFrame,
    as_of: pd.Timestamp,
    session_start_hour: int = SESSION_START_HOUR,
) -> Dict[str, Any]:
    """Intraday session stats: chg_pct, high, low, range_position."""
    session_start = as_of.normalize() + pd.Timedelta(hours=session_start_hour)
    if as_of < session_start:
        session_start -= pd.Timedelta(days=1)

    candles = df_15m_closed[df_15m_closed.index >= session_start]
    if candles.empty:
        return {"chg_pct": None, "high": None, "low": None, "range_position": None}

    s_open = float(candles.iloc[0]["open"])
    s_close = float(candles.iloc[-1]["close"])
    s_high = float(candles["high"].max())
    s_low = float(candles["low"].min())

    chg_pct = (s_close - s_open) / s_open * 100
    s_range = s_high - s_low
    range_pos = (s_close - s_low) / s_range if s_range > 0 else 0.5

    return {
        "chg_pct": round(chg_pct, 4),
        "high": round(s_high, 6),
        "low": round(s_low, 6),
        "range_position": round(range_pos, 4),
    }


# ---------------------------------------------------------------------------
# BTC Correlation
# ---------------------------------------------------------------------------


def compute_correlation_btc(
    df_crypto_15m_closed: pd.DataFrame,
    df_btc_15m_closed: pd.DataFrame,
    window_hours: int = 24,
) -> Optional[float]:
    """Log-return Pearson correlation with BTC over the last N hours.

    Indexes are aligned via inner join before computation.
    """
    n_periods = window_hours * 4  # 4 bars per hour at 15m

    crypto_close = df_crypto_15m_closed["close"].tail(n_periods + 1)
    btc_close = df_btc_15m_closed["close"].tail(n_periods + 1)

    crypto_ret = np.log(crypto_close / crypto_close.shift(1)).dropna()
    btc_ret = np.log(btc_close / btc_close.shift(1)).dropna()

    aligned = pd.DataFrame({"crypto": crypto_ret, "btc": btc_ret}).dropna()

    if len(aligned) < 10:
        return None

    corr = float(aligned["crypto"].corr(aligned["btc"]))
    return round(corr, 4) if not np.isnan(corr) else None


# ---------------------------------------------------------------------------
# Anonymization
# ---------------------------------------------------------------------------


def anonymize_and_format(
    features: Dict[str, Any], mapping: Dict[str, str]
) -> Dict[str, Any]:
    """Replace real symbol names with anonymous IDs (ASSET_A, ASSET_B, ...).

    Mapping is parameterizable and randomizable between runs.
    No absolute timestamps remain — only cycle_index and minutes_to_close.
    """
    result = json.loads(json.dumps(features))  # deep copy
    for asset in result.get("assets", []):
        real_symbol = asset.pop("_symbol", None)
        if real_symbol and real_symbol in mapping:
            asset["id"] = mapping[real_symbol]
        elif real_symbol:
            asset["id"] = real_symbol  # fallback
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _price_change(
    df_closed: pd.DataFrame, as_of: pd.Timestamp, hours: float
) -> Optional[float]:
    """% change of close price over the last N hours."""
    if df_closed.empty:
        return None
    current = float(df_closed.iloc[-1]["close"])
    target_time = as_of - pd.Timedelta(hours=hours)
    past = df_closed[df_closed.index <= target_time]
    if past.empty:
        return None
    past_price = float(past.iloc[-1]["close"])
    if past_price == 0:
        return None
    return round((current - past_price) / past_price * 100, 4)


def compute_features(
    data_sources: Dict[Tuple[str, str], pd.DataFrame],
    as_of: pd.Timestamp,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Main orchestrator. Compute all features at timestamp as_of.

    Args:
        data_sources: Dict from load_all_data().
        as_of: Current timestamp (UTC naive).
        context: External context (never computed here):
            - positions_open_count (int)
            - total_exposure_pct (float)
            - btc_dominance (dict with value, chg_24h_pct) — optional
            - funding_rates (dict symbol->rate) — optional

    Returns:
        JSON-serializable dict of features (with real symbols in _symbol field).
        Call anonymize_and_format() afterward to replace symbols.
    """
    if as_of.tzinfo is not None:
        raise ValueError("as_of must be UTC naive (no timezone)")

    # --- BTC baseline (needed for correlation + global section) ---
    btc_15m_key = ("BTCUSDT", "15m")
    if btc_15m_key not in data_sources:
        raise ValueError("BTCUSDT 15m data required in data_sources")

    df_btc_15m = filter_closed_candles(data_sources[btc_15m_key], 15, as_of)
    btc_price = float(df_btc_15m.iloc[-1]["close"])

    # --- Global section ---
    close_time = as_of.normalize() + pd.Timedelta(hours=22)
    if as_of >= close_time:
        minutes_to_close = 0
    else:
        minutes_to_close = int((close_time - as_of).total_seconds() / 60)

    cycle_index = int((as_of - _EPOCH).total_seconds() / (15 * 60))

    btc_dom = context.get("btc_dominance", {"value": None, "chg_24h_pct": None})
    funding_rates = context.get("funding_rates", {})

    global_section = {
        "btc": {
            "price": btc_price,
            "chg_1h_pct": _price_change(df_btc_15m, as_of, 1),
            "chg_24h_pct": _price_change(df_btc_15m, as_of, 24),
        },
        "btc_dominance": btc_dom,
        "time": {"minutes_to_close": minutes_to_close},
        "portfolio": {
            "open_positions": context.get("positions_open_count", 0),
            "total_exposure_pct": context.get("total_exposure_pct", 0.0),
        },
    }

    # --- Per-asset features ---
    assets = []
    for symbol in SYMBOLS:
        # Validate data exists
        for tf_label in TIMEFRAMES:
            key = (symbol, tf_label)
            if key not in data_sources:
                raise ValueError(f"Missing data for {symbol} {tf_label}")

        # Filter closed candles per timeframe
        df_15m = filter_closed_candles(data_sources[(symbol, "15m")], 15, as_of)
        df_1h = filter_closed_candles(data_sources[(symbol, "1h")], 60, as_of)
        df_4h = filter_closed_candles(data_sources[(symbol, "4h")], 240, as_of)

        current_price = float(df_15m.iloc[-1]["close"])

        # Indicators per TF
        ind_15m = compute_indicators(df_15m, "15m")
        ind_1h = compute_indicators(df_1h, "1h")
        ind_4h = compute_indicators(df_4h, "4h")

        # Current bars (1h and 4h only, reconstructed from closed 15m)
        cb_1h = build_current_bar(df_15m, 60, as_of)
        cb_4h = build_current_bar(df_15m, 240, as_of)

        atr_15m_abs = ind_15m.get("atr14_abs")
        avg_vol_15m = None
        if "vol_ma20" in df_15m.columns:
            v = df_15m.iloc[-1].get("vol_ma20")
            if v is not None and not pd.isna(v):
                avg_vol_15m = float(v)

        cb_1h_feat = build_current_bar_features(cb_1h, atr_15m_abs, avg_vol_15m, 60)
        cb_4h_feat = build_current_bar_features(cb_4h, atr_15m_abs, avg_vol_15m, 240)

        # Correlation with BTC (skip for BTC itself)
        corr_btc = None
        if symbol != "BTCUSDT":
            corr_btc = compute_correlation_btc(df_15m, df_btc_15m, window_hours=24)

        asset_dict = {
            "_symbol": symbol,
            "price": current_price,
            "session": compute_session_context(df_15m, as_of),
            "trend": {
                "ema20_15m": {
                    "slope_pct": ind_15m.get("ema20_slope_pct"),
                    "dist_pct": ind_15m.get("ema20_dist_pct"),
                },
                "ema50_1h": {
                    "slope_pct": ind_1h.get("ema50_slope_pct"),
                    "dist_pct": ind_1h.get("ema50_dist_pct"),
                },
                "ema200_4h": {
                    "slope_pct": ind_4h.get("ema200_slope_pct"),
                    "dist_pct": ind_4h.get("ema200_dist_pct"),
                },
            },
            "regime": {"adx_1h": ind_1h.get("adx14")},
            "momentum": {
                "rsi_15m": ind_15m.get("rsi14"),
                "rsi_1h": ind_1h.get("rsi14"),
            },
            "volatility": {
                "atr_15m_abs": ind_15m.get("atr14_abs"),
                "atr_15m_pct": ind_15m.get("atr14_pct"),
                "atr_ratio_vs_avg50": ind_15m.get("atr14_ratio_vs_avg50"),
                "bb_position_15m": ind_15m.get("bb_position"),
            },
            "volume": {
                "vol_rel_15m": ind_15m.get("vol_rel"),
                "vol_rel_1h": ind_1h.get("vol_rel"),
                "vwap_dist_pct": ind_15m.get("vwap_dist_pct"),
            },
            "structure": detect_levels(df_15m, current_price, symbol),
            "correlation": {"corr_btc_24h": corr_btc},
            "sentiment": {
                "funding_rate_perp_8h": funding_rates.get(symbol),
            },
            "current_bar": {
                "tf_1h": cb_1h_feat,
                "tf_4h": cb_4h_feat,
            },
        }
        assets.append(asset_dict)

    return {
        "cycle_index": cycle_index,
        "timeframe_reference": "15m",
        "global": global_section,
        "assets": assets,
    }


# ---------------------------------------------------------------------------
# __main__ — demo usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "src/data_trad"

    print(f"Loading data from {data_dir}/ ...")
    data = load_all_data(data_dir)
    print(f"Loaded {len(data)} datasets.")

    # Pick a T in the middle of the data
    btc_15m = data[("BTCUSDT", "15m")]
    mid_idx = len(btc_15m) // 2
    as_of = btc_15m.index[mid_idx]
    # Align to 15m boundary
    as_of = as_of.floor("15min")
    print(f"Computing features at as_of = {as_of} ...")

    context = {
        "positions_open_count": 2,
        "total_exposure_pct": 4.0,
        "btc_dominance": {"value": 54.3, "chg_24h_pct": -0.2},
        "funding_rates": {
            "BTCUSDT": 0.0001,
            "ETHUSDT": 0.00015,
            "SOLUSDT": 0.0002,
            "XRPUSDT": -0.0001,
            "BNBUSDT": 0.00005,
        },
    }

    features = compute_features(data, as_of, context)

    mapping = {
        "BTCUSDT": "ASSET_A",
        "ETHUSDT": "ASSET_B",
        "SOLUSDT": "ASSET_C",
        "XRPUSDT": "ASSET_D",
        "BNBUSDT": "ASSET_E",
    }
    anon = anonymize_and_format(features, mapping)

    print(json.dumps(anon, indent=2, default=str))
