"""
feature_builder.py — Calcul des 22+ channels OHLCV-only pour le pipeline v5.0.

Lit un CSV BTCUSD 5min, calcule les 4 groupes de features (A bougies / B microstructure
/ C niveaux / D multi-TF) ainsi que les 60+ patterns TA-Lib, et sauvegarde un parquet
dense (~150 MB) servant de cache pour event_detector.py et dataset_builder.py.

Le parquet est intermédiaire (gitignored). Il sera filtré aux events lors de la
construction du dataset PatchTST.

Usage:
    python -m experiments.patchtst_v5.feature_builder \\
        --csv data_trad/BTCUSD_all_5m.csv \\
        --output data/patchtst_v5/features_btc.parquet

Voir STATUS_v5.0.md et experiments/patchtst_v5/README.md pour le contexte.
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import talib

logger = logging.getLogger("patchtst_v5.feature_builder")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

ATR_WINDOW = 14
VOLUME_ZSCORE_WINDOW = 20
YANG_ZHANG_WINDOW = 20
HIGH_LOW_LOOKBACK = 20
POC_LOOKBACK_DAYS = 5
POC_BINS = 50
TREND_1H_BARS = 12   # 12 × 5min = 1h
TREND_4H_BARS = 48   # 48 × 5min = 4h
VOL_1H_LOOKBACK_HOURS = 24

# Group E — Statistical signatures
PERM_ENTROPY_WINDOW = 50
PERM_ENTROPY_M = 3       # Bandt-Pompe order (6 patterns)
HURST_WINDOW = 100
HURST_LAGS = (4, 8, 16, 32)
PACF_WINDOW = 50
PACF_LAG = 5

EPS = 1e-12


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    """Load BTCUSD-style OHLCV CSV with auto-detected separator and column names."""
    df = pd.read_csv(path)
    if len(df.columns) == 1 and ";" in df.columns[0]:
        df = pd.read_csv(path, sep=";")
    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={"date": "timestamp", "time": "timestamp"}, inplace=True)

    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columns missing in CSV: {missing}")
    if "volume" not in df.columns:
        logger.warning("No 'volume' column — defaulting to 1.0 (microstructure features will be unreliable)")
        df["volume"] = 1.0

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Division with denominator clipped to EPS to avoid Inf/NaN."""
    return num / np.where(np.abs(den) < EPS, EPS, den)


def compute_atr(df: pd.DataFrame, window: int = ATR_WINDOW) -> np.ndarray:
    """Wilder's ATR via TA-Lib."""
    return talib.ATR(df["high"].values, df["low"].values, df["close"].values, timeperiod=window)


# ---------------------------------------------------------------------------
# Group A — Continuous candle features (5 channels)
# ---------------------------------------------------------------------------

def compute_candle_continuous(df: pd.DataFrame, atr: np.ndarray) -> pd.DataFrame:
    """5 continuous candle features: body/wicks ratios, close_location_value, gap."""
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    rng = h - l
    body = np.abs(c - o)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    prev_close = np.concatenate([[c[0]], c[:-1]])

    out = pd.DataFrame({
        "body_ratio": _safe_div(body, rng),
        "upper_wick_ratio": _safe_div(upper_wick, rng),
        "lower_wick_ratio": _safe_div(lower_wick, rng),
        "close_location_value": _safe_div(c - l, rng),
        "gap_norm": _safe_div(o - prev_close, atr),
    })
    return out.astype("float32")


# ---------------------------------------------------------------------------
# Group A — TA-Lib pattern recognition (60+ channels)
# ---------------------------------------------------------------------------

def get_talib_pattern_names() -> list[str]:
    """All 60+ candlestick pattern recognition functions exposed by TA-Lib."""
    return list(talib.get_function_groups()["Pattern Recognition"])


def compute_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute every TA-Lib CDL* pattern; values in {-200, -100, 0, +100, +200}."""
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    names = get_talib_pattern_names()
    out = {}
    for name in names:
        fn = getattr(talib, name)
        try:
            arr = fn(o, h, l, c)
        except Exception as exc:  # pragma: no cover
            logger.warning("Pattern %s failed: %s — filling 0", name, exc)
            arr = np.zeros_like(c, dtype=np.int32)
        out[name] = np.nan_to_num(arr, nan=0.0).astype("int16")
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Group B — Microstructure (5 channels)
# ---------------------------------------------------------------------------

def _rogers_satchell(o, h, l, c) -> np.ndarray:
    """Rogers-Satchell per-bar variance (drift-independent)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)


def _yang_zhang(o, h, l, c, window: int) -> np.ndarray:
    """Yang-Zhang volatility (rolling). Uses overnight + open-to-close + Rogers-Satchell."""
    prev_c = np.concatenate([[c[0]], c[:-1]])
    overnight = np.log(_safe_div(o, prev_c))
    open_to_close = np.log(_safe_div(c, o))
    rs = _rogers_satchell(o, h, l, c)

    s_over = pd.Series(overnight ** 2).rolling(window).mean().values
    s_oc = pd.Series(open_to_close ** 2).rolling(window).mean().values
    s_rs = pd.Series(rs).rolling(window).mean().values

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    return np.sqrt(np.clip(s_over + k * s_oc + (1 - k) * s_rs, 0.0, None))


def _corwin_schultz_spread(h: np.ndarray, l: np.ndarray) -> np.ndarray:
    """Corwin-Schultz spread proxy (using 2 consecutive bars).

    Reference: Corwin & Schultz (2012), Journal of Finance.
    """
    h_pair = np.maximum(h[1:], h[:-1])
    l_pair = np.minimum(l[1:], l[:-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.log(h[1:] / l[1:]) ** 2 + np.log(h[:-1] / l[:-1]) ** 2
        gamma = np.log(h_pair / l_pair) ** 2
    denom = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom - np.sqrt(np.clip(gamma / denom, 0.0, None))
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    spread = np.where(np.isfinite(spread), spread, np.nan)
    return np.concatenate([[np.nan], spread])  # align back to original length


def compute_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    """5 microstructure features derived from OHLCV alone."""
    o, h, l, c, v = (df[k].values for k in ("open", "high", "low", "close", "volume"))
    prev_c = np.concatenate([[c[0]], c[:-1]])

    cs_spread = _corwin_schultz_spread(h, l)

    with np.errstate(divide="ignore", invalid="ignore"):
        gk_var = 0.5 * np.log(h / l) ** 2 - (2.0 * np.log(2.0) - 1.0) * np.log(c / o) ** 2
    gk_vol = np.sqrt(np.clip(gk_var, 0.0, None))

    yz_vol = _yang_zhang(o, h, l, c, YANG_ZHANG_WINDOW)

    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.abs(np.log(_safe_div(c, prev_c)))
    dollar_vol = c * v
    amihud = _safe_div(ret, dollar_vol)

    vol_mean = pd.Series(v).rolling(VOLUME_ZSCORE_WINDOW).mean().values
    vol_std = pd.Series(v).rolling(VOLUME_ZSCORE_WINDOW).std().values
    vol_z = _safe_div(v - vol_mean, vol_std)

    out = pd.DataFrame({
        "corwin_schultz_spread": cs_spread,
        "garman_klass_vol": gk_vol,
        "yang_zhang_vol": yz_vol,
        "amihud_illiq": amihud,
        "volume_zscore_20p": vol_z,
    })
    return out.astype("float32")


# ---------------------------------------------------------------------------
# Group C — Levels & context (5 channels)
# ---------------------------------------------------------------------------

def _session_vwap(df: pd.DataFrame, session_freq: str = "1D") -> np.ndarray:
    """VWAP cumulé par session (reset à minuit UTC pour 1D)."""
    typical = (df["high"].values + df["low"].values + df["close"].values) / 3.0
    pv = typical * df["volume"].values
    session_id = df["timestamp"].dt.floor(session_freq)
    df_tmp = pd.DataFrame({"pv": pv, "v": df["volume"].values, "session": session_id})
    cum_pv = df_tmp.groupby("session")["pv"].cumsum().values
    cum_v = df_tmp.groupby("session")["v"].cumsum().values
    return _safe_div(cum_pv, cum_v)


def _camarilla_pivots(df: pd.DataFrame) -> pd.DataFrame:
    """Pivots Camarilla calculés sur le H/L/C de la veille (UTC), forward-fillés au 5min."""
    daily = df.set_index("timestamp")[["high", "low", "close"]].resample("1D").agg(
        {"high": "max", "low": "min", "close": "last"}
    )
    rng = daily["high"] - daily["low"]
    prev_close = daily["close"].shift(1)
    prev_rng = rng.shift(1)
    levels = pd.DataFrame({
        "h1": prev_close + prev_rng * 1.1 / 12,
        "h2": prev_close + prev_rng * 1.1 / 6,
        "h3": prev_close + prev_rng * 1.1 / 4,
        "h4": prev_close + prev_rng * 1.1 / 2,
        "l1": prev_close - prev_rng * 1.1 / 12,
        "l2": prev_close - prev_rng * 1.1 / 6,
        "l3": prev_close - prev_rng * 1.1 / 4,
        "l4": prev_close - prev_rng * 1.1 / 2,
    })
    levels = levels.reindex(df.set_index("timestamp").index, method="ffill").reset_index(drop=True)
    return levels


def _poc_rolling(df: pd.DataFrame, lookback_days: int = POC_LOOKBACK_DAYS,
                 bins: int = POC_BINS) -> np.ndarray:
    """Daily POC (Point of Control) over a rolling N-day window, ffilled to 5min.

    Pour chaque jour D, on regarde la fenêtre [D-N, D-1] et on trouve le bin de prix
    qui a accumulé le plus de volume. POC = midpoint du bin.

    Optimisé : pré-calcule (prices, volumes) par jour, concatène les N précédents.
    """
    typical = (df["high"].values + df["low"].values + df["close"].values) / 3.0
    vol = df["volume"].values
    day_floor = df["timestamp"].dt.floor("1D")
    unique_days = day_floor.drop_duplicates().reset_index(drop=True)

    # Indexe le slicing par jour (positions de début/fin dans typical/vol)
    day_arr = day_floor.values
    day_starts = np.searchsorted(day_arr, unique_days.values, side="left")
    day_ends = np.append(day_starts[1:], len(df))

    poc_per_day = np.full(len(unique_days), np.nan, dtype="float64")
    for i in range(1, len(unique_days)):
        start_idx = max(0, i - lookback_days)
        s = day_starts[start_idx]
        e = day_ends[i - 1] + 1  # inclusif jour i-1
        prices = typical[s:e]
        volumes = vol[s:e]
        if prices.size == 0:
            continue
        lo, hi = prices.min(), prices.max()
        if hi - lo < EPS:
            poc_per_day[i] = lo
            continue
        edges = np.linspace(lo, hi + EPS, bins + 1)
        idx = np.clip(np.digitize(prices, edges) - 1, 0, bins - 1)
        vol_in_bin = np.bincount(idx, weights=volumes, minlength=bins)
        peak = int(np.argmax(vol_in_bin))
        poc_per_day[i] = 0.5 * (edges[peak] + edges[peak + 1])

    poc_series = pd.Series(poc_per_day, index=pd.DatetimeIndex(unique_days))
    return poc_series.reindex(day_floor, method="ffill").values


def compute_levels(df: pd.DataFrame, atr: np.ndarray) -> pd.DataFrame:
    """5 features de niveaux normalisées par ATR."""
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values

    vwap_session = _session_vwap(df, session_freq="1D")
    pivots = _camarilla_pivots(df)
    pivot_arr = pivots[["h1", "h2", "h3", "h4", "l1", "l2", "l3", "l4"]].values
    diff = c[:, None] - pivot_arr
    nearest_idx = np.argmin(np.abs(diff), axis=1)
    nearest_pivot = pivot_arr[np.arange(len(c)), nearest_idx]
    dist_camarilla = c - nearest_pivot

    poc_5d = _poc_rolling(df)

    high_20 = pd.Series(h).rolling(HIGH_LOW_LOOKBACK).max().values
    low_20 = pd.Series(l).rolling(HIGH_LOW_LOOKBACK).min().values

    out = pd.DataFrame({
        "dist_vwap_session_norm": _safe_div(c - vwap_session, atr),
        "dist_camarilla_nearest_norm": _safe_div(dist_camarilla, atr),
        "dist_poc_5d_norm": _safe_div(c - poc_5d, atr),
        "dist_high_20p_norm": _safe_div(c - high_20, atr),
        "dist_low_20p_norm": _safe_div(c - low_20, atr),
    })
    return out.astype("float32")


# ---------------------------------------------------------------------------
# Group D — Multi-TF (4 channels)
# ---------------------------------------------------------------------------

def _rolling_slope(series: np.ndarray, window: int) -> np.ndarray:
    """Slope d'une régression linéaire sur fenêtre glissante (closed-form, vectorisé)."""
    n = window
    x = np.arange(n, dtype="float64")
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()
    s = pd.Series(series)

    def _slope(y):
        return ((x - x_mean) * (y - y.mean())).sum() / denom

    return s.rolling(n).apply(_slope, raw=True).values


def _daily_open(df: pd.DataFrame) -> np.ndarray:
    """First Open of each UTC day, broadcasted back to 5min granularity.

    Différent du VWAP session : capture où le prix est par rapport au point
    d'ouverture quotidien, ancrage utile en intraday scalping.
    """
    daily_floor = df["timestamp"].dt.floor("1D")
    df_tmp = pd.DataFrame({"open": df["open"].values, "day": daily_floor})
    return df_tmp.groupby("day")["open"].transform("first").values


# ---------------------------------------------------------------------------
# Group E — Statistical signatures (3 channels)
# ---------------------------------------------------------------------------

def _perm_entropy_window(window_arr: np.ndarray, m: int, perms_dict: dict) -> float:
    """Bandt-Pompe permutation entropy on one window. Normalized by log(m!)."""
    n = len(window_arr)
    if n < m:
        return np.nan
    n_perms = len(perms_dict)
    counts = np.zeros(n_perms, dtype="int32")
    for i in range(n - m + 1):
        sub = window_arr[i:i + m]
        ranks = tuple(np.argsort(sub).argsort())
        counts[perms_dict[ranks]] += 1
    total = counts.sum()
    if total == 0:
        return np.nan
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log(probs)) / np.log(n_perms))


def _permutation_entropy_rolling(series: np.ndarray, window: int, m: int) -> np.ndarray:
    """Rolling permutation entropy. O(N × W × m) via pandas rolling.apply."""
    perms = list(itertools.permutations(range(m)))
    perms_dict = {p: i for i, p in enumerate(perms)}
    return pd.Series(series).rolling(window).apply(
        lambda x: _perm_entropy_window(x, m, perms_dict), raw=True
    ).values


def _hurst_rs_window(window_arr: np.ndarray, lags: tuple[int, ...]) -> float:
    """Hurst exponent via R/S analysis on one window. Slope of log(R/S) vs log(lag)."""
    n = len(window_arr)
    rs_values: list[tuple[float, float]] = []
    for lag in lags:
        if lag >= n:
            continue
        chunks = n // lag
        if chunks < 1:
            continue
        rs_lag: list[float] = []
        for i in range(chunks):
            chunk = window_arr[i * lag: (i + 1) * lag]
            mean_c = chunk.mean()
            std_c = chunk.std(ddof=1) if len(chunk) > 1 else 0.0
            if std_c < EPS:
                continue
            cumdev = np.cumsum(chunk - mean_c)
            R = cumdev.max() - cumdev.min()
            rs_lag.append(R / std_c)
        if not rs_lag:
            continue
        rs_values.append((np.log(lag), np.log(np.mean(rs_lag))))
    if len(rs_values) < 2:
        return np.nan
    log_lags, log_rs = zip(*rs_values)
    return float(np.polyfit(log_lags, log_rs, 1)[0])


def _hurst_rolling(series: np.ndarray, window: int, lags: tuple[int, ...]) -> np.ndarray:
    """Rolling Hurst exponent via R/S."""
    return pd.Series(series).rolling(window).apply(
        lambda x: _hurst_rs_window(x, lags), raw=True
    ).values


def _pacf_yw_window(window_arr: np.ndarray, lag: int) -> float:
    """Partial autocorrelation at given lag via Yule-Walker (Toeplitz solve)."""
    n = len(window_arr)
    if n <= 2 * lag:
        return np.nan
    arr = window_arr - window_arr.mean()
    var = float(np.dot(arr, arr) / n)
    if var < EPS:
        return np.nan
    gammas = np.empty(lag + 1, dtype="float64")
    gammas[0] = var
    for k in range(1, lag + 1):
        gammas[k] = float(np.dot(arr[:-k], arr[k:]) / n)
    R_mat = np.array([[gammas[abs(i - j)] for j in range(lag)] for i in range(lag)])
    rhs = gammas[1:lag + 1]
    try:
        phi = np.linalg.solve(R_mat, rhs)
    except np.linalg.LinAlgError:
        return np.nan
    return float(phi[-1])


def _pacf_lag_rolling(series: np.ndarray, window: int, lag: int) -> np.ndarray:
    """Rolling PACF at fixed lag via Yule-Walker."""
    return pd.Series(series).rolling(window).apply(
        lambda x: _pacf_yw_window(x, lag), raw=True
    ).values


def compute_statistical_signatures(df: pd.DataFrame) -> pd.DataFrame:
    """3 statistical features capturant la signature stochastique du log-return série.

    - permutation_entropy_50p : Bandt-Pompe entropy (m=3, window=50). Mesure la complexité
      ordinale. Valeurs ~1 = aléatoire, valeurs basses = structurel.
    - hurst_dfa_100p          : Hurst exponent via R/S (window=100). >0.5 = persistance,
      <0.5 = mean-reverting, =0.5 = random walk.
    - pacf_lag5               : Partial autocorrelation lag-5 via Yule-Walker (window=50).
      Capture la dépendance résiduelle à 5 bars après contrôle des lags 1-4.
    """
    c = df["close"].values.astype("float64")
    log_returns = np.diff(np.log(np.maximum(c, EPS)), prepend=np.log(max(c[0], EPS)))

    perm_ent = _permutation_entropy_rolling(log_returns, PERM_ENTROPY_WINDOW, PERM_ENTROPY_M)
    hurst = _hurst_rolling(log_returns, HURST_WINDOW, HURST_LAGS)
    pacf5 = _pacf_lag_rolling(log_returns, PACF_WINDOW, PACF_LAG)

    out = pd.DataFrame({
        "permutation_entropy_50p": perm_ent,
        "hurst_dfa_100p": hurst,
        "pacf_lag5": pacf5,
    })
    return out.astype("float32")


def compute_multitf(df: pd.DataFrame, atr: np.ndarray) -> pd.DataFrame:
    """4 features multi-timeframe (1h et 4h) calculées sur le 5min directement."""
    c = df["close"].values
    v = df["volume"].values

    trend_1h = _rolling_slope(c, TREND_1H_BARS)
    trend_4h = _rolling_slope(c, TREND_4H_BARS)

    # Volume horaire = somme glissante 12 bougies, z-scored sur 24h (288 bougies 5min)
    vol_1h = pd.Series(v).rolling(TREND_1H_BARS).sum().values
    lookback_5m = VOL_1H_LOOKBACK_HOURS * TREND_1H_BARS  # 288
    vol_1h_mean = pd.Series(vol_1h).rolling(lookback_5m).mean().values
    vol_1h_std = pd.Series(vol_1h).rolling(lookback_5m).std().values
    vol_1h_z = _safe_div(vol_1h - vol_1h_mean, vol_1h_std)

    daily_open = _daily_open(df)

    out = pd.DataFrame({
        "trend_1h_slope": trend_1h,
        "trend_4h_slope": trend_4h,
        "vol_1h_zscore": vol_1h_z,
        "dist_open_daily_norm": _safe_div(c - daily_open, atr),
    })
    return out.astype("float32")


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Group I — Pure Indicators (paradigme v5.2 indicators-only)
# ---------------------------------------------------------------------------

def compute_pure_indicators(df: pd.DataFrame, atr: np.ndarray) -> pd.DataFrame:
    """Indicateurs techniques classiques calculés via TA-Lib.

    Paradigme v5.2 indicators-only : on alimente PatchTST avec UNIQUEMENT des
    indicateurs (pas de raw OHLC, pas de bar shape, pas de microstructure).
    Ces 16 channels représentent les abstractions trader classiques.
    """
    o = df["open"].values.astype("float64")
    h = df["high"].values.astype("float64")
    l = df["low"].values.astype("float64")
    c = df["close"].values.astype("float64")
    v = df["volume"].values.astype("float64")

    # Multi-horizon RSI (3)
    rsi_7 = talib.RSI(c, timeperiod=7)
    rsi_14 = talib.RSI(c, timeperiod=14)
    rsi_21 = talib.RSI(c, timeperiod=21)

    # MACD (2 channels: line + signal)
    macd_line, macd_signal, _ = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)

    # CCI (1)
    cci_20 = talib.CCI(h, l, c, timeperiod=20)

    # Stochastic Oscillator (2: %K et %D)
    stoch_k, stoch_d = talib.STOCH(h, l, c, fastk_period=14, slowk_period=3, slowd_period=3)

    # Williams %R (1)
    williams_r = talib.WILLR(h, l, c, timeperiod=14)

    # ADX + DI+ / DI- (3)
    adx_14 = talib.ADX(h, l, c, timeperiod=14)
    di_plus = talib.PLUS_DI(h, l, c, timeperiod=14)
    di_minus = talib.MINUS_DI(h, l, c, timeperiod=14)

    # ATR normalisé (1) — adimensionnel ATR/Close (différent de atr_14 raw du parquet)
    atr_norm = _safe_div(atr, c)

    # Bollinger Bands %B (1) — position relative dans les bandes
    bb_upper, bb_middle, bb_lower = talib.BBANDS(c, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
    bb_pct_b = _safe_div(c - bb_lower, bb_upper - bb_lower)

    # On-Balance Volume slope (1) — OBV est cumulatif, on prend la pente sur 20 bars
    obv = talib.OBV(c, v)
    obv_slope_20 = _rolling_slope(obv, 20)

    # Money Flow Index (1)
    mfi_14 = talib.MFI(h, l, c, v, timeperiod=14)

    out = pd.DataFrame({
        # Momentum multi-horizon (3)
        "rsi_7": rsi_7,
        "rsi_14": rsi_14,
        "rsi_21": rsi_21,
        # MACD (2)
        "macd_line": macd_line,
        "macd_signal_line": macd_signal,
        # CCI (1)
        "cci_20": cci_20,
        # Stoch (2)
        "stoch_k_14": stoch_k,
        "stoch_d_14": stoch_d,
        # Williams (1)
        "williams_r_14": williams_r,
        # Trend strength (3)
        "adx_14": adx_14,
        "di_plus_14": di_plus,
        "di_minus_14": di_minus,
        # Volatility (2)
        "atr_14_norm": atr_norm,
        "bbands_pct_b_20": bb_pct_b,
        # Volume (2)
        "obv_slope_20": obv_slope_20,
        "mfi_14": mfi_14,
    })
    return out.astype("float32")


def build_features(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Compute all 4 groups + 60 patterns and return a single dense DataFrame."""
    n = len(df)
    logger.info("Computing ATR(%d) on %d bars", ATR_WINDOW, n)
    atr = compute_atr(df, ATR_WINDOW)

    t0 = time.time()
    cont = compute_candle_continuous(df, atr)
    logger.info("Group A continuous (5)         done in %.1fs", time.time() - t0)

    t0 = time.time()
    patterns = compute_candle_patterns(df)
    logger.info("Group A patterns TA-Lib (%d)   done in %.1fs", patterns.shape[1], time.time() - t0)

    t0 = time.time()
    micro = compute_microstructure(df)
    logger.info("Group B microstructure (5)     done in %.1fs", time.time() - t0)

    t0 = time.time()
    levels = compute_levels(df, atr)
    logger.info("Group C levels (5)             done in %.1fs", time.time() - t0)

    t0 = time.time()
    mtf = compute_multitf(df, atr)
    logger.info("Group D multi-TF (4)           done in %.1fs", time.time() - t0)

    t0 = time.time()
    stat = compute_statistical_signatures(df)
    logger.info("Group E stat signatures (3)    done in %.1fs", time.time() - t0)

    t0 = time.time()
    indi = compute_pure_indicators(df, atr)
    logger.info("Group I pure indicators (%d)  done in %.1fs", indi.shape[1], time.time() - t0)

    out = pd.DataFrame({
        "timestamp": df["timestamp"].values,
        "asset": np.full(n, asset, dtype="object"),
        "open": df["open"].astype("float32").values,
        "high": df["high"].astype("float32").values,
        "low": df["low"].astype("float32").values,
        "close": df["close"].astype("float32").values,
        "volume": df["volume"].astype("float32").values,
        "atr_14": atr.astype("float32"),
    })
    out = pd.concat([out, cont, patterns, micro, levels, mtf, stat, indi], axis=1)
    return out


def summarize(df: pd.DataFrame) -> None:
    n = len(df)
    n_cols = df.shape[1]
    pattern_cols = [c for c in df.columns if c.startswith("CDL")]
    n_patterns_active = (df[pattern_cols] != 0).any(axis=1).sum()
    nan_pct = df.isna().mean().mean() * 100

    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("Bars             : %d", n)
    logger.info("Period           : %s → %s", df["timestamp"].min(), df["timestamp"].max())
    logger.info("Total columns    : %d (incl. %d TA-Lib patterns)", n_cols, len(pattern_cols))
    logger.info("Bars with ≥1 pattern firing : %d  (%.2f%%)", n_patterns_active, 100 * n_patterns_active / n)
    logger.info("Mean NaN %% across all cols  : %.2f%%", nan_pct)
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=Path("data_trad/BTCUSD_all_5m.csv"),
                   help="Input OHLCV CSV (default: data_trad/BTCUSD_all_5m.csv)")
    p.add_argument("--output", type=Path, default=Path("data/patchtst_v5/features_btc.parquet"),
                   help="Output parquet path (default: data/patchtst_v5/features_btc.parquet)")
    p.add_argument("--asset", type=str, default="BTC", help="Asset name (default: BTC)")
    p.add_argument("--max-bars", type=int, default=None,
                   help="Optional limit on number of bars (head N) — useful for smoke tests")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    logger.info("Loading CSV: %s", args.csv)
    df = load_ohlcv_csv(args.csv)
    if args.max_bars:
        df = df.head(args.max_bars).reset_index(drop=True)
        logger.info("Limited to %d bars (--max-bars)", len(df))
    logger.info("Loaded %d bars (%s → %s)", len(df), df["timestamp"].iloc[0], df["timestamp"].iloc[-1])

    features = build_features(df, args.asset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing parquet: %s", args.output)
    features.to_parquet(args.output, compression="snappy", index=False)
    summarize(features)
    return 0


if __name__ == "__main__":
    sys.exit(main())
