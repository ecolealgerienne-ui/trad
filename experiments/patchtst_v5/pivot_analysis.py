"""
pivot_analysis.py — Étude statistique exploratoire des niveaux Camarilla.

PAS DE MODÈLE ML. Pure analyse empirique sur 8 ans BTC 5min pour répondre:

1. Quels niveaux Camarilla (H1-H4, L1-L4) ont un edge directionnel mesurable ?
2. À quel horizon (30min, 1h, 2h, 4h) le signal est-il fort/faible ?
3. La session (Asie/Europe/US) impacte-t-elle le signal ?
4. Une stratégie naïve "bounce" est-elle profitable par niveau ?
5. La distribution des returns post-touch est-elle asymétrique ?

Output: pivot_analysis.json + tableau console détaillé.

Usage:
    python -m experiments.patchtst_v5.pivot_analysis \\
        --csv data_trad/BTCUSD_all_5m.csv \\
        --output data/patchtst_v5/pivot_analysis.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import talib

logger = logging.getLogger("patchtst_v5.pivot_analysis")

PIVOT_LEVELS = ["h1", "h2", "h3", "h4", "l1", "l2", "l3", "l4"]
HORIZONS_BARS = [6, 12, 24, 48, 96]   # 30min, 1h, 2h, 4h, 8h
TOUCH_THRESHOLD_ATR = 0.10              # touch si |close - level| < 10% × ATR
NAIVE_TP_ATR = 1.0
NAIVE_SL_ATR = 1.0
NAIVE_TIME_BARRIER = 24
FEES_PCT = 0.04  # maker round-trip = 0.08%


# ---------------------------------------------------------------------------
# Loading & Camarilla
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df.columns) == 1 and ";" in df.columns[0]:
        df = pd.read_csv(path, sep=";")
    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={"date": "timestamp", "time": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    return df


def compute_camarilla_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Camarilla pivots from prev day, ffill au 5min (causal)."""
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
    return levels.reindex(df["timestamp"], method="ffill").reset_index(drop=True)


def session_label(hour_utc: int) -> str:
    if 0 <= hour_utc < 8:
        return "asia"
    elif 8 <= hour_utc < 16:
        return "europe"
    else:
        return "us"


# ---------------------------------------------------------------------------
# Touch detection
# ---------------------------------------------------------------------------

def detect_touches(close: np.ndarray, level: np.ndarray, atr: np.ndarray,
                   threshold: float, min_gap_bars: int = 12) -> np.ndarray:
    """First-touch indicator: True quand close passe pour la 1re fois dans
    la zone [level ± threshold*ATR]. min_gap_bars empêche de compter un touch
    multiple fois dans la même séquence."""
    in_band = np.abs(close - level) < threshold * atr
    valid = ~np.isnan(level) & ~np.isnan(atr)
    in_band = in_band & valid

    # First entrée dans la bande après être hors bande pendant min_gap_bars
    rolled_out = ~pd.Series(in_band).rolling(min_gap_bars).max().fillna(0).astype(bool)
    rolled_out_shifted = rolled_out.shift(1, fill_value=True).values
    first_touch = in_band & rolled_out_shifted
    return first_touch


# ---------------------------------------------------------------------------
# Per-touch statistics
# ---------------------------------------------------------------------------

def collect_touch_returns(df: pd.DataFrame, levels: pd.DataFrame, atr: np.ndarray,
                          horizons: list[int]) -> pd.DataFrame:
    """Pour chaque touch détecté, calcule le return % à plusieurs horizons."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    records = []
    for level_name in PIVOT_LEVELS:
        level_arr = levels[level_name].values
        touches = detect_touches(close, level_arr, atr, TOUCH_THRESHOLD_ATR)

        for idx in np.where(touches)[0]:
            entry = close[idx]
            level_val = level_arr[idx]
            atr_val = atr[idx]
            ts = df["timestamp"].iloc[idx]

            row = {
                "idx": idx,
                "timestamp": ts,
                "level_name": level_name,
                "level_type": "h" if level_name.startswith("h") else "l",
                "level_strength": int(level_name[1]),  # 1..4
                "entry_price": entry,
                "level_price": level_val,
                "atr": atr_val,
                "atr_norm": atr_val / entry,
                "session": session_label(ts.hour),
            }

            # Returns à plusieurs horizons
            for h in horizons:
                if idx + h >= n:
                    row[f"ret_pct_{h}"] = np.nan
                    row[f"ret_atr_{h}"] = np.nan
                    continue
                future_close = close[idx + h]
                row[f"ret_pct_{h}"] = (future_close - entry) / entry * 100
                row[f"ret_atr_{h}"] = (future_close - entry) / atr_val

            records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Stats par niveau / horizon
# ---------------------------------------------------------------------------

def stats_per_level_horizon(touches: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Stats agrégées : count, mean/std return, % positive."""
    rows = []
    for level_name in PIVOT_LEVELS:
        sub = touches[touches["level_name"] == level_name]
        for h in horizons:
            r = sub[f"ret_atr_{h}"].dropna()
            if len(r) < 10:
                continue
            rows.append({
                "level": level_name,
                "horizon_bars": h,
                "n_touches": len(r),
                "mean_atr_return": r.mean(),
                "std_atr_return": r.std(),
                "median_atr_return": r.median(),
                "skewness": r.skew(),
                "pct_positive": (r > 0).mean() * 100,
                "pct_above_05_atr": (r > 0.5).mean() * 100,
                "pct_below_neg05_atr": (r < -0.5).mean() * 100,
                "t_stat": r.mean() / (r.std() / np.sqrt(len(r))),  # one-sample t-test contre 0
            })
    return pd.DataFrame(rows)


def stats_per_level_session(touches: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """Stats par niveau × session pour un horizon donné."""
    rows = []
    for level_name in PIVOT_LEVELS:
        for session in ["asia", "europe", "us"]:
            sub = touches[(touches["level_name"] == level_name) &
                          (touches["session"] == session)]
            r = sub[f"ret_atr_{horizon}"].dropna()
            if len(r) < 10:
                continue
            rows.append({
                "level": level_name,
                "session": session,
                "n_touches": len(r),
                "mean_atr_return": r.mean(),
                "pct_positive": (r > 0).mean() * 100,
                "t_stat": r.mean() / (r.std() / np.sqrt(len(r))) if r.std() > 0 else 0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stratégie naïve par niveau (LONG sur L*, SHORT sur H*)
# ---------------------------------------------------------------------------

def naive_strategy_backtest(df: pd.DataFrame, levels: pd.DataFrame, atr: np.ndarray,
                             tp_atr: float = NAIVE_TP_ATR, sl_atr: float = NAIVE_SL_ATR,
                             time_barrier: int = NAIVE_TIME_BARRIER,
                             fees_pct: float = FEES_PCT) -> pd.DataFrame:
    """Naive bounce strategy:
    - LONG à chaque touch L*, TP = entry + tp_atr*ATR, SL = entry - sl_atr*ATR
    - SHORT à chaque touch H*, TP = entry - tp_atr*ATR, SL = entry + sl_atr*ATR
    - Time barrier = N bars max
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    rows = []
    for level_name in PIVOT_LEVELS:
        is_long = level_name.startswith("l")
        level_arr = levels[level_name].values
        touches = detect_touches(close, level_arr, atr, TOUCH_THRESHOLD_ATR)

        results = []
        for idx in np.where(touches)[0]:
            if idx + 1 + time_barrier > n:
                continue
            entry = close[idx]
            atr_val = atr[idx]

            if is_long:
                tp = entry + tp_atr * atr_val
                sl = entry - sl_atr * atr_val
            else:
                tp = entry - tp_atr * atr_val
                sl = entry + sl_atr * atr_val

            sub_high = high[idx + 1: idx + 1 + time_barrier]
            sub_low = low[idx + 1: idx + 1 + time_barrier]

            if is_long:
                tp_hit = sub_high >= tp
                sl_hit = sub_low <= sl
            else:
                tp_hit = sub_low <= tp
                sl_hit = sub_high >= sl

            first_tp = np.argmax(tp_hit) if tp_hit.any() else time_barrier
            first_sl = np.argmax(sl_hit) if sl_hit.any() else time_barrier

            if first_tp < first_sl:
                exit_price = tp
                outcome = "TP"
            elif first_sl < first_tp:
                exit_price = sl
                outcome = "SL"
            else:
                exit_price = close[idx + time_barrier]
                outcome = "TIMEOUT"

            if is_long:
                pnl_pct = (exit_price - entry) / entry * 100
            else:
                pnl_pct = (entry - exit_price) / entry * 100

            pnl_net = pnl_pct - 2 * fees_pct
            results.append({
                "level": level_name,
                "outcome": outcome,
                "pnl_pct": pnl_pct,
                "pnl_net": pnl_net,
            })

        if not results:
            continue

        sub = pd.DataFrame(results)
        rows.append({
            "level": level_name,
            "direction": "long" if is_long else "short",
            "n_trades": len(sub),
            "win_rate": (sub["pnl_net"] > 0).mean() * 100,
            "tp_rate": (sub["outcome"] == "TP").mean() * 100,
            "sl_rate": (sub["outcome"] == "SL").mean() * 100,
            "timeout_rate": (sub["outcome"] == "TIMEOUT").mean() * 100,
            "mean_pnl_net": sub["pnl_net"].mean(),
            "cumul_pnl_net": sub["pnl_net"].sum(),
            "sharpe_per_trade": sub["pnl_net"].mean() / sub["pnl_net"].std() if sub["pnl_net"].std() > 0 else 0,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(title: str, df: pd.DataFrame, columns: list[str], precision: int = 3):
    logger.info("=" * 110)
    logger.info(title)
    logger.info("=" * 110)
    if df.empty:
        logger.info("(empty)")
        return
    fmt = df[columns].copy()
    for col in fmt.select_dtypes(include=[np.floating]).columns:
        fmt[col] = fmt[col].round(precision)
    logger.info(fmt.to_string(index=False))
    logger.info("")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=Path("data_trad/BTCUSD_all_5m.csv"))
    p.add_argument("--output", type=Path, default=Path("data/patchtst_v5/pivot_analysis.json"))
    p.add_argument("--horizons", type=str, default="6,12,24,48,96",
                   help="Horizons en bars 5min séparés par virgules")
    p.add_argument("--max-bars", type=int, default=None)
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    logger.info("Loading: %s", args.csv)
    df = load_csv(args.csv)
    if args.max_bars:
        df = df.head(args.max_bars).reset_index(drop=True)
    logger.info("Loaded %d bars (%s → %s)", len(df), df["timestamp"].iloc[0], df["timestamp"].iloc[-1])

    logger.info("Computing Camarilla pivots ...")
    levels = compute_camarilla_5min(df)

    logger.info("Computing ATR(14) ...")
    atr = talib.ATR(df["high"].values, df["low"].values, df["close"].values, timeperiod=14)

    logger.info("Detecting touches and collecting returns at horizons %s ...", horizons)
    touches = collect_touch_returns(df, levels, atr, horizons)
    logger.info("Total touches detected: %d", len(touches))

    # Analyse 1: stats par niveau × horizon
    s1 = stats_per_level_horizon(touches, horizons)
    print_table(
        "1. STATS PAR NIVEAU × HORIZON (return en multiples d'ATR)",
        s1,
        ["level", "horizon_bars", "n_touches", "mean_atr_return", "median_atr_return",
         "std_atr_return", "skewness", "pct_positive", "t_stat"],
    )

    # Analyse 2: par session pour horizon 24 (2h)
    s2 = stats_per_level_session(touches, horizon=24)
    print_table(
        "2. STATS PAR NIVEAU × SESSION (horizon 2h)",
        s2,
        ["level", "session", "n_touches", "mean_atr_return", "pct_positive", "t_stat"],
    )

    # Analyse 3: stratégie naïve par niveau
    logger.info("Running naive bounce strategy backtest ...")
    s3 = naive_strategy_backtest(df, levels, atr)
    print_table(
        "3. STRATÉGIE NAÏVE (LONG sur L*, SHORT sur H*, RR 1:1, time 24 bars, fees 0.08%%)",
        s3,
        ["level", "direction", "n_trades", "win_rate", "tp_rate", "sl_rate",
         "timeout_rate", "mean_pnl_net", "cumul_pnl_net", "sharpe_per_trade"],
    )

    # Synthèse
    logger.info("=" * 110)
    logger.info("SYNTHESE")
    logger.info("=" * 110)
    profitable_levels = s3[s3["mean_pnl_net"] > 0.001]
    if len(profitable_levels):
        logger.info("Niveaux profitables (mean PnL net > 0.001%%) :")
        for _, row in profitable_levels.iterrows():
            logger.info("  %s (%s) : WR=%.1f%%, mean PnL net=%+.4f%%, %d trades",
                        row["level"], row["direction"], row["win_rate"],
                        row["mean_pnl_net"], row["n_trades"])
    else:
        logger.info("⚠️ Aucun niveau profitable en stratégie naïve.")

    significant_t = s1[(s1["n_touches"] > 100) & (s1["t_stat"].abs() > 2.0)]
    if len(significant_t):
        logger.info("\nSignaux statistiquement significatifs (|t| > 2, n > 100):")
        for _, row in significant_t.iterrows():
            logger.info("  %s @ horizon %d : mean=%+.3f ATR, t=%.2f, n=%d",
                        row["level"], row["horizon_bars"], row["mean_atr_return"],
                        row["t_stat"], row["n_touches"])
    else:
        logger.info("\nAucun signal directionnel statistiquement significatif (|t|>2).")

    # Save JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "n_total_touches": int(len(touches)),
        "horizons_bars": horizons,
        "stats_per_level_horizon": s1.to_dict(orient="records"),
        "stats_per_level_session": s2.to_dict(orient="records"),
        "naive_strategy": s3.to_dict(orient="records"),
    }
    args.output.write_text(json.dumps(out_data, indent=2, default=str))
    logger.info("JSON saved: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
