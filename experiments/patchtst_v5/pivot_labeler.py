"""
pivot_labeler.py — Triple Barrier Method ATR-adaptatif (étape 4 v5.0).

Pour chaque event détecté par event_detector.py, calcule:
  - TP = entry ± tp_atr × ATR (long/short)
  - SL selon --sl-mode:
      * from_entry  (default): SL = entry ∓ sl_atr × ATR (symétrique)
      * from_signal:           SL = signal_low/high ∓ sl_atr × ATR
  - Time barrier = 24 bougies (2h)

Walk forward through OHLC bars and detect which barrier hits first.
  - Label = 1 si TP touché avant SL avant timeout
  - Label = 0 sinon (SL ou timeout négatif)

ATR-adaptatif: les barrières s'ajustent automatiquement à la volatilité du moment.

sl_mode=from_entry vs from_signal:
  Pour les events Engulfing (73% des cas), la bougie de signal est large et
  signal_low se trouve loin du close. Le mode from_signal produisait alors une
  asymétrie SL/TP défavorable (SL ~1.5-2.0×ATR vs TP 1.0×ATR), neutralisant
  l'edge même avec WR 63% au top 1%. Le mode from_entry impose une symétrie
  TP/SL contrôlée et donne un breakeven net WR ≈ 50% (au lieu de ~58%).

Usage:
    python -m experiments.patchtst_v5.pivot_labeler \\
        --features data/patchtst_v5/features_btc.parquet \\
        --events data/patchtst_v5/events_btc.parquet \\
        --output data/patchtst_v5/labels_btc.parquet

Voir STATUS_v5.0.md et experiments/patchtst_v5/README.md.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger("patchtst_v5.pivot_labeler")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEFAULT_TP_ATR = 1.0       # TP = entry ± 1.0 × ATR
DEFAULT_SL_ATR = 1.0       # SL distance in ATR (interprétation selon --sl-mode)
DEFAULT_SL_MODE = "from_entry"  # 'from_entry' (symétrique) ou 'from_signal'
DEFAULT_TIME_BARRIER = 24  # 24 bougies × 5min = 2h
DEFAULT_FEES_PCT = 0.04    # 0.04% taker fee (Binance) — applied 2× round trip


# ---------------------------------------------------------------------------
# Triple Barrier (vectorized walk-forward over events)
# ---------------------------------------------------------------------------

def label_events(
    events: pd.DataFrame,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    tp_atr: float,
    sl_atr: float,
    sl_mode: str,
    time_barrier: int,
    fees_pct: float,
) -> pd.DataFrame:
    """
    Apply Triple Barrier to every event. Returns the events DataFrame enriched with
    label, exit_bars, exit_price, exit_reason, pnl_pct, pnl_after_fees.

    Implementation: Python loop over events (~36k rows). Inner loop is vectorized
    on the time_barrier window via numpy. Total cost ~1-2s.
    """
    if sl_mode not in ("from_entry", "from_signal"):
        raise ValueError(f"sl_mode must be 'from_entry' or 'from_signal', got {sl_mode!r}")

    n_events = len(events)
    n_bars = len(high)
    logger.info("Labeling %d events on %d bars of OHLC", n_events, n_bars)
    logger.info("TP = entry ± %.2f × ATR", tp_atr)
    if sl_mode == "from_entry":
        logger.info("SL = entry ∓ %.2f × ATR (mode: from_entry, symétrique)", sl_atr)
    else:
        logger.info("SL = signal_low/high ∓ %.2f × ATR (mode: from_signal)", sl_atr)
    logger.info("Time barrier = %d bars (~%d min)", time_barrier, time_barrier * 5)
    logger.info("Round-trip fees = 2 × %.3f%% = %.3f%%", fees_pct, 2 * fees_pct)

    direction_arr = events["direction"].values.astype("int8")
    feature_idx_arr = events["feature_idx"].values.astype("int64")
    signal_close = events["close"].values.astype("float64")
    signal_low = events["low"].values.astype("float64")
    signal_high = events["high"].values.astype("float64")
    signal_atr = events["atr_14"].values.astype("float64")

    # Pre-allocate outputs
    label = np.zeros(n_events, dtype="int8")
    exit_bars = np.zeros(n_events, dtype="int16")
    exit_price = np.zeros(n_events, dtype="float64")
    exit_reason = np.empty(n_events, dtype=object)
    tp_price_out = np.zeros(n_events, dtype="float64")
    sl_price_out = np.zeros(n_events, dtype="float64")
    pnl_pct = np.zeros(n_events, dtype="float64")

    # Skip events whose lookahead window goes past the available data
    skipped_oof = 0

    for k in range(n_events):
        idx = feature_idx_arr[k]
        end = idx + 1 + time_barrier
        if end > n_bars:
            skipped_oof += 1
            label[k] = -1  # mark as ignored, will be filtered out
            exit_reason[k] = "OUT_OF_DATA"
            continue

        direction = direction_arr[k]
        entry = signal_close[k]
        atr_t = signal_atr[k]

        if direction > 0:  # long
            tp = entry + tp_atr * atr_t
            if sl_mode == "from_entry":
                sl = entry - sl_atr * atr_t
            else:  # from_signal
                sl = signal_low[k] - sl_atr * atr_t
        else:              # short
            tp = entry - tp_atr * atr_t
            if sl_mode == "from_entry":
                sl = entry + sl_atr * atr_t
            else:  # from_signal
                sl = signal_high[k] + sl_atr * atr_t

        tp_price_out[k] = tp
        sl_price_out[k] = sl

        # Vectorized scan of the next time_barrier bars
        sub_high = high[idx + 1: end]
        sub_low = low[idx + 1: end]

        if direction > 0:
            tp_hit = sub_high >= tp
            sl_hit = sub_low <= sl
        else:
            tp_hit = sub_low <= tp
            sl_hit = sub_high >= sl

        first_tp = np.argmax(tp_hit) if tp_hit.any() else time_barrier
        first_sl = np.argmax(sl_hit) if sl_hit.any() else time_barrier

        if first_tp < first_sl:
            # TP hits first
            exit_bars[k] = int(first_tp + 1)
            exit_price[k] = tp
            exit_reason[k] = "TP"
            label[k] = 1
        elif first_sl < first_tp:
            # SL hits first
            exit_bars[k] = int(first_sl + 1)
            exit_price[k] = sl
            exit_reason[k] = "SL"
            label[k] = 0
        elif first_tp == first_sl and first_tp < time_barrier:
            # Same bar — ambiguous, treat conservatively as SL hit
            exit_bars[k] = int(first_tp + 1)
            exit_price[k] = sl
            exit_reason[k] = "AMBIGUOUS"
            label[k] = 0
        else:
            # Timeout — close at last bar's close
            last_close = close[end - 1]
            exit_bars[k] = time_barrier
            exit_price[k] = last_close
            exit_reason[k] = "TIMEOUT"
            if direction > 0:
                label[k] = 1 if last_close > entry else 0
            else:
                label[k] = 1 if last_close < entry else 0

        # Compute realized PnL %
        if direction > 0:
            pnl_pct[k] = 100.0 * (exit_price[k] - entry) / entry
        else:
            pnl_pct[k] = 100.0 * (entry - exit_price[k]) / entry

    if skipped_oof:
        logger.warning("%d events skipped (lookahead beyond data end)", skipped_oof)

    pnl_after_fees = pnl_pct - 2 * fees_pct  # round-trip fees applied symmetrically

    out = events.copy()
    out["tp_price"] = tp_price_out.astype("float32")
    out["sl_price"] = sl_price_out.astype("float32")
    out["label"] = label
    out["exit_bars"] = exit_bars
    out["exit_price"] = exit_price.astype("float32")
    out["exit_reason"] = exit_reason
    out["pnl_pct"] = pnl_pct.astype("float32")
    out["pnl_after_fees_pct"] = pnl_after_fees.astype("float32")

    # Filter out OUT_OF_DATA events
    valid_mask = out["label"] != -1
    if (~valid_mask).any():
        out = out[valid_mask].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_labels(events: pd.DataFrame) -> None:
    n = len(events)
    n_pos = int((events["label"] == 1).sum())
    n_neg = int((events["label"] == 0).sum())
    pos_rate = 100 * n_pos / n if n else 0.0

    logger.info("=" * 70)
    logger.info("LABEL SUMMARY")
    logger.info("=" * 70)
    logger.info("Total labeled events : %d", n)
    logger.info("Class balance        : Label=1 %d (%.1f%%)  |  Label=0 %d (%.1f%%)",
                n_pos, pos_rate, n_neg, 100 - pos_rate)

    # Exit reason distribution
    reason_counts = events["exit_reason"].value_counts()
    logger.info("Exit reasons:")
    for reason, count in reason_counts.items():
        logger.info("  %-12s : %6d (%.1f%%)", reason, count, 100 * count / n)

    # Time-to-exit distribution
    bars = events["exit_bars"]
    logger.info("Bars-to-exit         : min=%d  median=%.1f  mean=%.2f  max=%d",
                bars.min(), bars.median(), bars.mean(), bars.max())

    # PnL stats
    pnl = events["pnl_pct"]
    pnl_net = events["pnl_after_fees_pct"]
    logger.info("PnL gross %% (per-trade): mean=%+.3f  median=%+.3f  std=%.3f",
                pnl.mean(), pnl.median(), pnl.std())
    logger.info("PnL net   %% (per-trade): mean=%+.3f  median=%+.3f  std=%.3f",
                pnl_net.mean(), pnl_net.median(), pnl_net.std())
    logger.info("Cumul PnL gross  : %+.1f%%   |   Cumul PnL net : %+.1f%%",
                pnl.sum(), pnl_net.sum())

    # Win rate by direction
    long_mask = events["direction"] > 0
    short_mask = events["direction"] < 0
    n_long = int(long_mask.sum())
    n_short = int(short_mask.sum())
    if n_long:
        wr_long = 100 * (events.loc[long_mask, "label"] == 1).sum() / n_long
        logger.info("Win rate LONG  : %5.1f%% (%d events)", wr_long, n_long)
    if n_short:
        wr_short = 100 * (events.loc[short_mask, "label"] == 1).sum() / n_short
        logger.info("Win rate SHORT : %5.1f%% (%d events)", wr_short, n_short)

    # Win rate by year
    years = pd.to_datetime(events["timestamp"]).dt.year
    logger.info("Win rate per year:")
    for year in sorted(years.unique()):
        mask = years == year
        n_y = int(mask.sum())
        wr_y = 100 * (events.loc[mask, "label"] == 1).sum() / n_y
        logger.info("  %d : %5.1f%% (%d events)", int(year), wr_y, n_y)
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=Path("data/patchtst_v5/features_btc.parquet"),
                   help="Features parquet (for OHLC walk-forward). Default: data/patchtst_v5/features_btc.parquet")
    p.add_argument("--events", type=Path, default=Path("data/patchtst_v5/events_btc.parquet"),
                   help="Events parquet (output of event_detector). Default: data/patchtst_v5/events_btc.parquet")
    p.add_argument("--output", type=Path, default=Path("data/patchtst_v5/labels_btc.parquet"),
                   help="Output labeled events parquet. Default: data/patchtst_v5/labels_btc.parquet")
    p.add_argument("--tp-atr", type=float, default=DEFAULT_TP_ATR,
                   help=f"TP distance in ATR multiples (default: {DEFAULT_TP_ATR})")
    p.add_argument("--sl-atr", type=float, default=DEFAULT_SL_ATR,
                   help=f"SL distance in ATR (default: {DEFAULT_SL_ATR}). "
                        f"With from_entry: symétrique entry ∓ sl_atr × ATR. "
                        f"With from_signal: signal_low/high ∓ sl_atr × ATR.")
    p.add_argument("--sl-mode", type=str, default=DEFAULT_SL_MODE,
                   choices=["from_entry", "from_signal"],
                   help=f"SL anchor: 'from_entry' (symétrique avec TP, default) ou 'from_signal' (depuis swing low/high)")
    p.add_argument("--time-barrier", type=int, default=DEFAULT_TIME_BARRIER,
                   help=f"Max bars to hold before timeout (default: {DEFAULT_TIME_BARRIER})")
    p.add_argument("--fees-pct", type=float, default=DEFAULT_FEES_PCT,
                   help=f"One-way fee in %% (default: {DEFAULT_FEES_PCT}%% = Binance taker)")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    logger.info("Loading features (for OHLC walk-forward): %s", args.features)
    feat_cols = ["high", "low", "close"]
    features = pd.read_parquet(args.features, columns=feat_cols)
    high = features["high"].values.astype("float64")
    low = features["low"].values.astype("float64")
    close = features["close"].values.astype("float64")
    atr = np.zeros_like(close)  # not needed once events carry their own atr_14
    logger.info("Loaded %d OHLC bars", len(features))

    logger.info("Loading events: %s", args.events)
    events = pd.read_parquet(args.events)
    logger.info("Loaded %d events", len(events))

    labeled = label_events(
        events=events,
        high=high, low=low, close=close, atr=atr,
        tp_atr=args.tp_atr,
        sl_atr=args.sl_atr,
        sl_mode=args.sl_mode,
        time_barrier=args.time_barrier,
        fees_pct=args.fees_pct,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing labeled events parquet: %s", args.output)
    labeled.to_parquet(args.output, compression="snappy", index=False)

    report_labels(labeled)
    logger.info("Done. %d labeled events written to %s", len(labeled), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
