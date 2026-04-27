"""
test_sl_variations.py — Grid search sur (TP, SL, Time barrier).

Hypothèse user: le SL trop serré est touché par du noise (stop hunt) avant
que le prix ne reparte dans la bonne direction. Élargir le SL devrait:
  - Réduire le taux de SL hit
  - Augmenter le taux de TP hit
  - Augmenter l'Oracle PnL (sélection parfaite des Label=1)

Test: pour chaque combinaison (TP_atr, SL_atr, time_barrier), recalcule les
labels Triple Barrier sur les events existants et compare:
  - Class balance
  - Mean win / mean loss
  - Breakeven WR
  - Oracle cumul PnL annualisé
  - Win rate maximum théorique vs minimum profitable

Usage:
    python -m experiments.patchtst_v5.test_sl_variations \\
        --features data/patchtst_v5/features_btc.parquet \\
        --events data/patchtst_v5/events_btc.parquet \\
        --output data/patchtst_v5/sl_grid_search.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger("patchtst_v5.test_sl_variations")

DEFAULT_TP_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
DEFAULT_SL_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
DEFAULT_TIME_GRID = [24, 48, 96]
DEFAULT_FEES_PCT = 0.04   # maker round-trip = 0.08%


def label_events_grid(events: pd.DataFrame, high: np.ndarray, low: np.ndarray,
                      close: np.ndarray, tp_atr: float, sl_atr: float,
                      time_barrier: int, fees_pct: float) -> pd.DataFrame:
    """Re-label events avec TP/SL/time donnés. Retourne (labels, pnl_net)."""
    n_events = len(events)
    n_bars = len(high)
    direction_arr = events["direction"].values.astype("int8")
    feature_idx_arr = events["feature_idx"].values.astype("int64")
    signal_close = events["close"].values.astype("float64")
    signal_atr = events["atr_14"].values.astype("float64")

    label = np.full(n_events, -1, dtype="int8")
    pnl_net = np.full(n_events, np.nan, dtype="float64")
    exit_reason = np.empty(n_events, dtype=object)

    for k in range(n_events):
        idx = feature_idx_arr[k]
        end = idx + 1 + time_barrier
        if end > n_bars:
            continue
        direction = direction_arr[k]
        entry = signal_close[k]
        atr_t = signal_atr[k]

        if direction > 0:  # long
            tp = entry + tp_atr * atr_t
            sl = entry - sl_atr * atr_t
        else:
            tp = entry - tp_atr * atr_t
            sl = entry + sl_atr * atr_t

        sub_high = high[idx + 1: end]
        sub_low = low[idx + 1: end]

        if direction > 0:
            tp_hit = sub_high >= tp
            sl_hit = sub_low <= sl
        else:
            tp_hit = sub_low <= tp
            sl_hit = sub_high >= sl

        first_tp = int(np.argmax(tp_hit)) if tp_hit.any() else time_barrier
        first_sl = int(np.argmax(sl_hit)) if sl_hit.any() else time_barrier

        if first_tp < first_sl:
            exit_price = tp
            label[k] = 1
            exit_reason[k] = "TP"
        elif first_sl < first_tp:
            exit_price = sl
            label[k] = 0
            exit_reason[k] = "SL"
        elif first_tp == first_sl and first_tp < time_barrier:
            exit_price = sl  # AMBIGUOUS conservative
            label[k] = 0
            exit_reason[k] = "AMBIGUOUS"
        else:
            exit_price = close[end - 1]
            exit_reason[k] = "TIMEOUT"
            if direction > 0:
                label[k] = 1 if exit_price > entry else 0
            else:
                label[k] = 1 if exit_price < entry else 0

        if direction > 0:
            pnl = 100.0 * (exit_price - entry) / entry
        else:
            pnl = 100.0 * (entry - exit_price) / entry
        pnl_net[k] = pnl - 2 * fees_pct

    valid = label != -1
    return pd.DataFrame({
        "label": label[valid],
        "pnl_net": pnl_net[valid],
        "exit_reason": exit_reason[valid],
    })


def evaluate_config(labels: pd.DataFrame, span_years: float) -> dict:
    """Calcule métriques pour une configuration."""
    n = len(labels)
    n_pos = int((labels["label"] == 1).sum())
    n_neg = int((labels["label"] == 0).sum())
    pos_rate = n_pos / n if n else 0.0

    pos_mask = labels["label"] == 1
    neg_mask = labels["label"] == 0
    mean_win = float(labels.loc[pos_mask, "pnl_net"].mean()) if n_pos else 0.0
    mean_loss = float(labels.loc[neg_mask, "pnl_net"].mean()) if n_neg else 0.0

    # Breakeven WR
    if mean_win > 0 and mean_loss < 0:
        breakeven_wr = abs(mean_loss) / (mean_win + abs(mean_loss))
    else:
        breakeven_wr = float("nan")

    # Oracle = somme des PnL des Label=1 (sélection parfaite)
    oracle_cumul = float(labels.loc[pos_mask, "pnl_net"].sum())
    oracle_annualized = oracle_cumul / span_years if span_years > 0 else 0.0

    # Distribution des exits
    exit_counts = labels["exit_reason"].value_counts(normalize=True).to_dict()

    # All-events PnL (si on tradait tout)
    all_events_cumul = float(labels["pnl_net"].sum())
    all_events_annualized = all_events_cumul / span_years if span_years > 0 else 0.0

    return {
        "n_total": n,
        "n_label_1": n_pos,
        "class_1_ratio": pos_rate,
        "mean_win_net": mean_win,
        "mean_loss_net": mean_loss,
        "breakeven_wr": breakeven_wr,
        "oracle_cumul_net": oracle_cumul,
        "oracle_annualized_net": oracle_annualized,
        "all_events_cumul_net": all_events_cumul,
        "all_events_annualized_net": all_events_annualized,
        "tp_pct": exit_counts.get("TP", 0) * 100,
        "sl_pct": exit_counts.get("SL", 0) * 100,
        "timeout_pct": exit_counts.get("TIMEOUT", 0) * 100,
        "ambiguous_pct": exit_counts.get("AMBIGUOUS", 0) * 100,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=Path("data/patchtst_v5/features_btc.parquet"))
    p.add_argument("--events", type=Path, default=Path("data/patchtst_v5/events_btc.parquet"))
    p.add_argument("--output", type=Path, default=Path("data/patchtst_v5/sl_grid_search.json"))
    p.add_argument("--tp-grid", type=str, default=",".join(str(x) for x in DEFAULT_TP_GRID))
    p.add_argument("--sl-grid", type=str, default=",".join(str(x) for x in DEFAULT_SL_GRID))
    p.add_argument("--time-grid", type=str, default=",".join(str(x) for x in DEFAULT_TIME_GRID))
    p.add_argument("--fees-pct", type=float, default=DEFAULT_FEES_PCT)
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    tp_grid = [float(x) for x in args.tp_grid.split(",")]
    sl_grid = [float(x) for x in args.sl_grid.split(",")]
    time_grid = [int(x) for x in args.time_grid.split(",")]

    logger.info("Loading features ...")
    features = pd.read_parquet(args.features, columns=["high", "low", "close"])
    high = features["high"].values.astype("float64")
    low = features["low"].values.astype("float64")
    close = features["close"].values.astype("float64")

    logger.info("Loading events ...")
    events = pd.read_parquet(args.events)
    n_events = len(events)
    logger.info("Events: %d", n_events)

    # Span in years (test set period)
    events["ts"] = pd.to_datetime(events["timestamp"])
    span_years_full = (events["ts"].max() - events["ts"].min()).total_seconds() / (365.25 * 86400)
    logger.info("Full span: %.2f years", span_years_full)

    # Grid search
    logger.info("Running grid: %d × %d × %d = %d configs",
                len(tp_grid), len(sl_grid), len(time_grid),
                len(tp_grid) * len(sl_grid) * len(time_grid))

    rows = []
    for tp_atr, sl_atr, time_barrier in product(tp_grid, sl_grid, time_grid):
        labels = label_events_grid(events, high, low, close, tp_atr, sl_atr, time_barrier, args.fees_pct)
        metrics = evaluate_config(labels, span_years_full)
        metrics.update({
            "tp_atr": tp_atr,
            "sl_atr": sl_atr,
            "time_barrier": time_barrier,
            "rr_ratio": tp_atr / sl_atr,
        })
        rows.append(metrics)
        logger.info(
            "TP=%.1f SL=%.1f T=%d : Class1=%.1f%% TP=%.1f%% SL=%.1f%% TO=%.1f%% | "
            "Win=%+.3f%% Loss=%+.3f%% | BE=%.1f%% | Oracle=%+.1f%%/an",
            tp_atr, sl_atr, time_barrier, metrics["class_1_ratio"] * 100,
            metrics["tp_pct"], metrics["sl_pct"], metrics["timeout_pct"],
            metrics["mean_win_net"], metrics["mean_loss_net"],
            metrics["breakeven_wr"] * 100, metrics["oracle_annualized_net"],
        )

    df = pd.DataFrame(rows)

    # Tableau synthèse trié par Oracle annualisé
    logger.info("=" * 130)
    logger.info("TOP 10 CONFIGS PAR ORACLE ANNUALISÉ")
    logger.info("=" * 130)
    cols = ["tp_atr", "sl_atr", "time_barrier", "rr_ratio",
            "class_1_ratio", "tp_pct", "sl_pct", "timeout_pct",
            "mean_win_net", "mean_loss_net", "breakeven_wr",
            "oracle_annualized_net"]
    top = df.nlargest(10, "oracle_annualized_net")[cols].copy()
    for col in ["class_1_ratio", "tp_pct", "sl_pct", "timeout_pct",
                "mean_win_net", "mean_loss_net", "breakeven_wr"]:
        top[col] = (top[col] * 100).round(1) if col in ["class_1_ratio", "breakeven_wr"] else top[col].round(3)
    top["oracle_annualized_net"] = top["oracle_annualized_net"].round(1)
    logger.info(top.to_string(index=False))
    logger.info("")

    # Configs avec breakeven_WR le plus bas (les plus exploitables par un modèle imparfait)
    logger.info("=" * 130)
    logger.info("TOP 10 CONFIGS PAR BREAKEVEN WR LE PLUS BAS (les plus tolérantes)")
    logger.info("=" * 130)
    df_valid = df.dropna(subset=["breakeven_wr"])
    bottom_be = df_valid.nsmallest(10, "breakeven_wr")[cols].copy()
    for col in ["class_1_ratio", "tp_pct", "sl_pct", "timeout_pct",
                "mean_win_net", "mean_loss_net", "breakeven_wr"]:
        bottom_be[col] = (bottom_be[col] * 100).round(1) if col in ["class_1_ratio", "breakeven_wr"] else bottom_be[col].round(3)
    bottom_be["oracle_annualized_net"] = bottom_be["oracle_annualized_net"].round(1)
    logger.info(bottom_be.to_string(index=False))
    logger.info("")

    # Comparaison directe baseline (TP=2 SL=1 T=24) vs élargir SL (TP=2 SL=2/3 T=24)
    logger.info("=" * 130)
    logger.info("TEST DIRECT HYPOTHESE STOP HUNT — TP=2.0 fixé, SL varie, T=24")
    logger.info("=" * 130)
    sub = df[(df["tp_atr"] == 2.0) & (df["time_barrier"] == 24)].sort_values("sl_atr")
    sub_disp = sub[cols].copy()
    for col in ["class_1_ratio", "tp_pct", "sl_pct", "timeout_pct",
                "mean_win_net", "mean_loss_net", "breakeven_wr"]:
        sub_disp[col] = (sub_disp[col] * 100).round(1) if col in ["class_1_ratio", "breakeven_wr"] else sub_disp[col].round(3)
    sub_disp["oracle_annualized_net"] = sub_disp["oracle_annualized_net"].round(1)
    logger.info(sub_disp.to_string(index=False))
    logger.info("")

    # Sauvegarde
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(df.to_dict(orient="records"), indent=2, default=str))
    logger.info("JSON saved: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
