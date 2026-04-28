"""
pivot_labeler_levels.py — Triple Barrier avec niveaux pivot Camarilla comme TP/SL.

Pour chaque event:
  - Trouve les pivots Camarilla autour de l'entry close
  - LONG : TP = pivot Camarilla immédiat au-dessus
            SL = pivot Camarilla rang `sl_level` en-dessous (défaut 2)
  - SHORT: TP = pivot Camarilla immédiat en-dessous
            SL = pivot Camarilla rang `sl_level` au-dessus (défaut 2)
  - Skip events où le niveau de SL demandé n'existe pas (ex : sl_level=3 mais
    seulement 2 pivots au-dessus de l'entry → NO_PIVOT_LEVEL_3)
  - Time barrier configurable (default 24 bars = 2h)

Distances TP/SL varient naturellement selon position du trade et `sl_level`.
RR effectif varie par trade : plus `sl_level` est grand, plus le SL est lointain
(R/R plus défavorable mais moins de stop hunts).

Usage:
    python -m experiments.patchtst_v5.pivot_labeler_levels \\
        --features data/patchtst_v5/features_btc.parquet \\
        --events data/patchtst_v5/events_btc.parquet \\
        --output data/patchtst_v5/labels_btc_pivot_levels_sl3.parquet \\
        --sl-mode beyond --sl-level 3 --time-barrier 24 --fees-pct 0.02
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger("patchtst_v5.pivot_labeler_levels")

PIVOT_COLS = ["h1", "h2", "h3", "h4", "l1", "l2", "l3", "l4"]


def compute_camarilla_5min(timestamp: pd.Series, high: np.ndarray,
                            low: np.ndarray, close: np.ndarray) -> pd.DataFrame:
    """Recalcule les 8 niveaux Camarilla causaux et les broadcast au 5min."""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamp, utc=True, errors="coerce"),
        "high": high, "low": low, "close": close,
    }).dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

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
    return levels.reindex(df.set_index("timestamp").index, method="ffill").reset_index(drop=True)


def find_neighbor_levels(entry: float, levels: np.ndarray,
                          n_beyond: int = 2) -> tuple[float, float, float, float]:
    """Trouve les 4 niveaux pertinents autour de entry:
      - above: niveau immédiatement au-dessus (1er resistance)
      - below: niveau immédiatement en-dessous (1er support)
      - beyond_above: nième niveau au-dessus (n_beyond=2 → 2e résistance,
                       n_beyond=3 → 3e résistance, etc.)
      - beyond_below: nième niveau en-dessous (idem, n_beyond=2 par défaut)
    """
    valid = np.sort(levels[~np.isnan(levels)])
    above_levels = valid[valid > entry]
    below_levels = valid[valid < entry]

    above = above_levels[0] if len(above_levels) >= 1 else np.nan
    below = below_levels[-1] if len(below_levels) >= 1 else np.nan
    beyond_above = above_levels[n_beyond - 1] if len(above_levels) >= n_beyond else np.nan
    beyond_below = below_levels[-n_beyond] if len(below_levels) >= n_beyond else np.nan
    return above, below, beyond_above, beyond_below


def label_events(events: pd.DataFrame, features: pd.DataFrame,
                 high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 time_barrier: int, sl_mode: str, sl_buffer_atr: float,
                 fees_pct: float, sl_level: int = 2,
                 label_mode: str = "tp_first",
                 pnl_threshold_pct: float = 0.0) -> pd.DataFrame:
    """Triple Barrier avec niveaux Camarilla comme TP/SL dynamiques.

    sl_mode:
      - 'immediate-with-buffer': SL = pivot immédiat ± sl_buffer_atr × ATR (mix)
      - 'beyond': SL = niveau Camarilla au n-ième rang opposé (pur pivot)
        n contrôlé par sl_level (défaut 2 = 2e pivot opposé, configurable 2..4)

    label_mode:
      - 'tp_first' (défaut, legacy) : class1 = first_tp < first_sl
      - 'pnl_threshold' : class1 = pnl_after_fees_pct > pnl_threshold_pct
        (cible la profitabilité réelle après frais, indépendamment de la
        position de TP vs SL — corrige le biais du label trick)
    """
    n_events = len(events)
    n_bars = len(high)
    direction_arr = events["direction"].values.astype("int8")
    feature_idx_arr = events["feature_idx"].values.astype("int64")
    signal_close = events["close"].values.astype("float64")
    signal_atr = events["atr_14"].values.astype("float64")

    # Niveaux Camarilla au moment de chaque event
    levels_at_event = features[PIVOT_COLS].iloc[feature_idx_arr].values  # (n_events, 8)

    label = np.full(n_events, -1, dtype="int8")
    tp_price = np.full(n_events, np.nan, dtype="float64")
    sl_price = np.full(n_events, np.nan, dtype="float64")
    pnl_net = np.full(n_events, np.nan, dtype="float64")
    exit_bars = np.full(n_events, -1, dtype="int16")
    exit_reason = np.empty(n_events, dtype=object)
    rr_ratio = np.full(n_events, np.nan, dtype="float64")
    skipped_reason = np.empty(n_events, dtype=object)

    n_skipped_no_target = 0
    n_skipped_oof = 0

    for k in range(n_events):
        idx = feature_idx_arr[k]
        end = idx + 1 + time_barrier
        if end > n_bars:
            n_skipped_oof += 1
            skipped_reason[k] = "OUT_OF_DATA"
            continue

        direction = direction_arr[k]
        entry = signal_close[k]
        atr_t = signal_atr[k]

        above, below, beyond_above, beyond_below = find_neighbor_levels(
            entry, levels_at_event[k], n_beyond=sl_level)

        # TP toujours = pivot immédiat dans la direction du trade (pure pivot)
        if direction > 0:
            tp = above
        else:
            tp = below

        if np.isnan(tp):
            n_skipped_no_target += 1
            skipped_reason[k] = "NO_PIVOT_TARGET"
            continue

        # SL selon sl_mode
        if sl_mode == "beyond":
            # Pure pivot: SL = niveau Camarilla SUIVANT au-delà du support/résistance immédiat
            if direction > 0:
                sl = beyond_below
            else:
                sl = beyond_above
            if np.isnan(sl):
                n_skipped_no_target += 1
                skipped_reason[k] = f"NO_PIVOT_LEVEL_{sl_level}"
                continue
        else:  # immediate-with-buffer
            if direction > 0:
                sl_lvl = below
            else:
                sl_lvl = above
            if np.isnan(sl_lvl):
                n_skipped_no_target += 1
                skipped_reason[k] = "NO_PIVOT_TARGET"
                continue
            if direction > 0:
                sl = sl_lvl - sl_buffer_atr * atr_t
            else:
                sl = sl_lvl + sl_buffer_atr * atr_t

        tp_price[k] = tp
        sl_price[k] = sl

        # RR effectif (récompense/risque)
        if direction > 0:
            reward = abs(tp - entry)
            risk = abs(entry - sl)
        else:
            reward = abs(entry - tp)
            risk = abs(sl - entry)
        rr_ratio[k] = reward / risk if risk > 1e-12 else np.nan

        # Walk-forward exit detection
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
            exit_p = tp
            label[k] = 1
            exit_reason[k] = "TP"
            exit_bars[k] = first_tp + 1
        elif first_sl < first_tp:
            exit_p = sl
            label[k] = 0
            exit_reason[k] = "SL"
            exit_bars[k] = first_sl + 1
        elif first_tp == first_sl and first_tp < time_barrier:
            exit_p = sl
            label[k] = 0
            exit_reason[k] = "AMBIGUOUS"
            exit_bars[k] = first_tp + 1
        else:
            last_close = close[end - 1]
            exit_p = last_close
            exit_reason[k] = "TIMEOUT"
            exit_bars[k] = time_barrier
            if direction > 0:
                label[k] = 1 if last_close > entry else 0
            else:
                label[k] = 1 if last_close < entry else 0

        if direction > 0:
            pnl = 100.0 * (exit_p - entry) / entry
        else:
            pnl = 100.0 * (entry - exit_p) / entry
        pnl_net[k] = pnl - 2 * fees_pct

        # Override du label si mode pnl_threshold : cible la profitabilité
        # réelle au lieu de "TP first". Garde exit_reason inchangé pour debug.
        if label_mode == "pnl_threshold":
            label[k] = 1 if pnl_net[k] > pnl_threshold_pct else 0

    valid = label != -1
    out = events.copy()
    out["tp_price"] = tp_price.astype("float32")
    out["sl_price"] = sl_price.astype("float32")
    out["rr_effective"] = rr_ratio.astype("float32")
    out["label"] = label
    out["exit_bars"] = exit_bars
    out["exit_reason"] = exit_reason
    out["pnl_after_fees_pct"] = pnl_net.astype("float32")
    out = out.loc[valid].reset_index(drop=True)

    logger.info("Skipped: %d OUT_OF_DATA, %d NO_PIVOT_TARGET (extremes)", n_skipped_oof, n_skipped_no_target)
    return out


def report(labels: pd.DataFrame) -> None:
    n = len(labels)
    n_pos = int((labels["label"] == 1).sum())
    pos_rate = 100 * n_pos / n if n else 0.0

    logger.info("=" * 110)
    logger.info("LABEL SUMMARY (Pivot levels TP/SL)")
    logger.info("=" * 110)
    logger.info("Total labeled events : %d", n)
    logger.info("Class balance        : Label=1 %d (%.1f%%) | Label=0 %d (%.1f%%)",
                n_pos, pos_rate, n - n_pos, 100 - pos_rate)

    rr_stats = labels["rr_effective"].describe()
    logger.info("RR effectif (reward/risk):")
    logger.info("  min=%.2f q25=%.2f median=%.2f mean=%.2f q75=%.2f max=%.2f",
                rr_stats["min"], rr_stats["25%"], rr_stats["50%"],
                rr_stats["mean"], rr_stats["75%"], rr_stats["max"])

    reason_counts = labels["exit_reason"].value_counts()
    logger.info("Exit reasons:")
    for reason, count in reason_counts.items():
        logger.info("  %-12s : %6d (%.1f%%)", reason, count, 100 * count / n)

    pnl = labels["pnl_after_fees_pct"]
    logger.info("PnL net per-trade: mean=%+.4f%% median=%+.4f%% std=%.4f%%",
                pnl.mean(), pnl.median(), pnl.std())
    logger.info("Cumul PnL net    : %+.2f%%", pnl.sum())

    pos_mask = labels["label"] == 1
    neg_mask = labels["label"] == 0
    if pos_mask.any() and neg_mask.any():
        mean_win = labels.loc[pos_mask, "pnl_after_fees_pct"].mean()
        mean_loss = labels.loc[neg_mask, "pnl_after_fees_pct"].mean()
        breakeven = abs(mean_loss) / (mean_win + abs(mean_loss)) if mean_win > 0 else float("nan")
        oracle = labels.loc[pos_mask, "pnl_after_fees_pct"].sum()
        logger.info("Mean win net  : %+.4f%%   Mean loss net : %+.4f%%", mean_win, mean_loss)
        logger.info("Breakeven WR  : %.1f%%   Oracle cumul  : %+.1f%%", breakeven * 100, oracle)

        # Span estimé
        ts = pd.to_datetime(labels["timestamp"])
        span_years = (ts.max() - ts.min()).total_seconds() / (365.25 * 86400)
        logger.info("Span: %.2f years   Oracle annualisé: %+.1f%%/an", span_years, oracle / span_years)
    logger.info("=" * 110)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=Path("data/patchtst_v5/features_btc.parquet"))
    p.add_argument("--events", type=Path, default=Path("data/patchtst_v5/events_btc.parquet"))
    p.add_argument("--output", type=Path, default=Path("data/patchtst_v5/labels_btc_pivot_levels.parquet"))
    p.add_argument("--time-barrier", type=int, default=24, help="Bars max (24 = 2h)")
    p.add_argument("--sl-mode", type=str, default="beyond",
                   choices=["beyond", "immediate-with-buffer"],
                   help="'beyond' = SL au n-ième pivot opposé (pure pivot, no ATR). "
                        "'immediate-with-buffer' = SL au pivot immédiat - buffer×ATR")
    p.add_argument("--sl-level", type=int, default=2,
                   help="(beyond mode only) profondeur du SL. 2 = 2e pivot opposé "
                        "(défaut), 3 = 3e pivot, 4 = 4e pivot. Camarilla a 4 pivots "
                        "par côté donc max=4. Plus la valeur est haute, plus le SL "
                        "est lointain (R/R plus défavorable mais moins de stop hunts).")
    p.add_argument("--sl-buffer-atr", type=float, default=0.0,
                   help="(immediate-with-buffer mode only) buffer ATR au-delà du niveau immédiat")
    p.add_argument("--fees-pct", type=float, default=0.02, help="One-way fee %% (0.02 = maker)")
    p.add_argument("--label-mode", type=str, default="tp_first",
                   choices=["tp_first", "pnl_threshold"],
                   help="'tp_first' (legacy) : class1 = first_tp < first_sl. "
                        "'pnl_threshold' : class1 = pnl_after_fees_pct > "
                        "--pnl-threshold-pct. Recommandé pour corriger le biais "
                        "trick (TP collé). Garde la même logique TP/SL/timeout "
                        "pour le PnL — seul le label binaire change.")
    p.add_argument("--pnl-threshold-pct", type=float, default=0.10,
                   help="Seuil de profitabilité (en %%) pour label_mode=pnl_threshold. "
                        "0.10 = +0.10%% net après frais. Défaut 0.10.")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    logger.info("Loading features: %s", args.features)
    features = pd.read_parquet(args.features, columns=["timestamp", "high", "low", "close"])
    high = features["high"].values.astype("float64")
    low = features["low"].values.astype("float64")
    close = features["close"].values.astype("float64")
    logger.info("Computing Camarilla pivots (causal, prev day H/L/C → ffill 5min) ...")
    pivot_levels = compute_camarilla_5min(features["timestamp"], high, low, close)
    features = pd.concat([features.reset_index(drop=True), pivot_levels], axis=1)

    logger.info("Loading events: %s", args.events)
    events = pd.read_parquet(args.events)
    logger.info("Events: %d", len(events))
    logger.info("Time barrier: %d bars (~%d min)", args.time_barrier, args.time_barrier * 5)
    if args.sl_mode == "beyond":
        if args.sl_level < 2:
            raise SystemExit("--sl-level must be >= 2 (1 = immediate pivot, use immediate-with-buffer mode for that)")
        logger.info("SL mode: BEYOND (pure pivot — SL = pivot Camarilla rang %d opposé)", args.sl_level)
    else:
        logger.info("SL mode: IMMEDIATE-WITH-BUFFER (SL = pivot immédiat - %.2f × ATR)",
                    args.sl_buffer_atr)
    logger.info("Fees: %.3f%% one-way (round-trip = %.3f%%)", args.fees_pct, 2 * args.fees_pct)
    logger.info("Label mode: %s%s", args.label_mode,
                f" (pnl_threshold={args.pnl_threshold_pct:.3f}%)"
                if args.label_mode == "pnl_threshold" else "")

    labeled = label_events(events, features, high, low, close,
                           args.time_barrier, args.sl_mode, args.sl_buffer_atr,
                           args.fees_pct, sl_level=args.sl_level,
                           label_mode=args.label_mode,
                           pnl_threshold_pct=args.pnl_threshold_pct)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(args.output, compression="snappy", index=False)
    logger.info("Output: %s (%d events)", args.output, len(labeled))
    report(labeled)
    return 0


if __name__ == "__main__":
    sys.exit(main())
