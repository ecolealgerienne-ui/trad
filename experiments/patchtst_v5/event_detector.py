"""
event_detector.py — Détection des triggers d'entrée scalping (étape 3 v5.0).

Lit le parquet de features (output de feature_builder.py) et identifie les bougies
où une combinaison de conditions (pattern reversal + proximité pivot Camarilla +
spike de volume) constitue un signal d'entrée potentiel.

Logique trigger (par défaut, configurable via CLI) :
  1. ≥1 pattern reversal directionnel fire (5 patterns: ENGULFING, HAMMER, INVERTEDHAMMER,
     SHOOTINGSTAR, HANGINGMAN — Doji exclu par défaut car non-directionnel)
  2. La somme signée des patterns donne une direction non-ambiguë (pas mixed)
  3. |dist_camarilla_nearest_norm| < pivot_distance (default 0.3 ATR)
  4. volume_zscore_20p > volume_threshold (default 1.5)

Output: parquet avec une ligne par event, colonnes prêtes pour pivot_labeler.py.

Usage:
    python -m experiments.patchtst_v5.event_detector \\
        --features data/patchtst_v5/features_btc.parquet \\
        --output data/patchtst_v5/events_btc.parquet

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

logger = logging.getLogger("patchtst_v5.event_detector")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Patterns reversal directionnels (exclut explicitement les patterns d'indécision)
DEFAULT_TRIGGER_PATTERNS = [
    "CDLENGULFING",        # 17% — bullish/bearish engulfing 2 bars
    "CDLHAMMER",           # 3% — hammer reversal at low
    "CDLINVERTEDHAMMER",   # 0.7% — inverted hammer reversal at low
    "CDLSHOOTINGSTAR",     # 0.7% — shooting star reversal at high
    "CDLHANGINGMAN",       # 2% — hanging man reversal at high
]

# Patterns ajoutables sur demande (--include-doji)
NON_DIRECTIONAL_PATTERNS = ["CDLDOJI", "CDLDRAGONFLYDOJI", "CDLGRAVESTONEDOJI", "CDLLONGLEGGEDDOJI"]

DEFAULT_PIVOT_DISTANCE = 0.3      # |dist_camarilla_nearest_norm| < 0.3 (en unités ATR)
DEFAULT_VOLUME_THRESHOLD = 1.5    # volume_zscore_20p > 1.5

# Colonnes minimales attendues dans le parquet de features
REQUIRED_FEATURE_COLS = [
    "timestamp", "asset", "open", "high", "low", "close", "volume", "atr_14",
    "dist_camarilla_nearest_norm", "volume_zscore_20p",
]


# ---------------------------------------------------------------------------
# Détection
# ---------------------------------------------------------------------------

def detect_events(
    features: pd.DataFrame,
    trigger_patterns: list[str],
    pivot_distance: float,
    volume_threshold: float,
    warmup_bars: int = 400,
) -> pd.DataFrame:
    """Apply trigger logic and return one row per event."""
    n = len(features)
    logger.info("Total bars: %d", n)
    logger.info("Trigger patterns (%d): %s", len(trigger_patterns), trigger_patterns)
    logger.info("Pivot distance threshold: %.3f ATR", pivot_distance)
    logger.info("Volume z-score threshold: > %.2f", volume_threshold)
    logger.info("Warmup bars skipped: %d", warmup_bars)

    missing_patterns = [p for p in trigger_patterns if p not in features.columns]
    if missing_patterns:
        raise ValueError(f"Missing patterns in features parquet: {missing_patterns}")

    pattern_block = features[trigger_patterns]
    pattern_score = pattern_block.sum(axis=1).astype("int32")          # signed strength
    pattern_abs = pattern_block.abs().sum(axis=1).astype("int32")      # 0 if no fire

    # Conditions
    cond_pattern = pattern_abs > 0
    cond_direction = pattern_score != 0       # exclude mixed (sum cancels out)
    cond_pivot = features["dist_camarilla_nearest_norm"].abs() < pivot_distance
    cond_volume = features["volume_zscore_20p"] > volume_threshold

    # Skip warmup (NaN-prone period: ATR/multi-TF rolling windows)
    cond_warmup = np.arange(n) >= warmup_bars

    mask = cond_pattern & cond_direction & cond_pivot & cond_volume & cond_warmup
    n_events = int(mask.sum())

    # Decomposition stats
    n_pat = int(cond_pattern.sum())
    n_dir = int((cond_pattern & cond_direction).sum())
    n_piv = int((cond_pattern & cond_direction & cond_pivot).sum())
    n_vol = int((cond_pattern & cond_direction & cond_pivot & cond_volume).sum())

    logger.info("=" * 70)
    logger.info("FILTER FUNNEL (cumulative)")
    logger.info("=" * 70)
    logger.info("  After pattern fire    : %8d (%.3f%%)", n_pat, 100 * n_pat / n)
    logger.info("  After direction != 0  : %8d (%.3f%%)", n_dir, 100 * n_dir / n)
    logger.info("  After pivot proximity : %8d (%.3f%%)", n_piv, 100 * n_piv / n)
    logger.info("  After volume spike    : %8d (%.3f%%)", n_vol, 100 * n_vol / n)
    logger.info("  After warmup skip     : %8d (%.3f%%)  ← FINAL", n_events, 100 * n_events / n)
    logger.info("=" * 70)

    if n_events == 0:
        raise RuntimeError("No events detected — relax thresholds (--pivot-distance, --volume-threshold)")

    # Build per-event records
    idx = np.where(mask.values)[0]
    direction = np.sign(pattern_score.values[idx]).astype("int8")

    # Names of fired patterns per event (string concat for traceability)
    pattern_values = pattern_block.values[idx]
    pattern_names = []
    for row in pattern_values:
        fired = [trigger_patterns[i] for i, v in enumerate(row) if v != 0]
        pattern_names.append("|".join(fired))

    out = pd.DataFrame({
        "timestamp": features["timestamp"].values[idx],
        "asset": features["asset"].values[idx],
        "feature_idx": idx.astype("int64"),
        "direction": direction,
        "pattern_score": pattern_score.values[idx].astype("int32"),
        "pattern_names": pattern_names,
        "open": features["open"].values[idx].astype("float32"),
        "high": features["high"].values[idx].astype("float32"),
        "low": features["low"].values[idx].astype("float32"),
        "close": features["close"].values[idx].astype("float32"),
        "atr_14": features["atr_14"].values[idx].astype("float32"),
        "dist_camarilla_nearest_norm": features["dist_camarilla_nearest_norm"].values[idx].astype("float32"),
        "volume_zscore_20p": features["volume_zscore_20p"].values[idx].astype("float32"),
    })
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_events(events: pd.DataFrame) -> None:
    n = len(events)
    long_count = int((events["direction"] > 0).sum())
    short_count = int((events["direction"] < 0).sum())
    long_pct = 100 * long_count / n if n else 0.0
    short_pct = 100 * short_count / n if n else 0.0

    logger.info("EVENTS SUMMARY")
    logger.info("=" * 70)
    logger.info("Total events     : %d", n)
    logger.info("Period           : %s → %s", events["timestamp"].min(), events["timestamp"].max())
    logger.info("Direction split  : LONG = %d (%.1f%%) | SHORT = %d (%.1f%%)",
                long_count, long_pct, short_count, short_pct)

    # Per-pattern contribution (a single event may fire multiple patterns)
    pattern_counts: dict[str, int] = {}
    for names in events["pattern_names"]:
        for p in names.split("|"):
            pattern_counts[p] = pattern_counts.get(p, 0) + 1
    logger.info("Per-pattern contribution (an event can fire multiple):")
    for p, c in sorted(pattern_counts.items(), key=lambda kv: -kv[1]):
        logger.info("  %-25s : %6d (%.1f%% of events)", p, c, 100 * c / n)

    # Yearly distribution
    years = pd.to_datetime(events["timestamp"]).dt.year
    by_year = years.value_counts().sort_index()
    logger.info("Events per year:")
    for year, count in by_year.items():
        logger.info("  %d : %5d", int(year), int(count))

    # Strength distribution (|pattern_score|)
    strength = events["pattern_score"].abs()
    logger.info("Pattern_score |abs| distribution:")
    logger.info("  min=%d  median=%d  mean=%.1f  max=%d",
                int(strength.min()), int(strength.median()), float(strength.mean()), int(strength.max()))
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=Path("data/patchtst_v5/features_btc.parquet"),
                   help="Input features parquet (default: data/patchtst_v5/features_btc.parquet)")
    p.add_argument("--output", type=Path, default=Path("data/patchtst_v5/events_btc.parquet"),
                   help="Output events parquet (default: data/patchtst_v5/events_btc.parquet)")
    p.add_argument("--patterns", type=str, default=",".join(DEFAULT_TRIGGER_PATTERNS),
                   help=f"Comma-separated TA-Lib pattern names (default: {','.join(DEFAULT_TRIGGER_PATTERNS)})")
    p.add_argument("--include-doji", action="store_true",
                   help="Add CDLDOJI to the trigger set (non-directional, increases events but adds noise)")
    p.add_argument("--pivot-distance", type=float, default=DEFAULT_PIVOT_DISTANCE,
                   help=f"|dist_camarilla_nearest_norm| threshold (default: {DEFAULT_PIVOT_DISTANCE} ATR)")
    p.add_argument("--volume-threshold", type=float, default=DEFAULT_VOLUME_THRESHOLD,
                   help=f"volume_zscore_20p > threshold (default: {DEFAULT_VOLUME_THRESHOLD})")
    p.add_argument("--warmup-bars", type=int, default=400,
                   help="Skip first N bars to avoid NaN warmup region (default: 400). "
                        "Doit couvrir vol_1h_zscore (288) + 96 bars lookback du dataset_builder "
                        "= 384 minimum pour garantir aucun NaN dans les fenêtres extraites.")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    logger.info("Loading features: %s", args.features)
    features = pd.read_parquet(args.features)
    missing = set(REQUIRED_FEATURE_COLS) - set(features.columns)
    if missing:
        raise ValueError(f"Required columns missing in features parquet: {missing}")

    trigger_patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]
    if args.include_doji and "CDLDOJI" not in trigger_patterns:
        trigger_patterns.append("CDLDOJI")
        logger.info("--include-doji enabled, adding CDLDOJI to trigger set")

    events = detect_events(
        features=features,
        trigger_patterns=trigger_patterns,
        pivot_distance=args.pivot_distance,
        volume_threshold=args.volume_threshold,
        warmup_bars=args.warmup_bars,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing events parquet: %s", args.output)
    events.to_parquet(args.output, compression="snappy", index=False)

    report_events(events)
    logger.info("Done. %d events written to %s", len(events), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
