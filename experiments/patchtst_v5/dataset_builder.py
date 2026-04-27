"""
dataset_builder.py — Construction du dataset PatchTST (étape 5 v5.0).

Pour chaque event labellisé, extrait la fenêtre [t-window:t] sur N channels
sélectionnés depuis le parquet de features. Split chronologique 70/15/15
(sans shuffle), avec purge optionnelle pour éviter le label leakage entre splits.

Output: 3 NPZ files (train/val/test) avec
  X: (n_events, window, n_channels) float32
  y: (n_events,) int8 — label binaire 0/1
  direction: (n_events,) int8 — +1 long / -1 short
  timestamp: (n_events,) datetime64[ns, UTC] — pour analyse temporelle
  pnl_after_fees_pct: (n_events,) float32 — pour backtest réaliste
  feature_idx: (n_events,) int64 — index de l'event dans features parquet

Channel sets disponibles via --patterns:
  top5 (default)  : 5 trigger patterns (ENGULFING, HAMMER, INVHAMMER, SHOOTSTAR, HANGMAN)
  directional10  : top5 + 5 autres reversal directionnels (MORNINGSTAR, EVENINGSTAR,
                    3WHITESOLDIERS, 3BLACKCROWS, PIERCING)
  all            : tous les CDL* du parquet (~61)
  none           : aucun pattern (juste 19 channels continus)
  custom (CSV)   : --patterns "CDLENGULFING,CDLDOJI,..."

Usage:
    python -m experiments.patchtst_v5.dataset_builder \\
        --features data/patchtst_v5/features_btc.parquet \\
        --labels data/patchtst_v5/labels_btc.parquet \\
        --output-dir data/patchtst_v5/

Voir STATUS_v5.0.md et experiments/patchtst_v5/README.md.
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

logger = logging.getLogger("patchtst_v5.dataset_builder")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEFAULT_WINDOW = 96  # 96 × 5min = 8h lookback (PatchTST plan: 8 patches × 12 bars)

# Channel presets — selectable via CLI --channel-preset

# v5.0 hybrid: bar shape + microstructure + levels + multi-TF + statistical
# v5.4 fix: features price-scale normalisées /close ; vol features z-scorées rolling
HYBRID_CHANNELS = [
    # Group A continuous (5)
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
    "close_location_value", "gap_norm",
    # Group B microstructure (5) — vols z-scorées (stationnaires)
    "corwin_schultz_spread_z", "garman_klass_vol_z", "yang_zhang_vol_z",
    "amihud_illiq", "volume_zscore_20p",
    # Group C levels (5)
    "dist_vwap_session_norm", "dist_camarilla_nearest_norm",
    "dist_poc_5d_norm", "dist_high_20p_norm", "dist_low_20p_norm",
    # Group D multi-TF (4) — slopes en % du prix (stationnaires)
    "trend_1h_slope_pct", "trend_4h_slope_pct", "vol_1h_zscore", "dist_open_daily_norm",
    # Group E statistical signatures (3)
    "permutation_entropy_50p", "hurst_dfa_100p", "pacf_lag5",
]  # 22 channels (v5.0 paradigm + drift fixes)

# v5.2 indicators-only: pure classical indicators TA-Lib + statistical
INDICATORS_ONLY_CHANNELS = [
    # Group I momentum multi-horizon (3)
    "rsi_7", "rsi_14", "rsi_21",
    # Group I MACD (2) — normalisés /close (% du prix)
    "macd_line_pct", "macd_signal_pct",
    # Group I CCI (1)
    "cci_20",
    # Group I Stochastic (2)
    "stoch_k_14", "stoch_d_14",
    # Group I Williams (1)
    "williams_r_14",
    # Group I Trend strength (3)
    "adx_14", "di_plus_14", "di_minus_14",
    # Group I Volatility (2) — atr z-scoré rolling
    "atr_14_norm_z", "bbands_pct_b_20",
    # Group I Volume (2) — obv slope z-scoré
    "obv_slope_z", "mfi_14",
    # Group E statistical (2 — Hurst/Entropy comme indicateurs avancés)
    "hurst_dfa_100p", "permutation_entropy_50p",
    # Group B volume z-score (1 — réutilisé depuis microstructure)
    "volume_zscore_20p",
]  # 19 channels (v5.2 paradigm + drift fixes)

CHANNEL_PRESETS = {
    "v5_hybrid": HYBRID_CHANNELS,
    "v5_indicators_only": INDICATORS_ONLY_CHANNELS,
}

# Default kept for backward compatibility (used if --channel-preset not specified)
CONTINUOUS_CHANNELS = HYBRID_CHANNELS

# Pattern channel presets
PATTERN_PRESETS: dict[str, list[str]] = {
    "top5": [
        "CDLENGULFING", "CDLHAMMER", "CDLINVERTEDHAMMER",
        "CDLSHOOTINGSTAR", "CDLHANGINGMAN",
    ],
    "directional10": [
        "CDLENGULFING", "CDLHAMMER", "CDLINVERTEDHAMMER",
        "CDLSHOOTINGSTAR", "CDLHANGINGMAN",
        "CDLMORNINGSTAR", "CDLEVENINGSTAR",
        "CDL3WHITESOLDIERS", "CDL3BLACKCROWS",
        "CDLPIERCING",
    ],
    "all": ["__all__"],   # marker — resolved at runtime from features columns
    "none": [],
}

DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15  # test ratio = 1 - train - val = 0.15
DEFAULT_PURGE_BARS = 24   # embargo entre splits (= time_barrier du labeler)


# ---------------------------------------------------------------------------
# Channel resolution
# ---------------------------------------------------------------------------

def resolve_pattern_channels(spec: str, features_cols: list[str]) -> list[str]:
    """Resolve --patterns argument into an explicit list of column names."""
    if spec in PATTERN_PRESETS:
        preset = PATTERN_PRESETS[spec]
        if preset == ["__all__"]:
            return [c for c in features_cols if c.startswith("CDL")]
        return preset
    # Custom comma-separated
    return [c.strip() for c in spec.split(",") if c.strip()]


def select_channels(features_cols: list[str], pattern_channels: list[str],
                    continuous_channels: list[str] | None = None) -> list[str]:
    """Build the final ordered list of channels (continuous first, then patterns)."""
    if continuous_channels is None:
        continuous_channels = CONTINUOUS_CHANNELS
    chosen: list[str] = []
    for col in continuous_channels:
        if col not in features_cols:
            raise ValueError(f"Required continuous channel missing in features: {col}")
        chosen.append(col)
    for col in pattern_channels:
        if col not in features_cols:
            raise ValueError(f"Pattern channel missing in features: {col}")
        chosen.append(col)
    return chosen


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_windows(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    channels: list[str],
    window: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Extract a (n_events, window, n_channels) tensor by walking back `window` bars
    from each event's feature_idx (inclusive of the event bar).

    Skips events whose lookback would dip below bar 0 or whose window contains NaN.
    Returns the tensor and a filtered labels DataFrame aligned to it.
    """
    feat_arr = features[channels].astype("float32").values  # (n_bars, n_channels)
    n_bars, n_channels = feat_arr.shape
    logger.info("Feature matrix: %d bars × %d channels", n_bars, n_channels)

    feature_idx_arr = labels["feature_idx"].values.astype("int64")
    n_events = len(labels)

    X = np.empty((n_events, window, n_channels), dtype="float32")
    valid = np.ones(n_events, dtype=bool)

    for k in range(n_events):
        idx = feature_idx_arr[k]
        start = idx - window + 1
        if start < 0:
            valid[k] = False
            continue
        block = feat_arr[start: idx + 1]
        if np.isnan(block).any():
            valid[k] = False
            continue
        X[k] = block

    n_dropped = int((~valid).sum())
    if n_dropped:
        logger.warning("%d events dropped (lookback < 0 or NaN in window)", n_dropped)
        X = X[valid]
        labels = labels[valid].reset_index(drop=True)

    logger.info("Final tensor shape: %s", X.shape)
    return X, labels


def chronological_split(
    n: int,
    train_ratio: float,
    val_ratio: float,
    purge_bars: int,
    feature_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronological split with purge embargo to avoid label leakage:
    each split's last event must be at least `purge_bars` bars before the next
    split's first event in the original feature timeline.
    """
    n_train_target = int(np.floor(n * train_ratio))
    n_val_target = int(np.floor(n * val_ratio))

    train_idx = np.arange(n_train_target)

    # Purge: skip val events whose feature_idx < (last train event feature_idx + purge_bars)
    last_train_fi = feature_idx[train_idx[-1]]
    val_start = n_train_target
    n_purged_train_val = 0
    while val_start < n and feature_idx[val_start] < last_train_fi + purge_bars:
        val_start += 1
        n_purged_train_val += 1

    val_end = min(val_start + n_val_target, n)
    val_idx = np.arange(val_start, val_end)

    if len(val_idx) == 0:
        raise RuntimeError("Validation split is empty after purge — relax --purge-bars or data too short")

    last_val_fi = feature_idx[val_idx[-1]]
    test_start = val_end
    n_purged_val_test = 0
    while test_start < n and feature_idx[test_start] < last_val_fi + purge_bars:
        test_start += 1
        n_purged_val_test += 1
    test_idx = np.arange(test_start, n)

    if len(test_idx) == 0:
        raise RuntimeError("Test split is empty after purge — relax --purge-bars or data too short")

    logger.info("Purge embargo (%d bars): %d events skipped train→val, %d events skipped val→test",
                purge_bars, n_purged_train_val, n_purged_val_test)
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_split(
    output_dir: Path,
    name: str,
    X: np.ndarray,
    labels: pd.DataFrame,
    channels: list[str],
    window: int,
) -> Path:
    """Write one split to NPZ + companion metadata JSON."""
    out_path = output_dir / f"{name}.npz"
    np.savez_compressed(
        out_path,
        X=X,
        y=labels["label"].values.astype("int8"),
        direction=labels["direction"].values.astype("int8"),
        timestamp=labels["timestamp"].values.astype("datetime64[ns]"),
        pnl_after_fees_pct=labels["pnl_after_fees_pct"].values.astype("float32"),
        feature_idx=labels["feature_idx"].values.astype("int64"),
    )
    logger.info("Saved %s: %s  (X=%s)", name, out_path, X.shape)
    return out_path


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_split(name: str, labels: pd.DataFrame) -> None:
    n = len(labels)
    n_pos = int((labels["label"] == 1).sum())
    pos_rate = 100 * n_pos / n if n else 0.0
    n_long = int((labels["direction"] > 0).sum())
    n_short = int((labels["direction"] < 0).sum())
    ts_min = pd.to_datetime(labels["timestamp"]).min()
    ts_max = pd.to_datetime(labels["timestamp"]).max()
    logger.info(
        "%-5s | n=%5d | Label=1 %.1f%% | LONG=%d SHORT=%d | %s → %s",
        name.upper(), n, pos_rate, n_long, n_short, ts_min, ts_max,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=Path("data/patchtst_v5/features_btc.parquet"))
    p.add_argument("--labels", type=Path, default=Path("data/patchtst_v5/labels_btc.parquet"))
    p.add_argument("--output-dir", type=Path, default=Path("data/patchtst_v5/"))
    p.add_argument("--channel-preset", type=str, default="v5_hybrid",
                   choices=list(CHANNEL_PRESETS.keys()),
                   help="Continuous channels preset: 'v5_hybrid' (22 ch v5.0 paradigm) "
                        "ou 'v5_indicators_only' (19 ch v5.2 paradigm — pure indicators TA-Lib + Hurst/Entropy/volume_zscore, NO bar shape, NO patterns)")
    p.add_argument("--patterns", type=str, default="top5",
                   help="Pattern channels: 'top5' | 'directional10' | 'all' | 'none' | comma-separated CDL* names "
                        "(default: top5; force to 'none' if --channel-preset=v5_indicators_only)")
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                   help=f"Lookback window in bars (default: {DEFAULT_WINDOW})")
    p.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    p.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    p.add_argument("--purge-bars", type=int, default=DEFAULT_PURGE_BARS,
                   help=f"Embargo bars between splits (default: {DEFAULT_PURGE_BARS} = time_barrier)")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    logger.info("Loading features: %s", args.features)
    features = pd.read_parquet(args.features)
    logger.info("Loading labels: %s", args.labels)
    labels = pd.read_parquet(args.labels)
    logger.info("Features: %d bars × %d cols   |   Labels: %d events",
                len(features), features.shape[1], len(labels))

    # Resolve channels
    continuous_channels = CHANNEL_PRESETS[args.channel_preset]
    if args.channel_preset == "v5_indicators_only" and args.patterns != "none":
        logger.info("--channel-preset=v5_indicators_only → forcing --patterns=none "
                    "(pure indicators paradigm, no candlestick patterns)")
        pattern_channels: list[str] = []
    else:
        pattern_channels = resolve_pattern_channels(args.patterns, list(features.columns))
    channels = select_channels(list(features.columns), pattern_channels, continuous_channels)
    logger.info("Channel preset: %s (%d continuous + %d patterns = %d total)",
                args.channel_preset, len(continuous_channels), len(pattern_channels), len(channels))
    logger.info("  Continuous: %s", continuous_channels)
    logger.info("  Patterns:   %s", pattern_channels if pattern_channels else "(none)")

    # Build (n_events, window, n_channels)
    X, labels = build_windows(features, labels, channels, args.window)
    n = len(X)

    # Chronological split with purge
    feature_idx_arr = labels["feature_idx"].values.astype("int64")
    train_idx, val_idx, test_idx = chronological_split(
        n, args.train_ratio, args.val_ratio, args.purge_bars, feature_idx_arr,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_split(args.output_dir, "train", X[train_idx], labels.iloc[train_idx].reset_index(drop=True),
               channels, args.window)
    save_split(args.output_dir, "val",   X[val_idx],   labels.iloc[val_idx].reset_index(drop=True),
               channels, args.window)
    save_split(args.output_dir, "test",  X[test_idx],  labels.iloc[test_idx].reset_index(drop=True),
               channels, args.window)

    # Companion metadata for the entire build (channel order, window, etc.)
    meta = {
        "window": args.window,
        "n_channels": len(channels),
        "channels": channels,
        "n_total": int(n),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "purge_bars": args.purge_bars,
        "channel_preset": args.channel_preset,
        "patterns_spec": args.patterns,
        "patterns_resolved": pattern_channels,
    }
    meta_path = args.output_dir / "dataset_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    logger.info("Metadata written: %s", meta_path)

    # Final report
    logger.info("=" * 70)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 70)
    report_split("train", labels.iloc[train_idx].reset_index(drop=True))
    report_split("val",   labels.iloc[val_idx].reset_index(drop=True))
    report_split("test",  labels.iloc[test_idx].reset_index(drop=True))
    logger.info("=" * 70)
    logger.info("Done. Run train.py next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
