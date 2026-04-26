"""
diagnose_drift.py — Analyse détaillée du distribution shift entre train/val/test.

Objectif: identifier QUELLES features ont changé entre 2017-2023 (train) et
2024-2026 (test), pour comprendre pourquoi le modèle hybrid (run 7) atteint
AUC 0.74 sur train mais s'effondre à 0.51 sur test.

Méthode:
  1. Per-channel statistics (mean, std, percentiles) sur chaque split
  2. KS test (Kolmogorov-Smirnov) train vs test pour chaque channel
  3. Per-percentile shift table pour les pires drifters
  4. Per-year statistics si timestamp disponible
  5. Class balance par split (déjà connu mais redocumenté)

Usage:
    python -m experiments.patchtst_v5.diagnose_drift \\
        --data-dir data/patchtst_v5_rr2_hybrid/

Note: la valeur "à l'event" = X[:, -1, :] (dernière bar de la fenêtre 96).
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
from scipy import stats

logger = logging.getLogger("patchtst_v5.diagnose_drift")


def load_split(npz_path: Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=False)
    return {
        "X": data["X"].astype("float32"),
        "y": data["y"].astype("int8"),
        "timestamp": data["timestamp"],
    }


def per_channel_stats(values: np.ndarray) -> dict:
    """Calcule mean/std/percentiles d'une série 1D."""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p01": float(np.percentile(values, 1)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p99": float(np.percentile(values, 99)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def drift_analysis(splits: dict[str, dict], channel_names: list[str]) -> pd.DataFrame:
    """Calcule KS test + statistiques pour chaque channel × split."""
    train_X = splits["train"]["X"][:, -1, :]  # values à l'event time
    val_X = splits["val"]["X"][:, -1, :]
    test_X = splits["test"]["X"][:, -1, :]

    rows = []
    for c, name in enumerate(channel_names):
        train_vals = train_X[:, c]
        val_vals = val_X[:, c]
        test_vals = test_X[:, c]

        # Skip channels with constant values (no variance)
        if train_vals.std() < 1e-9 or test_vals.std() < 1e-9:
            continue

        # KS test train vs test
        ks_stat, ks_pvalue = stats.ks_2samp(train_vals, test_vals)
        ks_train_val_stat, _ = stats.ks_2samp(train_vals, val_vals)

        # Mean shift normalisé par std train
        mean_shift_norm = (test_vals.mean() - train_vals.mean()) / max(train_vals.std(), 1e-9)
        std_ratio = test_vals.std() / max(train_vals.std(), 1e-9)

        rows.append({
            "channel": name,
            "train_mean": train_vals.mean(),
            "test_mean": test_vals.mean(),
            "mean_shift_norm": mean_shift_norm,
            "train_std": train_vals.std(),
            "test_std": test_vals.std(),
            "std_ratio": std_ratio,
            "ks_train_test": ks_stat,
            "ks_train_val": ks_train_val_stat,
            "ks_pvalue": ks_pvalue,
        })

    df = pd.DataFrame(rows).sort_values("ks_train_test", ascending=False)
    return df


def detailed_top_drifters(
    splits: dict[str, dict],
    channel_names: list[str],
    drift_df: pd.DataFrame,
    top_n: int = 5,
) -> dict:
    """Pour les top N drifters, sortie percentile-by-percentile."""
    train_X = splits["train"]["X"][:, -1, :]
    test_X = splits["test"]["X"][:, -1, :]

    out = {}
    for _, row in drift_df.head(top_n).iterrows():
        name = row["channel"]
        c = channel_names.index(name)
        train_vals = train_X[:, c]
        test_vals = test_X[:, c]

        percentiles = [1, 5, 25, 50, 75, 95, 99]
        train_pct = np.percentile(train_vals, percentiles)
        test_pct = np.percentile(test_vals, percentiles)
        out[name] = {
            "ks_stat": float(row["ks_train_test"]),
            "percentiles": percentiles,
            "train": [float(v) for v in train_pct],
            "test": [float(v) for v in test_pct],
            "shifts": [float(t - tr) for tr, t in zip(train_pct, test_pct)],
        }
    return out


def class_balance_per_year(splits: dict[str, dict]) -> pd.DataFrame:
    """Class balance et n_events par année pour validation drift labels."""
    rows = []
    for split_name, data in splits.items():
        y = data["y"]
        ts = pd.to_datetime(data["timestamp"])
        years = ts.year
        for year in sorted(np.unique(years)):
            mask = years == year
            n = int(mask.sum())
            if n == 0:
                continue
            n_pos = int(y[mask].sum())
            rows.append({
                "split": split_name,
                "year": int(year),
                "n_events": n,
                "class_1_pct": 100 * n_pos / n,
            })
    return pd.DataFrame(rows)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Directory with train.npz, val.npz, test.npz")
    p.add_argument("--metadata", type=Path, default=None,
                   help="dataset_metadata.json (default: <data-dir>/dataset_metadata.json)")
    p.add_argument("--top-n", type=int, default=5, help="Top N drifters à détailler")
    p.add_argument("--output", type=Path, default=None,
                   help="Output JSON file (default: <data-dir>/drift_diagnostic.json)")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    metadata_path = args.metadata or (args.data_dir / "dataset_metadata.json")
    if not metadata_path.exists():
        logger.error("Metadata file not found: %s", metadata_path)
        return 1
    metadata = json.loads(metadata_path.read_text())
    channel_names = metadata["channels"]
    logger.info("Loaded %d channels from metadata", len(channel_names))

    splits = {}
    for name in ("train", "val", "test"):
        npz = args.data_dir / f"{name}.npz"
        if not npz.exists():
            logger.error("Missing %s", npz)
            return 1
        splits[name] = load_split(npz)
        logger.info("Loaded %s: %s events", name, len(splits[name]["y"]))

    # 1. Drift analysis
    logger.info("Running KS test + statistics per channel ...")
    drift_df = drift_analysis(splits, channel_names)

    # Display sorted by KS
    logger.info("=" * 110)
    logger.info("DRIFT ANALYSIS — sorted by KS train→test (highest = most drifted)")
    logger.info("=" * 110)
    logger.info(
        "%-30s %12s %12s %12s %12s %12s %12s",
        "Channel", "Train_mean", "Test_mean", "MeanShift", "Train_std", "Test_std", "KS",
    )
    logger.info("-" * 110)
    for _, row in drift_df.iterrows():
        marker = " 🚨" if row["ks_train_test"] > 0.20 else (" ⚠️" if row["ks_train_test"] > 0.10 else "")
        logger.info(
            "%-30s %12.4f %12.4f %+12.3f %12.4f %12.4f %12.4f%s",
            row["channel"], row["train_mean"], row["test_mean"],
            row["mean_shift_norm"], row["train_std"], row["test_std"],
            row["ks_train_test"], marker,
        )
    logger.info("=" * 110)

    # 2. Top drifters detailed
    top = detailed_top_drifters(splits, channel_names, drift_df, top_n=args.top_n)
    logger.info("TOP %d DRIFTERS — percentile-by-percentile shift", args.top_n)
    logger.info("=" * 110)
    for name, info in top.items():
        logger.info("Channel: %s   (KS = %.4f)", name, info["ks_stat"])
        logger.info("  %-9s %12s %12s %12s", "Pctile", "Train", "Test", "Shift")
        for p, tr, te, sh in zip(info["percentiles"], info["train"], info["test"], info["shifts"]):
            logger.info("  P%-8d %12.4f %12.4f %+12.4f", p, tr, te, sh)
        logger.info("")
    logger.info("=" * 110)

    # 3. Class balance per year
    cb_df = class_balance_per_year(splits)
    logger.info("CLASS BALANCE PER YEAR")
    logger.info("=" * 110)
    logger.info("%-7s %6s %12s %12s", "Split", "Year", "n_events", "Class=1 %")
    for _, row in cb_df.iterrows():
        logger.info("%-7s %6d %12d %12.2f", row["split"], row["year"], row["n_events"], row["class_1_pct"])
    logger.info("=" * 110)

    # 4. Synthèse
    n_critical = (drift_df["ks_train_test"] > 0.20).sum()
    n_warning = ((drift_df["ks_train_test"] > 0.10) & (drift_df["ks_train_test"] <= 0.20)).sum()
    n_clean = (drift_df["ks_train_test"] <= 0.10).sum()
    logger.info("SYNTHESE")
    logger.info("  Channels avec drift critique  (KS > 0.20) : %d", n_critical)
    logger.info("  Channels avec drift modéré    (KS 0.10-0.20) : %d", n_warning)
    logger.info("  Channels stables              (KS ≤ 0.10) : %d", n_clean)

    # Output JSON
    output_path = args.output or (args.data_dir / "drift_diagnostic.json")
    output_data = {
        "drift_table": drift_df.to_dict(orient="records"),
        "top_drifters_detail": top,
        "class_balance_per_year": cb_df.to_dict(orient="records"),
        "synthesis": {
            "n_critical": int(n_critical),
            "n_warning": int(n_warning),
            "n_clean": int(n_clean),
            "n_total": int(len(drift_df)),
        },
    }
    output_path.write_text(json.dumps(output_data, indent=2, default=str))
    logger.info("JSON saved: %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
