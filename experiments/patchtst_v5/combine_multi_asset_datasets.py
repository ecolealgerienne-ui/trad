"""
combine_multi_asset_datasets.py — Combine per-asset datasets into a single
multi-asset training/val/test set.

Each input dataset directory must contain train.npz, val.npz, test.npz with
the same channel set (features must be asset-agnostic). The script verifies
channel consistency, concatenates events, optionally adds an asset_id
column, and writes a combined dataset_metadata.json.

Chronological coherence: each per-asset dataset has its own
chronological 70/15/15 split. The combined train/val/test is the union
of per-asset splits, preserving causality within each asset (no future
leakage). Across assets, dates may overlap but events are independent.

Usage:
    python -m experiments.patchtst_v5.combine_multi_asset_datasets \\
        --input-dirs data/patchtst_v5_pivot_sl4_btc/ \\
                     data/patchtst_v5_pivot_sl4_eth/ \\
                     data/patchtst_v5_pivot_sl4_bnb/ \\
                     data/patchtst_v5_pivot_sl4_ada/ \\
                     data/patchtst_v5_pivot_sl4_ltc/ \\
        --asset-names BTC ETH BNB ADA LTC \\
        --output-dir data/patchtst_v5_pivot_sl4_multi/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

logger = logging.getLogger("patchtst_v5.combine_multi_asset")


def load_metadata(path: Path) -> dict:
    return json.loads(path.read_text())


def load_split_npz(path: Path) -> dict:
    """Load all arrays from a split NPZ as a dict (no hardcoded keys)."""
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def combine_split(dirs: list[Path], asset_ids: list[int],
                   asset_names: list[str], split: str,
                   out_path: Path) -> dict:
    """Concatenate NPZ arrays from multiple dirs for a single split."""
    arrays_by_key: dict = {}
    asset_id_per_event: list[np.ndarray] = []
    n_per_asset: list[int] = []

    for d, aid, aname in zip(dirs, asset_ids, asset_names):
        npz_path = d / f"{split}.npz"
        if not npz_path.exists():
            raise SystemExit(f"Missing {npz_path}")
        data = load_split_npz(npz_path)
        n = len(next(iter(data.values())))
        n_per_asset.append(n)
        for k, v in data.items():
            arrays_by_key.setdefault(k, []).append(v)
        asset_id_per_event.append(np.full(n, aid, dtype="int8"))
        logger.info("  %s %s : %d events", aname, split, n)

    combined = {k: np.concatenate(v, axis=0) for k, v in arrays_by_key.items()}
    combined["asset_id"] = np.concatenate(asset_id_per_event)

    # Sort all arrays by timestamp to restore chronological order across assets
    if "timestamp" in combined:
        order = np.argsort(combined["timestamp"])
        combined = {k: v[order] for k, v in combined.items()}

    np.savez_compressed(out_path, **combined)
    logger.info("  saved %s : n_total=%d (combined)", out_path, len(combined["asset_id"]))
    return {
        "n_total": int(len(combined["asset_id"])),
        "n_per_asset": dict(zip(asset_names, n_per_asset)),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dirs", type=Path, nargs="+", required=True,
                   help="Per-asset dataset directories (each must contain train/val/test.npz)")
    p.add_argument("--asset-names", type=str, nargs="+", required=True,
                   help="Asset names matching --input-dirs (e.g., BTC ETH BNB)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    if len(args.input_dirs) != len(args.asset_names):
        raise SystemExit("--input-dirs and --asset-names must have the same length")

    # Load and validate metadata across all assets
    metas: list[dict] = []
    for d, name in zip(args.input_dirs, args.asset_names):
        meta_path = d / "dataset_metadata.json"
        if not meta_path.exists():
            raise SystemExit(f"Missing {meta_path}")
        m = load_metadata(meta_path)
        metas.append(m)
        logger.info("Loaded %s: window=%d n_channels=%d n_total=%d",
                    name, m["window"], m["n_channels"], m["n_total"])

    # Verify channel consistency
    ref_channels = metas[0]["channels"]
    ref_window = metas[0]["window"]
    for m, name in zip(metas[1:], args.asset_names[1:]):
        if m["channels"] != ref_channels:
            raise SystemExit(f"Channel mismatch between {args.asset_names[0]} and {name}")
        if m["window"] != ref_window:
            raise SystemExit(f"Window mismatch between {args.asset_names[0]} and {name}")
    logger.info("All %d assets share the same %d channels and window=%d",
                len(args.asset_names), len(ref_channels), ref_window)

    # asset_id encoding
    asset_ids = list(range(len(args.asset_names)))
    asset_id_map = dict(zip(args.asset_names, asset_ids))
    logger.info("asset_id encoding: %s", asset_id_map)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Combine each split
    summary: dict = {"per_split": {}}
    for split in ("train", "val", "test"):
        logger.info("Combining %s ...", split)
        s = combine_split(
            args.input_dirs, asset_ids, args.asset_names,
            split, args.output_dir / f"{split}.npz"
        )
        summary["per_split"][split] = s

    # Combined metadata
    combined_meta = {
        "window": ref_window,
        "n_channels": len(ref_channels),
        "channels": ref_channels,
        "asset_id_map": asset_id_map,
        "n_total": sum(summary["per_split"][s]["n_total"] for s in ("train", "val", "test")),
        "n_train": summary["per_split"]["train"]["n_total"],
        "n_val": summary["per_split"]["val"]["n_total"],
        "n_test": summary["per_split"]["test"]["n_total"],
        "per_asset_per_split": {s: summary["per_split"][s]["n_per_asset"]
                                 for s in ("train", "val", "test")},
        "purge_bars": metas[0].get("purge_bars"),
        "channel_preset": metas[0].get("channel_preset"),
        "patterns_spec": metas[0].get("patterns_spec"),
        "patterns_resolved": metas[0].get("patterns_resolved"),
        "source_dirs": [str(d) for d in args.input_dirs],
    }
    meta_path = args.output_dir / "dataset_metadata.json"
    meta_path.write_text(json.dumps(combined_meta, indent=2, default=str))
    logger.info("Combined metadata: %s", meta_path)

    # Final report
    logger.info("=" * 70)
    logger.info("MULTI-ASSET DATASET SUMMARY")
    logger.info("=" * 70)
    logger.info("%-5s | %5s | %5s | %5s | %s",
                "Split", "Train", "Val", "Test", "Per asset (train)")
    for split in ("train", "val", "test"):
        s = summary["per_split"][split]
        per_asset = " ".join(f"{a}:{n}" for a, n in s["n_per_asset"].items())
        logger.info("%-5s | %5d | %s", split.upper(), s["n_total"], per_asset)
    logger.info("=" * 70)
    logger.info("Combined total: train=%d val=%d test=%d",
                combined_meta["n_train"], combined_meta["n_val"], combined_meta["n_test"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
