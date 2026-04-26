"""
evaluate.py — Analyse fine du modèle PatchTST entraîné (étape 7 v5.0).

Charge predictions_test.npz généré par train.py (ou recharge le best model et
ré-évalue) puis produit:
  - Threshold sweep (precision/recall/F1/n_trades par seuil 0.30..0.90)
  - Top-K%% sweep (granularité fine 1, 2, 5, 10, 25, 50, 100%)
  - Calibration de la confiance (10 bins de probabilité prédite vs WR réel)
  - Per-segment breakdown (LONG/SHORT, par année, par pattern)
  - Comparaison vs baseline class

Usage:
    python -m experiments.patchtst_v5.evaluate \\
        --predictions models/patchtst_v5/predictions_test.npz \\
        --output models/patchtst_v5/evaluation_report.json

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

logger = logging.getLogger("patchtst_v5.evaluate")

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
TOP_K_PCTS = [1, 2, 5, 10, 25, 50, 100]
CALIBRATION_BINS = 10


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_predictions(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {
        "scores": data["scores"].astype("float64"),
        "y_true": data["y_true"].astype("int8"),
        "direction": data["direction"].astype("int8"),
        "timestamp": data["timestamp"],
        "pnl_after_fees_pct": data["pnl_after_fees_pct"].astype("float64"),
    }


def load_test_metadata(test_npz: Path | None) -> dict[str, np.ndarray] | None:
    """Optional: load extra metadata from test.npz (e.g., feature_idx)."""
    if test_npz is None or not test_npz.exists():
        return None
    data = np.load(test_npz, allow_pickle=False)
    return {"feature_idx": data["feature_idx"]}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def threshold_sweep(scores: np.ndarray, y: np.ndarray, pnl: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    for t in THRESHOLDS:
        mask = scores >= t
        n = int(mask.sum())
        if n == 0:
            rows.append({"threshold": t, "n_trades": 0, "wr": float("nan"),
                         "precision": float("nan"), "recall": float("nan"),
                         "f1": float("nan"), "mean_pnl_net": float("nan")})
            continue
        y_sel = y[mask]
        pnl_sel = pnl[mask]
        n_pos = int(y_sel.sum())
        precision = n_pos / n
        recall = n_pos / max(int(y.sum()), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        rows.append({
            "threshold": t,
            "n_trades": n,
            "wr": precision,            # alias for win rate
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_pnl_net": float(pnl_sel.mean()),
            "cumul_pnl_net": float(pnl_sel.sum()),
        })
    return rows


def top_k_sweep(scores: np.ndarray, y: np.ndarray, pnl: np.ndarray) -> list[dict]:
    n = len(scores)
    sorted_idx = np.argsort(-scores)
    rows: list[dict] = []
    for k in TOP_K_PCTS:
        n_top = max(1, int(n * k / 100))
        idx = sorted_idx[:n_top]
        wr = float(y[idx].mean())
        rows.append({
            "top_k_pct": k,
            "n_trades": n_top,
            "wr": wr,
            "mean_pnl_net": float(pnl[idx].mean()),
            "cumul_pnl_net": float(pnl[idx].sum()),
            "min_score": float(scores[idx].min()),
        })
    return rows


def calibration(scores: np.ndarray, y: np.ndarray, n_bins: int = CALIBRATION_BINS) -> list[dict]:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    rows: list[dict] = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin_lo": lo, "bin_hi": hi, "n": 0,
                         "predicted_mean": float("nan"), "actual_wr": float("nan")})
            continue
        rows.append({
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "n": n,
            "predicted_mean": float(scores[mask].mean()),
            "actual_wr": float(y[mask].mean()),
        })
    return rows


def per_segment(scores: np.ndarray, y: np.ndarray, direction: np.ndarray,
                timestamps: np.ndarray, threshold: float = 0.5) -> dict:
    out: dict = {}

    # By direction
    for label, mask in [("long", direction > 0), ("short", direction < 0)]:
        if not mask.any():
            continue
        sel_scores = scores[mask]
        sel_y = y[mask]
        out[f"by_direction_{label}"] = {
            "n_total": int(mask.sum()),
            "baseline_wr": float(sel_y.mean()),
            "wr_above_threshold": float(sel_y[sel_scores >= threshold].mean()) if (sel_scores >= threshold).any() else float("nan"),
            "n_above_threshold": int((sel_scores >= threshold).sum()),
        }

    # By year
    years = pd.to_datetime(timestamps).year
    by_year = []
    for year in sorted(np.unique(years)):
        mask = years == year
        sel_scores = scores[mask]
        sel_y = y[mask]
        by_year.append({
            "year": int(year),
            "n_total": int(mask.sum()),
            "baseline_wr": float(sel_y.mean()),
            "n_above_threshold": int((sel_scores >= threshold).sum()),
            "wr_above_threshold": float(sel_y[sel_scores >= threshold].mean()) if (sel_scores >= threshold).any() else float("nan"),
        })
    out["by_year"] = by_year
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(title: str, rows: list[dict], cols: list[str], fmt: dict[str, str]) -> None:
    if not rows:
        return
    logger.info("-" * 80)
    logger.info(title)
    logger.info("-" * 80)
    header = "  ".join(f"{c:>14}" for c in cols)
    logger.info(header)
    for r in rows:
        line = "  ".join(fmt.get(c, "{:>14}").format(r.get(c, "")) for c in cols)
        logger.info(line)
    logger.info("-" * 80)


def report(predictions: dict, output_path: Path | None) -> dict:
    scores = predictions["scores"]
    y = predictions["y_true"]
    direction = predictions["direction"]
    timestamps = predictions["timestamp"]
    pnl = predictions["pnl_after_fees_pct"]

    n = len(scores)
    baseline_wr = float(y.mean())
    logger.info("=" * 80)
    logger.info("EVALUATION REPORT — n_test=%d, baseline WR=%.4f", n, baseline_wr)
    logger.info("=" * 80)

    threshold_rows = threshold_sweep(scores, y, pnl)
    top_k_rows = top_k_sweep(scores, y, pnl)
    calib_rows = calibration(scores, y)
    seg = per_segment(scores, y, direction, timestamps)

    fmt_thr = {
        "threshold": "{:>14.2f}", "n_trades": "{:>14d}",
        "wr": "{:>14.4f}", "precision": "{:>14.4f}", "recall": "{:>14.4f}",
        "f1": "{:>14.4f}", "mean_pnl_net": "{:>14.4f}", "cumul_pnl_net": "{:>14.2f}",
    }
    print_table(
        "THRESHOLD SWEEP",
        threshold_rows,
        ["threshold", "n_trades", "wr", "precision", "recall", "f1", "mean_pnl_net", "cumul_pnl_net"],
        fmt_thr,
    )

    fmt_topk = {
        "top_k_pct": "{:>14d}", "n_trades": "{:>14d}", "wr": "{:>14.4f}",
        "mean_pnl_net": "{:>14.4f}", "cumul_pnl_net": "{:>14.2f}", "min_score": "{:>14.4f}",
    }
    print_table(
        "TOP-K%% SWEEP",
        top_k_rows,
        ["top_k_pct", "n_trades", "wr", "mean_pnl_net", "cumul_pnl_net", "min_score"],
        fmt_topk,
    )

    fmt_calib = {
        "bin_lo": "{:>14.2f}", "bin_hi": "{:>14.2f}", "n": "{:>14d}",
        "predicted_mean": "{:>14.4f}", "actual_wr": "{:>14.4f}",
    }
    print_table(
        "CALIBRATION (10 bins prob predicted vs actual WR)",
        calib_rows,
        ["bin_lo", "bin_hi", "n", "predicted_mean", "actual_wr"],
        fmt_calib,
    )

    logger.info("PER-SEGMENT (threshold=0.5)")
    for k, v in seg.items():
        if k.startswith("by_direction"):
            logger.info("  %-25s : n=%d baseline_wr=%.3f n_above=%d wr_above=%.3f",
                        k, v["n_total"], v["baseline_wr"], v["n_above_threshold"], v["wr_above_threshold"])
    logger.info("  by_year:")
    for row in seg["by_year"]:
        logger.info("    %d : n=%4d baseline_wr=%.3f  n_above=%4d  wr_above=%.3f",
                    row["year"], row["n_total"], row["baseline_wr"],
                    row["n_above_threshold"], row["wr_above_threshold"])
    logger.info("=" * 80)

    out = {
        "n_test": n,
        "baseline_wr": baseline_wr,
        "threshold_sweep": threshold_rows,
        "top_k_sweep": top_k_rows,
        "calibration": calib_rows,
        "per_segment": seg,
    }
    if output_path is not None:
        output_path.write_text(json.dumps(out, indent=2, default=str))
        logger.info("Report saved: %s", output_path)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, default=Path("models/patchtst_v5/predictions_test.npz"))
    p.add_argument("--output", type=Path, default=Path("models/patchtst_v5/evaluation_report.json"))
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    predictions = load_predictions(args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report(predictions, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
