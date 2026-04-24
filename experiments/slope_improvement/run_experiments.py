"""
Orchestrator for slope improvement experiments.

Runs Étape 0 (setup + baseline reproduction) and Étape 1 (innovation
diagnostic) end-to-end and writes:

  artifacts/
    meta.json                       # split metadata, configs
    ground_truth_slope_rts.npy      # primary GT on full series (PRE-split)
    ground_truth_slope_ma.npy       # secondary GT on full series
    baseline_slope_train.npy        # KF slope estimates per split
    baseline_slope_val.npy
    baseline_slope_test.npy
    baseline_innovations_train.npy  # raw v_t and S_t (for diagnostic reruns)
    baseline_innov_S_train.npy
    baseline_sigma2_train.npy       # adaptation trace
    baseline_metrics.json           # 5 metrics on val + test (vs both GTs)
    step1_diagnostic.json           # innovation diagnostic on TRAIN
    step1_plots/                    # histogram, ACF, QQ
    report_step1.md                 # human-readable gate report

Usage
-----
    # From project root:
    python experiments/slope_improvement/run_experiments.py \
        --csv data_trad/BTCUSD_all_5m.csv \
        --start-date 2022-01-01

The script prints a SUMMARY at the end indicating the gate verdict:
    EXPLOITABLE | MARGINAL | WHITE_NOISE
Claude will read that output to decide whether to proceed to Étape 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np

# Make sibling modules importable when run as a script.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_loader import make_splits, SplitData   # noqa: E402
from ground_truth import compute_full_ground_truth, GroundTruth  # noqa: E402
from kf_baseline import run_kf_baseline, KFBaselineResult  # noqa: E402
from metrics import compute_all, SlopeMetrics  # noqa: E402
from diagnostics import run_diagnostic, make_plots  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, default=str)


def _metrics_both_gt(
    slope_est: np.ndarray,
    gt_split: GroundTruth,
) -> Dict[str, dict]:
    m_rts = compute_all(slope_est, gt_split.slope_rts)
    m_ma = compute_all(slope_est, gt_split.slope_ma)
    return {
        "vs_rts": m_rts.to_dict(),
        "vs_ma": m_ma.to_dict(),
    }


# ---------------------------------------------------------------------------
# Étape 0 + Étape 1
# ---------------------------------------------------------------------------

def run(
    csv_path: str,
    start_date: str,
    end_date: str | None,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "step1_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. Load splits ----------------
    print("=" * 70)
    print("ÉTAPE 0.1 — Chargement & splits")
    print("=" * 70)
    train, val, test, split_meta = make_splits(
        csv_path=csv_path, start_date=start_date, end_date=end_date,
    )
    print(json.dumps(split_meta, indent=2, default=str))

    # Reconstruct full RSI series (pre-split) for ground truth computation
    rsi_full = np.concatenate([train.rsi, val.rsi, test.rsi])
    n_full = len(rsi_full)
    assert n_full == split_meta["n_total"]

    # ---------------- 2. Ground truth on FULL series, once ----------------
    print("\n" + "=" * 70)
    print("ÉTAPE 0.2 — Ground truth RTS (full-pass, non-causal) + MA centrée")
    print("=" * 70)
    gt_full = compute_full_ground_truth(rsi_full)
    np.save(out_dir / "ground_truth_slope_rts.npy", gt_full.slope_rts)
    np.save(out_dir / "ground_truth_slope_ma.npy", gt_full.slope_ma)
    print(f"  RTS slope: mean={gt_full.slope_rts.mean():.6f}  std={gt_full.slope_rts.std():.6f}")
    m_ma_valid = np.isfinite(gt_full.slope_ma).sum()
    print(f"  MA slope : mean={np.nanmean(gt_full.slope_ma):.6f}  std={np.nanstd(gt_full.slope_ma):.6f}  (n_valid={m_ma_valid})")

    # Split GT by index
    gt_train = gt_full.split(train.idx_start, train.idx_end)
    gt_val = gt_full.split(val.idx_start, val.idx_end)
    gt_test = gt_full.split(test.idx_start, test.idx_end)

    # ---------------- 3. Run baseline KF on each split ----------------
    # IMPORTANT: Running KF INDEPENDENTLY per split would give slightly
    # different warmup behavior at the boundaries. For a fair and realistic
    # causal baseline, we run it on the FULL RSI series (still causal), then
    # slice. This also matches prod behavior where the filter has seen all
    # history up to `now`.
    print("\n" + "=" * 70)
    print("ÉTAPE 1.1 — KF baseline (2D CV, σ² adaptatif scalaire)")
    print("=" * 70)
    kf_full = run_kf_baseline(rsi_full)
    print(f"  σ² init={kf_full.sigma2_trace[np.isfinite(kf_full.sigma2_trace)][0]:.6g}")
    print(f"  σ² final={kf_full.sigma2_trace[-1]:.6g}")
    print(f"  σ² range=[{np.nanmin(kf_full.sigma2_trace):.6g}, {np.nanmax(kf_full.sigma2_trace):.6g}]")

    def _slice(r: KFBaselineResult, a: int, b: int) -> KFBaselineResult:
        return KFBaselineResult(
            level=r.level[a:b].copy(),
            slope=r.slope[a:b].copy(),
            innovations=r.innovations[a:b].copy(),
            S=r.S[a:b].copy(),
            sigma2_trace=r.sigma2_trace[a:b].copy(),
            P_diag=r.P_diag[a:b].copy(),
        )

    kf_train = _slice(kf_full, train.idx_start, train.idx_end)
    kf_val = _slice(kf_full, val.idx_start, val.idx_end)
    kf_test = _slice(kf_full, test.idx_start, test.idx_end)

    # Save slope per split
    np.save(out_dir / "baseline_slope_train.npy", kf_train.slope)
    np.save(out_dir / "baseline_slope_val.npy", kf_val.slope)
    np.save(out_dir / "baseline_slope_test.npy", kf_test.slope)
    # Save innovations/S for train (needed for step 1 diagnostic)
    np.save(out_dir / "baseline_innovations_train.npy", kf_train.innovations)
    np.save(out_dir / "baseline_innov_S_train.npy", kf_train.S)
    np.save(out_dir / "baseline_sigma2_train.npy", kf_train.sigma2_trace)

    # ---------------- 4. Metrics on val + test (both GTs) ----------------
    print("\n" + "=" * 70)
    print("ÉTAPE 1.2 — Metrics baseline (val + test, vs RTS & MA)")
    print("=" * 70)
    metrics_val = _metrics_both_gt(kf_val.slope, gt_val)
    metrics_test = _metrics_both_gt(kf_test.slope, gt_test)
    metrics_all = {"val": metrics_val, "test": metrics_test}
    _dump_json(out_dir / "baseline_metrics.json", metrics_all)

    def _print_metric_block(label: str, m: Dict[str, dict]) -> None:
        print(f"  [{label}]")
        for gt_name, md in m.items():
            print(
                f"    {gt_name:8s}  MSE={md['mse']:.6g}  MAE={md['mae']:.6g}  "
                f"Pearson={md['pearson']:.4f}  DirMatch={md['direction_match']:.4f}  "
                f"Lag={md['latency_bars']:+.1f}  n={md['n_valid']}"
            )
    _print_metric_block("val ", metrics_val)
    _print_metric_block("test", metrics_test)

    # ---------------- 5. Innovation diagnostic on TRAIN ----------------
    print("\n" + "=" * 70)
    print("ÉTAPE 1.3 — Diagnostic innovations (TRAIN) → GATE")
    print("=" * 70)
    diag = run_diagnostic(kf_train.innovations, kf_train.S, split_name="train")
    _dump_json(out_dir / "step1_diagnostic.json", diag.to_dict())

    print(f"  n (innovations valides) = {diag.n}")
    print(f"  mean z = {diag.mean:.4f}   std z = {diag.std:.4f}   (attendu ≈ 0, 1)")
    print(f"  ACF(1..10) = [" + ", ".join(f"{v:+.4f}" for v in diag.acf_1_to_10) + "]")
    print(f"  max|ACF(1..10)| = {diag.acf_max_abs_1_10:.4f}")
    print(f"  Ljung-Box h=10 : Q={diag.ljung_box_h10['statistic']:.2f}  p={diag.ljung_box_h10['p_value']:.3e}  Q/n={diag.ljung_box_h10['q_per_n']:.4f}")
    print(f"  Ljung-Box h=20 : Q={diag.ljung_box_h20['statistic']:.2f}  p={diag.ljung_box_h20['p_value']:.3e}  Q/n={diag.ljung_box_h20['q_per_n']:.4f}")
    print(f"  Jarque-Bera    : stat={diag.jarque_bera['statistic']:.2f}  p={diag.jarque_bera['p_value']:.3e}  skew={diag.jarque_bera['skewness']:.3f}  excess_kurt={diag.jarque_bera['excess_kurtosis']:.3f}")
    print()
    print(f"  GATE VERDICT : {diag.gate_verdict}")
    print(f"  REASON       : {diag.gate_reason}")

    # Plots
    plot_paths = make_plots(kf_train.innovations, kf_train.S, plots_dir, prefix="baseline_train")
    if plot_paths is None:
        print("  [matplotlib indisponible — plots omis]")
    else:
        print(f"  Plots sauvegardés : {', '.join(p.name for p in plot_paths)}")

    # ---------------- 6. Write markdown report ----------------
    report_path = out_dir / "report_step1.md"
    _write_report(
        report_path=report_path,
        split_meta=split_meta,
        metrics_val=metrics_val,
        metrics_test=metrics_test,
        diag=diag,
        sigma2_trace=kf_train.sigma2_trace,
        plot_paths=plot_paths,
    )
    print(f"\nRapport écrit : {report_path}")

    # ---------------- 7. Persist meta ----------------
    meta_out = {
        "split": split_meta,
        "gate_verdict": diag.gate_verdict,
        "gate_reason": diag.gate_reason,
        "metrics_val": metrics_val,
        "metrics_test": metrics_test,
        "artifacts": {
            "ground_truth_slope_rts": "ground_truth_slope_rts.npy",
            "ground_truth_slope_ma": "ground_truth_slope_ma.npy",
            "baseline_slope_train": "baseline_slope_train.npy",
            "baseline_slope_val": "baseline_slope_val.npy",
            "baseline_slope_test": "baseline_slope_test.npy",
            "baseline_innovations_train": "baseline_innovations_train.npy",
            "baseline_innov_S_train": "baseline_innov_S_train.npy",
            "baseline_sigma2_train": "baseline_sigma2_train.npy",
            "step1_diagnostic": "step1_diagnostic.json",
            "report_step1": "report_step1.md",
        },
    }
    _dump_json(out_dir / "meta.json", meta_out)

    # ---------------- 8. Print final summary ----------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Verdict GATE  : {diag.gate_verdict}")
    print(f"  Baseline VAL  (vs RTS): MSE={metrics_val['vs_rts']['mse']:.6g}  Pearson={metrics_val['vs_rts']['pearson']:.4f}  DirMatch={metrics_val['vs_rts']['direction_match']:.4f}")
    print(f"  Baseline TEST (vs RTS): MSE={metrics_test['vs_rts']['mse']:.6g}  Pearson={metrics_test['vs_rts']['pearson']:.4f}  DirMatch={metrics_test['vs_rts']['direction_match']:.4f}")
    print(f"  Baseline VAL  (vs MA) : MSE={metrics_val['vs_ma']['mse']:.6g}  Pearson={metrics_val['vs_ma']['pearson']:.4f}  DirMatch={metrics_val['vs_ma']['direction_match']:.4f}")
    print(f"  Baseline TEST (vs MA) : MSE={metrics_test['vs_ma']['mse']:.6g}  Pearson={metrics_test['vs_ma']['pearson']:.4f}  DirMatch={metrics_test['vs_ma']['direction_match']:.4f}")
    print("=" * 70)

    return meta_out


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt(x: float, sig: int = 6) -> str:
    if not np.isfinite(x):
        return "NaN"
    return f"{x:.{sig}g}"


def _write_report(
    report_path: Path,
    split_meta: dict,
    metrics_val: Dict[str, dict],
    metrics_test: Dict[str, dict],
    diag,
    sigma2_trace: np.ndarray,
    plot_paths,
) -> None:
    sigma2_valid = sigma2_trace[np.isfinite(sigma2_trace)]
    lines = []
    lines.append("# Étape 1 — Rapport baseline + diagnostic d'innovations")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Dataset : `{split_meta['csv_path']}`")
    lines.append(f"- Période : {split_meta['start_date']} → {split_meta['end_date'] or 'fin'}")
    lines.append(f"- RSI period : {split_meta['rsi_period']}")
    lines.append(f"- N total : {split_meta['n_total']:,}")
    lines.append(f"- Split chronologique 50/25/25 :")
    lines.append(f"  - train : {split_meta['n_train']:,} ({split_meta['train_start']} → {split_meta['train_end']})")
    lines.append(f"  - val   : {split_meta['n_val']:,} ({split_meta['val_start']} → {split_meta['val_end']})")
    lines.append(f"  - test  : {split_meta['n_test']:,} ({split_meta['test_start']} → {split_meta['test_end']})")
    lines.append("")
    lines.append("## Modèle baseline")
    lines.append("")
    lines.append("- État x = [level, velocity] — dim 2")
    lines.append("- F = [[1, 1], [0, 1]], H = [[1, 0]]")
    lines.append("- Q = σ_vel² · G·G^T,  G = [1, 1]^T  (rank-1, adaptation scalaire)")
    lines.append("- R = 0.1 (fixe)")
    lines.append(f"- σ² : range empirique [{_fmt(sigma2_valid.min(), 4)}, {_fmt(sigma2_valid.max(), 4)}], final = {_fmt(sigma2_valid[-1], 4)}")
    lines.append("- Adaptation : Myers-Tapley scalaire, fenêtre = 30 innovations, clip [σ²_init·0.1, σ²_init·10]")
    lines.append("")
    lines.append("## Metrics (val & test, vs 2 ground truths)")
    lines.append("")
    lines.append("### vs RTS full-pass (primary)")
    lines.append("")
    lines.append("| Split | MSE | MAE | Pearson | DirMatch | Latency | n |")
    lines.append("|-------|-----|-----|---------|----------|---------|---|")
    for name, m in [("val", metrics_val["vs_rts"]), ("test", metrics_test["vs_rts"])]:
        lines.append(
            f"| {name} | {_fmt(m['mse'])} | {_fmt(m['mae'])} | {_fmt(m['pearson'], 4)} | "
            f"{_fmt(m['direction_match'], 4)} | {_fmt(m['latency_bars'], 2)} | {m['n_valid']:,} |"
        )
    lines.append("")
    lines.append("### vs MA centrée window=21 (secondary)")
    lines.append("")
    lines.append("| Split | MSE | MAE | Pearson | DirMatch | Latency | n |")
    lines.append("|-------|-----|-----|---------|----------|---------|---|")
    for name, m in [("val", metrics_val["vs_ma"]), ("test", metrics_test["vs_ma"])]:
        lines.append(
            f"| {name} | {_fmt(m['mse'])} | {_fmt(m['mae'])} | {_fmt(m['pearson'], 4)} | "
            f"{_fmt(m['direction_match'], 4)} | {_fmt(m['latency_bars'], 2)} | {m['n_valid']:,} |"
        )
    lines.append("")
    lines.append("## Diagnostic d'innovations (TRAIN)")
    lines.append("")
    lines.append(f"- n innovations valides : {diag.n:,}")
    lines.append(f"- mean(z) = {diag.mean:.4f}  (attendu ≈ 0)")
    lines.append(f"- std(z)  = {diag.std:.4f}  (attendu ≈ 1)")
    lines.append("")
    lines.append("### ACF(1..10) des innovations normalisées")
    lines.append("")
    lines.append("| Lag | ACF |")
    lines.append("|-----|-----|")
    for k, v in enumerate(diag.acf_1_to_10, start=1):
        lines.append(f"| {k} | {v:+.4f} |")
    lines.append("")
    lines.append(f"- **max|ACF(1..10)| = {diag.acf_max_abs_1_10:.4f}**")
    lines.append("")
    lines.append("### Tests statistiques")
    lines.append("")
    lines.append("| Test | Statistique | p-value | Métrique normalisée |")
    lines.append("|------|-------------|---------|----------------------|")
    lb10 = diag.ljung_box_h10
    lb20 = diag.ljung_box_h20
    jb = diag.jarque_bera
    lines.append(f"| Ljung-Box h=10 | {lb10['statistic']:.2f} | {lb10['p_value']:.3e} | Q/n = {lb10['q_per_n']:.4f} |")
    lines.append(f"| Ljung-Box h=20 | {lb20['statistic']:.2f} | {lb20['p_value']:.3e} | Q/n = {lb20['q_per_n']:.4f} |")
    lines.append(f"| Jarque-Bera    | {jb['statistic']:.2f} | {jb['p_value']:.3e} | skew={jb['skewness']:.3f}, excess_kurt={jb['excess_kurtosis']:.3f} |")
    lines.append("")
    lines.append("## Gate verdict (Étape 1 → Étape 2)")
    lines.append("")
    lines.append(f"**{diag.gate_verdict}**")
    lines.append("")
    lines.append(f"{diag.gate_reason}")
    lines.append("")
    lines.append("### Règles de décision")
    lines.append("")
    lines.append("- `max|ACF(1..10)| > 0.05` ET `LB p < 0.05` → **EXPLOITABLE** : Étape 2 justifiée")
    lines.append("- `max|ACF(1..10)| ∈ [0.02, 0.05]` ET `LB p < 0.05` → **MARGINAL** : confirmer avant Étape 2")
    lines.append("- `max|ACF(1..10)| < 0.02` → **WHITE_NOISE** : skip Étape 2")
    lines.append("")
    if plot_paths:
        lines.append("## Plots")
        lines.append("")
        for p in plot_paths:
            lines.append(f"- `{p.relative_to(report_path.parent)}`")
        lines.append("")
    report_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Étape 0+1 orchestrator")
    p.add_argument("--csv", default="data_trad/BTCUSD_all_5m.csv", help="Chemin CSV BTC 5min")
    p.add_argument("--start-date", default="2022-01-01")
    p.add_argument("--end-date", default=None)
    p.add_argument("--out-dir", default=str(_HERE / "artifacts"),
                   help="Répertoire de sortie (default: experiments/slope_improvement/artifacts)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir).resolve()
    run(
        csv_path=args.csv,
        start_date=args.start_date,
        end_date=args.end_date,
        out_dir=out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
