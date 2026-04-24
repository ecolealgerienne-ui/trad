"""
Étape B.4 — Validation du ground truth officiel + recalibration R.

Objectif (utilisateur) :
    > "30 min d'investigation complémentaire fondent la validité de tout le
    > reste du projet."

Séquence :
    1. Estimation empirique multi-méthode de R (4 estimateurs)  → R*
    2. Calcul du Ground Truth OFFICIEL (3D WNA, MLE global)     → gt_official_slope.npy
    3. Sweep baselines 2D avec R ∈ {0.5·R*, 1·R*, 2·R*}, bornes σ² originales
    4. Validation : std(z) ∈ [0.85, 1.15] ET clip_hi σ² < 5%
    5. Sélection du R de baseline qui donne std(z) le plus proche de 1

Réutilise :
    - data_loader.make_splits
    - kf_baseline.run_kf_baseline (r_scalar déjà exposé)
    - diagnostics.run_diagnostic
    - metrics.compute_all
    - estimate_R.estimate_R_multimethod   (nouveau)
    - gt_3d.compute_official_ground_truth (nouveau)

Artefacts produits :
    artifacts/gt_official_slope.npy           (PRIMARY GT pour toute la suite)
    artifacts/gt_official_level.npy
    artifacts/gt_official_accel.npy
    artifacts/gt_official_metadata.json       (σ²_accel*, R*, NLL*, diagnostics)
    artifacts/R_estimation.json               (les 4 méthodes + choix)
    artifacts/validate_R_fix_results.json     (baseline sweep results)
    artifacts/validate_R_fix_report.md        (rapport humain)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

# Local imports
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_PROJECT_ROOT = _HERE.parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from constants import KALMAN_PROCESS_VAR  # noqa: E402

from data_loader import make_splits  # noqa: E402
from kf_baseline import run_kf_baseline  # noqa: E402
from diagnostics import run_diagnostic  # noqa: E402
from metrics import compute_all  # noqa: E402
from estimate_R import estimate_R_multimethod, format_estimation_table  # noqa: E402
from gt_3d import compute_official_ground_truth  # noqa: E402


# ---------------------------------------------------------------------------
# Per-R-run result container
# ---------------------------------------------------------------------------

@dataclass
class BaselineRun:
    r_value: float
    r_label: str
    sigma2_min: float
    sigma2_max: float
    sigma2_mean: float
    sigma2_median: float
    frac_clip_lo: float
    frac_clip_hi: float
    z_mean: float
    z_std: float
    acf_1_10: List[float]
    acf_max_abs: float
    lb_p: float
    mse_val_gt: float
    mae_val_gt: float
    pearson_val_gt: float
    dirmatch_val_gt: float
    mse_test_gt: float
    pearson_test_gt: float
    passes_std_z: bool
    passes_clip_hi: bool


def _frac_at(trace: np.ndarray, bound: float, rel_tol: float = 1e-6) -> float:
    mask = np.isfinite(trace)
    if not mask.any():
        return float("nan")
    v = trace[mask]
    return float(np.mean(np.abs(v - bound) < rel_tol * max(abs(bound), 1e-30)))


def run_baseline_with_R(
    rsi_full: np.ndarray,
    train_end: int,
    val_start: int,
    val_end: int,
    test_start: int,
    test_end: int,
    gt_slope: np.ndarray,
    r_value: float,
    r_label: str,
) -> BaselineRun:
    """One baseline run with given R and original σ² bounds."""
    s2_init = KALMAN_PROCESS_VAR
    s2_min = s2_init * 0.1
    s2_max = s2_init * 10.0

    kf = run_kf_baseline(
        rsi_full,
        sigma2_init=s2_init,
        sigma2_min=s2_min,
        sigma2_max=s2_max,
        r_scalar=r_value,
    )

    s2_train = kf.sigma2_trace[:train_end]
    v_train = kf.innovations[:train_end]
    S_train = kf.S[:train_end]

    slope_val = kf.slope[val_start:val_end]
    slope_test = kf.slope[test_start:test_end]

    gt_val = gt_slope[val_start:val_end]
    gt_test = gt_slope[test_start:test_end]

    diag = run_diagnostic(v_train, S_train, split_name=f"train_R{r_label}")
    s2_valid = s2_train[np.isfinite(s2_train)]

    m_val = compute_all(slope_val, gt_val)
    m_test = compute_all(slope_test, gt_test)

    clip_hi = _frac_at(s2_train, s2_max)
    clip_lo = _frac_at(s2_train, s2_min)
    passes_std = 0.85 <= diag.std <= 1.15
    passes_clip = clip_hi < 0.05

    return BaselineRun(
        r_value=float(r_value),
        r_label=r_label,
        sigma2_min=s2_min,
        sigma2_max=s2_max,
        sigma2_mean=float(np.mean(s2_valid)),
        sigma2_median=float(np.median(s2_valid)),
        frac_clip_lo=clip_lo,
        frac_clip_hi=clip_hi,
        z_mean=diag.mean,
        z_std=diag.std,
        acf_1_10=[float(v) for v in diag.acf_1_to_10],
        acf_max_abs=diag.acf_max_abs_1_10,
        lb_p=diag.ljung_box_h10["p_value"],
        mse_val_gt=m_val.mse,
        mae_val_gt=m_val.mae,
        pearson_val_gt=m_val.pearson,
        dirmatch_val_gt=m_val.direction_match,
        mse_test_gt=m_test.mse,
        pearson_test_gt=m_test.pearson,
        passes_std_z=passes_std,
        passes_clip_hi=passes_clip,
    )


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_markdown_report(
    path: Path,
    split_meta: dict,
    r_estimation,
    gt_info: dict,
    runs: List[BaselineRun],
    retained: BaselineRun,
    final_reasoning: str,
) -> None:
    def _fmt(x, s=6):
        return "NaN" if not np.isfinite(x) else f"{x:.{s}g}"

    L = []
    L.append("# Étape B.4 — Validation GT officiel + recalibration R")
    L.append("")
    L.append("## Configuration")
    L.append(f"- Dataset : `{split_meta['csv_path']}`")
    L.append(f"- Période : {split_meta['start_date']} → fin")
    L.append(f"- N total (post-warmup RSI) : {split_meta['n_total']:,}")
    L.append(f"- Splits : train={split_meta['n_train']:,}  val={split_meta['n_val']:,}  test={split_meta['n_test']:,}")
    L.append("")
    L.append("## 1. Estimation empirique de R (4 méthodes)")
    L.append("")
    L.append("| méthode | R (var) | R (MAD²) | n | détail |")
    L.append("|---------|---------|----------|---|--------|")
    m1 = r_estimation.m1_ma5
    m2 = r_estimation.m2_ma11
    m3 = r_estimation.m3_firstdiff
    m4 = r_estimation.m4_mle
    L.append(f"| {m1['method']} | {_fmt(m1['R'], 4)} | {_fmt(m1.get('R_mad', float('nan')), 4)} | {m1['n']:,} | skew={m1.get('residual_skew', 0):.2f} |")
    L.append(f"| {m2['method']} | {_fmt(m2['R'], 4)} | {_fmt(m2.get('R_mad', float('nan')), 4)} | {m2['n']:,} | skew={m2.get('residual_skew', 0):.2f} |")
    L.append(f"| {m3['method']} | {_fmt(m3['R'], 4)} | {_fmt(m3.get('R_mad', float('nan')), 4)} | {m3['n']:,} | raw_var={m3.get('raw_variance', 0):.3f} |")
    L.append(f"| {m4['method']} | {_fmt(m4['R'], 4)} | — | {m4['n']:,} | σ²={_fmt(m4.get('sigma2_proc', 0), 4)}  NLL={_fmt(m4.get('nll', float('nan')), 5)}  success={m4.get('success', False)} |")
    L.append("")
    L.append(f"**R retenu : {r_estimation.r_chosen:.4f}** (méthode : `{r_estimation.chosen_method}`)")
    L.append("")
    L.append(f"> Raison : {r_estimation.reasoning}")
    L.append("")
    L.append("## 2. Ground Truth OFFICIEL (3D WNA, MLE global)")
    L.append("")
    L.append("Modèle : `x = [level, slope, accel]` (3D White-Noise Acceleration), F et G de Bar-Shalom.")
    L.append("")
    L.append(f"- `σ²_accel*` MLE : {gt_info['sigma2_accel']:.6g}")
    L.append(f"- `R*` MLE        : {gt_info['r_scalar']:.6g}")
    L.append(f"- NLL optimale    : {gt_info['nll']:.2f}")
    L.append(f"- Samples MLE     : {gt_info['n_fit_samples']:,}")
    L.append(f"- Samples RTS     : {gt_info['n_full_samples']:,}")
    L.append("")
    L.append(f"> Rationale : le GT a 3 degrés de liberté (level, slope, accel). Le baseline 2D et les variantes 3D/IMM que nous testerons auront au maximum 3 degrés de liberté eux aussi, MAIS sans RTS backward (causal only). Le GT restera donc plus informé qu'aucun modèle testé.")
    L.append("")
    L.append("## 3. Baseline 2D — sweep R ∈ {0.5·R*, R*, 2·R*}")
    L.append("")
    L.append("Bornes σ² fixées aux valeurs originales `[σ²_init · 0.1, σ²_init · 10]` = `[0.001, 0.1]`.")
    L.append("")
    L.append("| label | R | σ²_mean | σ²_median | %clip_lo | %clip_hi | std(z) | max\\|ACF\\| | LB p | MSE val | Pearson val | DirMatch val | MSE test | Pearson test | std(z) OK | clip_hi OK |")
    L.append("|-------|---|---------|-----------|----------|----------|--------|-----------|------|---------|-------------|--------------|----------|--------------|-----------|------------|")
    for r in runs:
        L.append(
            f"| {r.r_label} | {_fmt(r.r_value, 4)} | {_fmt(r.sigma2_mean, 4)} | {_fmt(r.sigma2_median, 4)} | "
            f"{100*r.frac_clip_lo:.1f}% | {100*r.frac_clip_hi:.1f}% | "
            f"{r.z_std:.3f} | {r.acf_max_abs:.4f} | {_fmt(r.lb_p, 3)} | "
            f"{_fmt(r.mse_val_gt, 4)} | {r.pearson_val_gt:.4f} | {r.dirmatch_val_gt:.4f} | "
            f"{_fmt(r.mse_test_gt, 4)} | {r.pearson_test_gt:.4f} | "
            f"{'✅' if r.passes_std_z else '❌'} | {'✅' if r.passes_clip_hi else '❌'} |"
        )
    L.append("")
    L.append("### ACF(1..10)")
    L.append("")
    for r in runs:
        vals = ", ".join(f"{v:+.4f}" for v in r.acf_1_10)
        L.append(f"- **{r.r_label}** : [{vals}]")
    L.append("")
    L.append("## 4. R retenu pour le baseline officiel")
    L.append("")
    L.append(f"**R_baseline = {retained.r_value:.4f}** (label = {retained.r_label})")
    L.append("")
    L.append(f"{final_reasoning}")
    L.append("")
    L.append("### Critères de validation")
    L.append(f"- std(z) ∈ [0.85, 1.15] : {'✅ pass' if retained.passes_std_z else '❌ fail'} (std(z)={retained.z_std:.3f})")
    L.append(f"- %clip_hi σ² < 5%     : {'✅ pass' if retained.passes_clip_hi else '❌ fail'} (clip_hi={100*retained.frac_clip_hi:.2f}%)")
    L.append(f"- MSE val vs GT         : {retained.mse_val_gt:.4f}")
    L.append(f"- Pearson val           : {retained.pearson_val_gt:.4f}")
    L.append(f"- DirMatch val          : {retained.dirmatch_val_gt:.4f}")
    L.append("")
    L.append("## 5. Artefacts produits")
    L.append("- `gt_official_slope.npy`  — PRIMARY GT pour toute la suite (Étape 2, 3, 5)")
    L.append("- `gt_official_level.npy`")
    L.append("- `gt_official_accel.npy`")
    L.append("- `gt_official_metadata.json`")
    L.append("- `R_estimation.json`")
    L.append("- `validate_R_fix_results.json`")
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Étape B.4: official GT + R recalibration")
    parser.add_argument("--csv", default="data_trad/BTCUSD_all_5m.csv")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--out-dir", default=str(_HERE / "artifacts"))
    parser.add_argument("--mle-subsample", type=int, default=20_000,
                       help="Max samples for MLE fits (both R estimation and GT)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bar = "=" * 78
    print(bar)
    print("ÉTAPE B.4 — Validation GT officiel + recalibration R")
    print(bar)

    # ---- 1. Splits ---------------------------------------------------------
    print("\n[1/5] Chargement splits...")
    train, val, test, split_meta = make_splits(
        csv_path=args.csv, start_date=args.start_date, end_date=args.end_date,
    )
    rsi_full = np.concatenate([train.rsi, val.rsi, test.rsi])
    print(f"  N={len(rsi_full):,}  train={train.n:,}  val={val.n:,}  test={test.n:,}")

    # ---- 2. R estimation multi-méthode ------------------------------------
    print("\n[2/5] Estimation empirique de R (4 méthodes)...")
    r_est = estimate_R_multimethod(
        rsi_full,
        train_end_idx=train.idx_end,
        mle_subsample=args.mle_subsample,
    )
    print(format_estimation_table(r_est))

    # Save R estimation
    r_est_path = out_dir / "R_estimation.json"
    with r_est_path.open("w", encoding="utf-8") as f:
        json.dump({
            "m1_ma5": r_est.m1_ma5,
            "m2_ma11": r_est.m2_ma11,
            "m3_firstdiff": r_est.m3_firstdiff,
            "m4_mle": r_est.m4_mle,
            "r_chosen": r_est.r_chosen,
            "chosen_method": r_est.chosen_method,
            "reasoning": r_est.reasoning,
        }, f, indent=2, default=str, ensure_ascii=False)
    print(f"  → {r_est_path.name}")

    R_star = r_est.r_chosen

    # ---- 3. Official GT (3D WNA, MLE global) ------------------------------
    print("\n[3/5] Calcul du Ground Truth OFFICIEL (3D WNA, MLE)...")
    gt = compute_official_ground_truth(
        rsi_full,
        train_end_idx=train.idx_end,
        subsample_n=args.mle_subsample,
        init_sigma2=1e-3,
        init_r=R_star,           # seed MLE with empirical R for convergence
        verbose=True,
    )
    np.save(out_dir / "gt_official_slope.npy", gt.slope)
    np.save(out_dir / "gt_official_level.npy", gt.level)
    np.save(out_dir / "gt_official_accel.npy", gt.accel)
    gt_meta = {
        "sigma2_accel": gt.sigma2_accel,
        "r_scalar": gt.r_scalar,
        "nll": gt.nll,
        "n_fit_samples": gt.n_fit_samples,
        "n_full_samples": gt.n_full_samples,
        "slope_mean": float(gt.slope.mean()),
        "slope_std": float(gt.slope.std()),
        "level_mean": float(gt.level.mean()),
        "level_std": float(gt.level.std()),
    }
    with (out_dir / "gt_official_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(gt_meta, f, indent=2, default=str, ensure_ascii=False)
    print(f"  → gt_official_slope.npy  slope mean={gt.slope.mean():.5f}  std={gt.slope.std():.5f}")

    # ---- 4. Baseline sweep R ∈ {0.5R, R, 2R} ------------------------------
    print("\n[4/5] Baseline 2D — sweep R ∈ {0.5·R*, R*, 2·R*}")
    r_configs = [
        (0.5 * R_star, f"R=0.5·R* ({0.5*R_star:.3f})"),
        (1.0 * R_star, f"R=R* ({R_star:.3f})"),
        (2.0 * R_star, f"R=2·R* ({2.0*R_star:.3f})"),
    ]
    runs: List[BaselineRun] = []
    for r_val, r_label in r_configs:
        print(f"  → running {r_label}...")
        rr = run_baseline_with_R(
            rsi_full=rsi_full,
            train_end=train.idx_end,
            val_start=val.idx_start,
            val_end=val.idx_end,
            test_start=test.idx_start,
            test_end=test.idx_end,
            gt_slope=gt.slope,
            r_value=r_val,
            r_label=r_label,
        )
        runs.append(rr)
        print(
            f"    std(z)={rr.z_std:.3f}  %clip_hi={100*rr.frac_clip_hi:.2f}%  "
            f"σ²_med={rr.sigma2_median:.5f}  MSE val={rr.mse_val_gt:.4f}  "
            f"Pearson={rr.pearson_val_gt:.4f}  "
            f"std_z={'✅' if rr.passes_std_z else '❌'}  "
            f"clip={'✅' if rr.passes_clip_hi else '❌'}"
        )

    # Save run details
    with (out_dir / "validate_R_fix_results.json").open("w", encoding="utf-8") as f:
        json.dump({
            "splits": split_meta,
            "R_star": R_star,
            "R_estimation": {
                "m1_ma5": r_est.m1_ma5,
                "m2_ma11": r_est.m2_ma11,
                "m3_firstdiff": r_est.m3_firstdiff,
                "m4_mle": r_est.m4_mle,
                "chosen_method": r_est.chosen_method,
                "r_chosen": r_est.r_chosen,
            },
            "gt_official": gt_meta,
            "runs": [asdict(r) for r in runs],
        }, f, indent=2, default=str, ensure_ascii=False)

    # ---- 5. Verdict : pick best R ----------------------------------------
    print(f"\n[5/5] Sélection du R baseline optimal")
    # Rule: prefer runs that pass BOTH criteria, then among those pick std(z) closest to 1.
    passing = [r for r in runs if r.passes_std_z and r.passes_clip_hi]
    if passing:
        retained = min(passing, key=lambda r: abs(r.z_std - 1.0))
        reasoning = (
            f"Parmi les runs qui passent les deux critères (std(z) ∈ [0.85, 1.15] ET clip_hi < 5%), "
            f"choisi celui avec std(z) le plus proche de 1 → `{retained.r_label}`. "
            f"Baseline officiel : R = {retained.r_value:.4f}."
        )
    else:
        # Fallback: the one closest to std(z)=1 regardless
        retained = min(runs, key=lambda r: abs(r.z_std - 1.0))
        reasoning = (
            f"AUCUN run ne passe les deux critères simultanément. Choix par proximité à std(z)=1 → "
            f"`{retained.r_label}`. ⚠️ À investiguer : "
            f"std(z)={retained.z_std:.3f}, clip_hi={100*retained.frac_clip_hi:.2f}%. "
            f"Peut indiquer (a) R_empirical sous-estimé, (b) besoin d'adaptation R (Étape 3 plus tôt), "
            f"ou (c) inadéquation structurelle du modèle 2D (qui sera confirmée à l'Étape 2 3D WNA)."
        )

    print("\n" + "-" * 78)
    print(f"R RETENU : {retained.r_value:.4f}  ({retained.r_label})")
    print(f"  std(z)   = {retained.z_std:.3f}     {'✅' if retained.passes_std_z else '❌ hors [0.85, 1.15]'}")
    print(f"  clip_hi  = {100*retained.frac_clip_hi:.2f}%    {'✅' if retained.passes_clip_hi else '❌ > 5%'}")
    print(f"  MSE val  = {retained.mse_val_gt:.4f}")
    print(f"  Pearson val  = {retained.pearson_val_gt:.4f}")
    print(f"  DirMatch val = {retained.dirmatch_val_gt:.4f}")
    print(f"\n{reasoning}")
    print("-" * 78)

    # Markdown report
    md_path = out_dir / "validate_R_fix_report.md"
    write_markdown_report(
        path=md_path,
        split_meta=split_meta,
        r_estimation=r_est,
        gt_info=gt_meta,
        runs=runs,
        retained=retained,
        final_reasoning=reasoning,
    )
    print(f"\nRapport Markdown : {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
