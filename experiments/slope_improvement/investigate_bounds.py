"""
Investigation B — recalibration des bornes σ² et vérification de R.

Objectif : déterminer si la saturation observée à l'Étape 1 (σ²_final =
0.10 = borne haute, std(z) = 5.37) est due à des bornes trop serrées,
à un R mal calibré, ou à un problème structurel 2D.

Ce script NE modifie rien de l'existant. Il réutilise :
    - data_loader.make_splits
    - ground_truth.compute_full_ground_truth
    - kf_baseline.run_kf_baseline (les paramètres sigma2_min/max sont déjà
      exposés en CLI)
    - diagnostics.run_diagnostic (ACF, LB, JB, gate verdict)
    - metrics.compute_all (MSE, MAE, Pearson, DirMatch)

Tâches (cf. prompt utilisateur) :
    B.1 — Grid de bornes élargies :
        Run 1 : [σ²·0.1,   σ²·10]      (baseline actuel)
        Run 2 : [σ²·0.01,  σ²·100]
        Run 3 : [σ²·0.001, σ²·1000]
        Pour chaque : σ² mean/median, %clip haut/bas, std(z), LB, ACF, MSE val.

    B.2 — R empirique :
        Estimate R via variance du résidu RSI[t] - MA(RSI, window=5)[t]
        (approximation d'un bruit d'observation local).
        Compare à src.constants.KALMAN_MEASURE_VAR = 0.1.

    B.3 — Verdict automatique : classe le résultat en Cas 1/2/3 (prompt).

Output : console + artifacts/bounds_investigation.json + bounds_investigation.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

# Make sibling modules importable.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reuse project constants.
_PROJECT_ROOT = _HERE.parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from constants import KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR  # noqa: E402

from data_loader import make_splits  # noqa: E402
from ground_truth import compute_full_ground_truth, GroundTruth  # noqa: E402
from kf_baseline import run_kf_baseline, KFBaselineResult  # noqa: E402
from diagnostics import run_diagnostic  # noqa: E402
from metrics import compute_all  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fraction_at(trace: np.ndarray, bound: float, rel_tol: float = 1e-6) -> float:
    """
    Fraction of time where |trace - bound| / bound < rel_tol.
    Uses relative tolerance since σ² varies by orders of magnitude.
    """
    mask = np.isfinite(trace)
    if not mask.any():
        return float("nan")
    v = trace[mask]
    return float(np.mean(np.abs(v - bound) < rel_tol * max(abs(bound), 1e-30)))


def _empirical_R(rsi_full: np.ndarray, window: int = 5) -> Dict[str, float]:
    """
    Ordre-de-grandeur estimate of observation noise variance R.

    Residual = RSI[t] - centered_MA(RSI, window=5)[t]
    R_emp = Var(residual)

    Caveat : this conflates observation noise with signal curvature; useful
    as order-of-magnitude check only.
    """
    n = len(rsi_full)
    half = window // 2
    ma = np.full(n, np.nan)
    for t in range(half, n - half):
        w = rsi_full[t - half : t + half + 1]
        if np.all(np.isfinite(w)):
            ma[t] = np.mean(w)
    residual = rsi_full - ma
    finite = np.isfinite(residual)
    R_emp = float(np.var(residual[finite], ddof=0))
    # Also compute robust MAD-based estimate (immune to outliers)
    med = float(np.median(residual[finite]))
    mad = float(np.median(np.abs(residual[finite] - med)))
    # Convert MAD to std assuming normality: std ≈ 1.4826 * MAD
    R_mad = float((1.4826 * mad) ** 2)
    return {
        "R_current": float(KALMAN_MEASURE_VAR),
        "R_emp_var": R_emp,
        "R_emp_mad2": R_mad,
        "window": int(window),
        "n_residuals": int(finite.sum()),
        "residual_mean": float(np.mean(residual[finite])),
    }


# ---------------------------------------------------------------------------
# Single-run packaging
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    run_name: str
    sigma2_min: float
    sigma2_max: float
    sigma2_mean_train: float
    sigma2_median_train: float
    sigma2_p05_train: float
    sigma2_p95_train: float
    frac_at_min_train: float
    frac_at_max_train: float
    z_mean: float
    z_std: float
    acf_1_10: List[float]
    acf_max_abs: float
    lb_h10_stat: float
    lb_h10_p: float
    mse_val_rts: float
    mse_val_ma: float
    pearson_val_rts: float
    dirmatch_val_rts: float
    gate_verdict: str


def run_one(
    rsi_full: np.ndarray,
    gt_val: GroundTruth,
    train_idx_end: int,
    val_idx_start: int,
    val_idx_end: int,
    sigma2_min: float,
    sigma2_max: float,
    run_name: str,
) -> RunResult:
    """
    Run a single KF baseline on the full series with given bounds, extract
    metrics on train (diagnostic) and val (MSE).
    """
    kf = run_kf_baseline(
        rsi_full,
        sigma2_init=KALMAN_PROCESS_VAR,
        sigma2_min=sigma2_min,
        sigma2_max=sigma2_max,
    )

    # Train slice for innovations diagnostic
    s2_train = kf.sigma2_trace[0:train_idx_end]
    v_train = kf.innovations[0:train_idx_end]
    S_train = kf.S[0:train_idx_end]

    # Val slice for MSE
    slope_val = kf.slope[val_idx_start:val_idx_end]

    diag = run_diagnostic(v_train, S_train, split_name=f"train_{run_name}")
    s2_valid = s2_train[np.isfinite(s2_train)]

    m_rts = compute_all(slope_val, gt_val.slope_rts)
    m_ma = compute_all(slope_val, gt_val.slope_ma)

    return RunResult(
        run_name=run_name,
        sigma2_min=sigma2_min,
        sigma2_max=sigma2_max,
        sigma2_mean_train=float(np.mean(s2_valid)),
        sigma2_median_train=float(np.median(s2_valid)),
        sigma2_p05_train=float(np.percentile(s2_valid, 5)),
        sigma2_p95_train=float(np.percentile(s2_valid, 95)),
        frac_at_min_train=_fraction_at(s2_train, sigma2_min),
        frac_at_max_train=_fraction_at(s2_train, sigma2_max),
        z_mean=diag.mean,
        z_std=diag.std,
        acf_1_10=[float(v) for v in diag.acf_1_to_10],
        acf_max_abs=diag.acf_max_abs_1_10,
        lb_h10_stat=diag.ljung_box_h10["statistic"],
        lb_h10_p=diag.ljung_box_h10["p_value"],
        mse_val_rts=m_rts.mse,
        mse_val_ma=m_ma.mse,
        pearson_val_rts=m_rts.pearson,
        dirmatch_val_rts=m_rts.direction_match,
        gate_verdict=diag.gate_verdict,
    )


# ---------------------------------------------------------------------------
# Decision logic (B.3)
# ---------------------------------------------------------------------------

def decide_case(results: List[RunResult], R_info: Dict[str, float]) -> Dict[str, str]:
    """
    Classify outcome into Case 1 / 2 / 3 per user spec.

    Case 1: σ² se stabilise dans [σ²·0.1, σ²·10] avec std(z) ≈ 1
        → baseline OK, investiguer R
    Case 2: σ² se stabilise à valeur très au-dessus de la borne actuelle,
           std(z) se rapproche de 1
        → baseline sous-calibré, adopter nouvelles bornes
    Case 3: σ² explose sans stabilisation même avec bornes très larges
        → problème structurel 2D, aller en 3D
    """
    r1, r2, r3 = results  # run1=narrow, run2=medium, run3=wide

    # Is R possibly miscalibrated?
    R_current = R_info["R_current"]
    R_emp = R_info["R_emp_var"]
    R_mad = R_info["R_emp_mad2"]
    ratio_R = R_emp / R_current if R_current > 0 else float("inf")
    ratio_R_mad = R_mad / R_current if R_current > 0 else float("inf")

    def _ratio_display(r: float) -> str:
        if r >= 10 or r <= 0.1:
            return f"DÉCALÉ (×{r:.2f})"
        return f"cohérent (×{r:.2f})"

    # Case 3 detection: even widest bounds saturate at upper bound significantly
    if r3.frac_at_max_train > 0.05 and r3.z_std > 2.0:
        case = "CASE_3"
        reason = (
            f"Même avec bornes [{r3.sigma2_min:.1e}, {r3.sigma2_max:.1e}], "
            f"σ² sature au plafond ({100*r3.frac_at_max_train:.1f}% du temps) et "
            f"std(z)={r3.z_std:.2f} (>> 1). Problème structurel 2D → "
            f"probablement dominance d'un terme d'accélération ou de mean-reversion "
            f"que le modèle CV ne capture pas. Étape 2 (3D WNA) justifiée."
        )

    # Case 1 detection: run1 (narrow) already had good std(z) ≈ 1
    elif 0.75 <= r1.z_std <= 1.25:
        case = "CASE_1"
        reason = (
            f"Le baseline actuel (run1) avait std(z)={r1.z_std:.2f} ≈ 1. "
            f"Les bornes [{r1.sigma2_min:.1e}, {r1.sigma2_max:.1e}] étaient OK, "
            f"le std(z)=5.37 rapporté à l'Étape 1 était un artefact de l'ancien run. "
            f"R check : empirique vs courant {_ratio_display(ratio_R)}."
        )

    # Case 2 detection: widening bounds brings std(z) toward 1
    elif r3.z_std < r1.z_std * 0.7:
        # Choose the "best" run as the one with std(z) closest to 1
        best_run = min(results, key=lambda r: abs(r.z_std - 1.0))
        case = "CASE_2"
        reason = (
            f"L'élargissement des bornes converge : std(z) {r1.z_std:.2f} → {r2.z_std:.2f} → {r3.z_std:.2f}. "
            f"Baseline sous-calibré. Run recommandé comme nouveau baseline : {best_run.run_name} "
            f"(bornes [{best_run.sigma2_min:.1e}, {best_run.sigma2_max:.1e}], "
            f"std(z)={best_run.z_std:.2f}, MSE val RTS={best_run.mse_val_rts:.4f}). "
            f"R check : empirique vs courant {_ratio_display(ratio_R)}."
        )

    # Fallback ambiguous case
    else:
        case = "AMBIGU"
        reason = (
            f"Tendance peu claire : std(z) {r1.z_std:.2f} → {r2.z_std:.2f} → {r3.z_std:.2f}. "
            f"Ni convergence nette vers 1 ni saturation persistante. "
            f"R check : empirique vs courant {_ratio_display(ratio_R)}. "
            f"Décision humaine requise."
        )

    return {
        "case": case,
        "reason": reason,
        "R_current": R_current,
        "R_empirical_var": R_emp,
        "R_empirical_mad2": R_mad,
        "R_ratio_var": ratio_R,
        "R_ratio_mad": ratio_R_mad,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(x: float, sig: int = 4) -> str:
    if not np.isfinite(x):
        return "NaN"
    return f"{x:.{sig}g}"


def print_summary(results: List[RunResult], R_info: Dict[str, float], verdict: Dict) -> None:
    bar = "=" * 90
    print("\n" + bar)
    print("SYNTHÈSE BOUNDS INVESTIGATION (B.1 + B.2 + B.3)")
    print(bar)

    # Table header
    hdr = (
        f"{'run':10s} {'σ²_min':>10s} {'σ²_max':>10s} "
        f"{'σ²_mean':>10s} {'σ²_med':>10s} "
        f"{'%clip_lo':>9s} {'%clip_hi':>9s} "
        f"{'std(z)':>7s} {'max|ACF|':>9s} "
        f"{'MSE_val':>9s} {'Pearson':>8s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r.run_name:10s} "
            f"{_fmt(r.sigma2_min):>10s} {_fmt(r.sigma2_max):>10s} "
            f"{_fmt(r.sigma2_mean_train):>10s} {_fmt(r.sigma2_median_train):>10s} "
            f"{100*r.frac_at_min_train:>8.2f}% {100*r.frac_at_max_train:>8.2f}% "
            f"{r.z_std:>7.3f} {r.acf_max_abs:>9.4f} "
            f"{_fmt(r.mse_val_rts):>9s} {r.pearson_val_rts:>8.4f}"
        )

    print("\nACF(1..10) par run :")
    for r in results:
        vals = " ".join(f"{v:+.4f}" for v in r.acf_1_10)
        print(f"  {r.run_name}: [{vals}]  LB_stat={r.lb_h10_stat:.1f}  LB_p={r.lb_h10_p:.3e}")

    print("\n" + "-" * len(hdr))
    print("VÉRIFICATION R (B.2) :")
    print(f"  R courant (KALMAN_MEASURE_VAR) = {R_info['R_current']:.6g}")
    print(f"  R empirique (Var(RSI - MA5))   = {R_info['R_emp_var']:.6g}")
    print(f"  R empirique (MAD² * 1.4826²)   = {R_info['R_emp_mad2']:.6g}  (robuste aux outliers)")
    print(f"  n résidus utilisés             = {R_info['n_residuals']:,}")
    print(f"  ratio var / courant            = {R_info['R_emp_var']/R_info['R_current']:.3f}")
    print(f"  ratio MAD² / courant           = {R_info['R_emp_mad2']/R_info['R_current']:.3f}")

    print("\n" + "-" * len(hdr))
    print(f"VERDICT : {verdict['case']}")
    print(f"{verdict['reason']}")
    print(bar + "\n")


def write_report(
    results: List[RunResult],
    R_info: Dict[str, float],
    verdict: Dict,
    report_path: Path,
) -> None:
    lines = []
    lines.append("# Investigation B — Bornes σ² + R")
    lines.append("")
    lines.append("## B.1 — Grid de bornes")
    lines.append("")
    lines.append("| run | σ²_min | σ²_max | mean σ² | median σ² | %clip_lo | %clip_hi | std(z) | max\\|ACF\\| | LB p | MSE val (RTS) | Pearson | DirMatch |")
    lines.append("|-----|--------|--------|---------|-----------|----------|----------|--------|-----------|------|---------------|---------|----------|")
    for r in results:
        lines.append(
            f"| {r.run_name} | {_fmt(r.sigma2_min)} | {_fmt(r.sigma2_max)} | "
            f"{_fmt(r.sigma2_mean_train)} | {_fmt(r.sigma2_median_train)} | "
            f"{100*r.frac_at_min_train:.2f}% | {100*r.frac_at_max_train:.2f}% | "
            f"{r.z_std:.3f} | {r.acf_max_abs:.4f} | {r.lb_h10_p:.3e} | "
            f"{_fmt(r.mse_val_rts)} | {r.pearson_val_rts:.4f} | {r.dirmatch_val_rts:.4f} |"
        )
    lines.append("")
    lines.append("### ACF(1..10) par run")
    lines.append("")
    for r in results:
        vals = ", ".join(f"{v:+.4f}" for v in r.acf_1_10)
        lines.append(f"- **{r.run_name}** : [{vals}]")
    lines.append("")
    lines.append("## B.2 — Vérification R")
    lines.append("")
    lines.append(f"- R courant (KALMAN_MEASURE_VAR) = **{R_info['R_current']:.6g}**")
    lines.append(f"- R empirique via Var(RSI − MA5) = **{R_info['R_emp_var']:.6g}** (ratio courant × {R_info['R_emp_var']/R_info['R_current']:.3f})")
    lines.append(f"- R empirique via MAD (robuste)  = **{R_info['R_emp_mad2']:.6g}** (ratio courant × {R_info['R_emp_mad2']/R_info['R_current']:.3f})")
    lines.append(f"- n résidus = {R_info['n_residuals']:,}, mean résidu = {R_info['residual_mean']:.4e}")
    lines.append("")
    lines.append("> Note : l'estimation via MA5 conflate bruit d'observation et courbure locale.")
    lines.append("> Sert d'ordre-de-grandeur uniquement.")
    lines.append("")
    lines.append("## B.3 — Verdict")
    lines.append("")
    lines.append(f"**{verdict['case']}**")
    lines.append("")
    lines.append(verdict["reason"])
    lines.append("")
    lines.append("### Règles")
    lines.append("- **CASE_1** : σ² run1 stable dans [0.1σ²₀, 10σ²₀] ET std(z) ≈ 1 → baseline OK, corriger R")
    lines.append("- **CASE_2** : élargir les bornes ramène std(z) vers 1 → adopter nouvelles bornes comme baseline")
    lines.append("- **CASE_3** : saturation persiste même avec bornes très larges → problème structurel 2D, aller en 3D")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Investigation B: σ² bounds + R")
    parser.add_argument("--csv", default="data_trad/BTCUSD_all_5m.csv")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--out-dir", default=str(_HERE / "artifacts"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load splits
    train, val, test, split_meta = make_splits(
        csv_path=args.csv, start_date=args.start_date, end_date=args.end_date,
    )
    rsi_full = np.concatenate([train.rsi, val.rsi, test.rsi])
    print(f"N total post-warmup = {len(rsi_full):,} (train={train.n:,}  val={val.n:,}  test={test.n:,})")

    # 2. Ground truth on full series (once)
    gt_full = compute_full_ground_truth(rsi_full)
    gt_val = gt_full.split(val.idx_start, val.idx_end)

    # 3. Bounds grid
    sig2_0 = KALMAN_PROCESS_VAR  # 0.01
    runs_config = [
        ("run1_narrow",   sig2_0 * 0.1,    sig2_0 * 10.0),    # baseline actuel
        ("run2_medium",   sig2_0 * 0.01,   sig2_0 * 100.0),
        ("run3_wide",     sig2_0 * 0.001,  sig2_0 * 1000.0),
    ]

    results: List[RunResult] = []
    for name, s2_min, s2_max in runs_config:
        print(f"\n--- Running {name}  bounds=[{s2_min:.1e}, {s2_max:.1e}] ---")
        r = run_one(
            rsi_full=rsi_full,
            gt_val=gt_val,
            train_idx_end=train.idx_end,
            val_idx_start=val.idx_start,
            val_idx_end=val.idx_end,
            sigma2_min=s2_min,
            sigma2_max=s2_max,
            run_name=name,
        )
        results.append(r)
        print(
            f"  σ² mean={r.sigma2_mean_train:.4g}  median={r.sigma2_median_train:.4g}  "
            f"%clip_lo={100*r.frac_at_min_train:.2f}%  %clip_hi={100*r.frac_at_max_train:.2f}%  "
            f"std(z)={r.z_std:.3f}  MSE val={r.mse_val_rts:.4f}"
        )

    # 4. R empirical check
    R_info = _empirical_R(rsi_full, window=5)

    # 5. Decision
    verdict = decide_case(results, R_info)

    # 6. Print + save
    print_summary(results, R_info, verdict)

    report_path = out_dir / "bounds_investigation.md"
    write_report(results, R_info, verdict, report_path)
    print(f"Rapport : {report_path}")

    # Save JSON
    blob = {
        "split": split_meta,
        "runs": [asdict(r) for r in results],
        "R_info": R_info,
        "verdict": verdict,
    }
    json_path = out_dir / "bounds_investigation.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2, default=str, ensure_ascii=False)
    print(f"JSON     : {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
