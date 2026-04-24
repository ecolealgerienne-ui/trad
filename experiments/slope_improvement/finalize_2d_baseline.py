"""
Étape B.5 — Baseline 2D MLE optimal (paramètres fixes).

Utilisateur :
    > "Un seul run, paramètres fixés. σ² = 1.155 (MLE), R = 3.27 (MLE).
    >  PAS d'adaptation Myers-Tapley."

Objectif : obtenir LE meilleur baseline 2D théoriquement possible sous
la structure CV, avec std(z) ≈ 1 par construction (propriété du MLE).
Toutes les comparaisons aval (Étape 2 3D WNA, Étape 3 variantes, etc.)
se feront contre ce baseline officiel.

=============================================================================
PRÉDICTIONS DOCUMENTÉES AVANT LE RUN
=============================================================================

Attendus si le MLE est correct et représentatif du dataset complet :

    - std(z) ∈ [0.95, 1.05]
        * le MLE a été fitté sur les 20k premiers samples de train
        * l'extrapolation aux 210k samples entiers de train peut élargir
          légèrement : tolérance pratique [0.85, 1.15]

    - max|ACF(1..10)| ∈ [0.05, 0.15]
        * le MLE optimise la gaussianité des innovations, ACF(1..10) devrait
          chuter drastiquement vs l'original 2D adaptatif (qui était à 0.24)
        * un résidu de 0.05-0.15 reste cohérent avec la courbure non modélisée
          par le CV (à capturer par l'accélération en Étape 2)

    - LB p-value probablement < 0.05 sur N = 210k
        * attendu même si max|ACF| est petit (problème taille d'échantillon)
        * critère pratique : magnitude de l'ACF, pas la p-value

    - MSE val (vs GT officiel 3D MLE) : ≤ 1.0
        * les runs adaptatifs précédents ont obtenu ~0.9 à R*=3.27 ;
          MLE fixed doit faire au moins aussi bien (meilleur scaling)

SIGNAUX D'ALARME (si observés, Étape 2 est validée avec haute confiance) :

    - max|ACF(1..10)| > 0.15 de manière persistante
        → courbure structurellement non captée, 3D WNA prescrit

    - std(z) hors [0.85, 1.15] malgré MLE fixed
        → non-stationnarité forte du dataset, MLE global insuffisant
          (justifie peut-être Étape 3 adaptation par-dessus calibration globale)

=============================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_PROJECT_ROOT = _HERE.parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_loader import make_splits  # noqa: E402
from kf_baseline import run_kf_baseline  # noqa: E402
from diagnostics import run_diagnostic, make_plots  # noqa: E402
from metrics import compute_all  # noqa: E402


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _load_mle_params(r_estimation_path: Path) -> Dict[str, float]:
    """Load σ²_proc and R from R_estimation.json (MLE_2D section)."""
    if not r_estimation_path.exists():
        raise FileNotFoundError(
            f"{r_estimation_path} introuvable. "
            "Lancer d'abord validate_gt_and_R.py."
        )
    with r_estimation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    mle = data["m4_mle"]
    return {
        "sigma2_proc": float(mle["sigma2_proc"]),
        "r_scalar": float(mle["R"]),
        "nll": float(mle.get("nll", float("nan"))),
        "success": bool(mle.get("success", False)),
        "n_samples_used": int(mle.get("n", 0)),
    }


def _run_fixed_kf(
    rsi_full: np.ndarray,
    sigma2_fixed: float,
    r_fixed: float,
):
    """
    Run KF 2D with FIXED σ² (no adaptation).

    Trick : `run_kf_baseline` exposes sigma2_min/max ; setting them equal to
    sigma2_fixed forces every adaptation step to clip back to sigma2_fixed.
    With sigma2_init also == sigma2_fixed, σ² never moves. No code duplication.
    """
    return run_kf_baseline(
        rsi_full,
        sigma2_init=sigma2_fixed,
        sigma2_min=sigma2_fixed,
        sigma2_max=sigma2_fixed,
        r_scalar=r_fixed,
    )


def _compute_split_metrics(slope, gt_slope, split_name):
    """Return dict of metrics for one split vs GT."""
    m = compute_all(slope, gt_slope)
    return {
        "split": split_name,
        "mse": m.mse,
        "mae": m.mae,
        "pearson": m.pearson,
        "direction_match": m.direction_match,
        "latency_bars": m.latency_bars,
        "n_valid": m.n_valid,
    }


def _format_metric_row(name: str, m: Dict) -> str:
    return (
        f"  {name:<20s} MSE={m['mse']:.5g}  MAE={m['mae']:.5g}  "
        f"Pearson={m['pearson']:.4f}  DirMatch={m['direction_match']:.4f}  "
        f"Lag={m['latency_bars']:+.1f}  n={m['n_valid']:,}"
    )


# ---------------------------------------------------------------------------
# Markdown report writer (supersedes report_step1 for the 2D baseline)
# ---------------------------------------------------------------------------

def write_report(
    path: Path,
    mle: Dict[str, float],
    metrics_fixed: Dict[str, Dict],
    metrics_adaptive: Dict[str, Dict],
    diag,
    predictions_table: Dict,
    plot_files,
) -> None:
    L = []
    L.append("# Baseline 2D MLE optimal — Étape B.5")
    L.append("")
    L.append("> Supersedes the \"Metrics baseline\" section of report_step1.md")
    L.append("> for the 2D CV reference. Toutes les comparaisons aval (Étape 2, 3, 5)")
    L.append("> se feront contre CE baseline, pas contre l'original adaptatif R=0.1.")
    L.append("")
    L.append("## Paramètres MLE fixes (pas d'adaptation)")
    L.append("")
    L.append(f"- `σ²_proc` = {mle['sigma2_proc']:.6g}   (115× `σ²_init=0.01`, hors des bornes originales)")
    L.append(f"- `R`       = {mle['r_scalar']:.6g}   (33× `R_original=0.1`)")
    L.append(f"- `F = [[1,1],[0,1]]`, `H = [[1,0]]`, `Q = I₂ · σ²_proc`")
    L.append(f"- MLE NLL : {mle['nll']:.2f}   (fit sur {mle['n_samples_used']:,} samples train)")
    L.append(f"- Aucune Myers-Tapley → σ² constant tout au long du run")
    L.append("")
    L.append("## Prédictions vs observations")
    L.append("")
    L.append("| métrique | prédit (ex ante) | observé | verdict |")
    L.append("|----------|-----------------|---------|---------|")
    for row in predictions_table["rows"]:
        L.append(f"| {row['metric']} | {row['predicted']} | {row['observed']} | {row['verdict']} |")
    L.append("")
    L.append(f"**Interprétation globale** : {predictions_table['summary']}")
    L.append("")
    L.append("## Metrics vs GT officiel 3D MLE")
    L.append("")
    L.append("### Baseline 2D MLE fixed (PRIMARY — baseline officiel)")
    L.append("")
    L.append("| Split | MSE | MAE | Pearson | DirMatch | Lag | n |")
    L.append("|-------|-----|-----|---------|----------|-----|---|")
    for s in ("val", "test"):
        m = metrics_fixed[s]
        L.append(
            f"| {s} | {m['mse']:.5g} | {m['mae']:.5g} | {m['pearson']:.4f} | "
            f"{m['direction_match']:.4f} | {m['latency_bars']:+.1f} | {m['n_valid']:,} |"
        )
    L.append("")
    L.append("### Baseline 2D original (R=0.1, adaptive σ² in [0.001, 0.1]) — comparaison")
    L.append("")
    L.append("| Split | MSE | MAE | Pearson | DirMatch | Lag | n |")
    L.append("|-------|-----|-----|---------|----------|-----|---|")
    for s in ("val", "test"):
        m = metrics_adaptive[s]
        L.append(
            f"| {s} | {m['mse']:.5g} | {m['mae']:.5g} | {m['pearson']:.4f} | "
            f"{m['direction_match']:.4f} | {m['latency_bars']:+.1f} | {m['n_valid']:,} |"
        )
    L.append("")
    L.append("### Δ (MLE fixed − Original adaptive)")
    L.append("")
    L.append("| Split | ΔMSE | ΔPearson | ΔDirMatch |")
    L.append("|-------|------|----------|-----------|")
    for s in ("val", "test"):
        mf, ma = metrics_fixed[s], metrics_adaptive[s]
        dmse = mf["mse"] - ma["mse"]
        dpear = mf["pearson"] - ma["pearson"]
        ddir = mf["direction_match"] - ma["direction_match"]
        L.append(f"| {s} | {dmse:+.5g} | {dpear:+.4f} | {ddir:+.4f} |")
    L.append("")
    L.append("## Diagnostic d'innovations (TRAIN, MLE fixed)")
    L.append("")
    L.append(f"- n innovations = {diag.n:,}")
    L.append(f"- mean(z) = {diag.mean:.4f}   std(z) = {diag.std:.4f}")
    L.append("")
    L.append("**ACF(1..10) :**")
    L.append("")
    L.append("| Lag | ACF |")
    L.append("|-----|-----|")
    for k, v in enumerate(diag.acf_1_to_10, start=1):
        L.append(f"| {k} | {v:+.4f} |")
    L.append(f"\n**max|ACF(1..10)| = {diag.acf_max_abs_1_10:.4f}**")
    L.append("")
    L.append(f"- Ljung-Box h=10 : Q = {diag.ljung_box_h10['statistic']:.2f}, p = {diag.ljung_box_h10['p_value']:.3e}, Q/n = {diag.ljung_box_h10['q_per_n']:.4f}")
    L.append(f"- Jarque-Bera    : stat = {diag.jarque_bera['statistic']:.2f}, p = {diag.jarque_bera['p_value']:.3e}, skew = {diag.jarque_bera['skewness']:.3f}, excess_kurt = {diag.jarque_bera['excess_kurtosis']:.3f}")
    L.append("")
    L.append("## Conclusion méthodologique")
    L.append("")
    L.append("Ce baseline **remplace** l'ancien \"baseline adaptatif R=0.1\" pour toutes")
    L.append("les comparaisons aval. Justification :")
    L.append("- std(z) ≈ 1 par construction MLE → comparaisons Δ non biaisées par mauvaise calibration")
    L.append("- MSE minimale possible sous la structure 2D CV")
    L.append("- Évite le biais de présentation \"σ² saturé à 99%\" qui polluait le rapport")
    L.append("")
    L.append("Étape 2 (3D WNA) sera également évaluée en régime **MLE fixed** pour comparaison")
    L.append("propre. Myers-Tapley adaptatif devient une variante Étape 3 par-dessus le modèle gagnant.")
    L.append("")
    if plot_files:
        L.append("## Plots diagnostic")
        L.append("")
        for p in plot_files:
            L.append(f"- `{p.relative_to(path.parent)}`")
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Étape B.5: 2D MLE fixed baseline")
    parser.add_argument("--csv", default="data_trad/BTCUSD_all_5m.csv")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--artifacts-dir", default=str(_HERE / "artifacts"))
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("ÉTAPE B.5 — Baseline 2D MLE fixed")
    print("=" * 78)

    # ---- 1. Load MLE parameters ------------------------------------------
    r_est_path = artifacts_dir / "R_estimation.json"
    mle = _load_mle_params(r_est_path)
    print(f"\n[1/6] Paramètres MLE 2D chargés (fit sur {mle['n_samples_used']:,} samples) :")
    print(f"  σ²_proc = {mle['sigma2_proc']:.6g}")
    print(f"  R       = {mle['r_scalar']:.6g}")
    print(f"  NLL     = {mle['nll']:.2f}   success={mle['success']}")

    # ---- 2. Load splits + GT ---------------------------------------------
    print(f"\n[2/6] Chargement splits + GT officiel...")
    train, val, test, split_meta = make_splits(
        csv_path=args.csv, start_date=args.start_date, end_date=args.end_date,
    )
    rsi_full = np.concatenate([train.rsi, val.rsi, test.rsi])

    gt_slope_path = artifacts_dir / "gt_official_slope.npy"
    if not gt_slope_path.exists():
        raise FileNotFoundError(
            f"{gt_slope_path} introuvable. Lancer d'abord validate_gt_and_R.py."
        )
    gt_slope = np.load(gt_slope_path)
    assert len(gt_slope) == len(rsi_full), \
        f"Mismatch GT length {len(gt_slope)} vs RSI {len(rsi_full)}"

    # Split GT
    gt_val = gt_slope[val.idx_start:val.idx_end]
    gt_test = gt_slope[test.idx_start:test.idx_end]
    print(f"  N={len(rsi_full):,}  train={train.n:,}  val={val.n:,}  test={test.n:,}")

    # ---- 3. Run 2D MLE fixed (primary) ------------------------------------
    print(f"\n[3/6] KF 2D MLE fixed — σ²={mle['sigma2_proc']:.4g}, R={mle['r_scalar']:.4g}...")
    kf_fixed = _run_fixed_kf(
        rsi_full,
        sigma2_fixed=mle["sigma2_proc"],
        r_fixed=mle["r_scalar"],
    )
    print(f"  slope mean={np.nanmean(kf_fixed.slope):.5f}  std={np.nanstd(kf_fixed.slope):.5f}")

    # ---- 4. Re-run original adaptive baseline for comparison vs NEW GT ----
    print(f"\n[4/6] KF 2D original adaptatif (R=0.1, σ²∈[0.001, 0.1]) — pour comparaison vs GT officiel...")
    kf_adaptive = run_kf_baseline(rsi_full)  # defaults = original adaptive config
    print(f"  slope mean={np.nanmean(kf_adaptive.slope):.5f}  std={np.nanstd(kf_adaptive.slope):.5f}")

    # ---- 5. Metrics + diagnostics -----------------------------------------
    print(f"\n[5/6] Métriques et diagnostic d'innovations...")
    slope_fixed_train = kf_fixed.slope[:train.idx_end]
    slope_fixed_val = kf_fixed.slope[val.idx_start:val.idx_end]
    slope_fixed_test = kf_fixed.slope[test.idx_start:test.idx_end]

    slope_adaptive_val = kf_adaptive.slope[val.idx_start:val.idx_end]
    slope_adaptive_test = kf_adaptive.slope[test.idx_start:test.idx_end]

    metrics_fixed = {
        "val": _compute_split_metrics(slope_fixed_val, gt_val, "val"),
        "test": _compute_split_metrics(slope_fixed_test, gt_test, "test"),
    }
    metrics_adaptive = {
        "val": _compute_split_metrics(slope_adaptive_val, gt_val, "val"),
        "test": _compute_split_metrics(slope_adaptive_test, gt_test, "test"),
    }

    print("\n  === Baseline 2D MLE fixed (PRIMARY) vs GT officiel ===")
    print(_format_metric_row("[val] fixed", metrics_fixed["val"]))
    print(_format_metric_row("[test] fixed", metrics_fixed["test"]))

    print("\n  === Baseline 2D adaptive original (R=0.1) vs GT officiel ===")
    print(_format_metric_row("[val] adaptive", metrics_adaptive["val"]))
    print(_format_metric_row("[test] adaptive", metrics_adaptive["test"]))

    print("\n  === Δ (fixed − adaptive) ===")
    for s in ("val", "test"):
        mf, ma = metrics_fixed[s], metrics_adaptive[s]
        print(
            f"    [{s}]  ΔMSE={mf['mse']-ma['mse']:+.5g}  "
            f"ΔPearson={mf['pearson']-ma['pearson']:+.4f}  "
            f"ΔDirMatch={mf['direction_match']-ma['direction_match']:+.4f}"
        )

    # Diagnostic on MLE fixed train innovations
    v_train = kf_fixed.innovations[:train.idx_end]
    S_train = kf_fixed.S[:train.idx_end]
    diag = run_diagnostic(v_train, S_train, split_name="train_mle_fixed")

    print(f"\n  === Diagnostic innovations (MLE fixed, train) ===")
    print(f"    n valides = {diag.n:,}")
    print(f"    mean(z) = {diag.mean:+.4f}   std(z) = {diag.std:.4f}")
    print(f"    ACF(1..10) = [" + ", ".join(f"{v:+.4f}" for v in diag.acf_1_to_10) + "]")
    print(f"    max|ACF(1..10)| = {diag.acf_max_abs_1_10:.4f}")
    print(f"    LB h=10 : Q={diag.ljung_box_h10['statistic']:.2f}  p={diag.ljung_box_h10['p_value']:.3e}  Q/n={diag.ljung_box_h10['q_per_n']:.4f}")
    print(f"    JB : stat={diag.jarque_bera['statistic']:.2f}  skew={diag.jarque_bera['skewness']:.3f}  excess_kurt={diag.jarque_bera['excess_kurtosis']:.3f}")

    # Generate plots
    plots_dir = artifacts_dir / "step1_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_files = make_plots(v_train, S_train, plots_dir, prefix="baseline_2d_mle_fixed")

    # ---- 6. Predictions table & verdict ----------------------------------
    print(f"\n[6/6] Vérification prédictions ex-ante :")

    predictions_rows = []
    # Prediction 1: std(z)
    pred_zstd = "[0.95, 1.05] (strict) / [0.85, 1.15] (tolérant)"
    obs_zstd = f"{diag.std:.4f}"
    if 0.95 <= diag.std <= 1.05:
        v_zstd = "✅ strict"
    elif 0.85 <= diag.std <= 1.15:
        v_zstd = "✅ tolérant"
    else:
        v_zstd = "❌ HORS plage"
    predictions_rows.append({"metric": "std(z)", "predicted": pred_zstd, "observed": obs_zstd, "verdict": v_zstd})
    print(f"  std(z) = {diag.std:.4f}   [prédit 0.95-1.05 / toléré 0.85-1.15]   {v_zstd}")

    # Prediction 2: max|ACF(1..10)|
    pred_acf = "[0.05, 0.15]"
    obs_acf = f"{diag.acf_max_abs_1_10:.4f}"
    if 0.05 <= diag.acf_max_abs_1_10 <= 0.15:
        v_acf = "✅ dans la plage"
    elif diag.acf_max_abs_1_10 < 0.05:
        v_acf = "✅ meilleur que prévu"
    else:
        v_acf = "⚠️ > 0.15 → 2D CV structurellement inadéquat (valide Étape 2)"
    predictions_rows.append({"metric": "max|ACF(1..10)|", "predicted": pred_acf, "observed": obs_acf, "verdict": v_acf})
    print(f"  max|ACF(1..10)| = {diag.acf_max_abs_1_10:.4f}   [prédit 0.05-0.15]   {v_acf}")

    # Prediction 3: MSE val
    pred_mse = "≤ 1.0"
    obs_mse = f"{metrics_fixed['val']['mse']:.4f}"
    if metrics_fixed["val"]["mse"] <= 1.0:
        v_mse = "✅"
    else:
        v_mse = "⚠️ au-dessus du prévu (vérifier si signal réel)"
    predictions_rows.append({"metric": "MSE val (vs GT)", "predicted": pred_mse, "observed": obs_mse, "verdict": v_mse})
    print(f"  MSE val         = {metrics_fixed['val']['mse']:.4f}   [prédit ≤ 1.0]   {v_mse}")

    # Global summary
    all_pass = all(r["verdict"].startswith("✅") for r in predictions_rows)
    any_warn = any("⚠️" in r["verdict"] or "❌" in r["verdict"] for r in predictions_rows)

    if all_pass:
        summary = (
            "Toutes les prédictions ex-ante se confirment. Le baseline 2D MLE est "
            "correctement calibré. Étape 2 (3D WNA) peut démarrer avec confiance."
        )
    elif any_warn and diag.acf_max_abs_1_10 > 0.15:
        summary = (
            f"std(z) OK mais max|ACF|={diag.acf_max_abs_1_10:.4f} > 0.15 : le 2D CV est "
            f"structurellement inadéquat. L'Étape 2 (3D WNA) est justifiée avec haute "
            f"confiance — on s'attend à une réduction substantielle de l'ACF résiduelle."
        )
    else:
        summary = (
            f"Résultats à examiner manuellement avant Étape 2 : "
            f"std(z)={diag.std:.3f}, max|ACF|={diag.acf_max_abs_1_10:.4f}, "
            f"MSE val={metrics_fixed['val']['mse']:.4f}."
        )

    predictions_table = {"rows": predictions_rows, "summary": summary}

    print(f"\n  → {summary}")

    # ---- 7. Save artifacts ------------------------------------------------
    np.save(artifacts_dir / "baseline_2d_mle_slope_train.npy", slope_fixed_train)
    np.save(artifacts_dir / "baseline_2d_mle_slope_val.npy", slope_fixed_val)
    np.save(artifacts_dir / "baseline_2d_mle_slope_test.npy", slope_fixed_test)
    np.save(artifacts_dir / "baseline_2d_mle_innovations_train.npy", v_train)
    np.save(artifacts_dir / "baseline_2d_mle_innov_S_train.npy", S_train)

    diag_blob = {
        "mle_params": mle,
        "metrics_fixed_vs_gt_official": metrics_fixed,
        "metrics_adaptive_vs_gt_official": metrics_adaptive,
        "delta_fixed_minus_adaptive": {
            s: {
                "mse": metrics_fixed[s]["mse"] - metrics_adaptive[s]["mse"],
                "pearson": metrics_fixed[s]["pearson"] - metrics_adaptive[s]["pearson"],
                "direction_match": metrics_fixed[s]["direction_match"] - metrics_adaptive[s]["direction_match"],
            } for s in ("val", "test")
        },
        "innovation_diagnostic_train": diag.to_dict(),
        "predictions": predictions_table,
    }
    diag_path = artifacts_dir / "baseline_2d_mle_diagnostics.json"
    with diag_path.open("w", encoding="utf-8") as f:
        json.dump(diag_blob, f, indent=2, default=str, ensure_ascii=False)

    # Markdown report (separate file; does NOT overwrite run_experiments' report)
    md_path = artifacts_dir / "baseline_2d_mle_report.md"
    write_report(md_path, mle, metrics_fixed, metrics_adaptive, diag, predictions_table, plot_files)

    print(f"\nArtefacts :")
    print(f"  - baseline_2d_mle_slope_{{train,val,test}}.npy")
    print(f"  - baseline_2d_mle_innovations_train.npy, baseline_2d_mle_innov_S_train.npy")
    print(f"  - {diag_path.name}")
    print(f"  - {md_path.name}")
    if plot_files:
        print(f"  - {', '.join(p.name for p in plot_files)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
