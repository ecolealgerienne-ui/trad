"""
Étape 4 — Fixed-Lag Smoother (FLKS) sweep sur baseline 2D MLE fixed.

Utilisateur — spécifications :
  Grid : lag ∈ {0, 1, 2, 3, 5, 8, 13, 21, 50, 200, ∞}
  Détection coude : dernier lag où ΔMSE relatif > 5%
  Références : GT 3D, GT 4D, MA51 (décision cohérente sur 3 refs)
  Diagnostic innovations au coude : std(z), ACF, LB
  Scénarios :
    X : coude ≤ lag 8 ET gain cumulé > 25%
        → FLKS(lag=coude) devient pipeline de référence
    Y : pas de coude net (décroissance monotone lente)
        → forward 2D quasi-optimal en causal
    Z : FLKS(lag=∞) plafonne bien au-dessus GT 3D
        → reconsidérer validité GT

Caveat méthodologique important (documenté) :
  kf_baseline.run_kf_baseline utilise Q = σ² · G·G^T avec G=[1,1]^T
  (rank-1), mais estimate_R.r_via_mle_2d a fitté σ²=1.155 sous
  Q = σ² · I (diagonal). Pour consistance avec le baseline B.5,
  on garde G=[1,1]^T rank-1 avec σ²=1.155. La std(z)=1.10 observée
  en B.5 montre que le décalage est marginal. Ce script utilise la
  même convention pour que les résultats FLKS s'enchaînent
  proprement avec B.5 et l'Étape 2.

Réutilise :
  - data_loader.make_splits
  - ground_truth.slope_centered_ma (MA51)
  - gt_3d / gt_4d : GT slopes chargés depuis .npy pré-existants
  - flks.fixed_lag_smoother + full_rts_smoother
  - metrics.compute_all
  - diagnostics.run_diagnostic
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_loader import make_splits  # noqa: E402
from ground_truth import slope_centered_ma  # noqa: E402
from flks import fixed_lag_smoother, full_rts_smoother, FLKSResult  # noqa: E402
from metrics import compute_all  # noqa: E402
from diagnostics import run_diagnostic, acf, ljung_box  # noqa: E402


# ---------------------------------------------------------------------------
# 2D CV model (consistent with kf_baseline.py rank-1 Q)
# ---------------------------------------------------------------------------

F_2 = np.array([[1.0, 1.0], [0.0, 1.0]])
H_2 = np.array([[1.0, 0.0]])
G_2 = np.array([[1.0], [1.0]])


# Lag grid requested by user
LAG_GRID: List[float] = [0, 1, 2, 3, 5, 8, 13, 21, 50, 200, np.inf]


# ---------------------------------------------------------------------------
# Per-lag result
# ---------------------------------------------------------------------------

@dataclass
class LagResult:
    lag: float                # np.inf for full RTS
    lag_label: str            # "inf" or str(int)
    # Per-ref per-split metrics
    metrics: Dict[str, Dict[str, Dict[str, float]]]  # metrics[split][ref] = {mse, mae, pearson, dir_match, n}
    # Forward residuals on train (for diagnostic at chosen lag)
    residuals_train_acf_1_10: List[float]
    residuals_train_acf_max: float
    residuals_train_std: float
    residuals_train_lb_p: float


def _run_lag(
    rsi_full: np.ndarray,
    lag: float,
    sigma2: float,
    r_scalar: float,
    train_idx_end: int,
) -> "FLKSResult":
    """Thin dispatcher on lag=inf vs finite lag."""
    if np.isinf(lag):
        return full_rts_smoother(rsi_full, F_2, H_2, G_2, sigma2, r_scalar)
    return fixed_lag_smoother(rsi_full, F_2, H_2, G_2, sigma2, r_scalar, lag=int(lag))


def _evaluate_lag(
    flks_result: FLKSResult,
    lag: float,
    train_idx_end: int,
    val_idx_start: int,
    val_idx_end: int,
    test_idx_start: int,
    test_idx_end: int,
    refs: Dict[str, Dict[str, np.ndarray]],
    rsi_full: np.ndarray,
) -> LagResult:
    """Compute per-split per-ref metrics + train residual ACF/LB."""
    slope = flks_result.x_smoothed[:, 1]

    slope_val = slope[val_idx_start:val_idx_end]
    slope_test = slope[test_idx_start:test_idx_end]
    slope_train = slope[:train_idx_end]

    metrics: Dict[str, Dict[str, Dict[str, float]]] = {"val": {}, "test": {}}
    for split_name, split_slope, split_range in [
        ("val", slope_val, (val_idx_start, val_idx_end)),
        ("test", slope_test, (test_idx_start, test_idx_end)),
    ]:
        for ref_name, ref_slopes in refs.items():
            m = compute_all(split_slope, ref_slopes[split_name])
            metrics[split_name][ref_name] = {
                "mse": m.mse, "mae": m.mae,
                "pearson": m.pearson, "dir_match": m.direction_match,
                "n": m.n_valid,
            }

    # Residuals diagnostic on train
    res_train = flks_result.residuals[:train_idx_end]
    res_finite = res_train[np.isfinite(res_train)]
    if len(res_finite) >= 100:
        # Treat residuals as "innovations-like": standardize by their own std
        std_r = float(np.std(res_finite))
        z = res_finite / (std_r + 1e-30)
        acf_vals = acf(z, max_lag=10)
        lb = ljung_box(z, lags=10)
        acf_1_10 = acf_vals[1:11].tolist()
        acf_max = float(np.max(np.abs(acf_vals[1:11])))
        lb_p = float(lb["p_value"])
        z_std = 1.0  # by construction
    else:
        acf_1_10 = [float("nan")] * 10
        acf_max = float("nan")
        lb_p = float("nan")
        z_std = float("nan")

    lag_label = "inf" if np.isinf(lag) else str(int(lag))
    return LagResult(
        lag=float(lag),
        lag_label=lag_label,
        metrics=metrics,
        residuals_train_acf_1_10=[float(v) for v in acf_1_10],
        residuals_train_acf_max=acf_max,
        residuals_train_std=z_std,
        residuals_train_lb_p=lb_p,
    )


# ---------------------------------------------------------------------------
# Elbow detection & scenario classification
# ---------------------------------------------------------------------------

def detect_elbow(
    lag_results: List[LagResult],
    ref_name: str = "GT_3D",
    split: str = "val",
    rel_threshold: float = 0.05,
) -> Dict:
    """
    Elbow = last finite lag L where the relative MSE decrease between
    the previous lag and L is > rel_threshold (= 5% by default).
    """
    # Sort finite lags in ascending order
    finite = [lr for lr in lag_results if not np.isinf(lr.lag)]
    finite_sorted = sorted(finite, key=lambda lr: lr.lag)
    mses = [lr.metrics[split][ref_name]["mse"] for lr in finite_sorted]
    lags = [lr.lag for lr in finite_sorted]

    # Relative decrease vs previous lag (index 0 has no previous)
    rel_decreases = [np.nan]
    for i in range(1, len(mses)):
        prev, cur = mses[i - 1], mses[i]
        if prev > 0 and np.isfinite(prev) and np.isfinite(cur):
            rel_decreases.append((prev - cur) / prev)
        else:
            rel_decreases.append(np.nan)

    # Elbow = last lag where rel_decrease > rel_threshold
    elbow_idx = 0  # default: lag=0 (no useful lag)
    for i in range(1, len(rel_decreases)):
        if np.isfinite(rel_decreases[i]) and rel_decreases[i] > rel_threshold:
            elbow_idx = i

    elbow_lag = lags[elbow_idx] if elbow_idx >= 0 else 0
    mse_at_0 = mses[0]
    mse_at_elbow = mses[elbow_idx]
    cumulative_gain_pct = (mse_at_0 - mse_at_elbow) / max(mse_at_0, 1e-30) * 100.0

    # Also info on full RTS (lag=inf)
    rts = [lr for lr in lag_results if np.isinf(lr.lag)]
    if rts:
        mse_rts = rts[0].metrics[split][ref_name]["mse"]
        gain_rts_vs_0 = (mse_at_0 - mse_rts) / max(mse_at_0, 1e-30) * 100.0
        gain_rts_vs_elbow = (mse_at_elbow - mse_rts) / max(mse_at_elbow, 1e-30) * 100.0
    else:
        mse_rts = float("nan")
        gain_rts_vs_0 = float("nan")
        gain_rts_vs_elbow = float("nan")

    return {
        "ref_used": ref_name,
        "split_used": split,
        "rel_threshold": rel_threshold,
        "lags": lags,
        "mses": [float(v) for v in mses],
        "rel_decreases": [float(v) if np.isfinite(v) else None for v in rel_decreases],
        "elbow_lag": float(elbow_lag),
        "elbow_idx": int(elbow_idx),
        "mse_at_lag0": float(mse_at_0),
        "mse_at_elbow": float(mse_at_elbow),
        "cumulative_gain_pct_elbow": float(cumulative_gain_pct),
        "mse_rts_inf": float(mse_rts),
        "cumulative_gain_pct_rts_inf": float(gain_rts_vs_0),
        "gain_pct_rts_vs_elbow": float(gain_rts_vs_elbow),
    }


def classify_scenario(elbow_info: Dict) -> Dict:
    """
    X : elbow_lag ≤ 8 AND cumulative_gain_pct_elbow ≥ 25 %
    Y : elbow_lag <= 8 but gain < 25, OR elbow > 8 with slow decrease, OR no elbow
    Z : FLKS(lag=∞) plafonne haut vs forward (gain_rts_vs_0 < 30%)

    Logique :
      - si Z vrai → scenario Z (même si X ou Y aussi vrais)
      - sinon si X vrai → scenario X
      - sinon Y
    """
    elbow = elbow_info["elbow_lag"]
    gain_elbow = elbow_info["cumulative_gain_pct_elbow"]
    gain_rts_inf = elbow_info["cumulative_gain_pct_rts_inf"]

    # Scenario Z
    if np.isfinite(gain_rts_inf) and gain_rts_inf < 30.0:
        return {
            "scenario": "Z",
            "verdict": "FLKS(lag=∞) plafonne près de forward → lissage bidirectionnel n'aide pas.",
            "recommended_action": (
                f"Gain lag=∞ = {gain_rts_inf:.1f}% < 30%. Reconsidérer validité GT ou pivot "
                f"sur R adaptatif / IMM (l'information backward est limitée)."
            ),
        }

    # Scenario X : clear elbow with meaningful gain
    if elbow <= 8 and gain_elbow >= 25.0:
        return {
            "scenario": "X",
            "verdict": f"Coude à lag={int(elbow)} avec gain cumulé {gain_elbow:.1f}% (≥25%).",
            "recommended_action": (
                f"FLKS(lag={int(elbow)}) devient le pipeline de référence pour l'estimation "
                f"de pente. Pratiquement exploitable si latence ≤ {int(elbow)} barres acceptable. "
                f"Projet essentiellement conclu sur l'axe KF."
            ),
        }

    # Scenario Y : slow/no clear elbow
    return {
        "scenario": "Y",
        "verdict": (
            f"Coude à lag={int(elbow)}, gain cumulé {gain_elbow:.1f}% "
            f"(critère X requiert ≤ 8 ET ≥ 25%). Pas de point de rendements décroissants nets."
        ),
        "recommended_action": (
            f"Le forward 2D est quasi-optimal en causal. Dernier recours : Option B "
            f"(R adaptatif sur 2D) avant conclusion finale."
        ),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _format_table(results: List[LagResult], ref: str = "GT_3D", split: str = "val") -> str:
    lines = []
    bar = "-" * 100
    lines.append(bar)
    lines.append(f"FLKS sweep — MSE vs {ref} on {split}")
    lines.append(bar)
    hdr = f"  {'lag':<6s} {'MSE':>10s} {'Δ% vs prev':>11s} {'MAE':>10s} {'Pearson':>9s} {'DirMatch':>9s} {'ACF max':>9s} {'LB p':>10s}"
    lines.append(hdr)
    lines.append(bar)
    prev_mse = None
    # Sort by lag ascending
    sorted_r = sorted(results, key=lambda lr: lr.lag)
    for lr in sorted_r:
        m = lr.metrics[split][ref]
        if prev_mse is not None and prev_mse > 0:
            rel = (prev_mse - m["mse"]) / prev_mse * 100
            rel_str = f"{rel:>+10.2f}%"
        else:
            rel_str = "       —"
        lines.append(
            f"  {lr.lag_label:<6s} {m['mse']:>10.5f} {rel_str:>11s} {m['mae']:>10.5f} "
            f"{m['pearson']:>9.4f} {m['dir_match']:>9.4f} {lr.residuals_train_acf_max:>9.4f} "
            f"{lr.residuals_train_lb_p:>10.2e}"
        )
        prev_mse = m["mse"]
    lines.append(bar)
    return "\n".join(lines)


def _write_markdown_report(
    path: Path,
    results: List[LagResult],
    elbow_info_per_ref: Dict[str, Dict],
    scenario: Dict,
    sigma2: float,
    r_scalar: float,
    split_meta: Dict,
) -> None:
    L = []
    L.append("# Étape 4 — FLKS sweep sur baseline 2D MLE fixed")
    L.append("")
    L.append("## Configuration")
    L.append(f"- Dataset : `{split_meta['csv_path']}` ({split_meta['start_date']} → fin)")
    L.append(f"- Splits : train={split_meta['n_train']:,}  val={split_meta['n_val']:,}  test={split_meta['n_test']:,}")
    L.append(f"- Modèle : 2D CV, F=[[1,1],[0,1]], H=[[1,0]], G=[1,1]^T (rank-1 Q)")
    L.append(f"- Paramètres fixés MLE : σ²_proc = {sigma2:.4g}, R = {r_scalar:.4f}")
    L.append("")
    L.append("> **Caveat** : le MLE 2D a été fitté sous Q = σ²·I₂ (diagonal), le baseline")
    L.append("> utilise Q = σ²·G·G^T rank-1. std(z)=1.10 en B.5 → décalage marginal accepté")
    L.append("> pour préserver la comparabilité avec B.5 et l'Étape 2.")
    L.append("")
    L.append("## Tableau principal — FLKS MSE vs 3 références")
    L.append("")
    # Combined table with MSE per ref
    sorted_r = sorted(results, key=lambda lr: lr.lag)
    L.append("| lag | MSE vs GT_3D (val) | MSE vs GT_4D (val) | MSE vs MA51 (val) | MSE vs GT_3D (test) | Pearson val (GT_3D) | DirM val (GT_3D) | ACF max train | LB p train |")
    L.append("|-----|--------------------|--------------------|-------------------|---------------------|---------------------|------------------|---------------|------------|")
    for lr in sorted_r:
        m_val_gt3 = lr.metrics["val"]["GT_3D"]
        m_val_gt4 = lr.metrics["val"]["GT_4D"]
        m_val_ma = lr.metrics["val"][list(lr.metrics["val"].keys())[-1]]  # MA51
        m_test_gt3 = lr.metrics["test"]["GT_3D"]
        L.append(
            f"| {lr.lag_label} | {m_val_gt3['mse']:.5f} | {m_val_gt4['mse']:.5f} | {m_val_ma['mse']:.5f} | "
            f"{m_test_gt3['mse']:.5f} | {m_val_gt3['pearson']:.4f} | {m_val_gt3['dir_match']:.4f} | "
            f"{lr.residuals_train_acf_max:.4f} | {lr.residuals_train_lb_p:.2e} |"
        )
    L.append("")
    L.append("## Détection du coude — critère quantitatif")
    L.append("")
    L.append("Coude = dernier lag où la baisse relative de MSE vs lag précédent dépasse 5%.")
    L.append("")
    L.append("| Référence | split | coude lag | MSE à lag=0 | MSE au coude | Gain cumulé (%) | MSE à lag=∞ | Gain lag=∞ vs 0 (%) |")
    L.append("|-----------|-------|-----------|-------------|--------------|------------------|-------------|----------------------|")
    for ref_name, info in elbow_info_per_ref.items():
        L.append(
            f"| {ref_name} | {info['split_used']} | {int(info['elbow_lag'])} | "
            f"{info['mse_at_lag0']:.5f} | {info['mse_at_elbow']:.5f} | {info['cumulative_gain_pct_elbow']:+.2f}% | "
            f"{info['mse_rts_inf']:.5f} | {info['cumulative_gain_pct_rts_inf']:+.2f}% |"
        )
    L.append("")
    L.append("## Scénario de sortie")
    L.append("")
    L.append(f"**{scenario['scenario']}**")
    L.append("")
    L.append(scenario["verdict"])
    L.append("")
    L.append(f"→ {scenario['recommended_action']}")
    L.append("")
    L.append("## Règles de décision (rappel)")
    L.append("")
    L.append("- **X** : coude ≤ lag 8 AND gain cumulé ≥ 25% → FLKS(lag=coude) pipeline officiel")
    L.append("- **Y** : pas de coude net (décroissance monotone lente) → forward 2D quasi-optimal en causal")
    L.append("- **Z** : FLKS(lag=∞) plafonne (gain < 30% vs lag=0) → reconsidérer validité GT")
    L.append("")
    L.append("Si Z détecté, il prime (ordre de priorité : Z, X, Y).")
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Étape 4: FLKS sweep")
    parser.add_argument("--csv", default="data_trad/BTCUSD_all_5m.csv")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--artifacts-dir", default=str(_HERE / "artifacts"))
    parser.add_argument("--ma-window", type=int, default=51)
    parser.add_argument("--elbow-ref", default="GT_3D", choices=["GT_3D", "GT_4D"],
                        help="Reference used for elbow detection (primary)")
    parser.add_argument("--elbow-split", default="val", choices=["val", "test"])
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("ÉTAPE 4 — FLKS sweep sur baseline 2D MLE fixed")
    print("=" * 78)

    # ---- 1. Load data + refs ---------------------------------------------
    print("\n[1/5] Chargement splits + références...")
    train, val, test, split_meta = make_splits(
        csv_path=args.csv, start_date=args.start_date, end_date=args.end_date,
    )
    rsi_full = np.concatenate([train.rsi, val.rsi, test.rsi])
    print(f"  N={len(rsi_full):,}  train={train.n:,}  val={val.n:,}  test={test.n:,}")

    gt3_slope = np.load(artifacts_dir / "gt_official_slope.npy")
    gt4_slope = np.load(artifacts_dir / "gt_official_4d_slope.npy")
    ma_slope = slope_centered_ma(rsi_full, window=args.ma_window)

    refs = {
        "GT_3D": {
            "val": gt3_slope[val.idx_start:val.idx_end],
            "test": gt3_slope[test.idx_start:test.idx_end],
        },
        "GT_4D": {
            "val": gt4_slope[val.idx_start:val.idx_end],
            "test": gt4_slope[test.idx_start:test.idx_end],
        },
        f"MA{args.ma_window}": {
            "val": ma_slope[val.idx_start:val.idx_end],
            "test": ma_slope[test.idx_start:test.idx_end],
        },
    }

    # ---- 2. Load 2D MLE params ---------------------------------------------
    print("\n[2/5] Chargement params MLE 2D...")
    with (artifacts_dir / "R_estimation.json").open("r", encoding="utf-8") as f:
        r_est = json.load(f)
    sigma2 = float(r_est["m4_mle"]["sigma2_proc"])
    r_scalar = float(r_est["m4_mle"]["R"])
    print(f"  σ²_proc = {sigma2:.4g}   R = {r_scalar:.4f}")
    print(f"  (Note : caveat rank-1 vs diagonal Q documenté dans le rapport)")

    # ---- 3. FLKS sweep -----------------------------------------------------
    print(f"\n[3/5] FLKS sweep sur grid {LAG_GRID} (lag_inf = full RTS)...")
    lag_results: List[LagResult] = []
    for L in LAG_GRID:
        import time
        t0 = time.time()
        flks = _run_lag(rsi_full, L, sigma2, r_scalar, train.idx_end)
        result = _evaluate_lag(
            flks, L,
            train_idx_end=train.idx_end,
            val_idx_start=val.idx_start, val_idx_end=val.idx_end,
            test_idx_start=test.idx_start, test_idx_end=test.idx_end,
            refs=refs, rsi_full=rsi_full,
        )
        dt = time.time() - t0
        mse_val = result.metrics["val"]["GT_3D"]["mse"]
        lag_str = "inf" if np.isinf(L) else str(int(L))
        print(f"  lag={lag_str:<4s}  MSE val (GT_3D)={mse_val:.5f}  ACF_max={result.residuals_train_acf_max:.4f}  ({dt:.1f}s)")
        lag_results.append(result)

        # Save slope for this lag
        slope_val = flks.x_smoothed[val.idx_start:val.idx_end, 1]
        slope_test = flks.x_smoothed[test.idx_start:test.idx_end, 1]
        np.save(artifacts_dir / f"flks_lag{lag_str}_slope_val.npy", slope_val)
        np.save(artifacts_dir / f"flks_lag{lag_str}_slope_test.npy", slope_test)

    # ---- 4. Elbow detection per ref ---------------------------------------
    print(f"\n[4/5] Détection coude (critère ΔMSE rel > 5%)...")
    elbow_info_per_ref: Dict[str, Dict] = {}
    for ref_name in refs.keys():
        info = detect_elbow(lag_results, ref_name=ref_name, split=args.elbow_split, rel_threshold=0.05)
        elbow_info_per_ref[ref_name] = info
        print(f"  {ref_name} ({args.elbow_split}) : coude lag={int(info['elbow_lag'])} | "
              f"MSE 0 → coude = {info['mse_at_lag0']:.5f} → {info['mse_at_elbow']:.5f} "
              f"(gain {info['cumulative_gain_pct_elbow']:+.2f}%) | "
              f"MSE lag=∞ = {info['mse_rts_inf']:.5f} (gain vs 0 : {info['cumulative_gain_pct_rts_inf']:+.2f}%)")

    # Primary scenario = based on primary ref
    scenario = classify_scenario(elbow_info_per_ref[args.elbow_ref])

    # ---- 5. Print summary tables ------------------------------------------
    print("\n" + _format_table(lag_results, ref="GT_3D", split="val"))
    print()
    print(_format_table(lag_results, ref="GT_4D", split="val"))
    print()
    print(_format_table(lag_results, ref=f"MA{args.ma_window}", split="val"))

    # Diagnostic at elbow
    elbow_lag = elbow_info_per_ref[args.elbow_ref]["elbow_lag"]
    elbow_result = next(
        (lr for lr in lag_results if int(lr.lag) == int(elbow_lag) and not np.isinf(lr.lag)),
        lag_results[0]
    )
    print(f"\n  Diagnostic résidus smoothed (train) au coude lag={int(elbow_lag)} :")
    print(f"    ACF(1..10) = [" + ", ".join(f"{v:+.4f}" for v in elbow_result.residuals_train_acf_1_10) + "]")
    print(f"    max|ACF(1..10)| = {elbow_result.residuals_train_acf_max:.4f}")
    print(f"    LB p = {elbow_result.residuals_train_lb_p:.3e}")

    # ---- 6. Scenario + verdict --------------------------------------------
    print("\n" + "=" * 78)
    print(f"VERDICT FINAL — Scenario {scenario['scenario']}")
    print("=" * 78)
    print(scenario["verdict"])
    print(f"\n→ {scenario['recommended_action']}")
    print("=" * 78)

    # ---- 7. Save artifacts + report ---------------------------------------
    out = {
        "splits": split_meta,
        "model": {
            "F": F_2.tolist(), "H": H_2.tolist(), "G": G_2.ravel().tolist(),
            "sigma2_proc": sigma2, "r_scalar": r_scalar,
            "caveat": "MLE fitted with diagonal Q, baseline uses rank-1 Q (G=[1,1]^T). std(z)=1.10 in B.5.",
        },
        "lag_grid": [("inf" if np.isinf(L) else int(L)) for L in LAG_GRID],
        "results": [
            {
                "lag": lr.lag_label,
                "metrics": lr.metrics,
                "residuals_train_acf_1_10": lr.residuals_train_acf_1_10,
                "residuals_train_acf_max": lr.residuals_train_acf_max,
                "residuals_train_lb_p": lr.residuals_train_lb_p,
            }
            for lr in lag_results
        ],
        "elbow_per_ref": elbow_info_per_ref,
        "scenario": scenario,
        "elbow_ref_used": args.elbow_ref,
        "elbow_split_used": args.elbow_split,
    }
    json_path = artifacts_dir / "etape4_flks_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str, ensure_ascii=False)

    md_path = artifacts_dir / "etape4_flks_report.md"
    _write_markdown_report(md_path, lag_results, elbow_info_per_ref, scenario, sigma2, r_scalar, split_meta)

    # Optional MSE vs lag plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plots_dir = artifacts_dir / "step1_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        for ref_name in refs.keys():
            info = elbow_info_per_ref[ref_name]
            ax.plot(info["lags"], info["mses"], marker="o", label=ref_name)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("lag (symlog)")
        ax.set_ylabel(f"MSE vs ref ({args.elbow_split})")
        ax.set_title("FLKS MSE vs lag (2D MLE fixed baseline)")
        ax.axvline(elbow_info_per_ref[args.elbow_ref]["elbow_lag"], color="red", linestyle="--",
                   label=f"elbow (primary: {args.elbow_ref})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_path = plots_dir / "etape4_flks_mse_vs_lag.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=100)
        plt.close(fig)
        print(f"  plot : {plot_path.name}")
    except ImportError:
        pass

    print(f"\nArtefacts :")
    print(f"  - {json_path.name}")
    print(f"  - {md_path.name}")
    print(f"  - flks_lag{{0,1,2,...,inf}}_slope_{{val,test}}.npy")

    return 0


if __name__ == "__main__":
    sys.exit(main())
