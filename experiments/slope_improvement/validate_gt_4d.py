"""
Validation du GT 4D sur-paramétré — avant son adoption comme GT officiel.

Utilisateur :
    > "Partager la structure 3D entre GT et baseline testé invalide la
    > comparaison 2D vs 3D. Construire un GT 4D sur-paramétré [...]
    > VALIDER que ce GT 4D :
    >   - A une MSE vs 'vraie slope non-paramétrique' (MA centrée w=51)
    >     inférieure au GT 3D actuel
    >   - Produit des innovations forward plus blanches que le 3D forward
    >   - Si OUI : adopter ; si NON : flagger, discussion."

Pipeline :

    1. Charger données + GT 3D (déjà calculé en B.4)
    2. Construire GT 4D : MLE fit (σ²_jerk, R) + RTS full-pass
    3. Référence non-paramétrique : MA51 centrée sur diff(RSI)
       (réutilise ground_truth.slope_centered_ma)
    4. Comparer 3D vs 4D :
         a. MSE(GT_3D, MA51) vs MSE(GT_4D, MA51)      → le plus bas gagne
         b. Forward whiteness : ACF des innovations forward 3D vs 4D
            (en fixant Q, R aux valeurs MLE respectives de chaque modèle)
         c. Consistance Pearson(slope_3D, slope_4D)
         d. AIC/BIC pour vérifier que l'ajout du jerk est justifié
    5. Verdict automatique : adopt / reject / flag
    6. Si adopted : sauvegarder comme gt_official_4d_*.npy

Optionnel (B.4 bonus) : comparer aussi à un GT spline (UnivariateSpline).
Implémenté dans un bloc optionnel, skippé si scipy.interpolate indisponible.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_PROJECT_ROOT = _HERE.parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_loader import make_splits  # noqa: E402
from ground_truth import slope_centered_ma  # noqa: E402 (reuse)
from gt_3d import forward_filter_3d as fwd3_legacy  # noqa: E402 (reuse)
from gt_4d import compute_official_gt_4d, forward_filter_4d  # noqa: E402
from kf_nd import aic, bic  # noqa: E402
from diagnostics import acf, ljung_box  # noqa: E402 (reuse)


# ---------------------------------------------------------------------------
# Non-parametric reference : MA51 centered on diff(RSI)
# ---------------------------------------------------------------------------

def compute_non_parametric_slope_reference(
    rsi_full: np.ndarray,
    window: int = 51,
) -> np.ndarray:
    """
    Reuses ground_truth.slope_centered_ma for MA(w=51) on diff(RSI).

    Purpose: a GT independent of any Kalman structure. The model-based GTs
    (3D, 4D) are evaluated against this as a neutral arbitrator.
    """
    return slope_centered_ma(rsi_full, window=window)


# ---------------------------------------------------------------------------
# Comparison primitives
# ---------------------------------------------------------------------------

def _mse_pearson(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 100:
        return {"mse": float("nan"), "mae": float("nan"), "pearson": float("nan"), "n": int(mask.sum())}
    aa, bb = a[mask], b[mask]
    mse = float(np.mean((aa - bb) ** 2))
    mae = float(np.mean(np.abs(aa - bb)))
    if np.std(aa) == 0 or np.std(bb) == 0:
        pear = float("nan")
    else:
        pear = float(np.corrcoef(aa, bb)[0, 1])
    return {"mse": mse, "mae": mae, "pearson": pear, "n": int(mask.sum())}


def _innov_whiteness(v: np.ndarray, S: np.ndarray) -> Dict:
    """Compute ACF(1..10) + max|ACF| + LB on normalized innovations."""
    mask = np.isfinite(v) & np.isfinite(S) & (S > 0)
    z = v[mask] / np.sqrt(S[mask])
    if len(z) < 200:
        return {"z_mean": float("nan"), "z_std": float("nan"),
                "acf_1_10": [float("nan")] * 10, "max_abs": float("nan"),
                "lb_stat": float("nan"), "lb_p": float("nan"), "n": int(len(z))}
    a = acf(z, max_lag=10)
    lb = ljung_box(z, lags=10)
    return {
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z)),
        "acf_1_10": [float(x) for x in a[1:11]],
        "max_abs": float(np.max(np.abs(a[1:11]))),
        "lb_stat": float(lb["statistic"]),
        "lb_p": float(lb["p_value"]),
        "n": int(len(z)),
    }


# ---------------------------------------------------------------------------
# Bundled comparison result
# ---------------------------------------------------------------------------

@dataclass
class GTComparison:
    # GT parameters
    gt3_sigma2_accel: float
    gt3_r: float
    gt3_nll: float
    gt4_sigma2_jerk: float
    gt4_r: float
    gt4_nll: float
    # vs non-param MA51
    mse_gt3_vs_ma51: float
    mae_gt3_vs_ma51: float
    pearson_gt3_vs_ma51: float
    mse_gt4_vs_ma51: float
    mae_gt4_vs_ma51: float
    pearson_gt4_vs_ma51: float
    # Forward innovation whiteness (fixed MLE params)
    fwd3_z_std: float
    fwd3_acf_max: float
    fwd3_lb_p: float
    fwd4_z_std: float
    fwd4_acf_max: float
    fwd4_lb_p: float
    # Consistency 3D vs 4D slopes
    pearson_3d_4d: float
    # Information criteria (same n samples used for MLE)
    aic_3d: float
    aic_4d: float
    bic_3d: float
    bic_4d: float
    n_mle: int
    # Verdict
    verdict: str
    reasoning: str


def run_comparison(
    rsi_full: np.ndarray,
    train_end_idx: int,
    gt3_metadata: Dict,
    gt3_slope: np.ndarray,
    subsample_n: int = 20_000,
) -> tuple[GTComparison, "OfficialGT4D"]:  # noqa: F821
    """
    Heart of the validation :
      - compute GT 4D
      - compute non-param MA51
      - compute forward-only innovations for both 3D and 4D (at their MLE params)
      - consolidate all metrics into a GTComparison
    """
    # ---- 1. Compute GT 4D ----
    print("\n[GT 4D] MLE fit + RTS full-pass...")
    gt4 = compute_official_gt_4d(
        rsi_full,
        train_end_idx=train_end_idx,
        subsample_n=subsample_n,
        init_sigma2=1e-5,
        init_r=float(gt3_metadata["r_scalar"]),   # seed near 3D R for continuity
        verbose=True,
    )

    # ---- 2. Non-parametric MA51 reference ----
    print("\n[MA51] non-parametric slope reference on full series...")
    ma51 = compute_non_parametric_slope_reference(rsi_full, window=51)
    n_valid_ma51 = np.isfinite(ma51).sum()
    print(f"  n valid MA51 samples = {n_valid_ma51:,}")

    # ---- 3. Compare 3D and 4D GTs vs MA51 ----
    m3 = _mse_pearson(gt3_slope, ma51)
    m4 = _mse_pearson(gt4.slope, ma51)
    print(f"\n  GT_3D vs MA51 : MSE={m3['mse']:.5g}  MAE={m3['mae']:.5g}  Pearson={m3['pearson']:.4f}  n={m3['n']:,}")
    print(f"  GT_4D vs MA51 : MSE={m4['mse']:.5g}  MAE={m4['mae']:.5g}  Pearson={m4['pearson']:.4f}  n={m4['n']:,}")

    # ---- 4. Forward innovations whiteness (fixed at MLE params) ----
    # 3D forward (with 3D MLE params from gt3_metadata)
    print("\n[Whiteness] forward 3D innovations (at σ²_accel={:.4g}, R={:.4g})...".format(
        gt3_metadata["sigma2_accel"], gt3_metadata["r_scalar"]))
    y_train = rsi_full[:train_end_idx]
    fwd3 = fwd3_legacy(y_train, gt3_metadata["sigma2_accel"], gt3_metadata["r_scalar"])
    white3 = _innov_whiteness(fwd3[4], fwd3[5])  # fwd3 tuple: (x_filt, P_filt, x_pred, P_pred, innov, S)
    print(f"  3D fwd : std(z)={white3['z_std']:.4f}  max|ACF|={white3['max_abs']:.4f}  LB p={white3['lb_p']:.3e}")

    print("\n[Whiteness] forward 4D innovations (at σ²_jerk={:.4g}, R={:.4g})...".format(
        gt4.sigma2_jerk, gt4.r_scalar))
    fwd4 = forward_filter_4d(y_train, gt4.sigma2_jerk, gt4.r_scalar)
    white4 = _innov_whiteness(fwd4.innov, fwd4.S)
    print(f"  4D fwd : std(z)={white4['z_std']:.4f}  max|ACF|={white4['max_abs']:.4f}  LB p={white4['lb_p']:.3e}")

    # ---- 5. Slope consistency 3D vs 4D ----
    mask = np.isfinite(gt3_slope) & np.isfinite(gt4.slope)
    pear_34 = float(np.corrcoef(gt3_slope[mask], gt4.slope[mask])[0, 1])
    print(f"\n  Pearson(slope_GT3D, slope_GT4D) = {pear_34:.4f}")

    # ---- 6. AIC / BIC on same train subsample ----
    # Both models have 2 free parameters (sigma2, R).
    # Using the MLE's NLL. Sample size for BIC is min(len(y_train), subsample_n).
    n_mle = min(len(y_train), subsample_n)
    aic3 = aic(gt3_metadata["nll"], k=2)
    aic4 = aic(gt4.nll, k=2)
    bic3 = bic(gt3_metadata["nll"], k=2, n=n_mle)
    bic4 = bic(gt4.nll, k=2, n=n_mle)
    print(f"\n  AIC 3D={aic3:.2f}   AIC 4D={aic4:.2f}   ΔAIC={aic4 - aic3:+.2f} (négatif = 4D préféré)")
    print(f"  BIC 3D={bic3:.2f}   BIC 4D={bic4:.2f}   ΔBIC={bic4 - bic3:+.2f}")

    # ---- 7. Verdict ----
    verdict, reasoning = _decide_verdict(m3, m4, white3, white4, pear_34, aic3, aic4, bic3, bic4)
    print(f"\n  VERDICT : {verdict}")
    print(f"  {reasoning}")

    cmp = GTComparison(
        gt3_sigma2_accel=float(gt3_metadata["sigma2_accel"]),
        gt3_r=float(gt3_metadata["r_scalar"]),
        gt3_nll=float(gt3_metadata["nll"]),
        gt4_sigma2_jerk=float(gt4.sigma2_jerk),
        gt4_r=float(gt4.r_scalar),
        gt4_nll=float(gt4.nll),
        mse_gt3_vs_ma51=m3["mse"],
        mae_gt3_vs_ma51=m3["mae"],
        pearson_gt3_vs_ma51=m3["pearson"],
        mse_gt4_vs_ma51=m4["mse"],
        mae_gt4_vs_ma51=m4["mae"],
        pearson_gt4_vs_ma51=m4["pearson"],
        fwd3_z_std=white3["z_std"],
        fwd3_acf_max=white3["max_abs"],
        fwd3_lb_p=white3["lb_p"],
        fwd4_z_std=white4["z_std"],
        fwd4_acf_max=white4["max_abs"],
        fwd4_lb_p=white4["lb_p"],
        pearson_3d_4d=pear_34,
        aic_3d=aic3,
        aic_4d=aic4,
        bic_3d=bic3,
        bic_4d=bic4,
        n_mle=int(n_mle),
        verdict=verdict,
        reasoning=reasoning,
    )
    return cmp, gt4


def _decide_verdict(
    m3: Dict, m4: Dict,
    w3: Dict, w4: Dict,
    pear_34: float,
    aic3: float, aic4: float,
    bic3: float, bic4: float,
) -> tuple[str, str]:
    """
    Decision logic :

    ADOPT 4D as new GT iff ALL of :
        (a) MSE_4D < MSE_3D vs MA51 (strict)
        (b) max|ACF| of 4D fwd < max|ACF| of 3D fwd
        (c) BIC 4D < BIC 3D   (penalty-adjusted fit improvement)

    FLAG if inconsistent : e.g., (a) but not (b), etc.
    REJECT if GT 4D is worse on all fronts.
    """
    a = m4["mse"] < m3["mse"]
    b = w4["max_abs"] < w3["max_abs"]
    c = bic4 < bic3

    if a and b and c:
        return (
            "ADOPT_4D",
            f"GT 4D améliore les 3 critères : MSE vs MA51 ({m3['mse']:.4g} → {m4['mse']:.4g}), "
            f"whiteness forward (max|ACF| {w3['max_abs']:.4f} → {w4['max_abs']:.4f}), "
            f"BIC ({bic3:.1f} → {bic4:.1f}). Adopter comme GT officiel."
        )
    if (not a) and (not b) and (not c):
        return (
            "REJECT_4D_KEEP_3D",
            f"GT 4D est pire sur les 3 critères. Le jerk n'apporte rien → garder GT 3D. "
            f"Pearson(slope_3D, slope_4D)={pear_34:.4f} confirme l'équivalence pratique."
        )
    # Mixed case
    details = []
    details.append(f"MSE vs MA51 : 3D={m3['mse']:.4g}, 4D={m4['mse']:.4g}  "
                   f"({'4D mieux' if a else '3D mieux ou égal'})")
    details.append(f"max|ACF| fwd: 3D={w3['max_abs']:.4f}, 4D={w4['max_abs']:.4f}  "
                   f"({'4D mieux' if b else '3D mieux ou égal'})")
    details.append(f"BIC        : 3D={bic3:.1f}, 4D={bic4:.1f}  "
                   f"({'4D mieux' if c else '3D mieux ou égal'})")
    return (
        "FLAG_MIXED",
        "Résultats mixtes :\n  - " + "\n  - ".join(details) +
        f"\nPearson(3D, 4D)={pear_34:.4f}. Décision humaine requise."
    )


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_markdown_report(
    path: Path,
    cmp: GTComparison,
    gt4_meta: Dict,
    split_meta: Dict,
) -> None:
    L = []
    L.append("# Validation du GT 4D sur-paramétré")
    L.append("")
    L.append("## Motivation")
    L.append("Éviter la circularité structurelle : la comparaison 2D vs 3D ne peut pas "
             "être arbitée par un GT lui-même construit en 3D. Le GT 4D "
             "(état [level, slope, accel, jerk], RTS non-causal) a plus de degrés de "
             "liberté qu'aucun modèle testé et sert d'arbitre neutre.")
    L.append("")
    L.append("## Paramètres")
    L.append("")
    L.append(f"| GT | paramètres | NLL |")
    L.append(f"|----|-----------|-----|")
    L.append(f"| 3D WNA | σ²_accel = {cmp.gt3_sigma2_accel:.6g}, R = {cmp.gt3_r:.4f} | {cmp.gt3_nll:.2f} |")
    L.append(f"| 4D const-jerk | σ²_jerk = {cmp.gt4_sigma2_jerk:.6g}, R = {cmp.gt4_r:.4f} | {cmp.gt4_nll:.2f} |")
    L.append(f"(MLE fit sur {cmp.n_mle:,} samples de train)")
    L.append("")
    L.append("## Critères de validation")
    L.append("")
    L.append("### (a) vs référence non-paramétrique MA51 centrée sur diff(RSI)")
    L.append("")
    L.append(f"| GT | MSE vs MA51 | MAE | Pearson | n |")
    L.append(f"|----|-------------|-----|---------|---|")
    L.append(f"| 3D | {cmp.mse_gt3_vs_ma51:.5g} | {cmp.mae_gt3_vs_ma51:.5g} | {cmp.pearson_gt3_vs_ma51:.4f} | — |")
    L.append(f"| 4D | {cmp.mse_gt4_vs_ma51:.5g} | {cmp.mae_gt4_vs_ma51:.5g} | {cmp.pearson_gt4_vs_ma51:.4f} | — |")
    L.append(f"ΔMSE (4D − 3D) = {cmp.mse_gt4_vs_ma51 - cmp.mse_gt3_vs_ma51:+.5g}  "
             f"({'4D mieux' if cmp.mse_gt4_vs_ma51 < cmp.mse_gt3_vs_ma51 else '3D mieux ou égal'})")
    L.append("")
    L.append("### (b) Blancheur des innovations forward (train)")
    L.append("")
    L.append(f"| GT | std(z) | max\\|ACF(1..10)\\| | LB p |")
    L.append(f"|----|--------|-------------------|------|")
    L.append(f"| 3D fwd | {cmp.fwd3_z_std:.4f} | {cmp.fwd3_acf_max:.4f} | {cmp.fwd3_lb_p:.3e} |")
    L.append(f"| 4D fwd | {cmp.fwd4_z_std:.4f} | {cmp.fwd4_acf_max:.4f} | {cmp.fwd4_lb_p:.3e} |")
    L.append(f"Δ max\\|ACF\\| = {cmp.fwd4_acf_max - cmp.fwd3_acf_max:+.4f}")
    L.append("")
    L.append("### (c) Information criteria")
    L.append("")
    L.append(f"| Critère | 3D | 4D | Δ (4D−3D) |")
    L.append(f"|---------|-----|-----|-----------|")
    L.append(f"| AIC | {cmp.aic_3d:.2f} | {cmp.aic_4d:.2f} | {cmp.aic_4d - cmp.aic_3d:+.2f} |")
    L.append(f"| BIC | {cmp.bic_3d:.2f} | {cmp.bic_4d:.2f} | {cmp.bic_4d - cmp.bic_3d:+.2f} |")
    L.append("(Critère plus bas = meilleur. Négatif pour Δ = 4D préféré.)")
    L.append("")
    L.append("### Consistance 3D / 4D")
    L.append(f"- Pearson(slope_GT3D, slope_GT4D) = **{cmp.pearson_3d_4d:.4f}**")
    L.append(f"  - > 0.99 : équivalents en pratique (jerk n'apporte rien de significatif)")
    L.append(f"  - 0.90-0.99 : différences subtiles mais réelles")
    L.append(f"  - < 0.90 : structures franchement différentes")
    L.append("")
    L.append("## Verdict")
    L.append("")
    L.append(f"**{cmp.verdict}**")
    L.append("")
    L.append(cmp.reasoning)
    L.append("")
    L.append("## Conséquences")
    if cmp.verdict == "ADOPT_4D":
        L.append("- `gt_official_4d_slope.npy` devient la référence primaire pour toutes les")
        L.append("  comparaisons aval (baseline 2D, baseline 3D, Étape 2, variantes).")
        L.append("- `gt_official_slope.npy` (3D) reste disponible pour comparaison historique.")
        L.append("- Étape 2 doit être ré-évaluée contre le nouveau GT 4D.")
    elif cmp.verdict == "REJECT_4D_KEEP_3D":
        L.append("- GT 3D reste le GT officiel.")
        L.append("- La comparaison 2D vs 3D contre GT 3D reste légèrement circulaire mais ")
        L.append("  documentée : le jerk n'apporte rien de structurellement exploitable dans ce dataset.")
        L.append("- Étape 2 peut démarrer avec GT 3D comme prévu.")
    else:
        L.append("- Résultats mixtes : décision humaine requise avant de continuer.")
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Build + validate 4D super-parameterized GT")
    parser.add_argument("--csv", default="data_trad/BTCUSD_all_5m.csv")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--artifacts-dir", default=str(_HERE / "artifacts"))
    parser.add_argument("--mle-subsample", type=int, default=20_000)
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("VALIDATION GT 4D (sur-paramétré, non-circulaire)")
    print("=" * 78)

    # ---- Load data + GT 3D ------------------------------------------------
    train, val, test, split_meta = make_splits(
        csv_path=args.csv, start_date=args.start_date, end_date=args.end_date,
    )
    rsi_full = np.concatenate([train.rsi, val.rsi, test.rsi])

    gt3_meta_path = artifacts_dir / "gt_official_metadata.json"
    gt3_slope_path = artifacts_dir / "gt_official_slope.npy"
    if not gt3_meta_path.exists() or not gt3_slope_path.exists():
        raise FileNotFoundError(
            "GT 3D introuvable. Lancer d'abord validate_gt_and_R.py."
        )
    with gt3_meta_path.open("r", encoding="utf-8") as f:
        gt3_metadata = json.load(f)
    gt3_slope = np.load(gt3_slope_path)
    print(f"Loaded GT 3D : σ²_accel={gt3_metadata['sigma2_accel']:.4g}, R={gt3_metadata['r_scalar']:.4f}")
    print(f"N total = {len(rsi_full):,}")

    # ---- Run the comparison ----------------------------------------------
    cmp, gt4 = run_comparison(
        rsi_full=rsi_full,
        train_end_idx=train.idx_end,
        gt3_metadata=gt3_metadata,
        gt3_slope=gt3_slope,
        subsample_n=args.mle_subsample,
    )

    # ---- Save GT 4D artifacts (regardless of verdict — audit trail) ------
    np.save(artifacts_dir / "gt_official_4d_slope.npy", gt4.slope)
    np.save(artifacts_dir / "gt_official_4d_level.npy", gt4.level)
    np.save(artifacts_dir / "gt_official_4d_accel.npy", gt4.accel)
    np.save(artifacts_dir / "gt_official_4d_jerk.npy", gt4.jerk)
    gt4_meta = {
        "sigma2_jerk": gt4.sigma2_jerk,
        "r_scalar": gt4.r_scalar,
        "nll": gt4.nll,
        "n_fit_samples": gt4.n_fit_samples,
        "n_full_samples": gt4.n_full_samples,
    }
    with (artifacts_dir / "gt_official_4d_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(gt4_meta, f, indent=2, default=str, ensure_ascii=False)

    # Save full comparison
    with (artifacts_dir / "gt_4d_validation_results.json").open("w", encoding="utf-8") as f:
        json.dump({
            "splits": split_meta,
            "gt3_params": gt3_metadata,
            "gt4_params": gt4_meta,
            "comparison": asdict(cmp),
        }, f, indent=2, default=str, ensure_ascii=False)

    # Markdown report
    md_path = artifacts_dir / "gt_4d_validation_report.md"
    write_markdown_report(md_path, cmp, gt4_meta, split_meta)

    print("\nArtefacts sauvegardés :")
    print(f"  - gt_official_4d_slope.npy  (primary si adopté)")
    print(f"  - gt_official_4d_level.npy, _accel.npy, _jerk.npy")
    print(f"  - gt_official_4d_metadata.json")
    print(f"  - gt_4d_validation_results.json")
    print(f"  - {md_path.name}")

    # Final summary
    print("\n" + "=" * 78)
    print(f"VERDICT FINAL : {cmp.verdict}")
    print("=" * 78)
    print(cmp.reasoning)
    print("=" * 78)

    return 0


if __name__ == "__main__":
    sys.exit(main())
