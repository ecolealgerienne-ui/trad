"""
Étape 2 — Comparaison 2D MLE fixed vs 3D MLE fixed, multi-références.

Utilisateur — règles de décision PRÉ-DÉFINIES (à appliquer automatiquement) :

    Soit ΔMSE(X) = MSE(3D vs X) − MSE(2D vs X) pour référence X.
    Négatif = 3D gagne ; positif = 2D gagne.

    CASE_1_STRONG   : signe cohérent 3/3 ET max/min magnitudes ≤ 2
                      → conclusion forte, retenir gagnant
    CASE_2_MODERATE : signe cohérent 3/3 ET 2 < max/min ≤ 3
                      → gain exists, ampleur dépend de la ref
    CASE_2_WIDE     : signe cohérent 3/3 ET max/min > 3
                      → gain existe, grande variabilité
    CASE_3_MA51     : 2/3 cohérents ET MA51 diverge
                      → artefact de lissage probable, trust 2 GTs
    CASE_3_GT       : 2/3 cohérents ET un GT diverge
                      → red flag structure, discussion humaine
    CASE_4          : signes divergents / pas de cohérence
                      → "gain dimensionnel" artefactuel, STOP

Références évaluées en parallèle :
    1. GT 3D  (primary, caveat circularité documenté)
    2. GT 4D  (secondary, robustness)
    3. MA51   (tertiary, non-paramétrique)

Baselines comparés :
    A. 2D MLE fixed    — σ²=1.155, R=3.27 (chargé depuis R_estimation.json)
    B. 3D MLE fixed    — σ²_accel=0.0717, R=6.16 (gt_official_metadata.json)

Réutilise :
    - data_loader.make_splits
    - gt_3d.forward_filter_3d  (pour 3D baseline)
    - kf_baseline.run_kf_baseline (2D, si cache absent)
    - ground_truth.slope_centered_ma (pour MA51)
    - metrics.compute_all + diebold_mariano
    - diagnostics.run_diagnostic + make_plots
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_loader import make_splits  # noqa: E402
from ground_truth import slope_centered_ma  # noqa: E402
from gt_3d import forward_filter_3d  # noqa: E402
from kf_baseline import run_kf_baseline  # noqa: E402
from metrics import compute_all, diebold_mariano  # noqa: E402
from diagnostics import run_diagnostic, make_plots  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class RefComparison:
    ref_name: str
    split: str
    mse_2d: float
    mse_3d: float
    delta_mse: float
    mae_2d: float
    mae_3d: float
    pearson_2d: float
    pearson_3d: float
    dirmatch_2d: float
    dirmatch_3d: float
    n: int
    winner: str
    dm_stat: float
    dm_p: float


def _compute_ref_metrics(
    slope_2d: np.ndarray,
    slope_3d: np.ndarray,
    ref_slope: np.ndarray,
    ref_name: str,
    split: str,
    tie_rel_tol: float = 0.005,
) -> RefComparison:
    m2 = compute_all(slope_2d, ref_slope)
    m3 = compute_all(slope_3d, ref_slope)
    d = m3.mse - m2.mse
    rel = abs(d) / max(min(m2.mse, m3.mse), 1e-30)
    if rel < tie_rel_tol:
        winner = "TIE"
    elif d < 0:
        winner = "3D_wins"
    else:
        winner = "2D_wins"
    dm_stat, dm_p = diebold_mariano(slope_3d, slope_2d, ref_slope, loss="mse")
    return RefComparison(
        ref_name=ref_name, split=split,
        mse_2d=m2.mse, mse_3d=m3.mse, delta_mse=d,
        mae_2d=m2.mae, mae_3d=m3.mae,
        pearson_2d=m2.pearson, pearson_3d=m3.pearson,
        dirmatch_2d=m2.direction_match, dirmatch_3d=m3.direction_match,
        n=m2.n_valid, winner=winner, dm_stat=dm_stat, dm_p=dm_p,
    )


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------

def classify(comparisons: List[RefComparison], split: str = "val") -> Dict:
    subset = [c for c in comparisons if c.split == split]
    assert len(subset) == 3, f"Expected 3 refs for split {split}, got {len(subset)}"

    deltas = np.array([c.delta_mse for c in subset])
    refs = [c.ref_name for c in subset]
    winners = [c.winner for c in subset]

    non_tie = [(r, d, w) for r, d, w in zip(refs, deltas, winners) if w != "TIE"]
    if not non_tie:
        return {
            "case": "CASE_4",
            "winner_overall": "NONE",
            "reasoning": "Tous les refs tombent en TIE (ΔMSE relatif < 0.5%). Pas de gain mesurable.",
            "split_used": split,
            "per_ref": [asdict(c) for c in subset],
        }

    signs = np.sign([d for _, d, _ in non_tie])
    n_neg = int(np.sum(signs < 0))
    n_pos = int(np.sum(signs > 0))
    majority_sign = -1 if n_neg > n_pos else (1 if n_pos > n_neg else 0)

    if majority_sign == 0:
        return {
            "case": "CASE_4",
            "winner_overall": "NONE",
            "reasoning": f"Signes équilibrés : {n_neg} pour 3D, {n_pos} pour 2D. Pas de direction claire.",
            "split_used": split,
            "per_ref": [asdict(c) for c in subset],
        }

    winner_overall = "3D" if majority_sign < 0 else "2D"
    all_agree = (n_neg == len(non_tie)) or (n_pos == len(non_tie))
    ratio = float("nan")
    case = None
    reasoning = ""

    if all_agree:
        mags = np.abs(np.array([d for _, d, _ in non_tie]))
        if mags.size > 0 and mags.min() > 0:
            ratio = float(mags.max() / mags.min())
        else:
            ratio = float("inf")

        if ratio <= 2.0:
            case = "CASE_1_STRONG"
            reasoning = (
                f"Signe 3/3 cohérent → {winner_overall} gagne sur les 3 refs. "
                f"Ratio mag max/min = {ratio:.2f} ≤ 2 → conclusion FORTE."
            )
        elif ratio <= 3.0:
            case = "CASE_2_MODERATE"
            reasoning = (
                f"Signe 3/3 cohérent → {winner_overall} gagne. "
                f"Ratio mag = {ratio:.2f} ∈ (2, 3] → conclusion modérée."
            )
        else:
            case = "CASE_2_WIDE"
            reasoning = (
                f"Signe 3/3 cohérent → {winner_overall} gagne. "
                f"Ratio mag = {ratio:.2f} > 3 → gain variable selon la ref."
            )
    else:
        divergent = [r for r, d, _ in non_tie if np.sign(d) != majority_sign]
        div_name = divergent[0] if divergent else "?"
        if div_name.startswith("MA"):
            case = "CASE_3_MA51"
            reasoning = (
                f"Signe 2/3 : {winner_overall} gagne sur les 2 GTs mais PAS sur {div_name}. "
                f"MA étant passe-bas très large, divergence probablement artefact de lissage. "
                f"Faire confiance aux 2 GTs."
            )
        else:
            case = "CASE_3_GT"
            reasoning = (
                f"Signe 2/3 : {winner_overall} gagne sur 2 refs mais PAS sur {div_name} (un GT). "
                f"RED FLAG — la structure de {div_name} contredit les autres. Décision humaine requise."
            )

    return {
        "case": case,
        "winner_overall": winner_overall,
        "reasoning": reasoning,
        "split_used": split,
        "majority_sign": int(majority_sign),
        "n_agree": int(max(n_neg, n_pos)),
        "ratio_max_min": ratio,
        "per_ref": [asdict(c) for c in subset],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Étape 2 multi-références")
    parser.add_argument("--csv", default="data_trad/BTCUSD_all_5m.csv")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--artifacts-dir", default=str(_HERE / "artifacts"))
    parser.add_argument("--ma-window", type=int, default=51)
    parser.add_argument("--decision-split", default="val", choices=["val", "test"])
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("ÉTAPE 2 — 2D MLE fixed vs 3D MLE fixed, multi-références")
    print("=" * 78)

    # ---- 1. Splits + refs -------------------------------------------------
    print("\n[1/5] Chargement splits + références...")
    train, val, test, split_meta = make_splits(
        csv_path=args.csv, start_date=args.start_date, end_date=args.end_date,
    )
    rsi_full = np.concatenate([train.rsi, val.rsi, test.rsi])

    gt3_slope = np.load(artifacts_dir / "gt_official_slope.npy")
    gt4_slope = np.load(artifacts_dir / "gt_official_4d_slope.npy")
    ma_slope = slope_centered_ma(rsi_full, window=args.ma_window)
    print(f"  GT 3D : mean={np.nanmean(gt3_slope):+.5f}  std={np.nanstd(gt3_slope):.5f}")
    print(f"  GT 4D : mean={np.nanmean(gt4_slope):+.5f}  std={np.nanstd(gt4_slope):.5f}")
    print(f"  MA{args.ma_window}  : mean={np.nanmean(ma_slope):+.5f}  std={np.nanstd(ma_slope):.5f}  n_valid={np.isfinite(ma_slope).sum():,}")

    # ---- 2. MLE params ----------------------------------------------------
    print("\n[2/5] Chargement params MLE...")
    with (artifacts_dir / "R_estimation.json").open("r", encoding="utf-8") as f:
        r_est = json.load(f)
    sigma2_2d = float(r_est["m4_mle"]["sigma2_proc"])
    r_2d = float(r_est["m4_mle"]["R"])

    with (artifacts_dir / "gt_official_metadata.json").open("r", encoding="utf-8") as f:
        gt3_meta = json.load(f)
    sigma2_3d = float(gt3_meta["sigma2_accel"])
    r_3d = float(gt3_meta["r_scalar"])

    print(f"  2D : σ²_proc  = {sigma2_2d:.4g}   R = {r_2d:.4f}")
    print(f"  3D : σ²_accel = {sigma2_3d:.4g}   R = {r_3d:.4f}")

    # ---- 3. Run baselines -------------------------------------------------
    # Try to reuse cached 2D slope
    p2t = artifacts_dir / "baseline_2d_mle_slope_train.npy"
    p2v = artifacts_dir / "baseline_2d_mle_slope_val.npy"
    p2te = artifacts_dir / "baseline_2d_mle_slope_test.npy"
    if p2t.exists() and p2v.exists() and p2te.exists():
        print("\n[3a/5] Baseline 2D MLE : cache détecté, chargement...")
        slope_2d_train = np.load(p2t)
        slope_2d_val = np.load(p2v)
        slope_2d_test = np.load(p2te)
    else:
        print("\n[3a/5] Baseline 2D MLE fixed : run...")
        kf2 = run_kf_baseline(
            rsi_full,
            sigma2_init=sigma2_2d, sigma2_min=sigma2_2d, sigma2_max=sigma2_2d,
            r_scalar=r_2d,
        )
        slope_2d_train = kf2.slope[:train.idx_end]
        slope_2d_val = kf2.slope[val.idx_start:val.idx_end]
        slope_2d_test = kf2.slope[test.idx_start:test.idx_end]

    print("[3b/5] Baseline 3D MLE fixed : forward filter 3D...")
    fwd = forward_filter_3d(rsi_full, sigma2_3d, r_3d)
    x_filt = fwd[0]
    v_3d_train = fwd[4][:train.idx_end]
    S_3d_train = fwd[5][:train.idx_end]
    slope_3d_full = x_filt[:, 1]
    slope_3d_train = slope_3d_full[:train.idx_end]
    slope_3d_val = slope_3d_full[val.idx_start:val.idx_end]
    slope_3d_test = slope_3d_full[test.idx_start:test.idx_end]
    print(f"  slope 3D : mean={np.nanmean(slope_3d_full):+.5f}  std={np.nanstd(slope_3d_full):.5f}")

    # ---- 4. Metrics --------------------------------------------------------
    print("\n[4/5] Metrics × 3 refs × 2 splits...")
    refs = {
        "GT_3D": {"val": gt3_slope[val.idx_start:val.idx_end],
                  "test": gt3_slope[test.idx_start:test.idx_end]},
        "GT_4D": {"val": gt4_slope[val.idx_start:val.idx_end],
                  "test": gt4_slope[test.idx_start:test.idx_end]},
        f"MA{args.ma_window}": {"val": ma_slope[val.idx_start:val.idx_end],
                                "test": ma_slope[test.idx_start:test.idx_end]},
    }
    slopes_2d = {"val": slope_2d_val, "test": slope_2d_test}
    slopes_3d = {"val": slope_3d_val, "test": slope_3d_test}

    comparisons: List[RefComparison] = []
    for split in ("val", "test"):
        for ref_name, ref_slopes in refs.items():
            c = _compute_ref_metrics(
                slopes_2d[split], slopes_3d[split], ref_slopes[split],
                ref_name=ref_name, split=split,
            )
            comparisons.append(c)

    # Print table
    print("\n  Tableau principal :")
    bar = "-" * 128
    print(bar)
    hdr = (f"  {'split':<5s} {'ref':<8s} {'MSE 2D':>9s} {'MSE 3D':>9s} {'ΔMSE':>10s} {'Δ% rel':>9s} "
           f"{'Pear 2D':>8s} {'Pear 3D':>8s} {'DM 2D':>7s} {'DM 3D':>7s} {'DM stat':>8s} {'DM p':>10s} {'winner':<10s}")
    print(hdr)
    print(bar)
    for c in comparisons:
        rel = c.delta_mse / max(abs(c.mse_2d), 1e-30) * 100
        print(
            f"  {c.split:<5s} {c.ref_name:<8s} {c.mse_2d:>9.5f} {c.mse_3d:>9.5f} {c.delta_mse:>+10.5f} {rel:>+8.2f}% "
            f"{c.pearson_2d:>8.4f} {c.pearson_3d:>8.4f} {c.dirmatch_2d:>7.4f} {c.dirmatch_3d:>7.4f} "
            f"{c.dm_stat:>+8.2f} {c.dm_p:>10.2e} {c.winner:<10s}"
        )
    print(bar)

    # ---- 5. Decision + diagnostic -----------------------------------------
    print(f"\n[5/5] Décision (split = {args.decision_split})...")
    decision = classify(comparisons, split=args.decision_split)
    print(f"  CASE : {decision['case']}   winner : {decision['winner_overall']}")
    print(f"  {decision['reasoning']}")

    print("\n  Diagnostic 3D fwd innovations (train) :")
    diag_3d = run_diagnostic(v_3d_train, S_3d_train, split_name="train_3d_mle_fixed")
    print(f"    std(z)={diag_3d.std:.4f}   max|ACF(1..10)|={diag_3d.acf_max_abs_1_10:.4f}")
    print(f"    ACF(1..10) = [" + ", ".join(f"{v:+.4f}" for v in diag_3d.acf_1_to_10) + "]")
    print(f"    LB p={diag_3d.ljung_box_h10['p_value']:.3e}")

    # vs baseline 2D previously : max|ACF|=0.2195
    print(f"    [rappel baseline 2D MLE : max|ACF|=0.2195]")
    delta_acf = diag_3d.acf_max_abs_1_10 - 0.2195
    print(f"    Δ max|ACF| (3D − 2D) = {delta_acf:+.4f}  "
          f"({'3D plus blanc' if delta_acf < 0 else '3D moins blanc'})")

    plots_dir = artifacts_dir / "step1_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_files = make_plots(v_3d_train, S_3d_train, plots_dir, prefix="baseline_3d_mle_fixed")

    # ---- Save --------------------------------------------------------------
    np.save(artifacts_dir / "baseline_3d_mle_slope_train.npy", slope_3d_train)
    np.save(artifacts_dir / "baseline_3d_mle_slope_val.npy", slope_3d_val)
    np.save(artifacts_dir / "baseline_3d_mle_slope_test.npy", slope_3d_test)
    np.save(artifacts_dir / "baseline_3d_mle_innovations_train.npy", v_3d_train)
    np.save(artifacts_dir / "baseline_3d_mle_innov_S_train.npy", S_3d_train)

    out_json = {
        "splits": split_meta,
        "mle_params_2d": {"sigma2_proc": sigma2_2d, "R": r_2d},
        "mle_params_3d": {"sigma2_accel": sigma2_3d, "R": r_3d},
        "comparisons": [asdict(c) for c in comparisons],
        "decision": decision,
        "diagnostic_3d_train": diag_3d.to_dict(),
        "ma_window": args.ma_window,
        "decision_split": args.decision_split,
    }
    json_path = artifacts_dir / "etape2_multi_ref_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, default=str, ensure_ascii=False)

    md_path = artifacts_dir / "etape2_multi_ref_report.md"
    _write_md(md_path, decision, comparisons, diag_3d, sigma2_2d, r_2d, sigma2_3d, r_3d, split_meta, args)

    print(f"\nArtefacts sauvegardés :")
    print(f"  - baseline_3d_mle_slope_{{train,val,test}}.npy")
    print(f"  - baseline_3d_mle_innovations_train.npy")
    print(f"  - {json_path.name}")
    print(f"  - {md_path.name}")

    # ---- Final verdict / next action --------------------------------------
    print("\n" + "=" * 78)
    print(f"VERDICT FINAL : {decision['case']}   winner : {decision['winner_overall']}")
    print("=" * 78)
    print(decision["reasoning"])
    print()

    case = decision["case"]
    winner = decision["winner_overall"]
    if case in ("CASE_1_STRONG", "CASE_2_MODERATE") and winner == "3D":
        print("→ GO Étape 3 : Myers-Tapley adaptatif sur 3D WNA, par-dessus calibration MLE globale.")
    elif case == "CASE_2_WIDE" and winner == "3D":
        print("→ GO Étape 3 prudent : gain 3D existe mais variable, documenter.")
    elif case == "CASE_3_MA51" and winner == "3D":
        print("→ GO Étape 3 : MA51 = artefact, 3D reste gagnant.")
    elif case == "CASE_3_GT":
        print("→ DISCUSSION : un GT diverge, investiguer avant Étape 3.")
    elif case == "CASE_4":
        print("→ STOP axe 'dimension d'état'. Pivoter : R adaptatif / IMM / autre.")
    elif winner == "2D":
        print("→ 2D gagne : l'ajout d'accélération n'apporte rien. Pivot méthodologique.")
    print("=" * 78)

    return 0


def _write_md(
    path: Path,
    decision: Dict,
    comparisons: List[RefComparison],
    diag_3d,
    s2_2d, r_2d, s2_3d, r_3d,
    split_meta, args,
) -> None:
    def _f(x, d=5):
        return "NaN" if not np.isfinite(x) else f"{x:.{d}g}"

    L = []
    L.append("# Étape 2 — 2D MLE fixed vs 3D MLE fixed (multi-références)")
    L.append("")
    L.append("## Configuration")
    L.append(f"- Dataset : `{split_meta['csv_path']}` ({split_meta['start_date']} → {split_meta['end_date'] or 'fin'})")
    L.append(f"- Splits : train={split_meta['n_train']:,}  val={split_meta['n_val']:,}  test={split_meta['n_test']:,}")
    L.append("")
    L.append("## Baselines (MLE fixed, no adaptation)")
    L.append("")
    L.append(f"| Modèle | Params |")
    L.append(f"|--------|--------|")
    L.append(f"| **2D CV MLE**  | σ² = {s2_2d:.4g}, R = {r_2d:.4f} |")
    L.append(f"| **3D WNA MLE** | σ²_accel = {s2_3d:.4g}, R = {r_3d:.4f} |")
    L.append("")
    L.append("## Références")
    L.append("")
    L.append("| Référence | Type | Caveat |")
    L.append("|-----------|------|--------|")
    L.append("| GT 3D | Kalman RTS 3D (MLE) | Circularité partielle avec baseline 3D |")
    L.append("| GT 4D | Kalman RTS 4D const-jerk (MLE) | BIC-rejeté, conservé pour robustness |")
    L.append(f"| MA{args.ma_window}  | MA centrée sur Δ(RSI) | Biaisée vers smoothness |")
    L.append("")
    L.append("## Tableau principal")
    L.append("")
    L.append("| split | ref | MSE 2D | MSE 3D | ΔMSE | Δ% rel | Pear 2D | Pear 3D | DirM 2D | DirM 3D | DM stat | DM p | winner |")
    L.append("|-------|-----|--------|--------|------|--------|---------|---------|---------|---------|---------|------|--------|")
    for c in comparisons:
        rel = c.delta_mse / max(abs(c.mse_2d), 1e-30) * 100
        L.append(
            f"| {c.split} | {c.ref_name} | {_f(c.mse_2d)} | {_f(c.mse_3d)} | {c.delta_mse:+.5f} | {rel:+.2f}% | "
            f"{c.pearson_2d:.4f} | {c.pearson_3d:.4f} | {c.dirmatch_2d:.4f} | {c.dirmatch_3d:.4f} | "
            f"{c.dm_stat:+.2f} | {_f(c.dm_p, 3)} | {c.winner} |"
        )
    L.append("")
    L.append("Notes : ΔMSE = MSE(3D) − MSE(2D) ; négatif = 3D mieux. DM stat négatif = 3D statistiquement meilleur.")
    L.append("")
    L.append("## Diagnostic 3D forward innovations (train)")
    L.append("")
    L.append(f"- std(z) = {diag_3d.std:.4f}")
    L.append(f"- max|ACF(1..10)| = **{diag_3d.acf_max_abs_1_10:.4f}**  (baseline 2D : 0.2195)")
    L.append(f"- ACF(1..10) : [{', '.join(f'{v:+.4f}' for v in diag_3d.acf_1_to_10)}]")
    L.append(f"- Ljung-Box p = {diag_3d.ljung_box_h10['p_value']:.3e}")
    L.append("")
    L.append("## Règles de décision utilisateur")
    L.append("")
    L.append("| Case | Critère | Action |")
    L.append("|------|---------|--------|")
    L.append("| CASE_1_STRONG   | 3/3 cohérent, ratio mag ≤ 2 | Conclusion forte |")
    L.append("| CASE_2_MODERATE | 3/3 cohérent, ratio 2-3 | Conclusion modérée |")
    L.append("| CASE_2_WIDE     | 3/3 cohérent, ratio > 3 | Gain existe mais variable |")
    L.append("| CASE_3_MA51     | 2/3 + MA diverge | Artefact smoothness |")
    L.append("| CASE_3_GT       | 2/3 + GT diverge | RED FLAG |")
    L.append("| CASE_4          | pas de cohérence | STOP axe dim. |")
    L.append("")
    L.append("## Verdict")
    L.append("")
    L.append(f"**{decision['case']}** — Winner : `{decision['winner_overall']}`  (décidé sur {decision['split_used']})")
    L.append("")
    L.append(decision["reasoning"])
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
