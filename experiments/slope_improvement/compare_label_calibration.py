"""
Test rapide — Impact de la recalibration Kalman sur les labels existants.

Question :
    Les labels Phase 2.15 (`filtered[t] > filtered[t-1]`) générés par le
    pipeline actuel utilisent σ² = KALMAN_PROCESS_VAR = 0.01 et
    R = KALMAN_MEASURE_VAR = 0.1. Notre MLE 2D a trouvé σ² = 1.155 et
    R = 3.27 — ratios ×115 et ×33 vs les valeurs projet.

    Les labels CHANGENT-ILS substantiellement avec cette recalibration ?
    Si oui, tous les CNN-LSTM entraînés dessus opèrent sur des labels
    potentiellement bruités.

Règle de décision (spécifiée par l'utilisateur) :
    - Désaccord < 5%   → ancien pipeline robuste au mis-calibrage
    - Désaccord 5-10%  → marginal, à noter
    - Désaccord 10-20% → significatif, réentraînement recommandé
    - Désaccord > 20%  → réentraînement nécessaire

Implémentation :
    Reproduit exactement le pipeline `prepare_data_direction_only.py`
    (Kalman 2D smoother non-causal via pykalman, Q = σ²·I diagonale),
    avec les deux jeux de paramètres. Comparaison sur la période
    complète du CSV BTC 5min pour matcher les datasets d'entraînement
    historiques.

Artefacts :
    artifacts/compare_label_calibration.json  — statistiques complètes
    artifacts/compare_label_calibration.md    — rapport synthétique
    artifacts/labels_old_calib.npy            — labels avec params projet
    artifacts/labels_new_calib.npy            — labels avec params MLE
    artifacts/position_old_calib.npy          — position smoothed OLD
    artifacts/position_new_calib.npy          — position smoothed NEW

Réutilise :
    - src.indicators.calculate_rsi
    - src.constants.KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR, RSI_PERIOD
    - pykalman (même lib que prepare_data_direction_only.py)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from indicators import calculate_rsi  # noqa: E402
from constants import KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR, RSI_PERIOD  # noqa: E402


# ---------------------------------------------------------------------------
# Kalman dual (matches prepare_data_direction_only.py exactly)
# ---------------------------------------------------------------------------

def kalman_dual_smooth(
    data: np.ndarray,
    process_var: float,
    measure_var: float,
) -> np.ndarray:
    """
    2D Kalman with RTS smoothing, diagonal Q. Matches exactly the
    `kalman_filter_dual` function in `src/prepare_data_direction_only.py`
    lines 215-240.

    State: x = [position, velocity]
    F = [[1, 1], [0, 1]]
    H = [[1, 0]]
    Q = np.eye(2) * process_var   (diagonal, NOT rank-1)
    R = measure_var

    Returns shape (n, 2) = [position_smoothed, velocity_smoothed].
    """
    from pykalman import KalmanFilter

    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    observation = np.array([[1.0, 0.0]])

    kf = KalmanFilter(
        transition_matrices=transition,
        observation_matrices=observation,
        transition_covariance=np.eye(2) * process_var,
        observation_covariance=[[measure_var]],
        initial_state_mean=np.array([data[0], 0.0]),
        initial_state_covariance=np.eye(2),
    )
    smoothed, _ = kf.smooth(data.reshape(-1, 1))
    return np.asarray(smoothed)


# ---------------------------------------------------------------------------
# Label computation (Phase 2.15 formula: filtered[t] > filtered[t-1])
# ---------------------------------------------------------------------------

def compute_direction_labels(position: np.ndarray) -> np.ndarray:
    """
    Phase 2.15 direction label:
        label[t] = 1 if position[t] > position[t-1] else 0
    label[0] is NaN (no predecessor).
    """
    n = len(position)
    out = np.full(n, np.nan)
    out[1:] = (position[1:] > position[:-1]).astype(float)
    return out


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class CalibrationComparison:
    n_samples: int
    n_labels_valid: int
    # OLD calibration
    sigma2_old: float
    R_old: float
    pct_up_old: float
    # NEW calibration (MLE)
    sigma2_new: float
    R_new: float
    pct_up_new: float
    # Agreement
    agreement_rate: float          # fraction where labels identical
    disagreement_rate: float       # 1 - agreement
    # Position-level metrics
    pos_rmse: float                # sqrt(mean((pos_old - pos_new)^2))
    pos_mae: float
    pos_correlation: float
    # Classification of impact
    verdict: str
    verdict_detail: str


def _classify_impact(disagreement_rate: float) -> Tuple[str, str]:
    """User-specified thresholds for action."""
    pct = disagreement_rate * 100
    if pct < 5:
        return "ROBUST", (
            f"Désaccord {pct:.2f}% < 5% → pipeline historique robuste au "
            f"mis-calibrage. Les labels Phase 2.15 restent valides, pas de "
            f"réentraînement nécessaire."
        )
    elif pct < 10:
        return "MARGINAL", (
            f"Désaccord {pct:.2f}% ∈ [5%, 10%] → marginal. À noter, mais "
            f"pas obligatoire de réentraîner. Les accuracies 87-92% "
            f"historiques sont probablement légèrement biaisées."
        )
    elif pct < 20:
        return "SIGNIFICATIVE", (
            f"Désaccord {pct:.2f}% ∈ [10%, 20%] → impact significatif. "
            f"Réentraînement recommandé sur labels recalibrés. Une part "
            f"notable de la performance historique reflète la calibration "
            f"spécifique, pas la dynamique intrinsèque."
        )
    else:
        return "CRITIQUE", (
            f"Désaccord {pct:.2f}% > 20% → impact critique. Les CNN-LSTM "
            f"existants opèrent sur des labels fortement dépendants des "
            f"params Kalman choisis. Réentraînement nécessaire."
        )


def compare_calibrations(
    rsi: np.ndarray,
    sigma2_old: float = KALMAN_PROCESS_VAR,
    R_old: float = KALMAN_MEASURE_VAR,
    sigma2_new: float = 1.155,
    R_new: float = 3.27,
) -> Tuple[CalibrationComparison, Dict[str, np.ndarray]]:
    """Run both calibrations, return comparison + smoothed positions/labels."""
    # Drop initial NaNs from RSI warmup
    finite = np.isfinite(rsi)
    if not finite.any():
        raise RuntimeError("RSI entièrement NaN")
    first = int(np.argmax(finite))
    rsi_clean = rsi[first:]

    print(f"  Running Kalman OLD (σ²={sigma2_old}, R={R_old}) on {len(rsi_clean):,} samples...")
    state_old = kalman_dual_smooth(rsi_clean, sigma2_old, R_old)
    pos_old = state_old[:, 0]

    print(f"  Running Kalman NEW (σ²={sigma2_new}, R={R_new}) on {len(rsi_clean):,} samples...")
    state_new = kalman_dual_smooth(rsi_clean, sigma2_new, R_new)
    pos_new = state_new[:, 0]

    # Labels (Phase 2.15 formula)
    label_old = compute_direction_labels(pos_old)
    label_new = compute_direction_labels(pos_new)

    # Drop first sample (NaN labels)
    mask_both = np.isfinite(label_old) & np.isfinite(label_new)
    lo, ln = label_old[mask_both], label_new[mask_both]
    agreement_rate = float(np.mean(lo == ln))
    disagreement_rate = 1.0 - agreement_rate

    pct_up_old = float(np.mean(lo))
    pct_up_new = float(np.mean(ln))

    # Position-level comparison
    pos_diff = pos_old - pos_new
    pos_rmse = float(np.sqrt(np.mean(pos_diff ** 2)))
    pos_mae = float(np.mean(np.abs(pos_diff)))
    pos_correlation = float(np.corrcoef(pos_old, pos_new)[0, 1])

    verdict, detail = _classify_impact(disagreement_rate)

    result = CalibrationComparison(
        n_samples=len(rsi_clean),
        n_labels_valid=int(mask_both.sum()),
        sigma2_old=float(sigma2_old), R_old=float(R_old), pct_up_old=pct_up_old,
        sigma2_new=float(sigma2_new), R_new=float(R_new), pct_up_new=pct_up_new,
        agreement_rate=agreement_rate,
        disagreement_rate=disagreement_rate,
        pos_rmse=pos_rmse,
        pos_mae=pos_mae,
        pos_correlation=pos_correlation,
        verdict=verdict,
        verdict_detail=detail,
    )
    artifacts = {
        "position_old": pos_old,
        "position_new": pos_new,
        "labels_old": label_old,
        "labels_new": label_new,
        "rsi_clean": rsi_clean,
        "first_valid_idx": first,
    }
    return result, artifacts


# ---------------------------------------------------------------------------
# Per-period analysis (detect regime-specific impact)
# ---------------------------------------------------------------------------

def per_period_agreement(
    artifacts: Dict[str, np.ndarray],
    timestamps: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Agreement rate broken down by year. Helps spot regime-specific effects.
    """
    label_old = artifacts["labels_old"]
    label_new = artifacts["labels_new"]
    first = artifacts["first_valid_idx"]
    ts = timestamps[first:]
    ts_pd = pd.to_datetime(ts)
    df = pd.DataFrame({
        "year": ts_pd.year,
        "label_old": label_old,
        "label_new": label_new,
    }).dropna()
    if df.empty:
        return {}

    out = {}
    for year, group in df.groupby("year"):
        agreement = float(np.mean(group["label_old"].values == group["label_new"].values))
        out[str(int(year))] = {
            "n": int(len(group)),
            "agreement": agreement,
            "pct_up_old": float(group["label_old"].mean()),
            "pct_up_new": float(group["label_new"].mean()),
        }
    return out


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_report(
    path: Path,
    cmp: CalibrationComparison,
    per_year: Dict[str, Dict[str, float]],
    csv_path: str,
) -> None:
    L = []
    L.append("# Impact de la recalibration Kalman sur les labels existants")
    L.append("")
    L.append("## Configuration")
    L.append(f"- Dataset : `{csv_path}` (période complète)")
    L.append(f"- Indicateur : RSI({RSI_PERIOD}) via `src.indicators.calculate_rsi`")
    L.append(f"- Kalman : 2D CV, Q = σ²·I (diagonal), smoother RTS via pykalman")
    L.append(f"  → reproduit exactement `src/prepare_data_direction_only.py:kalman_filter_dual`")
    L.append(f"- Label : Phase 2.15 (`filtered[t] > filtered[t-1]`)")
    L.append("")
    L.append("## Paramètres comparés")
    L.append("")
    L.append("| Calibration | σ² (process) | R (measure) | Source |")
    L.append("|-------------|--------------|-------------|--------|")
    L.append(f"| OLD (projet actuel) | {cmp.sigma2_old} | {cmp.R_old} | `src/constants.py` KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR |")
    L.append(f"| NEW (MLE 2D) | {cmp.sigma2_new} | {cmp.R_new} | Fit MLE sur 20k samples train 2022+ (Étape B.4) |")
    L.append(f"| Ratio | ×{cmp.sigma2_new/cmp.sigma2_old:.1f} | ×{cmp.R_new/cmp.R_old:.1f} | |")
    L.append("")
    L.append("## Résultats — Agrément des labels")
    L.append("")
    L.append("| Métrique | Valeur |")
    L.append("|----------|--------|")
    L.append(f"| Samples analysés | {cmp.n_samples:,} |")
    L.append(f"| Labels valides (post-warmup) | {cmp.n_labels_valid:,} |")
    L.append(f"| **Taux d'accord** | **{cmp.agreement_rate*100:.2f}%** |")
    L.append(f"| **Taux de désaccord** | **{cmp.disagreement_rate*100:.2f}%** |")
    L.append(f"| % UP (OLD) | {cmp.pct_up_old*100:.2f}% |")
    L.append(f"| % UP (NEW) | {cmp.pct_up_new*100:.2f}% |")
    L.append(f"| Δ % UP | {(cmp.pct_up_new - cmp.pct_up_old)*100:+.2f} pp |")
    L.append("")
    L.append("## Résultats — Écart des positions smoothed")
    L.append("")
    L.append("| Métrique | Valeur |")
    L.append("|----------|--------|")
    L.append(f"| RMSE position (OLD vs NEW) | {cmp.pos_rmse:.4f} |")
    L.append(f"| MAE position | {cmp.pos_mae:.4f} |")
    L.append(f"| Pearson correlation | {cmp.pos_correlation:.6f} |")
    L.append("")
    if per_year:
        L.append("## Désagrégation par année")
        L.append("")
        L.append("| Année | N | Agreement | % UP OLD | % UP NEW |")
        L.append("|-------|---|-----------|----------|----------|")
        for year in sorted(per_year.keys()):
            d = per_year[year]
            L.append(f"| {year} | {d['n']:,} | {d['agreement']*100:.2f}% | "
                     f"{d['pct_up_old']*100:.2f}% | {d['pct_up_new']*100:.2f}% |")
        L.append("")
    L.append("## Verdict")
    L.append("")
    L.append(f"**{cmp.verdict}**")
    L.append("")
    L.append(cmp.verdict_detail)
    L.append("")
    L.append("### Règles de décision (pré-définies par l'utilisateur)")
    L.append("")
    L.append("| Désaccord | Verdict | Action |")
    L.append("|-----------|---------|--------|")
    L.append("| < 5% | ROBUST | Pipeline OK, pas de réentraînement |")
    L.append("| 5-10% | MARGINAL | À noter, optionnel |")
    L.append("| 10-20% | SIGNIFICATIVE | Réentraînement recommandé |")
    L.append("| > 20% | CRITIQUE | Réentraînement nécessaire |")
    L.append("")
    L.append("## Note méthodologique")
    L.append("")
    L.append("Le MLE a été calibré sur un sous-échantillon 2022+ de 20k samples.")
    L.append("Les labels historiques couvrent toute la période du CSV (2017-2026).")
    L.append("Pour un test plus rigoureux, on pourrait refitter le MLE par régime")
    L.append("temporel, mais ce test rapide mesure l'ordre de grandeur de l'effet.")
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Kalman calibrations on direction labels")
    parser.add_argument("--csv", default="data_trad/BTCUSD_all_5m.csv")
    parser.add_argument("--start-date", default=None,
                        help="Optional : filter from date (default = full CSV = matches historical training)")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--sigma2-new", type=float, default=1.155,
                        help="MLE σ² from Étape B.4 (default 1.155)")
    parser.add_argument("--r-new", type=float, default=3.27,
                        help="MLE R from Étape B.4 (default 3.27)")
    parser.add_argument("--artifacts-dir", default=str(_HERE / "artifacts"))
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("TEST RAPIDE — Impact recalibration Kalman sur labels Phase 2.15")
    print("=" * 78)

    # Load CSV
    print(f"\n[1/4] Chargement CSV : {args.csv}")
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = _PROJECT_ROOT / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable : {csv_path}")

    df = pd.read_csv(csv_path)
    date_col = None
    for c in ["date", "datetime", "time", "timestamp", "Date", "Datetime"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"Aucune colonne date trouvée dans {csv_path}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df.columns = df.columns.str.lower()
    if args.start_date:
        df = df.loc[args.start_date:]
    if args.end_date:
        df = df.loc[:args.end_date]
    df = df.dropna(subset=["close"])
    print(f"  Période : {df.index[0]} → {df.index[-1]}")
    print(f"  N barres : {len(df):,}")

    # RSI
    print(f"\n[2/4] Calcul RSI({RSI_PERIOD})...")
    close = df["close"].to_numpy()
    rsi = calculate_rsi(close, period=RSI_PERIOD)
    timestamps = df.index.to_numpy()
    n_rsi_valid = np.isfinite(rsi).sum()
    print(f"  RSI valid : {n_rsi_valid:,} / {len(rsi):,}")

    # Compare
    print(f"\n[3/4] Comparaison Kalman OLD vs NEW...")
    cmp, artifacts = compare_calibrations(
        rsi,
        sigma2_old=KALMAN_PROCESS_VAR, R_old=KALMAN_MEASURE_VAR,
        sigma2_new=args.sigma2_new, R_new=args.r_new,
    )

    # Print results
    print("\n" + "-" * 78)
    print(f"  % UP labels OLD : {cmp.pct_up_old*100:.2f}%")
    print(f"  % UP labels NEW : {cmp.pct_up_new*100:.2f}%")
    print(f"  Δ distribution  : {(cmp.pct_up_new - cmp.pct_up_old)*100:+.2f} pp")
    print("-" * 78)
    print(f"  Agreement rate       : {cmp.agreement_rate*100:.2f}%")
    print(f"  Disagreement rate    : {cmp.disagreement_rate*100:.2f}%")
    print("-" * 78)
    print(f"  Position RMSE        : {cmp.pos_rmse:.4f}")
    print(f"  Position MAE         : {cmp.pos_mae:.4f}")
    print(f"  Position Pearson     : {cmp.pos_correlation:.6f}")
    print("-" * 78)

    # Per-year analysis
    print(f"\n[4/4] Analyse par année...")
    per_year = per_period_agreement(artifacts, timestamps)
    for year in sorted(per_year.keys()):
        d = per_year[year]
        print(f"  {year} : agreement={d['agreement']*100:.2f}%  "
              f"(%UP: old={d['pct_up_old']*100:.1f}%, new={d['pct_up_new']*100:.1f}%, n={d['n']:,})")

    # Save artifacts
    np.save(artifacts_dir / "labels_old_calib.npy", artifacts["labels_old"])
    np.save(artifacts_dir / "labels_new_calib.npy", artifacts["labels_new"])
    np.save(artifacts_dir / "position_old_calib.npy", artifacts["position_old"])
    np.save(artifacts_dir / "position_new_calib.npy", artifacts["position_new"])

    out_json = {
        "config": {
            "csv": str(csv_path),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "rsi_period": RSI_PERIOD,
            "kalman_model": "2D CV, Q = σ²·I diagonal, RTS smoother via pykalman",
            "label_formula": "Phase 2.15: filtered[t] > filtered[t-1]",
        },
        "comparison": asdict(cmp),
        "per_year_agreement": per_year,
    }
    with (artifacts_dir / "compare_label_calibration.json").open("w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, default=str, ensure_ascii=False)

    # Markdown
    md_path = artifacts_dir / "compare_label_calibration.md"
    write_report(md_path, cmp, per_year, str(csv_path))

    # Verdict
    print("\n" + "=" * 78)
    print(f"VERDICT : {cmp.verdict}")
    print("=" * 78)
    print(cmp.verdict_detail)
    print("=" * 78)

    print(f"\nArtefacts :")
    print(f"  - labels_old_calib.npy, labels_new_calib.npy")
    print(f"  - position_old_calib.npy, position_new_calib.npy")
    print(f"  - compare_label_calibration.json")
    print(f"  - {md_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
