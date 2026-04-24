"""
FLKS vs CNN-LSTM Clock-Injected — comparaison directe sur pipeline identique.

Reproduit EXACTEMENT le pipeline `src/prepare_data_30min.py` pour les labels
et l'alignement 30min→5min. Ne diffère que par l'étape "modèle" :
  - CNN-LSTM : prédit le label depuis features 5min+30min+Step Index (≈83% RSI)
  - FLKS     : applique un Fixed-Lag Smoother sur le RSI 30min brut,
               puis sign(first difference) comme prédiction

Ne touche RIEN dans src/. Réutilise :
  - src.indicators.calculate_rsi (Wilder)
  - src.constants.KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR, RSI_PERIOD
  - pykalman (pour reproduire les labels du pipeline 30min)
  - flks.fixed_lag_smoother (pour le modèle FLKS)

Méthodologie :
  1. Charger BTC 5min complet
  2. Calculer RSI 5min
  3. Resample 5min → 30min (closed='left', label='left' comme le pipeline)
  4. Calculer RSI 30min sur les OHLC 30min
  5. Filtrer RSI 30min via Kalman global (σ²=0.01, R=0.1 — pipeline historique)
  6. Labels 30min : sign(filtered[k-1] - filtered[k-2])
  7. Aligner labels 30min → 5min (forward-fill)
  8. Calculer Step Index (1-6) pour chaque 5min
  9. FLKS sur RSI 30min brut, grid lag ∈ {0, 1, 2, 3, 5, 8, ∞} (unités 30min)
  10. Prédictions 30min : sign(FLKS_level[k-1] - FLKS_level[k-2])
  11. Forward-fill les prédictions aux 5min
  12. Comparer accuracy FLKS vs label, global ET par Step Index
  13. Afficher tableau + comparaison avec historique Clock-Injected RSI (83.0%)

Paramètres FLKS testés :
  - Projet actuel : σ²=0.01, R=0.1 (KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR)
  - MLE 2D      : σ²=1.155, R=3.27 (Étape B.4, fitté sur 5min)
    Note : on les applique tels quels ; un refit MLE sur 30min serait plus rigoureux.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_PROJECT_ROOT = _HERE.parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from indicators import calculate_rsi  # noqa: E402
from constants import KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR, RSI_PERIOD  # noqa: E402
from flks import fixed_lag_smoother, full_rts_smoother  # noqa: E402


# 2D CV model matrices for FLKS
F_2 = np.array([[1.0, 1.0], [0.0, 1.0]])
H_2 = np.array([[1.0, 0.0]])
G_2 = np.array([[1.0], [1.0]])


# Lag grid (units = 30min bars)
LAG_GRID = [0, 1, 2, 3, 5, 8, np.inf]


# ---------------------------------------------------------------------------
# Pipeline reproduction
# ---------------------------------------------------------------------------

def load_btc_5min_full(csv_path: Path, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Load BTC 5min OHLCV (matches data_loader.load_btc_5min structure)."""
    df = pd.read_csv(csv_path)
    date_col = None
    for c in ["date", "datetime", "time", "timestamp", "Date", "Datetime"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"No date column in {csv_path}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df.columns = df.columns.str.lower()
    if start_date:
        df = df.loc[start_date:]
    if end_date:
        df = df.loc[:end_date]
    return df.dropna(subset=["close"])


def resample_5min_to_30min(df_5min: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduit src/prepare_data_30min.py:resample_5min_to_30min.
    closed='left', label='left' : 10:00-10:29 → bougie indexée 10:00.
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df_5min.columns:
        agg["volume"] = "sum"
    return df_5min.resample("30min", closed="left", label="left").agg(agg).dropna()


def kalman_smooth_1d(
    data: np.ndarray,
    process_var: float,
    measure_var: float,
) -> np.ndarray:
    """
    2D CV Kalman with RTS smoothing, Q = σ²·I diagonal (matches historique).
    Returns smoothed position (level) array.
    """
    from pykalman import KalmanFilter
    kf = KalmanFilter(
        transition_matrices=np.array([[1.0, 1.0], [0.0, 1.0]]),
        observation_matrices=np.array([[1.0, 0.0]]),
        transition_covariance=np.eye(2) * process_var,
        observation_covariance=[[measure_var]],
        initial_state_mean=np.array([data[0], 0.0]),
        initial_state_covariance=np.eye(2),
    )
    smoothed, _ = kf.smooth(data.reshape(-1, 1))
    return np.asarray(smoothed)[:, 0]


def compute_30min_labels(filtered_30min: np.ndarray) -> np.ndarray:
    """
    Label[k] = 1 si filtered[k-1] > filtered[k-2] else 0 (matches
    src.indicators.generate_labels). Les deux premiers échantillons = -1 (invalid).
    """
    n = len(filtered_30min)
    labels = np.full(n, -1, dtype=np.int8)
    for k in range(2, n):
        labels[k] = 1 if filtered_30min[k - 1] > filtered_30min[k - 2] else 0
    return labels


def forward_fill_30min_to_5min(
    values_30min: np.ndarray,
    index_30min: pd.DatetimeIndex,
    index_5min: pd.DatetimeIndex,
    fill_invalid: int = -1,
) -> np.ndarray:
    """
    Forward-fill valeurs 30min sur timestamps 5min.
    À timestamp 5min t, on prend la valeur du bucket 30min contenant t.
    """
    # Construire une série indexée 30min
    s_30 = pd.Series(values_30min, index=index_30min)
    # Reindex + forward-fill sur l'index 5min
    aligned = s_30.reindex(index_5min, method="ffill")
    # Fill remaining NaN (avant premier bucket) par fill_invalid
    aligned = aligned.fillna(fill_invalid)
    return aligned.to_numpy()


def compute_step_index(index_5min: pd.DatetimeIndex) -> np.ndarray:
    """Step Index 1-6 pour chaque 5min (1 = minute 00, 6 = minute 25)."""
    minutes = index_5min.minute
    step_index = (minutes % 30) // 5 + 1  # 1, 2, 3, 4, 5, 6
    return step_index.values


# ---------------------------------------------------------------------------
# FLKS-based prediction
# ---------------------------------------------------------------------------

def flks_predictions_30min(
    rsi_30min: np.ndarray,
    sigma2: float,
    r_scalar: float,
    lag: float,
) -> np.ndarray:
    """
    Applique FLKS sur la série RSI 30min avec le lag donné (unités 30min).
    Retourne un array de prédictions binaires indexé comme rsi_30min :
        pred[k] = 1 si FLKS_level[k-1] > FLKS_level[k-2], sinon 0.
        pred[0] = pred[1] = -1 (invalid)

    Le label du pipeline a la même formule → comparaison directe.
    """
    if np.isinf(lag):
        flks = full_rts_smoother(rsi_30min, F_2, H_2, G_2, sigma2, r_scalar)
    else:
        flks = fixed_lag_smoother(rsi_30min, F_2, H_2, G_2, sigma2, r_scalar, lag=int(lag))
    level = flks.x_smoothed[:, 0]
    n = len(level)
    pred = np.full(n, -1, dtype=np.int8)
    for k in range(2, n):
        pred[k] = 1 if level[k - 1] > level[k - 2] else 0
    return pred


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class LagEval:
    lag_label: str
    sigma2: float
    r_scalar: float
    accuracy_global: float
    accuracy_by_step: Dict[int, float]      # {1: 0.xx, 2: ..., 6: ...}
    n_valid: int
    n_per_step: Dict[int, int]


def evaluate_lag(
    labels_5min: np.ndarray,
    step_index: np.ndarray,
    preds_5min: np.ndarray,
    lag_label: str,
    sigma2: float,
    r_scalar: float,
) -> LagEval:
    valid = (labels_5min != -1) & (preds_5min != -1)
    acc_global = float(np.mean(labels_5min[valid] == preds_5min[valid]))

    acc_by_step = {}
    n_per_step = {}
    for s in range(1, 7):
        mask = valid & (step_index == s)
        if mask.sum() > 0:
            acc_by_step[s] = float(np.mean(labels_5min[mask] == preds_5min[mask]))
            n_per_step[s] = int(mask.sum())
        else:
            acc_by_step[s] = float("nan")
            n_per_step[s] = 0

    return LagEval(
        lag_label=lag_label,
        sigma2=sigma2,
        r_scalar=r_scalar,
        accuracy_global=acc_global,
        accuracy_by_step=acc_by_step,
        n_valid=int(valid.sum()),
        n_per_step=n_per_step,
    )


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_comparison(
    csv_path: Path,
    start_date: str,
    end_date: str,
    sigma2_calibrations: List[Tuple[str, float, float]],
    test_split_frac: float = 0.15,
    artifacts_dir: Path = None,
) -> Dict:
    """
    Orchestrateur : charge data, reproduit pipeline, applique FLKS sous
    plusieurs calibrations, évalue.

    test_split_frac : frac des dernières données à utiliser comme 'test'
                      (les accuracies historiques Clock-Injected sont sur TEST).
    """
    bar = "=" * 78
    print(bar)
    print("FLKS vs Clock-Injected CNN-LSTM — pipeline 30min buckets + 5min sub-steps")
    print(bar)

    # ---- 1. Load data --------------------------------------------------
    print(f"\n[1/6] Chargement BTC 5min...")
    df_5min = load_btc_5min_full(csv_path, start_date, end_date)
    print(f"  Période : {df_5min.index[0]} → {df_5min.index[-1]}")
    print(f"  N 5min : {len(df_5min):,}")

    # ---- 2. 5min RSI (pour référence, pas utilisé par FLKS) ------------
    print(f"\n[2/6] Calcul RSI 5min (référence)...")
    rsi_5min = calculate_rsi(df_5min["close"].to_numpy(), period=RSI_PERIOD)

    # ---- 3. Resample 30min ---------------------------------------------
    print(f"\n[3/6] Resample 5min → 30min (closed='left', label='left')...")
    df_30min = resample_5min_to_30min(df_5min)
    print(f"  N 30min : {len(df_30min):,}")

    # ---- 4. RSI 30min --------------------------------------------------
    print(f"\n[4/6] RSI 30min + filtrage Kalman historique (σ²=0.01, R=0.1)...")
    rsi_30min_raw = calculate_rsi(df_30min["close"].to_numpy(), period=RSI_PERIOD)
    # Drop warmup NaN
    first_valid = int(np.argmax(np.isfinite(rsi_30min_raw)))
    rsi_30min = rsi_30min_raw[first_valid:]
    df_30min_trimmed = df_30min.iloc[first_valid:]
    # Filter with pipeline constants for labels
    filtered_30min = kalman_smooth_1d(rsi_30min, KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR)
    labels_30min = compute_30min_labels(filtered_30min)
    print(f"  RSI 30min valides : {len(rsi_30min):,}")
    print(f"  Labels 30min valides : {(labels_30min != -1).sum():,}")
    pct_up = float(np.mean(labels_30min[labels_30min != -1] == 1))
    print(f"  Distribution labels : {pct_up*100:.2f}% UP")

    # ---- 5. Align labels 30min → 5min ----------------------------------
    print(f"\n[5/6] Alignement labels 30min → 5min (forward-fill)...")
    labels_5min = forward_fill_30min_to_5min(
        labels_30min.astype(float),
        df_30min_trimmed.index,
        df_5min.index,
        fill_invalid=-1,
    ).astype(np.int8)
    step_index_5min = compute_step_index(df_5min.index)
    print(f"  5min labels valides : {(labels_5min != -1).sum():,} / {len(labels_5min):,}")

    # Test split : les dernières test_split_frac % (chronologique)
    n = len(df_5min)
    n_test = int(n * test_split_frac)
    test_mask = np.zeros(n, dtype=bool)
    test_mask[-n_test:] = True
    print(f"  Test split (derniers {test_split_frac*100:.0f}%) : {n_test:,} 5min samples")

    # ---- 6. FLKS evaluations -------------------------------------------
    print(f"\n[6/6] FLKS sweep sur {len(LAG_GRID)} lags × {len(sigma2_calibrations)} calibrations...")
    all_results: List[LagEval] = []
    for cal_name, sigma2, r_scalar in sigma2_calibrations:
        print(f"\n  === Calibration : {cal_name}  (σ²={sigma2:.4g}, R={r_scalar:.4g}) ===")
        for lag in LAG_GRID:
            lag_label = "inf" if np.isinf(lag) else str(int(lag))
            # Run FLKS on 30min RAW RSI
            pred_30min = flks_predictions_30min(rsi_30min, sigma2, r_scalar, lag)
            # Forward-fill to 5min
            pred_5min = forward_fill_30min_to_5min(
                pred_30min.astype(float),
                df_30min_trimmed.index,
                df_5min.index,
                fill_invalid=-1,
            ).astype(np.int8)
            # Evaluate on TEST split only (match historical 83.0% on test)
            pred_test = pred_5min.copy()
            pred_test[~test_mask] = -1  # mask out non-test samples
            labels_test = labels_5min.copy()
            labels_test[~test_mask] = -1

            res = evaluate_lag(
                labels_test, step_index_5min, pred_test,
                lag_label=f"{cal_name}_lag{lag_label}",
                sigma2=sigma2, r_scalar=r_scalar,
            )
            all_results.append(res)
            print(
                f"    lag={lag_label:<3s} : acc_global={res.accuracy_global*100:.2f}%  "
                f"n_valid_test={res.n_valid:,}"
            )

    # ---- Output -------------------------------------------------------
    print("\n" + "=" * 110)
    print("TABLEAU COMPARATIF — Accuracy par calibration × lag × step_index")
    print("=" * 110)
    hdr = f"{'calibration':<20s} {'lag':<5s} {'global':>8s}" + "".join(f"  step{s}" for s in range(1, 7))
    print(hdr)
    print("-" * len(hdr))
    for r in all_results:
        cal, lag_lbl = r.lag_label.rsplit("_lag", 1)
        row = f"{cal:<20s} {lag_lbl:<5s} {r.accuracy_global*100:>7.2f}%"
        for s in range(1, 7):
            row += f"  {r.accuracy_by_step[s]*100:>5.2f}%"
        print(row)
    print("-" * len(hdr))

    # ---- Historical reference ----
    print("\nRéférences historiques (Clock-Injected 7 features, test set) :")
    print("  RSI  : 83.0%  (cible de comparaison directe)")
    print("  CCI  : 85.6%")
    print("  MACD : 86.8%")
    print("  Moyenne 3 indicateurs : 85.1%")

    # ---- Save artifacts ----
    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "config": {
                "csv": str(csv_path),
                "start_date": start_date,
                "end_date": end_date,
                "test_split_frac": test_split_frac,
                "rsi_period": RSI_PERIOD,
                "lag_grid": [("inf" if np.isinf(L) else int(L)) for L in LAG_GRID],
                "sigma2_calibrations": [{"name": n, "sigma2": s, "R": r}
                                        for n, s, r in sigma2_calibrations],
                "label_formula": "filtered_30min[k-1] > filtered_30min[k-2]  (generate_labels)",
                "kalman_for_labels": "σ²=0.01, R=0.1 (KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR)",
                "historical_reference_RSI": 0.830,
            },
            "results": [
                {
                    "calibration": r.lag_label.rsplit("_lag", 1)[0],
                    "lag": r.lag_label.rsplit("_lag", 1)[1],
                    "sigma2": r.sigma2,
                    "r_scalar": r.r_scalar,
                    "accuracy_global": r.accuracy_global,
                    "accuracy_by_step": r.accuracy_by_step,
                    "n_valid": r.n_valid,
                    "n_per_step": r.n_per_step,
                }
                for r in all_results
            ],
            "counts": {
                "n_5min_total": int(n),
                "n_5min_test": int(n_test),
                "n_30min_valid": int((labels_30min != -1).sum()),
                "n_5min_labels_valid": int((labels_5min != -1).sum()),
                "pct_up": float(pct_up),
            },
        }
        json_path = artifacts_dir / "flks_vs_clockinjected.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str, ensure_ascii=False)
        print(f"\nRésultats sauvegardés : {json_path.name}")

    # ---- Verdict interprétation ----
    print("\n" + "=" * 78)
    print("INTERPRÉTATION")
    print("=" * 78)

    # Best FLKS per calibration
    cal_names = sorted(set(r.lag_label.rsplit("_lag", 1)[0] for r in all_results))
    for cal in cal_names:
        cal_results = [r for r in all_results if r.lag_label.startswith(cal + "_lag")]
        best = max(cal_results, key=lambda r: r.accuracy_global)
        lag_lbl = best.lag_label.rsplit("_lag", 1)[1]
        delta_vs_ci = (best.accuracy_global - 0.830) * 100
        print(f"  {cal:<20s} best : lag={lag_lbl}, acc={best.accuracy_global*100:.2f}% "
              f"(Δ vs Clock-Injected RSI 83.0% : {delta_vs_ci:+.2f} pp)")

    print("=" * 78)

    return {"results": all_results}


def main() -> int:
    parser = argparse.ArgumentParser(description="FLKS vs CNN-LSTM Clock-Injected")
    parser.add_argument("--csv", default="data_trad/BTCUSD_all_5m.csv")
    parser.add_argument("--start-date", default=None,
                        help="Default = full CSV (matches historical CNN-LSTM training)")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--test-split-frac", type=float, default=0.15,
                        help="Fraction finale utilisée comme test (default 0.15 = Clock-Injected split)")
    parser.add_argument("--artifacts-dir", default=str(_HERE / "artifacts"))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = _PROJECT_ROOT / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable : {csv_path}")

    sigma2_calibrations = [
        ("projet_actuel", KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR),     # σ²=0.01, R=0.1
        ("MLE_2D_5min",   1.155,              3.27),                   # Étape B.4
    ]

    run_comparison(
        csv_path=csv_path,
        start_date=args.start_date,
        end_date=args.end_date,
        sigma2_calibrations=sigma2_calibrations,
        test_split_frac=args.test_split_frac,
        artifacts_dir=Path(args.artifacts_dir).resolve(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
