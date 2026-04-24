"""
FLKS Sub-Step Convergence — comparaison calibrations A (historique) vs B (MLE).

Reproduit EXACTEMENT le test `src/signal_processing/flks_substep_convergence.py`
(Test 1 FLKS 30min pur + Test 2 FLKS avec k=1..6 sous-pas 5min live), mais
sur RSI uniquement, et avec **deux calibrations Kalman** exécutées en parallèle :

    A — Historique  : σ² = 0.01,  R = 0.1   (KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR)
    B — MLE fixed   : σ² = 1.155, R = 3.27  (fit MLE Étape B.4)

L'Oracle (pykalman.smooth global) utilise les paramètres historiques (σ²=0.01, R=0.1)
en référence commune pour les deux calibrations — choix validé par l'utilisateur.

Configuration identique au script historique :
    - 5000 bougies 30min (les plus récentes du CSV)
    - eval_start = 1000 (warmup, exclu de l'évaluation)
    - Indicateur RSI (période = 14 comme dans le script historique)
    - Métrique : % concordance de signe vs Oracle (all + at transitions)

Réutilise (import direct, sans modification) depuis src/signal_processing/ :
    - load_csv, resample_ohlcv, compute_bucket_close_mask
    - calculate_rsi (30min standard)
    - compute_rsi_live (frozen/provisional sur 5min)
    - compute_oracle (pykalman.smooth avec params historiques)
    - sign_concordance, find_oracle_transitions, sign_concordance_at_transitions

Réimplémente avec paramétrisation (params dépendent de la calibration) :
    - kf_update, kf_predict_sub
    - forward_filter_30m
    - compute_slopes_test1, compute_slopes_test2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_PROJECT_ROOT = _HERE.parents[1]
_SRC_SIG = _PROJECT_ROOT / "src" / "signal_processing"
if str(_SRC_SIG) not in sys.path:
    sys.path.insert(0, str(_SRC_SIG))

# Import from historical script (sans modifier)
from flks_substep_convergence import (  # noqa: E402
    load_csv,
    resample_ohlcv,
    compute_bucket_close_mask,
    calculate_rsi,
    compute_rsi_live,
    compute_oracle,
    sign_concordance,
    find_oracle_transitions,
    sign_concordance_at_transitions,
)


# ---------------------------------------------------------------------------
# KF primitives parameterisables (A, H, Q, R, A_SUB, Q_SUB dépendent de calib)
# ---------------------------------------------------------------------------

def _kf_update(x_p, P_p, z_obs, H, R):
    y = z_obs - H @ x_p
    S = H @ P_p @ H.T + R
    K = P_p @ H.T / S[0, 0]
    return x_p + (K @ y).ravel(), (np.eye(2) - K @ H) @ P_p


def _forward_filter_30m(indicator_30m, A, H, Q, R):
    """Forward Kalman filter 30min avec params fournis. Retourne x_filt, P_filt, x_pred, P_pred, C."""
    n = len(indicator_30m)
    first_valid_val = indicator_30m[~np.isnan(indicator_30m)][0]

    x_filt = np.zeros((n, 2))
    P_filt = np.zeros((n, 2, 2))
    x_pred = np.zeros((n, 2))
    P_pred = np.zeros((n, 2, 2))

    for t in range(n):
        if t == 0:
            x_p = np.array([first_valid_val, 0.0])
            P_p = np.eye(2)
        else:
            x_p = A @ x_filt[t - 1]
            P_p = A @ P_filt[t - 1] @ A.T + Q
        x_pred[t] = x_p
        P_pred[t] = P_p

        if np.isnan(indicator_30m[t]):
            x_filt[t] = x_p
            P_filt[t] = P_p
        else:
            x_filt[t], P_filt[t] = _kf_update(x_p, P_p, indicator_30m[t], H, R)

    # Précalcul des gains RTS C[t] = P_filt[t] @ A.T @ inv(P_pred[t+1])
    C = np.zeros((n, 2, 2))
    for t in range(n - 1):
        P_pk1 = P_pred[t + 1]
        det = P_pk1[0, 0] * P_pk1[1, 1] - P_pk1[0, 1] * P_pk1[1, 0]
        if abs(det) > 1e-15:
            inv_P = np.array([[P_pk1[1, 1], -P_pk1[0, 1]],
                              [-P_pk1[1, 0], P_pk1[0, 0]]]) / det
        else:
            inv_P = np.linalg.pinv(P_pk1)
        C[t] = P_filt[t] @ A.T @ inv_P

    return x_filt, P_filt, x_pred, P_pred, C


# ---------------------------------------------------------------------------
# Forward filter ADAPTIVE (Myers-Tapley AQ-KF) parameterisable
# ---------------------------------------------------------------------------

def _inv2x2(M):
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if abs(det) > 1e-15:
        return np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]]) / det
    return np.linalg.pinv(M)


def _is_pos_semidef(M):
    return M[0, 0] >= 0 and M[1, 1] >= 0 and (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]) >= -1e-12


def _forward_filter_30m_adaptive(
    indicator_30m, A, H, R,
    Q_init, Q_floor, Q_ceil,
    window: int = 30,
):
    """
    AQ-KF Myers-Tapley — forward filter avec Q adaptatif en ligne.

    Identique à `src/signal_processing/flks_substep_convergence.py:forward_filter_30m_adaptive`
    mais paramètres (A, H, R, Q_init, Q_floor, Q_ceil, window) injectés.

    Returns (x_filt, P_filt, x_pred, P_pred, C, Q_trace).
    """
    n = len(indicator_30m)
    first_valid_val = indicator_30m[~np.isnan(indicator_30m)][0]

    x_filt = np.zeros((n, 2))
    P_filt = np.zeros((n, 2, 2))
    x_pred = np.zeros((n, 2))
    P_pred = np.zeros((n, 2, 2))

    Q_current = Q_init.copy()
    innovation_buffer = []
    Q_trace = np.full((n, 2, 2), np.nan)

    for t in range(n):
        # 1. Predict
        if t == 0:
            x_p = np.array([first_valid_val, 0.0])
            P_p = np.eye(2)
        else:
            x_p = A @ x_filt[t - 1]
            P_p = A @ P_filt[t - 1] @ A.T + Q_current
        x_pred[t] = x_p
        P_pred[t] = P_p
        Q_trace[t] = Q_current

        # 2. Update
        if np.isnan(indicator_30m[t]):
            x_filt[t] = x_p
            P_filt[t] = P_p
            continue

        S_t = (H @ P_p @ H.T + R)[0, 0]
        x_filt[t], P_filt[t] = _kf_update(x_p, P_p, indicator_30m[t], H, R)

        # 3. Innovation
        v_t = indicator_30m[t] - (H @ x_p)[0]
        innovation_buffer.append(v_t)
        if len(innovation_buffer) > window:
            innovation_buffer.pop(0)

        # 4. Adaptive Q update (Myers-Tapley)
        if len(innovation_buffer) >= window and t > 0:
            C_vv = np.mean(np.array(innovation_buffer) ** 2)
            delta = C_vv - S_t
            if delta > 0:
                P_pred_next = A @ P_filt[t] @ A.T + Q_current
                C_rts = P_filt[t] @ A.T @ _inv2x2(P_pred_next)
                Q_candidate = delta * (C_rts @ C_rts.T)
                if _is_pos_semidef(Q_candidate):
                    Q_current = np.clip(Q_candidate, Q_floor, Q_ceil)

    # Précalcul gains RTS
    C_gains = np.zeros((n, 2, 2))
    for t in range(n - 1):
        C_gains[t] = P_filt[t] @ A.T @ _inv2x2(P_pred[t + 1])

    return x_filt, P_filt, x_pred, P_pred, C_gains, Q_trace


def _compute_slopes_test0_forward(x_filt):
    """
    Test 0 — Pure forward causal, lag 0 STRICT.

    Aucun backward smoothing. À l'instant t, on prédit la pente entre t-1
    et t-2 en utilisant UNIQUEMENT les états filtrés forward (pas
    d'observation à t).

        slopes[t] = x_filt[t-1, 0] - x_filt[t-2, 0]

    Ainsi la prédiction faite à l'instant t-1 (info disponible jusqu'à t-1)
    est évaluée à l'index t. Lag d'information = 0 (par rapport à t-1).
    """
    n = len(x_filt)
    slopes = np.full(n, np.nan)
    for t in range(2, n):
        slopes[t] = x_filt[t - 1, 0] - x_filt[t - 2, 0]
    return slopes


def _compute_slopes_test1(x_filt, x_pred, C):
    """Test 1 : FLKS 30min pur. Identique au script historique."""
    n = len(x_filt)
    slopes = np.full(n, np.nan)
    for t in range(2, n):
        sm_t1 = x_filt[t - 1] + C[t - 1] @ (x_filt[t] - x_pred[t])
        sm_t2 = x_filt[t - 2] + C[t - 2] @ (sm_t1 - x_pred[t - 1])
        slopes[t] = sm_t1[0] - sm_t2[0]
    return slopes


def _compute_slopes_test2(x_filt, P_filt, x_pred, C,
                          live_per_candle, n_substeps,
                          A_SUB, Q_SUB, H, R):
    """Test 2 : FLKS + n_substeps sous-pas 5min live de bougie t+1."""
    n = len(x_filt)
    slopes = np.full(n, np.nan)
    for t in range(2, n - 1):
        x_cur = x_filt[t].copy()
        P_cur = P_filt[t].copy()

        live_vals = live_per_candle[t + 1]
        valid_vals = [v for v in live_vals if not np.isnan(v)]
        use = valid_vals[:n_substeps]

        if len(use) > 0:
            for m5 in use:
                # Prédiction sous-pas
                x_cur = A_SUB @ x_cur
                P_cur = A_SUB @ P_cur @ A_SUB.T + Q_SUB
                # Update avec observation live
                x_cur, P_cur = _kf_update(x_cur, P_cur, m5, H, R)

        x_prov = x_cur
        k_actual = len(use) if len(use) > 0 else 1

        # Pas 1 : lisser t avec x_prov (backward depuis t + k/6 vers t)
        A_k = np.linalg.matrix_power(A_SUB, k_actual)
        Q_k = Q_SUB * k_actual
        x_pred_partial = A_k @ x_filt[t]
        P_pred_partial = A_k @ P_filt[t] @ A_k.T + Q_k
        det = P_pred_partial[0, 0] * P_pred_partial[1, 1] - P_pred_partial[0, 1] * P_pred_partial[1, 0]
        if abs(det) > 1e-15:
            inv_Pp = np.array([[P_pred_partial[1, 1], -P_pred_partial[0, 1]],
                               [-P_pred_partial[1, 0], P_pred_partial[0, 0]]]) / det
        else:
            inv_Pp = np.linalg.pinv(P_pred_partial)
        C_partial = P_filt[t] @ A_k.T @ inv_Pp
        sm_t = x_filt[t] + C_partial @ (x_prov - x_pred_partial)

        # Pas 2 : lisser t-1 avec smoothed[t]
        sm_t1 = x_filt[t - 1] + C[t - 1] @ (sm_t - x_pred[t])

        # Pas 3 : lisser t-2 avec smoothed[t-1]
        sm_t2 = x_filt[t - 2] + C[t - 2] @ (sm_t1 - x_pred[t - 1])

        slopes[t] = sm_t1[0] - sm_t2[0]
    return slopes


# ---------------------------------------------------------------------------
# Évaluation d'une calibration complète
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    name: str
    sigma2: float
    r_scalar: float
    mode: str           # 'fixed' | 'adaptive'
    t0_all: float = float("nan")   # Test 0 — forward pur, lag 0 strict, all samples
    t0_tr: float = float("nan")    # Test 0 — at transitions
    t1_all: float = float("nan")   # Test 1 — FLKS backward depuis x_filt[t], all
    t1_tr: float = float("nan")    # Test 1 — at transitions
    k_all: Dict[int, float] = field(default_factory=dict)   # Test 2 per k (1..6), all
    k_tr: Dict[int, float] = field(default_factory=dict)    # Test 2 per k, at transitions
    sigma2_mean: float = float("nan")   # For adaptive: mean of σ² trace
    sigma2_p95: float = float("nan")    # For adaptive: P95 of σ² trace
    frac_at_ceil: float = float("nan")  # For adaptive: fraction of time at upper bound


def run_calibration(
    name: str,
    sigma2: float,
    r_scalar: float,
    rsi_30m: np.ndarray,
    rsi_live_pc: List[np.ndarray],
    slopes_oracle: np.ndarray,
    trans_mask: np.ndarray,
    eval_start: int,
    n30: int,
) -> CalibrationResult:
    """Exécute Test 1 + Test 2 (k=1..6) pour une calibration donnée."""
    print(f"\n  --- {name}  (σ²={sigma2:.4g}, R={r_scalar:.4g}) ---")

    # Matrices pour cette calibration
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * sigma2
    R = np.array([[r_scalar]])
    DT_SUB = 1.0 / 6.0
    A_SUB = np.array([[1.0, DT_SUB], [0.0, 1.0]])
    Q_SUB = Q * DT_SUB

    # Forward filter
    x_filt, P_filt, x_pred, P_pred, C = _forward_filter_30m(rsi_30m, A, H, Q, R)

    # Test 0 — Forward pur (lag 0 strict, no backward smoothing)
    slopes_t0 = _compute_slopes_test0_forward(x_filt)
    c_t0_all, _ = sign_concordance(slopes_t0, slopes_oracle, eval_start, n30)
    c_t0_tr, _ = sign_concordance_at_transitions(slopes_t0, slopes_oracle, eval_start, n30, trans_mask)
    print(f"    Test 0 (forward pur) : all = {c_t0_all:6.2f}%   trans = {c_t0_tr:6.2f}%")

    # Test 1
    slopes_t1 = _compute_slopes_test1(x_filt, x_pred, C)
    c_t1_all, _ = sign_concordance(slopes_t1, slopes_oracle, eval_start, n30)
    c_t1_tr, _ = sign_concordance_at_transitions(slopes_t1, slopes_oracle, eval_start, n30, trans_mask)
    print(f"    Test 1 (30m + 1 lag) : all = {c_t1_all:6.2f}%   trans = {c_t1_tr:6.2f}%")

    # Test 2, k=1..6
    k_all_dict: Dict[int, float] = {}
    k_tr_dict: Dict[int, float] = {}
    for k in range(1, 7):
        slopes_k = _compute_slopes_test2(
            x_filt, P_filt, x_pred, C, rsi_live_pc, k,
            A_SUB, Q_SUB, H, R,
        )
        ck_all, _ = sign_concordance(slopes_k, slopes_oracle, eval_start, n30)
        ck_tr, _ = sign_concordance_at_transitions(slopes_k, slopes_oracle, eval_start, n30, trans_mask)
        k_all_dict[k] = ck_all
        k_tr_dict[k] = ck_tr
        print(f"    Test 2 k={k} ({k*5:2d}min) : all = {ck_all:6.2f}%   trans = {ck_tr:6.2f}%")

    return CalibrationResult(
        name=name,
        sigma2=float(sigma2),
        r_scalar=float(r_scalar),
        mode="fixed",
        t0_all=float(c_t0_all),
        t0_tr=float(c_t0_tr),
        t1_all=float(c_t1_all),
        t1_tr=float(c_t1_tr),
        k_all={int(k): float(v) for k, v in k_all_dict.items()},
        k_tr={int(k): float(v) for k, v in k_tr_dict.items()},
    )


def run_calibration_adaptive(
    name: str,
    sigma2_init: float,
    r_scalar: float,
    q_floor_factor: float,
    q_ceil_factor: float,
    window: int,
    rsi_30m: np.ndarray,
    rsi_live_pc: List[np.ndarray],
    slopes_oracle: np.ndarray,
    trans_mask: np.ndarray,
    eval_start: int,
    n30: int,
) -> CalibrationResult:
    """Exécute Test 1 + Test 2 en mode AQ-KF (Myers-Tapley adaptive Q)."""
    print(f"\n  --- {name}  (σ²_init={sigma2_init:.4g}, R={r_scalar:.4g}, "
          f"clip=[{sigma2_init*q_floor_factor:.4g}, {sigma2_init*q_ceil_factor:.4g}], W={window}) ---")

    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    R = np.array([[r_scalar]])
    Q_init = np.eye(2) * sigma2_init
    Q_floor = Q_init * q_floor_factor
    Q_ceil = Q_init * q_ceil_factor
    DT_SUB = 1.0 / 6.0
    A_SUB = np.array([[1.0, DT_SUB], [0.0, 1.0]])
    # Q_SUB : on utilise Q_final moyenné (approximation) — cohérent avec
    # l'implémentation du script historique qui fige Q au dernier niveau
    # adaptatif pour les sous-pas.

    x_filt, P_filt, x_pred, P_pred, C, Q_trace = _forward_filter_30m_adaptive(
        rsi_30m, A, H, R, Q_init, Q_floor, Q_ceil, window=window,
    )

    # Stats σ² adaptatif (Q_trace[:, 0, 0] = diagonale)
    sigma2_series = Q_trace[:, 0, 0]
    sigma2_series = sigma2_series[np.isfinite(sigma2_series)]
    sigma2_mean = float(np.mean(sigma2_series)) if len(sigma2_series) > 0 else float("nan")
    sigma2_p95 = float(np.percentile(sigma2_series, 95)) if len(sigma2_series) > 0 else float("nan")
    ceil_val = Q_ceil[0, 0]
    frac_at_ceil = float(np.mean(np.abs(sigma2_series - ceil_val) < 1e-6 * ceil_val)) if len(sigma2_series) > 0 else float("nan")
    print(f"    σ² adaptatif : mean={sigma2_mean:.4g}, P95={sigma2_p95:.4g}, "
          f"frac_at_ceil={frac_at_ceil*100:.1f}%")

    # Test 0 — Forward pur (lag 0 strict)
    slopes_t0 = _compute_slopes_test0_forward(x_filt)
    c_t0_all, _ = sign_concordance(slopes_t0, slopes_oracle, eval_start, n30)
    c_t0_tr, _ = sign_concordance_at_transitions(slopes_t0, slopes_oracle, eval_start, n30, trans_mask)
    print(f"    Test 0 (forward pur) : all = {c_t0_all:6.2f}%   trans = {c_t0_tr:6.2f}%")

    # Test 1
    slopes_t1 = _compute_slopes_test1(x_filt, x_pred, C)
    c_t1_all, _ = sign_concordance(slopes_t1, slopes_oracle, eval_start, n30)
    c_t1_tr, _ = sign_concordance_at_transitions(slopes_t1, slopes_oracle, eval_start, n30, trans_mask)
    print(f"    Test 1 (30m + 1 lag) : all = {c_t1_all:6.2f}%   trans = {c_t1_tr:6.2f}%")

    # Test 2 — pour les sous-pas, on utilise Q_sub = Q_mean_recent * DT_SUB
    # (approximation raisonnable : le script historique fige aussi Q au dernier niveau)
    Q_sub = np.eye(2) * sigma2_mean * DT_SUB

    k_all_dict: Dict[int, float] = {}
    k_tr_dict: Dict[int, float] = {}
    for k in range(1, 7):
        slopes_k = _compute_slopes_test2(
            x_filt, P_filt, x_pred, C, rsi_live_pc, k,
            A_SUB, Q_sub, H, R,
        )
        ck_all, _ = sign_concordance(slopes_k, slopes_oracle, eval_start, n30)
        ck_tr, _ = sign_concordance_at_transitions(slopes_k, slopes_oracle, eval_start, n30, trans_mask)
        k_all_dict[k] = ck_all
        k_tr_dict[k] = ck_tr
        print(f"    Test 2 k={k} ({k*5:2d}min) : all = {ck_all:6.2f}%   trans = {ck_tr:6.2f}%")

    return CalibrationResult(
        name=name,
        sigma2=float(sigma2_init),
        r_scalar=float(r_scalar),
        mode="adaptive",
        t0_all=float(c_t0_all),
        t0_tr=float(c_t0_tr),
        t1_all=float(c_t1_all),
        t1_tr=float(c_t1_tr),
        k_all={int(k): float(v) for k, v in k_all_dict.items()},
        k_tr={int(k): float(v) for k, v in k_tr_dict.items()},
        sigma2_mean=sigma2_mean,
        sigma2_p95=sigma2_p95,
        frac_at_ceil=frac_at_ceil,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="FLKS sub-step convergence — historique (A) vs MLE fixe (B) sur RSI")
    parser.add_argument("--csv", default="data_trad/BTCUSD_all_5m.csv")
    parser.add_argument("--n-candles-30m", type=int, default=5000)
    parser.add_argument("--eval-start", type=int, default=1000)
    parser.add_argument("--artifacts-dir", default=str(_HERE / "artifacts"))
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = _PROJECT_ROOT / csv_path

    bar = "=" * 78
    print(bar)
    print("FLKS SUB-STEP CONVERGENCE — A (historique) vs B (MLE fixe) — RSI")
    print(bar)

    # ---- 1. Load ---------------------------------------------------------
    print(f"\n[1/5] Loading {csv_path} ...")
    df_5m = load_csv(str(csv_path))
    print(f"       {len(df_5m):,} 5min candles")

    # ---- 2. Resample to 30min + crop last N -------------------------------
    print(f"\n[2/5] Resampling 5min → 30min + crop last {args.n_candles_30m:,} ...")
    df_30m = resample_ohlcv(df_5m, 30)
    if len(df_30m) > args.n_candles_30m:
        df_30m = df_30m.iloc[-args.n_candles_30m:]
    df_5m = df_5m.loc[df_30m.index[0] : df_30m.index[-1] + pd.Timedelta(minutes=29)]
    n30 = len(df_30m)
    print(f"       {n30:,} bougies 30min, {len(df_5m):,} bougies 5min")
    print(f"       Période : {df_30m.index[0]} → {df_30m.index[-1]}")

    # ---- 3. RSI 30min + RSI live (frozen/provisional) ---------------------
    print(f"\n[3/5] RSI 30min standard + RSI 5min live frozen/provisional ...")
    is_close = compute_bucket_close_mask(df_5m.index, 30)
    close_5m = df_5m["close"].values.astype(np.float64)
    rsi_30m = calculate_rsi(df_30m)
    rsi_live = compute_rsi_live(close_5m, is_close)

    # Group live values per 30min candle (identique au script historique)
    rsi_live_pc: List[np.ndarray] = []
    for ts_30m in df_30m.index:
        bucket_end = ts_30m + pd.Timedelta(minutes=29, seconds=59)
        mask = (df_5m.index >= ts_30m) & (df_5m.index <= bucket_end)
        rsi_live_pc.append(rsi_live[mask])

    # Coherence check
    max_err = 0.0
    n_checked = 0
    for t in range(n30):
        vals = [v for v in rsi_live_pc[t] if not np.isnan(v)]
        if len(vals) > 0 and not np.isnan(rsi_30m[t]):
            max_err = max(max_err, abs(vals[-1] - rsi_30m[t]))
            n_checked += 1
    print(f"       Coherence live vs standard : max err = {max_err:.2e} ({n_checked} candles)")

    # ---- 4. Oracle (params historiques, référence commune) ----------------
    print(f"\n[4/5] Oracle (pykalman.smooth, σ²=0.01, R=0.1 — référence commune) ...")
    _, slopes_oracle = compute_oracle(rsi_30m)
    trans_mask = find_oracle_transitions(slopes_oracle, args.eval_start, n30)
    n_trans = int(trans_mask.sum())
    EPSILON = 1e-8
    s_o = slopes_oracle[args.eval_start:n30]
    sign_o = np.where(np.abs(s_o) < EPSILON, 0, np.sign(s_o))
    valid_signs = sign_o[sign_o != 0]
    persistence = float(np.mean(valid_signs[1:] == valid_signs[:-1]) * 100.0) if len(valid_signs) > 1 else 0.0
    print(f"       Transitions oracle : {n_trans} ({n_trans/(n30-args.eval_start)*100:.1f}%)")
    print(f"       Persistence oracle : {persistence:.2f}%")

    # ---- 5. Run 2 calibrations -------------------------------------------
    print(f"\n[5/5] Test Kalman 2 calibrations ...")

    # --- Calibrations fixed (A + B) ---
    fixed_calibrations = [
        ("A_historique",  0.01,  0.1),
        ("B_MLE_fixe",    1.155, 3.27),
    ]
    results: List[CalibrationResult] = []
    for name, sigma2, r in fixed_calibrations:
        res = run_calibration(
            name=name, sigma2=sigma2, r_scalar=r,
            rsi_30m=rsi_30m, rsi_live_pc=rsi_live_pc,
            slopes_oracle=slopes_oracle, trans_mask=trans_mask,
            eval_start=args.eval_start, n30=n30,
        )
        results.append(res)

    # --- Calibrations adaptive (C1 historique + C2 unlocked) ---
    adaptive_calibrations = [
        # (name, sigma2_init, R, Q_floor_factor, Q_ceil_factor, window)
        ("C1_AQKF_historique", 0.01, 0.1, 0.1,  10.0,   30),    # clip [0.001, 0.1] (STATUS_v4.0)
        ("C2_AQKF_unlocked",   0.01, 0.1, 0.1,  1000.0, 30),    # clip [0.001, 10.0] permet adaptation MLE
    ]
    for name, s2_init, r, qfloor, qceil, win in adaptive_calibrations:
        res = run_calibration_adaptive(
            name=name, sigma2_init=s2_init, r_scalar=r,
            q_floor_factor=qfloor, q_ceil_factor=qceil, window=win,
            rsi_30m=rsi_30m, rsi_live_pc=rsi_live_pc,
            slopes_oracle=slopes_oracle, trans_mask=trans_mask,
            eval_start=args.eval_start, n30=n30,
        )
        results.append(res)

    # ---- Comparison table -------------------------------------------------
    print("\n" + "=" * 100)
    print("TABLEAU COMPARATIF — Concordance de signe vs Oracle (RSI)")
    print("=" * 100)

    # Header : T0 (lag 0 strict) puis k=0 (Test 1) puis k=1..6 (Test 2)
    col_labels = ["T0", "k=0", "k=1", "k=2", "k=3", "k=4", "k=5", "k=6"]
    hdr = f"{'Calibration':<22s} {'Test':<13s}" + "".join(f" {lbl:>7s}" for lbl in col_labels)
    print(hdr)
    print("-" * len(hdr))
    # Lignes : all + trans pour chaque calibration
    for res in results:
        row_all = f"{res.name:<22s} {'all':<13s} {res.t0_all:>7.2f} {res.t1_all:>7.2f}"
        row_tr = f"{res.name:<22s} {'transitions':<13s} {res.t0_tr:>7.2f} {res.t1_tr:>7.2f}"
        for k in range(1, 7):
            row_all += f" {res.k_all[k]:>7.2f}"
            row_tr += f" {res.k_tr[k]:>7.2f}"
        print(row_all)
        print(row_tr)
    print("-" * len(hdr))

    # Delta B - A (benchmark MLE vs historique)
    A_res = next(r for r in results if r.name == "A_historique")
    B_res = next(r for r in results if r.name == "B_MLE_fixe")
    row_delta_all = (f"{'Δ (B − A) MLE vs hist':<22s} {'all':<13s} "
                     f"{B_res.t0_all - A_res.t0_all:>+7.2f} {B_res.t1_all - A_res.t1_all:>+7.2f}")
    row_delta_tr = (f"{'Δ (B − A) MLE vs hist':<22s} {'transitions':<13s} "
                    f"{B_res.t0_tr - A_res.t0_tr:>+7.2f} {B_res.t1_tr - A_res.t1_tr:>+7.2f}")
    for k in range(1, 7):
        row_delta_all += f" {B_res.k_all[k] - A_res.k_all[k]:>+7.2f}"
        row_delta_tr += f" {B_res.k_tr[k] - A_res.k_tr[k]:>+7.2f}"
    print(row_delta_all)
    print(row_delta_tr)

    # Delta C1 - A (AQ-KF vs historique)
    C1_res = next(r for r in results if r.name == "C1_AQKF_historique")
    row_delta_c1_tr = (f"{'Δ (C1 − A) AQKF vs hist':<22s} {'transitions':<13s} "
                      f"{C1_res.t0_tr - A_res.t0_tr:>+7.2f} {C1_res.t1_tr - A_res.t1_tr:>+7.2f}")
    for k in range(1, 7):
        row_delta_c1_tr += f" {C1_res.k_tr[k] - A_res.k_tr[k]:>+7.2f}"
    print(row_delta_c1_tr)

    # Delta C2 - B (AQ-KF unlocked vs MLE)
    C2_res = next(r for r in results if r.name == "C2_AQKF_unlocked")
    row_delta_c2b_tr = (f"{'Δ (C2 − B) unlck vs MLE':<22s} {'transitions':<13s} "
                       f"{C2_res.t0_tr - B_res.t0_tr:>+7.2f} {C2_res.t1_tr - B_res.t1_tr:>+7.2f}")
    for k in range(1, 7):
        row_delta_c2b_tr += f" {C2_res.k_tr[k] - B_res.k_tr[k]:>+7.2f}"
    print(row_delta_c2b_tr)

    print("=" * len(hdr))

    # Adaptive stats
    print("\n  Stats σ² adaptatif (pour C1, C2) :")
    for res in results:
        if res.mode == "adaptive":
            print(f"    {res.name:<22s} σ²_mean={res.sigma2_mean:.4g}   σ²_P95={res.sigma2_p95:.4g}   "
                  f"frac_at_ceil={res.frac_at_ceil*100:.1f}%")

    # ---- Interprétation ---------------------------------------------------
    print("\n  Lecture :")
    print("  - 'all' = % concordance sur tous les samples (eval_start..n30)")
    print("  - 'transitions' = % concordance UNIQUEMENT aux points de changement de signe de l'oracle")
    print("  - Colonne T0 = Test 0 (forward pur, lag 0 STRICT — aucun backward smoothing)")
    print("  - Colonne k=0 = Test 1 (FLKS backward 1 lag, ≈ 30min de futur)")
    print("  - Colonnes k=1..6 = Test 2 (FLKS + k sous-pas 5min live de la bougie t+1)")
    print("  - Δ > 0 → B (MLE) mieux que A (historique). Δ < 0 → A mieux.")

    # ---- Save artifacts ---------------------------------------------------
    out = {
        "config": {
            "csv": str(csv_path),
            "n_candles_30m": int(n30),
            "eval_start": int(args.eval_start),
            "indicator": "RSI",
            "oracle_params": {"sigma2": 0.01, "r": 0.1, "mode": "historical_reference"},
            "fixed_calibrations": [{"name": n, "sigma2": s, "r": r} for n, s, r in fixed_calibrations],
            "adaptive_calibrations": [
                {"name": n, "sigma2_init": s, "r": r,
                 "q_floor_factor": qf, "q_ceil_factor": qc, "window": w}
                for n, s, r, qf, qc, w in adaptive_calibrations
            ],
            "n_transitions_oracle": int(n_trans),
            "persistence_oracle": persistence,
        },
        "results": [asdict(r) for r in results],
        "delta_B_minus_A": {
            "t0_all": B_res.t0_all - A_res.t0_all,
            "t0_tr": B_res.t0_tr - A_res.t0_tr,
            "t1_all": B_res.t1_all - A_res.t1_all,
            "t1_tr": B_res.t1_tr - A_res.t1_tr,
            "k_all": {int(k): B_res.k_all[k] - A_res.k_all[k] for k in range(1, 7)},
            "k_tr": {int(k): B_res.k_tr[k] - A_res.k_tr[k] for k in range(1, 7)},
        },
        "delta_C1_minus_A": {
            "t0_tr": C1_res.t0_tr - A_res.t0_tr,
            "t1_tr": C1_res.t1_tr - A_res.t1_tr,
            "k_tr": {int(k): C1_res.k_tr[k] - A_res.k_tr[k] for k in range(1, 7)},
        },
        "delta_C2_minus_B": {
            "t0_tr": C2_res.t0_tr - B_res.t0_tr,
            "t1_tr": C2_res.t1_tr - B_res.t1_tr,
            "k_tr": {int(k): C2_res.k_tr[k] - B_res.k_tr[k] for k in range(1, 7)},
        },
    }
    json_path = artifacts_dir / "flks_substep_mle_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nSauvegardé : {json_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
