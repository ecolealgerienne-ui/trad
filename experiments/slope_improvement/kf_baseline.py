"""
Baseline KF (2D constant-velocity, scalar Myers-Tapley Q adaptation).

Differences vs `src/prepare_multitf_csv_aqkf.py:compute_kalman_live`:
  - Barre-par-barre pur, sans logique closure/provisional multi-TF
    (le double-étage était utile en prod live sur TF > 5min ; ici on
    opère sur 5min natif).
  - Adaptation scalaire σ_velocity² au lieu de Q matriciel Myers-Tapley
    (conformément au cahier des charges pour fair comparison avec
    l'Étape 2 3D WNA qui adapte σ_accel² scalaire).

Modèle
------
État      x = [level, velocity]                     (dim 2)
Transition F = [[1, 1], [0, 1]]                     (dt = 1 barre)
Observation H = [[1, 0]]                             (on observe le level)
Process noise Q = σ_vel² · G · G^T avec G = [1, 1]^T (rank-1)
                => H G G^T H^T = (H G)² = 1
Measurement noise R = scalaire fixe (r_scalar)

Adaptation σ_vel² (scalaire Myers-Tapley)
-----------------------------------------
Buffer glissant W des dernières innovations v_t = y_t - H x_{t|t-1}.
Quand le buffer est plein:
    C_vv  = mean(v_t²) sur la fenêtre
    S_base = H · F P_filt[t-1] F^T · H^T + R      (variance d'innovation si σ²=0)
    σ²_target = max(C_vv - S_base, σ²_min)        (inversion directe)
    σ²_current = clip(σ²_target, σ²_min, σ²_max)

Puisque H G G^T H^T = 1, l'inversion directe est exacte : ajouter σ² à
H P_p H^T augmente S_t d'exactement σ². Pas de damping (identique en
esprit à l'update "direct" du Myers-Tapley existant, qui remplaçait
Q_current directement quand δ>0).

Causalité : toutes les estimations x[t] dépendent uniquement de y[0..t].

REUSE:
    - src.constants.KALMAN_PROCESS_VAR  (σ² initial)
    - src.constants.KALMAN_MEASURE_VAR  (R fixe)

Does NOT modify src/prepare_multitf_csv_aqkf.py; this is a parallel,
simplified wrapper for controlled experimentation (per user spec:
"Créer un wrapper KF simplifié barre-par-barre, SANS la logique
closure/provisional").
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Reuse project constants.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from constants import KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR  # noqa: E402

# Project defaults exposed at module level for convenience.
_Q_SCALAR_INIT = KALMAN_PROCESS_VAR
_R_DEFAULT = KALMAN_MEASURE_VAR


@dataclass
class KFBaselineResult:
    """Container for baseline KF output."""
    level: np.ndarray          # shape (n,)  = x_filt[:, 0]
    slope: np.ndarray          # shape (n,)  = x_filt[:, 1]
    innovations: np.ndarray    # shape (n,)  raw v_t (NaN before first update)
    S: np.ndarray              # shape (n,)  innovation variance H P_p H^T + R
    sigma2_trace: np.ndarray   # shape (n,)  σ_vel² used at each step (after update)
    P_diag: np.ndarray         # shape (n, 2) diagonal of filtered covariance

    @property
    def n(self) -> int:
        return len(self.level)


def run_kf_baseline(
    y: np.ndarray,
    *,
    sigma2_init: float = _Q_SCALAR_INIT,
    sigma2_min: float = _Q_SCALAR_INIT * 0.1,
    sigma2_max: float = _Q_SCALAR_INIT * 10.0,
    r_scalar: float = _R_DEFAULT,
    aq_window: int = 30,
    warmup: Optional[int] = None,
) -> KFBaselineResult:
    """
    Run the baseline adaptive-σ² Kalman filter on a univariate series.

    Parameters
    ----------
    y : array (n,)
        Observations (e.g., RSI values). NaN-tolerant: skipped observations
        trigger predict-only steps.
    sigma2_init : float
        Initial σ_vel². Default 0.01 (project KALMAN_PROCESS_VAR).
    sigma2_min, sigma2_max : float
        Clipping bounds. Defaults = σ²_init * {0.1, 10.0}, matching the prod
        AQ-KF bounds.
    r_scalar : float
        Measurement noise variance (fixed in this baseline).
    aq_window : int
        Innovation buffer size for Myers-Tapley adaptation. Default 30
        (matches prod AQ-KF).
    warmup : int, optional
        Number of initial samples during which σ² stays at sigma2_init
        without adaptation. If None, defaults to aq_window (adaptation only
        kicks in once the buffer is full).

    Returns
    -------
    KFBaselineResult

    Notes
    -----
    * Innovation v_t and variance S_t are stored BEFORE the update step, so
      they can be directly fed into diagnostics (z_t = v_t / sqrt(S_t)).
    * For the FIRST sample (t=0), we initialize x = [y[0], 0] with P = I.
      Innovations[0] and S[0] are NaN (no prior prediction to compare to).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        raise ValueError(f"Série trop courte: n={n}")

    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    G = np.array([[1.0], [1.0]])
    GGT = G @ G.T  # 2x2 rank-1 matrix [[1,1],[1,1]]
    HGGTHT = float((H @ GGT @ H.T)[0, 0])  # = 1.0 by design
    R = float(r_scalar)

    if warmup is None:
        warmup = aq_window

    # Output allocations
    level = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    innovations = np.full(n, np.nan)
    S_out = np.full(n, np.nan)
    sigma2_trace = np.full(n, np.nan)
    P_diag = np.full((n, 2), np.nan)

    # Innovation buffer (deque-like on a plain list for speed)
    innov_buf: list[float] = []

    # Find first finite observation
    start = 0
    while start < n and not np.isfinite(y[start]):
        start += 1
    if start >= n - 2:
        return KFBaselineResult(level, slope, innovations, S_out, sigma2_trace, P_diag)

    # Initialize state
    x = np.array([y[start], 0.0])
    P = np.eye(2)
    sigma2 = sigma2_init
    level[start] = x[0]
    slope[start] = x[1]
    sigma2_trace[start] = sigma2
    P_diag[start] = np.diag(P).copy()

    for t in range(start + 1, n):
        # Current Q based on current σ²
        Q = sigma2 * GGT

        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        if not np.isfinite(y[t]):
            # No observation -> state = prediction
            x = x_pred
            P = P_pred
            level[t] = x[0]
            slope[t] = x[1]
            sigma2_trace[t] = sigma2
            P_diag[t] = np.diag(P).copy()
            continue

        # Innovation BEFORE update
        y_hat = float((H @ x_pred)[0])
        v = float(y[t] - y_hat)
        S = float((H @ P_pred @ H.T)[0, 0] + R)
        innovations[t] = v
        S_out[t] = S

        # Update
        K = (P_pred @ H.T / S).ravel()  # shape (2,)
        x = x_pred + K * v
        P = (np.eye(2) - np.outer(K, H.ravel())) @ P_pred

        level[t] = x[0]
        slope[t] = x[1]
        P_diag[t] = np.diag(P).copy()

        # --- Scalar Myers-Tapley adaptation of σ² ---
        innov_buf.append(v)
        if len(innov_buf) > aq_window:
            innov_buf.pop(0)

        adapt_ready = (t >= start + warmup) and (len(innov_buf) >= aq_window)
        if adapt_ready:
            C_vv = float(np.mean(np.square(innov_buf)))
            # Expected innovation variance if σ² were 0 at the NEXT step.
            # Use current P (post-update) to project: P_pred_next = F P F^T + 0·GG^T
            P_pred_next_noQ = F @ P @ F.T
            S_base_next = float((H @ P_pred_next_noQ @ H.T)[0, 0] + R)
            sigma2_target = (C_vv - S_base_next) / HGGTHT
            if sigma2_target < sigma2_min:
                sigma2 = sigma2_min
            elif sigma2_target > sigma2_max:
                sigma2 = sigma2_max
            else:
                sigma2 = sigma2_target

        sigma2_trace[t] = sigma2

    return KFBaselineResult(
        level=level,
        slope=slope,
        innovations=innovations,
        S=S_out,
        sigma2_trace=sigma2_trace,
        P_diag=P_diag,
    )


if __name__ == "__main__":
    # Smoke test on synthetic signal: linear trend that changes regime
    rng = np.random.default_rng(0)
    n = 5000
    t = np.arange(n)
    true_vel = np.where(t < n // 2, 0.02, -0.01)
    level_true = np.cumsum(true_vel)
    y = level_true + rng.standard_normal(n) * 0.5
    res = run_kf_baseline(y, sigma2_init=1e-4, r_scalar=0.25)
    print(f"slope[-1] = {res.slope[-1]:.4f}  (true = {true_vel[-1]:.4f})")
    print(f"σ² final  = {res.sigma2_trace[-1]:.6f}")
    innov = res.innovations[np.isfinite(res.innovations)]
    S_v = res.S[np.isfinite(res.S)]
    z = innov / np.sqrt(S_v)
    print(f"z mean={z.mean():.4f}  z std={z.std():.4f}  (expect ~0, ~1 if well-specified)")
