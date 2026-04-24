"""
3D White-Noise Acceleration (WNA) Kalman — utility module.

Used for :
  (a) computing the OFFICIAL ground truth for slope (RTS full-pass
      with (σ²_accel, R) fitted by global MLE on train).
  (b) Étape 2 du plan d'amélioration (sera importé depuis ici).

Model (Bar-Shalom standard WNA):

    x_{t+1} = F · x_t + G · w_t       w_t ~ N(0, σ²_accel)
    y_t     = H · x_t + ε_t           ε_t ~ N(0, R)

With dt = 1 (1 barre):

    F = [[1, 1, 0.5],
         [0, 1,   1],
         [0, 0,   1]]

    G = [0.5, 1, 1]^T

    Q = σ²_accel · G · G^T   (rank-1, driven by a single scalar)

    H = [[1, 0, 0]]

State x = [level, slope, accel]. Slope = x[:, 1].

MLE parameters : θ = (log σ²_accel, log R). Positivity enforced by
parameterization, no explicit bounds needed.

IMPORTANT :
  - This module does NOT implement adaptive Q. (σ²_accel, R) are fixed
    per run. The adaptation layer belongs to kf_augmented.py (Étape 2).
  - Reused directly by validate_gt_and_R.py to build the official GT.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy import optimize

# Reuse project constants.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from constants import KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR  # noqa: E402


# ---------------------------------------------------------------------------
# Model matrices (module-level constants, built once)
# ---------------------------------------------------------------------------

F_3 = np.array([
    [1.0, 1.0, 0.5],
    [0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0],
])
H_3 = np.array([[1.0, 0.0, 0.0]])
G_3 = np.array([[0.5], [1.0], [1.0]])
GGT_3 = G_3 @ G_3.T  # 3x3 rank-1

INIT_COV_DIFFUSE = np.eye(3) * 100.0  # "diffuse" prior — standard in MLE


# ---------------------------------------------------------------------------
# Forward filter (fixed Q, R — no adaptation)
# ---------------------------------------------------------------------------

def forward_filter_3d(
    y: np.ndarray,
    sigma2_accel: float,
    r_scalar: float,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard forward Kalman pass on 3D WNA model.

    Returns
    -------
    x_filt : (n, 3)   filtered state means
    P_filt : (n, 3, 3) filtered covariances
    x_pred : (n, 3)   one-step-ahead predicted state (x_{t|t-1})
    P_pred : (n, 3, 3)
    innov  : (n,)    v_t = y_t - H x_{t|t-1}  (NaN for skipped/first)
    S_arr  : (n,)    H P_{t|t-1} H^T + R        (NaN for skipped/first)
    """
    n = len(y)
    Q = sigma2_accel * GGT_3
    R = np.array([[r_scalar]])

    x_filt = np.zeros((n, 3))
    P_filt = np.zeros((n, 3, 3))
    x_pred = np.zeros((n, 3))
    P_pred = np.zeros((n, 3, 3))
    innov = np.full(n, np.nan)
    S_arr = np.full(n, np.nan)

    # First finite observation
    start = 0
    while start < n and not np.isfinite(y[start]):
        start += 1
    if start >= n:
        return x_filt, P_filt, x_pred, P_pred, innov, S_arr

    x = x0.copy() if x0 is not None else np.array([y[start], 0.0, 0.0])
    P = P0.copy() if P0 is not None else INIT_COV_DIFFUSE.copy()
    x_filt[start] = x
    P_filt[start] = P
    x_pred[start] = x.copy()
    P_pred[start] = P.copy()

    for t in range(start + 1, n):
        # Predict
        x_p = F_3 @ x
        P_p = F_3 @ P @ F_3.T + Q
        x_pred[t] = x_p
        P_pred[t] = P_p

        if not np.isfinite(y[t]):
            x = x_p
            P = P_p
            x_filt[t] = x
            P_filt[t] = P
            continue

        # Innovation
        y_hat = float((H_3 @ x_p)[0])
        v = float(y[t] - y_hat)
        S = float((H_3 @ P_p @ H_3.T)[0, 0] + r_scalar)
        innov[t] = v
        S_arr[t] = S

        # Update
        K = (P_p @ H_3.T / S).ravel()  # (3,)
        x = x_p + K * v
        P = (np.eye(3) - np.outer(K, H_3.ravel())) @ P_p
        x_filt[t] = x
        P_filt[t] = P

    return x_filt, P_filt, x_pred, P_pred, innov, S_arr


# ---------------------------------------------------------------------------
# RTS smoother (non-causal backward pass)
# ---------------------------------------------------------------------------

def rts_smoother_3d(
    y: np.ndarray,
    sigma2_accel: float,
    r_scalar: float,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Full RTS smoother on 3D WNA. Returns (n, 3) smoothed state means.
    """
    x_filt, P_filt, _, _, _, _ = forward_filter_3d(y, sigma2_accel, r_scalar, x0, P0)
    n = len(y)
    Q = sigma2_accel * GGT_3

    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()

    for t in range(n - 2, -1, -1):
        P_pred_next = F_3 @ P_filt[t] @ F_3.T + Q
        try:
            C = P_filt[t] @ F_3.T @ np.linalg.inv(P_pred_next)
        except np.linalg.LinAlgError:
            C = P_filt[t] @ F_3.T @ np.linalg.pinv(P_pred_next)
        x_smooth[t] = x_filt[t] + C @ (x_smooth[t + 1] - F_3 @ x_filt[t])
        P_smooth[t] = P_filt[t] + C @ (P_smooth[t + 1] - P_pred_next) @ C.T

    return x_smooth


# ---------------------------------------------------------------------------
# Negative log-likelihood (for MLE)
# ---------------------------------------------------------------------------

def neg_log_likelihood_3d(
    y: np.ndarray,
    sigma2_accel: float,
    r_scalar: float,
    warmup: int = 50,
) -> float:
    """
    Gaussian NLL of the innovations under the 3D WNA model.

        NLL = 0.5 · Σ_{t > warmup} [log(2π S_t) + v_t² / S_t]

    `warmup` drops the first warmup samples to avoid the diffuse initial
    covariance dominating the likelihood.
    """
    _, _, _, _, v, S = forward_filter_3d(y, sigma2_accel, r_scalar)
    n = len(y)
    if warmup >= n:
        return float("inf")
    v_use = v[warmup:]
    S_use = S[warmup:]
    mask = np.isfinite(v_use) & np.isfinite(S_use) & (S_use > 0)
    if mask.sum() < 10:
        return float("inf")
    v_ = v_use[mask]
    S_ = S_use[mask]
    ll = -0.5 * np.sum(np.log(2 * np.pi * S_) + v_ * v_ / S_)
    return float(-ll)


# ---------------------------------------------------------------------------
# MLE fit
# ---------------------------------------------------------------------------

@dataclass
class MLEResult:
    sigma2_accel: float
    r_scalar: float
    nll: float
    success: bool
    n_iter: int
    n_eval: int
    n_samples_used: int
    init_sigma2: float
    init_r: float


def mle_fit_3d_wna(
    y: np.ndarray,
    init_sigma2: float = 1e-3,
    init_r: float = 3.0,
    subsample_n: Optional[int] = 20_000,
    warmup: int = 50,
    verbose: bool = False,
) -> MLEResult:
    """
    Fit (σ²_accel, R) by maximum likelihood via Nelder-Mead in log-space.

    `subsample_n` : if not None, uses only the first `subsample_n` samples
    of `y` for the MLE (keeps fit < 2 min on 420k-sample series).
    """
    if subsample_n is not None and len(y) > subsample_n:
        y_fit = y[:subsample_n]
    else:
        y_fit = y

    def _obj(theta: np.ndarray) -> float:
        log_s2, log_r = theta
        return neg_log_likelihood_3d(
            y_fit,
            sigma2_accel=float(np.exp(log_s2)),
            r_scalar=float(np.exp(log_r)),
            warmup=warmup,
        )

    x0 = np.array([np.log(init_sigma2), np.log(init_r)])
    res = optimize.minimize(
        _obj, x0,
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-2, "maxiter": 200, "disp": verbose},
    )

    s2_opt = float(np.exp(res.x[0]))
    r_opt = float(np.exp(res.x[1]))
    return MLEResult(
        sigma2_accel=s2_opt,
        r_scalar=r_opt,
        nll=float(res.fun),
        success=bool(res.success),
        n_iter=int(res.nit),
        n_eval=int(res.nfev),
        n_samples_used=int(len(y_fit)),
        init_sigma2=float(init_sigma2),
        init_r=float(init_r),
    )


# ---------------------------------------------------------------------------
# Official Ground Truth builder
# ---------------------------------------------------------------------------

@dataclass
class OfficialGT:
    slope: np.ndarray           # (n,) smoothed slope = primary GT
    level: np.ndarray           # (n,)
    accel: np.ndarray           # (n,)
    sigma2_accel: float         # MLE fitted
    r_scalar: float             # MLE fitted
    nll: float                  # fit NLL
    n_fit_samples: int          # samples used for MLE
    n_full_samples: int         # samples in full RTS pass


def compute_official_ground_truth(
    rsi_full: np.ndarray,
    train_end_idx: Optional[int] = None,
    subsample_n: int = 20_000,
    init_sigma2: float = 1e-3,
    init_r: float = 3.0,
    warmup: int = 50,
    verbose: bool = True,
) -> OfficialGT:
    """
    Build the OFFICIAL non-contaminated ground truth for the project:
        1. Fit (σ²_accel, R) by MLE on train subsample (3D WNA model).
        2. Run full RTS smoother on the FULL series with those parameters.
        3. Return smoothed [level, slope, accel].

    Rationale (utilisateur):
        > "le RTS GT doit avoir PLUS de degrés de liberté que les modèles
        > testés [...] baseline 2D et variantes 3D/IMM seront toutes évaluées
        > contre un GT plus riche qu'aucun d'entre eux, ce qui rend la MSE
        > informative et les classements fiables."

    Parameters
    ----------
    rsi_full : (n,)
        RSI series post-warmup (concat of train+val+test).
    train_end_idx : int, optional
        Index marking end of train split. MLE uses only first
        min(train_end_idx, subsample_n) samples. If None, uses first
        `subsample_n` globally.
    subsample_n : int
        Max samples used for MLE fit.
    """
    if train_end_idx is not None:
        y_available = rsi_full[:train_end_idx]
    else:
        y_available = rsi_full

    if verbose:
        print(f"  [GT MLE] fitting (σ²_accel, R) on {min(len(y_available), subsample_n):,} samples...")

    mle = mle_fit_3d_wna(
        y_available,
        init_sigma2=init_sigma2,
        init_r=init_r,
        subsample_n=subsample_n,
        warmup=warmup,
        verbose=False,
    )
    if verbose:
        print(f"  [GT MLE] σ²_accel = {mle.sigma2_accel:.6g}  R = {mle.r_scalar:.6g}")
        print(f"  [GT MLE] NLL = {mle.nll:.2f}  success={mle.success}  iters={mle.n_iter}  evals={mle.n_eval}")
        print(f"  [GT RTS] running full-pass smoother on {len(rsi_full):,} samples...")

    x_smooth = rts_smoother_3d(rsi_full, mle.sigma2_accel, mle.r_scalar)

    return OfficialGT(
        slope=x_smooth[:, 1],
        level=x_smooth[:, 0],
        accel=x_smooth[:, 2],
        sigma2_accel=mle.sigma2_accel,
        r_scalar=mle.r_scalar,
        nll=mle.nll,
        n_fit_samples=mle.n_samples_used,
        n_full_samples=len(rsi_full),
    )


if __name__ == "__main__":
    # Smoke test: synthetic WNA signal
    rng = np.random.default_rng(0)
    n = 5000
    true_s2_accel = 1e-3
    true_r = 2.0
    x = np.zeros((n, 3))
    x[0] = [50.0, 0.0, 0.0]
    for t in range(1, n):
        w = rng.standard_normal() * np.sqrt(true_s2_accel)
        x[t] = F_3 @ x[t - 1] + G_3.ravel() * w
    y = x[:, 0] + rng.standard_normal(n) * np.sqrt(true_r)

    mle = mle_fit_3d_wna(y, init_sigma2=1e-4, init_r=1.0, subsample_n=5000)
    print(f"True (σ²={true_s2_accel}, R={true_r})")
    print(f"MLE  (σ²={mle.sigma2_accel:.6g}, R={mle.r_scalar:.6g})  NLL={mle.nll:.2f}  success={mle.success}")

    gt = compute_official_ground_truth(y, subsample_n=5000, init_sigma2=1e-4, init_r=1.0, verbose=False)
    # Correlation: smoothed slope vs true slope
    corr = np.corrcoef(x[100:-100, 1], gt.slope[100:-100])[0, 1]
    print(f"Pearson(true slope, RTS smoothed) = {corr:.4f}")
