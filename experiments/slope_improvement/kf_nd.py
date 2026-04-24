"""
Generic N-dimensional Kalman filter toolkit.

Designed to be reused for arbitrary state-space dimensions (2D CV, 3D WNA,
4D constant-jerk, IMM sub-models, etc.) without code duplication.

Model (scalar-driven process noise):

    x_{t+1} = F · x_t + G · w_t          w_t ~ N(0, σ²_drive)
    y_t     = H · x_t + ε_t              ε_t ~ N(0, R)

Process covariance Q = σ²_drive · G · G^T (rank-1, single scalar drives Q).

This design keeps MLE parameterization minimal (log σ²_drive, log R) and
avoids over-parameterization of Q when higher-order state components appear.

Provided functions (all generic, any state dim N):
    forward_filter  : causal KF pass, returns filtered + predicted states
                      and innovations
    rts_smoother    : non-causal backward smoother (Rauch-Tung-Striebel)
    neg_log_lik     : Gaussian NLL of innovations, with warmup
    mle_fit         : Nelder-Mead optimization of (σ²_drive, R) in log-space
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import optimize


@dataclass
class NDForwardResult:
    x_filt: np.ndarray       # (n, N)
    P_filt: np.ndarray       # (n, N, N)
    x_pred: np.ndarray       # (n, N)
    P_pred: np.ndarray       # (n, N, N)
    innov: np.ndarray        # (n,)   v_t = y - H x_{t|t-1}
    S: np.ndarray            # (n,)   H P_{t|t-1} H^T + R


def forward_filter(
    y: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    G: np.ndarray,
    sigma2_drive: float,
    r_scalar: float,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
) -> NDForwardResult:
    """
    Causal forward Kalman pass in arbitrary dimension.

    Parameters
    ----------
    y : (n,)       scalar observations (NaN tolerated: predict-only step)
    F : (N, N)     transition matrix
    H : (1, N)     observation matrix (scalar obs)
    G : (N, 1) or (N,)  process-noise coupling vector
    sigma2_drive : float, single scalar driving Q = σ² · G G^T
    r_scalar     : float, observation noise variance
    x0           : (N,) initial state mean. Default [y[first_finite], 0, …, 0]
    P0           : (N, N) initial cov. Default 100 · I_N  (diffuse prior)
    """
    y = np.asarray(y, dtype=float)
    N = F.shape[0]
    if G.ndim == 1:
        G = G.reshape(-1, 1)
    assert F.shape == (N, N), f"F must be {N}x{N}"
    assert H.shape == (1, N), f"H must be 1x{N}"
    assert G.shape == (N, 1), f"G must be {N}x1"

    GGT = G @ G.T
    R_mat = np.array([[r_scalar]])
    Q = sigma2_drive * GGT

    n = len(y)
    x_filt = np.zeros((n, N))
    P_filt = np.zeros((n, N, N))
    x_pred = np.zeros((n, N))
    P_pred = np.zeros((n, N, N))
    innov = np.full(n, np.nan)
    S = np.full(n, np.nan)

    # First finite observation
    start = 0
    while start < n and not np.isfinite(y[start]):
        start += 1
    if start >= n:
        return NDForwardResult(x_filt, P_filt, x_pred, P_pred, innov, S)

    if x0 is None:
        x = np.zeros(N)
        x[0] = y[start]
    else:
        x = x0.copy()
    P = P0.copy() if P0 is not None else np.eye(N) * 100.0

    x_filt[start] = x
    P_filt[start] = P
    x_pred[start] = x.copy()
    P_pred[start] = P.copy()

    I_N = np.eye(N)
    for t in range(start + 1, n):
        # Predict
        x_p = F @ x
        P_p = F @ P @ F.T + Q
        x_pred[t] = x_p
        P_pred[t] = P_p

        if not np.isfinite(y[t]):
            x = x_p
            P = P_p
            x_filt[t] = x
            P_filt[t] = P
            continue

        # Innovation
        y_hat = float((H @ x_p)[0])
        v = float(y[t] - y_hat)
        S_t = float((H @ P_p @ H.T)[0, 0] + r_scalar)
        innov[t] = v
        S[t] = S_t

        # Update
        K = (P_p @ H.T / S_t).ravel()   # (N,)
        x = x_p + K * v
        P = (I_N - np.outer(K, H.ravel())) @ P_p
        x_filt[t] = x
        P_filt[t] = P

    return NDForwardResult(x_filt, P_filt, x_pred, P_pred, innov, S)


def rts_smoother(
    y: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    G: np.ndarray,
    sigma2_drive: float,
    r_scalar: float,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Full RTS backward smoother. Returns smoothed state means (n, N).
    """
    fwd = forward_filter(y, F, H, G, sigma2_drive, r_scalar, x0, P0)
    if G.ndim == 1:
        G = G.reshape(-1, 1)
    Q = sigma2_drive * (G @ G.T)
    n = len(y)
    N = F.shape[0]

    x_s = fwd.x_filt.copy()
    P_s = fwd.P_filt.copy()

    for t in range(n - 2, -1, -1):
        P_pred_next = F @ fwd.P_filt[t] @ F.T + Q
        try:
            C = fwd.P_filt[t] @ F.T @ np.linalg.inv(P_pred_next)
        except np.linalg.LinAlgError:
            C = fwd.P_filt[t] @ F.T @ np.linalg.pinv(P_pred_next)
        x_s[t] = fwd.x_filt[t] + C @ (x_s[t + 1] - F @ fwd.x_filt[t])
        P_s[t] = fwd.P_filt[t] + C @ (P_s[t + 1] - P_pred_next) @ C.T

    return x_s


def neg_log_lik(
    y: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    G: np.ndarray,
    sigma2_drive: float,
    r_scalar: float,
    warmup: int = 50,
) -> float:
    """
    Gaussian NLL of the innovations, skipping the first `warmup` samples
    to mitigate diffuse-prior influence.
    """
    fwd = forward_filter(y, F, H, G, sigma2_drive, r_scalar)
    n = len(y)
    if warmup >= n:
        return float("inf")
    v = fwd.innov[warmup:]
    S = fwd.S[warmup:]
    mask = np.isfinite(v) & np.isfinite(S) & (S > 0)
    if mask.sum() < 10:
        return float("inf")
    v_ = v[mask]
    S_ = S[mask]
    ll = -0.5 * np.sum(np.log(2 * np.pi * S_) + v_ * v_ / S_)
    return float(-ll)


@dataclass
class MLENDResult:
    sigma2_drive: float
    r_scalar: float
    nll: float
    success: bool
    n_iter: int
    n_eval: int
    n_samples_used: int
    init_sigma2: float
    init_r: float


def mle_fit(
    y: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    G: np.ndarray,
    init_sigma2: float,
    init_r: float,
    subsample_n: Optional[int] = 20_000,
    warmup: int = 50,
    maxiter: int = 200,
    verbose: bool = False,
) -> MLENDResult:
    """
    Fit (σ²_drive, R) by Nelder-Mead in log-space for any N-D model.
    """
    if subsample_n is not None and len(y) > subsample_n:
        y_fit = y[:subsample_n]
    else:
        y_fit = y

    def _obj(theta: np.ndarray) -> float:
        return neg_log_lik(
            y_fit, F, H, G,
            sigma2_drive=float(np.exp(theta[0])),
            r_scalar=float(np.exp(theta[1])),
            warmup=warmup,
        )

    x0 = np.array([np.log(init_sigma2), np.log(init_r)])
    res = optimize.minimize(
        _obj, x0, method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-2, "maxiter": maxiter, "disp": verbose},
    )

    return MLENDResult(
        sigma2_drive=float(np.exp(res.x[0])),
        r_scalar=float(np.exp(res.x[1])),
        nll=float(res.fun),
        success=bool(res.success),
        n_iter=int(res.nit),
        n_eval=int(res.nfev),
        n_samples_used=int(len(y_fit)),
        init_sigma2=float(init_sigma2),
        init_r=float(init_r),
    )


# ---------------------------------------------------------------------------
# Information criteria (for fair model comparison)
# ---------------------------------------------------------------------------

def aic(nll: float, k: int) -> float:
    """Akaike Information Criterion. k = number of free parameters."""
    return 2 * k + 2 * nll


def bic(nll: float, k: int, n: int) -> float:
    """Bayesian Information Criterion. n = effective sample size."""
    return k * np.log(n) + 2 * nll


if __name__ == "__main__":
    # Smoke test: 3D WNA with known truth
    rng = np.random.default_rng(0)
    n = 3000
    F = np.array([[1.0, 1.0, 0.5], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    H = np.array([[1.0, 0.0, 0.0]])
    G = np.array([[0.5], [1.0], [1.0]])
    true_s2 = 1e-3
    true_r = 2.0
    x = np.zeros((n, 3))
    x[0] = [50.0, 0.0, 0.0]
    for t in range(1, n):
        w = rng.standard_normal() * np.sqrt(true_s2)
        x[t] = F @ x[t - 1] + G.ravel() * w
    y = x[:, 0] + rng.standard_normal(n) * np.sqrt(true_r)

    mle = mle_fit(y, F, H, G, init_sigma2=1e-4, init_r=1.0, subsample_n=n)
    print(f"3D MLE: σ²={mle.sigma2_drive:.5g} (true {true_s2})  R={mle.r_scalar:.3f} (true {true_r})")
    print(f"       NLL={mle.nll:.2f}  success={mle.success}")

    # Smoke test: 4D constant-jerk
    F4 = np.array([
        [1.0, 1.0, 0.5, 1.0 / 6.0],
        [0.0, 1.0, 1.0, 0.5],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    H4 = np.array([[1.0, 0.0, 0.0, 0.0]])
    G4 = np.array([[1.0 / 6.0], [0.5], [1.0], [1.0]])
    mle4 = mle_fit(y, F4, H4, G4, init_sigma2=1e-5, init_r=1.0, subsample_n=n)
    print(f"4D MLE: σ²_jerk={mle4.sigma2_drive:.5g}  R={mle4.r_scalar:.3f}")
    print(f"        NLL={mle4.nll:.2f}  ΔNLL(4D-3D)={mle4.nll - mle.nll:+.2f}  success={mle4.success}")
    # AIC/BIC comparison
    print(f"AIC 3D={aic(mle.nll, 2):.2f}   AIC 4D={aic(mle4.nll, 2):.2f}")
