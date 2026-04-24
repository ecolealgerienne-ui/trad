"""
4D constant-jerk Kalman model — used to build a super-parameterized GT
that is structurally richer than any model tested downstream.

Rationale (utilisateur) :
    > "Partager la structure 3D entre GT et baseline testé invalide la
    >  comparaison 2D vs 3D. [...] construire un GT sur-paramétré 4D."

Model :

    x = [level, slope, accel, jerk]    (dim 4)

    F = [[1,  1, 0.5, 1/6],
         [0,  1,  1,  0.5],
         [0,  0,  1,   1 ],
         [0,  0,  0,   1 ]]

    G = [1/6, 0.5, 1, 1]^T   (constant-jerk white-noise coupling)

    H = [1, 0, 0, 0]

    Q = σ²_jerk · G · G^T       (rank-1, single scalar drives Q)

The 4D GT, via RTS smoother, produces a smoothed slope (x[:, 1]) which
serves as the OFFICIAL ground-truth reference for comparing 2D CV, 3D WNA
and any future variant. Since none of the tested models includes a jerk
term or backward smoothing, the GT 4D is strictly richer in structure.

All Kalman machinery is delegated to kf_nd.py to avoid duplication.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from kf_nd import (
    forward_filter,
    rts_smoother,
    neg_log_lik,
    mle_fit,
    MLENDResult,
)


# ---------------------------------------------------------------------------
# 4D constant-jerk matrices
# ---------------------------------------------------------------------------

F_4 = np.array([
    [1.0, 1.0, 0.5, 1.0 / 6.0],
    [0.0, 1.0, 1.0, 0.5],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
])
H_4 = np.array([[1.0, 0.0, 0.0, 0.0]])
G_4 = np.array([[1.0 / 6.0], [0.5], [1.0], [1.0]])
INIT_COV_DIFFUSE_4 = np.eye(4) * 100.0


# ---------------------------------------------------------------------------
# Thin convenience wrappers
# ---------------------------------------------------------------------------

def forward_filter_4d(
    y: np.ndarray,
    sigma2_jerk: float,
    r_scalar: float,
    **kwargs,
):
    """Forward KF pass with 4D constant-jerk model."""
    return forward_filter(y, F_4, H_4, G_4, sigma2_jerk, r_scalar, **kwargs)


def rts_smoother_4d(
    y: np.ndarray,
    sigma2_jerk: float,
    r_scalar: float,
) -> np.ndarray:
    """Full RTS smoother with 4D constant-jerk model → (n, 4) smoothed state."""
    return rts_smoother(y, F_4, H_4, G_4, sigma2_jerk, r_scalar)


def neg_log_lik_4d(
    y: np.ndarray,
    sigma2_jerk: float,
    r_scalar: float,
    warmup: int = 50,
) -> float:
    return neg_log_lik(y, F_4, H_4, G_4, sigma2_jerk, r_scalar, warmup=warmup)


def mle_fit_4d(
    y: np.ndarray,
    init_sigma2: float = 1e-5,
    init_r: float = 6.0,
    subsample_n: Optional[int] = 20_000,
    warmup: int = 50,
    maxiter: int = 300,
    verbose: bool = False,
) -> MLENDResult:
    """
    MLE fit of (σ²_jerk, R) on 4D constant-jerk model.

    Defaults :
        init_sigma2 = 1e-5  (jerk noise is expected smaller than accel noise
                             from the 3D fit, σ²_accel ≈ 0.07)
        init_r      = 6.0   (seed near the 3D MLE R for continuity)
    """
    return mle_fit(
        y, F_4, H_4, G_4,
        init_sigma2=init_sigma2,
        init_r=init_r,
        subsample_n=subsample_n,
        warmup=warmup,
        maxiter=maxiter,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Full ground-truth builder
# ---------------------------------------------------------------------------

@dataclass
class OfficialGT4D:
    slope: np.ndarray          # (n,)  PRIMARY
    level: np.ndarray          # (n,)
    accel: np.ndarray          # (n,)
    jerk: np.ndarray           # (n,)
    sigma2_jerk: float
    r_scalar: float
    nll: float
    n_fit_samples: int
    n_full_samples: int


def compute_official_gt_4d(
    rsi_full: np.ndarray,
    train_end_idx: Optional[int] = None,
    subsample_n: int = 20_000,
    init_sigma2: float = 1e-5,
    init_r: float = 6.0,
    warmup: int = 50,
    verbose: bool = True,
) -> OfficialGT4D:
    """
    Build the 4D super-parameterized GT:
        1. MLE fit of (σ²_jerk, R) on train subsample
        2. Full RTS smoother on entire series
        3. Return smoothed [level, slope, accel, jerk]
    """
    y_avail = rsi_full if train_end_idx is None else rsi_full[:train_end_idx]
    if verbose:
        print(f"  [GT 4D MLE] fitting (σ²_jerk, R) on {min(len(y_avail), subsample_n):,} samples...")

    mle = mle_fit_4d(
        y_avail,
        init_sigma2=init_sigma2,
        init_r=init_r,
        subsample_n=subsample_n,
        warmup=warmup,
        verbose=False,
    )
    if verbose:
        print(f"  [GT 4D MLE] σ²_jerk={mle.sigma2_drive:.6g}  R={mle.r_scalar:.4f}")
        print(f"  [GT 4D MLE] NLL={mle.nll:.2f}  success={mle.success}  iters={mle.n_iter}  evals={mle.n_eval}")
        print(f"  [GT 4D RTS] running full-pass smoother on {len(rsi_full):,} samples...")

    x_smooth = rts_smoother_4d(rsi_full, mle.sigma2_drive, mle.r_scalar)

    return OfficialGT4D(
        level=x_smooth[:, 0],
        slope=x_smooth[:, 1],
        accel=x_smooth[:, 2],
        jerk=x_smooth[:, 3],
        sigma2_jerk=mle.sigma2_drive,
        r_scalar=mle.r_scalar,
        nll=mle.nll,
        n_fit_samples=mle.n_samples_used,
        n_full_samples=len(rsi_full),
    )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 3000
    # Synthetic signal with actual jerk dynamics
    true_s2_jerk = 1e-5
    true_r = 3.0
    x = np.zeros((n, 4))
    x[0] = [50.0, 0.0, 0.0, 0.0]
    for t in range(1, n):
        w = rng.standard_normal() * np.sqrt(true_s2_jerk)
        x[t] = F_4 @ x[t - 1] + G_4.ravel() * w
    y = x[:, 0] + rng.standard_normal(n) * np.sqrt(true_r)

    gt = compute_official_gt_4d(y, subsample_n=n, init_sigma2=1e-6, init_r=1.0, verbose=True)
    print(f"True σ²_jerk={true_s2_jerk}, R={true_r}")
    print(f"RTS recovers slope with Pearson = {np.corrcoef(x[50:-50, 1], gt.slope[50:-50])[0, 1]:.4f}")
