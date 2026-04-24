"""
Ground-truth slope references for evaluation only.

PRIMARY: RTS (Rauch-Tung-Striebel) full-pass non-causal smoother, run on the
         RSI signal through a fixed-Q 2D Kalman model (same structure as the
         project's prod AQ-KF). Produces slope_truth_rts = smoothed state[:, 1].
         NON-CAUSAL. Used as reference, never fed back as input.

SECONDARY: Centered moving-average of first differences of RSI, window=21.
           Simpler sanity check that doesn't depend on the KF model.

REUSE:
    - pykalman.KalmanFilter (same library used by src/filters.py and
      src/prepare_multitf_csv_aqkf.py:132). Using it directly here — not
      through src/filters.py:kalman_filter because that wrapper is 1D-state
      and doesn't expose the velocity component we need.
    - src.constants.KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR

Model is identical to the one in src/prepare_multitf_csv_aqkf.py:kalman_filter_standard:
    F = [[1, 1], [0, 1]],  H = [[1, 0]],  Q = I*q_scalar,  R = r_scalar

IMPORTANT: ground truth is computed ONCE on the FULL filtered series, THEN
split. See user constraint: "Ne PAS recalculer le RTS ground truth par fold".
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Reuse project constants without modifying anything.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from constants import KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR  # noqa: E402


# ---------------------------------------------------------------------------
# Primary: RTS full-pass smoother on 2D CV model
# ---------------------------------------------------------------------------

def _rts_via_pykalman(
    y: np.ndarray,
    q_scalar: float,
    r_scalar: float,
) -> np.ndarray:
    """RTS smoother via pykalman (preferred when available)."""
    from pykalman import KalmanFilter  # imported lazily so fallback works
    # Same configuration as src/prepare_multitf_csv_aqkf.py:kalman_filter_standard
    kf = KalmanFilter(
        transition_matrices=[[1.0, 1.0], [0.0, 1.0]],
        observation_matrices=[[1.0, 0.0]],
        transition_covariance=np.eye(2) * q_scalar,
        observation_covariance=[[r_scalar]],
        initial_state_mean=[y[0], 0.0],
        initial_state_covariance=np.eye(2),
    )
    # pykalman .smooth() runs forward filter + RTS backward pass
    smoothed_means, _ = kf.smooth(y.reshape(-1, 1))
    return np.asarray(smoothed_means)


def _rts_hand_rolled(
    y: np.ndarray,
    q_scalar: float,
    r_scalar: float,
) -> np.ndarray:
    """
    Hand-rolled RTS smoother (fallback if pykalman unavailable).

    Standard two-pass algorithm (Rauch-Tung-Striebel 1965).
    Identical model to _rts_via_pykalman above.
    """
    n = len(y)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * q_scalar
    R = np.array([[r_scalar]])

    x_filt = np.zeros((n, 2))
    P_filt = np.zeros((n, 2, 2))

    # First finite observation
    start = 0
    while start < n and not np.isfinite(y[start]):
        start += 1
    x_filt[start] = np.array([y[start], 0.0])
    P_filt[start] = np.eye(2)

    # Forward pass
    for t in range(start + 1, n):
        x_p = F @ x_filt[t - 1]
        P_p = F @ P_filt[t - 1] @ F.T + Q
        if not np.isfinite(y[t]):
            x_filt[t] = x_p
            P_filt[t] = P_p
            continue
        S = (H @ P_p @ H.T + R)[0, 0]
        K = (P_p @ H.T / S).ravel()
        x_filt[t] = x_p + K * (y[t] - (H @ x_p)[0])
        P_filt[t] = (np.eye(2) - np.outer(K, H.ravel())) @ P_p

    # Backward pass (RTS)
    x_smooth = x_filt.copy()
    for t in range(n - 2, start - 1, -1):
        P_pred_next = F @ P_filt[t] @ F.T + Q
        try:
            C = P_filt[t] @ F.T @ np.linalg.inv(P_pred_next)
        except np.linalg.LinAlgError:
            C = P_filt[t] @ F.T @ np.linalg.pinv(P_pred_next)
        x_smooth[t] = x_filt[t] + C @ (x_smooth[t + 1] - F @ x_filt[t])

    return x_smooth


def rts_smoother_2d(
    y: np.ndarray,
    q_scalar: float = KALMAN_PROCESS_VAR,
    r_scalar: float = KALMAN_MEASURE_VAR,
) -> np.ndarray:
    """
    RTS full-pass non-causal smoother on 2D CV model.

    Uses pykalman if available (same library as project), falls back to a
    hand-rolled equivalent otherwise. Both paths implement identical model.

    Returns
    -------
    x_smooth : (n, 2)
        Smoothed state [level, velocity]. Column 1 is the ground-truth slope.
    """
    y = np.asarray(y, dtype=float)
    try:
        return _rts_via_pykalman(y, q_scalar, r_scalar)
    except ImportError:
        return _rts_hand_rolled(y, q_scalar, r_scalar)


# ---------------------------------------------------------------------------
# Secondary: centered moving average of first differences
# ---------------------------------------------------------------------------

def slope_centered_ma(rsi: np.ndarray, window: int = 21) -> np.ndarray:
    """
    Centered moving average of RSI first differences, window=21 (= lags -10..+10).

    NON-CAUSAL by construction. Reference only. Edge points with insufficient
    lookaround yield NaN.
    """
    if window % 2 == 0:
        raise ValueError("window must be odd for a centered MA")
    n = len(rsi)
    half = window // 2

    diff = np.full(n, np.nan)
    diff[1:] = np.diff(rsi)

    out = np.full(n, np.nan)
    # Use a vectorized cumulative-sum trick on finite windows
    for t in range(half, n - half):
        w = diff[t - half : t + half + 1]
        if np.all(np.isfinite(w)):
            out[t] = np.mean(w)
    return out


# ---------------------------------------------------------------------------
# Bundled API
# ---------------------------------------------------------------------------

@dataclass
class GroundTruth:
    """Container holding both ground-truth references aligned with RSI."""
    slope_rts: np.ndarray       # PRIMARY  (n,)
    slope_ma: np.ndarray        # SECONDARY (n,)
    level_rts: np.ndarray       # Smoothed level (diagnostics only)

    def split(self, idx_start: int, idx_end: int) -> "GroundTruth":
        return GroundTruth(
            slope_rts=self.slope_rts[idx_start:idx_end].copy(),
            slope_ma=self.slope_ma[idx_start:idx_end].copy(),
            level_rts=self.level_rts[idx_start:idx_end].copy(),
        )


def compute_full_ground_truth(
    rsi_full: np.ndarray,
    q_scalar: float = KALMAN_PROCESS_VAR,
    r_scalar: float = KALMAN_MEASURE_VAR,
    ma_window: int = 21,
) -> GroundTruth:
    """Compute both ground-truth references on the FULL series in one pass."""
    x_smooth = rts_smoother_2d(rsi_full, q_scalar=q_scalar, r_scalar=r_scalar)
    slope_rts = x_smooth[:, 1]
    level_rts = x_smooth[:, 0]
    slope_ma = slope_centered_ma(rsi_full, window=ma_window)
    return GroundTruth(slope_rts=slope_rts, slope_ma=slope_ma, level_rts=level_rts)


if __name__ == "__main__":
    # Smoke test: synthetic piecewise-linear trend
    rng = np.random.default_rng(0)
    n = 5000
    t = np.arange(n)
    true_vel = np.where(t < n // 2, 0.02, -0.01)
    level_true = np.cumsum(true_vel)
    y = level_true + rng.standard_normal(n) * 0.5
    gt = compute_full_ground_truth(y, q_scalar=1e-4, r_scalar=0.25)
    # Trim edges for fair correlation (centered MA has NaN edges)
    m = np.isfinite(gt.slope_rts) & np.isfinite(gt.slope_ma)
    corr_rts = np.corrcoef(true_vel[m], gt.slope_rts[m])[0, 1]
    corr_ma = np.corrcoef(true_vel[m], gt.slope_ma[m])[0, 1]
    print(f"Pearson(true_vel, RTS) = {corr_rts:.4f}")
    print(f"Pearson(true_vel, MA)  = {corr_ma:.4f}")
