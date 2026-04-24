"""
Estimation metrics for slope evaluation.

All metrics are aligned pointwise: est[t] vs truth[t]. NaN entries on either
side are excluded from the computation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
from scipy import stats


@dataclass
class SlopeMetrics:
    mse: float
    mae: float
    pearson: float
    direction_match: float  # fraction in [0, 1] where sign agrees
    latency_bars: float     # cross-correlation argmax lag of est vs truth
    n_valid: int

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _valid_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & np.isfinite(b)


def mse(est: np.ndarray, truth: np.ndarray) -> float:
    m = _valid_mask(est, truth)
    if not m.any():
        return float("nan")
    return float(np.mean((est[m] - truth[m]) ** 2))


def mae(est: np.ndarray, truth: np.ndarray) -> float:
    m = _valid_mask(est, truth)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(est[m] - truth[m])))


def pearson(est: np.ndarray, truth: np.ndarray) -> float:
    m = _valid_mask(est, truth)
    if m.sum() < 3:
        return float("nan")
    if np.std(est[m]) == 0 or np.std(truth[m]) == 0:
        return float("nan")
    r, _ = stats.pearsonr(est[m], truth[m])
    return float(r)


def direction_match(est: np.ndarray, truth: np.ndarray, zero_tol: float = 0.0) -> float:
    """
    Fraction of samples where sign(est) == sign(truth).

    Samples where |truth| <= zero_tol are excluded (ambiguous sign).
    """
    m = _valid_mask(est, truth)
    if zero_tol > 0:
        m = m & (np.abs(truth) > zero_tol)
    if not m.any():
        return float("nan")
    s_est = np.sign(est[m])
    s_truth = np.sign(truth[m])
    # Treat zero-slope estimates as neutral (no match, no mismatch) by excluding
    neutral = s_est == 0
    if neutral.all():
        return float("nan")
    s_est = s_est[~neutral]
    s_truth = s_truth[~neutral]
    return float(np.mean(s_est == s_truth))


def cross_correlation_lag(
    est: np.ndarray,
    truth: np.ndarray,
    max_lag: int = 20,
) -> float:
    """
    Estimate the lag (in bars) between est and truth by cross-correlation.

    Returns the lag l (positive => est lags truth, negative => est leads truth)
    that maximizes Pearson correlation on the overlap.

    This is a coarse estimate in [-max_lag, +max_lag].
    """
    m = _valid_mask(est, truth)
    if m.sum() < 100:
        return float("nan")
    e = est[m] - np.mean(est[m])
    t = truth[m] - np.mean(truth[m])
    e_std = np.std(e)
    t_std = np.std(t)
    if e_std == 0 or t_std == 0:
        return float("nan")
    best_lag = 0
    best_r = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a = e[lag:]
            b = t[: len(e) - lag]
        else:
            a = e[: len(e) + lag]
            b = t[-lag:]
        if len(a) < 100:
            continue
        r = float(np.mean(a * b) / (np.std(a) * np.std(b) + 1e-30))
        if r > best_r:
            best_r = r
            best_lag = lag
    return float(best_lag)


def compute_all(
    est: np.ndarray,
    truth: np.ndarray,
    *,
    declared_latency: int = 0,
    direction_zero_tol: float = 0.0,
    max_lag_scan: int = 20,
) -> SlopeMetrics:
    """
    Compute the 5 project metrics on a single split.

    `declared_latency` is the design-time latency (e.g., FLKS lag). If > 0,
    the effective metric `latency_bars` returned is max(declared, measured).
    """
    measured_lag = cross_correlation_lag(est, truth, max_lag=max_lag_scan)
    m = _valid_mask(est, truth)
    return SlopeMetrics(
        mse=mse(est, truth),
        mae=mae(est, truth),
        pearson=pearson(est, truth),
        direction_match=direction_match(est, truth, zero_tol=direction_zero_tol),
        latency_bars=float(max(declared_latency, measured_lag)) if np.isfinite(measured_lag) else float(declared_latency),
        n_valid=int(m.sum()),
    )


# ---------------------------------------------------------------------------
# Diebold-Mariano test (for comparing two forecasts)
# ---------------------------------------------------------------------------

def diebold_mariano(
    est1: np.ndarray,
    est2: np.ndarray,
    truth: np.ndarray,
    loss: str = "mse",
    h: int = 1,
) -> Tuple[float, float]:
    """
    Diebold-Mariano test on squared (or absolute) loss differentials.

    H0: est1 and est2 have equal predictive accuracy.
    H1: est1 and est2 differ in predictive accuracy.

    Returns
    -------
    dm_stat : float
        Test statistic. Negative => est1 has lower loss (better).
    p_value : float
        Two-sided p-value from N(0, 1) asymptotic distribution.

    Parameters
    ----------
    h : int
        Forecast horizon. For 1-step, h=1. We use Newey-West HAC with
        truncation lag h-1 (i.e. no autocorrelation adjustment for h=1).
    """
    m = _valid_mask(est1, truth) & _valid_mask(est2, truth)
    if m.sum() < 50:
        return float("nan"), float("nan")
    e1 = est1[m] - truth[m]
    e2 = est2[m] - truth[m]

    if loss == "mse":
        d = e1 ** 2 - e2 ** 2
    elif loss == "mae":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    n = len(d)
    d_mean = np.mean(d)

    # Long-run variance estimator (Newey-West style), truncation = h-1
    gamma0 = np.var(d, ddof=0)
    lrv = gamma0
    for k in range(1, h):
        cov_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
        lrv += 2.0 * (1.0 - k / h) * cov_k
    if lrv <= 0 or not np.isfinite(lrv):
        return float("nan"), float("nan")

    dm_stat = d_mean / np.sqrt(lrv / n)
    # Two-sided p-value
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


if __name__ == "__main__":
    # Smoke test
    rng = np.random.default_rng(0)
    truth = rng.standard_normal(1000)
    est1 = truth + 0.1 * rng.standard_normal(1000)
    est2 = truth + 0.3 * rng.standard_normal(1000)
    m1 = compute_all(est1, truth)
    m2 = compute_all(est2, truth)
    print(f"est1 MSE={m1.mse:.4f}  Pearson={m1.pearson:.4f}  DirMatch={m1.direction_match:.4f}")
    print(f"est2 MSE={m2.mse:.4f}  Pearson={m2.pearson:.4f}  DirMatch={m2.direction_match:.4f}")
    dm, p = diebold_mariano(est1, est2, truth, loss="mse")
    print(f"DM stat={dm:.4f} p={p:.4e}  (negative => est1 better)")
