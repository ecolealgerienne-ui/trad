"""
Innovation diagnostics for Kalman filter residuals.

All tests run on normalized innovations z_t = v_t / sqrt(S_t), which should
be ~ iid N(0, 1) under correct model specification.

Ljung-Box is implemented from scratch (no statsmodels dependency).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Core tests
# ---------------------------------------------------------------------------

def acf(x: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """
    Autocorrelation function of x at lags 0..max_lag.

    Uses the biased (divide-by-n) estimator, matching statsmodels default.
    Returns array of shape (max_lag + 1,) with acf[0] = 1.0.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < max_lag + 2:
        raise ValueError(f"Série trop courte: n={n}, max_lag={max_lag}")

    x = x - np.mean(x)
    var = np.mean(x ** 2)
    if var == 0:
        return np.zeros(max_lag + 1)

    out = np.empty(max_lag + 1)
    out[0] = 1.0
    for k in range(1, max_lag + 1):
        out[k] = np.mean(x[:-k] * x[k:]) / var
    return out


def ljung_box(x: np.ndarray, lags: int = 10) -> Dict[str, float]:
    """
    Ljung-Box test for autocorrelation at lags 1..lags.

        Q = n*(n+2) * sum_{k=1..h} rho_k^2 / (n - k)

    Under H0 of no autocorrelation, Q ~ chi2(h).

    Returns
    -------
    dict with keys:
        - statistic : Q
        - p_value   : P(chi2(lags) > Q)
        - lags      : int
        - n         : sample size used
        - q_per_n   : Q / n  (size-normalized statistic for large-N comparability)
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < lags + 10:
        raise ValueError(f"n={n} trop petit pour lags={lags}")

    rhos = acf(x, max_lag=lags)[1:]  # drop rho_0
    q = n * (n + 2) * np.sum(rhos ** 2 / (n - np.arange(1, lags + 1)))
    p = 1.0 - stats.chi2.cdf(q, df=lags)
    return {
        "statistic": float(q),
        "p_value": float(p),
        "lags": int(lags),
        "n": int(n),
        "q_per_n": float(q / n),
    }


def jarque_bera(x: np.ndarray) -> Dict[str, float]:
    """
    Jarque-Bera test for normality.

    Returns statistic, p_value, skewness, excess_kurtosis.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    jb, p = stats.jarque_bera(x)
    skew = float(stats.skew(x))
    kurt = float(stats.kurtosis(x))  # excess kurtosis (Fisher's definition)
    return {
        "statistic": float(jb),
        "p_value": float(p),
        "skewness": skew,
        "excess_kurtosis": kurt,
        "n": int(len(x)),
    }


# ---------------------------------------------------------------------------
# Full diagnostic report
# ---------------------------------------------------------------------------

@dataclass
class InnovationDiagnostic:
    """Container for innovation diagnostic results on one split."""
    split_name: str
    n: int
    mean: float
    std: float
    acf_1_to_10: List[float]
    acf_max_abs_1_10: float
    ljung_box_h10: Dict[str, float]
    ljung_box_h20: Dict[str, float]
    jarque_bera: Dict[str, float]
    gate_verdict: str  # "EXPLOITABLE", "MARGINAL", "WHITE_NOISE"
    gate_reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _gate_decision(max_acf: float, lb_pvalue: float) -> tuple[str, str]:
    """
    Étape 1 gate logic (user-calibrated for large N).

    Decision rules:
      - max(|ACF(1..10)|) > 0.05 AND LB p-value < 0.05
          => EXPLOITABLE  (structure présente, Étape 2 justifiée)
      - max(|ACF(1..10)|) in [0.02, 0.05] AND LB p-value < 0.05
          => MARGINAL     (flag & attendre confirmation)
      - max(|ACF(1..10)|) < 0.02
          => WHITE_NOISE  (skip Étape 2 quelle que soit la p-value)
    """
    if max_acf < 0.02:
        return "WHITE_NOISE", f"max|ACF(1..10)|={max_acf:.4f} < 0.02 → bruit blanc pratique"
    if max_acf > 0.05 and lb_pvalue < 0.05:
        return "EXPLOITABLE", (
            f"max|ACF(1..10)|={max_acf:.4f} > 0.05 ET LB p={lb_pvalue:.3e} < 0.05 → "
            "structure exploitable, Étape 2 justifiée"
        )
    if 0.02 <= max_acf <= 0.05 and lb_pvalue < 0.05:
        return "MARGINAL", (
            f"max|ACF(1..10)|={max_acf:.4f} ∈ [0.02, 0.05] ET LB p={lb_pvalue:.3e} → "
            "marginal, confirmation requise avant Étape 2"
        )
    # Remaining case: max_acf >= 0.02 but LB p >= 0.05 (unlikely on large N)
    return "MARGINAL", (
        f"max|ACF(1..10)|={max_acf:.4f}, LB p={lb_pvalue:.3e} — "
        "situation ambiguë, flag & confirmer"
    )


def run_diagnostic(
    innovations: np.ndarray,
    S: np.ndarray,
    split_name: str = "train",
) -> InnovationDiagnostic:
    """
    Full diagnostic on normalized innovations z_t = v_t / sqrt(S_t).

    Parameters
    ----------
    innovations : array of shape (n,)
        Raw innovations v_t (NaN allowed, will be dropped).
    S : array of shape (n,)
        Innovation variance estimates S_t = H P_{t|t-1} H^T + R.
    split_name : str
        Label ("train" / "val" / "test") for reporting.
    """
    innovations = np.asarray(innovations, dtype=float)
    S = np.asarray(S, dtype=float)
    mask = np.isfinite(innovations) & np.isfinite(S) & (S > 0)
    if mask.sum() < 100:
        raise ValueError(f"Innovations valides insuffisantes: {mask.sum()}")

    z = innovations[mask] / np.sqrt(S[mask])

    # ACF up to lag 50, we report 1..10 for the gate
    acf_vals = acf(z, max_lag=50)
    acf_1_10 = acf_vals[1:11].tolist()
    max_abs_1_10 = float(np.max(np.abs(acf_vals[1:11])))

    lb10 = ljung_box(z, lags=10)
    lb20 = ljung_box(z, lags=20)
    jb = jarque_bera(z)

    verdict, reason = _gate_decision(max_abs_1_10, lb10["p_value"])

    return InnovationDiagnostic(
        split_name=split_name,
        n=int(mask.sum()),
        mean=float(np.mean(z)),
        std=float(np.std(z)),
        acf_1_to_10=[float(v) for v in acf_1_10],
        acf_max_abs_1_10=max_abs_1_10,
        ljung_box_h10=lb10,
        ljung_box_h20=lb20,
        jarque_bera=jb,
        gate_verdict=verdict,
        gate_reason=reason,
    )


# ---------------------------------------------------------------------------
# Optional plotting (skipped silently if matplotlib unavailable)
# ---------------------------------------------------------------------------

def make_plots(
    innovations: np.ndarray,
    S: np.ndarray,
    out_dir: str | Path,
    prefix: str = "baseline",
) -> Optional[List[Path]]:
    """
    Generate diagnostic plots:
      - histogram of z_t vs N(0,1)
      - ACF of z_t (lags 1..50) with ±1.96/sqrt(n) bands
      - QQ plot of z_t vs normal

    Returns list of saved file paths, or None if matplotlib is missing.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask = np.isfinite(innovations) & np.isfinite(S) & (S > 0)
    z = innovations[mask] / np.sqrt(S[mask])
    n = len(z)

    saved: List[Path] = []

    # Histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(z, bins=80, density=True, alpha=0.65, label=f"z_t (n={n})")
    xs = np.linspace(-5, 5, 500)
    ax.plot(xs, stats.norm.pdf(xs), "r-", lw=1.5, label="N(0,1)")
    ax.set_xlim(-5, 5)
    ax.set_xlabel("Innovation normalisée z_t")
    ax.set_ylabel("Densité")
    ax.set_title(f"Histogramme des innovations normalisées ({prefix})")
    ax.legend()
    path = out_dir / f"{prefix}_innov_hist.png"
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    saved.append(path)

    # ACF
    acf_vals = acf(z, max_lag=50)
    ci = 1.96 / np.sqrt(n)
    fig, ax = plt.subplots(figsize=(9, 4))
    lags = np.arange(0, 51)
    ax.vlines(lags[1:], 0, acf_vals[1:], colors="steelblue", lw=1.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(ci, color="red", ls="--", lw=0.8, label=f"±1.96/√n = ±{ci:.4f}")
    ax.axhline(-ci, color="red", ls="--", lw=0.8)
    ax.set_xlim(0, 51)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.set_title(f"ACF des innovations normalisées ({prefix})")
    ax.legend()
    path = out_dir / f"{prefix}_innov_acf.png"
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    saved.append(path)

    # QQ plot
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    stats.probplot(z, dist="norm", plot=ax)
    ax.set_title(f"QQ-plot vs Normal ({prefix})")
    path = out_dir / f"{prefix}_innov_qq.png"
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    saved.append(path)

    return saved


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 50_000
    # White noise test
    z = rng.standard_normal(n)
    S = np.ones(n)
    d = run_diagnostic(z, S, split_name="smoke_white")
    print(f"White noise: verdict={d.gate_verdict} max|ACF|={d.acf_max_abs_1_10:.4f} LB p={d.ljung_box_h10['p_value']:.4e}")

    # AR(1) test
    phi = 0.3
    x = np.zeros(n)
    eps = rng.standard_normal(n)
    for i in range(1, n):
        x[i] = phi * x[i-1] + eps[i]
    d2 = run_diagnostic(x, np.ones(n), split_name="smoke_ar1")
    print(f"AR(1) phi={phi}: verdict={d2.gate_verdict} max|ACF|={d2.acf_max_abs_1_10:.4f} LB p={d2.ljung_box_h10['p_value']:.4e}")
