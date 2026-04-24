"""
Estimation empirique de R — 4 méthodes complémentaires.

Utilisateur :
    > "Avant de fixer R = 3.0, documenter explicitement :
    >   - Méthode d'estimation utilisée (RSI - MA(k), quel k ?)
    >   - Sensibilité au choix de méthode
    >   - Reporter les 4 estimations, choisir la médiane ou justifier le choix"

Méthodes :
    M1. var(RSI[t] - MA5[t])   — résidu / moyenne mobile centrée width=5
    M2. var(RSI[t] - MA11[t])  — résidu / moyenne mobile centrée width=11
        (l'utilisateur a dit "MA10" mais la centered MA requiert width impair ;
        width=11 est le plus proche symétrique)
    M3. var(RSI[t] - RSI[t-1]) / 2  — estimateur différence première
        (classique pour y = x_smooth + ε : E[(Δy)²] = 2 E[ε²])
    M4. MLE (Q, R) sur un sous-échantillon train avec un modèle 2D CV
        — intègre la structure state-space, lit R comme le paramètre qui
        maximise la vraisemblance des innovations.

Choix retenu : MLE si son résultat est dans ±30% de la médiane des 3
autres (sanity), sinon médiane des 3 moving-stats (moins sujet aux
optimums locaux). Le critère est documenté dans l'output.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy import optimize

# Reuse project constants.
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from constants import KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR  # noqa: E402


# ---------------------------------------------------------------------------
# Moving-average based estimates (M1, M2)
# ---------------------------------------------------------------------------

def _centered_ma(x: np.ndarray, window: int) -> np.ndarray:
    """Centered MA of odd width. Edges -> NaN."""
    if window % 2 == 0:
        raise ValueError("window must be odd for a centered MA")
    n = len(x)
    half = window // 2
    out = np.full(n, np.nan)
    # Use a cumulative-sum trick for speed on large arrays
    finite = np.isfinite(x)
    xc = np.where(finite, x, 0.0)
    count = finite.astype(int)
    csum = np.concatenate([[0.0], np.cumsum(xc)])
    ccount = np.concatenate([[0], np.cumsum(count)])
    for t in range(half, n - half):
        s = csum[t + half + 1] - csum[t - half]
        c = ccount[t + half + 1] - ccount[t - half]
        if c == window:
            out[t] = s / c
    return out


def r_via_ma(rsi: np.ndarray, window: int) -> Dict[str, float]:
    """R estimate from residual variance against a centered MA."""
    ma = _centered_ma(rsi, window)
    residual = rsi - ma
    finite = np.isfinite(residual)
    if finite.sum() < 100:
        return {"method": f"MA{window}", "R": float("nan"), "n": int(finite.sum())}
    r = residual[finite]
    # Both variance (sensitive to outliers) and MAD (robust) for cross-check
    var = float(np.var(r, ddof=0))
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    mad_std2 = float((1.4826 * mad) ** 2)
    return {
        "method": f"MA{window}",
        "R": var,
        "R_mad": mad_std2,
        "n": int(finite.sum()),
        "residual_mean": float(np.mean(r)),
        "residual_skew": float(((r - np.mean(r)) ** 3).mean() / (np.std(r) ** 3 + 1e-30)),
    }


# ---------------------------------------------------------------------------
# First-difference estimator (M3)
# ---------------------------------------------------------------------------

def r_via_first_diff(rsi: np.ndarray) -> Dict[str, float]:
    """
    Classical estimator : R ≈ Var(Δy) / 2.

    Assumption : y[t] = x_smooth[t] + ε[t], with x_smooth changing slowly
    relative to ε's noise scale. Then Δy ≈ Δε, so Var(Δy) ≈ 2 Var(ε).

    BIASED UPWARD if the true signal has non-zero local slope variance
    (that fraction of the diff gets attributed to noise). We report
    anyway for comparison.
    """
    d = np.diff(rsi)
    d = d[np.isfinite(d)]
    if len(d) < 10:
        return {"method": "FirstDiff", "R": float("nan"), "n": 0}
    var_d = float(np.var(d, ddof=0))
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    mad_std2 = float((1.4826 * mad) ** 2)
    return {
        "method": "FirstDiff",
        "R": var_d / 2.0,
        "R_mad": mad_std2 / 2.0,
        "n": int(len(d)),
        "raw_variance": var_d,
    }


# ---------------------------------------------------------------------------
# MLE estimate (M4) on a 2D CV model
# ---------------------------------------------------------------------------

def _forward_2d_nll(
    y: np.ndarray,
    sigma2_proc: float,
    r_scalar: float,
    warmup: int = 50,
) -> float:
    """Quick forward KF + NLL for 2D CV model (Q = σ²·I₂, diagonal)."""
    n = len(y)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * sigma2_proc

    x = np.array([y[0], 0.0])
    P = np.eye(2) * 100.0  # diffuse init

    ll = 0.0
    count = 0
    for t in range(1, n):
        x_p = F @ x
        P_p = F @ P @ F.T + Q
        if not np.isfinite(y[t]):
            x = x_p
            P = P_p
            continue
        y_hat = float((H @ x_p)[0])
        v = float(y[t] - y_hat)
        S = float((H @ P_p @ H.T)[0, 0] + r_scalar)
        if S <= 0:
            return float("inf")
        if t > warmup:
            ll += -0.5 * (np.log(2 * np.pi * S) + v * v / S)
            count += 1
        K = (P_p @ H.T / S).ravel()
        x = x_p + K * v
        P = (np.eye(2) - np.outer(K, H.ravel())) @ P_p
    if count < 10:
        return float("inf")
    return -ll


def r_via_mle_2d(
    rsi: np.ndarray,
    n_subsample: int = 20_000,
    init_sigma2: float = KALMAN_PROCESS_VAR,
    init_r: float = 1.0,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Joint MLE of (σ², R) on a 2D CV model, over `n_subsample` train bars.

    Uses Nelder-Mead in log-space. Returns R (primary) + σ² (secondary).
    """
    y = rsi[:n_subsample] if len(rsi) > n_subsample else rsi

    def _obj(theta):
        s2 = float(np.exp(theta[0]))
        r = float(np.exp(theta[1]))
        return _forward_2d_nll(y, s2, r)

    x0 = np.array([np.log(init_sigma2), np.log(init_r)])
    res = optimize.minimize(
        _obj, x0, method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-2, "maxiter": 200, "disp": verbose},
    )
    return {
        "method": "MLE_2D",
        "R": float(np.exp(res.x[1])),
        "sigma2_proc": float(np.exp(res.x[0])),
        "nll": float(res.fun),
        "success": bool(res.success),
        "n_iter": int(res.nit),
        "n_eval": int(res.nfev),
        "n": int(len(y)),
    }


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

@dataclass
class REstimation:
    m1_ma5: Dict[str, float]
    m2_ma11: Dict[str, float]
    m3_firstdiff: Dict[str, float]
    m4_mle: Dict[str, float]
    r_chosen: float
    chosen_method: str
    reasoning: str


def estimate_R_multimethod(
    rsi_full: np.ndarray,
    train_end_idx: Optional[int] = None,
    mle_subsample: int = 20_000,
) -> REstimation:
    """
    Compute all 4 R estimates, choose the one to use in the baseline.

    Selection rule :
      - compute median of M1, M2, M3 (the three moving-stats estimators)
      - if M4 (MLE) is within ±30% of the median → trust MLE
        (model-aware, cleanest theoretically)
      - otherwise → use median (robust consensus)
    """
    y_train = rsi_full if train_end_idx is None else rsi_full[:train_end_idx]

    m1 = r_via_ma(y_train, window=5)
    m2 = r_via_ma(y_train, window=11)
    m3 = r_via_first_diff(y_train)
    m4 = r_via_mle_2d(y_train, n_subsample=mle_subsample)

    # Build a consensus from the three stats-based estimators
    r_values = [m1["R"], m2["R"], m3["R"]]
    r_values = [v for v in r_values if np.isfinite(v)]
    r_median = float(np.median(r_values)) if r_values else float("nan")

    r_mle = m4["R"]
    if np.isfinite(r_mle) and m4.get("success", False) and r_median > 0:
        ratio = r_mle / r_median
        if 0.70 <= ratio <= 1.30:
            chosen = "MLE_2D"
            r_out = r_mle
            reason = (
                f"MLE R={r_mle:.4f} est cohérent avec la médiane stats {r_median:.4f} "
                f"(ratio {ratio:.2f} ∈ [0.70, 1.30]). MLE retenu car model-aware."
            )
        else:
            chosen = "Median_M1_M2_M3"
            r_out = r_median
            reason = (
                f"MLE R={r_mle:.4f} diverge de la médiane stats {r_median:.4f} "
                f"(ratio {ratio:.2f}). Fallback : médiane M1/M2/M3."
            )
    else:
        chosen = "Median_M1_M2_M3"
        r_out = r_median
        reason = (
            f"MLE peu fiable (success={m4.get('success', False)} ou NaN). "
            f"Fallback : médiane M1/M2/M3 = {r_median:.4f}."
        )

    return REstimation(
        m1_ma5=m1,
        m2_ma11=m2,
        m3_firstdiff=m3,
        m4_mle=m4,
        r_chosen=float(r_out),
        chosen_method=chosen,
        reasoning=reason,
    )


def format_estimation_table(est: REstimation) -> str:
    """Pretty print of the 4 estimators + verdict."""
    lines = []
    lines.append("\n" + "=" * 78)
    lines.append("ESTIMATION EMPIRIQUE DE R — 4 MÉTHODES")
    lines.append("=" * 78)
    hdr = f"{'méthode':<12s} {'R (var)':>10s} {'R (MAD²)':>10s} {'n':>10s}  {'détail':<30s}"
    lines.append(hdr)
    lines.append("-" * 78)
    m1 = est.m1_ma5
    lines.append(
        f"{m1['method']:<12s} {m1['R']:>10.4f} {m1.get('R_mad', float('nan')):>10.4f} "
        f"{m1['n']:>10,d}  residu_skew={m1.get('residual_skew', 0):.2f}"
    )
    m2 = est.m2_ma11
    lines.append(
        f"{m2['method']:<12s} {m2['R']:>10.4f} {m2.get('R_mad', float('nan')):>10.4f} "
        f"{m2['n']:>10,d}  residu_skew={m2.get('residual_skew', 0):.2f}"
    )
    m3 = est.m3_firstdiff
    lines.append(
        f"{m3['method']:<12s} {m3['R']:>10.4f} {m3.get('R_mad', float('nan')):>10.4f} "
        f"{m3['n']:>10,d}  raw_var={m3.get('raw_variance', 0):.3f}"
    )
    m4 = est.m4_mle
    nll_str = f"NLL={m4.get('nll', float('nan')):.1f}" if np.isfinite(m4.get('nll', float('nan'))) else "NLL=NaN"
    lines.append(
        f"{m4['method']:<12s} {m4['R']:>10.4f} {'—':>10s} "
        f"{m4['n']:>10,d}  σ²={m4.get('sigma2_proc', 0):.5g}  {nll_str}  succ={m4.get('success', False)}"
    )
    lines.append("-" * 78)
    lines.append(f"R retenu    : {est.r_chosen:.4f}  (méthode: {est.chosen_method})")
    lines.append(f"Raison      : {est.reasoning}")
    lines.append("=" * 78)
    return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test on synthetic RSI-like series with known noise
    rng = np.random.default_rng(0)
    n = 30_000
    true_r = 2.5
    # Slow drifting smooth underlying signal
    smooth = 50.0 + 20.0 * np.sin(np.linspace(0, 30, n)) + 5.0 * np.cos(np.linspace(0, 80, n))
    y = smooth + rng.standard_normal(n) * np.sqrt(true_r)
    est = estimate_R_multimethod(y, train_end_idx=n, mle_subsample=15_000)
    print(format_estimation_table(est))
    print(f"\nTRUE R = {true_r:.3f}")
    print(f"RETAINED R = {est.r_chosen:.3f}")
