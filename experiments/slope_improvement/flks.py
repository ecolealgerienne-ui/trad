"""
Fixed-Lag Smoother (FLKS) — générique N-dimensionnel, version vectorisée.

Calcule pour un lag L fixé :

    x_s[t | t+L] = E[x_t | y_{0:t+L}]

Lag 0 = forward filter pur. Lag ≥ n = RTS full-pass équivalent.

Implémentation vectorisée :
  1. Forward pass unique → x_filt[t], P_filt[t] pour tout t
  2. Précalcul des gains RTS C[k] = P_filt[k] · F^T · (F P_filt[k] F^T + Q)^(-1)
     Indépendants de t, calculés une fois.
  3. Backward pass en L étapes vectorisées sur TOUS les t simultanément
     À l'étape i, tous les t ont avancé i pas en arrière depuis leur
     point de départ t+L. Avec numpy einsum, chaque étape traite ~n points
     en une op matricielle, soit ~5 ms par étape.
  4. Edge handling pour t > n-1-L : lag effectif raccourci (gracefully).

Complexité totale : O(n·L) opérations mais latence réelle ~L·3ms grâce à
la vectorisation. Grid lag∈{0,1,2,3,5,8,13,21,50,200} : ~1-2 sec total.

Réutilise kf_nd.forward_filter et kf_nd.rts_smoother.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from kf_nd import forward_filter, rts_smoother


@dataclass
class FLKSResult:
    x_smoothed: np.ndarray        # (n, N)
    residuals: np.ndarray         # (n,) y[t] - H @ x_s[t]
    effective_lag: np.ndarray     # (n,) min(L, n-1-t)


def _precompute_C(
    P_filt: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
) -> np.ndarray:
    """
    Précalcule C[k] = P_filt[k] · F^T · (F P_filt[k] F^T + Q)^(-1) pour tout k.

    Retourne shape (n-1, N, N) — C[k] pour k=0..n-2.
    """
    n = P_filt.shape[0]
    N = F.shape[0]
    # F @ P_filt @ F.T + Q, batch version
    FP = np.einsum('nk,tkl->tnl', F, P_filt[:n - 1])        # (n-1, N, N)
    P_pred_next = np.einsum('tnk,lk->tnl', FP, F) + Q       # (n-1, N, N)
    # Batched inverse — 2x2 and 3x3 have closed form but np.linalg.inv is fast enough
    try:
        P_pred_inv = np.linalg.inv(P_pred_next)
    except np.linalg.LinAlgError:
        P_pred_inv = np.array([np.linalg.pinv(P_pred_next[i]) for i in range(n - 1)])
    # P_filt[k] @ F^T
    PF = np.einsum('tnk,lk->tnl', P_filt[:n - 1], F)        # (n-1, N, N)
    C = np.einsum('tnk,tkl->tnl', PF, P_pred_inv)           # (n-1, N, N)
    return C


def fixed_lag_smoother(
    y: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    G: np.ndarray,
    sigma2_drive: float,
    r_scalar: float,
    lag: int,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
) -> FLKSResult:
    """
    Vectorized FLKS. Computes x_s[t | t+L] for all t.

    Edge handling : for t > n-1-L, the effective lag shrinks to n-1-t.
    These tail points are processed in a small Python loop (only L tail
    points affected, negligible cost).
    """
    assert lag >= 0
    if G.ndim == 1:
        G = G.reshape(-1, 1)
    N = F.shape[0]

    # ---- Forward pass once ----
    fwd = forward_filter(y, F, H, G, sigma2_drive, r_scalar, x0, P0)
    x_filt = fwd.x_filt        # (n, N)
    P_filt = fwd.P_filt        # (n, N, N)
    n = len(y)

    if lag == 0:
        x_smoothed = x_filt.copy()
        residuals = np.full(n, np.nan)
        finite = np.isfinite(y)
        residuals[finite] = y[finite] - (x_smoothed[finite] @ H.T).ravel()
        return FLKSResult(
            x_smoothed=x_smoothed,
            residuals=residuals,
            effective_lag=np.zeros(n, dtype=np.int32),
        )

    Q = sigma2_drive * (G @ G.T)

    # ---- Precompute C[k] for all k ----
    C = _precompute_C(P_filt, F, Q)         # (n-1, N, N)

    # ---- Precompute F @ x_filt[k] for all k ----
    F_xfilt = x_filt @ F.T                  # (n, N)  (F @ x_filt[k] for each k)

    x_smoothed = x_filt.copy()
    effective_lag = np.full(n, 0, dtype=np.int32)

    # ---- Main vectorized backward passes ----
    # For t ∈ [0, n-1-L], the backward pass starts at t+L and goes L steps to t.
    # We process these in a BATCHED backward iteration of L steps, where at step i,
    # all t's simultaneously advance by 1 step backward (from k = t+L-i to k = t+L-i-1).
    L = lag
    t_max = n - 1 - L   # last t that has full lag L
    if t_max >= 0:
        n_batch = t_max + 1         # number of t's with full lag
        # Initial backward state : x_s_batch[t] = x_filt[t + L]
        x_s_batch = x_filt[L : L + n_batch].copy()      # (n_batch, N)

        for i in range(L):
            # At step i, for each t in [0, n_batch), current k = t + L - 1 - i
            # So k_arr = L - 1 - i + arange(n_batch) = slice(L-1-i, L-1-i+n_batch)
            k_start = L - 1 - i
            k_end = k_start + n_batch                   # exclusive
            # Safety
            if k_start < 0 or k_end > n - 1:
                break

            Ck = C[k_start : k_end]                     # (n_batch, N, N)
            x_filt_k = x_filt[k_start : k_end]          # (n_batch, N)
            F_xfilt_k = F_xfilt[k_start : k_end]        # (n_batch, N)

            diff = x_s_batch - F_xfilt_k                # (n_batch, N)
            # x_s = x_filt[k] + C[k] @ diff
            Cdiff = np.einsum('tnk,tk->tn', Ck, diff)   # (n_batch, N)
            x_s_batch = x_filt_k + Cdiff

        # After L steps, x_s_batch[t] = x_s[t | t+L] for t in [0, n_batch)
        x_smoothed[:n_batch] = x_s_batch
        effective_lag[:n_batch] = L

    # ---- Edge cases : t ∈ [t_max+1, n-1] → lag_eff = n-1-t < L ----
    # Small Python loop (at most L iterations)
    for t in range(max(0, t_max + 1), n):
        end = n - 1
        if end == t:
            effective_lag[t] = 0
            continue
        lag_eff = end - t
        effective_lag[t] = lag_eff
        x_s = x_filt[end].copy()
        for k in range(end - 1, t - 1, -1):
            x_s = x_filt[k] + C[k] @ (x_s - F_xfilt[k])
        x_smoothed[t] = x_s

    # ---- Residuals ----
    residuals = np.full(n, np.nan)
    finite = np.isfinite(y)
    residuals[finite] = y[finite] - (x_smoothed[finite] @ H.T).ravel()

    return FLKSResult(
        x_smoothed=x_smoothed,
        residuals=residuals,
        effective_lag=effective_lag,
    )


def full_rts_smoother(
    y: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    G: np.ndarray,
    sigma2_drive: float,
    r_scalar: float,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
) -> FLKSResult:
    """
    Full RTS smoother (equivalent to lag = infinity). Delegates to kf_nd.rts_smoother.
    """
    if G.ndim == 1:
        G = G.reshape(-1, 1)
    x_smooth = rts_smoother(y, F, H, G, sigma2_drive, r_scalar, x0, P0)
    n = len(y)
    residuals = np.full(n, np.nan)
    finite = np.isfinite(y)
    residuals[finite] = y[finite] - (x_smooth[finite] @ H.T).ravel()
    # Effective lag = n (sentinel)
    effective_lag = np.full(n, n, dtype=np.int32)
    return FLKSResult(
        x_smoothed=x_smooth,
        residuals=residuals,
        effective_lag=effective_lag,
    )


if __name__ == "__main__":
    # Smoke tests on small synthetic series
    rng = np.random.default_rng(0)
    n = 1000
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    G = np.array([[1.0], [1.0]])
    sigma2 = 0.01
    r = 0.5
    y = np.cumsum(rng.standard_normal(n) * 0.1) + rng.standard_normal(n) * np.sqrt(r)

    fwd = forward_filter(y, F, H, G, sigma2, r)
    flks_0 = fixed_lag_smoother(y, F, H, G, sigma2, r, lag=0)
    flks_5 = fixed_lag_smoother(y, F, H, G, sigma2, r, lag=5)
    flks_20 = fixed_lag_smoother(y, F, H, G, sigma2, r, lag=20)
    flks_inf = full_rts_smoother(y, F, H, G, sigma2, r)

    # lag=0 must equal forward
    d0 = np.max(np.abs(flks_0.x_smoothed - fwd.x_filt))
    assert d0 < 1e-10, f"lag=0 should equal forward, got max|diff|={d0}"

    # Monotonic improvement expected in MSE vs y
    mse = lambda a: float(np.mean((y[50:] - a[50:, 0]) ** 2))
    mse_0 = mse(flks_0.x_smoothed)
    mse_5 = mse(flks_5.x_smoothed)
    mse_20 = mse(flks_20.x_smoothed)
    mse_inf = mse(flks_inf.x_smoothed)
    print(f"FLKS MSE vs y : lag=0: {mse_0:.4f}  lag=5: {mse_5:.4f}  lag=20: {mse_20:.4f}  lag=inf: {mse_inf:.4f}")
    print("Expect monotonically decreasing (smoothing lowers MSE)")
    print("OK" if mse_0 >= mse_5 >= mse_20 >= mse_inf else "⚠️ non-monotonic")
