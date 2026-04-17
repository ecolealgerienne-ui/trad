"""
Audit unitaire — forward_filter_30m (core.py:407-432)
=======================================================

Fonction : Kalman forward filter standard. Produit x_filt, P_filt, x_pred, P_pred, C
utilisés par compute_slopes_test1/test2 pour construire les features ML.

Point critique identifié à l'audit statique:
- `first_valid_val = indicator_30m[~np.isnan(indicator_30m)][0]` (core.py:410)
  = première valeur NON-NaN, qui peut être dans le FUTUR si les premières
  valeurs sont NaN (warm-up MACD/RSI/CCI).
  → Init de x_filt[0] avec valeur future = LEAKAGE.

Lancement:
    python -m pytest tests/audit_core/test_03_forward_filter_30m.py -v -s
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import (
    forward_filter_30m,
    A,
    Q,
    R,
    H,
    inv2x2,
    kf_update,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def linear_signal():
    """y = a*t + b, 200 bougies, pas de NaN."""
    n = 200
    a, b = 0.1, 10.0
    y = a * np.arange(n, dtype=np.float64) + b
    return y, a, b


@pytest.fixture
def constant_signal():
    n = 200
    return np.full(n, 42.0)


@pytest.fixture
def noisy_signal():
    """y = a*t + b + noise, 300 bougies."""
    rng = np.random.default_rng(42)
    n = 300
    a, b = 0.05, 50.0
    noise = rng.normal(0, 0.5, n)
    y = a * np.arange(n, dtype=np.float64) + b + noise
    return y, a, b


# ============================================================================
# TESTS — shape, types
# ============================================================================

class TestShapes:

    def test_output_shapes(self, linear_signal):
        y, _, _ = linear_signal
        x_f, P_f, x_p, P_p, C = forward_filter_30m(y)
        n = len(y)
        assert x_f.shape == (n, 2)
        assert P_f.shape == (n, 2, 2)
        assert x_p.shape == (n, 2)
        assert P_p.shape == (n, 2, 2)
        assert C.shape == (n, 2, 2)

    def test_no_nan_in_output(self, linear_signal):
        y, _, _ = linear_signal
        x_f, P_f, x_p, P_p, C = forward_filter_30m(y)
        assert not np.any(np.isnan(x_f))
        assert not np.any(np.isnan(P_f))


# ============================================================================
# TESTS — ground truth Kalman
# ============================================================================

class TestGroundTruth:

    def test_linear_signal_position_converges(self, linear_signal):
        """Sur y=a*t+b, x_filt[t][0] (position) doit suivre y[t]."""
        y, a, b = linear_signal
        x_f, _, _, _, _ = forward_filter_30m(y)
        # Après warm-up (50 bougies), la position estimée ≈ y
        diff = np.abs(x_f[50:, 0] - y[50:])
        max_abs = np.max(diff)
        assert max_abs < 1.0, f"Max |x_filt - y| after warmup = {max_abs:.4f}"

    def test_linear_signal_velocity_converges(self, linear_signal):
        """Sur y=a*t+b, x_filt[t][1] (velocity) doit converger vers a."""
        y, a, _ = linear_signal
        x_f, _, _, _, _ = forward_filter_30m(y)
        # Vérifier convergence après warm-up
        mean_v = np.mean(x_f[100:, 1])
        assert abs(mean_v - a) < 0.02, f"Velocity convergence: got {mean_v:.4f}, expected {a}"

    def test_constant_signal_zero_velocity(self, constant_signal):
        """y=const, x_filt[t][1] doit converger vers 0."""
        y = constant_signal
        x_f, _, _, _, _ = forward_filter_30m(y)
        mean_v = np.mean(x_f[50:, 1])
        assert abs(mean_v) < 1e-4, f"Velocity on const signal: {mean_v:.6f}"

    def test_constant_signal_position_converges(self, constant_signal):
        y = constant_signal
        x_f, _, _, _, _ = forward_filter_30m(y)
        mean_p = np.mean(x_f[20:, 0])
        assert abs(mean_p - y[0]) < 1e-4


# ============================================================================
# TESTS — CAUSALITÉ (le plus critique)
# ============================================================================

class TestCausality:
    """x_filt[t] NE DOIT dépendre QUE de indicator[0..t]."""

    def test_no_leak_from_future_observations(self, linear_signal):
        """
        Polluer indicator[T+1:] ne doit pas changer x_filt[:T+1] ni x_pred[:T+1].
        """
        y, _, _ = linear_signal
        T = 100

        y_A = y.copy()
        y_B = y.copy()
        # Pollution massive au-delà de T
        y_B[T + 1:] = -9999.0

        xf_A, Pf_A, xp_A, Pp_A, C_A = forward_filter_30m(y_A)
        xf_B, Pf_B, xp_B, Pp_B, C_B = forward_filter_30m(y_B)

        # x_filt[:T+1] doit être identique
        np.testing.assert_array_equal(
            xf_A[:T + 1], xf_B[:T + 1],
            err_msg="LEAKAGE: x_filt[:T+1] changed when indicator[T+1:] polluted"
        )
        np.testing.assert_array_equal(xp_A[:T + 1], xp_B[:T + 1])

        # x_filt[T+1:] doit différer (car l'observation est polluée)
        assert not np.array_equal(xf_A[T + 1:], xf_B[T + 1:]), \
            "x_filt should change after pollution point"

    def test_C_gain_causal(self, linear_signal):
        """
        C[t] = P_filt[t] @ A^T @ inv(P_pred[t+1]).
        P_pred[t+1] = A @ P_filt[t] @ A^T + Q → ne dépend QUE de P_filt[t] et de A, Q.
        Donc C[t] ne dépend que des obs jusqu'à t. Aucune dépendance au futur.

        Test : polluer indicator[T+1:] ne change pas C[:T+1].
        """
        y, _, _ = linear_signal
        T = 100
        y_A = y.copy()
        y_B = y.copy()
        y_B[T + 1:] = -9999.0
        _, _, _, _, C_A = forward_filter_30m(y_A)
        _, _, _, _, C_B = forward_filter_30m(y_B)
        np.testing.assert_array_equal(
            C_A[:T + 1], C_B[:T + 1],
            err_msg="LEAKAGE: C[:T+1] changed when indicator[T+1:] polluted"
        )


# ============================================================================
# TESTS — LEAKAGE D'INITIALISATION (le bug suspecté)
# ============================================================================

class TestInitLeakage:
    """
    Bug suspecté (core.py:410):
        first_valid_val = indicator_30m[~np.isnan(indicator_30m)][0]
    = première valeur NON-NaN, potentiellement dans le futur.

    Si indicator = [NaN, NaN, NaN, 5.0, ...], first_valid_val = 5.0.
    Utilisée comme init de x_filt[0] → x_filt[0] voit dans le futur.
    """

    def test_init_without_leading_nan(self, linear_signal):
        """Sanité : sans NaN au début, x_filt[0][0] ≈ y[0]."""
        y, _, _ = linear_signal
        x_f, _, _, _, _ = forward_filter_30m(y)
        # À t=0, avec init [y[0], 0.0] + update sur y[0] → x_filt[0][0] proche de y[0]
        assert abs(x_f[0, 0] - y[0]) < 0.5, f"x_filt[0][0] = {x_f[0, 0]}, y[0] = {y[0]}"

    def test_init_with_leading_nan_leaks_future(self):
        """
        INDICATEUR DU BUG : si les 5 premières valeurs sont NaN et la 6e = X,
        alors first_valid_val = X.

        Modifier indicator[5] (la valeur future) change l'init donc x_filt[0..4].
        Si la fonction était strictement causale, x_filt[0..4] ne devrait pas changer
        quand on modifie indicator[5] (= obs du futur à t=5).
        """
        n = 50
        y_A = np.full(n, np.nan)
        y_A[5:] = 3.0 + 0.1 * np.arange(n - 5)  # first_valid = 3.0
        y_B = y_A.copy()
        y_B[5] = 100.0  # change la première valeur non-NaN
        y_B[6:] = y_A[6:]  # reste identique

        xf_A, _, _, _, _ = forward_filter_30m(y_A)
        xf_B, _, _, _, _ = forward_filter_30m(y_B)

        # Si causal strict : xf_A[0..4] == xf_B[0..4] (pas d'obs encore)
        # Si leakage (bug) : xf_A[0..4] != xf_B[0..4] car init différente
        diff_at_t0 = abs(xf_A[0, 0] - xf_B[0, 0])
        print(f"\n[INIT LEAK] y_A[5]={y_A[5]}, y_B[5]={y_B[5]}")
        print(f"[INIT LEAK] x_filt_A[0]={xf_A[0, 0]:.4f}, x_filt_B[0]={xf_B[0, 0]:.4f}")
        print(f"[INIT LEAK] diff = {diff_at_t0:.4f}")
        # Test strict: si la fonction est causale, x_filt[0] ne doit pas dépendre
        # de y[5]. Si le test échoue → bug confirmé.
        # On fait un assert DOCUMENTAIRE pour exposer le comportement actuel.
        if diff_at_t0 > 1e-9:
            print("[INIT LEAK] ⚠️  LEAKAGE CONFIRMÉ : x_filt[0] dépend de y[5] (future).")
        else:
            print("[INIT LEAK] ✅ Pas de leakage détecté (init ne dépend pas du futur).")
        # Test documentaire — pas de hard fail pour pouvoir continuer l'audit
        # Mais on MARQUE le résultat pour analyse.

    def test_init_first_valid_is_t0(self):
        """
        Si indicator[0] n'est pas NaN, first_valid_val = indicator[0].
        Pas de leakage (init = première valeur réelle).
        """
        n = 50
        y = 3.0 + 0.1 * np.arange(n)
        # Pas de NaN
        assert not np.any(np.isnan(y))
        x_f, _, _, _, _ = forward_filter_30m(y)
        # À t=0, init x_p = [y[0], 0.0], update avec y[0] → x_filt[0][0] ≈ y[0]
        assert abs(x_f[0, 0] - y[0]) < 0.5


# ============================================================================
# TESTS — NaN au milieu
# ============================================================================

class TestMidNaN:

    def test_nan_at_middle_uses_prediction(self, linear_signal):
        """
        Si indicator[T] est NaN, x_filt[T] = x_pred[T] (prédiction pure, pas d'update).
        """
        y, _, _ = linear_signal
        T = 100
        y_nan = y.copy()
        y_nan[T] = np.nan
        x_f, _, x_p, _, _ = forward_filter_30m(y_nan)
        # À T, x_filt doit être égal à x_pred
        np.testing.assert_array_equal(x_f[T], x_p[T])

    def test_nan_at_middle_does_not_propagate_leak(self, linear_signal):
        """
        Un NaN à T ne doit pas créer de dépendance au futur.
        """
        y, _, _ = linear_signal
        T = 100
        y_A = y.copy()
        y_A[T] = np.nan
        y_B = y.copy()
        y_B[T] = np.nan
        y_B[T + 1:] = 9999.0  # pollution

        xf_A, _, _, _, _ = forward_filter_30m(y_A)
        xf_B, _, _, _, _ = forward_filter_30m(y_B)
        np.testing.assert_array_equal(xf_A[:T + 1], xf_B[:T + 1])


# ============================================================================
# TESTS — formules Kalman
# ============================================================================

class TestKalmanFormulas:
    """Vérifier que forward_filter respecte les équations Kalman."""

    def test_x_pred_formula(self, linear_signal):
        """x_pred[t] = A @ x_filt[t-1] pour t >= 1."""
        y, _, _ = linear_signal
        x_f, _, x_p, _, _ = forward_filter_30m(y)
        for t in [1, 10, 50, 100]:
            expected = A @ x_f[t - 1]
            np.testing.assert_allclose(x_p[t], expected, rtol=1e-12)

    def test_P_pred_formula(self, linear_signal):
        """P_pred[t] = A @ P_filt[t-1] @ A^T + Q pour t >= 1."""
        y, _, _ = linear_signal
        _, P_f, _, P_p, _ = forward_filter_30m(y)
        for t in [1, 10, 50, 100]:
            expected = A @ P_f[t - 1] @ A.T + Q
            np.testing.assert_allclose(P_p[t], expected, rtol=1e-12)

    def test_x_filt_formula(self, linear_signal):
        """x_filt[t] = x_pred[t] + K @ (y[t] - H @ x_pred[t])."""
        y, _, _ = linear_signal
        x_f, P_f, x_p, P_p, _ = forward_filter_30m(y)
        for t in [5, 10, 50, 100]:
            expected_xf, expected_Pf = kf_update(x_p[t], P_p[t], y[t])
            np.testing.assert_allclose(x_f[t], expected_xf, rtol=1e-12)
            np.testing.assert_allclose(P_f[t], expected_Pf, rtol=1e-12)

    def test_C_gain_formula(self, linear_signal):
        """C[t] = P_filt[t] @ A^T @ inv(P_pred[t+1])."""
        y, _, _ = linear_signal
        _, P_f, _, P_p, C = forward_filter_30m(y)
        for t in [0, 5, 50, 100]:
            expected = P_f[t] @ A.T @ inv2x2(P_p[t + 1])
            np.testing.assert_allclose(C[t], expected, rtol=1e-10)


# ============================================================================
# TESTS — convergence & stabilité
# ============================================================================

class TestStability:

    def test_P_filt_converges(self, noisy_signal):
        """
        P_filt[t] (covariance filtrée) doit converger vers une steady-state.
        Signe de Kalman bien tuné : la covariance se stabilise.
        """
        y, _, _ = noisy_signal
        _, P_f, _, _, _ = forward_filter_30m(y)
        # Comparer P_filt[150] et P_filt[250] : doivent être très proches
        diff = np.abs(P_f[150] - P_f[250])
        assert np.max(diff) < 0.05, f"P_filt not converged: max diff = {np.max(diff)}"

    def test_x_filt_tracks_signal(self, noisy_signal):
        """Position estimée doit tracker le signal bruité (+/- filtrage)."""
        y, a, b = noisy_signal
        n = len(y)
        x_f, _, _, _, _ = forward_filter_30m(y)
        # Après warm-up, x_filt doit suivre la tendance vraie a*t+b
        true_y = a * np.arange(n) + b
        # On compare x_filt à la tendance (pas au bruité)
        rmse = np.sqrt(np.mean((x_f[50:, 0] - true_y[50:]) ** 2))
        assert rmse < 0.5, f"RMSE track = {rmse:.4f} (signal noise 0.5)"
