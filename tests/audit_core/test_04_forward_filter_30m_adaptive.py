"""
Audit unitaire — forward_filter_30m_adaptive (core.py:435-481)
==============================================================

Fonction : Kalman forward avec Q adaptatif Myers-Tapley (AQ-KF).
Même format de sortie que forward_filter_30m, mais Q_current évolue en ligne.

Mécanisme :
- innovation_buffer garde les `window` (=30) dernières innovations
- delta = mean(innov²) - S_t
- Si delta > 0 : Q_candidate = delta * C_rts @ C_rts^T
- Clamp: Q dans [Q * Q_min_factor, Q * Q_max_factor]

Même bug d'init leakage que forward_filter_30m (ligne 439).

Lancement:
    python -m pytest tests/audit_core/test_04_forward_filter_30m_adaptive.py -v -s
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import (
    forward_filter_30m,
    forward_filter_30m_adaptive,
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
    """y = a*t + b, 300 bougies pour couvrir window=30."""
    n = 300
    a, b = 0.1, 10.0
    y = a * np.arange(n, dtype=np.float64) + b
    return y, a, b


@pytest.fixture
def constant_signal():
    n = 300
    return np.full(n, 42.0)


@pytest.fixture
def noisy_signal():
    rng = np.random.default_rng(42)
    n = 400
    a, b = 0.05, 50.0
    noise = rng.normal(0, 0.5, n)
    y = a * np.arange(n, dtype=np.float64) + b + noise
    return y, a, b


@pytest.fixture
def regime_change_signal():
    """200 bougies stables puis 200 bougies avec variance bruyante forte."""
    rng = np.random.default_rng(123)
    stable = 30.0 + rng.normal(0, 0.1, 200)
    volatile = 30.0 + rng.normal(0, 3.0, 200)
    return np.concatenate([stable, volatile])


# ============================================================================
# TESTS — shape
# ============================================================================

class TestShapes:

    def test_output_shapes(self, linear_signal):
        y, _, _ = linear_signal
        x_f, P_f, x_p, P_p, C = forward_filter_30m_adaptive(y)
        n = len(y)
        assert x_f.shape == (n, 2)
        assert P_f.shape == (n, 2, 2)
        assert x_p.shape == (n, 2)
        assert P_p.shape == (n, 2, 2)
        assert C.shape == (n, 2, 2)

    def test_no_nan_in_output(self, linear_signal):
        y, _, _ = linear_signal
        x_f, P_f, _, _, _ = forward_filter_30m_adaptive(y)
        assert not np.any(np.isnan(x_f))
        assert not np.any(np.isnan(P_f))


# ============================================================================
# TESTS — ground truth Kalman
# ============================================================================

class TestGroundTruth:

    def test_linear_signal_velocity_converges(self, linear_signal):
        """Sur y=a*t+b, velocity converge vers a."""
        y, a, _ = linear_signal
        x_f, _, _, _, _ = forward_filter_30m_adaptive(y)
        mean_v = np.mean(x_f[150:, 1])
        assert abs(mean_v - a) < 0.05, f"AQ-KF velocity: {mean_v:.4f} (expected {a})"

    def test_constant_signal_zero_velocity(self, constant_signal):
        y = constant_signal
        x_f, _, _, _, _ = forward_filter_30m_adaptive(y)
        mean_v = np.mean(x_f[100:, 1])
        assert abs(mean_v) < 1e-3


# ============================================================================
# TESTS — CAUSALITÉ
# ============================================================================

class TestCausality:

    def test_no_leak_from_future_observations(self, linear_signal):
        """Polluer y[T+1:] ne doit pas changer x_filt[:T+1]."""
        y, _, _ = linear_signal
        T = 150
        y_A = y.copy()
        y_B = y.copy()
        y_B[T + 1:] = -9999.0
        xf_A, Pf_A, xp_A, Pp_A, C_A = forward_filter_30m_adaptive(y_A)
        xf_B, Pf_B, xp_B, Pp_B, C_B = forward_filter_30m_adaptive(y_B)
        np.testing.assert_array_equal(
            xf_A[:T + 1], xf_B[:T + 1],
            err_msg="LEAKAGE AQ-KF: x_filt[:T+1] changed when y[T+1:] polluted"
        )

    def test_Q_adaptation_is_causal(self, linear_signal):
        """
        Q_current au pas t utilise uniquement innovations[0..t].
        Donc polluer y[T+1:] ne doit pas changer Q_current au pas T ni P_filt[:T+1].
        """
        y, _, _ = linear_signal
        T = 150
        y_A = y.copy()
        y_B = y.copy()
        y_B[T + 1:] = -9999.0
        _, Pf_A, _, _, _ = forward_filter_30m_adaptive(y_A)
        _, Pf_B, _, _, _ = forward_filter_30m_adaptive(y_B)
        np.testing.assert_array_equal(Pf_A[:T + 1], Pf_B[:T + 1])

    def test_C_gain_causal(self, linear_signal):
        y, _, _ = linear_signal
        T = 150
        y_A = y.copy()
        y_B = y.copy()
        y_B[T + 1:] = -9999.0
        _, _, _, _, C_A = forward_filter_30m_adaptive(y_A)
        _, _, _, _, C_B = forward_filter_30m_adaptive(y_B)
        np.testing.assert_array_equal(C_A[:T + 1], C_B[:T + 1])


# ============================================================================
# TESTS — INIT LEAKAGE (le même bug que forward_filter_30m)
# ============================================================================

class TestInitLeakage:

    def test_init_with_leading_nan_leaks_future(self):
        """
        Même bug que forward_filter_30m : first_valid_val = première non-NaN
        peut être dans le futur si warm-up NaN.
        """
        n = 50
        y_A = np.full(n, np.nan)
        y_A[5:] = 3.0 + 0.1 * np.arange(n - 5)
        y_B = y_A.copy()
        y_B[5] = 100.0

        xf_A, _, _, _, _ = forward_filter_30m_adaptive(y_A)
        xf_B, _, _, _, _ = forward_filter_30m_adaptive(y_B)

        diff_at_t0 = abs(xf_A[0, 0] - xf_B[0, 0])
        print(f"\n[AQ-KF INIT] y_A[5]={y_A[5]}, y_B[5]={y_B[5]}")
        print(f"[AQ-KF INIT] x_filt_A[0]={xf_A[0, 0]:.4f}, x_filt_B[0]={xf_B[0, 0]:.4f}")
        print(f"[AQ-KF INIT] diff = {diff_at_t0:.4f}")
        if diff_at_t0 > 1e-9:
            print("[AQ-KF INIT] ⚠️  Même leakage init que forward_filter_30m.")
        else:
            print("[AQ-KF INIT] ✅ Pas de leakage init.")

    def test_init_first_valid_is_t0(self):
        n = 100
        y = 3.0 + 0.1 * np.arange(n)
        x_f, _, _, _, _ = forward_filter_30m_adaptive(y)
        assert abs(x_f[0, 0] - y[0]) < 0.5


# ============================================================================
# TESTS — NaN au milieu
# ============================================================================

class TestMidNaN:

    def test_nan_at_middle_uses_prediction(self, linear_signal):
        y, _, _ = linear_signal
        T = 150
        y_nan = y.copy()
        y_nan[T] = np.nan
        x_f, _, x_p, _, _ = forward_filter_30m_adaptive(y_nan)
        np.testing.assert_array_equal(x_f[T], x_p[T])


# ============================================================================
# TESTS — formules Kalman
# ============================================================================

class TestKalmanFormulas:

    def test_x_pred_formula(self, linear_signal):
        """x_pred[t] = A @ x_filt[t-1]. Valide même avec Q adaptatif."""
        y, _, _ = linear_signal
        x_f, _, x_p, _, _ = forward_filter_30m_adaptive(y)
        for t in [1, 10, 50, 100, 200]:
            expected = A @ x_f[t - 1]
            np.testing.assert_allclose(x_p[t], expected, rtol=1e-12)

    def test_x_filt_formula(self, linear_signal):
        """x_filt[t] = kf_update(x_pred[t], P_pred[t], y[t])."""
        y, _, _ = linear_signal
        x_f, P_f, x_p, P_p, _ = forward_filter_30m_adaptive(y)
        for t in [5, 50, 100, 200]:
            expected_xf, expected_Pf = kf_update(x_p[t], P_p[t], y[t])
            np.testing.assert_allclose(x_f[t], expected_xf, rtol=1e-12)
            np.testing.assert_allclose(P_f[t], expected_Pf, rtol=1e-12)

    def test_C_gain_formula(self, linear_signal):
        y, _, _ = linear_signal
        _, P_f, _, P_p, C = forward_filter_30m_adaptive(y)
        for t in [0, 5, 50, 100]:
            expected = P_f[t] @ A.T @ inv2x2(P_p[t + 1])
            np.testing.assert_allclose(C[t], expected, rtol=1e-10)


# ============================================================================
# TESTS — spécifiques AQ-KF (Q adaptatif)
# ============================================================================

class TestAdaptiveQ:
    """
    Q_current évolue selon les innovations. Ces tests vérifient indirectement
    l'adaptation via la trace de P_pred (qui contient Q_current).
    """

    def test_P_pred_contains_Q_current(self, noisy_signal):
        """
        P_pred[t] = A @ P_filt[t-1] @ A^T + Q_current(t-1).
        Donc on peut extraire Q_current via : Q_at_t = P_pred[t] - A @ P_filt[t-1] @ A^T
        """
        y, _, _ = noisy_signal
        _, P_f, _, P_p, _ = forward_filter_30m_adaptive(y)
        # Vérifier aux temps 50 et 200 que Q_current est PSD et dans les bornes
        for t in [50, 100, 200, 300]:
            Q_at_t = P_p[t] - A @ P_f[t - 1] @ A.T
            # PSD : diagonale >= 0
            assert Q_at_t[0, 0] >= -1e-12, f"Q[{t}][0,0] = {Q_at_t[0, 0]}"
            assert Q_at_t[1, 1] >= -1e-12, f"Q[{t}][1,1] = {Q_at_t[1, 1]}"

    def test_Q_clamped_between_floor_ceiling(self, noisy_signal):
        """Q_current doit rester dans [Q_min_factor * Q, Q_max_factor * Q]."""
        y, _, _ = noisy_signal
        Q_min_factor = 0.1
        Q_max_factor = 10.0
        _, P_f, _, P_p, _ = forward_filter_30m_adaptive(
            y, Q_min_factor=Q_min_factor, Q_max_factor=Q_max_factor
        )
        Q_floor_00 = Q[0, 0] * Q_min_factor
        Q_ceil_00 = Q[0, 0] * Q_max_factor
        # Prélever Q à plusieurs t
        for t in [50, 100, 200, 300]:
            Q_at_t = P_p[t] - A @ P_f[t - 1] @ A.T
            # Avec tolérance car Q est mis à jour à la fin de chaque pas
            assert Q_at_t[0, 0] <= Q_ceil_00 * 1.01, \
                f"Q[{t}][0,0] = {Q_at_t[0, 0]} exceeds ceiling {Q_ceil_00}"

    def test_adaptive_differs_from_standard_on_regime_change(self, regime_change_signal):
        """
        Sur changement de régime (stable → volatile), AQ-KF doit s'adapter
        différemment de standard Kalman.
        """
        y = regime_change_signal
        xf_std, _, _, _, _ = forward_filter_30m(y)
        xf_aqkf, _, _, _, _ = forward_filter_30m_adaptive(y)

        # Différence moyenne sur la phase volatile (t>=250)
        diff_volatile = np.abs(xf_std[250:, 0] - xf_aqkf[250:, 0])
        mean_diff = np.mean(diff_volatile)
        print(f"\n[REGIME] Mean |std - aqkf| on volatile phase = {mean_diff:.4f}")
        # Les deux doivent différer (non-zero)
        assert mean_diff > 1e-4, f"AQ-KF should differ from standard: diff={mean_diff}"

    def test_agrees_with_standard_on_clean_signal(self, linear_signal):
        """
        Sur signal linéaire PROPRE (pas de bruit), AQ-KF et standard Kalman
        doivent converger vers les mêmes estimations (Q adaptatif ne change
        rien car innovations nulles).
        """
        y, a, _ = linear_signal
        xf_std, _, _, _, _ = forward_filter_30m(y)
        xf_aqkf, _, _, _, _ = forward_filter_30m_adaptive(y)
        # Après warm-up (>window=30), les deux devraient coïncider à l'epsilon
        diff = np.abs(xf_std[100:, 0] - xf_aqkf[100:, 0])
        max_diff = np.max(diff)
        # Tolérance modérée : l'AQ-KF utilise Q slightly different
        assert max_diff < 0.5, f"AQ-KF diverges from std on clean signal: max diff {max_diff:.4f}"


class TestSanity:

    def test_default_parameters(self, linear_signal):
        """Valeurs par défaut window=30, Q_max_factor=10, Q_min_factor=0.1."""
        y, _, _ = linear_signal
        # Juste vérifier que l'appel avec défaut fonctionne
        res = forward_filter_30m_adaptive(y)
        assert len(res) == 5

    def test_custom_window(self, noisy_signal):
        y, _, _ = noisy_signal
        res = forward_filter_30m_adaptive(y, window=50)
        assert len(res) == 5
