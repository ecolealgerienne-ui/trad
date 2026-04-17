"""
Audit unitaire — primitives Kalman (core.py:380-400)
=====================================================

4 fonctions utilitaires utilisées par forward_filter, compute_slopes_test2,
compute_kalman_live, etc.

- kf_update(x_p, P_p, z): measurement update step
- kf_predict_sub(x, P): prediction step for 5min sub-step (dt = 1/6)
- inv2x2(M): 2x2 matrix inversion with fallback pinv
- is_pos_semidef(M): PSD check for 2x2

Lancement:
    python -m pytest tests/audit_core/test_10_kalman_primitives.py -v -s
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import (
    kf_update,
    kf_predict_sub,
    inv2x2,
    is_pos_semidef,
    A,
    A_SUB,
    Q,
    Q_SUB,
    R,
    H,
    DT_SUB,
    KALMAN_PROCESS_VAR,
    KALMAN_MEASURE_VAR,
)


# ============================================================================
# TESTS — kf_update
# ============================================================================

class TestKfUpdate:

    def test_returns_correct_shapes(self):
        x_p = np.array([1.0, 0.5])
        P_p = np.eye(2)
        z = 1.2
        x_new, P_new = kf_update(x_p, P_p, z)
        assert x_new.shape == (2,)
        assert P_new.shape == (2, 2)

    def test_manual_kalman_formula(self):
        """Vérifier la formule Kalman standard:
            y = z - H x_p
            S = H P_p H^T + R
            K = P_p H^T / S
            x_new = x_p + K y
            P_new = (I - K H) P_p
        """
        x_p = np.array([2.0, 0.3])
        P_p = np.array([[1.0, 0.2], [0.2, 0.5]])
        z = 2.5
        x_new, P_new = kf_update(x_p, P_p, z)

        y = z - (H @ x_p)[0]
        S = (H @ P_p @ H.T + R)[0, 0]
        K = (P_p @ H.T / S).ravel()
        x_expected = x_p + K * y
        P_expected = (np.eye(2) - np.outer(K, H[0])) @ P_p

        np.testing.assert_allclose(x_new, x_expected, rtol=1e-12)
        np.testing.assert_allclose(P_new, P_expected, rtol=1e-12)

    def test_perfect_observation_pulls_state(self):
        """Si R petit (observation précise), x_new est tiré vers z."""
        x_p = np.array([0.0, 0.0])
        P_p = np.eye(2) * 10.0  # grande incertitude → faire confiance à obs
        z = 5.0
        x_new, P_new = kf_update(x_p, P_p, z)
        # La position estimée doit être proche de z
        assert abs(x_new[0] - z) < 1.0, \
            f"Large P_p + observation z=5 should pull state close to 5, got {x_new[0]}"

    def test_no_observation_weight_when_high_R(self):
        """Si R grand, faible crédit à l'observation, x_new reste proche de x_p."""
        from src.signal_processing.core import kf_update as _kfu
        # Pour ce test on va recréer manuellement car R est global
        # Mais on peut vérifier avec P_p très petit (prior très confiant)
        x_p = np.array([1.0, 0.0])
        P_p = np.eye(2) * 0.001  # très confiant
        z = 10.0
        x_new, _ = kf_update(x_p, P_p, z)
        # x_new devrait rester proche de x_p
        assert abs(x_new[0] - x_p[0]) < 0.5

    def test_P_filtered_is_smaller(self):
        """P_new < P_p (observation réduit l'incertitude)."""
        x_p = np.array([0.0, 0.0])
        P_p = np.eye(2) * 2.0
        z = 1.0
        _, P_new = kf_update(x_p, P_p, z)
        # Trace de P_new < trace de P_p
        assert np.trace(P_new) < np.trace(P_p)

    def test_P_filtered_symmetric(self):
        """P_new doit rester symétrique."""
        x_p = np.array([1.0, 0.2])
        P_p = np.array([[1.5, 0.3], [0.3, 0.8]])
        z = 1.2
        _, P_new = kf_update(x_p, P_p, z)
        assert np.allclose(P_new, P_new.T, atol=1e-12), \
            f"P_new should be symmetric, got {P_new}"

    def test_P_filtered_positive_semidefinite(self):
        x_p = np.array([0.0, 0.0])
        P_p = np.eye(2) * 1.5
        z = 1.0
        _, P_new = kf_update(x_p, P_p, z)
        eigvals = np.linalg.eigvalsh(P_new)
        assert np.all(eigvals >= -1e-12), \
            f"P_new eigenvalues should be >= 0, got {eigvals}"


# ============================================================================
# TESTS — kf_predict_sub
# ============================================================================

class TestKfPredictSub:

    def test_position_advances_by_velocity(self):
        """
        A_SUB = [[1, dt_sub], [0, 1]] avec dt_sub = 1/6.
        x_new[0] = x[0] + dt_sub * x[1]
        """
        x = np.array([10.0, 6.0])  # position 10, vitesse 6 par bougie 30m
        P = np.eye(2)
        x_new, _ = kf_predict_sub(x, P)
        expected_pos = 10.0 + DT_SUB * 6.0  # = 10 + 1
        assert abs(x_new[0] - expected_pos) < 1e-12, \
            f"Position advance: got {x_new[0]}, expected {expected_pos}"

    def test_velocity_unchanged(self):
        """Vitesse inchangée après prédiction (constant velocity model)."""
        x = np.array([5.0, 2.3])
        P = np.eye(2)
        x_new, _ = kf_predict_sub(x, P)
        assert x_new[1] == x[1]

    def test_covariance_grows(self):
        """P_new = A_SUB P A_SUB^T + Q_SUB, avec Q_SUB > 0 → trace augmente."""
        x = np.array([0.0, 0.0])
        P = np.eye(2) * 1.0
        _, P_new = kf_predict_sub(x, P)
        # Trace doit avoir augmenté (incertitude augmente pendant prédiction)
        assert np.trace(P_new) > np.trace(P)

    def test_A_SUB_matches_dt(self):
        """A_SUB = [[1, DT_SUB], [0, 1]]."""
        expected = np.array([[1.0, DT_SUB], [0.0, 1.0]])
        np.testing.assert_allclose(A_SUB, expected)

    def test_Q_SUB_is_Q_scaled(self):
        """Q_SUB = Q * DT_SUB."""
        np.testing.assert_allclose(Q_SUB, Q * DT_SUB)

    def test_covariance_symmetric(self):
        x = np.array([1.0, 0.5])
        P = np.array([[1.5, 0.3], [0.3, 0.8]])
        _, P_new = kf_predict_sub(x, P)
        assert np.allclose(P_new, P_new.T, atol=1e-12)


# ============================================================================
# TESTS — inv2x2
# ============================================================================

class TestInv2x2:

    def test_identity_inverse_is_identity(self):
        I = np.eye(2)
        Iinv = inv2x2(I)
        np.testing.assert_allclose(Iinv, I, rtol=1e-12)

    def test_standard_matrix_inverse(self):
        """A @ inv(A) = I pour matrice bien conditionnée."""
        A_mat = np.array([[2.0, 1.0], [1.0, 3.0]])
        Ainv = inv2x2(A_mat)
        product = A_mat @ Ainv
        np.testing.assert_allclose(product, np.eye(2), atol=1e-12)

    def test_matches_numpy_linalg(self):
        A_mat = np.array([[4.0, -2.0], [1.0, 1.0]])
        Ainv_custom = inv2x2(A_mat)
        Ainv_numpy = np.linalg.inv(A_mat)
        np.testing.assert_allclose(Ainv_custom, Ainv_numpy, rtol=1e-12)

    def test_singular_matrix_uses_pinv(self):
        """Matrice singulière → fallback np.linalg.pinv."""
        M_singular = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank 1
        assert abs(np.linalg.det(M_singular)) < 1e-10
        Minv = inv2x2(M_singular)
        # Doit retourner une valeur finie (pas de crash, pas d'inf)
        assert np.all(np.isfinite(Minv)), f"Singular inverse should use pinv: {Minv}"

    def test_near_singular_threshold(self):
        """Déterminant très petit mais > 1e-15 → utilise formule normale."""
        M = np.array([[1e-10, 0.0], [0.0, 1e-10]])  # det = 1e-20 (below 1e-15)
        Minv = inv2x2(M)
        # det = 1e-20 est en dessous du seuil 1e-15, donc utilise pinv
        assert np.all(np.isfinite(Minv))


# ============================================================================
# TESTS — is_pos_semidef
# ============================================================================

class TestIsPosSemidef:

    def test_identity_is_psd(self):
        assert is_pos_semidef(np.eye(2))

    def test_negative_identity_not_psd(self):
        assert not is_pos_semidef(-np.eye(2))

    def test_negative_diagonal_not_psd(self):
        M = np.array([[-1.0, 0.0], [0.0, 1.0]])
        assert not is_pos_semidef(M)
        M2 = np.array([[1.0, 0.0], [0.0, -1.0]])
        assert not is_pos_semidef(M2)

    def test_zero_matrix_is_psd(self):
        """Matrice nulle = PSD (tous les eigvals = 0, diag = 0, det = 0)."""
        assert is_pos_semidef(np.zeros((2, 2)))

    def test_positive_diagonal_and_positive_det(self):
        """Diag positif et déterminant positif → PSD."""
        M = np.array([[2.0, 1.0], [1.0, 2.0]])
        assert is_pos_semidef(M)

    def test_positive_diagonal_but_negative_det_not_psd(self):
        """
        Cas subtil : diag positif mais déterminant négatif → indéfini.
        [[1, 2], [2, 1]] a det = 1 - 4 = -3 < 0 → pas PSD.
        """
        M = np.array([[1.0, 2.0], [2.0, 1.0]])
        assert not is_pos_semidef(M)

    def test_tolerance_on_det(self):
        """La fonction accepte det >= -1e-12 (tolérance numérique)."""
        M = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1, det = 0
        assert is_pos_semidef(M)

    def test_small_negative_det_within_tolerance(self):
        """det légèrement < 0 mais dans la tolérance → encore PSD."""
        # Construire M avec det = -1e-13 (dans tolérance 1e-12)
        M = np.array([[1.0, 1.0], [1.0, 1.0 - 1e-13]])
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        assert det < 0
        assert det > -1e-12  # dans tolérance
        assert is_pos_semidef(M)


# ============================================================================
# TESTS — intégration primitives
# ============================================================================

class TestIntegration:
    """Vérifier que les primitives s'utilisent correctement ensemble."""

    def test_predict_then_update_yields_valid_state(self):
        """Predict + update produit des x, P valides."""
        x0 = np.array([100.0, 0.5])
        P0 = np.eye(2)
        x_pred, P_pred = kf_predict_sub(x0, P0)
        assert is_pos_semidef(P_pred)
        z = 101.0
        x_upd, P_upd = kf_update(x_pred, P_pred, z)
        assert is_pos_semidef(P_upd)
        # x_upd[0] doit être entre x_pred[0] et z
        assert min(x_pred[0], z) - 1e-9 <= x_upd[0] <= max(x_pred[0], z) + 1e-9

    def test_inv2x2_with_P_pred(self):
        """inv2x2(P_pred) doit être applicable (P_pred est PSD donc inversible si non-singulier)."""
        x = np.array([0.0, 0.0])
        P = np.eye(2)
        _, P_pred = kf_predict_sub(x, P)
        Pinv = inv2x2(P_pred)
        # P_pred @ Pinv ≈ I
        product = P_pred @ Pinv
        np.testing.assert_allclose(product, np.eye(2), atol=1e-10)
