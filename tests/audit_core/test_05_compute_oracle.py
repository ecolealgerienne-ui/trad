"""
Audit unitaire — compute_oracle (core.py:230-252)
==================================================

Fonction : produit les LABELS (non-causal par design).
Utilise pykalman.smooth() = RTS smoother qui regarde tout l'historique.

Signature : (positions, slopes) = compute_oracle(indicator_30m)
Convention : slopes[t] = positions[t-1] - positions[t-2]
             → représente la pente passée (entre t-2 et t-1)
             → DOIT être identique à celle de compute_slopes_test2

IMPORTANT : l'oracle est VOLONTAIREMENT non-causal. Les tests vérifient:
  1. Shape, NaN handling (sanité)
  2. Ground truth (slope linéaire = a, constant = 0)
  3. Convention slopes[t] = positions[t-1] - positions[t-2] (alignement critique)
  4. Non-causalité assumée (modifier y[T+k] change positions[T])

Lancement:
    python -m pytest tests/audit_core/test_05_compute_oracle.py -v -s
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import (
    compute_oracle,
    forward_filter_30m,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def linear_signal():
    n = 200
    a, b = 0.1, 10.0
    y = a * np.arange(n, dtype=np.float64) + b
    return y, a, b


@pytest.fixture
def constant_signal():
    return np.full(200, 42.0)


@pytest.fixture
def step_signal():
    n = 200
    return np.concatenate([np.zeros(100), np.full(100, 10.0)])


@pytest.fixture
def sine_signal():
    n = 400
    period = 50
    y = np.sin(2 * np.pi * np.arange(n) / period)
    return y, period


# ============================================================================
# TESTS — shapes et NaN handling
# ============================================================================

class TestShapes:

    def test_output_shapes(self, linear_signal):
        y, _, _ = linear_signal
        pos, slopes = compute_oracle(y)
        assert pos.shape == y.shape
        assert slopes.shape == y.shape

    def test_slopes_borders_are_nan(self, linear_signal):
        """slopes[0] et slopes[1] = NaN (pas de t-1 ou t-2)."""
        y, _, _ = linear_signal
        _, slopes = compute_oracle(y)
        assert np.isnan(slopes[0])
        assert np.isnan(slopes[1])
        assert not np.isnan(slopes[2])

    def test_positions_all_finite(self, linear_signal):
        """Sans NaN en entrée, positions tous finis."""
        y, _, _ = linear_signal
        pos, _ = compute_oracle(y)
        assert not np.any(np.isnan(pos))

    def test_less_than_3_valid_returns_nan(self):
        """Si moins de 3 valeurs valides, retour NaN complet."""
        y = np.array([np.nan, np.nan, 1.0, np.nan, np.nan])
        pos, slopes = compute_oracle(y)
        assert np.all(np.isnan(pos))
        assert np.all(np.isnan(slopes))


# ============================================================================
# TESTS — GROUND TRUTH
# ============================================================================

class TestGroundTruth:

    def test_linear_signal_slope_equals_a(self, linear_signal):
        """Sur y = a*t+b, oracle slope ≈ a."""
        y, a, _ = linear_signal
        _, slopes = compute_oracle(y)
        # Le smoother étant non-causal, converge très vite
        mid = slopes[20:-10]
        mean_s = np.nanmean(mid)
        assert abs(mean_s - a) < 0.01, f"Oracle slope: {mean_s} vs a={a}"

    def test_constant_signal_zero_slope(self, constant_signal):
        y = constant_signal
        _, slopes = compute_oracle(y)
        mid = slopes[20:-10]
        max_abs = np.nanmax(np.abs(mid))
        assert max_abs < 1e-6, f"Oracle slope on constant: max |slope| = {max_abs}"

    def test_step_signal_anticipates_transition(self, step_signal):
        """
        Step à t=100. Le smoother est non-causal, donc la pente oracle
        "anticipe" le step : slopes[t] commence à monter AVANT t=100.

        C'est exactement ce qu'on veut pour un LABEL.
        """
        y = step_signal
        _, slopes = compute_oracle(y)
        # Zone avant le step : slopes doit devenir positif à l'approche
        # de t=100 (le smoother voit le step venir)
        max_before = np.nanmax(slopes[80:100])
        print(f"\n[ORACLE STEP] max slope in [80, 100) = {max_before:.6f}")
        print(f"[ORACLE STEP] slopes around step:")
        for t in [95, 98, 99, 100, 101, 102, 105]:
            print(f"  slopes[{t}] = {slopes[t]:.6f}")
        # Le smoother doit anticiper le step
        assert max_before > 0.01, f"Smoother should anticipate step (non-causal): got {max_before}"

    def test_sine_signal_sign_matches_derivative(self, sine_signal):
        """slopes oracle doit suivre le signe de d(sin)/dt = cos."""
        y, period = sine_signal
        n = len(y)
        w = 2 * np.pi / period
        true_slope = w * np.cos(w * np.arange(n))
        _, slopes = compute_oracle(y)
        mask = (np.arange(n) > 10) & (np.abs(true_slope) > 0.01) & ~np.isnan(slopes)
        concordance = np.mean(np.sign(slopes[mask]) == np.sign(true_slope[mask]))
        print(f"\n[ORACLE SINE] sign concordance = {concordance:.3f}")
        assert concordance > 0.95, f"Oracle sign concordance with cos: {concordance}"


# ============================================================================
# TESTS — ALIGNEMENT DE CONVENTION (CRITIQUE)
# ============================================================================

class TestSlopeConvention:
    """
    Vérifier que slopes[t] = positions[t-1] - positions[t-2].
    Cette convention DOIT être identique à celle de compute_slopes_test2
    (qui calcule sm_t1 - sm_t2 = smoothed(t-1) - smoothed(t-2)).
    Sinon les labels ne sont pas alignés avec les features.
    """

    def test_slopes_formula(self, linear_signal):
        y, _, _ = linear_signal
        pos, slopes = compute_oracle(y)
        for t in [5, 50, 100, 150]:
            expected = pos[t - 1] - pos[t - 2]
            assert abs(slopes[t] - expected) < 1e-12, (
                f"slopes[{t}] = {slopes[t]}, expected pos[{t-1}] - pos[{t-2}] = {expected}"
            )

    def test_slopes_represent_past_slope(self, linear_signal):
        """
        Pour signal linéaire y = a*t+b avec a=0.1 :
        - positions[t] ≈ y[t] = 0.1*t + 10
        - slopes[t] = positions[t-1] - positions[t-2] ≈ 0.1
        - Sur 1 bougie de différence: slope = a*(t-1 - (t-2)) = a

        Donc slopes[t] représente la pente MOYENNE sur [t-2, t-1].
        """
        y, a, _ = linear_signal
        pos, slopes = compute_oracle(y)
        for t in [10, 50, 100]:
            # slope attendue = a (pente constante sur signal linéaire)
            # Plus précis : positions[t-1] - positions[t-2] ≈ a (si pos ≈ y)
            assert abs(slopes[t] - a) < 0.01


# ============================================================================
# TESTS — NON-CAUSALITÉ (comportement attendu)
# ============================================================================

class TestNonCausality:
    """
    L'oracle UTILISE LE FUTUR par design (smoother backward).
    Ces tests vérifient que ce comportement est bien présent.

    Si l'oracle était causal par erreur, il ne produirait pas de labels parfaits
    et la performance des features serait atteignable par un modèle causal → pas
    de gain avec le smoother.
    """

    def test_modifying_future_changes_past_positions(self, linear_signal):
        """
        Modifier y[T+k] DOIT changer positions[T] (smoother non-causal).
        Si pas de changement → l'oracle n'est pas un vrai smoother.
        """
        y, _, _ = linear_signal
        T = 50
        y_A = y.copy()
        y_B = y.copy()
        y_B[T + 10] = 1000.0  # modif future

        pos_A, slopes_A = compute_oracle(y_A)
        pos_B, slopes_B = compute_oracle(y_B)

        diff = abs(pos_A[T] - pos_B[T])
        print(f"\n[ORACLE NON-CAUSAL] pos_A[{T}]={pos_A[T]:.4f}, pos_B[{T}]={pos_B[T]:.4f}")
        print(f"[ORACLE NON-CAUSAL] diff = {diff:.6f}")
        assert diff > 1e-6, (
            "Oracle should be non-causal (smoother). "
            f"Modifying y[{T+10}] must change positions[{T}]"
        )

    def test_oracle_differs_from_forward_filter(self, step_signal):
        """
        Sur un step, oracle smoother et forward filter causal doivent différer :
        l'oracle a "vu" le step avant (anticipation), le filter causal doit attendre.

        Avec ratio R/Q = 0.1/0.01 = 10, le smoother anticipe sur ~3 bougies.
        On compare donc tout près du step (t=98-99) où l'anticipation est visible.
        """
        y = step_signal
        _, slopes_oracle = compute_oracle(y)
        x_f, _, _, _, _ = forward_filter_30m(y)
        velocity_fwd = x_f[:, 1]
        # À t=99 (juste avant step à t=100) : oracle anticipe fortement, filter ne voit rien
        t_before = 99
        print(f"\n[ORACLE vs FILTER] slopes_oracle[{t_before}] = {slopes_oracle[t_before]:.4f}")
        print(f"[ORACLE vs FILTER] velocity_fwd[{t_before}] = {velocity_fwd[t_before]:.4f}")
        # Oracle doit avoir une pente positive significative (anticipation)
        assert slopes_oracle[t_before] > 0.5, \
            f"Oracle should anticipate step at t={t_before}: got {slopes_oracle[t_before]}"
        # Le filter causal n'a pas encore vu le step → velocity ≈ 0
        assert abs(velocity_fwd[t_before]) < 0.01, \
            f"Causal filter should have near-zero velocity before step: got {velocity_fwd[t_before]}"


# ============================================================================
# TESTS — gestion NaN
# ============================================================================

class TestNaNHandling:

    def test_leading_nan_preserves_nan(self):
        """Si y[0..4] = NaN, positions[0..4] doit rester NaN."""
        n = 100
        y = np.full(n, np.nan)
        y[5:] = 1.0 + 0.1 * np.arange(n - 5)
        pos, _ = compute_oracle(y)
        # positions[0..4] doit être NaN
        for t in range(5):
            assert np.isnan(pos[t]), f"positions[{t}] should be NaN, got {pos[t]}"
        # positions[5:] doit être fini
        assert not np.any(np.isnan(pos[5:]))

    def test_middle_nan_preserves_nan(self, linear_signal):
        """NaN au milieu préserve NaN aux mêmes positions."""
        y, _, _ = linear_signal
        y_nan = y.copy()
        y_nan[50] = np.nan
        pos, slopes = compute_oracle(y_nan)
        assert np.isnan(pos[50])
        # slopes qui dépendent de pos[50] = NaN
        assert np.isnan(slopes[51])  # = pos[50] - pos[49] → NaN
        assert np.isnan(slopes[52])  # = pos[51] - pos[50] → NaN

    def test_trailing_nan_preserves_nan(self):
        """NaN à la fin préserve NaN."""
        n = 100
        y = np.arange(n, dtype=np.float64)
        y[-5:] = np.nan
        pos, _ = compute_oracle(y)
        for t in range(n - 5, n):
            assert np.isnan(pos[t])


# ============================================================================
# TESTS — comparaison smoother vs filter
# ============================================================================

class TestSmootherProperty:

    def test_smoother_lower_rmse_on_noisy_signal(self):
        """
        Le vrai gain du smoother RTS : sur un signal bruité, la RMSE globale
        par rapport au signal vrai doit être INFÉRIEURE à celle du filter causal.

        (Note : sur un step brutal, le smoother LISSE la transition donc peut
        être moins précis localement, mais gagne en RMSE globale.)
        """
        rng = np.random.default_rng(42)
        n = 300
        true_y = 50.0 + 0.1 * np.arange(n)
        noisy_y = true_y + rng.normal(0, 1.0, n)
        pos_oracle, _ = compute_oracle(noisy_y)
        x_f, _, _, _, _ = forward_filter_30m(noisy_y)
        pos_filter = x_f[:, 0]
        # RMSE hors warm-up
        rmse_oracle = np.sqrt(np.mean((pos_oracle[50:] - true_y[50:]) ** 2))
        rmse_filter = np.sqrt(np.mean((pos_filter[50:] - true_y[50:]) ** 2))
        print(f"\n[SMOOTHER RMSE] oracle = {rmse_oracle:.4f}, filter = {rmse_filter:.4f}")
        assert rmse_oracle < rmse_filter, \
            f"Smoother should have lower RMSE than filter on noisy signal"

    def test_smoother_step_transition_is_smoothed(self, step_signal):
        """
        Corollaire (pas un 'gain') : le smoother LISSE la transition du step.
        Documentaire : vérifie que pos_oracle autour de t=100 est une
        transition graduelle, pas un saut.
        """
        y = step_signal
        pos_oracle, _ = compute_oracle(y)
        # Entre t=95 et t=105, la position oracle doit être monotone croissante
        segment = pos_oracle[95:106]
        diffs = np.diff(segment)
        # Toutes positives (montée continue)
        assert np.all(diffs >= 0), f"Oracle should rise monotonically across step: diffs={diffs}"
        # Début < fin
        assert segment[0] < 5.0 < segment[-1]
        print(f"\n[SMOOTHER STEP] pos_oracle[95..105] = {[f'{x:.2f}' for x in segment]}")
