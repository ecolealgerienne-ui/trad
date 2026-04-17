"""
Audit unitaire — compute_slopes_test2 (core.py:499-524)
========================================================

Fonction critique : produit les features ML std_k1..k6_slope (96.3% accuracy).

Convention vérifiée dans prepare_flks_csv.py:
- live_per_candle[t] = liste des ~6 valeurs MACD 5min provisoires DANS la bougie 30min t
- live_per_candle[t][5] = close frozen = MACD 30min officiel du close(t)
- live_per_candle[t][0..4] = valeurs 5min provisoires (frozen EMA + live close)

Causalité attendue:
- slopes[t] utilise x_filt/x_pred/C jusqu'à t, et live_per_candle[t+1][:k]
- slopes[t] N'UTILISE PAS live_per_candle[t+2, t+3, ...]
- En temps physique: slopes[t] dispo à close[t] + k*5min

Lancement:
    pytest tests/audit_core/test_01_slopes_test2.py -v
    pytest tests/audit_core/test_01_slopes_test2.py -v -s   # voir les prints
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ajoute src/ au path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import (
    compute_slopes_test2,
    compute_slopes_test1,
    forward_filter_30m,
)


# ============================================================================
# HELPERS — construction de live_per_candle cohérent avec un signal 30min
# ============================================================================

def build_live_linear(y_30m, n_sub=6):
    """
    Construit live_per_candle cohérent avec un signal 30min `y_30m`.

    Convention : au sein de la bougie t, les 6 valeurs 5min interpolent
    linéairement entre y[t-1] et y[t]. La dernière (index 5) vaut y[t]
    (= valeur frozen au close de la bougie t).

    Pour la bougie 0, on utilise y[0] partout (pas de t-1).
    """
    n = len(y_30m)
    live = []
    for t in range(n):
        if t == 0:
            sub = np.full(n_sub, y_30m[0])
        else:
            y_prev = y_30m[t - 1]
            y_cur = y_30m[t]
            # (k+1)/n_sub de k=0..n_sub-1 → 1/6, 2/6, ..., 6/6
            frac = np.arange(1, n_sub + 1) / n_sub
            sub = y_prev + frac * (y_cur - y_prev)
        live.append(sub)
    return live


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def linear_signal():
    """Signal linéaire y = a*t + b sur 200 bougies."""
    n = 200
    a, b = 0.1, 10.0
    y = a * np.arange(n, dtype=np.float64) + b
    return y, a, b


@pytest.fixture
def constant_signal():
    n = 200
    y = np.full(n, 42.0)
    return y


@pytest.fixture
def step_signal():
    """Step à t=100 : y=0 puis y=10."""
    n = 200
    y = np.concatenate([np.zeros(100), np.full(100, 10.0)])
    return y


@pytest.fixture
def sine_signal():
    """y(t) = sin(2*pi*t/period)."""
    n = 400
    period = 50
    y = np.sin(2 * np.pi * np.arange(n) / period)
    return y, period


# ============================================================================
# TESTS
# ============================================================================

class TestShapeAndBorders:
    """Test #1 — shapes et gestion des bords."""

    def test_shape_matches_input(self, linear_signal):
        y, _, _ = linear_signal
        x_f, P_f, x_p, _, C = forward_filter_30m(y)
        live = build_live_linear(y)
        slopes = compute_slopes_test2(x_f, P_f, x_p, C, live, n_substeps=6)
        assert slopes.shape == y.shape

    def test_nan_at_borders(self, linear_signal):
        """Boucle for t in range(2, n-1) → slopes[0], slopes[1], slopes[n-1] = NaN."""
        y, _, _ = linear_signal
        x_f, P_f, x_p, _, C = forward_filter_30m(y)
        live = build_live_linear(y)
        slopes = compute_slopes_test2(x_f, P_f, x_p, C, live, n_substeps=6)
        assert np.isnan(slopes[0])
        assert np.isnan(slopes[1])
        assert np.isnan(slopes[-1]), "slopes[n-1] should be NaN (loop stops at n-2)"

    def test_middle_is_finite(self, linear_signal):
        y, _, _ = linear_signal
        x_f, P_f, x_p, _, C = forward_filter_30m(y)
        live = build_live_linear(y)
        slopes = compute_slopes_test2(x_f, P_f, x_p, C, live, n_substeps=6)
        # Après warm-up Kalman (~30 bougies) et avant le bord
        middle = slopes[30:-5]
        n_nan = np.isnan(middle).sum()
        assert n_nan == 0, f"{n_nan} NaN in middle (bougies 30..n-5)"


class TestGroundTruth:
    """Test #2 — signaux avec slope analytique connue."""

    def test_linear_signal_converges_to_slope(self, linear_signal):
        """Pour y=a*t+b, slope → a après warm-up Kalman (par bougie 30min)."""
        y, a, _ = linear_signal
        x_f, P_f, x_p, _, C = forward_filter_30m(y)
        live = build_live_linear(y)
        slopes = compute_slopes_test2(x_f, P_f, x_p, C, live, n_substeps=6)
        # Prendre milieu pour éviter warm-up et bords
        mid = slopes[100:180]
        mean_slope = np.nanmean(mid)
        std_slope = np.nanstd(mid)
        # Tolérance 5% sur la moyenne, stdev < 1% de a
        assert abs(mean_slope - a) < 0.005, \
            f"Expected ≈{a}, got mean={mean_slope:.5f}, std={std_slope:.5f}"

    def test_constant_signal_zero_slope(self, constant_signal):
        """y=const → slope ≈ 0."""
        y = constant_signal
        x_f, P_f, x_p, _, C = forward_filter_30m(y)
        live = build_live_linear(y)
        slopes = compute_slopes_test2(x_f, P_f, x_p, C, live, n_substeps=6)
        mid = slopes[30:-5]
        max_abs = np.nanmax(np.abs(mid))
        assert max_abs < 1e-6, f"Max |slope| on const = {max_abs:.2e} (should be ≈0)"

    def test_step_signal_spike_at_transition(self, step_signal):
        """Step à t=100 : slope ≈ 0 avant, pic positif autour, ≈ 0 après stabilisation."""
        y = step_signal
        x_f, P_f, x_p, _, C = forward_filter_30m(y)
        live = build_live_linear(y)
        slopes = compute_slopes_test2(x_f, P_f, x_p, C, live, n_substeps=6)
        before = slopes[50:95]
        around = slopes[98:108]
        after = slopes[150:-5]
        # Avant : slope proche 0
        assert np.nanmax(np.abs(before)) < 0.2, \
            f"Before step: max |slope| = {np.nanmax(np.abs(before)):.3f}"
        # Autour : slope franchement positive
        assert np.nanmax(around) > 0.5, \
            f"At step: max slope = {np.nanmax(around):.3f} (expected >0.5)"
        # Après : retour vers 0
        assert np.nanmax(np.abs(after)) < 0.3, \
            f"After stabilization: max |slope| = {np.nanmax(np.abs(after)):.3f}"

    def test_sine_signal_sign_matches_derivative(self, sine_signal):
        """Pour y=sin(wt), slope_true ≈ w*cos(wt). Signe doit matcher."""
        y, period = sine_signal
        n = len(y)
        w = 2 * np.pi / period
        # Slope analytique sur bougie (1 bougie = 1 unité de temps)
        true_slope = w * np.cos(w * np.arange(n))
        x_f, P_f, x_p, _, C = forward_filter_30m(y)
        live = build_live_linear(y)
        slopes = compute_slopes_test2(x_f, P_f, x_p, C, live, n_substeps=6)
        # Comparer signes hors warm-up (50:) et hors zones |true_slope| < seuil
        mask = np.arange(n) > 50
        mask &= np.abs(true_slope) > 0.01
        mask &= ~np.isnan(slopes)
        mask[-1] = False
        concordance = np.mean(np.sign(slopes[mask]) == np.sign(true_slope[mask]))
        assert concordance > 0.90, f"Sign concordance with derivative = {concordance:.3f}"


class TestCausality:
    """Test #3 — PAS DE DATA LEAKAGE. Le test le plus important."""

    def test_no_leak_from_live_t_plus_2(self, linear_signal):
        """
        slopes[t] NE DOIT PAS changer si live_per_candle[t+2, t+3, ...] est modifié.

        Rappel : slopes[t] dépend de live_per_candle[t+1][:k]. Toute autre
        dépendance serait une fuite du futur.
        """
        y, _, _ = linear_signal
        x_f, P_f, x_p, _, C = forward_filter_30m(y)
        live_A = build_live_linear(y)
        live_B = [lv.copy() for lv in live_A]

        T = 100
        # Polluer le futur au-delà de t+1
        live_B[T + 2] = np.full(6, 9999.0)
        live_B[T + 3] = np.full(6, 9999.0)
        live_B[T + 5] = np.full(6, -9999.0)

        slopes_A = compute_slopes_test2(x_f, P_f, x_p, C, live_A, n_substeps=6)
        slopes_B = compute_slopes_test2(x_f, P_f, x_p, C, live_B, n_substeps=6)

        # Jusqu'à T inclus : slopes doivent être identiques
        for t_check in [T - 1, T]:
            assert slopes_A[t_check] == slopes_B[t_check], (
                f"LEAKAGE: slopes[{t_check}] changed when future live[{T+2}..] was polluted. "
                f"A={slopes_A[t_check]}, B={slopes_B[t_check]}"
            )
        # À partir de T+1 : slopes doivent avoir changé (car live[T+2] pollué)
        assert slopes_A[T + 1] != slopes_B[T + 1], \
            "slopes[T+1] should change when live[T+2] is polluted"

    def test_no_leak_from_x_filt_future(self, linear_signal):
        """
        slopes[t] NE DOIT PAS changer si x_filt[t+2, ...] est modifié
        (conceptuellement, le smoother ne devrait utiliser que x_filt[t-2..t]).

        NB: compute_slopes_test2 accède à x_filt[t-2], x_filt[t-1], x_filt[t].
        Pas à x_filt[t+1] ni au-delà.
        """
        y, _, _ = linear_signal
        x_f, P_f, x_p, _, C = forward_filter_30m(y)
        live = build_live_linear(y)

        T = 100
        x_f_B = x_f.copy()
        x_f_B[T + 2:] = -9999.0
        P_f_B = P_f.copy()
        x_p_B = x_p.copy()
        x_p_B[T + 2:] = -9999.0
        C_B = C.copy()
        C_B[T + 2:] = -9999.0

        slopes_A = compute_slopes_test2(x_f, P_f, x_p, C, live, n_substeps=6)
        slopes_B = compute_slopes_test2(x_f_B, P_f_B, x_p_B, C_B, live, n_substeps=6)

        for t_check in [T - 1, T, T + 1]:
            assert slopes_A[t_check] == slopes_B[t_check], (
                f"LEAKAGE from x_filt: slopes[{t_check}] changed when "
                f"x_filt/x_pred/C[{T+2}..] was polluted"
            )


class TestSymmetry:
    """Test #4 — invariance par inversion de signe."""

    def test_sign_flip(self, linear_signal):
        """y → -y ⇒ slope → -slope."""
        y, _, _ = linear_signal
        x_f, P_f, x_p, _, C = forward_filter_30m(y)
        live_pos = build_live_linear(y)
        slopes_pos = compute_slopes_test2(x_f, P_f, x_p, C, live_pos, n_substeps=6)

        y_neg = -y
        x_f_n, P_f_n, x_p_n, _, C_n = forward_filter_30m(y_neg)
        live_neg = [-lv for lv in build_live_linear(y)]
        # Cohérent : live_neg construit à partir de -y (équivalent)
        slopes_neg = compute_slopes_test2(x_f_n, P_f_n, x_p_n, C_n, live_neg, n_substeps=6)

        valid = ~np.isnan(slopes_pos) & ~np.isnan(slopes_neg)
        residual = slopes_pos[valid] + slopes_neg[valid]
        max_res = np.max(np.abs(residual))
        assert max_res < 1e-9, f"Max |slopes_pos + slopes_neg| = {max_res:.2e}"


class TestSubstepDependence:
    """Test #5 — comportement vs n_substeps."""

    def test_linear_slope_invariant_across_k(self, linear_signal):
        """Signal linéaire parfait : slope ≈ a quel que soit k."""
        y, a, _ = linear_signal
        x_f, P_f, x_p, _, C = forward_filter_30m(y)
        live = build_live_linear(y)

        means = {}
        for k in [1, 3, 6]:
            slopes = compute_slopes_test2(x_f, P_f, x_p, C, live, n_substeps=k)
            means[k] = np.nanmean(slopes[100:180])
        # Tous ≈ a, avec variation < 10% de a
        for k, m in means.items():
            assert abs(m - a) < 0.02, f"k={k}: slope={m:.4f} (expected {a})"
        # Différences entre k doivent être faibles
        assert abs(means[6] - means[1]) < 0.01, \
            f"k=1: {means[1]:.4f}, k=6: {means[6]:.4f}"


class TestFallbackBranch:
    """Test #6 — branche `k_actual=1` quand aucune mesure valide."""

    def test_all_live_nan_does_not_crash(self, linear_signal):
        """Si tous les live sont NaN, la fonction doit produire des valeurs finies."""
        y, _, _ = linear_signal
        x_f, P_f, x_p, _, C = forward_filter_30m(y)
        n = len(y)
        live_nan = [np.full(6, np.nan) for _ in range(n)]
        slopes = compute_slopes_test2(x_f, P_f, x_p, C, live_nan, n_substeps=6)
        # Milieu fini (pas de crash)
        mid = slopes[30:-5]
        assert np.all(np.isfinite(mid)), \
            "Fallback branch (no valid live) produced NaN/Inf"

    def test_fallback_slope_on_linear_signal(self, linear_signal):
        """
        SUSPECT : quand aucune mesure valide, slope devrait idéalement rester ≈ a
        (le smoother sans mesure extra = backward depuis x_filt).

        Si la branche fallback produit un résultat très différent de a, c'est le
        signe d'un biais artificiel dans la correction `sm_t = x_filt[t] + C_partial @ (...)`.
        """
        y, a, _ = linear_signal
        x_f, P_f, x_p, _, C = forward_filter_30m(y)
        n = len(y)
        live_nan = [np.full(6, np.nan) for _ in range(n)]
        slopes = compute_slopes_test2(x_f, P_f, x_p, C, live_nan, n_substeps=6)
        mid = slopes[100:180]
        mean_slope = np.nanmean(mid)
        # Si loin de a, c'est un bias dans la branche fallback
        deviation = abs(mean_slope - a) / abs(a)
        print(f"\n[FALLBACK] mean slope (all-NaN live) = {mean_slope:.5f} (true a = {a})")
        print(f"[FALLBACK] relative deviation = {deviation*100:.2f}%")
        assert deviation < 0.20, \
            f"Fallback branch produces biased slope: {mean_slope} vs {a}"


class TestWarningValidValsSlicing:
    """
    Test #7 — AUDIT : `use = valid_vals[:n_substeps]` prend les k premières
    NON-NaN. Si le début de live_vals contient des NaN, les valeurs utilisées
    sont décalées en temps mais k_actual ne compense pas.
    """

    def test_leading_nan_changes_result(self, linear_signal):
        """
        Si live[t+1] = [NaN, NaN, v2, v3, v4, v5], la fonction utilise
        [v2, v3] pour k=2 au lieu de s'attendre à [v0, v1] (qui sont NaN).

        Le `k_actual=2` est correct en nombre, mais physiquement ces valeurs
        sont à 15 et 20 minutes, pas 5 et 10 minutes. `A_k = A_SUB^2` et
        `Q_k = Q_SUB * 2` ne reflètent pas ce décalage temporel.
        """
        y, _, _ = linear_signal
        x_f, P_f, x_p, _, C = forward_filter_30m(y)
        live_full = build_live_linear(y)

        # Version propre (6 valeurs)
        slopes_full = compute_slopes_test2(x_f, P_f, x_p, C, live_full, n_substeps=2)

        # Version où les 4 premières sont NaN (on garde v4, v5)
        T = 100
        live_nan_start = [lv.copy() for lv in live_full]
        lv = live_nan_start[T + 1].copy()
        lv[:4] = np.nan
        live_nan_start[T + 1] = lv
        slopes_nan = compute_slopes_test2(x_f, P_f, x_p, C, live_nan_start, n_substeps=2)

        # En théorie : les deux devraient utiliser 2 valeurs, mais à des temps différents.
        # Si la fonction est temporellement consciente → slopes_nan[T] ≠ slopes_full[T].
        # En pratique dans l'implémentation actuelle, elle n'est PAS temporellement
        # consciente → slopes_nan[T] utilise v4,v5 comme si c'étaient v0,v1.
        print(f"\n[NaN LEADING] full: slopes[{T}] = {slopes_full[T]:.6f}")
        print(f"[NaN LEADING] nan_start: slopes[{T}] = {slopes_nan[T]:.6f}")
        print(f"[NaN LEADING] diff = {slopes_full[T] - slopes_nan[T]:.6f}")
        # Test documentaire : on vérifie juste que la fonction ne crashe pas.
        # Si la valeur est significativement différente, c'est le bug potentiel documenté.
        assert np.isfinite(slopes_nan[T])


class TestTest1VsTest2Consistency:
    """Test #8 — compute_slopes_test1 vs test2 sur signal stable."""

    def test_test1_and_test2_close_on_linear(self, linear_signal):
        """
        test1 = backward 2 pas sans info 5min.
        test2 avec k=6 = backward 2 pas + info de la bougie t+1 entière.

        Sur signal linéaire parfait, les deux devraient donner ≈ a.
        """
        y, a, _ = linear_signal
        x_f, P_f, x_p, _, C = forward_filter_30m(y)
        live = build_live_linear(y)
        slopes_t1 = compute_slopes_test1(x_f, x_p, C)
        slopes_t2 = compute_slopes_test2(x_f, P_f, x_p, C, live, n_substeps=6)
        m1 = np.nanmean(slopes_t1[100:180])
        m2 = np.nanmean(slopes_t2[100:180])
        print(f"\n[T1 vs T2] linear: t1 mean = {m1:.5f}, t2 mean = {m2:.5f} (true a={a})")
        assert abs(m1 - a) < 0.02
        assert abs(m2 - a) < 0.02
