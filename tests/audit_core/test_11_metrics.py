"""
Audit unitaire — fonctions de métriques (core.py:531-560)
==========================================================

3 fonctions utilisées pour mesurer la qualité des slopes prédites vs oracle:
- sign_concordance: % de signes identiques entre test et oracle
- find_oracle_transitions: masque True aux transitions de signe de l'oracle
- sign_concordance_at_transitions: concordance spécifiquement aux transitions

EPSILON = 1e-8 : seuil pour ignorer les zéros (valeurs < EPSILON = "ambiguës").

Lancement:
    python -m pytest tests/audit_core/test_11_metrics.py -v -s
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import (
    sign_concordance,
    find_oracle_transitions,
    sign_concordance_at_transitions,
)


# ============================================================================
# TESTS — sign_concordance
# ============================================================================

class TestSignConcordance:

    def test_identity_is_100_pct(self):
        """slopes_test = slopes_oracle → 100%."""
        oracle = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
        test = oracle.copy()
        pct, n = sign_concordance(test, oracle, 0, 5)
        assert pct == 100.0
        assert n == 5

    def test_opposite_is_0_pct(self):
        """slopes_test = -slopes_oracle → 0%."""
        oracle = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
        test = -oracle
        pct, n = sign_concordance(test, oracle, 0, 5)
        assert pct == 0.0
        assert n == 5

    def test_half_match(self):
        """2 matches / 4 total → 50%."""
        oracle = np.array([1.0, -1.0, 1.0, -1.0])
        test = np.array([1.0, 1.0, -1.0, -1.0])  # match at 0 and 3
        pct, n = sign_concordance(test, oracle, 0, 4)
        assert pct == 50.0
        assert n == 4

    def test_nan_ignored(self):
        """NaN dans test ou oracle → ignoré."""
        oracle = np.array([1.0, np.nan, 2.0, 3.0])
        test = np.array([1.0, 1.0, np.nan, 3.0])
        pct, n = sign_concordance(test, oracle, 0, 4)
        # Seuls les indices 0 et 3 ont les deux valides
        assert n == 2
        assert pct == 100.0

    def test_oracle_near_zero_ignored(self):
        """Oracle avec |s_o| < EPSILON (1e-8) → ignoré."""
        oracle = np.array([1.0, 1e-10, -1.0, 1e-9])  # indices 1, 3 sont ≈ 0
        test = np.array([1.0, -1.0, -1.0, 1.0])
        pct, n = sign_concordance(test, oracle, 0, 4)
        # Seuls indices 0 et 2 sont valides
        assert n == 2
        assert pct == 100.0

    def test_start_end_bounds(self):
        """start/end respectés."""
        oracle = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
        test = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
        # Sur [0, 3) : indices 0, 1, 2 → test [+, +, -] vs oracle [+, +, +] → 2/3 = 66.67%
        pct, n = sign_concordance(test, oracle, 0, 3)
        assert n == 3
        assert abs(pct - 200.0 / 3.0) < 1e-6

    def test_empty_range_returns_nan(self):
        """Si aucun point valide, retour (NaN, 0)."""
        oracle = np.array([np.nan, np.nan])
        test = np.array([1.0, 1.0])
        pct, n = sign_concordance(test, oracle, 0, 2)
        assert np.isnan(pct)
        assert n == 0

    def test_returns_percentage(self):
        """Retour en % (×100), pas en fraction."""
        oracle = np.array([1.0, -1.0, 1.0])
        test = np.array([1.0, -1.0, 1.0])
        pct, _ = sign_concordance(test, oracle, 0, 3)
        assert pct == 100.0  # pas 1.0


# ============================================================================
# TESTS — find_oracle_transitions
# ============================================================================

class TestFindOracleTransitions:

    def test_shape_equals_slice_length(self):
        oracle = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trans = find_oracle_transitions(oracle, 0, 5)
        assert len(trans) == 5
        assert trans.dtype == bool

    def test_all_positive_no_transitions(self):
        """Tous positifs → aucune transition."""
        oracle = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trans = find_oracle_transitions(oracle, 0, 5)
        assert not np.any(trans)

    def test_single_transition_detected(self):
        """[+, +, +, -, -] → transition à index 3."""
        oracle = np.array([1.0, 2.0, 3.0, -1.0, -2.0])
        trans = find_oracle_transitions(oracle, 0, 5)
        expected = np.array([False, False, False, True, False])
        np.testing.assert_array_equal(trans, expected)

    def test_multiple_transitions(self):
        """[+, -, +, -, +] → 4 transitions à 1, 2, 3, 4."""
        oracle = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        trans = find_oracle_transitions(oracle, 0, 5)
        expected = np.array([False, True, True, True, True])
        np.testing.assert_array_equal(trans, expected)

    def test_zeros_not_counted_as_transitions(self):
        """
        Transition via zéro: [+, 0, -]. sign_o = [+1, 0, -1].
        La condition `sign_o[i] != 0 AND sign_o[i-1] != 0 AND sign_o[i] != sign_o[i-1]`
        n'est pas satisfaite entre un +1 et un 0, ni entre un 0 et un -1.
        Donc trans = [F, F, F].
        """
        oracle = np.array([1.0, 0.0, -1.0])  # 0.0 < EPSILON
        trans = find_oracle_transitions(oracle, 0, 3)
        expected = np.array([False, False, False])
        np.testing.assert_array_equal(trans, expected)

    def test_near_zero_treated_as_zero(self):
        """|s_o| < EPSILON (1e-8) → traité comme 0."""
        oracle = np.array([1.0, 1e-10, -1.0])  # 1e-10 < EPSILON
        trans = find_oracle_transitions(oracle, 0, 3)
        # Pareil que test_zeros_not_counted : pas de transition via zéro
        expected = np.array([False, False, False])
        np.testing.assert_array_equal(trans, expected)

    def test_start_offset(self):
        """start != 0 : trans retourne pour la tranche seulement."""
        oracle = np.array([1.0, 2.0, -1.0, -2.0, 1.0])  # transition à 2, 4
        # Tranche [1, 4) : [2, -1, -2] → transition à index 1 (relatif à la tranche)
        trans = find_oracle_transitions(oracle, 1, 4)
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(trans, expected)

    def test_first_index_never_transition(self):
        """L'index 0 de la tranche n'est jamais une transition (pas de précédent)."""
        oracle = np.array([-1.0, 1.0, 1.0])
        trans = find_oracle_transitions(oracle, 0, 3)
        # Même si oracle commence par -1 et devient +, le 1er index n'est pas une transition
        assert not trans[0]
        # Mais l'index 1 est une transition (-1 → +1)
        assert trans[1]


# ============================================================================
# TESTS — sign_concordance_at_transitions
# ============================================================================

class TestSignConcordanceAtTransitions:

    def test_match_at_transitions_only(self):
        """Concordance seulement aux transitions de l'oracle."""
        oracle = np.array([1.0, 1.0, -1.0, -1.0])  # transition à index 2
        test = np.array([-1.0, 1.0, -1.0, 1.0])
        trans = find_oracle_transitions(oracle, 0, 4)
        # trans = [F, F, T, F] → seul index 2 est pris en compte
        # test[2] = -1, oracle[2] = -1 → match → 100%
        pct, n = sign_concordance_at_transitions(test, oracle, 0, 4, trans)
        assert n == 1
        assert pct == 100.0

    def test_no_transitions_returns_nan(self):
        """Aucune transition → (NaN, 0)."""
        oracle = np.array([1.0, 2.0, 3.0, 4.0])  # tous +
        test = np.array([1.0, 2.0, 3.0, 4.0])
        trans = find_oracle_transitions(oracle, 0, 4)
        pct, n = sign_concordance_at_transitions(test, oracle, 0, 4, trans)
        assert n == 0
        assert np.isnan(pct)

    def test_nan_at_transition_is_skipped(self):
        oracle = np.array([1.0, -1.0, 1.0])  # transitions à 1, 2
        test = np.array([1.0, np.nan, 1.0])
        trans = find_oracle_transitions(oracle, 0, 3)
        # trans = [F, T, T], mais test[1] = NaN → skip
        pct, n = sign_concordance_at_transitions(test, oracle, 0, 3, trans)
        assert n == 1  # seul index 2
        assert pct == 100.0

    def test_transition_mismatch(self):
        """Test prédit le mauvais signe à la transition."""
        oracle = np.array([1.0, -1.0])  # transition à 1
        test = np.array([1.0, 1.0])     # test dit encore +
        trans = find_oracle_transitions(oracle, 0, 2)
        pct, n = sign_concordance_at_transitions(test, oracle, 0, 2, trans)
        assert n == 1
        assert pct == 0.0  # mismatch à la transition


# ============================================================================
# TESTS — cohérence inter-fonctions
# ============================================================================

class TestInterFunctionConsistency:

    def test_concordance_at_all_transitions_equals_subset(self):
        """
        sign_concordance_at_transitions doit être cohérent avec sign_concordance
        quand on restreint aux transitions.
        """
        oracle = np.array([1.0, -1.0, 1.0, -1.0, 1.0])  # transitions partout sauf 0
        test = np.array([1.0, 1.0, 1.0, -1.0, 1.0])  # divergences
        trans = find_oracle_transitions(oracle, 0, 5)
        pct, n = sign_concordance_at_transitions(test, oracle, 0, 5, trans)
        # transitions à 1, 2, 3, 4
        # match: idx 1: oracle=-, test=+ → X
        #        idx 2: oracle=+, test=+ → OK
        #        idx 3: oracle=-, test=- → OK
        #        idx 4: oracle=+, test=+ → OK
        # 3/4 = 75%
        assert n == 4
        assert pct == 75.0

    def test_global_concordance_with_full_trans_mask(self):
        """
        sign_concordance_at_transitions avec trans=tous-True devrait donner
        le même résultat que sign_concordance (modulo EPSILON).
        """
        oracle = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        test = np.array([1.0, -1.0, 1.0, 1.0, 1.0])
        # Full concordance
        pct_full, n_full = sign_concordance(test, oracle, 0, 5)
        # Avec trans = tout True
        trans_all = np.ones(5, dtype=bool)
        pct_at, n_at = sign_concordance_at_transitions(test, oracle, 0, 5, trans_all)
        # Doivent être identiques (pas de filtrage EPSILON ici car tous > EPSILON)
        assert n_full == n_at
        assert pct_full == pct_at
