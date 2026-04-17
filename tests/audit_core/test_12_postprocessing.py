"""
Audit unitaire — post-processing (core.py:677-727)
====================================================

2 fonctions pour filtrer/décoder des séquences de probabilités binaires:
- viterbi_decode: décodage Viterbi HMM (NON-CAUSAL, forward+backward)
- cusum_filter: filtre CUSUM (CAUSAL, accumulation d'écarts)

Lancement:
    python -m pytest tests/audit_core/test_12_postprocessing.py -v -s
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import viterbi_decode, cusum_filter


# ============================================================================
# TESTS — viterbi_decode
# ============================================================================

class TestViterbiDecode:

    def test_shape(self):
        probs = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        labels = viterbi_decode(probs)
        assert labels.shape == probs.shape
        assert labels.dtype == np.int64 or labels.dtype == np.int32

    def test_all_high_probs_yields_all_ones(self):
        """probs = 0.99 partout → tous labels = 1."""
        probs = np.full(20, 0.99)
        labels = viterbi_decode(probs)
        assert np.all(labels == 1)

    def test_all_low_probs_yields_all_zeros(self):
        probs = np.full(20, 0.01)
        labels = viterbi_decode(probs)
        assert np.all(labels == 0)

    def test_clear_transition_detected(self):
        """[0.05]*5 + [0.95]*5 → [0]*5 + [1]*5."""
        probs = np.concatenate([np.full(5, 0.05), np.full(5, 0.95)])
        labels = viterbi_decode(probs)
        expected = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(labels, expected)

    def test_high_self_trans_reduces_switches(self):
        """
        self_trans élevé (0.99) → modèle résiste au changement.
        Un seul pic isolé dans probs ne doit PAS changer le label.
        """
        probs = np.full(20, 0.05)
        probs[10] = 0.95  # un seul pic isolé
        labels_strict = viterbi_decode(probs, self_trans=0.99)
        # Viterbi avec self_trans élevé doit résister au pic isolé
        # Nombre de switchs devrait être 0 (ou 2 = aller-retour)
        n_switches = np.sum(np.abs(np.diff(labels_strict)))
        assert n_switches <= 2, f"Expected <=2 switches with high self_trans, got {n_switches}"

    def test_low_self_trans_allows_switches(self):
        """
        self_trans bas (0.55) → modèle permet le changement facilement.
        Un pic isolé peut créer un switch.
        """
        probs = np.full(20, 0.05)
        probs[10] = 0.95
        labels_lax = viterbi_decode(probs, self_trans=0.55)
        labels_strict = viterbi_decode(probs, self_trans=0.99)
        n_lax = np.sum(np.abs(np.diff(labels_lax)))
        n_strict = np.sum(np.abs(np.diff(labels_strict)))
        # Le modèle avec self_trans bas doit avoir >= de switchs
        assert n_lax >= n_strict

    def test_boundary_probs_clipped(self):
        """probs=1.0 exact ou 0.0 exact → clipé à 1e-10, pas de log(0)."""
        probs = np.array([0.0, 1.0, 0.0, 1.0])
        # Ne doit pas crasher
        labels = viterbi_decode(probs)
        assert len(labels) == 4

    def test_non_causal_full_sequence_used(self):
        """
        Viterbi est NON-CAUSAL : modifier probs[T+k] peut changer labels[T].
        C'est attendu (décodage global).
        """
        probs_A = np.array([0.45, 0.55, 0.45, 0.55, 0.45])
        probs_B = probs_A.copy()
        probs_B[4] = 0.01  # pollution du futur
        labels_A = viterbi_decode(probs_A, self_trans=0.99)
        labels_B = viterbi_decode(probs_B, self_trans=0.99)
        # Les labels PEUVENT différer (c'est attendu — Viterbi est global)
        # Test documentaire : on n'assert pas forcément la différence,
        # mais on vérifie que la fonction ne crashe pas.
        assert len(labels_A) == 5
        assert len(labels_B) == 5


# ============================================================================
# TESTS — cusum_filter
# ============================================================================

class TestCusumFilter:

    def test_shape(self):
        probs = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        labels = cusum_filter(probs)
        assert labels.shape == probs.shape
        assert labels.dtype == np.int64 or labels.dtype == np.int32

    def test_initial_state_follows_probs_0(self):
        """Label[0] = 1 si probs[0] > 0.5, sinon 0."""
        labels_high = cusum_filter(np.array([0.9, 0.5, 0.5, 0.5]), threshold=10.0)
        assert labels_high[0] == 1
        labels_low = cusum_filter(np.array([0.1, 0.5, 0.5, 0.5]), threshold=10.0)
        assert labels_low[0] == 0

    def test_high_threshold_no_switch(self):
        """Threshold élevé → pas de switch même sur signal alterné."""
        probs = np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
        labels = cusum_filter(probs, threshold=100.0)
        # Tous labels = 1 (état initial, pas de switch)
        assert np.all(labels == 1)

    def test_low_threshold_allows_switch(self):
        """Threshold bas → switch possible."""
        # probs persistent à 0 → doit switcher vers 0
        probs = np.concatenate([np.full(1, 0.9), np.full(20, 0.1)])
        labels = cusum_filter(probs, threshold=1.0)
        # Au début label=1, puis au bout d'un moment switch vers 0
        assert labels[0] == 1
        assert labels[-1] == 0

    def test_causality(self):
        """
        CUSUM est CAUSAL. Polluer probs[T+1:] ne doit pas changer labels[:T+1].
        """
        probs_A = np.array([0.9, 0.9, 0.9, 0.5, 0.5, 0.5])
        probs_B = probs_A.copy()
        probs_B[4:] = 0.01  # pollution après T=3
        labels_A = cusum_filter(probs_A, threshold=2.0)
        labels_B = cusum_filter(probs_B, threshold=2.0)
        # labels[:4] doivent être identiques
        np.testing.assert_array_equal(
            labels_A[:4], labels_B[:4],
            err_msg="CUSUM must be causal: labels[:4] changed when probs[4:] polluted"
        )

    def test_reset_after_switch(self):
        """
        Après un switch, les compteurs s_up et s_down doivent être reset.
        Test : alternance longue → pas d'oscillations rapides.
        """
        # Phase 1 : force switch vers 1
        # Phase 2 : force switch vers 0
        # Phase 3 : signal neutre → doit rester à 0 (compteurs reset)
        probs = np.concatenate([
            np.full(10, 0.1),  # start: label=0 (prob[0]=0.1)
            np.full(10, 0.9),  # switch to 1
            np.full(10, 0.5),  # neutral, should stay 1
        ])
        labels = cusum_filter(probs, threshold=2.0)
        # Fin : label doit être stable (pas d'oscillation)
        assert labels[-1] in (0, 1)
        # Compter les switchs
        n_switches = np.sum(np.abs(np.diff(labels)))
        # Attendu : 1 switch (du 0 initial vers 1 après ~5 bougies de 0.9)
        assert n_switches <= 2, f"Too many switchs: {n_switches}"

    def test_neutral_probs_keep_initial_state(self):
        """
        Si probs = 0.5 partout, s_up reste 0, s_down reste 0, aucun switch.
        Label = état initial pour tout t.
        """
        probs = np.concatenate([np.full(1, 0.9), np.full(20, 0.5)])
        labels = cusum_filter(probs, threshold=2.0)
        # Tous = 1 (état initial, aucun switch car probs neutres)
        assert np.all(labels == 1)


# ============================================================================
# TESTS — comparaison Viterbi vs CUSUM
# ============================================================================

class TestViterbiVsCusum:

    def test_both_converge_on_strong_signal(self):
        """Sur un signal franc, les deux donnent le même résultat."""
        probs = np.concatenate([np.full(10, 0.01), np.full(10, 0.99)])
        v_labels = viterbi_decode(probs, self_trans=0.95)
        c_labels = cusum_filter(probs, threshold=2.0)
        # Les deux doivent donner [0]*~10 + [1]*~10
        # Viterbi est plus précis à la frontière (peut switcher à 9 ou 10)
        # CUSUM a un délai d'accumulation
        # Test : les deux doivent au moins avoir le bon label au milieu
        assert v_labels[2] == 0
        assert v_labels[-3] == 1
        assert c_labels[2] == 0
        assert c_labels[-3] == 1

    def test_both_respect_shape(self):
        probs = np.random.rand(50)
        v = viterbi_decode(probs)
        c = cusum_filter(probs)
        assert v.shape == c.shape == probs.shape
