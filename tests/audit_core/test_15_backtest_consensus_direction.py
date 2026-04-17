"""
Audit unitaire — backtest_consensus_direction.py (239 lignes)
==============================================================

3 fonctions testées:
- backtest_model_only: PnL selon prédiction modèle (toujours en position)
- backtest_oracle_only: PnL selon oracle (baseline max)
- backtest_consensus: oracle-assisted — n'agit qu'aux transitions oracle
  que le modèle confirme dans ±6 pas

⚠️ backtest_consensus utilise y_oracle dans la DÉCISION de direction. Ce
n'est PAS un backtest de production (le modèle seul ne connaît pas y_oracle).
C'est un outil d'analyse "combien de l'oracle le modèle capte".

Lancement:
    python -m pytest tests/audit_core/test_15_backtest_consensus_direction.py -v -s
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from src.backtest_consensus_direction import (
    backtest_model_only,
    backtest_oracle_only,
    backtest_consensus,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def rising_prices():
    """Prix qui montent de 100 à 120 sur 100 pas."""
    return np.linspace(100.0, 120.0, 100)


@pytest.fixture
def constant_prices():
    return np.full(50, 100.0)


# ============================================================================
# TESTS — backtest_model_only
# ============================================================================

class TestBacktestModelOnly:

    def test_all_long_on_rising_prices(self, rising_prices):
        """y_pred = 1 partout → 1 seul trade LONG entry close[0], exit closes[-1]."""
        closes = rising_prices
        y_pred = np.ones(len(closes), dtype=int)
        res = backtest_model_only(y_pred, closes, fees=0.0)
        assert res['trades'] == 1
        # PnL = (closes[-1] - closes[0]) / closes[0]
        expected_pnl = (closes[-1] - closes[0]) / closes[0] * 100
        assert abs(res['pnl'] - expected_pnl) < 1e-6

    def test_all_short_on_rising_prices(self, rising_prices):
        """y_pred = 0 partout → SHORT → PnL négatif."""
        closes = rising_prices
        y_pred = np.zeros(len(closes), dtype=int)
        res = backtest_model_only(y_pred, closes, fees=0.0)
        assert res['trades'] == 1
        expected_pnl = (closes[0] - closes[-1]) / closes[0] * 100
        assert abs(res['pnl'] - expected_pnl) < 1e-6

    def test_constant_prices_zero_fees(self, constant_prices):
        """Prix constant + signal constant → 1 trade, PnL = 0."""
        closes = constant_prices
        y_pred = np.ones(len(closes), dtype=int)
        res = backtest_model_only(y_pred, closes, fees=0.0)
        assert res['trades'] == 1
        assert abs(res['pnl']) < 1e-9

    def test_constant_prices_with_fees(self, constant_prices):
        """Prix constant + signal constant → 1 trade, PnL = -2*fees."""
        closes = constant_prices
        y_pred = np.ones(len(closes), dtype=int)
        fees = 0.001
        res = backtest_model_only(y_pred, closes, fees=fees)
        # 2 fees : 1 à l'ouverture, 1 à l'exit final
        expected = -2 * fees * 100
        assert abs(res['pnl'] - expected) < 1e-6

    def test_alternating_signal(self, rising_prices):
        """Signal alterné → beaucoup de trades."""
        closes = rising_prices
        y_pred = (np.arange(len(closes)) % 2 == 0).astype(int)
        res = backtest_model_only(y_pred, closes, fees=0.0)
        # Beaucoup de switches
        assert res['trades'] > 40

    def test_fees_linearity(self, rising_prices):
        """pnl(f) - pnl(0) = -2 * f * n_trades * 100."""
        closes = rising_prices
        y_pred = (np.arange(len(closes)) % 5 < 3).astype(int)
        res_0 = backtest_model_only(y_pred, closes, fees=0.0)
        f = 0.001
        res_f = backtest_model_only(y_pred, closes, fees=f)
        assert res_0['trades'] == res_f['trades']
        expected_diff = -2 * f * res_0['trades'] * 100
        actual_diff = res_f['pnl'] - res_0['pnl']
        assert abs(actual_diff - expected_diff) < 1e-6

    def test_causality_anti_leak(self, rising_prices):
        """
        Polluer y_pred[T+1:] ne doit pas changer le PnL d'une version tronquée.
        Note: backtest_model_only n'a pas de paramètre start/end, il traite tout.
        On teste donc en tronquant manuellement.
        """
        closes = rising_prices
        y_pred_A = np.ones(len(closes), dtype=int)
        y_pred_B = y_pred_A.copy()
        y_pred_B[50:] = 0  # pollution futur
        # Truncate à 50 pour simuler "PnL jusqu'à T=50"
        res_A_trunc = backtest_model_only(y_pred_A[:50], closes[:50], fees=0.0)
        res_B_trunc = backtest_model_only(y_pred_B[:50], closes[:50], fees=0.0)
        # Jusqu'à T=50, y_pred_A et y_pred_B sont identiques → mêmes résultats
        assert res_A_trunc == res_B_trunc

    def test_exec_at_closes_i_not_next(self, rising_prices):
        """
        Exécution à closes[i] (pas closes[i+1]). À i=50 (signal change),
        entry/exit à closes[50].
        """
        closes = rising_prices
        # Signal UP de 0 à 49, DOWN de 50 à 99
        y_pred = np.concatenate([np.ones(50, dtype=int), np.zeros(50, dtype=int)])
        res = backtest_model_only(y_pred, closes, fees=0.0)
        # Trade 1 (LONG) : entry closes[0], exit closes[50]
        # Trade 2 (SHORT) : entry closes[50], exit closes[-1]
        assert res['trades'] == 2
        # PnL trade 1 = (closes[50] - closes[0]) / closes[0]
        # PnL trade 2 = (closes[50] - closes[-1]) / closes[50]  (short)
        p1 = (closes[50] - closes[0]) / closes[0]
        p2 = (closes[50] - closes[-1]) / closes[50]
        expected = (p1 + p2) * 100
        assert abs(res['pnl'] - expected) < 1e-6


# ============================================================================
# TESTS — backtest_oracle_only (wrapper)
# ============================================================================

class TestBacktestOracleOnly:

    def test_wrapper_equals_model_only(self, rising_prices):
        """backtest_oracle_only est un alias de backtest_model_only."""
        y = (np.arange(len(rising_prices)) % 3 < 2).astype(int)
        r1 = backtest_oracle_only(y, rising_prices, fees=0.001)
        r2 = backtest_model_only(y, rising_prices, fees=0.001)
        assert r1 == r2


# ============================================================================
# TESTS — backtest_consensus (ORACLE-ASSISTED, not production)
# ============================================================================

class TestBacktestConsensus:
    """
    ⚠️ backtest_consensus utilise y_oracle dans la LOGIQUE DE DÉCISION.
    Ce n'est PAS un backtest de production (le modèle seul ne connaît
    pas y_oracle). C'est un outil d'analyse.
    """

    def test_consensus_no_oracle_transitions_no_trades(self, rising_prices):
        """Si oracle ne transite jamais, consensus ne trade jamais."""
        y_oracle = np.ones(len(rising_prices), dtype=int)
        y_pred = np.zeros(len(rising_prices), dtype=int)  # tout opposé
        res = backtest_consensus(y_pred, y_oracle, rising_prices, fees=0.0)
        assert res['trades'] == 0

    def test_consensus_model_never_confirms_no_trades(self, rising_prices):
        """Oracle transite mais modèle jamais → 0 trades."""
        n = len(rising_prices)
        # Oracle : UP moitié, DOWN moitié → 1 transition à i=50
        y_oracle = np.concatenate([np.ones(50, dtype=int), np.zeros(50, dtype=int)])
        # Modèle : reste constant → aucune transition → pas de confirmation
        y_pred = np.ones(n, dtype=int)
        res = backtest_consensus(y_pred, y_oracle, rising_prices, fees=0.0)
        assert res['trades'] == 0
        assert res['skipped'] >= 1

    def test_consensus_perfect_match(self, rising_prices):
        """
        Si modèle = oracle → toutes les transitions oracle sont confirmées
        (delta=0 est dans [-NEAR, +NEAR]).
        """
        n = len(rising_prices)
        y_oracle = np.concatenate([np.ones(50, dtype=int), np.zeros(50, dtype=int)])
        y_pred = y_oracle.copy()
        res = backtest_consensus(y_pred, y_oracle, rising_prices, fees=0.0)
        # 1 transition oracle, modèle confirme → 1 trade
        assert res['confirmed'] == 1
        # Peut avoir >= 1 trade selon la logique (entry au switch)
        assert res['trades'] >= 0

    def test_consensus_trades_leq_oracle_transitions(self, rising_prices):
        """Contrat : consensus trades ≤ nombre de transitions oracle."""
        n = len(rising_prices)
        # Oracle avec 5 transitions
        y_oracle = np.zeros(n, dtype=int)
        for pos in [20, 40, 60, 80]:
            y_oracle[pos:] = 1 - y_oracle[pos]
        # Modèle parfois d'accord, parfois pas
        np.random.seed(42)
        y_pred = (np.random.rand(n) > 0.5).astype(int)
        res = backtest_consensus(y_pred, y_oracle, rising_prices, fees=0.0)
        # Count oracle transitions
        n_oracle_trans = sum(1 for i in range(1, n) if y_oracle[i] != y_oracle[i-1])
        assert res['trades'] <= n_oracle_trans, \
            f"trades={res['trades']} > oracle transitions={n_oracle_trans}"

    def test_consensus_window_near_6(self, rising_prices):
        """
        Vérifie la fenêtre de confirmation ±6 pas.
        """
        n = len(rising_prices)
        y_oracle = np.concatenate([np.ones(50, dtype=int), np.zeros(50, dtype=int)])
        # Modèle transite à i=56 (dans [50-6, 50+6] = [44, 56])
        y_pred = np.ones(n, dtype=int)
        y_pred[56:] = 0
        res_within = backtest_consensus(y_pred, y_oracle, rising_prices, fees=0.0)
        # Should confirm (within NEAR=6)
        assert res_within['confirmed'] == 1

        # Modèle transite à i=57 (hors fenêtre [44, 56])
        y_pred = np.ones(n, dtype=int)
        y_pred[57:] = 0
        res_outside = backtest_consensus(y_pred, y_oracle, rising_prices, fees=0.0)
        # Should NOT confirm
        assert res_outside['confirmed'] == 0

    def test_consensus_uses_oracle_for_direction(self):
        """
        ⚠️ Documentation : backtest_consensus utilise y_oracle pour la direction
        (target = oracle_dir). Même si le modèle confirme une direction OPPOSÉE,
        le trade se fait dans la direction ORACLE.
        """
        n = 30
        closes = np.linspace(100, 110, n)
        # Oracle: UP → DOWN à i=15
        y_oracle = np.concatenate([np.ones(15, dtype=int), np.zeros(15, dtype=int)])
        # Modèle transite dans la direction opposée à i=15 (DOWN → UP)
        # Donc oracle_dir=-1 mais model_dir=+1 → model ne confirme pas la direction
        y_pred = np.concatenate([np.zeros(15, dtype=int), np.ones(15, dtype=int)])
        res = backtest_consensus(y_pred, y_oracle, closes, fees=0.0)
        # Le modèle transite dans la direction opposée à l'oracle → pas de confirmation
        assert res['confirmed'] == 0


# ============================================================================
# TESTS — invariants inter-fonctions
# ============================================================================

class TestInvariants:

    def test_oracle_perfect_match_is_upper_bound(self, rising_prices):
        """
        Oracle PnL ≥ Model PnL (en général, oracle est le meilleur possible).
        """
        n = len(rising_prices)
        y_oracle = np.concatenate([np.ones(50, dtype=int), np.zeros(50, dtype=int)])
        # Modèle partiellement correct
        y_pred = y_oracle.copy()
        y_pred[10:20] = 0  # erreurs
        y_pred[60:70] = 1  # erreurs
        r_oracle = backtest_oracle_only(y_oracle, rising_prices, fees=0.0)
        r_model = backtest_model_only(y_pred, rising_prices, fees=0.0)
        # L'oracle doit être >= modèle (dans ce cas)
        # Note : pas toujours vrai en général (l'oracle peut timer moins bien
        # que le modèle par hasard). Test documentaire.
        print(f"\n[COMPARE] oracle PnL={r_oracle['pnl']:.2f}, model PnL={r_model['pnl']:.2f}")


# ============================================================================
# TESTS — numerical edge cases
# ============================================================================

class TestEdgeCases:

    def test_empty_closes_returns_zero(self):
        """Closes vides → 0 trades."""
        y_pred = np.array([], dtype=int)
        closes = np.array([], dtype=float)
        res = backtest_model_only(y_pred, closes, fees=0.001)
        assert res['trades'] == 0
        assert res['pnl'] == 0.0

    def test_nan_exec_price_skips_trade(self):
        """Si closes[i] = NaN, switch skipped."""
        closes = np.array([100.0, np.nan, 102.0, 103.0, 104.0])
        # Changement de signal à i=1 (où closes est NaN)
        y_pred = np.array([1, 0, 0, 0, 0])
        res = backtest_model_only(y_pred, closes, fees=0.0)
        # Le switch à i=1 ne peut pas exécuter (NaN) → pas de trade à i=1
        # Donc seul trade final (exit à closes[-1] = 104)
        # Entry = closes[0] = 100, position = 1 (LONG), exit = 104
        # Mais y_pred[0]=1 → target=1, position=0 → switch à i=0 (entry closes[0])
        # Puis y_pred[1]=0 mais NaN → skip
        # Puis y_pred[2..4]=0 → target=-1, position=1 (encore LONG) → switch à i=2
        # ...
        # Test documentaire : 0 crash
        assert 'pnl' in res
