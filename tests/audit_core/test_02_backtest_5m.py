"""
Audit unitaire — backtest_5m (core.py:617-662)
================================================

Fonction critique : produit le +870% PnL sur le test set.

Spec:
- À chaque t dans [start, end), si abs(slopes[t]) >= threshold, target = sign(slopes[t]).
- Si target != position et holding_min respecté: ferme position actuelle et ouvre la nouvelle.
- Prix d'exécution: closes_5m_per_candle[t+1][k_substep-1].
- Fees: déduites à l'ouverture ET à la fermeture (2*fees par roundtrip).
- Exit final: fermeture à closes_last[-1] si position != 0.

Causalité attendue:
- Le trade décidé à t utilise exec_price = closes_5m_per_candle[t+1][k-1]
  qui correspond à close[t] + k*5min.
- slopes[t] est dispo à ce même instant (si calculée avec k_substep=k dans
  compute_slopes_test2). Pas d'info du futur au-delà.
- PnL sur [start, end) ne doit pas dépendre de closes[t >= end+1].

Convention closes_5m_per_candle:
- Liste de length = n_bougies_30m
- closes_5m_per_candle[t] = array des ~6 close 5min dans la bougie 30m t
- closes_5m_per_candle[t][-1] = close de la bougie 30m (dernière 5min)

Lancement:
    python -m pytest tests/audit_core/test_02_backtest_5m.py -v -s
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import backtest_5m, _exec_trade


# ============================================================================
# HELPERS
# ============================================================================

def make_closes_from_prices_30m(prices_30m, n_sub=6):
    """
    Construit closes_5m_per_candle cohérent avec un prix 30min donné.
    closes_5m_per_candle[t][k] = interpolation linéaire entre price[t-1] et price[t].
    closes_5m_per_candle[0][k] = price[0] (pas de t-1).
    closes_5m_per_candle[t][-1] = price[t] (close officiel de la bougie t).
    """
    n = len(prices_30m)
    per_candle = []
    for t in range(n):
        if t == 0:
            sub = np.full(n_sub, prices_30m[0])
        else:
            frac = np.arange(1, n_sub + 1) / n_sub
            sub = prices_30m[t - 1] + frac * (prices_30m[t] - prices_30m[t - 1])
        per_candle.append(sub)
    return per_candle


def make_flat_closes(n, price, n_sub=6):
    return [np.full(n_sub, price) for _ in range(n)]


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def constant_prices():
    n = 100
    price = 100.0
    closes = make_flat_closes(n, price)
    return closes, price, n


@pytest.fixture
def linear_prices():
    """Prix qui monte linéairement de 100 à 120 sur 100 bougies."""
    n = 100
    prices = np.linspace(100.0, 120.0, n)
    closes = make_closes_from_prices_30m(prices)
    return closes, prices, n


# ============================================================================
# TESTS
# ============================================================================

class TestNoSignal:
    """slopes tous NaN ou sous threshold → 0 trades, 0 PnL."""

    def test_all_nan_slopes(self, constant_prices):
        closes, _, n = constant_prices
        slopes = np.full(n, np.nan)
        res = backtest_5m(slopes, closes, k_substep=6, start=0, end=n,
                           fees=0.001, threshold=0.0)
        assert res['trades'] == 0
        assert res['pnl_pct'] == 0.0
        assert res['win_rate'] == 0.0

    def test_slopes_below_threshold(self, constant_prices):
        closes, _, n = constant_prices
        slopes = np.full(n, 0.01)  # petites slopes positives
        res = backtest_5m(slopes, closes, k_substep=6, start=0, end=n,
                           fees=0.0, threshold=0.1)
        # Toutes les slopes < threshold → 0 trades
        assert res['trades'] == 0


class TestSingleTradeConstantPrice:
    """Prix constant + slope positive constante → 1 trade LONG, PnL = -2*fees."""

    def test_single_long_trade_zero_fees(self, constant_prices):
        closes, price, n = constant_prices
        slopes = np.full(n, 1.0)  # slope positive constante
        res = backtest_5m(slopes, closes, k_substep=6, start=0, end=n,
                           fees=0.0, threshold=0.0)
        assert res['trades'] == 1, f"Expected 1 trade (constant slope), got {res['trades']}"
        # Prix constant → 0% gain, 0 fees → PnL = 0
        assert abs(res['pnl_pct']) < 1e-9, f"PnL with 0 fees on flat price = {res['pnl_pct']:.6f}"

    def test_single_long_trade_with_fees(self, constant_prices):
        closes, price, n = constant_prices
        slopes = np.full(n, 1.0)
        fees = 0.001
        res = backtest_5m(slopes, closes, k_substep=6, start=0, end=n,
                           fees=fees, threshold=0.0)
        assert res['trades'] == 1
        # 1 trade = 2*fees (ouverture + fermeture)
        expected_pnl_pct = -2 * fees * 100
        assert abs(res['pnl_pct'] - expected_pnl_pct) < 1e-6, \
            f"Expected {expected_pnl_pct}, got {res['pnl_pct']}"


class TestPnLAnalyticalLinear:
    """Prix linéaire + slope LONG correcte → PnL analytique."""

    def test_long_on_rising_prices_zero_fees(self, linear_prices):
        """Prix monte de 100 à 120. LONG dès le début: PnL = (120-100)/100 = 20%."""
        closes, prices, n = linear_prices
        slopes = np.full(n, 1.0)  # LONG tout le temps
        res = backtest_5m(slopes, closes, k_substep=6, start=0, end=n,
                           fees=0.0, threshold=0.0)
        assert res['trades'] == 1
        # Entry: closes[1][5] = prices[1]
        # Exit final: closes[n-1][-1] = prices[n-1]
        entry = prices[1]
        exit_price = prices[n - 1]
        expected_pnl = (exit_price - entry) / entry * 100
        assert abs(res['pnl_pct'] - expected_pnl) < 1e-6, \
            f"Expected {expected_pnl:.4f}%, got {res['pnl_pct']:.4f}%"

    def test_short_on_rising_prices(self, linear_prices):
        """Prix monte, SHORT → PnL négatif."""
        closes, prices, n = linear_prices
        slopes = np.full(n, -1.0)  # SHORT tout le temps
        res = backtest_5m(slopes, closes, k_substep=6, start=0, end=n,
                           fees=0.0, threshold=0.0)
        assert res['trades'] == 1
        entry = prices[1]
        exit_price = prices[n - 1]
        # SHORT: PnL = (entry - exit) / entry
        expected_pnl = (entry - exit_price) / entry * 100
        assert abs(res['pnl_pct'] - expected_pnl) < 1e-6


class TestFeesLinearity:
    """pnl(fees=f) - pnl(fees=0) = -2 * f * n_trades * 100 (en %)."""

    def test_fees_linearity(self, linear_prices):
        closes, prices, n = linear_prices
        # Slopes alternées pour générer beaucoup de trades
        slopes = np.where(np.arange(n) % 4 < 2, 1.0, -1.0).astype(float)

        res_0 = backtest_5m(slopes, closes, k_substep=6, start=0, end=n,
                             fees=0.0, threshold=0.0)
        f = 0.001
        res_f = backtest_5m(slopes, closes, k_substep=6, start=0, end=n,
                             fees=f, threshold=0.0)

        # Les deux doivent avoir le même nombre de trades (les fees n'influencent pas les décisions)
        assert res_0['trades'] == res_f['trades'], \
            f"trades(0)={res_0['trades']}, trades(f)={res_f['trades']}"

        n_trades = res_0['trades']
        expected_diff = -2 * f * n_trades * 100
        actual_diff = res_f['pnl_pct'] - res_0['pnl_pct']
        assert abs(actual_diff - expected_diff) < 1e-6, \
            f"fee diff expected {expected_diff:.4f}%, got {actual_diff:.4f}%"


class TestCausality:
    """Anti-leakage : polluer slopes[T+...] ou closes[T+...] ne doit pas changer le PnL sur [0, T+1)."""

    def test_no_leak_from_future_slopes(self, linear_prices):
        """PnL sur [0, T+1) ne dépend pas de slopes[T+1, T+2, ...]."""
        closes, prices, n = linear_prices
        T = 50
        # Version A : slopes propres
        slopes_A = np.full(n, 1.0)
        # Version B : future slopes polluées (après T+1)
        slopes_B = slopes_A.copy()
        slopes_B[T + 2:] = -1.0  # pollution

        # Backtest limité à [0, T+1)
        # Mais attention : exit final utilise closes_5m_per_candle[end][-1]
        # Donc le résultat dépend du prix à l'instant end.
        res_A = backtest_5m(slopes_A, closes, k_substep=6, start=0, end=T + 1,
                             fees=0.0, threshold=0.0)
        res_B = backtest_5m(slopes_B, closes, k_substep=6, start=0, end=T + 1,
                             fees=0.0, threshold=0.0)
        # Le futur des slopes (après end=T+1) NE DOIT PAS changer le PnL
        assert res_A['trades'] == res_B['trades'], "Leakage: trades differ"
        assert abs(res_A['pnl_pct'] - res_B['pnl_pct']) < 1e-9, \
            f"Leakage: pnl_A={res_A['pnl_pct']}, pnl_B={res_B['pnl_pct']}"

    def test_no_leak_from_future_closes(self, constant_prices):
        """PnL sur [0, T+1) ne dépend pas de closes_5m_per_candle[T+2+]."""
        closes_A, price, n = constant_prices
        T = 50
        slopes = np.full(n, 1.0)
        # Polluer closes futurs au-delà de T+1 (closes[T+2..])
        closes_B = [c.copy() for c in closes_A]
        for tt in range(T + 2, n):
            closes_B[tt] = np.full(6, 9999.0)

        res_A = backtest_5m(slopes, closes_A, k_substep=6, start=0, end=T + 1,
                             fees=0.0, threshold=0.0)
        res_B = backtest_5m(slopes, closes_B, k_substep=6, start=0, end=T + 1,
                             fees=0.0, threshold=0.0)
        assert abs(res_A['pnl_pct'] - res_B['pnl_pct']) < 1e-9, \
            f"Leakage from closes future: pnl_A={res_A['pnl_pct']}, pnl_B={res_B['pnl_pct']}"
        assert res_A['trades'] == res_B['trades']


class TestExecPriceAlignment:
    """exec_price = closes_5m_per_candle[t+1][k-1]. Tester pour plusieurs k."""

    def test_exec_price_for_k6(self, linear_prices):
        """Avec k=6, exec = closes[t+1][5] = prices[t+1]."""
        closes, prices, n = linear_prices
        # NaN ailleurs pour ne pas déclencher d'autre trade (slopes=0 serait SHORT car 0>0 est False)
        slopes = np.full(n, np.nan)
        slopes[10] = 1.0  # trade déclenché à t=10
        end = 50
        res = backtest_5m(slopes, closes, k_substep=6, start=0, end=end,
                           fees=0.0, threshold=0.0)
        assert res['trades'] == 1
        # Entry à closes[11][5] = prices[11]
        # Exit final: last_candle = min(end, len-1) = min(50, 99) = 50, donc closes[50][-1] = prices[50]
        entry = prices[11]
        exit_price = closes[min(end, n - 1)][-1]
        expected_pnl = (exit_price - entry) / entry * 100
        assert abs(res['pnl_pct'] - expected_pnl) < 1e-6, \
            f"k=6 exec misalign: expected {expected_pnl:.4f}, got {res['pnl_pct']:.4f}"

    def test_exec_price_for_k3(self, linear_prices):
        """Avec k=3, exec = closes[t+1][2]."""
        closes, prices, n = linear_prices
        slopes = np.full(n, np.nan)
        slopes[10] = 1.0
        end = 50
        res = backtest_5m(slopes, closes, k_substep=3, start=0, end=end,
                           fees=0.0, threshold=0.0)
        assert res['trades'] == 1
        # Entry: closes[11][2] = prices[10] + 3/6*(prices[11]-prices[10])
        entry = closes[11][2]
        exit_price = closes[min(end, n - 1)][-1]
        expected_pnl = (exit_price - entry) / entry * 100
        assert abs(res['pnl_pct'] - expected_pnl) < 1e-6

    def test_exec_price_for_k1(self, linear_prices):
        """Avec k=1, exec = closes[t+1][0] = prices[t] + (1/6)(prices[t+1]-prices[t])."""
        closes, prices, n = linear_prices
        slopes = np.full(n, np.nan)
        slopes[10] = 1.0
        end = 50
        res = backtest_5m(slopes, closes, k_substep=1, start=0, end=end,
                           fees=0.0, threshold=0.0)
        assert res['trades'] == 1
        entry = closes[11][0]
        exit_price = closes[min(end, n - 1)][-1]
        expected_pnl = (exit_price - entry) / entry * 100
        assert abs(res['pnl_pct'] - expected_pnl) < 1e-6


class TestHoldingMin:
    """holding_min empêche switch avant N bougies."""

    def test_holding_min_blocks_switch(self, constant_prices):
        closes, _, n = constant_prices
        slopes = np.where(np.arange(n) % 2 == 0, 1.0, -1.0).astype(float)
        # Sans holding_min : trade à chaque bougie
        res_no_hold = backtest_5m(slopes, closes, k_substep=6, start=0, end=n,
                                   fees=0.0, threshold=0.0, holding_min=0)
        # Avec holding_min=5 : bien moins de trades
        res_hold = backtest_5m(slopes, closes, k_substep=6, start=0, end=n,
                                fees=0.0, threshold=0.0, holding_min=5)
        assert res_hold['trades'] < res_no_hold['trades'], \
            f"holding_min=5 should reduce trades: {res_hold['trades']} vs {res_no_hold['trades']}"


class TestStartEndBounds:
    """start/end bornes : trades seulement considérés dans [start, end)."""

    def test_start_offset(self, linear_prices):
        closes, prices, n = linear_prices
        slopes = np.full(n, 1.0)
        # Démarrer à t=20 au lieu de 0
        res = backtest_5m(slopes, closes, k_substep=6, start=20, end=n,
                           fees=0.0, threshold=0.0)
        # 1 seul trade ouvert à t=20
        assert res['trades'] == 1
        # Entry à closes[21][5] = prices[21]
        entry = prices[21]
        exit_price = prices[n - 1]
        expected_pnl = (exit_price - entry) / entry * 100
        assert abs(res['pnl_pct'] - expected_pnl) < 1e-6, \
            f"start=20: expected {expected_pnl:.4f}, got {res['pnl_pct']:.4f}"

    def test_end_before_last_candle(self, linear_prices):
        closes, prices, n = linear_prices
        slopes = np.full(n, 1.0)
        end = 60
        res = backtest_5m(slopes, closes, k_substep=6, start=0, end=end,
                           fees=0.0, threshold=0.0)
        # Exit final: closes[end][-1] = closes[60][-1] = prices[60]
        entry = prices[1]
        exit_price = closes[end][-1]
        expected_pnl = (exit_price - entry) / entry * 100
        assert abs(res['pnl_pct'] - expected_pnl) < 1e-6


class TestNanHandling:
    """NaN dans closes ou slopes → skip, pas de crash."""

    def test_nan_in_exec_price_skips_trade(self, constant_prices):
        closes, price, n = constant_prices
        closes = [c.copy() for c in closes]
        slopes = np.full(n, np.nan)
        slopes[10] = 1.0  # tentative de trade à t=10
        closes[11][5] = np.nan  # NaN au prix d'exécution pour k=6
        res = backtest_5m(slopes, closes, k_substep=6, start=0, end=50,
                           fees=0.0, threshold=0.0)
        # Le trade à t=10 doit être skipped
        assert res['trades'] == 0, \
            f"Trade should be skipped when exec_price is NaN, got {res['trades']} trades"


class TestWinRate:
    """Win rate = wins / trades * 100."""

    def test_win_rate_all_winning(self, linear_prices):
        """LONG sur prix montant → trade gagnant."""
        closes, prices, n = linear_prices
        slopes = np.full(n, 1.0)
        res = backtest_5m(slopes, closes, k_substep=6, start=0, end=n,
                           fees=0.0, threshold=0.0)
        assert res['trades'] == 1
        assert res['pnl_pct'] > 0
        assert res['win_rate'] == 100.0

    def test_win_rate_all_losing(self, linear_prices):
        """SHORT sur prix montant → trade perdant."""
        closes, prices, n = linear_prices
        slopes = np.full(n, -1.0)
        res = backtest_5m(slopes, closes, k_substep=6, start=0, end=n,
                           fees=0.0, threshold=0.0)
        assert res['trades'] == 1
        assert res['pnl_pct'] < 0
        assert res['win_rate'] == 0.0


class TestExitFinalAccounting:
    """Le dernier trade (exit final) doit compter dans n_trades et win_rate."""

    def test_last_open_position_is_closed(self, linear_prices):
        closes, prices, n = linear_prices
        # 1 seul signal à t=10, pas d'autre (NaN ailleurs pour skip)
        slopes = np.full(n, np.nan)
        slopes[10] = 1.0
        res = backtest_5m(slopes, closes, k_substep=6, start=0, end=n,
                           fees=0.0, threshold=0.0)
        # 1 trade compté, et comme prix monte → win
        assert res['trades'] == 1
        assert res['pnl_pct'] > 0
        assert res['win_rate'] == 100.0


class TestInvariantExecTrade:
    """_exec_trade primitive: vérifier la formule."""

    def test_long_pnl(self):
        # position=1 (LONG): pnl = (exit - entry) / entry - fees
        r = _exec_trade(1, entry_price=100.0, exec_price=110.0, fees=0.001)
        expected = (110 - 100) / 100 - 0.001
        assert abs(r - expected) < 1e-12

    def test_short_pnl(self):
        # position=-1 (SHORT): pnl = (entry - exit) / entry - fees
        r = _exec_trade(-1, entry_price=100.0, exec_price=90.0, fees=0.001)
        expected = (100 - 90) / 100 - 0.001
        assert abs(r - expected) < 1e-12

    def test_long_loss(self):
        r = _exec_trade(1, entry_price=100.0, exec_price=95.0, fees=0.0)
        assert abs(r - (-0.05)) < 1e-12

    def test_short_loss(self):
        r = _exec_trade(-1, entry_price=100.0, exec_price=105.0, fees=0.0)
        assert abs(r - (-0.05)) < 1e-12
