"""
Audit unitaire — load_test_data (core.py:766-809)
==================================================

Fonction critique pour le backtest : charge NPZ prédictions + closes alignés.

Bugs historiques corrigés dans le passé:
  #9 - CSV source différent entre training et backtest (désalignement total)
  #10 - Closes non incluses dans NPZ (trades = 0)
  #11 - Discriminabilité 88% post-hoc ≠ filtre temps réel (biais oracle)

Deux conventions de clés NPZ supportées:
  - y_test / y_test_pred (nouveau, FLKS)
  - test_labels / test_preds (ancien)

Chemin idéal: NPZ contient 'test_closes' → pas de fallback CSV nécessaire.
Chemin fallback: CSV avec dropna() + [-n_test:] (SUSPECT si NaN dans close).

Lancement:
    python -m pytest tests/audit_core/test_08_load_test_data.py -v -s
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import src.signal_processing.core as core
from src.signal_processing.core import load_test_data


# ============================================================================
# HELPERS
# ============================================================================

def make_minimal_npz(path, y_test, y_pred_proba, closes=None, dates=None,
                      convention='new'):
    """
    Crée un NPZ minimal pour tester load_test_data.

    convention='new' → clés y_test, y_test_pred
    convention='old' → clés test_labels, test_preds
    """
    data = {}
    if convention == 'new':
        data['y_test'] = y_test
        data['y_test_pred'] = y_pred_proba
    else:
        data['test_labels'] = y_test
        data['test_preds'] = y_pred_proba
    if closes is not None:
        data['test_closes'] = closes
    if dates is not None:
        data['test_dates'] = dates
    np.savez(path, **data)


def make_minimal_features_csv(path, n_rows, close_values=None, include_nan=False):
    """Crée un CSV minimal features pour le fallback."""
    dates = pd.date_range('2024-01-01', periods=n_rows, freq='30min')
    if close_values is None:
        close_values = np.arange(n_rows, dtype=float) + 100.0
    df = pd.DataFrame({
        'datetime': dates,
        'close': close_values,
    })
    if include_nan:
        df.loc[df.index[:10], 'close'] = np.nan  # NaN au début
    df.to_csv(path, index=False)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def tmp_prepared_dir(tmp_path, monkeypatch):
    """Crée un dossier temporaire et redirige PREPARED_DATA_DIR."""
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    monkeypatch.setattr(core, 'PREPARED_DATA_DIR', str(prepared))
    return prepared


# ============================================================================
# TESTS — missing file
# ============================================================================

class TestMissingFile:

    def test_file_not_found_raises(self, tmp_prepared_dir):
        with pytest.raises(FileNotFoundError, match="NPZ not found"):
            load_test_data(indicator='macd', timeframe='30m')


# ============================================================================
# TESTS — key conventions
# ============================================================================

class TestKeyConventions:
    """Supporte deux jeux de clés NPZ: (y_test, y_test_pred) OU (test_labels, test_preds)."""

    def test_new_convention_y_test(self, tmp_prepared_dir):
        """Clés y_test, y_test_pred chargées correctement."""
        n = 20
        y_test = np.random.randint(0, 2, n)
        y_pred = np.random.rand(n)
        closes = np.arange(n, dtype=float) + 100.0
        make_minimal_npz(tmp_prepared_dir / "macd_30m_dataset.npz",
                          y_test, y_pred, closes=closes, convention='new')

        y, p, b, c, n_, src = load_test_data(indicator='macd')
        np.testing.assert_array_equal(y, y_test)
        np.testing.assert_array_equal(p, y_pred)
        assert n_ == n

    def test_old_convention_test_labels(self, tmp_prepared_dir):
        """Clés test_labels, test_preds chargées correctement."""
        n = 20
        y_test = np.random.randint(0, 2, n)
        y_pred = np.random.rand(n)
        closes = np.arange(n, dtype=float) + 100.0
        make_minimal_npz(tmp_prepared_dir / "macd_30m_dataset.npz",
                          y_test, y_pred, closes=closes, convention='old')

        y, p, b, c, n_, src = load_test_data(indicator='macd')
        np.testing.assert_array_equal(y, y_test)
        np.testing.assert_array_equal(p, y_pred)


# ============================================================================
# TESTS — closes embedded (ideal path)
# ============================================================================

class TestClosesEmbedded:
    """Si test_closes présent dans NPZ, pas de fallback CSV."""

    def test_source_indicates_npz(self, tmp_prepared_dir):
        n = 20
        y_test = np.random.randint(0, 2, n)
        y_pred = np.random.rand(n)
        closes = np.arange(n, dtype=float) + 100.0
        make_minimal_npz(tmp_prepared_dir / "macd_30m_dataset.npz",
                          y_test, y_pred, closes=closes)

        _, _, _, _, _, src = load_test_data(indicator='macd')
        assert "NPZ" in src, f"Source should mention NPZ, got: {src}"
        assert "CSV" not in src or "fallback" not in src.lower(), \
            f"Source should not indicate fallback: {src}"

    def test_closes_loaded_exactly(self, tmp_prepared_dir):
        """closes_test doit être identique à ce qui a été sauvé dans le NPZ."""
        n = 20
        y_test = np.random.randint(0, 2, n)
        y_pred = np.random.rand(n)
        closes = np.arange(n, dtype=float) + 100.0
        make_minimal_npz(tmp_prepared_dir / "macd_30m_dataset.npz",
                          y_test, y_pred, closes=closes)

        _, _, _, c, _, _ = load_test_data(indicator='macd')
        np.testing.assert_array_equal(c, closes)

    def test_no_csv_required_when_npz_has_closes(self, tmp_prepared_dir):
        """Pas de CSV dans tmp → si closes embedded, ne doit pas crasher."""
        n = 10
        y_test = np.random.randint(0, 2, n)
        y_pred = np.random.rand(n)
        closes = np.arange(n, dtype=float) + 50.0
        make_minimal_npz(tmp_prepared_dir / "macd_30m_dataset.npz",
                          y_test, y_pred, closes=closes)
        # Pas de CSV créé
        # load_test_data ne doit pas chercher le CSV
        y, p, b, c, n_, src = load_test_data(indicator='macd')
        assert n_ == n


# ============================================================================
# TESTS — fallback CSV
# ============================================================================

class TestFallbackCSV:
    """Si test_closes absent, fallback sur CSV → potentiel désalignement."""

    def test_fallback_works_with_clean_csv(self, tmp_prepared_dir):
        """CSV propre (pas de NaN) → fallback marche."""
        n_csv = 100  # CSV contient 100 lignes
        n_test = 20  # NPZ sans closes, n_test = 20
        make_minimal_features_csv(tmp_prepared_dir / "BTCUSD_flks_features.csv",
                                    n_csv, include_nan=False)
        y_test = np.random.randint(0, 2, n_test)
        y_pred = np.random.rand(n_test)
        # Pas de closes dans NPZ
        make_minimal_npz(tmp_prepared_dir / "macd_30m_dataset.npz",
                          y_test, y_pred, closes=None)

        y, p, b, c, n_, src = load_test_data(indicator='macd')
        assert "CSV" in src and "fallback" in src.lower(), \
            f"Source should indicate fallback: {src}"
        assert len(c) == n_test

    def test_fallback_uses_last_n_rows(self, tmp_prepared_dir):
        """Fallback prend les n_test DERNIÈRES lignes du CSV (suppose split chrono)."""
        n_csv = 100
        n_test = 20
        csv_closes = np.arange(n_csv, dtype=float) + 1000.0
        make_minimal_features_csv(tmp_prepared_dir / "BTCUSD_flks_features.csv",
                                    n_csv, close_values=csv_closes)
        y_test = np.random.randint(0, 2, n_test)
        y_pred = np.random.rand(n_test)
        make_minimal_npz(tmp_prepared_dir / "macd_30m_dataset.npz",
                          y_test, y_pred, closes=None)

        _, _, _, c, _, _ = load_test_data(indicator='macd')
        # Doit correspondre aux 20 dernières valeurs du CSV
        np.testing.assert_array_equal(c, csv_closes[-n_test:])

    def test_fallback_with_nan_in_close_may_misalign(self, tmp_prepared_dir):
        """
        BUG POTENTIEL : si le CSV contient des NaN dans `close`, dropna() les
        supprime, et [-n_test:] ne correspond plus aux dernières dates du CSV.

        Ce test documente le comportement actuel.
        """
        n_csv = 100
        n_test = 20
        csv_closes = np.arange(n_csv, dtype=float) + 1000.0
        make_minimal_features_csv(tmp_prepared_dir / "BTCUSD_flks_features.csv",
                                    n_csv, close_values=csv_closes.copy(),
                                    include_nan=True)  # 10 NaN au début
        y_test = np.random.randint(0, 2, n_test)
        y_pred = np.random.rand(n_test)
        make_minimal_npz(tmp_prepared_dir / "macd_30m_dataset.npz",
                          y_test, y_pred, closes=None)

        _, _, _, c, _, _ = load_test_data(indicator='macd')
        # Après dropna, le CSV a 90 lignes valides.
        # [-20:] prend les dernières 20, qui correspondent aux valeurs 1080..1099
        # (car les NaN étaient au début, donc ont été supprimées, pas d'impact ici).
        # Mais si les NaN étaient ailleurs, l'alignement casserait.
        expected = csv_closes[-n_test:]  # quand NaN au début, même résultat
        np.testing.assert_array_equal(c, expected)
        print(f"\n[FALLBACK NaN] NaN-leading CSV behaves OK: last {n_test} after dropna = original tail")


# ============================================================================
# TESTS — threshold et binarisation
# ============================================================================

class TestThreshold:

    def test_threshold_default_0_5(self, tmp_prepared_dir):
        n = 10
        y_test = np.zeros(n, dtype=int)
        y_pred = np.array([0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.55])
        closes = np.arange(n, dtype=float)
        make_minimal_npz(tmp_prepared_dir / "macd_30m_dataset.npz",
                          y_test, y_pred, closes=closes)

        _, _, b, _, _, _ = load_test_data(indicator='macd', threshold=0.5)
        # > 0.5 strictement
        expected = (y_pred > 0.5).astype(int)
        np.testing.assert_array_equal(b, expected)

    def test_threshold_custom(self, tmp_prepared_dir):
        n = 10
        y_test = np.zeros(n, dtype=int)
        y_pred = np.array([0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.55])
        closes = np.arange(n, dtype=float)
        make_minimal_npz(tmp_prepared_dir / "macd_30m_dataset.npz",
                          y_test, y_pred, closes=closes)

        _, _, b_low, _, _, _ = load_test_data(indicator='macd', threshold=0.3)
        _, _, b_high, _, _, _ = load_test_data(indicator='macd', threshold=0.7)
        assert b_low.sum() > b_high.sum(), \
            "Lower threshold should yield more 1s"


# ============================================================================
# TESTS — shape coherence
# ============================================================================

class TestShapes:

    def test_all_outputs_have_same_length(self, tmp_prepared_dir):
        n = 25
        y_test = np.random.randint(0, 2, n)
        y_pred = np.random.rand(n)
        closes = np.arange(n, dtype=float)
        make_minimal_npz(tmp_prepared_dir / "macd_30m_dataset.npz",
                          y_test, y_pred, closes=closes)

        y, p, b, c, n_, _ = load_test_data(indicator='macd')
        assert len(y) == n
        assert len(p) == n
        assert len(b) == n
        assert len(c) == n
        assert n_ == n

    def test_returns_tuple_of_6(self, tmp_prepared_dir):
        n = 10
        y_test = np.random.randint(0, 2, n)
        y_pred = np.random.rand(n)
        closes = np.arange(n, dtype=float)
        make_minimal_npz(tmp_prepared_dir / "macd_30m_dataset.npz",
                          y_test, y_pred, closes=closes)

        result = load_test_data(indicator='macd')
        assert len(result) == 6, f"Expected 6-tuple, got {len(result)}"


# ============================================================================
# TESTS — indicator/timeframe parameters
# ============================================================================

class TestIndicatorTimeframe:

    def test_indicator_selects_file(self, tmp_prepared_dir):
        n = 10
        y_test = np.zeros(n, dtype=int)
        y_pred = np.ones(n) * 0.5
        closes = np.arange(n, dtype=float)
        # Deux fichiers différents
        make_minimal_npz(tmp_prepared_dir / "macd_30m_dataset.npz",
                          y_test, y_pred, closes=closes)
        make_minimal_npz(tmp_prepared_dir / "rsi_30m_dataset.npz",
                          y_test, y_pred + 0.1, closes=closes + 1)

        _, p_macd, _, c_macd, _, _ = load_test_data(indicator='macd')
        _, p_rsi, _, c_rsi, _, _ = load_test_data(indicator='rsi')
        # Valeurs différentes selon l'indicateur
        assert p_macd[0] != p_rsi[0]
        assert c_macd[0] != c_rsi[0]

    def test_missing_indicator_raises(self, tmp_prepared_dir):
        with pytest.raises(FileNotFoundError):
            load_test_data(indicator='xyz_nonexistent')
