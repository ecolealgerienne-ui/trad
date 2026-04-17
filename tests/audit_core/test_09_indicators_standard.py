"""
Audit unitaire — indicateurs et loaders standards (core.py:48-113)
===================================================================

Fonctions triviales groupées en 1 fichier :
- calculate_macd, calculate_rsi, calculate_cci : indicateurs standards
- resample_ohlcv : resample pandas OHLCV
- load_csv : lecture CSV avec auto-detection colonne date

Lancement:
    python -m pytest tests/audit_core/test_09_indicators_standard.py -v -s
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import (
    calculate_macd,
    calculate_rsi,
    calculate_cci,
    resample_ohlcv,
    load_csv,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    RSI_PERIOD,
    CCI_PERIOD,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def df_linear():
    """Prix montant de 100 à 200 sur 200 bougies."""
    n = 200
    idx = pd.date_range('2024-01-01', periods=n, freq='30min')
    close = np.linspace(100.0, 200.0, n)
    high = close + 0.5
    low = close - 0.5
    open_ = close - 0.1
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close,
        'volume': np.full(n, 100.0),
    }, index=idx)


@pytest.fixture
def df_constant():
    n = 100
    idx = pd.date_range('2024-01-01', periods=n, freq='30min')
    close = np.full(n, 100.0)
    return pd.DataFrame({
        'open': close, 'high': close, 'low': close, 'close': close,
        'volume': np.full(n, 100.0),
    }, index=idx)


@pytest.fixture
def df_5min_noisy():
    rng = np.random.default_rng(42)
    n = 200
    idx = pd.date_range('2024-01-01', periods=n, freq='5min')
    close = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
    return pd.DataFrame({
        'open': close, 'high': close + 0.5, 'low': close - 0.5, 'close': close,
        'volume': np.full(n, 100.0),
    }, index=idx)


# ============================================================================
# TESTS — calculate_macd
# ============================================================================

class TestMACD:

    def test_shape(self, df_linear):
        macd = calculate_macd(df_linear)
        assert macd.shape == (len(df_linear),)
        assert macd.dtype == np.float64

    def test_macd_stabilizes_on_linear(self, df_linear):
        """Prix linéaire → MACD tend vers valeur stable après warmup."""
        macd = calculate_macd(df_linear)
        # Après warmup (>50 bougies), std doit être faible relative à mean
        tail = macd[100:]
        std = np.std(tail)
        mean = np.mean(tail)
        assert abs(mean) > 0.01, f"MACD on rising prices should be positive, got mean={mean}"
        assert std / abs(mean) < 0.3, f"MACD should stabilize, got std/mean = {std/abs(mean):.3f}"

    def test_macd_zero_on_constant(self, df_constant):
        """Prix constant → MACD = 0."""
        macd = calculate_macd(df_constant)
        # EMA fast = EMA slow = close → MACD line = 0 → histogram = 0
        max_abs = np.max(np.abs(macd))
        assert max_abs < 1e-10, f"MACD on constant: max |macd| = {max_abs}"

    def test_macd_matches_manual_ewm(self, df_linear):
        """Vérifier que MACD = EMA_fast - EMA_slow - signal(EMA_fast-EMA_slow)."""
        close = df_linear['close']
        ef = close.ewm(span=MACD_FAST, adjust=False).mean()
        es = close.ewm(span=MACD_SLOW, adjust=False).mean()
        line = ef - es
        sig = line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        expected = (line - sig).values.astype(np.float64)
        actual = calculate_macd(df_linear)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)


# ============================================================================
# TESTS — calculate_rsi
# ============================================================================

class TestRSI:

    def test_shape(self, df_linear):
        rsi = calculate_rsi(df_linear)
        assert rsi.shape == (len(df_linear),)

    def test_rsi_approaches_100_on_rising(self, df_linear):
        """Prix strictement croissant → RSI → 100."""
        rsi = calculate_rsi(df_linear)
        # Warmup RSI = RSI_PERIOD (14). Après, RSI doit être proche de 100.
        tail = rsi[50:]
        # Prix strictement croissant → gain > 0 partout, loss = 0 → RSI = 100
        tail_clean = tail[~np.isnan(tail)]
        assert np.all(tail_clean > 95), f"RSI on rising: min={np.min(tail_clean)}"

    def test_rsi_approaches_0_on_falling(self):
        """Prix strictement décroissant → RSI → 0."""
        n = 200
        idx = pd.date_range('2024-01-01', periods=n, freq='30min')
        close = np.linspace(200.0, 100.0, n)
        df = pd.DataFrame({
            'open': close, 'high': close + 0.1, 'low': close - 0.1,
            'close': close, 'volume': np.full(n, 100.0),
        }, index=idx)
        rsi = calculate_rsi(df)
        tail = rsi[50:]
        tail_clean = tail[~np.isnan(tail)]
        assert np.all(tail_clean < 5), f"RSI on falling: max={np.max(tail_clean)}"

    def test_rsi_bounded_0_100(self, df_linear):
        """RSI doit être dans [0, 100]."""
        rsi = calculate_rsi(df_linear)
        clean = rsi[~np.isnan(rsi)]
        assert np.all(clean >= 0)
        assert np.all(clean <= 100)


# ============================================================================
# TESTS — calculate_cci
# ============================================================================

class TestCCI:

    def test_shape(self, df_linear):
        cci = calculate_cci(df_linear)
        assert cci.shape == (len(df_linear),)

    def test_cci_warmup_nan(self, df_linear):
        """Premières CCI_PERIOD-1 valeurs = NaN (rolling.mean warmup)."""
        cci = calculate_cci(df_linear)
        # Les CCI_PERIOD-1 premières valeurs sont NaN
        assert np.all(np.isnan(cci[:CCI_PERIOD - 1]))

    def test_cci_nan_on_constant(self, df_constant):
        """
        Sur signal constant, tp = const, sma = const, mad = 0.
        → CCI = (tp - sma) / (0.015 * 0) = 0 / 0 = NaN (division par zero).
        """
        cci = calculate_cci(df_constant)
        # Après warmup, mad = 0 → division → NaN
        tail = cci[50:]
        # Tous NaN ou tous ≈ 0 (selon impl. pandas de apply)
        # Acceptable : soit NaN (division 0/0) soit 0 (numerateur 0)
        non_nan = tail[~np.isnan(tail)]
        if len(non_nan) > 0:
            assert np.max(np.abs(non_nan)) < 1e-6


# ============================================================================
# TESTS — resample_ohlcv
# ============================================================================

class TestResampleOHLCV:

    def test_shape_6x_reduction(self, df_5min_noisy):
        """200 bougies 5min → ~34 bougies 30min (/6 avec arrondis)."""
        df_30m = resample_ohlcv(df_5min_noisy, 30)
        # 200/6 = 33.3 → min 33, max 34 buckets
        assert 32 <= len(df_30m) <= 35

    def test_open_is_first(self, df_5min_noisy):
        df_30m = resample_ohlcv(df_5min_noisy, 30)
        # open du 1er bucket 30min = open de la 1ère 5min
        ts_first = df_30m.index[0]
        mask_first = (df_5min_noisy.index >= ts_first) & \
                     (df_5min_noisy.index < ts_first + pd.Timedelta(minutes=30))
        expected_open = df_5min_noisy.loc[mask_first, 'open'].iloc[0]
        assert df_30m.iloc[0]['open'] == expected_open

    def test_close_is_last(self, df_5min_noisy):
        df_30m = resample_ohlcv(df_5min_noisy, 30)
        ts_first = df_30m.index[0]
        mask_first = (df_5min_noisy.index >= ts_first) & \
                     (df_5min_noisy.index < ts_first + pd.Timedelta(minutes=30))
        expected_close = df_5min_noisy.loc[mask_first, 'close'].iloc[-1]
        assert df_30m.iloc[0]['close'] == expected_close

    def test_high_is_max(self, df_5min_noisy):
        df_30m = resample_ohlcv(df_5min_noisy, 30)
        ts_first = df_30m.index[0]
        mask_first = (df_5min_noisy.index >= ts_first) & \
                     (df_5min_noisy.index < ts_first + pd.Timedelta(minutes=30))
        expected_high = df_5min_noisy.loc[mask_first, 'high'].max()
        assert df_30m.iloc[0]['high'] == expected_high

    def test_low_is_min(self, df_5min_noisy):
        df_30m = resample_ohlcv(df_5min_noisy, 30)
        ts_first = df_30m.index[0]
        mask_first = (df_5min_noisy.index >= ts_first) & \
                     (df_5min_noisy.index < ts_first + pd.Timedelta(minutes=30))
        expected_low = df_5min_noisy.loc[mask_first, 'low'].min()
        assert df_30m.iloc[0]['low'] == expected_low

    def test_volume_is_sum(self, df_5min_noisy):
        df_30m = resample_ohlcv(df_5min_noisy, 30)
        ts_first = df_30m.index[0]
        mask_first = (df_5min_noisy.index >= ts_first) & \
                     (df_5min_noisy.index < ts_first + pd.Timedelta(minutes=30))
        expected_vol = df_5min_noisy.loc[mask_first, 'volume'].sum()
        assert df_30m.iloc[0]['volume'] == expected_vol

    def test_dropna_removes_empty_buckets(self, df_5min_noisy):
        """resample.agg().dropna() — pas de NaN dans le résultat."""
        df_30m = resample_ohlcv(df_5min_noisy, 30)
        assert not df_30m.isna().any().any()


# ============================================================================
# TESTS — load_csv
# ============================================================================

class TestLoadCSV:

    def test_reads_file(self, tmp_path):
        """Lit un CSV simple avec colonne 'datetime'."""
        csv_path = tmp_path / "test.csv"
        df_in = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5, freq='30min'),
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [101.0, 102.0, 103.0, 104.0, 105.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [10.0] * 5,
        })
        df_in.to_csv(csv_path, index=False)
        df_out = load_csv(csv_path)
        assert len(df_out) == 5
        assert 'close' in df_out.columns

    def test_auto_detects_date_col(self, tmp_path):
        """Détecte 'date', 'datetime', 'time', 'timestamp', etc."""
        for col_name in ['date', 'datetime', 'time', 'timestamp']:
            csv_path = tmp_path / f"test_{col_name}.csv"
            df_in = pd.DataFrame({
                col_name: pd.date_range('2024-01-01', periods=3, freq='30min'),
                'close': [100.0, 101.0, 102.0],
            })
            df_in.to_csv(csv_path, index=False)
            df_out = load_csv(csv_path)
            assert df_out.index.name == 'datetime'

    def test_no_date_col_raises(self, tmp_path):
        """Si aucune colonne date détectable, raise ValueError."""
        csv_path = tmp_path / "bad.csv"
        df_in = pd.DataFrame({'foo': [1, 2, 3], 'bar': [4, 5, 6]})
        df_in.to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="No date column"):
            load_csv(csv_path)

    def test_columns_are_lowercase(self, tmp_path):
        """load_csv met toutes les colonnes en minuscules."""
        csv_path = tmp_path / "upper.csv"
        df_in = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=3, freq='30min'),
            'Open': [100.0, 101.0, 102.0],
            'Close': [100.5, 101.5, 102.5],
        })
        df_in.to_csv(csv_path, index=False)
        df_out = load_csv(csv_path)
        assert 'open' in df_out.columns
        assert 'close' in df_out.columns
        assert 'Open' not in df_out.columns

    def test_sorted_by_index(self, tmp_path):
        """load_csv retourne un DataFrame trié par index."""
        csv_path = tmp_path / "unsorted.csv"
        # Index dans l'ordre inverse
        df_in = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5, freq='30min')[::-1],
            'close': [100.0, 101.0, 102.0, 103.0, 104.0],
        })
        df_in.to_csv(csv_path, index=False)
        df_out = load_csv(csv_path)
        # Doit être trié (index croissant)
        assert df_out.index.is_monotonic_increasing


# ============================================================================
# TESTS — cohérence inter-fonctions
# ============================================================================

class TestCrossConsistency:

    def test_macd_and_compute_macd_live_agree_at_closes(self):
        """
        calculate_macd(df_30m) doit coïncider avec compute_macd_live
        aux closes 30min. Déjà testé en test_07, mais on re-vérifie ici.
        """
        # Import nécessaire
        from src.signal_processing.core import (
            compute_macd_live, compute_bucket_close_mask
        )
        rng = np.random.default_rng(42)
        n = 300
        idx = pd.date_range('2024-01-01', periods=n, freq='5min')
        close = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
        df_5m = pd.DataFrame({
            'open': close, 'high': close + 0.2, 'low': close - 0.2,
            'close': close, 'volume': np.full(n, 10.0),
        }, index=idx)
        df_30m = resample_ohlcv(df_5m, 30)
        macd_30m = calculate_macd(df_30m)
        mask = compute_bucket_close_mask(df_5m.index, 30)
        macd_live = compute_macd_live(df_5m['close'].values, mask)
        macd_live_at_closes = macd_live[mask]

        n_closes = min(len(macd_30m), len(macd_live_at_closes))
        # Après warmup (>26 closes)
        diff = np.abs(macd_30m[30:n_closes] - macd_live_at_closes[30:n_closes])
        assert np.max(diff) < 1e-8, f"MACD 30m ≠ MACD live at closes: max diff = {np.max(diff)}"
