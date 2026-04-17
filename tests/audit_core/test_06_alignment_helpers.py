"""
Audit unitaire — helpers d'alignement 5min / 30min
===================================================

Trois fonctions critiques pour l'alignement temporel du pipeline causal:

1. compute_bucket_close_mask(index_5min, tf_minutes) : True à la dernière 5min de chaque bucket 30min
2. compute_live_ohlcv(df_5min, tf_minutes) : OHLC 5min "live" (cumulés sur bucket)
3. group_per_candle(df_5m, df_30m, array_5m) : regroupe les 5min dans les buckets 30min

Lancement:
    python -m pytest tests/audit_core/test_06_alignment_helpers.py -v -s
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import (
    compute_bucket_close_mask,
    compute_live_ohlcv,
    group_per_candle,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def regular_5min_index():
    """Index 5min régulier sur 2h (24 bougies = 4 buckets 30min)."""
    start = pd.Timestamp('2024-01-01 10:00')
    return pd.date_range(start, periods=24, freq='5min')


@pytest.fixture
def regular_df_5min(regular_5min_index):
    """OHLCV 5min synthétique régulier."""
    rng = np.random.default_rng(42)
    n = len(regular_5min_index)
    # Prix montant de 100 à 120
    close = np.linspace(100.0, 120.0, n)
    high = close + 0.5
    low = close - 0.5
    open_ = close - 0.2
    volume = rng.integers(100, 1000, n).astype(float)
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }, index=regular_5min_index)
    return df


@pytest.fixture
def df_30min_matching(regular_df_5min):
    """Resample 30min aligné sur df_5min."""
    return regular_df_5min.resample('30min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()


# ============================================================================
# TESTS — compute_bucket_close_mask
# ============================================================================

class TestBucketCloseMask:
    """Masque True à la dernière 5min de chaque bucket 30min."""

    def test_shape(self, regular_5min_index):
        mask = compute_bucket_close_mask(regular_5min_index, 30)
        assert len(mask) == len(regular_5min_index)
        assert mask.dtype == bool

    def test_true_only_on_last_5min_of_bucket(self, regular_5min_index):
        """
        Index régulier 10:00 → 11:55. Buckets 30min:
        - 10:00: bougies 10:00, 10:05, 10:10, 10:15, 10:20, 10:25
        - 10:30: bougies 10:30, 10:35, 10:40, 10:45, 10:50, 10:55
        - ...
        Le mask doit être True UNIQUEMENT aux 5e indices dans chaque bucket
        (10:25, 10:55, 11:25, 11:55).
        """
        mask = compute_bucket_close_mask(regular_5min_index, 30)
        # 24 bougies, 4 buckets de 6 bougies chacun → 4 True (aux positions 5, 11, 17, 23)
        expected_true_pos = [5, 11, 17, 23]
        assert mask.sum() == 4, f"Expected 4 True, got {mask.sum()}"
        for pos in expected_true_pos:
            assert mask[pos], f"Expected True at pos {pos} ({regular_5min_index[pos]})"
        # Toutes les autres positions doivent être False
        for pos in range(24):
            if pos not in expected_true_pos:
                assert not mask[pos], f"Expected False at pos {pos} ({regular_5min_index[pos]})"

    def test_timestamp_of_true_is_xx25_or_xx55(self, regular_5min_index):
        """Les True doivent tomber sur les timestamps xx:25 et xx:55."""
        mask = compute_bucket_close_mask(regular_5min_index, 30)
        true_timestamps = regular_5min_index[mask]
        for ts in true_timestamps:
            minute = ts.minute
            assert minute in [25, 55], f"True at non-boundary timestamp {ts}"

    def test_last_element_always_true(self):
        """Le dernier élément de l'index est toujours True (next_bucket = NaT)."""
        idx = pd.date_range('2024-01-01 10:00', periods=5, freq='5min')
        mask = compute_bucket_close_mask(idx, 30)
        # Tous dans le bucket 10:00, mais le dernier (10:20) est True car next=NaT
        assert mask[-1], "Last element should be True (NaT next_bucket)"

    def test_gap_in_index(self):
        """
        Avec un gap : [10:00, 10:05, 10:25, 10:30]. 10:05 est censé être
        la dernière 5min avant 10:25 mais 10:25 est dans le même bucket.
        L'implémentation actuelle ne voit pas de transition de bucket entre
        10:05 et 10:25, donc 10:05 est False.
        """
        idx = pd.DatetimeIndex(['2024-01-01 10:00', '2024-01-01 10:05',
                                 '2024-01-01 10:25', '2024-01-01 10:30'])
        mask = compute_bucket_close_mask(idx, 30)
        # [10:00, 10:05, 10:25] dans bucket 10:00. [10:30] dans bucket 10:30.
        # Transition 10:25 → 10:30 : bucket différent, donc 10:25 est True.
        # Dernier (10:30) : NaT → True.
        expected = [False, False, True, True]
        np.testing.assert_array_equal(list(mask), expected)


# ============================================================================
# TESTS — compute_live_ohlcv
# ============================================================================

class TestLiveOHLCV:
    """
    OHLC 5min live : à chaque 5min, donne open/high/low/close cumulés
    comme si on observait la bougie 30min en cours.
    """

    def test_shape(self, regular_df_5min):
        live = compute_live_ohlcv(regular_df_5min, 30)
        assert len(live) == len(regular_df_5min)
        assert set(live.columns) == {'open', 'high', 'low', 'close'}

    def test_open_is_first_of_bucket(self, regular_df_5min):
        """
        Pour toutes les 5min d'un bucket, live['open'] = open du 1er 5min du bucket.
        """
        live = compute_live_ohlcv(regular_df_5min, 30)
        bucket_0_open = regular_df_5min.iloc[0]['open']  # open du 1er 5min du 1er bucket
        # Les 6 premières 5min doivent avoir live['open'] = bucket_0_open
        for i in range(6):
            assert abs(live.iloc[i]['open'] - bucket_0_open) < 1e-12, \
                f"live['open'] at i={i}: got {live.iloc[i]['open']}, expected {bucket_0_open}"
        # 7e à 12e : bucket suivant, open = regular_df_5min.iloc[6]['open']
        bucket_1_open = regular_df_5min.iloc[6]['open']
        for i in range(6, 12):
            assert abs(live.iloc[i]['open'] - bucket_1_open) < 1e-12

    def test_close_is_5min_close(self, regular_df_5min):
        """live['close'] = close 5min courant (pas cumulé)."""
        live = compute_live_ohlcv(regular_df_5min, 30)
        np.testing.assert_array_equal(
            live['close'].values, regular_df_5min['close'].values
        )

    def test_high_is_cummax_within_bucket(self, regular_df_5min):
        """
        live['high'] = cummax depuis le début du bucket.
        Pour vérifier, on compute manuellement pour le 1er bucket (6 premières 5min).
        """
        live = compute_live_ohlcv(regular_df_5min, 30)
        # 1er bucket : high cumulatif sur les 6 premières 5min
        expected_high = np.maximum.accumulate(regular_df_5min['high'].iloc[:6].values)
        np.testing.assert_allclose(live['high'].iloc[:6].values, expected_high, rtol=1e-12)
        # 2e bucket : reset
        expected_high_b1 = np.maximum.accumulate(regular_df_5min['high'].iloc[6:12].values)
        np.testing.assert_allclose(live['high'].iloc[6:12].values, expected_high_b1, rtol=1e-12)

    def test_low_is_cummin_within_bucket(self, regular_df_5min):
        live = compute_live_ohlcv(regular_df_5min, 30)
        expected_low = np.minimum.accumulate(regular_df_5min['low'].iloc[:6].values)
        np.testing.assert_allclose(live['low'].iloc[:6].values, expected_low, rtol=1e-12)

    def test_high_is_causal(self, regular_df_5min):
        """
        live['high'][i] ne doit dépendre que de high[start_of_bucket..i], pas du futur.
        Test : modifier high à t=3 ne doit pas changer live['high'][0..2].
        """
        df_A = regular_df_5min.copy()
        df_B = regular_df_5min.copy()
        df_B.iloc[3, df_B.columns.get_loc('high')] = 9999.0  # pollution à t=3
        live_A = compute_live_ohlcv(df_A, 30)
        live_B = compute_live_ohlcv(df_B, 30)
        # live['high'][0..2] doit être identique
        np.testing.assert_array_equal(
            live_A['high'].iloc[:3].values, live_B['high'].iloc[:3].values,
            err_msg="LEAKAGE: live['high'][0..2] changed when high[3] polluted"
        )
        # live['high'][3..] peut différer (car on a changé high[3])
        assert live_A['high'].iloc[3] != live_B['high'].iloc[3]

    def test_low_is_causal(self, regular_df_5min):
        df_A = regular_df_5min.copy()
        df_B = regular_df_5min.copy()
        df_B.iloc[3, df_B.columns.get_loc('low')] = -9999.0
        live_A = compute_live_ohlcv(df_A, 30)
        live_B = compute_live_ohlcv(df_B, 30)
        np.testing.assert_array_equal(
            live_A['low'].iloc[:3].values, live_B['low'].iloc[:3].values
        )

    def test_live_close_equals_real_close_at_bucket_end(self, regular_df_5min, df_30min_matching):
        """
        À la dernière 5min du bucket (xx:25 ou xx:55), live values doivent
        correspondre à la vraie bougie 30min :
            live['open'] = df_30m['open'], live['close'] = df_30m['close'],
            live['high'] = df_30m['high'] (max de toutes les 5min du bucket),
            live['low'] = df_30m['low'] (min).
        """
        live = compute_live_ohlcv(regular_df_5min, 30)
        mask = compute_bucket_close_mask(regular_df_5min.index, 30)
        closing_live = live[mask]
        # Les closing_live.index doivent être alignés à df_30min + 25min
        # Vérifier valeurs open/high/low/close
        for i, (ts_close, row) in enumerate(closing_live.iterrows()):
            ts_30m = ts_close.floor('30min')
            if ts_30m in df_30min_matching.index:
                row_30m = df_30min_matching.loc[ts_30m]
                assert abs(row['open'] - row_30m['open']) < 1e-12, \
                    f"open mismatch at {ts_close}"
                assert abs(row['high'] - row_30m['high']) < 1e-12, \
                    f"high mismatch at {ts_close}"
                assert abs(row['low'] - row_30m['low']) < 1e-12
                assert abs(row['close'] - row_30m['close']) < 1e-12


# ============================================================================
# TESTS — group_per_candle
# ============================================================================

class TestGroupPerCandle:
    """
    Regroupe les valeurs 5min selon les buckets 30min.
    per_candle[t] = valeurs 5min pour lesquelles ts_30m[t] <= idx_5m <= ts_30m[t]+29:59
    """

    def test_shape(self, regular_df_5min, df_30min_matching):
        array_5m = regular_df_5min['close'].values
        per_candle = group_per_candle(regular_df_5min, df_30min_matching, array_5m)
        assert len(per_candle) == len(df_30min_matching), \
            f"Expected {len(df_30min_matching)} buckets, got {len(per_candle)}"

    def test_six_values_per_bucket_regular(self, regular_df_5min, df_30min_matching):
        """Index régulier → 6 valeurs 5min par bucket 30min."""
        array_5m = regular_df_5min['close'].values
        per_candle = group_per_candle(regular_df_5min, df_30min_matching, array_5m)
        for i, arr in enumerate(per_candle):
            assert len(arr) == 6, f"Bucket {i}: expected 6 values, got {len(arr)}"

    def test_last_value_is_bucket_close(self, regular_df_5min, df_30min_matching):
        """
        per_candle[t][-1] doit être la valeur 5min à ts_30m + 25min,
        = close 5min de la bougie 30min.
        """
        array_5m = regular_df_5min['close'].values
        per_candle = group_per_candle(regular_df_5min, df_30min_matching, array_5m)
        for t, ts_30m in enumerate(df_30min_matching.index):
            ts_expected = ts_30m + pd.Timedelta(minutes=25)
            idx_5m = regular_df_5min.index.get_loc(ts_expected)
            expected_val = array_5m[idx_5m]
            assert per_candle[t][-1] == expected_val

    def test_first_value_is_bucket_open(self, regular_df_5min, df_30min_matching):
        """per_candle[t][0] doit être la valeur 5min à ts_30m."""
        array_5m = regular_df_5min['close'].values
        per_candle = group_per_candle(regular_df_5min, df_30min_matching, array_5m)
        for t, ts_30m in enumerate(df_30min_matching.index):
            idx_5m = regular_df_5min.index.get_loc(ts_30m)
            expected_val = array_5m[idx_5m]
            assert per_candle[t][0] == expected_val

    def test_missing_5min_in_bucket(self):
        """
        Si une 5min manque dans un bucket, per_candle[t] a moins de 6 valeurs.
        """
        # Index avec un gap : 10:00, 10:05, 10:10, 10:20, 10:25 (10:15 manque)
        idx = pd.DatetimeIndex([
            '2024-01-01 10:00', '2024-01-01 10:05',
            '2024-01-01 10:10', '2024-01-01 10:20',
            '2024-01-01 10:25',
        ])
        df_5m = pd.DataFrame({'close': [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
        df_30m = pd.DataFrame({'close': [5.0]}, index=pd.DatetimeIndex(['2024-01-01 10:00']))
        per_candle = group_per_candle(df_5m, df_30m, df_5m['close'].values)
        # 5 valeurs (pas 6)
        assert len(per_candle[0]) == 5

    def test_array_values_preserved(self, regular_df_5min, df_30min_matching):
        """Les valeurs retournées correspondent exactement à array_5m."""
        array_5m = regular_df_5min['close'].values
        per_candle = group_per_candle(regular_df_5min, df_30min_matching, array_5m)
        # Reconstituer la concaténation
        reconstructed = np.concatenate(per_candle)
        # Doit être identique à array_5m (car couvre tout l'index 5m)
        np.testing.assert_array_equal(reconstructed, array_5m)

    def test_with_nan_array(self, regular_df_5min, df_30min_matching):
        """Préservation des NaN dans array_5m."""
        array_nan = regular_df_5min['close'].values.copy()
        array_nan[3] = np.nan
        per_candle = group_per_candle(regular_df_5min, df_30min_matching, array_nan)
        assert np.isnan(per_candle[0][3])


# ============================================================================
# TESTS — cohérence inter-fonctions
# ============================================================================

class TestCrossConsistency:
    """Vérifier la cohérence entre les 3 helpers."""

    def test_bucket_close_mask_count_matches_group_per_candle(
        self, regular_df_5min, df_30min_matching
    ):
        """Le nombre de True dans bucket_close_mask = nombre de buckets dans group_per_candle."""
        mask = compute_bucket_close_mask(regular_df_5min.index, 30)
        array_5m = regular_df_5min['close'].values
        per_candle = group_per_candle(regular_df_5min, df_30min_matching, array_5m)
        assert mask.sum() == len(per_candle), \
            f"mask True count {mask.sum()} != per_candle len {len(per_candle)}"

    def test_closing_values_match_last_of_per_candle(
        self, regular_df_5min, df_30min_matching
    ):
        """
        array_5m[bucket_close_mask] doit correspondre aux valeurs finales
        de chaque per_candle[t].
        """
        mask = compute_bucket_close_mask(regular_df_5min.index, 30)
        array_5m = regular_df_5min['close'].values
        closing_vals = array_5m[mask]
        per_candle = group_per_candle(regular_df_5min, df_30min_matching, array_5m)
        last_of_each = np.array([arr[-1] for arr in per_candle])
        np.testing.assert_array_equal(closing_vals, last_of_each)
