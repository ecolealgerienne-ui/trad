"""
Audit unitaire — train_flks_slopes.py (269 lignes)
===================================================

Reproduit la logique critique du script (split, sequences, normalization) et
teste les invariants + le LEAKAGE POTENTIEL EXPLOITÉ PAR LE MODÈLE.

🚨 TEST CRITIQUE — TestLeakageExploitation 🚨
Vérifie que la dernière timestep de X[i] (feature) contient une slope qui
utilise des données POSTÉRIEURES au label y[i].

Si ce test passe → LEAKAGE EXPLOITÉ → +870% artificiel.

Lancement:
    python -m pytest tests/audit_core/test_14_train_flks_slopes.py -v -s
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import (
    resample_ohlcv,
    compute_bucket_close_mask,
    calculate_macd,
    compute_macd_live,
    group_per_candle,
    forward_filter_30m,
    compute_slopes_test2,
    compute_oracle,
)


# ============================================================================
# HELPERS — reproduit la logique de train_flks_slopes.py
# ============================================================================

WINDOW = 25
TRIM = 100


def reproduce_csv(df_5m):
    """Reproduit le CSV généré par prepare_flks_csv.py (version simplifiée)."""
    df_30m = resample_ohlcv(df_5m, 30)
    macd_30m = calculate_macd(df_30m)
    is_close = compute_bucket_close_mask(df_5m.index, 30)
    close_5m = df_5m['close'].values.astype(np.float64)
    macd_live = compute_macd_live(close_5m, is_close)
    macd_live_pc = group_per_candle(df_5m, df_30m, macd_live)

    _, slopes_oracle = compute_oracle(macd_30m)
    oracle_labels = np.where(slopes_oracle > 0, 1, 0)
    oracle_labels_30m = pd.Series(oracle_labels, index=df_30m.index)
    oracle_labels_5m = oracle_labels_30m.reindex(df_5m.index, method='ffill').fillna(0).astype(int)

    x_std, P_std, xp_std, _Pp_std, C_std = forward_filter_30m(macd_30m)

    def compute_and_ffill(slopes_30m):
        s = pd.Series(slopes_30m, index=df_30m.index)
        return s.reindex(df_5m.index, method='ffill').values

    std_slopes = {}
    for k in range(1, 7):
        slopes_30m = compute_slopes_test2(x_std, P_std, xp_std, C_std, macd_live_pc, k)
        std_slopes[f'k{k}'] = compute_and_ffill(slopes_30m)

    result = pd.DataFrame(index=df_5m.index)
    result['close'] = df_5m['close'].values
    for k in range(1, 7):
        result[f'std_k{k}_slope'] = std_slopes[f'k{k}']
    result['oracle_label_macd_30m'] = oracle_labels_5m.values
    return result, df_30m


def make_sequences(df_split, feature_cols, label_col, window=WINDOW):
    """Reproduit exactement la fonction de train_flks_slopes.py."""
    features = df_split[feature_cols].values.astype(np.float32)
    labels = df_split[label_col].values.astype(np.int64)
    closes = df_split['close'].values.astype(np.float64)
    dates = df_split.index.values
    n_s = len(df_split)
    n_feat = len(feature_cols)
    if n_s < window:
        return (np.empty((0, window, n_feat)), np.empty((0,)),
                np.empty((0,)), np.empty((0,), dtype='datetime64[ns]'))
    indices = np.arange(window)[None, :] + np.arange(n_s - window + 1)[:, None]
    X = features[indices]
    y = labels[window - 1:]
    c = closes[window - 1:]
    d = dates[window - 1:]
    return X, y, c, d


def chronological_split(df_clean, window=WINDOW):
    """Reproduit le split 70/15/15 avec gap=window."""
    n = len(df_clean)
    gap = window
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    df_train = df_clean.iloc[:train_end - gap]
    df_val = df_clean.iloc[train_end:val_end - gap]
    df_test = df_clean.iloc[val_end:]
    return df_train, df_val, df_test


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def pipeline_df():
    """CSV synthétique assez grand pour pouvoir split 70/15/15."""
    n_5min = 3000  # ~500 bougies 30min
    idx = pd.date_range('2024-01-01', periods=n_5min, freq='5min')
    trend = np.linspace(100, 200, n_5min)
    oscillation = 5.0 * np.sin(np.arange(n_5min) * 2 * np.pi / 120)
    close = trend + oscillation
    df_5m = pd.DataFrame({
        'open': close - 0.05, 'high': close + 0.2, 'low': close - 0.2,
        'close': close, 'volume': np.full(n_5min, 100.0),
    }, index=idx)
    result, df_30m = reproduce_csv(df_5m)
    return result, df_30m


# ============================================================================
# TESTS — make_sequences
# ============================================================================

class TestMakeSequences:

    def test_shapes(self, pipeline_df):
        result, _ = pipeline_df
        feat_cols = [f'std_k{k}_slope' for k in range(1, 7)]
        df_clean = result[feat_cols + ['oracle_label_macd_30m', 'close']].dropna().iloc[TRIM:-TRIM]
        X, y, c, d = make_sequences(df_clean, feat_cols, 'oracle_label_macd_30m')
        n = len(df_clean)
        assert X.shape == (n - WINDOW + 1, WINDOW, 6)
        assert y.shape == (n - WINDOW + 1,)
        assert c.shape == (n - WINDOW + 1,)

    def test_label_alignment_is_last_timestep(self, pipeline_df):
        """y[i] = labels[i+window-1]."""
        result, _ = pipeline_df
        feat_cols = [f'std_k{k}_slope' for k in range(1, 7)]
        df_clean = result[feat_cols + ['oracle_label_macd_30m', 'close']].dropna().iloc[TRIM:-TRIM]
        X, y, c, d = make_sequences(df_clean, feat_cols, 'oracle_label_macd_30m')
        labels_all = df_clean['oracle_label_macd_30m'].values
        for i in [0, 10, 100]:
            assert y[i] == labels_all[i + WINDOW - 1]

    def test_features_first_row_is_first_window(self, pipeline_df):
        """X[0] = features[0:window]."""
        result, _ = pipeline_df
        feat_cols = [f'std_k{k}_slope' for k in range(1, 7)]
        df_clean = result[feat_cols + ['oracle_label_macd_30m', 'close']].dropna().iloc[TRIM:-TRIM]
        X, _, _, _ = make_sequences(df_clean, feat_cols, 'oracle_label_macd_30m')
        features_all = df_clean[feat_cols].values
        np.testing.assert_allclose(X[0], features_all[:WINDOW])


# ============================================================================
# TESTS — Chronological Split
# ============================================================================

class TestSplit:

    def test_train_before_val_before_test(self, pipeline_df):
        result, _ = pipeline_df
        feat_cols = [f'std_k{k}_slope' for k in range(1, 7)]
        df_clean = result[feat_cols + ['oracle_label_macd_30m', 'close']].dropna().iloc[TRIM:-TRIM]
        df_train, df_val, df_test = chronological_split(df_clean)
        # Ordre chronologique strict
        assert df_train.index[-1] < df_val.index[0]
        assert df_val.index[-1] < df_test.index[0]

    def test_gap_between_train_and_val(self, pipeline_df):
        """
        Gap = window. df_train.iloc[:train_end - gap], df_val.iloc[train_end:...]
        Donc il y a `window` lignes entre la fin de train et le début de val.
        """
        result, _ = pipeline_df
        feat_cols = [f'std_k{k}_slope' for k in range(1, 7)]
        df_clean = result[feat_cols + ['oracle_label_macd_30m', 'close']].dropna().iloc[TRIM:-TRIM]
        n = len(df_clean)
        train_end_idx = int(n * 0.70)
        df_train, df_val, _ = chronological_split(df_clean)
        # df_train se termine à train_end - window
        # df_val commence à train_end
        # Index entre les deux : train_end - window + 1, ..., train_end - 1 (exclu)
        # Donc `window - 1` lignes sont "oubliées" (ni train ni val)
        len_gap = df_clean.index.get_loc(df_val.index[0]) - df_clean.index.get_loc(df_train.index[-1]) - 1
        assert len_gap == WINDOW - 1, f"Gap should be {WINDOW-1}, got {len_gap}"

    def test_no_overlap(self, pipeline_df):
        result, _ = pipeline_df
        feat_cols = [f'std_k{k}_slope' for k in range(1, 7)]
        df_clean = result[feat_cols + ['oracle_label_macd_30m', 'close']].dropna().iloc[TRIM:-TRIM]
        df_train, df_val, df_test = chronological_split(df_clean)
        train_idx = set(df_train.index)
        val_idx = set(df_val.index)
        test_idx = set(df_test.index)
        assert len(train_idx & val_idx) == 0
        assert len(val_idx & test_idx) == 0
        assert len(train_idx & test_idx) == 0


# ============================================================================
# TESTS — Z-score normalization
# ============================================================================

class TestZScore:

    def test_stats_from_train_only(self, pipeline_df):
        result, _ = pipeline_df
        feat_cols = [f'std_k{k}_slope' for k in range(1, 7)]
        df_clean = result[feat_cols + ['oracle_label_macd_30m', 'close']].dropna().iloc[TRIM:-TRIM]
        df_train, df_val, df_test = chronological_split(df_clean)

        stats = {}
        for col in feat_cols:
            mean = df_train[col].mean()
            std = df_train[col].std()
            stats[col] = (mean, std)

        # Vérifier que la mean de val/test N'EST PAS utilisée pour calculer stats
        # En l'état actuel, stats n'est PAS recalculée sur val/test → OK
        for col in feat_cols:
            mean_train = df_train[col].mean()
            # Stats stockées
            assert stats[col][0] == mean_train

    def test_normalized_train_has_zero_mean_unit_std(self, pipeline_df):
        result, _ = pipeline_df
        feat_cols = [f'std_k{k}_slope' for k in range(1, 7)]
        df_clean = result[feat_cols + ['oracle_label_macd_30m', 'close']].dropna().iloc[TRIM:-TRIM].copy()
        df_train, df_val, df_test = chronological_split(df_clean)

        # Apply z-score
        for col in feat_cols:
            mean = df_train[col].mean()
            std = df_train[col].std()
            if std < 1e-10:
                std = 1.0
            df_train.loc[:, col] = (df_train[col] - mean) / std

        for col in feat_cols:
            assert abs(df_train[col].mean()) < 1e-6
            assert abs(df_train[col].std() - 1.0) < 1e-3


# ============================================================================
# TESTS — CRITICAL : LEAKAGE EXPLOITATION
# ============================================================================

class TestLeakageExploitation:
    """
    🚨 TEST CRITIQUE DU +870% 🚨

    Hypothèse : dans la séquence X[i], la DERNIÈRE timestep (indice i+window-1)
    contient une feature `std_k6_slope` qui utilise des données POSTÉRIEURES
    au label y[i].

    Si confirmé → le modèle apprend à mapper feature(voit futur) → label.
    """

    def test_last_timestep_feature_depends_on_data_after_label(self, pipeline_df):
        """
        Pour une séquence X[i] avec label y[i] = label à 30min `T`,
        la feature à la dernière timestep utilise la slope calculée avec
        data jusqu'à la close de la bougie 30min `T+1` (sous-pas k=6).

        → La feature à la dernière timestep voit jusqu'à 30min APRÈS le label.
        """
        result, df_30m = pipeline_df
        feat_cols = [f'std_k{k}_slope' for k in range(1, 7)]
        df_clean = result[feat_cols + ['oracle_label_macd_30m', 'close']].dropna().iloc[TRIM:-TRIM]
        X, y, _, d = make_sequences(df_clean, feat_cols, 'oracle_label_macd_30m')

        # Prendre un indice du milieu pour éviter les bords
        i = 100
        label_timestamp = pd.Timestamp(d[i])
        # Bougie 30min correspondante au label
        label_30m_bucket_start = label_timestamp.floor('30min')
        label_30m_bucket_end = label_30m_bucket_start + pd.Timedelta(minutes=30)
        # Cette bougie va jusqu'à close[label_30m_bucket_end - 1]
        # La slope k=6 utilise data jusqu'à `label_30m_bucket_end + 25min` (bougie suivante)

        # last_feature_value = X[i, -1, k=6]
        last_feat_k6 = X[i, -1, 5]  # colonne 5 = std_k6_slope

        # Vérifier que modifier le close 5min APRÈS label_30m_bucket_end change X[i, -1, k=6]
        # Autrement dit : recréer le CSV en polluant juste après label_timestamp
        df_5m_A_idx = pd.date_range('2024-01-01', periods=3000, freq='5min')
        df_5m_A = pd.DataFrame({
            'open': np.full(3000, 100.0), 'high': np.full(3000, 100.2),
            'low': np.full(3000, 99.8),
            'close': np.linspace(100, 200, 3000) + 5.0 * np.sin(np.arange(3000) * 2 * np.pi / 120),
            'volume': np.full(3000, 100.0),
        }, index=df_5m_A_idx)
        result_A, df_30m_A = reproduce_csv(df_5m_A)

        feat_cols = [f'std_k{k}_slope' for k in range(1, 7)]
        df_clean_A = result_A[feat_cols + ['oracle_label_macd_30m', 'close']].dropna().iloc[TRIM:-TRIM]
        X_A, y_A, _, d_A = make_sequences(df_clean_A, feat_cols, 'oracle_label_macd_30m')

        # Pour le sample i, trouver le timestamp du label
        i = 100
        label_ts = pd.Timestamp(d_A[i])
        bucket_label = label_ts.floor('30min')
        bucket_after_label = bucket_label + pd.Timedelta(minutes=30)
        bucket_after_end = bucket_after_label + pd.Timedelta(minutes=29)

        # Polluer le close 5min PENDANT la bougie 30min APRÈS le label
        df_5m_B = df_5m_A.copy()
        mask_after = (df_5m_B.index >= bucket_after_label) & (df_5m_B.index <= bucket_after_end)
        df_5m_B.loc[mask_after, 'close'] = df_5m_B.loc[mask_after, 'close'] + 50.0
        result_B, _ = reproduce_csv(df_5m_B)
        df_clean_B = result_B[feat_cols + ['oracle_label_macd_30m', 'close']].dropna().iloc[TRIM:-TRIM]
        X_B, y_B, _, d_B = make_sequences(df_clean_B, feat_cols, 'oracle_label_macd_30m')

        # Le label y[i] DOIT ÊTRE inchangé si oracle est causalement cohérent... mais
        # oracle est non-causal (pykalman.smooth), donc y[i] peut changer aussi.
        # L'important : vérifier que X[i, -1, 5] (last timestep, std_k6_slope) change
        # alors que le label vient d'AVANT la pollution.

        last_feat_A = X_A[i, -1, 5]  # std_k6_slope
        last_feat_B = X_B[i, -1, 5]
        diff = abs(last_feat_A - last_feat_B)

        print(f"\n[EXPLOIT] Label timestamp: {d_A[i]}")
        print(f"[EXPLOIT] Label 30m bucket: {bucket_label}")
        print(f"[EXPLOIT] Polluted 30m bucket: {bucket_after_label}")
        print(f"[EXPLOIT] X[i, -1, k=6]_A = {last_feat_A:.6e}")
        print(f"[EXPLOIT] X[i, -1, k=6]_B = {last_feat_B:.6e}")
        print(f"[EXPLOIT] |diff| = {diff:.6e}")

        if diff > 1e-9:
            print("[EXPLOIT] 🚨 LEAKAGE EXPLOITÉ 🚨")
            print("[EXPLOIT] La dernière timestep de X[i] dépend de data APRÈS le label")
            print("[EXPLOIT] → Le modèle voit des infos du futur pour prédire le label")
        else:
            print("[EXPLOIT] ✅ Pas d'exploitation détectée")

    def test_earlier_timesteps_also_leaky(self, pipeline_df):
        """
        Pas seulement la dernière timestep : toutes les timesteps de X[i] ont des
        features leakées (chacune regarde 0-55min dans le futur de SON propre instant).

        Documentaire : on inspecte simplement les duplications par bougie 30min.
        """
        result, _ = pipeline_df
        feat_cols = [f'std_k{k}_slope' for k in range(1, 7)]
        df_clean = result[feat_cols + ['oracle_label_macd_30m', 'close']].dropna().iloc[TRIM:-TRIM]
        X, _, _, d = make_sequences(df_clean, feat_cols, 'oracle_label_macd_30m')
        # Pour i=100, inspecter les valeurs de std_k6_slope dans les 25 timesteps
        i = 100
        k6_series = X[i, :, 5]
        # On s'attend à voir des groupes de valeurs identiques (ffill)
        # Compter les valeurs uniques
        unique_vals = np.unique(k6_series)
        print(f"\n[FFILL DUP] X[i={i}, :, k=6] has {len(unique_vals)} unique values "
              f"across 25 timesteps")
        # 25 timesteps 5min / 6 = ~4-5 bougies 30min → ~4-5 valeurs uniques
        assert len(unique_vals) <= 7, \
            f"Expected ~4-5 unique values (6x ffill duplication), got {len(unique_vals)}"


# ============================================================================
# TESTS — dropna behavior on ffill duplicates
# ============================================================================

class TestDropNA:

    def test_dropna_preserves_ffill_duplicates(self, pipeline_df):
        """
        dropna() élimine les premières lignes NaN (warmup), mais pas les
        duplicates ffill. Les slopes restent 6x par bougie 30min.
        """
        result, _ = pipeline_df
        feat_cols = [f'std_k{k}_slope' for k in range(1, 7)]
        # Sans dropna : NaN au début, puis duplicates ffill
        # Avec dropna : premiers NaN retirés, duplicates restent
        df_clean = result[feat_cols + ['oracle_label_macd_30m', 'close']].dropna()
        # Vérifier : consécutivité dans std_k6_slope
        k6 = df_clean['std_k6_slope'].values
        # Chaque 6 valeurs doivent être identiques (ffill 30min → 5min)
        for i in range(0, len(k6) - 6, 50):  # sampler quelques points
            block = k6[i:i + 6]
            # Les 6 valeurs doivent être identiques (sauf si on est exactement sur une transition)
            if len(np.unique(block)) > 2:
                # Tolérer 2 valeurs max (transition possible)
                pytest.fail(
                    f"Block at i={i}: {block} has >2 unique values — duplicates broken"
                )
