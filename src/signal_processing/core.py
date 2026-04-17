"""
Signal Processing Core — shared functions for all FLKS/AQ-KF scripts
=====================================================================

All data loading, indicator calculation, Kalman filters, slope computation,
metrics, and backtest functions in one place. No duplication.

Import with:
    from signal_processing.core import *
"""

import numpy as np
import pandas as pd
from collections import deque


# ============================================================================
# PARAMETERS (from pipeline: prepare_multitf_csv.py)
# ============================================================================

KALMAN_PROCESS_VAR = 0.01
KALMAN_MEASURE_VAR = 0.1

A = np.array([[1.0, 1.0],
              [0.0, 1.0]])
H = np.array([[1.0, 0.0]])
Q = np.eye(2) * KALMAN_PROCESS_VAR
R = np.array([[KALMAN_MEASURE_VAR]])

DT_SUB = 1.0 / 6.0
A_SUB = np.array([[1.0, DT_SUB],
                   [0.0, 1.0]])
Q_SUB = Q * DT_SUB

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
CCI_PERIOD = 20

FEES = 0.001


# ============================================================================
# DATA LOADING
# ============================================================================

def load_csv(path):
    df = pd.read_csv(path)
    date_col = None
    for col in ['date', 'datetime', 'time', 'timestamp', 'Date', 'Datetime']:
        if col in df.columns:
            date_col = col
            break
    if date_col is None:
        raise ValueError(f"No date column found in {path}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = 'datetime'
    df.columns = df.columns.str.lower()
    return df.sort_index()


def resample_ohlcv(df_5min, tf_minutes):
    return df_5min.resample(f'{tf_minutes}min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()


def compute_bucket_close_mask(index_5min, tf_minutes):
    bucket = index_5min.floor(f'{tf_minutes}min').values
    next_bucket = np.append(bucket[1:], np.datetime64('NaT'))
    return (bucket != next_bucket) | pd.isna(next_bucket)


def compute_live_ohlcv(df_5min, tf_minutes):
    group = df_5min.index.floor(f'{tf_minutes}min')
    r = pd.DataFrame(index=df_5min.index)
    r['open'] = df_5min.groupby(group)['open'].transform('first')
    r['high'] = df_5min.groupby(group)['high'].cummax()
    r['low'] = df_5min.groupby(group)['low'].cummin()
    r['close'] = df_5min['close']
    return r


# ============================================================================
# INDICATORS — Standard 30min
# ============================================================================

def calculate_macd(df, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    """MACD histogram. Périodes paramétrables (défauts = constantes globales)."""
    ema_f = df['close'].ewm(span=fast, adjust=False).mean()
    ema_s = df['close'].ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    return (line - sig).values.astype(np.float64)


def calculate_rsi(df, period=RSI_PERIOD):
    """RSI. Période paramétrable (défaut = constante globale)."""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    ag = gain.ewm(span=period, adjust=False).mean()
    al = loss.ewm(span=period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).values.astype(np.float64)


def calculate_cci(df, period=CCI_PERIOD):
    """CCI. Période paramétrable (défaut = constante globale)."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    return ((tp - sma) / (0.015 * mad)).values.astype(np.float64)


def compute_indicator(df, indicator, **kwargs):
    """
    Point d'entrée unique pour calculer un indicateur standard sur n'importe
    quel df OHLC (tout timeframe).

    Args:
        df: DataFrame avec colonnes 'open', 'high', 'low', 'close'
            (index = timestamps).
        indicator: 'macd' | 'rsi' | 'cci'.
        **kwargs: params optionnels passés à calculate_<indicator> :
          - MACD: fast, slow, signal
          - RSI:  period
          - CCI:  period

    Returns:
        pd.Series (float64) indexée par les timestamps de df, de même longueur.
        name = indicator. Les NaN (warm-up ou input NaN) sont remplacés par 0.
    """
    dispatch = {
        'macd': calculate_macd,
        'rsi': calculate_rsi,
        'cci': calculate_cci,
    }
    key = indicator.lower()
    if key not in dispatch:
        raise ValueError(
            f"Unknown indicator '{indicator}'. Expected one of: {list(dispatch)}"
        )
    values = dispatch[key](df, **kwargs)
    # Remplacer les NaN par 0 (warm-up et bords). Tous les indicateurs ont
    # le même N valide = len(df) après cette étape.
    values = np.where(np.isnan(values), 0.0, values)
    return pd.Series(values, index=df.index, name=key)


def compute_indicator_live(df_5m, is_close, indicator, tf_minutes, **kwargs):
    """
    Dispatcher live (frozen/provisional EMAs) pour calculer un indicateur
    en résolution 5min, figé à la close de chaque bougie TF.

    Args:
        df_5m: DataFrame OHLCV 5min (index = timestamps 5min).
        is_close: np.ndarray[bool] de longueur len(df_5m), True à la dernière
                  5min de chaque bucket TF (voir compute_bucket_close_mask).
        indicator: 'macd' | 'rsi' | 'cci'.
        tf_minutes: taille du bucket (30 ou 60). Utilisé par CCI pour calculer
                    high_live/low_live (cummax/cummin dans le bucket).
        **kwargs: params de l'indicateur (ex: fast/slow/signal pour MACD, period pour RSI/CCI).

    Returns:
        pd.Series (float64) indexée par les timestamps 5min de df_5m.
        name = f'{indicator}_live'. Les NaN sont remplacés par 0.
    """
    key = indicator.lower()
    close_5m = df_5m['close'].values.astype(np.float64)

    if key == 'macd':
        values = compute_macd_live(close_5m, is_close, **kwargs)
    elif key == 'rsi':
        values = compute_rsi_live(close_5m, is_close, **kwargs)
    elif key == 'cci':
        # CCI a besoin de high_live et low_live (cummax/cummin dans le bucket)
        live_ohlcv = compute_live_ohlcv(df_5m, tf_minutes)
        values = compute_cci_live(
            live_ohlcv['high'].values,
            live_ohlcv['low'].values,
            close_5m,
            is_close,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown indicator '{indicator}'. Expected: ['macd', 'rsi', 'cci']"
        )
    # Remplacer les NaN par 0 (cohérent avec compute_indicator standard)
    values = np.where(np.isnan(values), 0.0, values)
    return pd.Series(values, index=df_5m.index, name=f'{key}_live')


def compute_oracle_labels(df, indicator,
                           Q_var=KALMAN_PROCESS_VAR, R_var=KALMAN_MEASURE_VAR,
                           indicator_params=None):
    """
    Dispatcher oracle : calcule les labels non-causaux (smoother RTS).

    ATTENTION: SMOOTHER NON-CAUSAL. Labels de training uniquement.

    Args:
        df: DataFrame OHLC.
        indicator: 'macd' | 'rsi' | 'cci'.
        Q_var, R_var: covariances Kalman smoother (défauts = globales).
        indicator_params: dict optionnel de params pour l'indicateur
                          (ex: {'fast': 8, 'slow': 17, 'signal': 9}).

    Returns:
        pd.DataFrame (position, slope, label), NaN remplacés par 0.
    """
    ind_kwargs = indicator_params or {}
    ind_series = compute_indicator(df, indicator, **ind_kwargs)
    ind_array = ind_series.values.astype(np.float64)
    positions, slopes = compute_oracle(ind_array, Q_var=Q_var, R_var=R_var)
    positions = np.where(np.isnan(positions), 0.0, positions)
    slopes = np.where(np.isnan(slopes), 0.0, slopes)
    labels = (slopes > 0).astype(int)
    return pd.DataFrame({
        'position': positions,
        'slope': slopes,
        'label': labels,
    }, index=df.index)


def compute_forward_filter(df, indicator, adaptive=False,
                            Q_var=KALMAN_PROCESS_VAR, R_var=KALMAN_MEASURE_VAR,
                            indicator_params=None):
    """
    Dispatcher forward filter Kalman.

    Args:
        df: DataFrame OHLC.
        indicator: 'macd' | 'rsi' | 'cci'.
        adaptive: False = Standard Kalman (Q fixe), True = AQ-KF adaptatif.
        Q_var, R_var: covariances Kalman (défauts = globales).
        indicator_params: dict optionnel de params pour l'indicateur.

    Returns:
        dict avec 'state' (DataFrame), 'P_filt', 'P_pred', 'C' (ndarrays),
        'indicator' (Series).
    """
    ind_kwargs = indicator_params or {}
    ind_series = compute_indicator(df, indicator, **ind_kwargs)
    ind_array = ind_series.values.astype(np.float64)

    if adaptive:
        x_filt, P_filt, x_pred, P_pred, C = forward_filter_30m_adaptive(
            ind_array, Q_var=Q_var, R_var=R_var)
    else:
        x_filt, P_filt, x_pred, P_pred, C = forward_filter_30m(
            ind_array, Q_var=Q_var, R_var=R_var)

    state = pd.DataFrame({
        'position': x_filt[:, 0],
        'velocity': x_filt[:, 1],
        'pred_position': x_pred[:, 0],
        'pred_velocity': x_pred[:, 1],
    }, index=df.index)

    return {
        'state': state,
        'P_filt': P_filt,
        'P_pred': P_pred,
        'C': C,
        'indicator': ind_series,
    }


def compute_flks_slopes(df_tf, df_5m, indicator, tf_minutes, k_range=(1, 6),
                         Q_var=KALMAN_PROCESS_VAR, R_var=KALMAN_MEASURE_VAR,
                         indicator_params=None):
    """
    Dispatcher FLKS-2 : calcule les backward slopes (Fixed-Lag Kalman Smoother)
    pour un indicateur, au timeframe TF, avec sous-pas 5min.

    Pipeline interne:
      1. compute_forward_filter(df_tf, indicator)
         → x_filt, P_filt, x_pred, C (états Kalman standard)
      2. compute_indicator_live(df_5m, is_close, indicator, tf_minutes)
         → indicateur live 5min (frozen + provisional)
      3. group_per_candle(df_5m, df_tf, live_5m)
         → live_per_candle[t] = values 5min durant la bougie t
      4. compute_slopes_test1(x_filt, x_pred, C)
         → slope_t1 (backward 2 pas, pas de sous-pas 5min)
      5. compute_slopes_test2(x_filt, P_filt, x_pred, C, live_per_candle, k)
         pour k ∈ [k_range[0]..k_range[1]]
         → slope_k1..k6 (backward 3 pas avec k updates Kalman supplémentaires)

    Args:
        df_tf: DataFrame OHLC au timeframe TF.
        df_5m: DataFrame OHLC au 5min (source des sous-pas).
        indicator: 'macd' | 'rsi' | 'cci'.
        tf_minutes: taille du bucket TF (30 ou 60).
        k_range: (k_min, k_max) inclus. Default (1, 6) → slope_k1 à slope_k6.

    Returns:
        pd.DataFrame indexée par df_tf.index, colonnes:
          [slope_t1, slope_k1, slope_k2, ..., slope_k<k_max>]
        NaN remplacés par 0 (cohérent avec le reste du pipeline).
    """
    ind_kwargs = indicator_params or {}

    # Étape 1: forward filter sur df_tf (avec Q/R et params indicateur)
    fwd = compute_forward_filter(df_tf, indicator, adaptive=False,
                                   Q_var=Q_var, R_var=R_var,
                                   indicator_params=ind_kwargs)
    x_filt_pos = fwd['state'][['position', 'velocity']].values
    x_pred_pos = fwd['state'][['pred_position', 'pred_velocity']].values
    P_filt = fwd['P_filt']
    C = fwd['C']

    # Étape 2: indicateur live 5min (avec params indicateur)
    is_close = compute_bucket_close_mask(df_5m.index, tf_minutes)
    live_series = compute_indicator_live(
        df_5m, is_close, indicator, tf_minutes, **ind_kwargs)

    # Étape 3: grouper les live 5min par bougie TF
    live_per_candle = group_per_candle(df_5m, df_tf, live_series.values)

    # Étape 4: slope_t1 (pas de sous-pas, pas de dt_sub)
    slopes_t1 = compute_slopes_test1(x_filt_pos, x_pred_pos, C)

    # Étape 5: slope_k1..k_max (avec sous-pas 5min et dt_sub dynamique)
    result = {'slope_t1': slopes_t1}
    k_min, k_max = k_range
    for k in range(k_min, k_max + 1):
        result[f'slope_k{k}'] = compute_slopes_test2(
            x_filt_pos, P_filt, x_pred_pos, C, live_per_candle, k,
            tf_minutes=tf_minutes,  # ← dt_sub dynamique dérivé de tf_minutes
            Q_var=Q_var, R_var=R_var,
        )

    # fillna(0) cohérent avec le reste du pipeline
    df_slopes = pd.DataFrame(result, index=df_tf.index)
    df_slopes = df_slopes.fillna(0.0)
    return df_slopes


# ============================================================================
# ML DATA PREPARATION
# ============================================================================

def prepare_features_and_labels(df_tf, df_5m, indicator, tf_minutes, trim=100,
                                  Q_var=KALMAN_PROCESS_VAR, R_var=KALMAN_MEASURE_VAR,
                                  indicator_params=None):
    """
    Prépare un DataFrame features + labels prêt pour split/normalize/sequences.

    Scope V1 (actuel — reproduit exactement le script train_flks_slopes.py):
        Features = [slope_k1, slope_k2, slope_k3, slope_k4, slope_k5, slope_k6]
        Soit 6 slopes FLKS avec sous-pas 5min (k=1..6).

    Chaîne interne :
        df_tf, df_5m → compute_flks_slopes   → slopes_k1..k6 (+ slope_t1 calculée
                                                 mais NON exposée en V1)
        df_tf        → compute_oracle_labels → label binary + continu
        df_tf        → close (pour backtest downstream)

    Args:
        df_tf: DataFrame OHLC au TF.
        df_5m: DataFrame OHLC au 5min.
        indicator: 'macd' | 'rsi' | 'cci'.
        tf_minutes: 30 ou 60.
        trim: nombre de bougies à retirer au début ET à la fin pour écarter
              les warm-up Kalman/oracle et les bords incomplets. Default 100.

    Returns:
        pd.DataFrame indexée, 9 colonnes:
          - slope_k1..slope_k6              (6 features FLKS)
          - label_binary                    (int : 1 si oracle slope > 0 sinon 0)
          - label_continuous                (float : oracle slope brute)
          - close                           (float : prix close pour backtest)

    Améliorations futures documentées (V2, V3):
        V2: ajouter slope_t1 (backward 2 pas, sans sous-pas) → 7 features
            Motivation: slope_t1 est une estimation sans injection 5min live.
            Peut servir à comparer l'apport des sous-pas vs un filtre plus pur.
        V3: ajouter position + velocity (x_filt du forward filter Kalman) → 9 features
            Motivation: capturer l'état Kalman en plus de ses différences.
            ATTENTION: position a un ordre de grandeur BEAUCOUP plus grand
            que les slopes (plusieurs dizaines vs ~1) → normalisation spécifique
            nécessaire (z-score OK si appliqué par colonne).
        Implémentation suggérée: paramètre `feature_set='v1'|'v2'|'v3'`.
    """
    # 1. Features FLKS : on garde uniquement slope_k1..k6 (V1)
    slopes = compute_flks_slopes(df_tf, df_5m, indicator, tf_minutes,
                                  Q_var=Q_var, R_var=R_var,
                                  indicator_params=indicator_params)
    v1_cols = [f'slope_k{k}' for k in range(1, 7)]
    slopes = slopes[v1_cols]  # drop slope_t1 (calculée mais non exposée en V1)

    # 2. Labels oracle (position, slope, label=binary)
    oracle = compute_oracle_labels(df_tf, indicator,
                                     Q_var=Q_var, R_var=R_var,
                                     indicator_params=indicator_params)

    # 3. Assembler
    result = slopes.copy()
    result['label_binary'] = oracle['label'].astype(int)
    result['label_continuous'] = oracle['slope'].astype(np.float64)
    result['close'] = df_tf['close'].astype(np.float64)

    # 4. TRIM début et fin pour écarter warm-up et bords incomplets
    if trim > 0 and len(result) > 2 * trim:
        result = result.iloc[trim:-trim]
    return result


def split_train_val_test(df, train_ratio=0.70, val_ratio=0.15, gap=0):
    """
    Split chronologique d'un DataFrame temporel en 3 splits disjoints.

    Args:
        df: DataFrame indexé par timestamps (déjà trimé).
        train_ratio: fraction pour train. Default 0.70.
        val_ratio: fraction pour val. Default 0.15. Test = 1 - train_ratio - val_ratio.
        gap: nombre de lignes exclues entre train et val, ET entre val et test.
             Utile pour éviter le leakage de séquences (gap = window - 1).

    Returns:
        (df_train, df_val, df_test) : 3 DataFrames chronologiquement disjoints.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    df_train = df.iloc[:train_end - gap] if gap > 0 else df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end - gap] if gap > 0 else df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]
    return df_train, df_val, df_test


def normalize_features(df_train, df_val, df_test, feature_cols):
    """
    Z-score normalization. Stats fittées SUR TRAIN UNIQUEMENT.

    Args:
        df_train, df_val, df_test: 3 DataFrames issus de split_train_val_test.
        feature_cols: liste des colonnes à normaliser.

    Returns:
        (df_train_norm, df_val_norm, df_test_norm, stats) où stats est un dict
        {col: (mean, std)} utile pour inférence en production.
    """
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()
    stats = {}
    for col in feature_cols:
        mean = df_train[col].mean()
        std = df_train[col].std()
        if std < 1e-10:
            std = 1.0  # évite division par zéro pour features constantes
        stats[col] = (float(mean), float(std))
        df_train[col] = (df_train[col] - mean) / std
        df_val[col] = (df_val[col] - mean) / std
        df_test[col] = (df_test[col] - mean) / std
    return df_train, df_val, df_test, stats


def make_sequences(df, feature_cols, label_cols, window=25):
    """
    Crée les séquences temporelles pour entraînement LSTM/XGBoost.

    Pour chaque i dans [0, n - window]:
      X[i] = features[i : i + window]
      y[i] = labels[i + window - 1]         (label à la DERNIÈRE timestep)
      close[i], date[i] idem

    Args:
        df: DataFrame (issu de split/normalize).
        feature_cols: liste des colonnes features.
        label_cols: liste de 1 ou N colonnes labels. Si 1 string, renvoie y 1D.
                   Si liste, renvoie dict {col: array}.
        window: taille de la séquence. Default 25.

    Returns:
        dict avec clés:
          'X'      : np.ndarray float32 shape (n_seq, window, n_features)
          'y'      : np.ndarray int64 (si 1 label) OU dict (si plusieurs)
          'closes' : np.ndarray float64 shape (n_seq,)
          'dates'  : np.ndarray datetime64 shape (n_seq,)
    """
    n = len(df)
    n_feat = len(feature_cols)
    if n < window:
        raise ValueError(f"DataFrame too short ({n}) for window={window}")

    features = df[feature_cols].values.astype(np.float32)
    closes = df['close'].values.astype(np.float64) if 'close' in df.columns \
             else np.full(n, np.nan)
    dates = df.index.values

    # Indices 2D pour construire X d'un coup (fancy indexing)
    indices = np.arange(window)[None, :] + np.arange(n - window + 1)[:, None]
    X = features[indices]  # (n_seq, window, n_feat)
    closes_out = closes[window - 1:]
    dates_out = dates[window - 1:]

    # Labels : simple string ou liste
    if isinstance(label_cols, str):
        y = df[label_cols].values[window - 1:]
        # Cast : int64 si dtype int, sinon garder float
        if np.issubdtype(y.dtype, np.integer):
            y = y.astype(np.int64)
        else:
            y = y.astype(np.float64)
    else:
        y = {}
        for col in label_cols:
            vals = df[col].values[window - 1:]
            if np.issubdtype(vals.dtype, np.integer):
                vals = vals.astype(np.int64)
            else:
                vals = vals.astype(np.float64)
            y[col] = vals

    return {
        'X': X,
        'y': y,
        'closes': closes_out,
        'dates': dates_out,
    }


# ============================================================================
# INDICATORS — Live frozen/provisional
# ============================================================================

def compute_macd_live(close_5min, is_close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    """MACD live 5min avec frozen + provisional EMAs. Périodes paramétrables."""
    n = len(close_5min)
    alpha_f = 2.0 / (fast + 1)
    alpha_s = 2.0 / (slow + 1)
    alpha_sig = 2.0 / (signal + 1)
    out = np.full(n, np.nan)
    ema_f_cl = ema_s_cl = ema_sig_cl = np.nan
    init = False
    for i in range(n):
        c = close_5min[i]
        if np.isnan(c):
            continue
        if not init:
            if is_close[i]:
                ema_f_cl = c
                ema_s_cl = c
                ema_sig_cl = 0.0
                out[i] = 0.0
                init = True
            continue
        ef = alpha_f * c + (1.0 - alpha_f) * ema_f_cl
        es = alpha_s * c + (1.0 - alpha_s) * ema_s_cl
        ml = ef - es
        esg = alpha_sig * ml + (1.0 - alpha_sig) * ema_sig_cl
        out[i] = ml - esg
        if is_close[i]:
            ema_f_cl = ef
            ema_s_cl = es
            ema_sig_cl = esg
    return out


def compute_rsi_live(close_5min, is_close, period=RSI_PERIOD):
    """RSI live 5min avec frozen + provisional EMAs. Période paramétrable."""
    n = len(close_5min)
    alpha = 2.0 / (period + 1)
    out = np.full(n, np.nan)
    closure_indices = []
    closure_closes = []
    for i in range(n):
        if not np.isnan(close_5min[i]) and is_close[i]:
            closure_indices.append(i)
            closure_closes.append(close_5min[i])
    if len(closure_closes) < 2:
        return out
    closes_arr = np.array(closure_closes)
    deltas = np.diff(closes_arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    gains_padded = np.concatenate([[0.0], gains])
    losses_padded = np.concatenate([[0.0], losses])
    ag = gains_padded[0]
    al = losses_padded[0]
    closure_states = [(ag, al, closure_closes[0])]
    for k in range(1, len(gains_padded)):
        ag = alpha * gains_padded[k] + (1.0 - alpha) * ag
        al = alpha * losses_padded[k] + (1.0 - alpha) * al
        closure_states.append((ag, al, closure_closes[k]))
    for k, ci in enumerate(closure_indices):
        ag_k, al_k, _ = closure_states[k]
        if al_k > 1e-15:
            out[ci] = 100.0 - 100.0 / (1.0 + ag_k / al_k)
    closure_set = set(closure_indices)
    current_k = -1
    ag_cl = 0.0
    al_cl = 0.0
    prev_cl = np.nan
    for i in range(n):
        c = close_5min[i]
        if np.isnan(c):
            continue
        if i in closure_set:
            current_k += 1
            ag_cl, al_cl, prev_cl = closure_states[current_k]
            continue
        if current_k >= 0 and not np.isnan(prev_cl):
            delta = c - prev_cl
            gn = max(delta, 0.0)
            ls = max(-delta, 0.0)
            ag_p = alpha * gn + (1.0 - alpha) * ag_cl
            al_p = alpha * ls + (1.0 - alpha) * al_cl
            if al_p > 1e-15:
                out[i] = 100.0 - 100.0 / (1.0 + ag_p / al_p)
    return out


def compute_cci_live(high_live, low_live, close_5min, is_close, period=CCI_PERIOD):
    """CCI live 5min. Période paramétrable."""
    n = len(close_5min)
    out = np.full(n, np.nan)
    tp_buf = deque(maxlen=period - 1)
    for i in range(n):
        c = close_5min[i]
        h = high_live[i]
        lo = low_live[i]
        if np.isnan(c) or np.isnan(h) or np.isnan(lo):
            continue
        tp = (h + lo + c) / 3.0
        if len(tp_buf) >= period - 1:
            all_tp = np.array(list(tp_buf) + [tp])
            sma = all_tp.mean()
            mad = np.abs(all_tp - sma).mean()
            out[i] = (tp - sma) / (0.015 * mad) if mad > 1e-15 else 0.0
        if is_close[i]:
            tp_buf.append(tp)
    return out


# ============================================================================
# ORACLE
# ============================================================================

def compute_oracle(indicator_30m, Q_var=KALMAN_PROCESS_VAR, R_var=KALMAN_MEASURE_VAR):
    """
    Smoother RTS (non-causal). Q_var et R_var paramétrables.
    """
    from pykalman import KalmanFilter as KF
    n = len(indicator_30m)
    valid = ~np.isnan(indicator_30m)
    if valid.sum() < 3:
        return np.full(n, np.nan), np.full(n, np.nan)
    vd = indicator_30m[valid]
    kf = KF(
        transition_matrices=[[1, 1], [0, 1]],
        observation_matrices=[[1, 0]],
        initial_state_mean=[vd[0], 0.0],
        initial_state_covariance=np.eye(2),
        observation_covariance=R_var,
        transition_covariance=np.eye(2) * Q_var,
    )
    smooth_means, _ = kf.smooth(vd)
    positions = np.full(n, np.nan)
    positions[valid] = smooth_means[:, 0]
    slopes = np.full(n, np.nan)
    for t in range(2, n):
        if not np.isnan(positions[t - 1]) and not np.isnan(positions[t - 2]):
            slopes[t] = positions[t - 1] - positions[t - 2]
    return positions, slopes


# ============================================================================
# KALMAN LIVE (5min resolution, frozen/provisional)
# ============================================================================

def compute_kalman_live_standard(indicator_live, is_close):
    """Standard Kalman (Q fixe) with frozen/provisional. Returns (n, 2) = [pos, vel]."""
    from pykalman import KalmanFilter as KF
    n = len(indicator_live)
    out = np.full((n, 2), np.nan)
    closure_indices = []
    closure_values = []
    for i in range(n):
        if not np.isnan(indicator_live[i]) and is_close[i]:
            closure_indices.append(i)
            closure_values.append(indicator_live[i])
    if len(closure_values) < 2:
        return out
    cv = np.array(closure_values)
    kf = KF(transition_matrices=A, observation_matrices=np.array([[1, 0]]),
            initial_state_mean=[cv[0], 0.0], initial_state_covariance=np.eye(2),
            observation_covariance=KALMAN_MEASURE_VAR,
            transition_covariance=np.eye(2) * KALMAN_PROCESS_VAR)
    state_means, state_covs = kf.filter(cv)
    for k, ci in enumerate(closure_indices):
        out[ci, 0] = state_means[k, 0]
        out[ci, 1] = state_means[k, 1]
    closure_set = set(closure_indices)
    current_k = -1
    sm_cl = np.array([cv[0], 0.0])
    sc_cl = np.eye(2)
    for i in range(n):
        obs = indicator_live[i]
        if np.isnan(obs):
            continue
        if i in closure_set:
            current_k += 1
            sm_cl = state_means[current_k]
            sc_cl = state_covs[current_k]
            continue
        if current_k >= 0:
            sm_p, _ = kf.filter_update(sm_cl, sc_cl, observation=obs)
            out[i, 0] = sm_p[0]
            out[i, 1] = sm_p[1]
    return out


def compute_kalman_live_aqkf(indicator_live, is_close, aq_window=30, Q_max_factor=10.0):
    """AQ-KF (adaptive Q) with frozen/provisional. Returns (n, 2) = [pos, vel]."""
    _Q_fixed = np.eye(2) * KALMAN_PROCESS_VAR
    _R = np.array([[KALMAN_MEASURE_VAR]])
    Q_FLOOR = _Q_fixed * 0.1
    Q_CEIL = _Q_fixed * Q_max_factor
    n = len(indicator_live)
    out = np.full((n, 2), np.nan)
    closure_indices = []
    closure_values = []
    for i in range(n):
        if not np.isnan(indicator_live[i]) and is_close[i]:
            closure_indices.append(i)
            closure_values.append(indicator_live[i])
    if len(closure_values) < 2:
        return out
    cv = np.array(closure_values)
    nc = len(cv)
    x_filt_cl = np.zeros((nc, 2))
    P_filt_cl = np.zeros((nc, 2, 2))
    Q_current = _Q_fixed.copy()
    innovation_buffer = []
    for k in range(nc):
        if k == 0:
            x_p = np.array([cv[0], 0.0])
            P_p = np.eye(2)
        else:
            x_p = A @ x_filt_cl[k - 1]
            P_p = A @ P_filt_cl[k - 1] @ A.T + Q_current
        y = cv[k] - H @ x_p
        S = (H @ P_p @ H.T + _R)[0, 0]
        K = P_p @ H.T / S
        x_filt_cl[k] = x_p + (K @ y).ravel()
        P_filt_cl[k] = (np.eye(2) - K @ H) @ P_p
        v_t = cv[k] - (H @ x_p)[0]
        innovation_buffer.append(v_t)
        if len(innovation_buffer) > aq_window:
            innovation_buffer.pop(0)
        if len(innovation_buffer) >= aq_window and k > 0:
            C_vv = np.mean(np.array(innovation_buffer) ** 2)
            delta = C_vv - S
            if delta > 0:
                P_pred_next = A @ P_filt_cl[k] @ A.T + Q_current
                C_rts = P_filt_cl[k] @ A.T @ inv2x2(P_pred_next)
                Q_candidate = delta * (C_rts @ C_rts.T)
                if is_pos_semidef(Q_candidate):
                    Q_current = np.clip(Q_candidate, Q_FLOOR, Q_CEIL)
    for k, ci in enumerate(closure_indices):
        out[ci, 0] = x_filt_cl[k, 0]
        out[ci, 1] = x_filt_cl[k, 1]
    closure_set = set(closure_indices)
    current_k = -1
    sm_cl = np.array([cv[0], 0.0])
    sc_cl = np.eye(2)
    for i in range(n):
        obs = indicator_live[i]
        if np.isnan(obs):
            continue
        if i in closure_set:
            current_k += 1
            sm_cl = x_filt_cl[current_k]
            sc_cl = P_filt_cl[current_k]
            continue
        if current_k >= 0:
            x_p = A @ sm_cl
            P_p = A @ sc_cl @ A.T + Q_current
            y_val = obs - (H @ x_p)[0]
            S_val = (H @ P_p @ H.T + _R)[0, 0]
            K_val = P_p @ H.T / S_val
            sm_p = x_p + (K_val * y_val).ravel()
            out[i, 0] = sm_p[0]
            out[i, 1] = sm_p[1]
    return out


# ============================================================================
# KALMAN PRIMITIVES
# ============================================================================

def kf_update(x_p, P_p, z_obs, R_mat=None):
    """
    Kalman measurement update. R_mat peut être override (défaut = R global).
    """
    R_use = R if R_mat is None else R_mat
    y = z_obs - H @ x_p
    S = H @ P_p @ H.T + R_use
    K = P_p @ H.T / S[0, 0]
    return x_p + (K @ y).ravel(), (np.eye(2) - K @ H) @ P_p


def kf_predict_sub(x, P, A_mat=None, Q_mat=None):
    """
    Kalman prediction sub-step. A_mat et Q_mat overridables (défaut = A_SUB, Q_SUB).
    Utilisé avec dt_sub dynamique selon tf_minutes.
    """
    A_use = A_SUB if A_mat is None else A_mat
    Q_use = Q_SUB if Q_mat is None else Q_mat
    return A_use @ x, A_use @ P @ A_use.T + Q_use


def inv2x2(M):
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if abs(det) > 1e-15:
        return np.array([[M[1, 1], -M[0, 1]],
                         [-M[1, 0], M[0, 0]]]) / det
    return np.linalg.pinv(M)


def is_pos_semidef(M):
    return M[0, 0] >= 0 and M[1, 1] >= 0 and (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]) >= -1e-12


# ============================================================================
# FORWARD FILTERS
# ============================================================================

def forward_filter_30m(indicator_30m, Q_var=KALMAN_PROCESS_VAR, R_var=KALMAN_MEASURE_VAR):
    """
    Standard Kalman forward filter. Returns (x_filt, P_filt, x_pred, P_pred, C).
    Q_var et R_var paramétrables (défauts = constantes globales).
    """
    Q_local = np.eye(2) * Q_var
    R_local = np.array([[R_var]])
    n = len(indicator_30m)
    first_valid_val = indicator_30m[~np.isnan(indicator_30m)][0]
    x_filt = np.zeros((n, 2))
    P_filt = np.zeros((n, 2, 2))
    x_pred = np.zeros((n, 2))
    P_pred = np.zeros((n, 2, 2))
    for t in range(n):
        if t == 0:
            x_p = np.array([first_valid_val, 0.0])
            P_p = np.eye(2)
        else:
            x_p = A @ x_filt[t - 1]
            P_p = A @ P_filt[t - 1] @ A.T + Q_local
        x_pred[t] = x_p
        P_pred[t] = P_p
        if np.isnan(indicator_30m[t]):
            x_filt[t] = x_p
            P_filt[t] = P_p
        else:
            x_filt[t], P_filt[t] = kf_update(x_p, P_p, indicator_30m[t], R_mat=R_local)
    C = np.zeros((n, 2, 2))
    for t in range(n - 1):
        C[t] = P_filt[t] @ A.T @ inv2x2(P_pred[t + 1])
    return x_filt, P_filt, x_pred, P_pred, C


def forward_filter_30m_adaptive(indicator_30m, window=30, Q_max_factor=10.0,
                                  Q_min_factor=0.1,
                                  Q_var=KALMAN_PROCESS_VAR, R_var=KALMAN_MEASURE_VAR):
    """
    AQ-KF forward filter (Myers-Tapley). Q_var et R_var paramétrables.
    """
    Q_local = np.eye(2) * Q_var
    R_local = np.array([[R_var]])
    n = len(indicator_30m)
    first_valid_val = indicator_30m[~np.isnan(indicator_30m)][0]
    x_filt = np.zeros((n, 2))
    P_filt = np.zeros((n, 2, 2))
    x_pred = np.zeros((n, 2))
    P_pred = np.zeros((n, 2, 2))
    Q_current = Q_local.copy()
    innovation_buffer = []
    Q_FLOOR = Q_local * Q_min_factor
    Q_CEIL = Q_local * Q_max_factor

    for t in range(n):
        if t == 0:
            x_p = np.array([first_valid_val, 0.0])
            P_p = np.eye(2)
        else:
            x_p = A @ x_filt[t - 1]
            P_p = A @ P_filt[t - 1] @ A.T + Q_current
        x_pred[t] = x_p
        P_pred[t] = P_p
        if np.isnan(indicator_30m[t]):
            x_filt[t] = x_p
            P_filt[t] = P_p
            continue
        S_t = (H @ P_p @ H.T + R_local)[0, 0]
        x_filt[t], P_filt[t] = kf_update(x_p, P_p, indicator_30m[t], R_mat=R_local)
        v_t = indicator_30m[t] - (H @ x_p)[0]
        innovation_buffer.append(v_t)
        if len(innovation_buffer) > window:
            innovation_buffer.pop(0)
        if len(innovation_buffer) >= window and t > 0:
            C_vv = np.mean(np.array(innovation_buffer) ** 2)
            delta = C_vv - S_t
            if delta > 0:
                P_pred_next = A @ P_filt[t] @ A.T + Q_current
                C_rts = P_filt[t] @ A.T @ inv2x2(P_pred_next)
                Q_candidate = delta * (C_rts @ C_rts.T)
                if is_pos_semidef(Q_candidate):
                    Q_current = np.clip(Q_candidate, Q_FLOOR, Q_CEIL)

    C_gains = np.zeros((n, 2, 2))
    for t in range(n - 1):
        C_gains[t] = P_filt[t] @ A.T @ inv2x2(P_pred[t + 1])
    return x_filt, P_filt, x_pred, P_pred, C_gains


# ============================================================================
# SLOPES (FLKS backward)
# ============================================================================

def compute_slopes_test1(x_filt, x_pred, C):
    """Backward 2 steps from x_filt[t]. slope[t] = smoothed[t-1] - smoothed[t-2]."""
    n = len(x_filt)
    slopes = np.full(n, np.nan)
    for t in range(2, n):
        sm_t1 = x_filt[t - 1] + C[t - 1] @ (x_filt[t] - x_pred[t])
        sm_t2 = x_filt[t - 2] + C[t - 2] @ (sm_t1 - x_pred[t - 1])
        slopes[t] = sm_t1[0] - sm_t2[0]
    return slopes


def compute_slopes_test2(x_filt, P_filt, x_pred, C, live_per_candle, n_substeps,
                          tf_minutes=30,
                          Q_var=KALMAN_PROCESS_VAR, R_var=KALMAN_MEASURE_VAR):
    """
    Backward 3 steps: x_prov (from sub-steps of candle t+1) → t → t-1 → t-2.

    Args:
        tf_minutes: taille du bucket TF. Détermine dt_sub = 5 / tf_minutes
                    pour les matrices A_sub et Q_sub dynamiques.
                    30 → dt_sub=1/6 (défaut), 60 → dt_sub=1/12, etc.
        Q_var, R_var: covariances Kalman process/measurement.
    """
    # Sub-step dynamique selon tf_minutes (5min est l'intervalle 5min par défaut)
    dt_sub = 5.0 / tf_minutes
    A_sub_local = np.array([[1.0, dt_sub], [0.0, 1.0]])
    Q_local = np.eye(2) * Q_var
    Q_sub_local = Q_local * dt_sub
    R_local = np.array([[R_var]])

    n = len(x_filt)
    slopes = np.full(n, np.nan)
    for t in range(2, n - 1):
        x_cur = x_filt[t].copy()
        P_cur = P_filt[t].copy()
        live_vals = live_per_candle[t + 1]
        valid_vals = [v for v in live_vals if not np.isnan(v)]
        use = valid_vals[:n_substeps]
        if len(use) > 0:
            for m5 in use:
                x_cur, P_cur = kf_predict_sub(
                    x_cur, P_cur, A_mat=A_sub_local, Q_mat=Q_sub_local)
                x_cur, P_cur = kf_update(x_cur, P_cur, m5, R_mat=R_local)
        x_prov = x_cur
        k_actual = len(use) if len(use) > 0 else 1
        A_k = np.linalg.matrix_power(A_sub_local, k_actual)
        Q_k = Q_sub_local * k_actual
        x_pred_partial = A_k @ x_filt[t]
        P_pred_partial = A_k @ P_filt[t] @ A_k.T + Q_k
        C_partial = P_filt[t] @ A_k.T @ inv2x2(P_pred_partial)
        sm_t = x_filt[t] + C_partial @ (x_prov - x_pred_partial)
        sm_t1 = x_filt[t - 1] + C[t - 1] @ (sm_t - x_pred[t])
        sm_t2 = x_filt[t - 2] + C[t - 2] @ (sm_t1 - x_pred[t - 1])
        slopes[t] = sm_t1[0] - sm_t2[0]
    return slopes


# ============================================================================
# METRICS
# ============================================================================

def sign_concordance(slopes_test, slopes_oracle, start, end):
    EPSILON = 1e-8
    s_t = slopes_test[start:end]
    s_o = slopes_oracle[start:end]
    mask = ~np.isnan(s_t) & ~np.isnan(s_o) & (np.abs(s_o) > EPSILON)
    n_valid = mask.sum()
    if n_valid == 0:
        return np.nan, 0
    return np.mean(np.sign(s_t[mask]) == np.sign(s_o[mask])) * 100.0, n_valid


def find_oracle_transitions(slopes_oracle, start, end):
    EPSILON = 1e-8
    s_o = slopes_oracle[start:end]
    sign_o = np.where(np.abs(s_o) < EPSILON, 0, np.sign(s_o))
    trans = np.zeros(len(s_o), dtype=bool)
    for i in range(1, len(s_o)):
        if sign_o[i] != 0 and sign_o[i - 1] != 0 and sign_o[i] != sign_o[i - 1]:
            trans[i] = True
    return trans


def sign_concordance_at_transitions(slopes_test, slopes_oracle, start, end, trans):
    s_t = slopes_test[start:end]
    s_o = slopes_oracle[start:end]
    mask = trans & ~np.isnan(s_t) & ~np.isnan(s_o)
    n_valid = mask.sum()
    if n_valid == 0:
        return np.nan, 0
    return np.mean(np.sign(s_t[mask]) == np.sign(s_o[mask])) * 100.0, n_valid


# ============================================================================
# BACKTEST
# ============================================================================

def _exec_trade(position, entry_price, exec_price, fees):
    if position == 1:
        pnl = (exec_price - entry_price) / entry_price
    else:
        pnl = (entry_price - exec_price) / entry_price
    return pnl - fees


def backtest_30m(slopes, closes_30m, start, end, fees,
                 threshold=0.0, holding_min=0):
    pnl_total = 0.0
    n_trades = 0
    n_wins = 0
    position = 0
    entry_price = 0.0
    entry_t = -holding_min
    for t in range(start, end):
        if np.isnan(slopes[t]) or abs(slopes[t]) < threshold:
            continue
        target = 1 if slopes[t] > 0 else -1
        if position == target:
            continue
        if position != 0 and (t - entry_t) < holding_min:
            continue
        if t + 1 >= len(closes_30m):
            continue
        exec_price = closes_30m[t]
        if np.isnan(exec_price):
            continue
        if position != 0:
            trade_pnl = _exec_trade(position, entry_price, exec_price, fees)
            pnl_total += trade_pnl
            if trade_pnl > 0:
                n_wins += 1
        entry_price = exec_price
        position = target
        n_trades += 1
        entry_t = t
        pnl_total -= fees
    if position != 0 and end < len(closes_30m):
        exec_price = closes_30m[min(end, len(closes_30m) - 1)]
        if not np.isnan(exec_price):
            trade_pnl = _exec_trade(position, entry_price, exec_price, fees)
            pnl_total += trade_pnl
            if trade_pnl > 0:
                n_wins += 1
    wr = (n_wins / n_trades * 100.0) if n_trades > 0 else 0.0
    return {'pnl_pct': pnl_total * 100, 'trades': n_trades, 'win_rate': wr}


def backtest_5m(slopes, closes_5m_per_candle, k_substep, start, end, fees,
                threshold=0.0, holding_min=0):
    pnl_total = 0.0
    n_trades = 0
    n_wins = 0
    position = 0
    entry_price = 0.0
    entry_t = -holding_min
    for t in range(start, end):
        if np.isnan(slopes[t]) or abs(slopes[t]) < threshold:
            continue
        target = 1 if slopes[t] > 0 else -1
        if position == target:
            continue
        if position != 0 and (t - entry_t) < holding_min:
            continue
        candle_idx = t + 1
        if candle_idx >= len(closes_5m_per_candle):
            continue
        closes_5m = closes_5m_per_candle[candle_idx]
        step_idx = k_substep - 1
        if step_idx >= len(closes_5m):
            continue
        exec_price = closes_5m[step_idx]
        if np.isnan(exec_price):
            continue
        if position != 0:
            trade_pnl = _exec_trade(position, entry_price, exec_price, fees)
            pnl_total += trade_pnl
            if trade_pnl > 0:
                n_wins += 1
        entry_price = exec_price
        position = target
        n_trades += 1
        entry_t = t
        pnl_total -= fees
    if position != 0:
        last_candle = min(end, len(closes_5m_per_candle) - 1)
        closes_last = closes_5m_per_candle[last_candle]
        if len(closes_last) > 0 and not np.isnan(closes_last[-1]):
            trade_pnl = _exec_trade(position, entry_price, closes_last[-1], fees)
            pnl_total += trade_pnl
            if trade_pnl > 0:
                n_wins += 1
    wr = (n_wins / n_trades * 100.0) if n_trades > 0 else 0.0
    return {'pnl_pct': pnl_total * 100, 'trades': n_trades, 'win_rate': wr}


def buy_and_hold(closes, start, end):
    c = closes[start:end + 1]
    valid = c[~np.isnan(c)]
    if len(valid) < 2:
        return 0.0
    return (valid[-1] - valid[0]) / valid[0] * 100


# ============================================================================
# POST-PROCESSING
# ============================================================================

def viterbi_decode(probs, self_trans=0.95):
    """Viterbi decoding on binary probability sequence."""
    n = len(probs)
    log_trans_same = np.log(self_trans)
    log_trans_switch = np.log(1 - self_trans)
    log_emit = np.zeros((n, 2))
    log_emit[:, 1] = np.log(np.clip(probs, 1e-10, 1 - 1e-10))
    log_emit[:, 0] = np.log(np.clip(1 - probs, 1e-10, 1 - 1e-10))
    V = np.zeros((n, 2))
    backptr = np.zeros((n, 2), dtype=int)
    V[0] = log_emit[0] + np.log(0.5)
    for t in range(1, n):
        for s in range(2):
            score_same = V[t-1, s] + log_trans_same
            other = 1 - s
            score_switch = V[t-1, other] + log_trans_switch
            if score_same >= score_switch:
                V[t, s] = score_same + log_emit[t, s]
                backptr[t, s] = s
            else:
                V[t, s] = score_switch + log_emit[t, s]
                backptr[t, s] = other
    labels = np.zeros(n, dtype=int)
    labels[-1] = np.argmax(V[-1])
    for t in range(n-2, -1, -1):
        labels[t] = backptr[t+1, labels[t+1]]
    return labels


def cusum_filter(probs, threshold=2.0):
    """CUSUM filter on probability sequence."""
    n = len(probs)
    labels = np.zeros(n, dtype=int)
    current_state = 1 if probs[0] > 0.5 else 0
    labels[0] = current_state
    s_up = 0.0
    s_down = 0.0
    for t in range(1, n):
        x = probs[t] - 0.5
        s_up = max(0, s_up + x)
        s_down = min(0, s_down + x)
        if current_state == 0 and s_up > threshold:
            current_state = 1
            s_up = 0.0
            s_down = 0.0
        elif current_state == 1 and -s_down > threshold:
            current_state = 0
            s_up = 0.0
            s_down = 0.0
        labels[t] = current_state
    return labels


# ============================================================================
# HELPERS
# ============================================================================

def group_per_candle(df_5m, df_30m, array_5m):
    """Group 5min values by 30min candle."""
    per_candle = []
    for ts_30m in df_30m.index:
        bucket_end = ts_30m + pd.Timedelta(minutes=29, seconds=59)
        mask = (df_5m.index >= ts_30m) & (df_5m.index <= bucket_end)
        per_candle.append(array_5m[mask])
    return per_candle


# ============================================================================
# DATA LOADING — NPZ + CSV aligned for backtests
# ============================================================================

PREPARED_DATA_DIR = 'data/prepared'

ASSET_CSV_MAP = {'BTC': 'BTCUSD'}


def find_features_csv():
    """Find the features CSV. Prefer FLKS features, fall back to old pipeline."""
    candidates = [
        f'{PREPARED_DATA_DIR}/BTCUSD_flks_features.csv',
        f'{PREPARED_DATA_DIR}/BTCUSD_multitf_macd_rsi_cci.csv',
    ]
    for c in candidates:
        from pathlib import Path
        if Path(c).exists():
            return c
    raise FileNotFoundError(f"No features CSV found. Tried: {candidates}")


def load_test_data(indicator='macd', timeframe='30m', threshold=0.5):
    """
    Load NPZ predictions + aligned closes.

    If NPZ contains 'test_closes' (saved by train_flks_slopes.py),
    uses those directly — guaranteed alignment.

    Returns:
        y_test: oracle labels
        y_pred_proba: model probabilities
        y_pred_binary: thresholded predictions
        closes_test: aligned close prices
        n_test: number of test samples
        source: description of data source
    """
    npz_path = f'{PREPARED_DATA_DIR}/{indicator}_{timeframe}_dataset.npz'
    from pathlib import Path
    if not Path(npz_path).exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    if 'y_test' in data:
        y_test = data['y_test']
        y_pred_proba = data['y_test_pred']
    else:
        y_test = data['test_labels']
        y_pred_proba = data['test_preds']

    n_test = len(y_test)
    y_pred_binary = (y_pred_proba > threshold).astype(int)

    # Closes + dates: prefer from NPZ (guaranteed alignment)
    if 'test_closes' in data:
        closes_test = data['test_closes']
        dates_test = data['test_dates'] if 'test_dates' in data else None
        source = f"NPZ (closes+dates embedded)"
    else:
        csv_path = find_features_csv()
        df = pd.read_csv(csv_path, parse_dates=['datetime']).set_index('datetime').sort_index()
        closes_test = df['close'].dropna().values[-n_test:]
        dates_test = None
        source = f"CSV fallback (last {n_test} rows — may be misaligned!)"

    return y_test, y_pred_proba, y_pred_binary, closes_test, n_test, source
