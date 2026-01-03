"""
Script de test des parametres d'indicateurs.

Test: 3 features (RSI, CCI, MACD avec params variables)
      -> 1 target (Kalman d'un indicateur avec params fixes)

Usage:
    python src/test_indicator_params.py
"""

import numpy as np
import pandas as pd
import subprocess
import sys
import json
import tempfile
from pathlib import Path
import logging
from itertools import product

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Imports locaux
from constants import (
    AVAILABLE_ASSETS_5M,
    KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR,
    SEQUENCE_LENGTH, MACD_SIGNAL,
)
from data_utils import load_crypto_data, trim_edges
from indicators import (
    calculate_rsi, calculate_cci, calculate_macd,
    normalize_cci, normalize_macd_histogram,
    create_sequences
)
from filters import kalman_filter


# =============================================================================
# CONFIGURATION DU TEST
# =============================================================================

# Parametres FIXES pour les targets (labels)
TARGET_RSI_PERIOD = 14
TARGET_CCI_PERIOD = 20
TARGET_MACD_FAST = 12
TARGET_MACD_SLOW = 26

# Parametres VARIABLES pour les features
RSI_PARAMS = [7, 10, 14, 20]
CCI_PARAMS = [10, 14, 20, 30]
MACD_PARAMS = [(8, 17), (12, 26), (16, 34)]  # (fast, slow)

# Limite de donnees
MAX_SAMPLES = 50000


# =============================================================================
# GENERATION DES LABELS
# =============================================================================

def generate_indicator_labels(df, indicator_type):
    """
    Genere les labels depuis Kalman(indicator) avec params FIXES.

    Args:
        df: DataFrame avec OHLCV
        indicator_type: 'RSI', 'CCI', ou 'MACD'

    Returns:
        labels: np.array de 0/1
    """
    if indicator_type == 'RSI':
        indicator = calculate_rsi(df['close'], period=TARGET_RSI_PERIOD)
    elif indicator_type == 'CCI':
        cci_raw = calculate_cci(df['high'], df['low'], df['close'], period=TARGET_CCI_PERIOD)
        indicator = normalize_cci(cci_raw)
    elif indicator_type == 'MACD':
        macd_data = calculate_macd(df['close'], fast_period=TARGET_MACD_FAST,
                                   slow_period=TARGET_MACD_SLOW, signal_period=MACD_SIGNAL)
        indicator = normalize_macd_histogram(macd_data['histogram'])
    else:
        raise ValueError(f"Unknown indicator: {indicator_type}")

    # Appliquer Kalman
    indicator = pd.Series(indicator).ffill().fillna(50.0).values
    filtered = kalman_filter(indicator, KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR)

    # Labels = pente du Kalman
    labels = np.zeros(len(filtered), dtype=int)
    for t in range(2, len(filtered)):
        if filtered[t-1] > filtered[t-2]:
            labels[t] = 1

    buy_pct = labels.sum() / len(labels) * 100
    logger.info(f"  Labels Kalman({indicator_type}): {buy_pct:.1f}% UP")

    return labels


# =============================================================================
# PREPARATION DES FEATURES
# =============================================================================

def prepare_dataset(df, rsi_period, cci_period, macd_fast, macd_slow, labels, output_path):
    """
    Prepare dataset avec 3 features (RSI, CCI, MACD) et params variables.
    """
    # Calculer les 3 indicateurs
    rsi = calculate_rsi(df['close'], period=rsi_period)
    rsi = pd.Series(rsi).ffill().fillna(50.0).values

    cci_raw = calculate_cci(df['high'], df['low'], df['close'], period=cci_period)
    cci = normalize_cci(cci_raw)
    cci = pd.Series(cci).ffill().fillna(50.0).values

    macd_data = calculate_macd(df['close'], fast_period=macd_fast,
                               slow_period=macd_slow, signal_period=MACD_SIGNAL)
    macd = normalize_macd_histogram(macd_data['histogram'])
    macd = pd.Series(macd).ffill().fillna(50.0).values

    # Stack en 3 features
    features = np.column_stack([rsi, cci, macd])
    labels_2d = labels.reshape(-1, 1)

    X, Y = create_sequences(features, labels_2d, sequence_length=SEQUENCE_LENGTH)

    # Limiter
    if len(X) > MAX_SAMPLES:
        X = X[:MAX_SAMPLES]
        Y = Y[:MAX_SAMPLES]

    # Split 70/15/15
    n = len(X)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test, Y_test = X[n_train+n_val:], Y[n_train+n_val:]

    metadata = {
        'n_features': 3,
        'rsi_period': rsi_period,
        'cci_period': cci_period,
        'macd_fast': macd_fast,
        'macd_slow': macd_slow,
    }

    np.savez_compressed(
        output_path,
        X_train=X_train, Y_train=Y_train,
        X_val=X_val, Y_val=Y_val,
        X_test=X_test, Y_test=Y_test,
        metadata=json.dumps(metadata)
    )

    return len(X_train), len(X_val), len(X_test)


def run_training(data_path):
    """Lance train.py et recupere la MEILLEURE val accuracy."""
    cmd = [
        sys.executable, 'src/train.py',
        '--data', str(data_path),
        '--epochs', '30',
        '--patience', '8',
        '--batch-size', '256',
        '--indicator', 'rsi',  # Force single-output mode
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )

    output = result.stdout + result.stderr

    if result.returncode != 0:
        logger.error(f"Train failed: {result.stderr[:500]}")
        return None

    # Collecter TOUTES les val accuracies
    val_accuracies = []
    for line in output.split('\n'):
        if 'Val   - Loss:' in line and 'Acc:' in line:
            try:
                acc_part = line.split('Acc:')[1].split(',')[0].strip()
                val_accuracies.append(float(acc_part))
            except:
                pass

    if not val_accuracies:
        logger.error("Aucune accuracy trouvee")
        return None

    best_acc = max(val_accuracies)
    logger.info(f"  Epochs: {len(val_accuracies)}, Best: {best_acc:.3f}")

    return best_acc


def load_btc_limited():
    """Charge les donnees BTC limitees."""
    df = load_crypto_data(AVAILABLE_ASSETS_5M['BTC'], asset_name='BTC')
    df = trim_edges(df, trim_start=100, trim_end=100)

    if len(df) > MAX_SAMPLES + 1000:
        df = df.tail(MAX_SAMPLES + 1000)

    logger.info(f"BTC: {len(df)} bougies")
    return df


def main():
    """Test principal."""
    logger.info("="*60)
    logger.info("TEST: 3 FEATURES (params var) -> 1 TARGET (params fixes)")
    logger.info("="*60)

    # Charger donnees
    df = load_btc_limited()

    # Pour chaque target (RSI, CCI, MACD)
    targets = ['RSI', 'CCI', 'MACD']
    all_results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_path = tmpdir / 'dataset.npz'

        for target in targets:
            logger.info("\n" + "="*60)
            logger.info(f"TARGET: Kalman({target})")
            logger.info("="*60)

            # Generer labels pour ce target
            labels = generate_indicator_labels(df, target)

            results = {}

            # Tester toutes les combinaisons de params
            for rsi_p, cci_p, (macd_f, macd_s) in product(RSI_PARAMS, CCI_PARAMS, MACD_PARAMS):
                params_str = f"RSI={rsi_p}, CCI={cci_p}, MACD={macd_f}/{macd_s}"
                logger.info(f"\n--- {params_str} ---")

                n_train, n_val, n_test = prepare_dataset(
                    df, rsi_p, cci_p, macd_f, macd_s, labels, data_path
                )

                acc = run_training(data_path)
                results[(rsi_p, cci_p, macd_f, macd_s)] = acc

                if acc:
                    logger.info(f"  -> Accuracy: {acc:.2%}")

            all_results[target] = results

    # === RESUME ===
    logger.info("\n" + "="*60)
    logger.info("RESUME DES RESULTATS")
    logger.info("="*60)

    for target, results in all_results.items():
        logger.info(f"\n=== Target: Kalman({target}) ===")
        logger.info(f"{'RSI':<6} {'CCI':<6} {'MACD':<8} {'Accuracy':<10}")

        # Trier par accuracy
        sorted_results = sorted(
            [(k, v) for k, v in results.items() if v is not None],
            key=lambda x: x[1],
            reverse=True
        )

        for (rsi_p, cci_p, macd_f, macd_s), acc in sorted_results[:5]:  # Top 5
            logger.info(f"{rsi_p:<6} {cci_p:<6} {macd_f}/{macd_s:<5} {acc:.2%}")

        if sorted_results:
            best = sorted_results[0]
            logger.info(f"\n  MEILLEUR: RSI={best[0][0]}, CCI={best[0][1]}, "
                       f"MACD={best[0][2]}/{best[0][3]} -> {best[1]:.2%}")


if __name__ == '__main__':
    main()
