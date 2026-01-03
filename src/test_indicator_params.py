"""
Script de test des parametres d'indicateurs vs Kalman(Close).

Teste differentes valeurs de parametres pour trouver celles qui predisent
le mieux la pente de Kalman(Close).

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

# Parametres norme et decrements de 20%
RSI_PARAMS = [14, 11, 9, 7]      # norme=14, -20% chaque fois
CCI_PARAMS = [20, 16, 13, 10]    # norme=20, -20% chaque fois
MACD_PARAMS = [               # (fast, slow) - norme=12/26
    (12, 26),
    (10, 21),
    (8, 17),
    (6, 13),
]

# Limite de donnees
MAX_SAMPLES = 50000


# =============================================================================
# FONCTIONS
# =============================================================================

def load_btc_limited():
    """Charge les donnees BTC limitees."""
    df = load_crypto_data(AVAILABLE_ASSETS_5M['BTC'], asset_name='BTC')
    df = trim_edges(df, trim_start=100, trim_end=100)

    # Limiter
    if len(df) > MAX_SAMPLES + 1000:
        df = df.tail(MAX_SAMPLES + 1000)

    logger.info(f"BTC: {len(df)} bougies")
    return df


def generate_close_labels(df):
    """Genere les labels depuis Kalman(Close)."""
    close = df['close'].values
    filtered = kalman_filter(close, KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR)

    labels = np.zeros(len(filtered), dtype=int)
    for t in range(2, len(filtered)):
        if filtered[t-1] > filtered[t-2]:
            labels[t] = 1

    return labels


def prepare_rsi_dataset(df, period, labels, output_path):
    """Prepare dataset avec RSI seul."""
    rsi = calculate_rsi(df['close'], period=period)
    rsi = pd.Series(rsi).ffill().fillna(50.0).values

    # Feature = RSI seul (shape n, 1)
    features = rsi.reshape(-1, 1)
    labels_2d = labels.reshape(-1, 1)

    X, Y = create_sequences(features, labels_2d, sequence_length=SEQUENCE_LENGTH)

    # Limiter
    if len(X) > MAX_SAMPLES:
        X = X[:MAX_SAMPLES]
        Y = Y[:MAX_SAMPLES]

    # Split simple 70/15/15
    n = len(X)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test, Y_test = X[n_train+n_val:], Y[n_train+n_val:]

    metadata = {'n_features': 1, 'indicator': 'RSI', 'period': period}

    np.savez_compressed(
        output_path,
        X_train=X_train, Y_train=Y_train,
        X_val=X_val, Y_val=Y_val,
        X_test=X_test, Y_test=Y_test,
        metadata=json.dumps(metadata)
    )

    return len(X_train), len(X_val), len(X_test)


def prepare_cci_dataset(df, period, labels, output_path):
    """Prepare dataset avec CCI seul."""
    cci_raw = calculate_cci(df['high'], df['low'], df['close'], period=period)
    cci = normalize_cci(cci_raw)
    cci = pd.Series(cci).ffill().fillna(50.0).values

    features = cci.reshape(-1, 1)
    labels_2d = labels.reshape(-1, 1)

    X, Y = create_sequences(features, labels_2d, sequence_length=SEQUENCE_LENGTH)

    if len(X) > MAX_SAMPLES:
        X = X[:MAX_SAMPLES]
        Y = Y[:MAX_SAMPLES]

    n = len(X)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test, Y_test = X[n_train+n_val:], Y[n_train+n_val:]

    metadata = {'n_features': 1, 'indicator': 'CCI', 'period': period}

    np.savez_compressed(
        output_path,
        X_train=X_train, Y_train=Y_train,
        X_val=X_val, Y_val=Y_val,
        X_test=X_test, Y_test=Y_test,
        metadata=json.dumps(metadata)
    )

    return len(X_train), len(X_val), len(X_test)


def prepare_macd_dataset(df, fast, slow, labels, output_path):
    """Prepare dataset avec MACD seul."""
    macd_data = calculate_macd(df['close'], fast_period=fast, slow_period=slow, signal_period=MACD_SIGNAL)
    macd = normalize_macd_histogram(macd_data['histogram'])
    macd = pd.Series(macd).ffill().fillna(50.0).values

    features = macd.reshape(-1, 1)
    labels_2d = labels.reshape(-1, 1)

    X, Y = create_sequences(features, labels_2d, sequence_length=SEQUENCE_LENGTH)

    if len(X) > MAX_SAMPLES:
        X = X[:MAX_SAMPLES]
        Y = Y[:MAX_SAMPLES]

    n = len(X)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test, Y_test = X[n_train+n_val:], Y[n_train+n_val:]

    metadata = {'n_features': 1, 'indicator': 'MACD', 'fast': fast, 'slow': slow}

    np.savez_compressed(
        output_path,
        X_train=X_train, Y_train=Y_train,
        X_val=X_val, Y_val=Y_val,
        X_test=X_test, Y_test=Y_test,
        metadata=json.dumps(metadata)
    )

    return len(X_train), len(X_val), len(X_test)


def run_training(data_path, model_path):
    """Lance train.py et recupere la val accuracy."""
    cmd = [
        sys.executable, 'src/train.py',
        '--data', str(data_path),
        '--epochs', '20',
        '--patience', '5',
        '--batch-size', '256',
        '--indicator', 'rsi',  # Force single-output mode (label = Close, pas RSI)
    ]

    # Ajouter chemin model temporaire
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )

    # Parser la sortie pour trouver val accuracy
    output = result.stdout + result.stderr

    # Chercher "Val   - Loss: X.XXXX, Acc: 0.XXX"
    val_acc = None
    for line in output.split('\n'):
        if 'Val   - Loss:' in line and 'Acc:' in line:
            try:
                # Extraire accuracy
                acc_part = line.split('Acc:')[1].split(',')[0].strip()
                val_acc = float(acc_part)
            except:
                pass

    # Chercher aussi "Meilleur modèle: Époque X, Val Loss: X.XXXX"
    # et la derniere val accuracy
    best_acc = val_acc

    return best_acc


def main():
    """Test principal."""
    logger.info("="*60)
    logger.info("TEST PARAMETRES INDICATEURS vs KALMAN(CLOSE)")
    logger.info("="*60)

    # Charger donnees
    df = load_btc_limited()

    # Generer labels depuis Close
    logger.info("\nGeneration labels Kalman(Close)...")
    labels = generate_close_labels(df)
    buy_pct = labels.sum() / len(labels) * 100
    logger.info(f"Labels: {buy_pct:.1f}% BUY")

    # Resultats
    results = {'RSI': {}, 'CCI': {}, 'MACD': {}}

    # Fichiers temporaires
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_path = tmpdir / 'dataset.npz'
        model_path = tmpdir / 'model.pth'

        # === TEST RSI ===
        logger.info("\n" + "="*60)
        logger.info("TEST RSI")
        logger.info("="*60)

        for period in RSI_PARAMS:
            logger.info(f"\n--- RSI period={period} ---")
            n_train, n_val, n_test = prepare_rsi_dataset(df, period, labels, data_path)
            logger.info(f"Dataset: train={n_train}, val={n_val}, test={n_test}")

            acc = run_training(data_path, model_path)
            results['RSI'][period] = acc
            logger.info(f"RSI({period}) -> Val Accuracy: {acc:.1%}" if acc else f"RSI({period}) -> ERREUR")

        # === TEST CCI ===
        logger.info("\n" + "="*60)
        logger.info("TEST CCI")
        logger.info("="*60)

        for period in CCI_PARAMS:
            logger.info(f"\n--- CCI period={period} ---")
            n_train, n_val, n_test = prepare_cci_dataset(df, period, labels, data_path)
            logger.info(f"Dataset: train={n_train}, val={n_val}, test={n_test}")

            acc = run_training(data_path, model_path)
            results['CCI'][period] = acc
            logger.info(f"CCI({period}) -> Val Accuracy: {acc:.1%}" if acc else f"CCI({period}) -> ERREUR")

        # === TEST MACD ===
        logger.info("\n" + "="*60)
        logger.info("TEST MACD")
        logger.info("="*60)

        for fast, slow in MACD_PARAMS:
            logger.info(f"\n--- MACD fast={fast}, slow={slow} ---")
            n_train, n_val, n_test = prepare_macd_dataset(df, fast, slow, labels, data_path)
            logger.info(f"Dataset: train={n_train}, val={n_val}, test={n_test}")

            acc = run_training(data_path, model_path)
            results['MACD'][(fast, slow)] = acc
            logger.info(f"MACD({fast}/{slow}) -> Val Accuracy: {acc:.1%}" if acc else f"MACD({fast}/{slow}) -> ERREUR")

    # === RESUME ===
    logger.info("\n" + "="*60)
    logger.info("RESUME DES RESULTATS")
    logger.info("="*60)

    logger.info("\n=== RSI ===")
    logger.info(f"{'Period':<10} {'Accuracy':<10}")
    for period, acc in results['RSI'].items():
        acc_str = f"{acc:.1%}" if acc else "ERREUR"
        logger.info(f"{period:<10} {acc_str:<10}")

    logger.info("\n=== CCI ===")
    logger.info(f"{'Period':<10} {'Accuracy':<10}")
    for period, acc in results['CCI'].items():
        acc_str = f"{acc:.1%}" if acc else "ERREUR"
        logger.info(f"{period:<10} {acc_str:<10}")

    logger.info("\n=== MACD ===")
    logger.info(f"{'Fast/Slow':<10} {'Accuracy':<10}")
    for (fast, slow), acc in results['MACD'].items():
        acc_str = f"{acc:.1%}" if acc else "ERREUR"
        logger.info(f"{fast}/{slow:<7} {acc_str:<10}")

    # Meilleurs
    logger.info("\n" + "="*60)
    logger.info("MEILLEURS PARAMETRES")
    logger.info("="*60)

    for ind, res in results.items():
        if res:
            valid = {k: v for k, v in res.items() if v is not None}
            if valid:
                best = max(valid.items(), key=lambda x: x[1])
                logger.info(f"{ind}: {best[0]} -> {best[1]:.1%}")


if __name__ == '__main__':
    main()
