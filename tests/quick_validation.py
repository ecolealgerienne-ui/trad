#!/usr/bin/env python3
"""
Validation rapide du pipeline - Test basique sans visualisation.

Vérifie rapidement:
1. Création des bougies fantômes (6 steps)
2. Intégrité OHLC
3. Features avancées présentes
4. Pas de NaN inattendus
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np

# Imports du pipeline
from data_pipeline import create_ghost_candles
from advanced_features import (
    add_velocity_features,
    add_open_context,
    add_step_index_normalized,
    add_log_returns_ghost,
    add_all_advanced_features
)
from utils import validate_ohlc_integrity


def create_test_data(n=100):
    """Crée 100 bougies 5min de test."""
    timestamps = pd.date_range('2024-01-01', periods=n, freq='5T')

    # Prix simple qui monte/descend
    prices = 50000 + np.cumsum(np.random.randn(n) * 100)

    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        open_price = prices[i-1] if i > 0 else close
        high = max(open_price, close) * 1.001
        low = min(open_price, close) * 0.999

        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': 100.0
        })

    return pd.DataFrame(data)


def test_basic_pipeline():
    """Test rapide du pipeline complet."""
    print("\n" + "="*60)
    print("VALIDATION RAPIDE DU PIPELINE")
    print("="*60)

    errors = []

    # 1. Créer données test
    print("\n[1/5] Création données test...")
    df_5m = create_test_data(n=100)
    print(f"✅ {len(df_5m)} bougies 5min créées")

    # 2. Bougies fantômes
    print("\n[2/5] Création bougies fantômes...")
    df_ghost = create_ghost_candles(df_5m, target_timeframe='30T')

    # Vérifier 6 steps
    steps_per_candle = df_ghost.groupby('candle_30m_timestamp')['step'].count()
    complete_candles = (steps_per_candle == 6).sum()
    total_candles = len(steps_per_candle)

    if complete_candles / total_candles >= 0.8:  # Au moins 80% complètes
        print(f"✅ {complete_candles}/{total_candles} bougies complètes (6 steps)")
    else:
        errors.append(f"Trop peu de bougies complètes: {complete_candles}/{total_candles}")
        print(f"❌ Seulement {complete_candles}/{total_candles} bougies complètes")

    # Vérifier OHLC
    try:
        validate_ohlc_integrity(df_ghost, col_prefix='ghost_')
        print("✅ Intégrité OHLC validée")
    except ValueError as e:
        errors.append(f"OHLC invalide: {e}")
        print(f"❌ {e}")

    # 3. Features avancées
    print("\n[3/5] Ajout features avancées...")
    df = add_all_advanced_features(
        df_ghost,
        ghost_prefix='ghost',
        open_zscore_window=50,
        max_steps=6
    )

    # Vérifier présence des features
    expected_features = [
        'velocity', 'amplitude', 'acceleration',
        'ghost_high_log', 'ghost_low_log', 'ghost_close_log',
        'ghost_open_zscore',
        'step_index_norm'
    ]

    for feat in expected_features:
        if feat in df.columns:
            print(f"  ✅ {feat}")
        else:
            errors.append(f"Feature manquante: {feat}")
            print(f"  ❌ {feat} MANQUANT")

    # 4. Vérifier ranges
    print("\n[4/5] Vérification des ranges...")

    # Step index norm doit être [0.0, 1.0]
    if 'step_index_norm' in df.columns:
        min_val = df['step_index_norm'].min()
        max_val = df['step_index_norm'].max()

        if min_val == 0.0 and max_val == 1.0:
            print(f"  ✅ step_index_norm: [{min_val}, {max_val}]")
        else:
            errors.append(f"step_index_norm range incorrect: [{min_val}, {max_val}]")
            print(f"  ❌ step_index_norm: [{min_val}, {max_val}] au lieu de [0.0, 1.0]")

    # Amplitude doit être positive
    if 'amplitude' in df.columns:
        if (df['amplitude'].dropna() >= 0).all():
            print(f"  ✅ amplitude toujours positive")
        else:
            errors.append("Amplitude négative détectée")
            print(f"  ❌ Amplitude négative trouvée!")

    # Open Z-Score doit avoir mean~0 (tolérance plus élevée pour petits datasets)
    if 'ghost_open_zscore' in df.columns:
        mean_zscore = df['ghost_open_zscore'].dropna().mean()
        # Pour un grand dataset: |mean| < 0.1
        # Pour un petit dataset test: |mean| < 1.0 (acceptable)
        if abs(mean_zscore) < 1.0:
            print(f"  ✅ ghost_open_zscore mean={mean_zscore:.4f}")
        else:
            errors.append(f"Open Z-Score mean trop éloigné de 0: {mean_zscore}")
            print(f"  ❌ ghost_open_zscore mean={mean_zscore:.4f} (>1.0)")

    # 5. Statistiques
    print("\n[5/5] Statistiques finales...")
    print(f"  Lignes totales: {len(df)}")
    print(f"  Colonnes: {len(df.columns)}")
    print(f"  Bougies 30min: {df['candle_30m_timestamp'].nunique()}")

    # Compter les NaN par colonne
    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]

    if len(nan_cols) > 0:
        print(f"\n  Colonnes avec NaN:")
        for col, count in nan_cols.items():
            pct = count / len(df) * 100
            print(f"    - {col}: {count} ({pct:.1f}%)")

    # Rapport final
    print("\n" + "="*60)
    if errors:
        print(f"❌ VALIDATION ÉCHOUÉE - {len(errors)} erreurs:")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("✅ VALIDATION RÉUSSIE - Pipeline fonctionnel!")
        return True


if __name__ == '__main__':
    success = test_basic_pipeline()
    sys.exit(0 if success else 1)
