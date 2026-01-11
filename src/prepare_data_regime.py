"""
Script de préparation des données pour Meta-Regime Trading (5min).

PRINCIPE CLÉ: Classification 4 régimes (TS × VC) + Features enrichies (~20)
===============================================================================

⚠️ NOUVELLE APPROCHE: Abandonner la prédiction directionnelle (MACD/RSI/CCI)
→ Prédire le RÉGIME de marché (4 classes) pour améliorer Win Rate et PF

Régimes:
- 0: RANGE LOW VOL  (TS < 0.4, VC ≤ P70)
- 1: RANGE HIGH VOL (TS < 0.4, VC > P70)
- 2: TREND LOW VOL  (TS > 0.6, VC ≤ P70)
- 3: TREND HIGH VOL (TS > 0.6, VC > P70)

Features (~20 colonnes):
  Trend: MA slopes, ADX, regression, Hurst, MACD histogram
  Volatility: ATR normalized, BB bands, realized vol, compression
  Volume/Micro: Volume ratio, spike, VWAP deviation, OBV derivative

Labels:
  - regime: 0-3 (4 classes)
  - trend_strength: 0-1 (score TS)
  - volatility_cluster: 0-1 (score VC)

Pipeline:
1. Charger données brutes (OHLCV 5min)
2. Calculer ~20 features de régime (regime_features.py)
3. Calculer labels régime (regime_labeler.py)
4. Créer séquences (12 timesteps × ~20 features)
5. Split temporel (70/15/15)
6. Sauvegarder dataset unique: dataset_<assets>_regime.npz

Usage:
    python src/prepare_data_regime.py --assets BTC ETH BNB ADA LTC

Génère:
    data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz

Author: Claude Code - Phase 1 (Data Layer)
Date: 2025-01-11
Version: 1.0
"""

import numpy as np
import pandas as pd
import argparse
import logging
import json
import os
from pathlib import Path
from datetime import datetime
import gc
from numpy.lib.stride_tricks import sliding_window_view
from joblib import Parallel, delayed
import psutil

logger = logging.getLogger(__name__)

# Import modules locaux
from constants import (
    AVAILABLE_ASSETS_5M, DEFAULT_ASSETS,
    TRIM_EDGES,
    PREPARED_DATA_DIR,
    SEQUENCE_LENGTH,
)

# Import modules de régime
from regime_features import calculate_all_regime_features, get_regime_feature_names
from regime_labeler import calculate_regime_labels, validate_regime_features

# Mapping Asset Name → Asset ID (pour encodage dans les matrices)
ASSET_ID_MAP = {
    'BTC': 0,
    'ETH': 1,
    'BNB': 2,
    'ADA': 3,
    'LTC': 4
}


# =============================================================================
# PARALLÉLISATION INTELLIGENTE
# =============================================================================

def get_safe_n_jobs(n_assets: int, ram_per_asset_gb: float = 4.0) -> int:
    """
    Calcule le nombre de jobs parallèles selon la RAM disponible.

    Args:
        n_assets: Nombre total d'assets à traiter
        ram_per_asset_gb: RAM peak estimée par asset (GB)

    Returns:
        Nombre de jobs sûrs (1 à min(n_assets, n_cores))
    """
    try:
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        max_by_ram = max(1, int(available_ram_gb / ram_per_asset_gb))
        n_cores = os.cpu_count() or 1
        max_by_cpu = max(1, n_cores - 1)
        n_jobs = min(max_by_ram, max_by_cpu, n_assets)

        logger.info(f"Parallélisation: {n_jobs} assets simultanés")
        logger.info(f"  RAM disponible: {available_ram_gb:.1f} GB")
        logger.info(f"  RAM par asset: {ram_per_asset_gb:.1f} GB")

        return n_jobs
    except Exception as e:
        logger.warning(f"Erreur détection parallélisme: {e}, fallback n_jobs=1")
        return 1


# =============================================================================
# SPLIT TEMPOREL
# =============================================================================

def temporal_split(df: pd.DataFrame,
                   train_ratio: float = 0.70,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15) -> dict:
    """
    Split temporel chronologique: Train → Val → Test.

    Args:
        df: DataFrame avec index temporel
        train_ratio: Ratio pour train (défaut: 70%)
        val_ratio: Ratio pour val (défaut: 15%)
        test_ratio: Ratio pour test (défaut: 15%)

    Returns:
        dict avec clés 'train', 'val', 'test'
    """
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train+n_val].copy()
    test_df = df.iloc[n_train+n_val:].copy()

    logger.info(f"  Split temporel: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }


# =============================================================================
# CRÉATION SÉQUENCES POUR RÉGIMES
# =============================================================================

def create_sequences_for_regime(df: pd.DataFrame,
                                 feature_cols: list,
                                 asset_name: str,
                                 asset_id: int,
                                 seq_length: int = SEQUENCE_LENGTH) -> tuple:
    """
    Crée les séquences pour le dataset régime.

    Structure:
    - X: (n, seq_length, n_features+2) = [timestamp, asset_id, features...]
    - Y: (n, 5) = [timestamp, asset_id, regime, ts_score, vc_score]
    - OHLCV: (n, 7) = [timestamp, asset_id, O, H, L, C, V]

    Args:
        df: DataFrame avec features + labels de régime
        feature_cols: Liste des features à utiliser (~20)
        asset_name: Nom de l'asset ('BTC', 'ETH', etc.)
        asset_id: ID encodé de l'asset (0-4)
        seq_length: Longueur des séquences (défaut: 12)

    Returns:
        X: (n, seq_length, n_features+2)
        Y: (n, 5)
        OHLCV: (n, 7)
    """
    # Colonnes label (3 pour régime: regime, ts_score, vc_score)
    label_cols = ['regime', 'trend_strength', 'volatility_cluster']

    # Colonnes OHLCV brutes
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']

    # Remplacer NaN par 0
    cols_needed = feature_cols + label_cols + ohlcv_cols
    df_clean = df[cols_needed].fillna(0)
    df_clean.index = df.index  # Préserver l'index temporel

    n_nans_filled = df[cols_needed].isna().sum().sum()
    logger.info(f"     NaN → 0: {n_nans_filled} valeurs remplacées")

    # Extraire arrays
    features = df_clean[feature_cols].values.astype(np.float32)  # (N, n_features)
    labels = df_clean[label_cols].values.astype(np.float32)      # (N, 3)
    ohlcv = df_clean[ohlcv_cols].values.astype(np.float32)       # (N, 5)

    N, n_features = features.shape

    # Validation
    if N < seq_length:
        logger.warning(f"     {asset_name}: Pas assez de données ({N} < {seq_length})")
        return None, None, None

    # Timestamps (Unix en secondes)
    timestamps = df_clean.index.astype(np.int64) // 10**9  # (N,)
    timestamps = timestamps.astype(np.float32)

    # Asset ID (répété)
    asset_ids = np.full(N, asset_id, dtype=np.float32)  # (N,)

    # ========================================================================
    # VECTORISATION SLIDING WINDOWS (×50 plus rapide)
    # ========================================================================

    # Features: (N, n_features) → (n_sequences, seq_length, n_features)
    X_features = sliding_window_view(features, window_shape=(seq_length, n_features)).squeeze(axis=1)

    # Timestamps: (N,) → (n_sequences, seq_length)
    X_timestamps = sliding_window_view(timestamps, window_shape=seq_length).reshape(-1, seq_length)

    # Asset IDs: (N,) → (n_sequences, seq_length)
    X_asset_ids = sliding_window_view(asset_ids, window_shape=seq_length).reshape(-1, seq_length)

    # Labels: Prendre le label à la FIN de chaque séquence
    Y_labels = labels[seq_length-1:]  # (n_sequences, 3)

    # Timestamps pour Y: Derniers timestamps de chaque séquence
    Y_timestamps = timestamps[seq_length-1:]  # (n_sequences,)

    # Asset IDs pour Y
    Y_asset_ids = asset_ids[seq_length-1:]  # (n_sequences,)

    # OHLCV: Prendre OHLCV à la FIN de chaque séquence
    OHLCV_data = ohlcv[seq_length-1:]  # (n_sequences, 5)
    OHLCV_timestamps = timestamps[seq_length-1:]
    OHLCV_asset_ids = asset_ids[seq_length-1:]

    n_sequences = X_features.shape[0]

    # Combiner X: [timestamp, asset_id, features...]
    # Shape: (n_sequences, seq_length, 2+n_features)
    X = np.concatenate([
        X_timestamps[..., np.newaxis],  # (n_seq, seq_len, 1)
        X_asset_ids[..., np.newaxis],   # (n_seq, seq_len, 1)
        X_features                      # (n_seq, seq_len, n_features)
    ], axis=2)

    # Combiner Y: [timestamp, asset_id, regime, ts_score, vc_score]
    # Shape: (n_sequences, 5)
    Y = np.column_stack([
        Y_timestamps,  # (n_seq,)
        Y_asset_ids,   # (n_seq,)
        Y_labels       # (n_seq, 3)
    ])

    # Combiner OHLCV: [timestamp, asset_id, O, H, L, C, V]
    # Shape: (n_sequences, 7)
    OHLCV = np.column_stack([
        OHLCV_timestamps,
        OHLCV_asset_ids,
        OHLCV_data
    ])

    logger.info(f"     Séquences créées: {n_sequences}")
    logger.info(f"       X: {X.shape} (timestamp, asset_id, {n_features} features)")
    logger.info(f"       Y: {Y.shape} (timestamp, asset_id, regime, ts, vc)")
    logger.info(f"       OHLCV: {OHLCV.shape}")

    return X, Y, OHLCV


# =============================================================================
# TRAITEMENT D'UN ASSET
# =============================================================================

def process_single_asset(asset_name: str,
                          clip_value: float = None,
                          max_samples: int = None) -> dict:
    """
    Traite un seul asset: charge, calcule features, labels, split, séquences.

    Args:
        asset_name: Nom de l'asset ('BTC', 'ETH', etc.)
        clip_value: Valeur de clipping des features (None = pas de clip)
        max_samples: Limite nombre de samples (None = tout)

    Returns:
        dict avec 'train', 'val', 'test' contenant (X, Y, OHLCV)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"ASSET: {asset_name}")
    logger.info('='*80)

    # Charger CSV
    csv_path = AVAILABLE_ASSETS_5M.get(asset_name)
    if csv_path is None:
        raise ValueError(f"Asset {asset_name} non disponible")

    logger.info(f"  Chargement: {csv_path}")
    df = pd.read_csv(csv_path)

    # Limiter nombre de samples pour tests
    if max_samples is not None and max_samples > 0:
        df = df.head(max_samples)
        logger.info(f"  Limité à {max_samples} samples (test)")

    logger.info(f"  Lignes chargées: {len(df)}")

    # Colonnes OHLCV (renommage si nécessaire)
    if 'Open' in df.columns:
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        })

    # Index temporel
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
    else:
        logger.warning("  Aucune colonne timestamp trouvée, utilisation index")

    # Trim edges (100 début + 100 fin)
    if len(df) > 2 * TRIM_EDGES:
        df = df.iloc[TRIM_EDGES:-TRIM_EDGES].copy()
        logger.info(f"  Trim ±{TRIM_EDGES}: {len(df)} lignes restantes")

    # ========================================================================
    # ÉTAPE 1: CALCULER FEATURES DE RÉGIME (~20 colonnes)
    # ========================================================================

    logger.info(f"\n  Calcul features de régime (~20 colonnes)...")
    df = calculate_all_regime_features(df)
    logger.info(f"  ✓ Features calculées: {df.shape[1]} colonnes")

    # ========================================================================
    # ÉTAPE 2: CALCULER LABELS DE RÉGIME (regime, ts_score, vc_score)
    # ========================================================================

    logger.info(f"\n  Calcul labels de régime (4 classes)...")
    try:
        validate_regime_features(df)
        regime_labels, ts_score, vc_score = calculate_regime_labels(df)

        # Ajouter au DataFrame
        df['regime'] = regime_labels
        df['trend_strength'] = ts_score
        df['volatility_cluster'] = vc_score

        logger.info(f"  ✓ Labels calculés")
    except Exception as e:
        logger.error(f"  ✗ Erreur calcul labels: {e}")
        raise

    # Remplacer NaN par 0 après tout le calcul
    df = df.fillna(0)

    # ========================================================================
    # ÉTAPE 3: SPLIT TEMPOREL (70/15/15)
    # ========================================================================

    logger.info(f"\n  Split temporel...")
    splits = temporal_split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    # ========================================================================
    # ÉTAPE 4: CRÉER SÉQUENCES POUR CHAQUE SPLIT
    # ========================================================================

    logger.info(f"\n  Création séquences (seq_length={SEQUENCE_LENGTH})...")

    # Features à utiliser (toutes les features de régime)
    feature_cols = get_regime_feature_names()
    logger.info(f"  Features utilisées ({len(feature_cols)}): {feature_cols}")

    # Clip si demandé
    if clip_value is not None:
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].clip(-clip_value, clip_value)
        logger.info(f"  Features clippées à ±{clip_value}")

    # Asset ID
    asset_id = ASSET_ID_MAP[asset_name]

    results = {}
    for split_name, split_df in splits.items():
        logger.info(f"\n    {split_name.upper()}:")
        X, Y, OHLCV = create_sequences_for_regime(
            split_df,
            feature_cols,
            asset_name,
            asset_id,
            seq_length=SEQUENCE_LENGTH
        )

        if X is not None:
            results[split_name] = (X, Y, OHLCV)
        else:
            logger.warning(f"    {split_name}: Pas de séquences créées")
            results[split_name] = (None, None, None)

    # Nettoyage mémoire
    del df, splits
    gc.collect()

    logger.info(f"\n✅ Asset {asset_name} traité")

    return results


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main():
    """
    Pipeline principal de préparation des données régime.
    """
    parser = argparse.ArgumentParser(description='Préparation données régime')
    parser.add_argument('--assets', nargs='+', default=DEFAULT_ASSETS,
                        choices=list(AVAILABLE_ASSETS_5M.keys()),
                        help='Liste des assets à inclure')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Répertoire de sortie (défaut: data/prepared)')
    parser.add_argument('--clip', type=float, default=None,
                        help='Valeur de clipping des features (None = pas de clip)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limiter le nombre de samples par asset (pour tests)')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger.info("="*80)
    logger.info("PRÉPARATION DONNÉES - META-REGIME TRADING")
    logger.info("="*80)
    logger.info(f"Assets: {args.assets}")
    logger.info(f"Sequence length: {SEQUENCE_LENGTH}")
    logger.info(f"Régimes: 4 classes (TS × VC)")
    logger.info(f"Features: ~20 colonnes (trend, volatility, volume)")

    # ========================================================================
    # PARALLÉLISATION MULTI-CORE
    # ========================================================================

    n_jobs = get_safe_n_jobs(len(args.assets), ram_per_asset_gb=8.0)
    logger.info(f"\n🚀 TRAITEMENT PARALLÈLE: {n_jobs} asset(s) simultané(s)")

    # Traiter les assets en parallèle
    all_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_asset)(
            asset_name, args.clip, args.max_samples
        ) for asset_name in args.assets
    )

    # ========================================================================
    # CONCATÉNATION DES RÉSULTATS
    # ========================================================================

    logger.info(f"\n{'='*80}")
    logger.info(f"CONCATÉNATION FINALE")
    logger.info('='*80)

    # Organiser par split
    datasets = {'train': [], 'val': [], 'test': []}

    for asset_results in all_results:
        for split_name in ['train', 'val', 'test']:
            if asset_results[split_name][0] is not None:
                datasets[split_name].append(asset_results[split_name])

    # Concaténer
    X_train = np.concatenate([s[0] for s in datasets['train']], axis=0)
    Y_train = np.concatenate([s[1] for s in datasets['train']], axis=0)
    OHLCV_train = np.concatenate([s[2] for s in datasets['train']], axis=0)

    X_val = np.concatenate([s[0] for s in datasets['val']], axis=0)
    Y_val = np.concatenate([s[1] for s in datasets['val']], axis=0)
    OHLCV_val = np.concatenate([s[2] for s in datasets['val']], axis=0)

    X_test = np.concatenate([s[0] for s in datasets['test']], axis=0)
    Y_test = np.concatenate([s[1] for s in datasets['test']], axis=0)
    OHLCV_test = np.concatenate([s[2] for s in datasets['test']], axis=0)

    logger.info(f"   Shapes concaténées:")
    logger.info(f"     Train: X={X_train.shape}, Y={Y_train.shape}, OHLCV={OHLCV_train.shape}")
    logger.info(f"     Val:   X={X_val.shape}, Y={Y_val.shape}, OHLCV={OHLCV_val.shape}")
    logger.info(f"     Test:  X={X_test.shape}, Y={Y_test.shape}, OHLCV={OHLCV_test.shape}")

    # Stats labels
    logger.info(f"\n   Balance labels régime:")
    for split_name, Y_split in [('Train', Y_train), ('Val', Y_val), ('Test', Y_test)]:
        # Y: [timestamp, asset_id, regime, ts_score, vc_score]
        regime_col = Y_split[:, 2].astype(int)
        regime_counts = pd.Series(regime_col).value_counts().sort_index()
        regime_pcts = (regime_counts / len(regime_col) * 100).round(1)

        logger.info(f"     {split_name}:")
        for regime_id, pct in regime_pcts.items():
            regime_name = {
                0: "RANGE LOW VOL",
                1: "RANGE HIGH VOL",
                2: "TREND LOW VOL",
                3: "TREND HIGH VOL"
            }.get(regime_id, f"UNKNOWN_{regime_id}")
            logger.info(f"       Régime {regime_id} ({regime_name}): {pct}%")

    # ========================================================================
    # SAUVEGARDE
    # ========================================================================

    if args.output_dir is None:
        output_dir = Path('data/prepared')
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    assets_str = '_'.join(args.assets).lower()
    output_path = output_dir / f"dataset_{assets_str}_regime.npz"

    # Features
    feature_cols = get_regime_feature_names()

    # Metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'assets': args.assets,
        'n_assets': len(args.assets),
        'sequence_length': SEQUENCE_LENGTH,
        'features': feature_cols,
        'n_features': len(feature_cols),
        'labels': ['regime', 'trend_strength', 'volatility_cluster'],
        'n_classes': 4,
        'regime_definition': {
            0: "RANGE LOW VOL (TS < 0.4, VC ≤ P70)",
            1: "RANGE HIGH VOL (TS < 0.4, VC > P70)",
            2: "TREND LOW VOL (TS > 0.6, VC ≤ P70)",
            3: "TREND HIGH VOL (TS > 0.6, VC > P70)"
        },
        'clip_value': args.clip,
        'max_samples_per_asset': args.max_samples,
        'splits': {
            'train': {'n_sequences': len(X_train), 'ratio': 0.70},
            'val': {'n_sequences': len(X_val), 'ratio': 0.15},
            'test': {'n_sequences': len(X_test), 'ratio': 0.15}
        }
    }

    # Sauvegarder .npz
    np.savez_compressed(
        output_path,
        X_train=X_train, Y_train=Y_train, OHLCV_train=OHLCV_train,
        X_val=X_val, Y_val=Y_val, OHLCV_val=OHLCV_val,
        X_test=X_test, Y_test=Y_test, OHLCV_test=OHLCV_test
    )

    # Sauvegarder metadata JSON
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\n✅ DATASET SAUVEGARDÉ:")
    logger.info(f"   NPZ:      {output_path}")
    logger.info(f"   Metadata: {metadata_path}")
    logger.info(f"   Taille:   {output_path.stat().st_size / (1024**2):.1f} MB")

    logger.info("\n" + "="*80)
    logger.info("✓ PRÉPARATION TERMINÉE")
    logger.info("="*80)
    logger.info("\nProchaine étape:")
    logger.info("  python src/train_regime_classifier.py \\")
    logger.info(f"    --data {output_path}")


if __name__ == '__main__':
    main()
