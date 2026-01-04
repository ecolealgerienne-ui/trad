"""
Script de préparation des données OHLC v2 - Avec synchronisation par index.

Approche rigoureuse:
1. Charger données brutes avec DatetimeIndex
2. Calculer indicateurs → ajouter au DataFrame
3. Calculer filtre → ajouter au DataFrame
4. Calculer labels → ajouter au DataFrame
5. Créer séquences avec vérification des index

Toutes les données restent synchronisées via l'index datetime.

Usage:
    python src/prepare_data_ohlc_v2.py --target close --assets BTC ETH BNB ADA LTC
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Import modules locaux
from constants import (
    AVAILABLE_ASSETS_5M, DEFAULT_ASSETS,
    TRIM_EDGES,
    PREPARED_DATA_DIR,
    SEQUENCE_LENGTH,
)

# Périodes STANDARD des indicateurs
RSI_PERIOD = 14
CCI_PERIOD = 20
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Configuration
OCTAVE_STEP = 0.20


# =============================================================================
# FILTRE OCTAVE
# =============================================================================

def octave_filter(data: np.ndarray, step: float = OCTAVE_STEP) -> np.ndarray:
    """Applique le filtre Octave (Butterworth ordre 3 + filtfilt)."""
    B, A = signal.butter(3, step, output='ba')
    filtered = signal.filtfilt(B, A, data)
    return filtered


# =============================================================================
# CHARGEMENT DONNÉES AVEC INDEX
# =============================================================================

def load_data_with_index(file_path: str, asset_name: str = "Asset") -> pd.DataFrame:
    """
    Charge les données CSV avec DatetimeIndex.

    Returns:
        DataFrame avec colonnes: open, high, low, close, volume
        Index: DatetimeIndex
    """
    df = pd.read_csv(file_path)

    # Identifier la colonne date
    date_col = None
    for col in ['date', 'datetime', 'time', 'timestamp', 'Date', 'Datetime']:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        raise ValueError(f"Colonne date non trouvée dans {file_path}")

    # Convertir en datetime et définir comme index
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = 'datetime'

    # Normaliser noms colonnes
    df.columns = df.columns.str.lower()

    # Vérifier colonnes requises
    required = ['open', 'high', 'low', 'close']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    # Trier par date
    df = df.sort_index()

    logger.info(f"  {asset_name}: {len(df):,} lignes, "
                f"{df.index[0]} → {df.index[-1]}")

    return df


# =============================================================================
# CALCUL INDICATEURS (ajoutés au DataFrame)
# =============================================================================

def add_indicators_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les indicateurs et les ajoute directement au DataFrame.
    Garde la synchronisation via l'index.
    """
    df = df.copy()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(span=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(span=RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    # CCI
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(CCI_PERIOD).mean()
    mad = tp.rolling(CCI_PERIOD).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (tp - sma_tp) / (0.015 * mad)

    # MACD
    ema_fast = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df['macd'] = macd_line - signal_line  # Histogram

    return df


# =============================================================================
# NORMALISATION OHLC (ajoutée au DataFrame)
# =============================================================================

def add_ohlc_features_to_df(df: pd.DataFrame, clip_value: float = 0.10) -> pd.DataFrame:
    """
    Calcule les features OHLC normalisées et les ajoute au DataFrame.
    """
    df = df.copy()

    prev_close = df['close'].shift(1)

    # 5 canaux OHLC normalisés
    df['o_ret'] = (df['open'] - prev_close) / prev_close
    df['h_ret'] = (df['high'] - prev_close) / prev_close
    df['l_ret'] = (df['low'] - prev_close) / prev_close
    df['c_ret'] = (df['close'] - prev_close) / prev_close
    df['range_ret'] = (df['high'] - df['low']) / prev_close

    # Clipper les outliers
    for col in ['o_ret', 'h_ret', 'l_ret', 'c_ret', 'range_ret']:
        df[col] = df[col].clip(-clip_value, clip_value)

    return df


# =============================================================================
# FILTRE ET LABELS (ajoutés au DataFrame)
# =============================================================================

def add_filtered_and_labels_to_df(df: pd.DataFrame, target: str,
                                   octave_step: float = OCTAVE_STEP) -> pd.DataFrame:
    """
    Calcule le signal filtré et les labels, les ajoute au DataFrame.

    Labels: label[t] = 1 si filtered[t-1] > filtered[t-2] else 0

    Alignement vérifié:
    - À l'index t, on a accès aux features OHLC jusqu'à t
    - Le label[t] = filtered[t-1] > filtered[t-2] (pente PASSÉE)
    - Donc on prédit la pente entre t-2 et t-1 avec les données jusqu'à t
    """
    df = df.copy()

    # Sélectionner la colonne cible
    target = target.lower()
    if target == 'close':
        raw_signal = df['close'].values
    elif target == 'rsi':
        raw_signal = df['rsi'].values
    elif target == 'cci':
        raw_signal = df['cci'].values
    elif target in ['macd', 'macd26']:
        raw_signal = df['macd'].values
    else:
        raise ValueError(f"Target inconnu: {target}")

    # Appliquer filtre Octave
    # Note: filtfilt a besoin de données sans NaN
    valid_mask = ~np.isnan(raw_signal)
    filtered = np.full(len(raw_signal), np.nan)
    if valid_mask.sum() > 10:
        filtered[valid_mask] = octave_filter(raw_signal[valid_mask], octave_step)

    df['filtered'] = filtered

    # Calculer labels: filtered[t-1] > filtered[t-2]
    # shift(1) : valeur à t-1
    # shift(2) : valeur à t-2
    df['filtered_t1'] = df['filtered'].shift(1)
    df['filtered_t2'] = df['filtered'].shift(2)

    # Label = 1 si pente positive (filtered[t-1] > filtered[t-2])
    df['label'] = (df['filtered_t1'] > df['filtered_t2']).astype(int)

    # Log pour vérification avec exemple
    logger.info(f"     Filtered: min={df['filtered'].min():.4f}, max={df['filtered'].max():.4f}")
    logger.info(f"     Labels: {df['label'].sum()}/{len(df)} = {df['label'].mean()*100:.1f}% UP")

    # Afficher quelques exemples pour vérification
    sample_idx = min(100, len(df) - 1)
    logger.info(f"     Exemple idx={sample_idx}:")
    logger.info(f"       filtered[{sample_idx-2}]={df['filtered'].iloc[sample_idx-2]:.4f}")
    logger.info(f"       filtered[{sample_idx-1}]={df['filtered'].iloc[sample_idx-1]:.4f}")
    logger.info(f"       label[{sample_idx}]={df['label'].iloc[sample_idx]} (filtered[{sample_idx-1}] > filtered[{sample_idx-2}])")

    return df


# =============================================================================
# CRÉATION SÉQUENCES AVEC VÉRIFICATION INDEX
# =============================================================================

def create_sequences_with_index(df: pd.DataFrame,
                                 feature_cols: list,
                                 label_col: str = 'label',
                                 seq_length: int = SEQUENCE_LENGTH) -> tuple:
    """
    Crée les séquences X, Y avec conservation des index pour vérification.

    Alignement:
        Pour chaque séquence i:
        - X[i] = features[i-12:i] → indices i-12, i-11, ..., i-1 (12 éléments)
        - Y[i] = label[i] = filtered[i-1] > filtered[i-2]
        - idx_feature[i] = date de la DERNIÈRE feature (indice i-1)
        - idx_label[i] = date du label (indice i)

    Returns:
        X: np.array (n_sequences, seq_length, n_features)
        Y: np.array (n_sequences, 1)
        indices: list de tuples (idx_feature, idx_label) pour vérification
    """
    # Supprimer lignes avec NaN dans les colonnes utilisées
    cols_needed = feature_cols + [label_col]
    df_clean = df.dropna(subset=cols_needed)

    logger.info(f"     Lignes valides: {len(df_clean)}/{len(df)} "
                f"({len(df) - len(df_clean)} supprimées pour NaN)")

    # Extraire arrays
    features = df_clean[feature_cols].values
    labels = df_clean[label_col].values
    dates = df_clean.index.tolist()

    # Créer séquences
    X_list = []
    Y_list = []
    idx_list = []

    for i in range(seq_length, len(features)):
        # Séquence de features: indices [i-seq_length, i-1] → 12 éléments
        X_list.append(features[i-seq_length:i])
        # Label: à l'index i
        Y_list.append(labels[i])
        # Stocker DEUX indices pour vérification:
        # - idx_feature: date de la dernière feature (i-1)
        # - idx_label: date du label (i)
        idx_list.append((dates[i-1], dates[i]))

    X = np.array(X_list)
    Y = np.array(Y_list).reshape(-1, 1)

    return X, Y, idx_list


# =============================================================================
# SPLIT CHRONOLOGIQUE
# =============================================================================

def split_chronological(X: np.ndarray, Y: np.ndarray, indices: list,
                        train_ratio: float = 0.70,
                        val_ratio: float = 0.15,
                        gap: int = SEQUENCE_LENGTH) -> dict:
    """
    Split chronologique avec GAP entre train/val et val/test.

    Returns:
        dict avec 'train', 'val', 'test', chacun contenant (X, Y, indices)
    """
    n = len(X)

    # Calcul des indices de split
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Appliquer GAP
    train_end_gap = train_end - gap
    val_start = train_end
    val_end_gap = val_end - gap
    test_start = val_end

    return {
        'train': (X[:train_end_gap], Y[:train_end_gap], indices[:train_end_gap]),
        'val': (X[val_start:val_end_gap], Y[val_start:val_end_gap], indices[val_start:val_end_gap]),
        'test': (X[test_start:], Y[test_start:], indices[test_start:])
    }


# =============================================================================
# VÉRIFICATION COHÉRENCE
# =============================================================================

def verify_alignment(df: pd.DataFrame, X: np.ndarray, Y: np.ndarray,
                     indices: list, feature_cols: list) -> bool:
    """
    Vérifie que X, Y sont bien alignés avec le DataFrame original.

    indices contient des tuples (idx_feature, idx_label) où:
    - idx_feature: date de la dernière feature dans X
    - idx_label: date du label dans Y
    """
    logger.info("     Vérification alignement...")

    # Prendre quelques échantillons aléatoires
    n_checks = min(5, len(indices))
    check_indices = np.random.choice(len(indices), n_checks, replace=False)

    all_ok = True
    for i in check_indices:
        idx_feature, idx_label = indices[i]

        # Récupérer depuis DataFrame
        df_row_feature = df.loc[idx_feature]
        df_row_label = df.loc[idx_label]

        df_label = df_row_label['label']
        df_features = df_row_feature[feature_cols].values

        # Comparer avec X, Y
        y_val = Y[i, 0]
        x_last = X[i, -1, :]  # Dernière feature de la séquence

        # Vérifier label
        if y_val != df_label:
            logger.error(f"     ❌ Label mismatch at {idx_label}: Y={y_val}, df={df_label}")
            all_ok = False

        # Vérifier features (dernière ligne de X = features à idx_feature)
        if not np.allclose(x_last, df_features, rtol=1e-5, equal_nan=True):
            logger.error(f"     ❌ Features mismatch at {idx_feature}")
            logger.error(f"        X[-1]={x_last}")
            logger.error(f"        df={df_features}")
            all_ok = False

    if all_ok:
        logger.info(f"     ✅ Alignement vérifié sur {n_checks} échantillons")

        # Afficher un exemple pour confirmation
        idx_f, idx_l = indices[0]
        logger.info(f"     Exemple séquence 0:")
        logger.info(f"       Dernière feature: {idx_f}")
        logger.info(f"       Label: {idx_l} → Y={Y[0,0]}")
        logger.info(f"       (label = filtered[{idx_f}] > filtered[précédent])")

    return all_ok


# =============================================================================
# PRÉPARATION D'UN ASSET
# =============================================================================

def prepare_single_asset(file_path: str, asset_name: str, target: str,
                         octave_step: float = OCTAVE_STEP,
                         clip_value: float = 0.10) -> tuple:
    """
    Prépare les données pour un seul asset.

    Pipeline:
    1. Charger données brutes avec DatetimeIndex
    2. Calculer indicateurs → ajouter au df
    3. Calculer features OHLC → ajouter au df
    4. Calculer filtre + labels → ajouter au df
    5. TRIM edges (après tous les calculs pour éviter effets de bord)
    6. Créer séquences avec conservation des index
    7. Vérifier alignement

    Returns:
        (X, Y, indices, df) pour vérification
    """
    logger.info(f"\n  {asset_name}: Préparation...")

    # 1. Charger avec index
    df = load_data_with_index(file_path, asset_name)
    logger.info(f"     Chargé: {len(df)} lignes")

    # 2. Ajouter indicateurs
    df = add_indicators_to_df(df)
    logger.info(f"     Indicateurs ajoutés: RSI, CCI, MACD")

    # 3. Ajouter features OHLC
    df = add_ohlc_features_to_df(df, clip_value)
    logger.info(f"     Features OHLC ajoutées (clip ±{clip_value*100:.0f}%)")

    # 4. Ajouter filtre et labels
    df = add_filtered_and_labels_to_df(df, target, octave_step)

    # 5. TRIM edges (après tous les calculs pour éviter effets de bord du filtre)
    df = df.iloc[TRIM_EDGES:-TRIM_EDGES]
    logger.info(f"     Après trim ±{TRIM_EDGES}: {len(df)} lignes")

    # 6. Créer séquences
    feature_cols = ['o_ret', 'h_ret', 'l_ret', 'c_ret', 'range_ret']
    X, Y, indices = create_sequences_with_index(df, feature_cols, 'label')
    logger.info(f"     Séquences: X={X.shape}, Y={Y.shape}")

    # 7. Vérifier alignement
    verify_alignment(df, X, Y, indices, feature_cols)

    return X, Y, indices, df


# =============================================================================
# PRÉPARATION ET SAUVEGARDE
# =============================================================================

def prepare_and_save(target: str,
                     assets: list = None,
                     output_path: str = None,
                     octave_step: float = OCTAVE_STEP,
                     clip_value: float = 0.10) -> str:
    """
    Prépare les données OHLC normalisées et les sauvegarde.
    """
    if assets is None:
        assets = DEFAULT_ASSETS

    # Valider les assets
    invalid_assets = [a for a in assets if a not in AVAILABLE_ASSETS_5M]
    if invalid_assets:
        raise ValueError(f"Assets invalides: {invalid_assets}")

    logger.info("="*80)
    logger.info("PRÉPARATION DES DONNÉES OHLC v2 (avec synchronisation index)")
    logger.info("="*80)
    logger.info(f"Target: FL_{target.upper()} (Octave step={octave_step})")
    logger.info(f"Assets: {', '.join(assets)}")
    logger.info(f"Formule labels: filtered[t-1] > filtered[t-2]")

    # Préparer chaque asset
    all_splits = {'train': [], 'val': [], 'test': []}

    for asset_name in assets:
        file_path = AVAILABLE_ASSETS_5M[asset_name]
        X, Y, indices, df = prepare_single_asset(
            file_path, asset_name, target, octave_step, clip_value
        )

        # Split chronologique
        splits = split_chronological(X, Y, indices)

        for split_name in ['train', 'val', 'test']:
            all_splits[split_name].append(splits[split_name])

        logger.info(f"     Split: Train={len(splits['train'][0])}, "
                   f"Val={len(splits['val'][0])}, Test={len(splits['test'][0])}")

    # Concaténer tous les assets
    X_train = np.concatenate([s[0] for s in all_splits['train']], axis=0)
    Y_train = np.concatenate([s[1] for s in all_splits['train']], axis=0)
    X_val = np.concatenate([s[0] for s in all_splits['val']], axis=0)
    Y_val = np.concatenate([s[1] for s in all_splits['val']], axis=0)
    X_test = np.concatenate([s[0] for s in all_splits['test']], axis=0)
    Y_test = np.concatenate([s[1] for s in all_splits['test']], axis=0)

    logger.info(f"\n📊 Shapes finales:")
    logger.info(f"   Train: X={X_train.shape}, Y={Y_train.shape}")
    logger.info(f"   Val:   X={X_val.shape}, Y={Y_val.shape}")
    logger.info(f"   Test:  X={X_test.shape}, Y={Y_test.shape}")

    # Stats labels
    logger.info(f"\n📈 Balance labels:")
    logger.info(f"   Train: {Y_train.mean()*100:.1f}% UP")
    logger.info(f"   Val:   {Y_val.mean()*100:.1f}% UP")
    logger.info(f"   Test:  {Y_test.mean()*100:.1f}% UP")

    # Sauvegarder
    if output_path is None:
        assets_str = '_'.join(assets).lower()
        output_path = f"data/prepared/dataset_{assets_str}_ohlcv2_{target}_octave{int(octave_step*100)}.npz"

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        'created_at': datetime.now().isoformat(),
        'version': 'v2_with_index_sync',
        'assets': assets,
        'target': target,
        'filter_type': 'octave',
        'octave_step': octave_step,
        'label_formula': 'filtered[t-1] > filtered[t-2]',
        'clip_value': clip_value,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'sequence_length': SEQUENCE_LENGTH,
        'n_features': 5,
        'features': ['o_ret', 'h_ret', 'l_ret', 'c_ret', 'range_ret'],
    }

    np.savez_compressed(
        output_path,
        X_train=X_train, Y_train=Y_train,
        X_val=X_val, Y_val=Y_val,
        X_test=X_test, Y_test=Y_test,
        metadata=json.dumps(metadata)
    )

    logger.info(f"\n✅ Données sauvegardées: {output_path}")

    # Sauvegarder métadonnées
    metadata_path = str(output_path).replace('.npz', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Point d'entrée CLI."""
    available_assets = list(AVAILABLE_ASSETS_5M.keys())
    available_targets = ['close', 'rsi', 'cci', 'macd']

    parser = argparse.ArgumentParser(
        description="Prépare les datasets OHLC v2 avec synchronisation par index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Exemples:
  python src/prepare_data_ohlc_v2.py --target close --assets BTC ETH BNB ADA LTC
  python src/prepare_data_ohlc_v2.py --target macd --assets BTC ETH

Targets disponibles: {', '.join(available_targets)}
Assets disponibles: {', '.join(available_assets)}
        """
    )

    parser.add_argument('--target', '-t', type=str, required=True,
                        choices=available_targets,
                        help=f'Indicateur cible')
    parser.add_argument('--assets', '-a', type=str, nargs='+',
                        default=DEFAULT_ASSETS,
                        choices=available_assets,
                        help=f'Assets à inclure')
    parser.add_argument('--octave-step', type=float, default=OCTAVE_STEP,
                        help=f'Paramètre du filtre Octave')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Chemin de sortie')
    parser.add_argument('--clip', type=float, default=0.10,
                        help='Valeur de clipping')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    output_path = prepare_and_save(
        target=args.target,
        assets=args.assets,
        output_path=args.output,
        octave_step=args.octave_step,
        clip_value=args.clip
    )

    print(f"\n✅ Terminé! Dataset: {output_path}")
    print(f"\nPour entraîner:")
    print(f"  python src/train.py --data {output_path} --indicator {args.target}")


if __name__ == '__main__':
    main()
