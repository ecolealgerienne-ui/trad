#!/usr/bin/env python3
"""
Training script for multi-timeframe pilot model Net_macd_30m.

Reads CSV files from prepare_multitf_csv.py pipeline.
Trains a CNN-LSTM binary classifier to predict oracle_label_{ind}_{tf}
from causal live features.

Pilot: Net_macd_30m
  Features: macd_30m_live, macd_30m_filtered (2 columns)
  Target:   oracle_label_macd_30m (binary 0/1)

Pipeline:
  1. Load 5 asset CSVs
  2. Drop warm-up NaN rows per asset
  3. Split chronological 70/15/15 with gap=25 per asset
  4. Z-score normalization per asset per feature (stats from TRAIN ONLY)
  5. Create sequences (window=25) per asset
  6. Concatenate assets per split
  7. Train CNN-LSTM with BCEWithLogitsLoss, early stopping

Usage:
    python src/train_multitf.py --indicator macd --timeframe 30m --epochs 100
    python src/train_multitf.py --indicator macd --timeframe 30m --epochs 3 --assets BTC  # debug
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
import json
import argparse
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from constants import PREPARED_DATA_DIR

# Defaults (same as train.py)
WINDOW = 25
BATCH_SIZE = 128
LR = 0.0001
EPOCHS = 100
PATIENCE = 10
GRAD_CLIP = 1.0
SEED = 42

ASSET_CSV_MAP = {
    'BTC': 'BTCUSD', 'ETH': 'ETHUSD', 'BNB': 'BNBUSD',
    'ADA': 'ADAUSD', 'LTC': 'LTCUSD',
}


# =============================================================================
# DATA LOADING
# =============================================================================

def find_csv(asset_name, indicator):
    """Find the multitf CSV for an asset. Tries multiple naming conventions."""
    base = ASSET_CSV_MAP[asset_name]
    candidates = [
        f'{PREPARED_DATA_DIR}/{base}_multitf_macd_rsi_cci.csv',
        f'{PREPARED_DATA_DIR}/{base}_multitf_{indicator}.csv',
        f'{PREPARED_DATA_DIR}/{base}_multitf.csv',
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    raise FileNotFoundError(
        f"No CSV found for {asset_name}. Tried: {candidates}\n"
        f"Run: python src/prepare_multitf_csv.py --assets {asset_name}")


def load_asset_data(asset_name, indicator, timeframe):
    """
    Load CSV and extract feature + label columns for one asset.

    Returns:
        DataFrame with columns: feature_0, feature_1, ..., label
        Index: DatetimeIndex
    """
    csv_path = find_csv(asset_name, indicator)
    df = pd.read_csv(csv_path, parse_dates=['datetime']).set_index('datetime').sort_index()

    feature_cols = [f'{indicator}_{timeframe}_live', f'{indicator}_{timeframe}_filtered']

    # Add velocity as 3rd feature if available
    vel_col = f'{indicator}_{timeframe}_velocity'
    if vel_col in df.columns:
        feature_cols.append(vel_col)

    label_col = f'oracle_label_{indicator}_{timeframe}'

    missing = [c for c in feature_cols + [label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    # Extract only needed columns
    n_features = len(feature_cols)
    result = df[feature_cols + [label_col]].copy()
    new_col_names = [f'feature_{i}' for i in range(n_features)] + ['label']
    result.columns = new_col_names

    # Drop warm-up NaN rows (features NaN at the start)
    n_before = len(result)
    result = result.dropna()
    n_dropped = n_before - len(result)

    logger.info(f"  {asset_name}: {len(result):,} rows (dropped {n_dropped:,} warm-up NaN)")

    return result


# =============================================================================
# SPLIT + NORMALIZATION + SEQUENCES
# =============================================================================

def split_chronological(df, train_ratio=0.70, val_ratio=0.15, gap=WINDOW):
    """
    Split DataFrame chronologically with gap between splits.
    Gap ensures no sequence in val contains bars from train.

    Returns:
        (df_train, df_val, df_test)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    df_train = df.iloc[:train_end - gap]
    df_val = df.iloc[train_end:val_end - gap]
    df_test = df.iloc[val_end:]

    return df_train, df_val, df_test


def compute_norm_stats(df_train, feature_cols):
    """
    Compute z-score normalization stats from TRAIN split ONLY.

    Returns:
        dict: {col: {'mean': float, 'std': float}}
    """
    stats = {}
    for col in feature_cols:
        mean = df_train[col].mean()
        std = df_train[col].std()
        if std < 1e-10:
            std = 1.0  # Avoid division by zero
            logger.warning(f"    WARNING: {col} has near-zero std, using 1.0")
        stats[col] = {'mean': float(mean), 'std': float(std)}
    return stats


def apply_norm(df, stats):
    """Apply z-score normalization using pre-computed stats."""
    df = df.copy()
    for col, s in stats.items():
        if col in df.columns:
            df[col] = (df[col] - s['mean']) / s['std']
    return df


def create_sequences(df, window=WINDOW):
    """
    Create sliding window sequences from DataFrame.

    X[i] = features at steps [i, i+1, ..., i+window-1]
    y[i] = label at step i+window-1 (last step of the window)

    Returns:
        X: (n_sequences, window, n_features) float32
        y: (n_sequences,) int
    """
    feat_cols = [c for c in df.columns if c.startswith('feature_')]
    features = df[feat_cols].values.astype(np.float32)
    labels = df['label'].values.astype(np.int64)

    n = len(df)
    n_feat = features.shape[1]
    if n < window:
        return np.empty((0, window, n_feat), dtype=np.float32), np.empty((0,), dtype=np.int64)

    # Sliding window (vectorized)
    indices = np.arange(window)[None, :] + np.arange(n - window + 1)[:, None]
    X = features[indices]  # (n_seq, window, n_feat)
    y = labels[window - 1:]  # label at last step of each window

    return X, y


def prepare_all_assets(assets, indicator, timeframe):
    """
    Full pipeline: load → split → normalize → sequences → concatenate.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test (numpy arrays)
        norm_stats: dict of per-asset normalization stats
        metadata: dict with pipeline info
    """
    feature_cols = [c for c in df_train.columns if c.startswith('feature_')]
    all_norm_stats = {}

    splits = {'train': [], 'val': [], 'test': []}

    for asset in assets:
        logger.info(f"\n  --- {asset} ---")

        # Load
        df = load_asset_data(asset, indicator, timeframe)

        # Split
        df_train, df_val, df_test = split_chronological(df)
        logger.info(f"    Split: train={len(df_train):,}, val={len(df_val):,}, test={len(df_test):,}")

        # Normalize (stats from TRAIN ONLY)
        stats = compute_norm_stats(df_train, feature_cols)
        all_norm_stats[asset] = stats
        logger.info(f"    Norm stats (train): "
                    f"f0 mean={stats['feature_0']['mean']:.6f} std={stats['feature_0']['std']:.6f}, "
                    f"f1 mean={stats['feature_1']['mean']:.6f} std={stats['feature_1']['std']:.6f}")

        df_train = apply_norm(df_train, stats)
        df_val = apply_norm(df_val, stats)
        df_test = apply_norm(df_test, stats)

        # Verify train is normalized
        t_mean_0 = df_train['feature_0'].mean()
        t_std_0 = df_train['feature_0'].std()
        t_mean_1 = df_train['feature_1'].mean()
        t_std_1 = df_train['feature_1'].std()
        logger.info(f"    Post-norm train: f0 mean={t_mean_0:.4f} std={t_std_0:.4f}, "
                    f"f1 mean={t_mean_1:.4f} std={t_std_1:.4f}")

        # Label distribution in train
        n_up = (df_train['label'] == 1).sum()
        n_down = (df_train['label'] == 0).sum()
        logger.info(f"    Train labels: {n_up:,} UP ({n_up/(n_up+n_down)*100:.1f}%), {n_down:,} DOWN")

        # Create sequences per asset (no cross-asset sequences)
        for split_name, df_split in [('train', df_train), ('val', df_val), ('test', df_test)]:
            X, y = create_sequences(df_split)
            splits[split_name].append((X, y))
            logger.info(f"    {split_name} sequences: {len(X):,}")

    # Concatenate all assets per split
    X_train = np.concatenate([s[0] for s in splits['train']])
    y_train = np.concatenate([s[1] for s in splits['train']])
    X_val = np.concatenate([s[0] for s in splits['val']])
    y_val = np.concatenate([s[1] for s in splits['val']])
    X_test = np.concatenate([s[0] for s in splits['test']])
    y_test = np.concatenate([s[1] for s in splits['test']])

    logger.info(f"\n  Concatenated:")
    logger.info(f"    Train: X={X_train.shape}, y={y_train.shape} (UP={y_train.mean()*100:.1f}%)")
    logger.info(f"    Val:   X={X_val.shape}, y={y_val.shape} (UP={y_val.mean()*100:.1f}%)")
    logger.info(f"    Test:  X={X_test.shape}, y={y_test.shape} (UP={y_test.mean()*100:.1f}%)")

    metadata = {
        'indicator': indicator,
        'timeframe': timeframe,
        'feature_names': [f'{indicator}_{timeframe}_live', f'{indicator}_{timeframe}_filtered'],
        'target_name': f'oracle_label_{indicator}_{timeframe}',
        'assets': assets,
        'window': WINDOW,
        'gap': WINDOW,
        'train_ratio': 0.70,
        'val_ratio': 0.15,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
    }

    return X_train, y_train, X_val, y_val, X_test, y_test, all_norm_stats, metadata


# =============================================================================
# MODEL (CNN-LSTM, uniform architecture, no conditional logic)
# =============================================================================

class CNNLSTMClassifier(nn.Module):
    """
    CNN-LSTM binary classifier. Uniform architecture for all indicators.

    Input:  (batch, window=25, n_features=2)
    Output: (batch, 1) — raw logits (no sigmoid, use BCEWithLogitsLoss)
    """

    def __init__(self, n_features=2, window=25,
                 cnn_filters=64, cnn_kernel=3,
                 lstm_hidden=64, lstm_layers=2, lstm_dropout=0.2,
                 dense_hidden=32, dense_dropout=0.3):
        super().__init__()

        # CNN: extract local patterns
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_filters, kernel_size=cnn_kernel, padding=cnn_kernel // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # LayerNorm: stabilize between CNN and LSTM
        self.layer_norm = nn.LayerNorm(cnn_filters)

        # LSTM: capture temporal dependencies
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
            batch_first=True,
        )

        # Dense head: last LSTM output → logit
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, dense_hidden),
            nn.ReLU(),
            nn.Dropout(dense_dropout),
            nn.Linear(dense_hidden, 1),  # Raw logit, NO sigmoid
        )

    def forward(self, x):
        """
        Args:
            x: (batch, window, n_features)
        Returns:
            logits: (batch, 1) — raw, apply sigmoid externally if needed
        """
        # CNN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, n_features, window)
        x = self.cnn(x)        # (batch, cnn_filters, window)
        x = x.transpose(1, 2)  # (batch, window, cnn_filters)

        # LayerNorm
        x = self.layer_norm(x)

        # LSTM
        x, _ = self.lstm(x)    # (batch, window, lstm_hidden)
        x = x[:, -1, :]        # Last timestep: (batch, lstm_hidden)

        # Head
        return self.head(x)    # (batch, 1)


# =============================================================================
# DATASET
# =============================================================================

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)  # (n, 1) for BCEWithLogitsLoss

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# TRAINING
# =============================================================================

def train_one_epoch(model, loader, loss_fn, optimizer, device, grad_clip):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()

        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * len(X_batch)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += len(X_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)

        total_loss += loss.item() * len(X_batch)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += len(X_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def generate_predictions(model, X, device, batch_size=512):
    """Generate probability predictions for a dataset."""
    model.eval()
    ds = SequenceDataset(X, np.zeros(len(X)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    for X_batch, _ in loader:
        logits = model(X_batch.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        preds.append(probs)
    return np.concatenate(preds).squeeze()


def train_model(model, train_loader, val_loader, loss_fn, optimizer, device,
                epochs, patience, grad_clip, save_path):
    """Full training loop with early stopping."""
    best_val_loss = float('inf')
    best_epoch = 0
    no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device, grad_clip)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        logger.info(f"  Epoch {epoch:3d}/{epochs} — "
                    f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
                    f"Val loss={val_loss:.4f} acc={val_acc:.4f}"
                    f"{' *' if val_loss < best_val_loss else ''}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    history['best_epoch'] = best_epoch
    history['best_val_loss'] = best_val_loss
    logger.info(f"\n  Best: epoch {best_epoch}, val_loss={best_val_loss:.4f}")
    return history


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train multi-timeframe pilot model')
    parser.add_argument('--indicator', default='macd', choices=['macd', 'rsi', 'cci'])
    parser.add_argument('--timeframe', default='30m', choices=['30m', '1h'])
    parser.add_argument('--assets', nargs='+', default=['BTC', 'ETH', 'BNB', 'ADA', 'LTC'])
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--grad-clip', type=float, default=GRAD_CLIP)
    parser.add_argument('--cnn-filters', type=int, default=64)
    parser.add_argument('--lstm-hidden', type=int, default=64)
    parser.add_argument('--lstm-layers', type=int, default=2)
    parser.add_argument('--lstm-dropout', type=float, default=0.2)
    parser.add_argument('--dense-hidden', type=int, default=32)
    parser.add_argument('--dense-dropout', type=float, default=0.3)
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if device == 'auto':
        device = 'cpu'

    model_name = f'{args.indicator}_{args.timeframe}'

    logger.info("=" * 60)
    logger.info(f"TRAINING Net_{model_name}")
    logger.info("=" * 60)
    logger.info(f"Features: {args.indicator}_{args.timeframe}_live, {args.indicator}_{args.timeframe}_filtered")
    logger.info(f"Target:   oracle_label_{args.indicator}_{args.timeframe}")
    logger.info(f"Assets:   {args.assets}")
    logger.info(f"Device:   {device}")
    logger.info(f"Epochs:   {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    # =========================================================================
    # 1. Load + Split + Normalize + Sequences
    # =========================================================================
    logger.info("\n1. Data preparation...")
    X_train, y_train, X_val, y_val, X_test, y_test, norm_stats, metadata = \
        prepare_all_assets(args.assets, args.indicator, args.timeframe)

    # Save norm stats
    norm_path = f'{PREPARED_DATA_DIR}/norm_stats_{model_name}.json'
    Path(norm_path).parent.mkdir(parents=True, exist_ok=True)
    with open(norm_path, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    logger.info(f"\n  Norm stats saved: {norm_path}")

    # =========================================================================
    # 2. DataLoaders
    # =========================================================================
    logger.info("\n2. Creating DataLoaders...")
    train_loader = DataLoader(
        SequenceDataset(X_train, y_train),
        batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=(device == 'cuda'))
    val_loader = DataLoader(
        SequenceDataset(X_val, y_val),
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=(device == 'cuda'))
    logger.info(f"  Train: {len(train_loader):,} batches, Val: {len(val_loader):,} batches")

    # =========================================================================
    # 3. Model
    # =========================================================================
    logger.info("\n3. Creating model...")
    n_features = X_train.shape[2]
    logger.info(f"  Detected {n_features} features")
    model = CNNLSTMClassifier(
        n_features=n_features, window=WINDOW,
        cnn_filters=args.cnn_filters, lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers, lstm_dropout=args.lstm_dropout,
        dense_hidden=args.dense_hidden, dense_dropout=args.dense_dropout,
    ).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {n_params:,}")
    logger.info(f"  Loss: BCEWithLogitsLoss (no sigmoid in model)")

    # =========================================================================
    # 4. Train
    # =========================================================================
    save_path = f'models/best_model_{model_name}.pth'
    logger.info(f"\n4. Training (save: {save_path})...")

    history = train_model(
        model, train_loader, val_loader, loss_fn, optimizer, device,
        args.epochs, args.patience, args.grad_clip, save_path)

    # Save history
    hist_path = f'models/training_history_{model_name}.json'
    Path(hist_path).parent.mkdir(parents=True, exist_ok=True)
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"  History saved: {hist_path}")

    # =========================================================================
    # 5. Generate predictions + save NPZ
    # =========================================================================
    logger.info("\n5. Generating predictions...")

    # Reload best model
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    y_train_pred = generate_predictions(model, X_train, device)
    y_val_pred = generate_predictions(model, X_val, device)
    y_test_pred = generate_predictions(model, X_test, device)

    logger.info(f"  Train pred: mean={y_train_pred.mean():.4f}, std={y_train_pred.std():.4f}")
    logger.info(f"  Val pred:   mean={y_val_pred.mean():.4f}, std={y_val_pred.std():.4f}")
    logger.info(f"  Test pred:  mean={y_test_pred.mean():.4f}, std={y_test_pred.std():.4f}")

    # Save NPZ
    npz_path = f'{PREPARED_DATA_DIR}/{model_name}_dataset.npz'
    np.savez_compressed(npz_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        y_train_pred=y_train_pred,
        y_val_pred=y_val_pred,
        y_test_pred=y_test_pred,
        metadata=metadata)
    logger.info(f"  NPZ saved: {npz_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info(f"DONE — Net_{model_name}")
    logger.info("=" * 60)
    logger.info(f"  Best epoch: {history['best_epoch']}")
    logger.info(f"  Best val loss: {history['best_val_loss']:.4f}")
    logger.info(f"  Best val acc:  {history['val_acc'][history['best_epoch']-1]:.4f}")
    logger.info(f"  Model: {save_path}")
    logger.info(f"  NPZ:   {npz_path}")
    logger.info(f"  Norms:  {norm_path}")
    logger.info(f"  History: {hist_path}")


if __name__ == '__main__':
    main()
