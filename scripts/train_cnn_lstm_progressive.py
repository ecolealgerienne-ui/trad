#!/usr/bin/env python3
"""
Entraîne un CNN-LSTM binaire sur le dataset progressif, en remplacement
du XGBoost. Pipeline strictement identique :
  - Même NPZ d'entrée : dataset_<ind>_<tf>_<period>_progressive.npz
  - Même format de sortie : preds_<ind>_<tf>_<period>_progressive_cnnlstm.npz
  - Compatible avec scripts/backtest_progressive.py
  - Compatible avec scripts/cross_validation_indicators.py (préfixe différent)

Différence vs train_progressive.py (XGBoost) :
  - Construit des séquences (rolling window) à partir de X tabulaire (n, 2)
  - X séquentiel : (n', window, 2) avec n' = n (padding début par copie de
    la première ligne, préserve l'alignement avec dates/closes/y)
  - Modèle PyTorch simple : Conv1D → LayerNorm → LSTM → Dense

Architecture (2 features slope_progressive + step_k) :
  Input (batch, window, 2)
    → Conv1D(32 filters, kernel=3, padding=same) + LayerNorm + ReLU + Dropout
    → LSTM(32 hidden, num_layers=2)
    → Dense(32 → 1) + BCEWithLogitsLoss
  Output : proba UP après sigmoid

Usage :
    python scripts/train_cnn_lstm_progressive.py --npz data/prepared/dataset_macd_30m_full_progressive.npz
    python scripts/train_cnn_lstm_progressive.py --window 12 --epochs 20 --batch-size 512
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("❌ PyTorch non installé. pip install torch")
    sys.exit(1)

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

PREP_DIR = Path('data/prepared')
MODELS_DIR = Path('models')


def parse_npz_path(npz_path):
    """Extrait (indicator, tf_label, period_tag, filter_tag) du nom du NPZ.

    Reconnaît tous les suffixes optionnels après '_progressive' :
      dataset_<ind>_<tf>_<period>_progressive.npz                   → filter_tag=''
      dataset_<ind>_<tf>_<period>_progressive_adaptive.npz          → filter_tag='_adaptive'
      dataset_<ind>_<tf>_<period>_progressive_lag0.npz              → filter_tag='_lag0'
      dataset_<ind>_<tf>_<period>_progressive_adaptive_lag0.npz     → filter_tag='_adaptive_lag0'
    """
    name = npz_path.stem
    if not name.startswith('dataset_'):
        return None, None, None, None
    if '_progressive' not in name:
        return None, None, None, None
    core_part, _, filter_part = name.partition('_progressive')
    core = core_part[len('dataset_'):]
    parts = core.split('_')
    if len(parts) != 3:
        return None, None, None, None
    return parts[0], parts[1], parts[2], filter_part


def build_sequences(X_tab, window):
    """
    Transforme X tabulaire (n, n_feat) en séquences (n, window, n_feat).

    Padding : les `window-1` premières lignes utilisent la 1ère ligne répétée
    (préserve l'alignement avec dates/closes/y, pas de perte de rows).

    Pour chaque i dans [0, n) :
        X_seq[i] = X_tab[max(0, i-window+1) : i+1]  padded si besoin
    """
    n, n_feat = X_tab.shape
    X_seq = np.empty((n, window, n_feat), dtype=np.float32)
    # Pad début avec X_tab[0] répété
    for i in range(n):
        start = max(0, i - window + 1)
        actual = X_tab[start:i + 1]  # (window ou moins, n_feat)
        if len(actual) < window:
            pad = np.tile(X_tab[0], (window - len(actual), 1))
            actual = np.concatenate([pad, actual], axis=0)
        X_seq[i] = actual
    return X_seq


class CNNLSTMProgressive(nn.Module):
    """Petit CNN-LSTM pour features progressives 2 canaux."""

    def __init__(self, n_features=2, cnn_filters=32, lstm_hidden=32,
                 lstm_layers=2, dense_hidden=32, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv1d(n_features, cnn_filters, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(cnn_filters)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(cnn_filters, lstm_hidden, num_layers=lstm_layers,
                              batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        self.fc1 = nn.Linear(lstm_hidden, dense_hidden)
        self.fc2 = nn.Linear(dense_hidden, 1)

    def forward(self, x):
        # x : (batch, window, n_features)
        x = x.transpose(1, 2)              # (batch, n_features, window)
        x = self.conv(x)                   # (batch, cnn_filters, window)
        x = x.transpose(1, 2)              # (batch, window, cnn_filters)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        _, (h_n, _) = self.lstm(x)         # h_n : (num_layers, batch, lstm_hidden)
        x = h_n[-1]                        # (batch, lstm_hidden)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                    # (batch, 1) logit
        return x.squeeze(-1)


def predict_proba(model, X, device, batch_size=2048):
    """Predict proba UP (sigmoid du logit)."""
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i+batch_size]).to(device)
            logits = model(batch)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', required=True,
                        help='Path NPZ progressif')
    parser.add_argument('--window', type=int, default=24,
                        help='Fenêtre de séquences en rows 5min (default 24 = 2h)')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--early-stop', type=int, default=3,
                        help='Patience val loss (0 = désactivé)')
    parser.add_argument('--cnn-filters', type=int, default=32)
    parser.add_argument('--lstm-hidden', type=int, default=32)
    parser.add_argument('--lstm-layers', type=int, default=2)
    parser.add_argument('--dense-hidden', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--pnl-per-row', default=None,
                        help='Option A — Path NPZ pnl_per_row (généré par '
                             'analyze_oracle_trade_thresholds.py --save-pnl-per-row). '
                             'Si fourni + --pnl-threshold > 0 → active sample weighting.')
    parser.add_argument('--pnl-threshold', type=float, default=0.0,
                        help='Option A — Seuil absolu |pnl_net| (default 0.0 = désactivé). '
                             'Ex: 0.002 = garde que les rows où le trade Oracle a '
                             '|PnL net| >= 0.2%% (filtre trades marginaux).')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    npz_path = Path(args.npz)
    if not npz_path.exists():
        print(f"❌ NPZ introuvable: {npz_path}")
        return
    indicator, tf_label, period_tag, filter_tag = parse_npz_path(npz_path)
    if indicator is None:
        print(f"❌ Impossible de parser: {npz_path.name}")
        return

    print("=" * 80)
    print(f"TRAIN CNN-LSTM PROGRESSIVE — {indicator.upper()} × {tf_label}  "
          f"period={period_tag}")
    print(f"  device={device}  window={args.window}  batch={args.batch_size}  "
          f"lr={args.learning_rate}  epochs={args.epochs}")
    print("=" * 80)

    # [1] Load NPZ
    print(f"\n[1/6] Load NPZ: {npz_path}")
    ds = np.load(npz_path, allow_pickle=True)
    X_train_tab = ds['X_train']
    y_train = ds['y_train_binary'].astype(np.float32)
    X_val_tab = ds['X_val']
    y_val = ds['y_val_binary'].astype(np.float32)
    X_test_tab = ds['X_test']
    y_test = ds['y_test_binary'].astype(np.float32)
    feature_cols = [str(c) for c in ds['feature_cols']]
    print(f"   Features: {feature_cols}")
    print(f"   X_train tab: {X_train_tab.shape}  |  "
          f"X_val tab: {X_val_tab.shape}  |  X_test tab: {X_test_tab.shape}")

    # [1.5] Option A : sample weighting via pnl_per_row (optionnel)
    use_sample_weighting = (args.pnl_per_row is not None
                             and args.pnl_threshold > 0)
    sw_tag = ''
    if use_sample_weighting:
        pnl_path = Path(args.pnl_per_row)
        if not pnl_path.exists():
            print(f"❌ pnl_per_row NPZ introuvable: {pnl_path}")
            return
        pnl_npz = np.load(pnl_path, allow_pickle=True)
        pnl_train = pnl_npz['pnl_per_row_train']
        pnl_val = pnl_npz['pnl_per_row_val']
        assert len(pnl_train) == len(X_train_tab), \
            f"Mismatch pnl_train ({len(pnl_train)}) vs X_train ({len(X_train_tab)})"
        assert len(pnl_val) == len(X_val_tab), \
            f"Mismatch pnl_val ({len(pnl_val)}) vs X_val ({len(X_val_tab)})"
        w_train = (np.abs(pnl_train) >= args.pnl_threshold).astype(np.float32)
        w_val = (np.abs(pnl_val) >= args.pnl_threshold).astype(np.float32)
        n_train_kept = int(w_train.sum())
        n_val_kept = int(w_val.sum())
        # Tag pour noms de fichiers
        sw_str = f'{args.pnl_threshold:.4f}'.rstrip('0').rstrip('.')
        sw_tag = f'_sw{sw_str.replace(".", "p")}'
        print(f"\n[1.5/6] Option A ACTIVÉE  pnl_threshold={args.pnl_threshold*100:.3f}%  "
              f"(suffix={sw_tag})")
        print(f"   Train : {n_train_kept:,}/{len(w_train):,} rows kept "
              f"({n_train_kept/len(w_train)*100:.2f}%)")
        print(f"   Val   : {n_val_kept:,}/{len(w_val):,} rows kept "
              f"({n_val_kept/len(w_val)*100:.2f}%)")
    else:
        w_train = np.ones(len(X_train_tab), dtype=np.float32)
        w_val = np.ones(len(X_val_tab), dtype=np.float32)
        if args.pnl_per_row is not None:
            print(f"\n[1.5/6] pnl_per_row fourni mais pnl_threshold=0 → Option A DÉSACTIVÉE")

    # [2] Build sequences (rolling window avec padding début)
    print(f"\n[2/6] Build sequences (window={args.window}, padding début)")
    X_train = build_sequences(X_train_tab, args.window)
    X_val = build_sequences(X_val_tab, args.window)
    X_test = build_sequences(X_test_tab, args.window)
    print(f"   X_train seq: {X_train.shape}  |  "
          f"X_val seq: {X_val.shape}  |  X_test seq: {X_test.shape}")
    print(f"   y_train UP ratio: {y_train.mean()*100:.2f}%  "
          f"(val: {y_val.mean()*100:.2f}%  test: {y_test.mean()*100:.2f}%)")

    # [3] Model
    print(f"\n[3/6] CNN-LSTM : conv={args.cnn_filters} "
          f"lstm={args.lstm_hidden}×{args.lstm_layers} "
          f"dense={args.dense_hidden} dropout={args.dropout}")
    model = CNNLSTMProgressive(
        n_features=len(feature_cols),
        cnn_filters=args.cnn_filters,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dense_hidden=args.dense_hidden,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   {n_params:,} paramètres")

    # DataLoaders (tensors incluent sample_weight pour Option A)
    train_ds = TensorDataset(torch.from_numpy(X_train),
                               torch.from_numpy(y_train),
                               torch.from_numpy(w_train))
    val_ds = TensorDataset(torch.from_numpy(X_val),
                             torch.from_numpy(y_val),
                             torch.from_numpy(w_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=(device.type == 'cuda'))

    # [4] Train
    print(f"\n[4/6] Training ..."
          + (f" (Option A sample weighting active)" if use_sample_weighting else ""))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # reduction='none' pour pondérer row par row
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_weight_sum = 0.0
        for xb, yb, wb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(xb)
            loss_per_sample = criterion(logits, yb)
            # Loss pondérée : somme(loss * w) / somme(w) (évite division par 0)
            w_sum = wb.sum()
            if w_sum.item() > 0:
                loss = (loss_per_sample * wb).sum() / w_sum
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * w_sum.item()
                train_weight_sum += w_sum.item()
        train_loss = train_loss_sum / max(train_weight_sum, 1e-8)

        # Validation (pondérée aussi pour cohérence early stop)
        model.eval()
        val_loss_sum = 0.0
        val_weight_sum = 0.0
        with torch.no_grad():
            for xb, yb, wb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                wb = wb.to(device, non_blocking=True)
                logits = model(xb)
                loss_per_sample = criterion(logits, yb)
                w_sum = wb.sum()
                if w_sum.item() > 0:
                    loss = (loss_per_sample * wb).sum() / w_sum
                    val_loss_sum += loss.item() * w_sum.item()
                    val_weight_sum += w_sum.item()
        val_loss = val_loss_sum / max(val_weight_sum, 1e-8)

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = ' ★'
        else:
            patience_counter += 1

        print(f"   Epoch {epoch:>2}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}{marker}")

        if args.early_stop > 0 and patience_counter >= args.early_stop:
            print(f"   Early stop (patience={args.early_stop})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"   Best val_loss: {best_val_loss:.4f}")

    # [5] Predict + metrics
    print(f"\n[5/6] Predict + métriques classification (threshold=0.5)")
    train_proba = predict_proba(model, X_train, device, args.batch_size)
    val_proba = predict_proba(model, X_val, device, args.batch_size)
    test_proba = predict_proba(model, X_test, device, args.batch_size)

    print(f"{'Split':<10} {'AUC':>8} {'Acc':>8} {'F1':>8}  {'Balance':>10}")
    print("-" * 50)
    for name, y, p in [('train', y_train, train_proba),
                        ('val', y_val, val_proba),
                        ('test', y_test, test_proba)]:
        y_int = y.astype(int)
        y_pred = (p > 0.5).astype(int)
        auc = roc_auc_score(y_int, p)
        acc = accuracy_score(y_int, y_pred)
        f1 = f1_score(y_int, y_pred)
        balance = y.mean() * 100
        print(f"{name:<10} {auc:>8.4f} {acc:>8.4f} {f1:>8.4f}  "
              f"{balance:>9.2f}%")

    # [6] Sauvegarde
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f'cnnlstm_progressive_{indicator}_{tf_label}_{period_tag}{filter_tag}{sw_tag}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'feature_cols': feature_cols,
        'indicator': indicator,
        'tf_label': tf_label,
        'period_tag': period_tag,
        'best_val_loss': best_val_loss,
    }, model_path)
    print(f"\n[6/6] Modèle sauvé: {model_path}  "
          f"({model_path.stat().st_size / 1024:.1f} KB)")

    preds_path = PREP_DIR / f'preds_{indicator}_{tf_label}_{period_tag}_progressive_cnnlstm{filter_tag}{sw_tag}.npz'
    np.savez(
        preds_path,
        train_preds_proba=train_proba.astype(np.float32),
        val_preds_proba=val_proba.astype(np.float32),
        test_preds_proba=test_proba.astype(np.float32),
        indicator=indicator,
        tf_label=tf_label,
        period_tag=period_tag,
        model_type='cnn_lstm',
        window=args.window,
        feature_cols=np.array(feature_cols),
    )
    print(f"   Preds sauvées: {preds_path}  "
          f"({preds_path.stat().st_size / 1024:.1f} KB)")

    print(f"\nPour backtester :")
    print(f"  python scripts/backtest_progressive.py \\")
    print(f"      --npz {npz_path} \\")
    print(f"      --preds {preds_path} \\")
    print(f"      --split test")


if __name__ == '__main__':
    main()
