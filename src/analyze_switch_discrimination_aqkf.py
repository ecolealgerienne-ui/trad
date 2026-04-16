#!/usr/bin/env python3
"""
Switch Discriminability Analysis — faux vs vrai switches, UP vs DOWN
====================================================================

Extracts feature windows around model switches and tests if false
switches are distinguishable from true switches.

4 groups:
  - faux_up   : model switches 0→1, far from any oracle transition (>20 steps)
  - vrai_up   : model switches 0→1, near oracle transition 0→1 (±6 steps)
  - faux_down : model switches 1→0, far from any oracle transition (>20 steps)
  - vrai_down : model switches 1→0, near oracle transition 1→0 (±6 steps)

2 discrimination tests:
  - UP:   faux_up vs vrai_up (XGBoost, accuracy > 70% = distinguishable)
  - DOWN: faux_down vs vrai_down

Usage:
    python src/analyze_switch_discrimination_aqkf.py --indicator macd --timeframe 30m
"""

import numpy as np
from pathlib import Path
import logging
import argparse
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from constants import PREPARED_DATA_DIR

HALF_WINDOW = 12  # 12 before + center + 12 after = 25 steps
NEAR_THRESHOLD = 6   # ±6 steps = "near" oracle transition
FAR_THRESHOLD = 20   # >20 steps = "far" from any oracle transition


def find_switches(labels):
    """Find indices where label changes. Returns (index, direction) pairs.
    direction: +1 for 0→1 (UP), -1 for 1→0 (DOWN)."""
    switches = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            direction = 1 if labels[i] > labels[i - 1] else -1
            switches.append((i, direction))
    return switches


def classify_model_switches(model_switches, oracle_switches, near_thr, far_thr):
    """Classify each model switch as vrai or faux, UP or DOWN."""
    oracle_indices = {idx: d for idx, d in oracle_switches}
    oracle_idx_arr = np.array([idx for idx, _ in oracle_switches])

    faux_up = []
    faux_down = []
    vrai_up = []
    vrai_down = []

    for m_idx, m_dir in model_switches:
        # Find nearest oracle transition
        if len(oracle_idx_arr) == 0:
            min_dist = 9999
            nearest_dir = 0
        else:
            dists = np.abs(oracle_idx_arr - m_idx)
            nearest_pos = np.argmin(dists)
            min_dist = dists[nearest_pos]
            nearest_dir = oracle_switches[nearest_pos][1]

        if m_dir == 1:  # UP switch
            if min_dist <= near_thr and nearest_dir == 1:
                vrai_up.append(m_idx)
            elif min_dist > far_thr:
                faux_up.append(m_idx)
        else:  # DOWN switch
            if min_dist <= near_thr and nearest_dir == -1:
                vrai_down.append(m_idx)
            elif min_dist > far_thr:
                faux_down.append(m_idx)

    return faux_up, vrai_up, faux_down, vrai_down


def extract_windows(indices, features, half_w):
    """Extract centered windows of features around each index."""
    n = len(features)
    windows = []
    valid_indices = []
    for idx in indices:
        start = idx - half_w
        end = idx + half_w + 1
        if start < 0 or end > n:
            continue
        windows.append(features[start:end].flatten())
        valid_indices.append(idx)
    if len(windows) == 0:
        return np.empty((0, 0)), valid_indices
    return np.array(windows), valid_indices


def discriminability_test(X_faux, X_vrai, label_faux, label_vrai, seed=42):
    """Train XGBoost to distinguish faux from vrai. Returns accuracy."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("XGBoost not installed. Run: pip install xgboost")
        return None, None

    n_faux = len(X_faux)
    n_vrai = len(X_vrai)

    if n_faux < 10 or n_vrai < 10:
        return None, None

    X = np.vstack([X_faux, X_vrai])
    y = np.concatenate([np.zeros(n_faux), np.ones(n_vrai)])

    # Shuffle
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    # Split 70/30
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=seed,
        eval_metric='logloss',
        early_stopping_rounds=10,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)

    train_acc = (model.predict(X_train) == y_train).mean()
    test_acc = (model.predict(X_test) == y_test).mean()

    return train_acc, test_acc


def main():
    parser = argparse.ArgumentParser(
        description='Switch discriminability — faux vs vrai, UP vs DOWN')
    parser.add_argument('--indicator', default='macd')
    parser.add_argument('--timeframe', default='30m')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    # Load NPZ
    npz_path = f'{PREPARED_DATA_DIR}/{args.indicator}_{args.timeframe}_dataset.npz'
    if not Path(npz_path).exists():
        logger.error(f"NPZ not found: {npz_path}")
        return

    data = np.load(npz_path, allow_pickle=True)
    if 'y_test' in data:
        y_test = data['y_test']
        y_pred_proba = data['y_test_pred']
    else:
        y_test = data['test_labels']
        y_pred_proba = data['test_preds']

    # Also load raw features from CSV for window extraction
    from constants import PREPARED_DATA_DIR
    import pandas as pd

    ASSET_CSV_MAP = {'BTC': 'BTCUSD'}
    base = ASSET_CSV_MAP['BTC']
    csv_candidates = [
        f'{PREPARED_DATA_DIR}/{base}_multitf_macd_rsi_cci.csv',
        f'{PREPARED_DATA_DIR}/{base}_multitf.csv',
    ]
    csv_path = None
    for c in csv_candidates:
        if Path(c).exists():
            csv_path = c
            break
    if csv_path is None:
        logger.error("CSV not found")
        return

    df = pd.read_csv(csv_path, parse_dates=['datetime']).set_index('datetime').sort_index()

    # Extract features (same as training pipeline)
    ind = args.indicator
    tf = args.timeframe
    feature_cols = [f'{ind}_{tf}_live', f'{ind}_{tf}_filtered']
    vel_col = f'{ind}_{tf}_velocity'
    if vel_col in df.columns:
        feature_cols.append(vel_col)

    # We need the TEST portion of the features
    # Same split as training: 70/15/15 with gap=25
    n_total = df.dropna(subset=feature_cols).shape[0]
    train_end = int(n_total * 0.70) - 25
    val_end = int(n_total * 0.85) - 25
    # Test starts at val_end, but we need sequences of 25
    # The test predictions correspond to the last portion
    # Use the last len(y_test) rows of features
    df_clean = df.dropna(subset=feature_cols)
    features_all = df_clean[feature_cols].values.astype(np.float32)

    # Test features = last len(y_test) + HALF_WINDOW*2 rows (need margin for windows)
    margin = HALF_WINDOW + 1
    test_start = len(features_all) - len(y_test) - margin
    if test_start < 0:
        test_start = 0
    features_test = features_all[test_start:]

    # Align: predictions start at index margin in features_test
    pred_offset = len(features_test) - len(y_test)

    y_pred_binary = (y_pred_proba > args.threshold).astype(int)

    # ==================================================================
    print(f"\n{'=' * 70}")
    print(f"  SWITCH DISCRIMINABILITY — {ind.upper()}_{tf}")
    print(f"  Test samples: {len(y_test):,}")
    print(f"{'=' * 70}")

    # Find switches
    oracle_switches = find_switches(y_test)
    model_switches = find_switches(y_pred_binary)

    print(f"  Oracle switches: {len(oracle_switches):,}")
    print(f"  Model switches:  {len(model_switches):,}")

    # Classify
    faux_up, vrai_up, faux_down, vrai_down = classify_model_switches(
        model_switches, oracle_switches, NEAR_THRESHOLD, FAR_THRESHOLD)

    print(f"\n  Classification des switches modèle:")
    print(f"    faux_up   (0→1 parasite): {len(faux_up):,}")
    print(f"    vrai_up   (0→1 correct):  {len(vrai_up):,}")
    print(f"    faux_down (1→0 parasite): {len(faux_down):,}")
    print(f"    vrai_down (1→0 correct):  {len(vrai_down):,}")
    print(f"    (non classé: {len(model_switches) - len(faux_up) - len(vrai_up) - len(faux_down) - len(vrai_down):,}"
          f" — entre {NEAR_THRESHOLD} et {FAR_THRESHOLD} steps)")

    # Extract feature windows
    print(f"\n  Extraction des fenêtres ({2 * HALF_WINDOW + 1} steps centrées)...")

    X_faux_up, _ = extract_windows(
        [idx + pred_offset for idx in faux_up], features_test, HALF_WINDOW)
    X_vrai_up, _ = extract_windows(
        [idx + pred_offset for idx in vrai_up], features_test, HALF_WINDOW)
    X_faux_down, _ = extract_windows(
        [idx + pred_offset for idx in faux_down], features_test, HALF_WINDOW)
    X_vrai_down, _ = extract_windows(
        [idx + pred_offset for idx in vrai_down], features_test, HALF_WINDOW)

    print(f"    faux_up windows:   {len(X_faux_up):,}")
    print(f"    vrai_up windows:   {len(X_vrai_up):,}")
    print(f"    faux_down windows: {len(X_faux_down):,}")
    print(f"    vrai_down windows: {len(X_vrai_down):,}")

    # Discrimination tests
    print(f"\n{'=' * 70}")
    print(f"  TEST DE DISCRIMINATION (XGBoost, 70/30 split)")
    print(f"{'=' * 70}")

    # UP test
    print(f"\n  --- UP: faux_up ({len(X_faux_up)}) vs vrai_up ({len(X_vrai_up)}) ---")
    if len(X_faux_up) >= 10 and len(X_vrai_up) >= 10:
        train_acc, test_acc = discriminability_test(X_faux_up, X_vrai_up, "faux_up", "vrai_up")
        if test_acc is not None:
            verdict = "DISTINGUABLE" if test_acc > 0.65 else "NON DISTINGUABLE"
            print(f"    Train accuracy: {train_acc:.1%}")
            print(f"    Test accuracy:  {test_acc:.1%}")
            print(f"    Verdict: {verdict}")
            if test_acc > 0.65:
                print(f"    → Hard negative mining peut marcher pour les UP")
            else:
                print(f"    → Les faux UP ressemblent aux vrais UP dans les features")
    else:
        print(f"    Pas assez de samples (min 10 par groupe)")

    # DOWN test
    print(f"\n  --- DOWN: faux_down ({len(X_faux_down)}) vs vrai_down ({len(X_vrai_down)}) ---")
    if len(X_faux_down) >= 10 and len(X_vrai_down) >= 10:
        train_acc, test_acc = discriminability_test(X_faux_down, X_vrai_down, "faux_down", "vrai_down")
        if test_acc is not None:
            verdict = "DISTINGUABLE" if test_acc > 0.65 else "NON DISTINGUABLE"
            print(f"    Train accuracy: {train_acc:.1%}")
            print(f"    Test accuracy:  {test_acc:.1%}")
            print(f"    Verdict: {verdict}")
            if test_acc > 0.65:
                print(f"    → Hard negative mining peut marcher pour les DOWN")
            else:
                print(f"    → Les faux DOWN ressemblent aux vrais DOWN dans les features")
    else:
        print(f"    Pas assez de samples (min 10 par groupe)")

    # Feature stats comparison
    print(f"\n{'=' * 70}")
    print(f"  STATISTIQUES DES FEATURES AU MOMENT DU SWITCH")
    print(f"{'=' * 70}")

    n_feat = len(feature_cols)
    for fi, fname in enumerate(feature_cols):
        print(f"\n  {fname}:")
        for group_name, X_group in [('faux_up', X_faux_up), ('vrai_up', X_vrai_up),
                                     ('faux_down', X_faux_down), ('vrai_down', X_vrai_down)]:
            if len(X_group) == 0:
                continue
            # Center value (step HALF_WINDOW = the switch itself)
            center_idx = HALF_WINDOW * n_feat + fi
            if center_idx < X_group.shape[1]:
                vals = X_group[:, center_idx]
                print(f"    {group_name:>12}: mean={vals.mean():>+10.4f}  "
                      f"std={vals.std():>8.4f}  "
                      f"median={np.median(vals):>+10.4f}")

    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == '__main__':
    main()
