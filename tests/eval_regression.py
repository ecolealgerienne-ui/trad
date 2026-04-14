#!/usr/bin/env python3
"""
Evaluation script for regression models (continuous slope prediction).

Computes on denormalized data:
- R² (coefficient of determination)
- MAE (mean absolute error)
- Pearson correlation
- Implicit binary accuracy: sign(pred) == sign(target)
- Residual distribution

Usage:
    python tests/eval_regression.py --indicator macd --timeframe 30m
    python tests/eval_regression.py --indicator macd --timeframe 30m --crossfeat
"""

import numpy as np
import json
import argparse
import logging
import sys
from pathlib import Path
from scipy import stats as scipy_stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from constants import PREPARED_DATA_DIR


def main():
    parser = argparse.ArgumentParser(description='Evaluate regression model')
    parser.add_argument('--indicator', default='macd', choices=['macd', 'rsi', 'cci'])
    parser.add_argument('--timeframe', default='30m', choices=['30m', '1h'])
    parser.add_argument('--crossfeat', action='store_true')
    args = parser.parse_args()

    suffix_parts = []
    if args.crossfeat:
        suffix_parts.append('crossfeat')
    suffix_parts.append('regression')
    suffix = '_' + '_'.join(suffix_parts)
    model_name = f'{args.indicator}_{args.timeframe}{suffix}'

    npz_path = f'{PREPARED_DATA_DIR}/{model_name}_dataset.npz'
    norm_path = f'{PREPARED_DATA_DIR}/norm_stats_{model_name}.json'

    if not Path(npz_path).exists():
        logger.error(f"NPZ not found: {npz_path}")
        return

    # Load predictions and targets (z-scored)
    data = np.load(npz_path, allow_pickle=True)
    y_test = data['y_test']        # z-scored targets
    y_test_pred = data['y_test_pred']  # raw model outputs (z-scored scale)
    y_train = data['y_train']
    y_val = data['y_val']

    # Load norm stats to denormalize
    # The target was z-scored per asset. For single-asset (BTC), we need BTC's stats.
    if Path(norm_path).exists():
        with open(norm_path) as f:
            norm_stats = json.load(f)
        # Find target stats (first asset that has 'target' key)
        target_mean = None
        target_std = None
        for asset, stats in norm_stats.items():
            if 'target' in stats:
                target_mean = stats['target']['mean']
                target_std = stats['target']['std']
                break
    else:
        target_mean = None
        target_std = None

    print(f"\n{'='*70}")
    print(f"REGRESSION EVALUATION — {model_name}")
    print(f"{'='*70}")
    print(f"  Test samples: {len(y_test):,}")
    print(f"  Train samples: {len(y_train):,}")
    print(f"  Val samples: {len(y_val):,}")

    # --- Z-SCORED metrics (as trained) ---
    print(f"\n--- Z-SCORED (as trained) ---")

    ss_res = ((y_test - y_test_pred) ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum()
    r2_zscore = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    mae_zscore = np.abs(y_test - y_test_pred).mean()
    corr_zscore = np.corrcoef(y_test, y_test_pred)[0, 1] if len(y_test) > 1 else 0

    print(f"  R²:          {r2_zscore:.4f}")
    print(f"  MAE:         {mae_zscore:.4f}")
    print(f"  Correlation: {corr_zscore:.4f}")
    print(f"  MSE:         {((y_test - y_test_pred)**2).mean():.6f}")

    # --- DENORMALIZED metrics (real scale) ---
    if target_mean is not None and target_std is not None:
        print(f"\n--- DENORMALIZED (real scale) ---")
        print(f"  Target stats (from train): mean={target_mean:.6f}, std={target_std:.6f}")

        y_test_real = y_test * target_std + target_mean
        y_pred_real = y_test_pred * target_std + target_mean

        ss_res_real = ((y_test_real - y_pred_real) ** 2).sum()
        ss_tot_real = ((y_test_real - y_test_real.mean()) ** 2).sum()
        r2_real = 1 - ss_res_real / ss_tot_real if ss_tot_real > 0 else 0
        mae_real = np.abs(y_test_real - y_pred_real).mean()
        corr_real = np.corrcoef(y_test_real, y_pred_real)[0, 1]

        print(f"  R²:          {r2_real:.4f}")
        print(f"  MAE:         {mae_real:.4f}")
        print(f"  Correlation: {corr_real:.4f}")
        print(f"  Pred range:  [{y_pred_real.min():.2f}, {y_pred_real.max():.2f}]")
        print(f"  True range:  [{y_test_real.min():.2f}, {y_test_real.max():.2f}]")
    else:
        y_test_real = y_test
        y_pred_real = y_test_pred
        r2_real = r2_zscore
        mae_real = mae_zscore
        corr_real = corr_zscore
        print(f"\n  (No norm stats found, using z-scored values)")

    # --- IMPLICIT BINARY ACCURACY ---
    print(f"\n--- IMPLICIT BINARY ACCURACY ---")
    print(f"  sign(pred) == sign(target)")

    sign_correct = (np.sign(y_test_pred) == np.sign(y_test)).mean()
    print(f"  Accuracy:    {sign_correct*100:.2f}%")

    # Compare with classification baselines
    persistence = np.zeros_like(y_test)
    persistence[1:] = y_test[:-1]
    persistence_sign_acc = (np.sign(persistence) == np.sign(y_test)).mean()
    print(f"  Persistence: {persistence_sign_acc*100:.2f}%")
    print(f"  Delta:       {(sign_correct - persistence_sign_acc)*100:+.2f}%")

    # --- RESIDUAL DISTRIBUTION ---
    print(f"\n--- RESIDUAL DISTRIBUTION (z-scored) ---")
    residuals = y_test - y_test_pred
    print(f"  Mean:   {residuals.mean():.4f}")
    print(f"  Std:    {residuals.std():.4f}")
    print(f"  Median: {np.median(residuals):.4f}")
    print(f"  P5:     {np.percentile(residuals, 5):.4f}")
    print(f"  P95:    {np.percentile(residuals, 95):.4f}")
    print(f"  Skew:   {scipy_stats.skew(residuals):.4f}")
    print(f"  Kurt:   {scipy_stats.kurtosis(residuals):.4f}")

    # --- PREDICTION STABILITY ---
    print(f"\n--- PREDICTION STABILITY ---")
    print(f"  Train pred: mean={data['y_train_pred'].mean():.4f}, std={data['y_train_pred'].std():.4f}")
    print(f"  Val pred:   mean={data['y_val_pred'].mean():.4f}, std={data['y_val_pred'].std():.4f}")
    print(f"  Test pred:  mean={y_test_pred.mean():.4f}, std={y_test_pred.std():.4f}")
    pred_std_ratio = y_test_pred.std() / data['y_train_pred'].std()
    print(f"  Test/Train std ratio: {pred_std_ratio:.2f}x {'⚠️ DIVERGENT' if pred_std_ratio > 1.5 else '✅ STABLE'}")

    # --- VERDICT ---
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    if r2_real > 0.3 and corr_real > 0.5:
        print(f"  ✅ SUBSTANTIAL SIGNAL (R²={r2_real:.4f}, corr={corr_real:.4f})")
        print(f"     The continuous slope contains information lost by binary classification")
    elif r2_real > 0.1:
        print(f"  ⚠️  WEAK SIGNAL (R²={r2_real:.4f}, corr={corr_real:.4f})")
        print(f"     Some signal but may not be exploitable")
    else:
        print(f"  ❌ STRUCTURAL CEILING CONFIRMED (R²={r2_real:.4f}, corr={corr_real:.4f})")
        print(f"     Regression does not improve over classification")

    # Save results
    results = {
        'model': model_name,
        'r2_zscore': float(r2_zscore),
        'r2_real': float(r2_real),
        'mae_zscore': float(mae_zscore),
        'mae_real': float(mae_real),
        'correlation': float(corr_real),
        'sign_accuracy': float(sign_correct),
        'persistence_sign_acc': float(persistence_sign_acc),
        'pred_std_ratio': float(pred_std_ratio),
    }

    json_path = f'models/regression_eval_{args.indicator}_{args.timeframe}.json'
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {json_path}")


if __name__ == '__main__':
    main()
