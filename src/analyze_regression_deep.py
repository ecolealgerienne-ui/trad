#!/usr/bin/env python3
"""
Deep analysis of regression model: is R²=0.91 real signal or just plateau prediction?

Analysis 1: R² conditional (transition zones vs plateau zones)
Analysis 2: Relative MAE distribution
Analysis 3: Magnitude threshold filtering (compare with binary classifier)

Usage:
    python src/analyze_regression_deep.py --indicator macd --timeframe 30m
"""

import numpy as np
import json
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from constants import PREPARED_DATA_DIR

PROXIMITY = 3


def find_transitions(labels_or_slopes):
    """Find transition indices from binary labels or sign changes in slopes."""
    signs = np.sign(labels_or_slopes)
    return np.where(np.diff(signs) != 0)[0] + 1


def distance_to_nearest_transition(n, transitions):
    """For each index, compute distance to nearest transition."""
    dist = np.full(n, 999999)
    for t in transitions:
        d = np.abs(np.arange(n) - t)
        dist = np.minimum(dist, d)
    return dist


def main():
    parser = argparse.ArgumentParser(description='Deep regression analysis')
    parser.add_argument('--indicator', default='macd', choices=['macd', 'rsi', 'cci'])
    parser.add_argument('--timeframe', default='30m', choices=['30m', '1h'])
    args = parser.parse_args()

    # Load regression predictions
    reg_name = f'{args.indicator}_{args.timeframe}_crossfeat_regression'
    reg_path = f'{PREPARED_DATA_DIR}/{reg_name}_dataset.npz'

    if not Path(reg_path).exists():
        logger.error(f"Not found: {reg_path}")
        return

    reg_data = np.load(reg_path, allow_pickle=True)
    y_test = reg_data['y_test']          # z-scored target slopes
    y_pred = reg_data['y_test_pred']     # z-scored predictions

    # Load norm stats for denormalization
    norm_path = f'{PREPARED_DATA_DIR}/norm_stats_{reg_name}.json'
    target_mean, target_std = 0.0, 1.0
    if Path(norm_path).exists():
        with open(norm_path) as f:
            ns = json.load(f)
        for asset, stats in ns.items():
            if 'target' in stats:
                target_mean = stats['target']['mean']
                target_std = stats['target']['std']
                break

    # Denormalize
    y_real = y_test * target_std + target_mean
    y_pred_real = y_pred * target_std + target_mean

    # Find oracle transitions (sign changes in target)
    oracle_trans = find_transitions(y_test)
    n = len(y_test)
    dist = distance_to_nearest_transition(n, oracle_trans)

    # Masks
    mask_transition = dist <= PROXIMITY
    mask_plateau = dist > PROXIMITY

    print(f"\n{'='*70}")
    print(f"DEEP REGRESSION ANALYSIS — {reg_name}")
    print(f"{'='*70}")
    print(f"  Test samples: {n:,}")
    print(f"  Oracle transitions: {len(oracle_trans):,}")
    print(f"  Transition zone (±{PROXIMITY} steps): {mask_transition.sum():,} ({mask_transition.mean()*100:.1f}%)")
    print(f"  Plateau zone: {mask_plateau.sum():,} ({mask_plateau.mean()*100:.1f}%)")

    # =========================================================================
    # ANALYSIS 1 — Conditional R²
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS 1 — CONDITIONAL R² (transition vs plateau)")
    print(f"{'='*70}")

    for zone_name, mask in [('TRANSITION', mask_transition), ('PLATEAU', mask_plateau), ('ALL', np.ones(n, dtype=bool))]:
        yt = y_test[mask]
        yp = y_pred[mask]

        if len(yt) < 2:
            continue

        ss_res = ((yt - yp) ** 2).sum()
        ss_tot = ((yt - yt.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        mae = np.abs(yt - yp).mean()
        corr = np.corrcoef(yt, yp)[0, 1] if len(yt) > 1 else 0

        yt_r = y_real[mask]
        yp_r = y_pred_real[mask]
        mae_real = np.abs(yt_r - yp_r).mean()

        print(f"\n  {zone_name} ({mask.sum():,} samples, {mask.mean()*100:.1f}%):")
        print(f"    R²:          {r2:.4f}")
        print(f"    MAE (z):     {mae:.4f}")
        print(f"    MAE (real):  {mae_real:.4f}")
        print(f"    Correlation: {corr:.4f}")
        print(f"    Target std:  {yt.std():.4f}")
        print(f"    Pred std:    {yp.std():.4f}")

    # =========================================================================
    # ANALYSIS 2 — Relative MAE
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS 2 — RELATIVE MAE")
    print(f"{'='*70}")

    epsilon = 0.01
    abs_error = np.abs(y_real - y_pred_real)
    rel_error = abs_error / np.maximum(np.abs(y_real), epsilon)

    print(f"\n  Overall relative MAE distribution:")
    print(f"    Median: {np.median(rel_error):.4f}")
    print(f"    P25:    {np.percentile(rel_error, 25):.4f}")
    print(f"    P75:    {np.percentile(rel_error, 75):.4f}")
    print(f"    P90:    {np.percentile(rel_error, 90):.4f}")
    print(f"    P95:    {np.percentile(rel_error, 95):.4f}")

    for zone_name, mask in [('TRANSITION', mask_transition), ('PLATEAU', mask_plateau)]:
        re = rel_error[mask]
        print(f"\n  {zone_name}:")
        print(f"    Median: {np.median(re):.4f}")
        print(f"    P25:    {np.percentile(re, 25):.4f}")
        print(f"    P75:    {np.percentile(re, 75):.4f}")
        print(f"    P90:    {np.percentile(re, 90):.4f}")

    # =========================================================================
    # ANALYSIS 3 — Magnitude threshold filtering
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS 3 — MAGNITUDE THRESHOLD FILTERING")
    print(f"{'='*70}")

    # Compute binary decisions from regression predictions
    pred_direction = np.sign(y_pred)  # -1, 0, +1

    # Oracle transitions for switch evaluation
    oracle_binary = (y_test > 0).astype(int)
    oracle_trans_binary = np.where(np.diff(oracle_binary) != 0)[0] + 1
    n_oracle_switches = len(oracle_trans_binary)

    print(f"\n  Oracle switches: {n_oracle_switches:,}")
    print(f"\n  {'Threshold':<12} {'Switches':>10} {'Ratio':>8} {'Justified%':>12} {'Spurious%':>10} {'Coverage%':>10}")
    print(f"  {'-'*65}")

    thresholds = [0, 0.1, 0.2, 0.5, 1.0, 2.0]
    threshold_results = []

    for thr in thresholds:
        # Apply threshold: only switch when |pred| > threshold
        # Build filtered direction: hold previous direction when |pred| <= threshold
        filtered_dir = np.zeros(n, dtype=int)
        current_dir = 0

        for i in range(n):
            if abs(y_pred[i]) > thr:
                current_dir = 1 if y_pred[i] > 0 else -1
            filtered_dir[i] = current_dir

        # Count switches
        switches = np.where(np.diff(filtered_dir) != 0)[0] + 1
        n_switches = len(switches)

        # Classify switches as justified/spurious
        if n_switches > 0 and len(oracle_trans_binary) > 0:
            justified = 0
            spurious = 0
            for s in switches:
                min_dist = np.abs(oracle_trans_binary.astype(int) - int(s)).min()
                if min_dist <= 6:
                    justified += 1
                elif min_dist > 20:
                    spurious += 1

            pct_justified = justified / n_switches * 100
            pct_spurious = spurious / n_switches * 100
        else:
            pct_justified = 0
            pct_spurious = 0

        ratio = n_switches / n_oracle_switches if n_oracle_switches > 0 else 0

        # Coverage: % of time we have a position (|pred| > threshold)
        coverage = (np.abs(y_pred) > thr).mean() * 100

        print(f"  {thr:<12.1f} {n_switches:>10,} {ratio:>7.1f}x {pct_justified:>11.1f}% {pct_spurious:>9.1f}% {coverage:>9.1f}%")

        threshold_results.append({
            'threshold': thr,
            'switches': n_switches,
            'ratio': ratio,
            'pct_justified': pct_justified,
            'pct_spurious': pct_spurious,
            'coverage': coverage,
        })

    # =========================================================================
    # COMPARISON WITH BINARY CLASSIFIER
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"COMPARISON: Regression vs Binary Classifier (crossfeat)")
    print(f"{'='*70}")

    # Try to load binary crossfeat for comparison
    bin_path = f'{PREPARED_DATA_DIR}/{args.indicator}_{args.timeframe}_crossfeat_dataset.npz'
    if Path(bin_path).exists():
        bin_data = np.load(bin_path, allow_pickle=True)
        bin_pred = (bin_data['y_test_pred'] > 0.5).astype(int)
        bin_switches = np.where(np.diff(bin_pred) != 0)[0] + 1
        n_bin_switches = len(bin_switches)

        # Find best regression threshold that matches binary switch count
        best_match_thr = None
        best_match_diff = float('inf')
        for tr in threshold_results:
            diff = abs(tr['switches'] - n_bin_switches)
            if diff < best_match_diff:
                best_match_diff = diff
                best_match_thr = tr

        print(f"\n  Binary classifier (crossfeat): {n_bin_switches:,} switches")
        if best_match_thr:
            print(f"  Closest regression threshold:  {best_match_thr['threshold']:.1f} → {best_match_thr['switches']:,} switches")
            print(f"\n  {'Metric':<25} {'Binary':>12} {'Regression':>12}")
            print(f"  {'-'*52}")
            print(f"  {'Switches':<25} {n_bin_switches:>12,} {best_match_thr['switches']:>12,}")
            print(f"  {'Ratio':<25} {n_bin_switches/n_oracle_switches:>11.1f}x {best_match_thr['ratio']:>11.1f}x")
            print(f"  {'Justified%':<25} {'(see KPI)':>12} {best_match_thr['pct_justified']:>11.1f}%")
            print(f"  {'Spurious%':<25} {'(see KPI)':>12} {best_match_thr['pct_spurious']:>11.1f}%")
    else:
        print(f"\n  Binary crossfeat not found: {bin_path}")

    # =========================================================================
    # VERDICT
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    # Get transition R²
    yt_trans = y_test[mask_transition]
    yp_trans = y_pred[mask_transition]
    if len(yt_trans) > 1:
        ss_r = ((yt_trans - yp_trans) ** 2).sum()
        ss_t = ((yt_trans - yt_trans.mean()) ** 2).sum()
        r2_trans = 1 - ss_r / ss_t if ss_t > 0 else 0
    else:
        r2_trans = 0

    # Get plateau R²
    yt_plat = y_test[mask_plateau]
    yp_plat = y_pred[mask_plateau]
    ss_r = ((yt_plat - yp_plat) ** 2).sum()
    ss_t = ((yt_plat - yt_plat.mean()) ** 2).sum()
    r2_plat = 1 - ss_r / ss_t if ss_t > 0 else 0

    if r2_trans > 0.3:
        print(f"  ✅ Regression captures transition dynamics (R² transition = {r2_trans:.4f})")
    else:
        print(f"  ❌ Regression only predicts plateaus well (R² transition = {r2_trans:.4f})")

    if r2_plat > 0.9:
        print(f"  ⚠️  R²=0.91 is mostly plateau prediction (R² plateau = {r2_plat:.4f})")

    # Check if magnitude threshold helps
    best_useful = None
    for tr in threshold_results:
        if tr['threshold'] > 0 and tr['pct_justified'] > 60 and tr['pct_spurious'] < 15:
            best_useful = tr
            break

    if best_useful:
        print(f"  ✅ Magnitude threshold useful: thr={best_useful['threshold']:.1f} → "
              f"{best_useful['pct_justified']:.0f}% justified, {best_useful['pct_spurious']:.0f}% spurious")
    else:
        print(f"  ❌ No magnitude threshold achieves >60% justified + <15% spurious")

    # Save
    results = {
        'model': reg_name,
        'r2_all': float(1 - ((y_test - y_pred)**2).sum() / ((y_test - y_test.mean())**2).sum()),
        'r2_transition': float(r2_trans),
        'r2_plateau': float(r2_plat),
        'n_transition': int(mask_transition.sum()),
        'n_plateau': int(mask_plateau.sum()),
        'thresholds': threshold_results,
    }

    json_path = 'models/regression_deep_analysis.json'
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {json_path}")


if __name__ == '__main__':
    main()
