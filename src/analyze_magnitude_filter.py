#!/usr/bin/env python3
"""
Analyze if regression magnitude distinguishes true from false switches.

For each switch (sign change in predicted slope), extract |pred| at the switch
and compare distributions between true switches (near oracle transition)
and false switches (far from any oracle transition).

Usage:
    python src/analyze_magnitude_filter.py --indicator macd --timeframe 30m
"""

import numpy as np
import json
import argparse
import logging
import sys
from pathlib import Path
from scipy.stats import mannwhitneyu

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from constants import PREPARED_DATA_DIR

TRUE_PROXIMITY = 3    # ±3 steps = true switch
FALSE_MIN_DIST = 20   # >20 steps = clearly false


def main():
    parser = argparse.ArgumentParser(description='Magnitude filter analysis')
    parser.add_argument('--indicator', default='macd', choices=['macd', 'rsi', 'cci'])
    parser.add_argument('--timeframe', default='30m', choices=['30m', '1h'])
    args = parser.parse_args()

    reg_name = f'{args.indicator}_{args.timeframe}_crossfeat_regression'
    npz_path = f'{PREPARED_DATA_DIR}/{reg_name}_dataset.npz'

    if not Path(npz_path).exists():
        logger.error(f"Not found: {npz_path}")
        return

    data = np.load(npz_path, allow_pickle=True)
    y_test = data['y_test']          # z-scored target slopes
    y_pred = data['y_test_pred']     # z-scored predictions
    n = len(y_test)

    # Oracle transitions: sign changes in target
    oracle_signs = np.sign(y_test)
    oracle_trans = np.where(np.diff(oracle_signs) != 0)[0] + 1

    # Model switches: sign changes in prediction
    pred_signs = np.sign(y_pred)
    model_switches = np.where(np.diff(pred_signs) != 0)[0] + 1

    print(f"\n{'='*70}")
    print(f"MAGNITUDE FILTER ANALYSIS — {reg_name}")
    print(f"{'='*70}")
    print(f"  Test samples: {n:,}")
    print(f"  Oracle transitions: {len(oracle_trans):,}")
    print(f"  Model switches: {len(model_switches):,}")

    # Classify each switch as TRUE, FALSE, or AMBIGUOUS
    true_magnitudes = []
    false_magnitudes = []
    all_switch_data = []

    for s in model_switches:
        mag = abs(y_pred[s])

        # Distance to nearest oracle transition
        if len(oracle_trans) > 0:
            min_dist = np.abs(oracle_trans.astype(int) - int(s)).min()
        else:
            min_dist = 999999

        is_true = min_dist <= TRUE_PROXIMITY
        is_false = min_dist > FALSE_MIN_DIST

        all_switch_data.append({
            'idx': int(s),
            'magnitude': float(mag),
            'min_dist': int(min_dist),
            'is_true': is_true,
            'is_false': is_false,
        })

        if is_true:
            true_magnitudes.append(mag)
        if is_false:
            false_magnitudes.append(mag)

    true_mag = np.array(true_magnitudes)
    false_mag = np.array(false_magnitudes)
    n_true = len(true_mag)
    n_false = len(false_mag)
    n_ambiguous = len(model_switches) - n_true - n_false

    print(f"\n  Switch classification:")
    print(f"    True (±{TRUE_PROXIMITY} steps):  {n_true:,}")
    print(f"    False (>{FALSE_MIN_DIST} steps): {n_false:,}")
    print(f"    Ambiguous:            {n_ambiguous:,}")

    # =========================================================================
    # ANALYSIS 1 — Distribution comparison
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS 1 — MAGNITUDE DISTRIBUTION (true vs false switches)")
    print(f"{'='*70}")

    print(f"\n  {'Stat':<12} {'True switches':>15} {'False switches':>15} {'Gap':>10}")
    print(f"  {'-'*55}")

    for name, fn in [('Median', np.median), ('P25', lambda x: np.percentile(x, 25)),
                      ('P75', lambda x: np.percentile(x, 75)), ('P90', lambda x: np.percentile(x, 90)),
                      ('Mean', np.mean), ('Std', np.std)]:
        tv = fn(true_mag) if n_true > 0 else 0
        fv = fn(false_mag) if n_false > 0 else 0
        print(f"  {name:<12} {tv:>15.4f} {fv:>15.4f} {tv-fv:>+10.4f}")

    # Mann-Whitney U test
    if n_true > 10 and n_false > 10:
        stat, pvalue = mannwhitneyu(true_mag, false_mag, alternative='greater')
        print(f"\n  Mann-Whitney U test (true > false):")
        print(f"    U-statistic: {stat:,.0f}")
        print(f"    p-value:     {pvalue:.2e}")
        if pvalue < 0.001:
            print(f"    ✅ Highly significant (p < 0.001): true switches have HIGHER magnitude")
        elif pvalue < 0.05:
            print(f"    ⚠️  Significant (p < 0.05)")
        else:
            print(f"    ❌ Not significant (p ≥ 0.05): magnitude does NOT discriminate")

    # =========================================================================
    # ANALYSIS 2 — Threshold filtering
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS 2 — THRESHOLD FILTERING")
    print(f"{'='*70}")

    thresholds = [0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

    print(f"\n  {'Threshold':<12} {'Remaining':>10} {'False filt%':>12} {'True filt%':>12} {'Ratio':>8}")
    print(f"  {'-'*58}")

    results_table = []

    for thr in thresholds:
        # Filter: keep only switches where magnitude >= threshold
        remaining_true = (true_mag >= thr).sum()
        remaining_false = (false_mag >= thr).sum()
        filtered_true = n_true - remaining_true
        filtered_false = n_false - remaining_false

        pct_false_filt = filtered_false / n_false * 100 if n_false > 0 else 0
        pct_true_filt = filtered_true / n_true * 100 if n_true > 0 else 0
        ratio = pct_false_filt / pct_true_filt if pct_true_filt > 0.1 else float('inf')

        remaining_total = remaining_true + remaining_false + n_ambiguous - \
                         sum(1 for s in all_switch_data
                             if not s['is_true'] and not s['is_false'] and s['magnitude'] < thr)

        ratio_str = f"{ratio:.1f}x" if ratio < 100 else "INF"
        print(f"  {thr:<12.2f} {remaining_true + remaining_false:>10,} "
              f"{pct_false_filt:>11.1f}% {pct_true_filt:>11.1f}% {ratio_str:>8}")

        results_table.append({
            'threshold': thr,
            'remaining_true': int(remaining_true),
            'remaining_false': int(remaining_false),
            'filtered_true': int(filtered_true),
            'filtered_false': int(filtered_false),
            'pct_false_filtered': float(pct_false_filt),
            'pct_true_filtered': float(pct_true_filt),
            'ratio': float(ratio) if ratio < 1000 else 999,
        })

    # =========================================================================
    # VERDICT
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    # Find best threshold with ratio >= 5x
    best_5x = None
    for r in results_table:
        if r['threshold'] > 0 and r['ratio'] >= 5 and r['pct_false_filtered'] > 10:
            if best_5x is None or r['pct_false_filtered'] > best_5x['pct_false_filtered']:
                best_5x = r

    if best_5x:
        print(f"  ✅ Threshold with ratio ≥ 5x found: {best_5x['threshold']:.2f}")
        print(f"     Filters {best_5x['pct_false_filtered']:.1f}% false, loses {best_5x['pct_true_filtered']:.1f}% true")
        print(f"     Ratio: {best_5x['ratio']:.1f}x")
    else:
        print(f"  ❌ No threshold achieves ratio ≥ 5x with >10% false filtered")

    # Find best overall ratio
    best_ratio = max([r for r in results_table if r['threshold'] > 0 and r['pct_false_filtered'] > 5],
                     key=lambda r: r['ratio'], default=None)
    if best_ratio:
        print(f"\n  Best overall: threshold={best_ratio['threshold']:.2f}")
        print(f"     Filters {best_ratio['pct_false_filtered']:.1f}% false, loses {best_ratio['pct_true_filtered']:.1f}% true")
        print(f"     Ratio: {best_ratio['ratio']:.1f}x")

    # Save
    output = {
        'model': reg_name,
        'n_true_switches': n_true,
        'n_false_switches': n_false,
        'n_ambiguous': n_ambiguous,
        'true_median': float(np.median(true_mag)) if n_true > 0 else None,
        'false_median': float(np.median(false_mag)) if n_false > 0 else None,
        'thresholds': results_table,
    }

    json_path = 'models/regression_magnitude_filter.json'
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {json_path}")


if __name__ == '__main__':
    main()
