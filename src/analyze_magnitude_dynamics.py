#!/usr/bin/env python3
"""
Analyze temporal dynamics of predicted slope magnitude around switches.

Hypothesis: true switches are preceded by a progressive magnitude increase
("crescendo") while false switches are isolated spikes.

Usage:
    python src/analyze_magnitude_dynamics.py --indicator macd --timeframe 30m
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

TRUE_PROXIMITY = 3
FALSE_MIN_DIST = 20
WINDOW_BEFORE = 10
WINDOW_AFTER = 5


def main():
    parser = argparse.ArgumentParser(description='Magnitude dynamics around switches')
    parser.add_argument('--indicator', default='macd', choices=['macd', 'rsi', 'cci'])
    parser.add_argument('--timeframe', default='30m', choices=['30m', '1h'])
    args = parser.parse_args()

    reg_name = f'{args.indicator}_{args.timeframe}_crossfeat_regression'
    npz_path = f'{PREPARED_DATA_DIR}/{reg_name}_dataset.npz'

    if not Path(npz_path).exists():
        logger.error(f"Not found: {npz_path}"); return

    data = np.load(npz_path, allow_pickle=True)
    y_pred = data['y_test_pred']
    y_test = data['y_test']
    n = len(y_pred)

    mag = np.abs(y_pred)  # magnitude at every step

    # Oracle transitions
    oracle_signs = np.sign(y_test)
    oracle_trans = np.where(np.diff(oracle_signs) != 0)[0] + 1

    # Model switches
    pred_signs = np.sign(y_pred)
    model_switches = np.where(np.diff(pred_signs) != 0)[0] + 1

    # Classify switches
    true_switches = []
    false_switches = []
    for s in model_switches:
        if len(oracle_trans) > 0:
            min_dist = np.abs(oracle_trans.astype(int) - int(s)).min()
        else:
            min_dist = 999999
        if min_dist <= TRUE_PROXIMITY:
            true_switches.append(s)
        elif min_dist > FALSE_MIN_DIST:
            false_switches.append(s)

    print(f"\n{'='*70}")
    print(f"MAGNITUDE DYNAMICS ANALYSIS — {reg_name}")
    print(f"{'='*70}")
    print(f"  Model switches: {len(model_switches):,}")
    print(f"  True: {len(true_switches):,}, False: {len(false_switches):,}")

    # =========================================================================
    # ANALYSIS 1 — Trajectory around switches
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS 1 — MEAN TRAJECTORY [t-{WINDOW_BEFORE}, t+{WINDOW_AFTER}]")
    print(f"{'='*70}")

    total_window = WINDOW_BEFORE + WINDOW_AFTER + 1

    def extract_trajectories(switch_list):
        trajs = []
        for s in switch_list:
            start = s - WINDOW_BEFORE
            end = s + WINDOW_AFTER + 1
            if start >= 0 and end <= n:
                trajs.append(mag[start:end])
        return np.array(trajs) if trajs else np.empty((0, total_window))

    true_trajs = extract_trajectories(true_switches)
    false_trajs = extract_trajectories(false_switches)

    true_mean = true_trajs.mean(axis=0) if len(true_trajs) > 0 else np.zeros(total_window)
    false_mean = false_trajs.mean(axis=0) if len(false_trajs) > 0 else np.zeros(total_window)
    true_std = true_trajs.std(axis=0) if len(true_trajs) > 0 else np.zeros(total_window)
    false_std = false_trajs.std(axis=0) if len(false_trajs) > 0 else np.zeros(total_window)

    offsets = list(range(-WINDOW_BEFORE, WINDOW_AFTER + 1))

    print(f"\n  Trajectories: {len(true_trajs):,} true, {len(false_trajs):,} false")
    print(f"\n  {'Offset':<8} {'True mean':>10} {'True std':>10} {'False mean':>11} {'False std':>10} {'Gap':>8}")
    print(f"  {'-'*60}")

    for i, off in enumerate(offsets):
        marker = ' <<<' if off == 0 else ''
        print(f"  {off:<8} {true_mean[i]:>10.4f} {true_std[i]:>10.4f} "
              f"{false_mean[i]:>11.4f} {false_std[i]:>10.4f} "
              f"{true_mean[i]-false_mean[i]:>+8.4f}{marker}")

    # Crescendo check: is mean magnitude increasing from t-5 to t for true?
    true_t5_to_t = true_mean[WINDOW_BEFORE-5:WINDOW_BEFORE+1]  # t-5 to t
    false_t5_to_t = false_mean[WINDOW_BEFORE-5:WINDOW_BEFORE+1]
    true_slope = np.polyfit(range(6), true_t5_to_t, 1)[0] if len(true_t5_to_t) == 6 else 0
    false_slope = np.polyfit(range(6), false_t5_to_t, 1)[0] if len(false_t5_to_t) == 6 else 0

    print(f"\n  Crescendo [t-5 → t]:")
    print(f"    True slope:  {true_slope:+.6f} {'(ascending ✅)' if true_slope > 0 else '(flat/descending ❌)'}")
    print(f"    False slope: {false_slope:+.6f} {'(ascending)' if false_slope > 0 else '(flat/descending)'}")

    # =========================================================================
    # ANALYSIS 2 — Dynamic filtering rules
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS 2 — DYNAMIC FILTERING RULES")
    print(f"{'='*70}")

    def eval_rule(name, keep_fn):
        true_kept = sum(1 for s in true_switches if keep_fn(s))
        false_kept = sum(1 for s in false_switches if keep_fn(s))
        n_t = len(true_switches)
        n_f = len(false_switches)
        pct_true = true_kept / n_t * 100 if n_t > 0 else 0
        pct_false = false_kept / n_f * 100 if n_f > 0 else 0
        # Ratio = % false FILTERED / % true FILTERED
        filt_false = 100 - pct_false
        filt_true = 100 - pct_true
        ratio = filt_false / filt_true if filt_true > 0.1 else float('inf')
        return {
            'name': name,
            'true_kept': true_kept, 'false_kept': false_kept,
            'pct_true_kept': pct_true, 'pct_false_kept': pct_false,
            'pct_false_filtered': filt_false, 'pct_true_filtered': filt_true,
            'ratio': ratio if ratio < 1000 else 999,
        }

    rules = []

    # R_avg_window: mean magnitude on [t-3, t] > threshold
    for thr in [0.05, 0.1, 0.2]:
        def make_fn(threshold):
            def fn(s):
                start = max(0, s - 3)
                return mag[start:s+1].mean() > threshold if s < n else False
            return fn
        rules.append(eval_rule(f'R_avg_window>{thr}', make_fn(thr)))

    # R_ascending: magnitude monotone increasing on [t-3, t]
    def ascending(s):
        if s < 3: return False
        window = mag[s-3:s+1]
        return all(window[i+1] >= window[i] for i in range(len(window)-1))
    rules.append(eval_rule('R_ascending [t-3,t]', ascending))

    # R_slope_mag: linear slope of magnitude on [t-5, t] > threshold
    for thr in [0.001, 0.005, 0.01]:
        def make_fn(threshold):
            def fn(s):
                if s < 5: return False
                window = mag[s-5:s+1]
                slope = np.polyfit(range(6), window, 1)[0]
                return slope > threshold
            return fn
        rules.append(eval_rule(f'R_slope_mag>{thr}', make_fn(thr)))

    # R_max_recent: max magnitude on [t-3, t] > threshold
    for thr in [0.1, 0.2, 0.5]:
        def make_fn(threshold):
            def fn(s):
                start = max(0, s - 3)
                return mag[start:s+1].max() > threshold if s < n else False
            return fn
        rules.append(eval_rule(f'R_max_recent>{thr}', make_fn(thr)))

    print(f"\n  {'Rule':<25} {'True kept%':>10} {'False kept%':>11} {'F filtered%':>12} {'T filtered%':>12} {'Ratio':>7}")
    print(f"  {'-'*80}")

    for r in rules:
        ratio_str = f"{r['ratio']:.1f}x" if r['ratio'] < 100 else "INF"
        print(f"  {r['name']:<25} {r['pct_true_kept']:>9.1f}% {r['pct_false_kept']:>10.1f}% "
              f"{r['pct_false_filtered']:>11.1f}% {r['pct_true_filtered']:>11.1f}% {ratio_str:>7}")

    # =========================================================================
    # ANALYSIS 3 — Combined rule
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS 3 — COMBINED RULES")
    print(f"{'='*70}")

    # R_max_recent > 0.2 AND R_slope_mag > 0
    def combined_1(s):
        if s < 5: return False
        start = max(0, s - 3)
        max_ok = mag[start:s+1].max() > 0.2
        window = mag[s-5:s+1]
        slope = np.polyfit(range(6), window, 1)[0]
        slope_ok = slope > 0
        return max_ok and slope_ok
    r_comb1 = eval_rule('max>0.2 AND slope>0', combined_1)

    # R_max_recent > 0.1 AND R_ascending
    def combined_2(s):
        if s < 3: return False
        start = max(0, s - 3)
        max_ok = mag[start:s+1].max() > 0.1
        window = mag[s-3:s+1]
        asc_ok = all(window[i+1] >= window[i] for i in range(len(window)-1))
        return max_ok and asc_ok
    r_comb2 = eval_rule('max>0.1 AND ascending', combined_2)

    # R_avg_window > 0.1 AND R_slope_mag > 0.005
    def combined_3(s):
        if s < 5: return False
        start = max(0, s - 3)
        avg_ok = mag[start:s+1].mean() > 0.1
        window = mag[s-5:s+1]
        slope = np.polyfit(range(6), window, 1)[0]
        slope_ok = slope > 0.005
        return avg_ok and slope_ok
    r_comb3 = eval_rule('avg>0.1 AND slope>0.005', combined_3)

    combined_rules = [r_comb1, r_comb2, r_comb3]

    print(f"\n  {'Rule':<25} {'True kept%':>10} {'False kept%':>11} {'F filtered%':>12} {'T filtered%':>12} {'Ratio':>7}")
    print(f"  {'-'*80}")
    for r in combined_rules:
        ratio_str = f"{r['ratio']:.1f}x" if r['ratio'] < 100 else "INF"
        print(f"  {r['name']:<25} {r['pct_true_kept']:>9.1f}% {r['pct_false_kept']:>10.1f}% "
              f"{r['pct_false_filtered']:>11.1f}% {r['pct_true_filtered']:>11.1f}% {ratio_str:>7}")

    # =========================================================================
    # VERDICT
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    all_rules = rules + combined_rules
    # Find best rule: highest ratio with >30% true kept
    viable = [r for r in all_rules if r['pct_true_kept'] > 30 and r['pct_false_filtered'] > 10]
    if viable:
        best = max(viable, key=lambda r: r['ratio'])
        print(f"  Best rule: {best['name']}")
        print(f"    True kept: {best['pct_true_kept']:.1f}%, False filtered: {best['pct_false_filtered']:.1f}%")
        print(f"    Ratio: {best['ratio']:.1f}x")
        if best['ratio'] >= 5:
            print(f"    ✅ BREAKTHROUGH — ratio ≥ 5x with {best['pct_true_kept']:.0f}% true kept")
        elif best['ratio'] >= 2:
            print(f"    ⚠️  Moderate — ratio {best['ratio']:.1f}x, useful but not decisive")
        else:
            print(f"    ❌ Insufficient — ratio < 2x")
    else:
        print(f"  ❌ No rule keeps >30% true with meaningful false filtering")

    # Save
    output = {
        'model': reg_name,
        'n_true': len(true_switches),
        'n_false': len(false_switches),
        'trajectory_true_mean': true_mean.tolist(),
        'trajectory_false_mean': false_mean.tolist(),
        'offsets': offsets,
        'true_crescendo_slope': float(true_slope),
        'false_crescendo_slope': float(false_slope),
        'rules': [r for r in all_rules],
    }

    json_path = 'models/magnitude_dynamics_analysis.json'
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {json_path}")


if __name__ == '__main__':
    main()
