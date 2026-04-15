#!/usr/bin/env python3
"""
Cross-Timeframe Switch Discrimination: can 1h confirm/filter 30m switches
and vice versa?

For each indicator (macd, rsi, cci), tests whether the model at one timeframe
can discriminate false from true switches at the other timeframe.

Analysis A: Filter 30m switches using same-indicator 1h model
Analysis B: Filter 1h switches using same-indicator 30m model

Usage:
    python src/analyze_cross_tf_discrimination.py
"""

import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from constants import PREPARED_DATA_DIR

INDICATORS = ['macd', 'cci', 'rsi']
PROXIMITY = 3


def load_model(indicator, timeframe):
    """Load test data from NPZ. Returns None if not found."""
    path = f'{PREPARED_DATA_DIR}/{indicator}_{timeframe}_dataset.npz'
    if not Path(path).exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        'y_true': data['y_test'],
        'y_pred_prob': data['y_test_pred'],
        'y_pred_bin': (data['y_test_pred'] > 0.5).astype(int),
    }


def find_switches(preds):
    return np.where(np.diff(preds) != 0)[0] + 1


def find_transitions(labels):
    return np.where(np.diff(labels) != 0)[0] + 1


def label_switches(switches, oracle_transitions, proximity=PROXIMITY):
    labels = []
    for s in switches:
        if len(oracle_transitions) == 0:
            labels.append(False)
            continue
        min_dist = np.abs(oracle_transitions.astype(int) - int(s)).min()
        labels.append(min_dist <= proximity)
    return np.array(labels)


def has_switch_in_range(other_switches, start, end):
    """Check if other model has any switch in [start, end]."""
    if len(other_switches) == 0:
        return False
    return np.any((other_switches >= start) & (other_switches <= end))


def analyze_pair(indicator, primary_tf, secondary_tf, primary_data, secondary_data):
    """
    Analyze whether secondary_tf model can filter false switches in primary_tf model.

    Both models are for the SAME indicator but different timeframes.
    """
    n = min(len(primary_data['y_true']), len(secondary_data['y_true']))

    p_bin = primary_data['y_pred_bin'][:n]
    p_true = primary_data['y_true'][:n]
    s_bin = secondary_data['y_pred_bin'][:n]
    s_prob = secondary_data['y_pred_prob'][:n]

    # Primary switches and their labels
    p_switches = find_switches(p_bin)
    p_oracle = find_transitions(p_true)
    p_labels = label_switches(p_switches, p_oracle)

    # Secondary switches
    s_switches = find_switches(s_bin)

    n_true = p_labels.sum()
    n_false = (~p_labels).sum()

    if n_true == 0 or n_false == 0:
        return None

    # Test rules
    rules = []

    for rule_name, rule_fn in [
        (f'{secondary_tf} no switch after [t,t+3]',
         lambda t: not has_switch_in_range(s_switches, t, min(t + 3, n - 1))),

        (f'{secondary_tf} no switch before [t-3,t]',
         lambda t: not has_switch_in_range(s_switches, max(t - 3, 0), t)),

        (f'{secondary_tf} no switch in [t-3,t+3]',
         lambda t: not has_switch_in_range(s_switches, max(t - 3, 0), min(t + 3, n - 1))),

        (f'{secondary_tf} direction disagrees at t+3',
         lambda t: (t + 3 < n) and (s_bin[min(t + 3, n - 1)] != p_bin[min(t, n - 1)])),
    ]:
        false_filt = sum(1 for i, s in enumerate(p_switches)
                        if not p_labels[i] and rule_fn(s))
        true_filt = sum(1 for i, s in enumerate(p_switches)
                       if p_labels[i] and rule_fn(s))

        pct_false = false_filt / n_false * 100
        pct_true = true_filt / n_true * 100
        ratio = pct_false / pct_true if pct_true > 0.1 else float('inf')

        rules.append({
            'name': rule_name,
            'false_filtered': false_filt,
            'true_filtered': true_filt,
            'pct_false': pct_false,
            'pct_true': pct_true,
            'ratio': ratio,
        })

    return {
        'indicator': indicator,
        'primary': f'{indicator}_{primary_tf}',
        'secondary': f'{indicator}_{secondary_tf}',
        'n_switches': len(p_switches),
        'n_true': int(n_true),
        'n_false': int(n_false),
        'n_samples_aligned': n,
        'rules': rules,
    }


def print_analysis(result):
    """Print analysis for one pair."""
    if result is None:
        return

    print(f"\n  {result['primary']} filtered by {result['secondary']}")
    print(f"  Switches: {result['n_false']:,} false + {result['n_true']:,} true "
          f"(aligned on {result['n_samples_aligned']:,} samples)")
    print(f"  {'Rule':<45} | {'FalseF%':>8} | {'TrueF%':>8} | {'Ratio':>6}")
    print(f"  {'-'*78}")

    for r in result['rules']:
        ratio_str = f"{r['ratio']:.1f}x" if r['ratio'] < 100 else "INF"
        print(f"  {r['name']:<45} | {r['pct_false']:>7.1f}% | {r['pct_true']:>7.1f}% | {ratio_str:>6}")


def main():
    print("=" * 80)
    print("CROSS-TIMEFRAME SWITCH DISCRIMINATION")
    print("=" * 80)

    # Load all 6 models
    models = {}
    for ind in INDICATORS:
        for tf in ['30m', '1h']:
            key = f'{ind}_{tf}'
            data = load_model(ind, tf)
            if data:
                models[key] = data
                print(f"  Loaded {key}: {len(data['y_true']):,} test samples")
            else:
                print(f"  SKIP {key}: not found")

    all_results = []

    for ind in INDICATORS:
        key_30m = f'{ind}_30m'
        key_1h = f'{ind}_1h'

        if key_30m not in models or key_1h not in models:
            print(f"\n  SKIP {ind}: need both 30m and 1h")
            continue

        print(f"\n{'='*80}")
        print(f"INDICATOR: {ind.upper()}")
        print(f"{'='*80}")

        # Analysis A: Filter 30m using 1h
        print(f"\n  --- Analysis A: Filter {ind}_30m switches using {ind}_1h ---")
        result_a = analyze_pair(ind, '30m', '1h', models[key_30m], models[key_1h])
        if result_a:
            print_analysis(result_a)
            all_results.append(result_a)

        # Analysis B: Filter 1h using 30m
        print(f"\n  --- Analysis B: Filter {ind}_1h switches using {ind}_30m ---")
        result_b = analyze_pair(ind, '1h', '30m', models[key_1h], models[key_30m])
        if result_b:
            print_analysis(result_b)
            all_results.append(result_b)

        # Leadership analysis
        if result_a and result_b:
            best_a = max([r for r in result_a['rules'] if r['pct_false'] > 15],
                        key=lambda r: r['ratio'], default=None)
            best_b = max([r for r in result_b['rules'] if r['pct_false'] > 15],
                        key=lambda r: r['ratio'], default=None)

            print(f"\n  --- Pair Summary: {ind.upper()} ---")
            if best_a:
                print(f"  Best filter for {ind}_30m: {best_a['name']} "
                      f"(ratio {best_a['ratio']:.1f}x, "
                      f"filters {best_a['pct_false']:.1f}% false, "
                      f"loses {best_a['pct_true']:.1f}% true)")
            if best_b:
                print(f"  Best filter for {ind}_1h:  {best_b['name']} "
                      f"(ratio {best_b['ratio']:.1f}x, "
                      f"filters {best_b['pct_false']:.1f}% false, "
                      f"loses {best_b['pct_true']:.1f}% true)")

            if best_a and best_b:
                if best_a['ratio'] > best_b['ratio'] * 1.3:
                    print(f"  Leadership: 1h leads (better at filtering 30m)")
                elif best_b['ratio'] > best_a['ratio'] * 1.3:
                    print(f"  Leadership: 30m leads (better at filtering 1h)")
                else:
                    print(f"  Leadership: quasi-simultaneous (similar ratios)")

    # Global summary
    print(f"\n{'='*80}")
    print(f"GLOBAL SUMMARY")
    print(f"{'='*80}")

    for r in all_results:
        viable = [rule for rule in r['rules'] if rule['pct_false'] > 15]
        if viable:
            best = max(viable, key=lambda rule: rule['ratio'])
            print(f"  {r['primary']:<12} filtered by {r['secondary']:<12}: "
                  f"best ratio={best['ratio']:.1f}x "
                  f"({best['pct_false']:.0f}% false / {best['pct_true']:.0f}% true) "
                  f"— {best['name']}")
        else:
            print(f"  {r['primary']:<12} filtered by {r['secondary']:<12}: no viable rule")

    # Save
    json_path = 'models/switch_discrimination_bidirectional.json'
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)

    # Clean for JSON serialization
    for r in all_results:
        for rule in r['rules']:
            if rule['ratio'] == float('inf'):
                rule['ratio'] = 999

    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {json_path}")


if __name__ == '__main__':
    main()
