#!/usr/bin/env python3
"""
Switch Discrimination Analysis: identify features that distinguish
false switches from true switches using cross-model signals.

For each 30m model X, when X switches direction:
- TRUE switch: oracle transition of X exists within ±3 steps
- FALSE switch: no oracle transition nearby

Then analyze whether the OTHER two models (Y, Z) can discriminate
false from true switches. Test filtering rules and measure their
benefit/cost ratio.

Usage:
    python src/analyze_switch_discrimination.py
"""

import numpy as np
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from constants import PREPARED_DATA_DIR


MODELS_30M = ['macd_30m', 'cci_30m', 'rsi_30m']
PROXIMITY = 3  # ±3 steps to count as "true" switch


def load_model_data(model_name):
    """Load test predictions and labels from NPZ."""
    path = f'{PREPARED_DATA_DIR}/{model_name}_dataset.npz'
    if not Path(path).exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        'y_true': data['y_test'],
        'y_pred_prob': data['y_test_pred'],
        'y_pred_bin': (data['y_test_pred'] > 0.5).astype(int),
    }


def find_switches(preds):
    """Indices where prediction changes."""
    return np.where(np.diff(preds) != 0)[0] + 1


def find_transitions(labels):
    """Indices where oracle label changes."""
    return np.where(np.diff(labels) != 0)[0] + 1


def label_switches(switches, oracle_transitions, proximity=PROXIMITY):
    """
    Label each switch as TRUE or FALSE.
    TRUE: oracle transition exists within ±proximity steps.
    FALSE: no oracle transition nearby.
    """
    labels = []
    for s in switches:
        if len(oracle_transitions) == 0:
            labels.append(False)
            continue
        min_dist = np.abs(oracle_transitions.astype(int) - int(s)).min()
        labels.append(min_dist <= proximity)
    return np.array(labels)


def compute_switch_features(switch_idx, switch_is_true, model_x_data,
                            other_models_data, other_names):
    """
    For each switch of model X, compute features from other models.

    Returns list of dicts with features for each switch.
    """
    x_bin = model_x_data['y_pred_bin']
    n = len(x_bin)
    features_list = []

    for i, s_idx in enumerate(switch_idx):
        if s_idx < 1 or s_idx >= n:
            continue

        # Direction of X before and after the switch
        x_dir_before = x_bin[s_idx - 1]
        x_dir_after = x_bin[s_idx]

        feat = {
            'idx': int(s_idx),
            'is_true': bool(switch_is_true[i]),
            'x_dir_before': int(x_dir_before),
            'x_dir_after': int(x_dir_after),
        }

        for other_data, other_name in zip(other_models_data, other_names):
            y_bin = other_data['y_pred_bin']
            y_prob = other_data['y_pred_prob']

            # Direction of Y at switch time
            feat[f'dir_{other_name}'] = int(y_bin[s_idx])

            # Agreement BEFORE: Y agrees with X's direction before the switch
            feat[f'agree_before_{other_name}'] = int(y_bin[s_idx] == x_dir_before)

            # Agreement AFTER: Y agrees with X's new direction after the switch
            feat[f'agree_after_{other_name}'] = int(y_bin[s_idx] == x_dir_after)

            # Stability: number of Y switches in [s_idx-20, s_idx-1]
            window_start = max(0, s_idx - 20)
            y_segment = y_bin[window_start:s_idx]
            if len(y_segment) > 1:
                feat[f'stability_{other_name}'] = int((np.diff(y_segment) != 0).sum())
            else:
                feat[f'stability_{other_name}'] = 0

            # Probability of Y at switch time
            feat[f'prob_{other_name}'] = float(y_prob[s_idx])

            # Mean probability of Y in [s_idx-5, s_idx+5]
            w_start = max(0, s_idx - 5)
            w_end = min(n, s_idx + 6)
            feat[f'prob_mean_{other_name}'] = float(y_prob[w_start:w_end].mean())

        features_list.append(feat)

    return features_list


def print_distribution_comparison(features_list, feature_name, binary=True):
    """Compare a feature between true and false switches."""
    true_vals = [f[feature_name] for f in features_list if f['is_true']]
    false_vals = [f[feature_name] for f in features_list if not f['is_true']]

    if not true_vals or not false_vals:
        return

    if binary:
        true_pct = np.mean(true_vals) * 100
        false_pct = np.mean(false_vals) * 100
        print(f"    {feature_name:<35} | False: {false_pct:5.1f}% | True: {true_pct:5.1f}% | "
              f"Gap: {abs(false_pct - true_pct):5.1f}pp")
    else:
        true_mean = np.mean(true_vals)
        false_mean = np.mean(false_vals)
        print(f"    {feature_name:<35} | False: {false_mean:6.3f} | True: {true_mean:6.3f} | "
              f"Gap: {abs(false_mean - true_mean):6.3f}")


def test_filtering_rules(features_list, other_names):
    """
    Test filtering rules and measure their discrimination power.
    Returns list of rule results.
    """
    true_switches = [f for f in features_list if f['is_true']]
    false_switches = [f for f in features_list if not f['is_true']]
    n_true = len(true_switches)
    n_false = len(false_switches)

    if n_true == 0 or n_false == 0:
        return []

    y1, y2 = other_names[0], other_names[1]
    rules = []

    # Rule 1: Both others agree with direction BEFORE (= disagree with switch)
    def rule1(f):
        return f[f'agree_before_{y1}'] == 1 and f[f'agree_before_{y2}'] == 1

    # Rule 2: At least 1 other agrees with direction BEFORE
    def rule2(f):
        return f[f'agree_before_{y1}'] == 1 or f[f'agree_before_{y2}'] == 1

    # Rule 3: Mean prob of others in window < 0.6 AND > 0.4 (low confidence)
    def rule3(f):
        p1 = f[f'prob_mean_{y1}']
        p2 = f[f'prob_mean_{y2}']
        avg = (p1 + p2) / 2
        return 0.4 < avg < 0.6

    # Rule 4: Both others are stable (0 recent switches) AND in opposite direction to X's new dir
    def rule4(f):
        stable = f[f'stability_{y1}'] == 0 and f[f'stability_{y2}'] == 0
        opposite = f[f'agree_after_{y1}'] == 0 and f[f'agree_after_{y2}'] == 0
        return stable and opposite

    # Rule 5: Both others agree AFTER (= confirm the switch) — inverse filter
    # This identifies switches we should KEEP, not filter
    def rule5_keep(f):
        return f[f'agree_after_{y1}'] == 1 and f[f'agree_after_{y2}'] == 1

    rule_defs = [
        ('R1: Both contradict switch', rule1),
        ('R2: At least 1 contradicts', rule2),
        ('R3: Others low confidence', rule3),
        ('R4: Others stable + opposite', rule4),
        ('R5: Both CONFIRM (keep rule)', rule5_keep),
    ]

    for name, rule_fn in rule_defs:
        false_filtered = sum(1 for f in false_switches if rule_fn(f))
        true_filtered = sum(1 for f in true_switches if rule_fn(f))

        pct_false = false_filtered / n_false * 100
        pct_true = true_filtered / n_true * 100

        ratio = pct_false / pct_true if pct_true > 0.1 else float('inf')

        rules.append({
            'name': name,
            'false_filtered': false_filtered,
            'true_filtered': true_filtered,
            'pct_false': pct_false,
            'pct_true': pct_true,
            'ratio': ratio,
            'n_true_lost': true_filtered,
        })

    return rules


def analyze_one_model(model_name, all_data):
    """Full analysis for one model X."""
    x_data = all_data[model_name]
    other_names = [m for m in MODELS_30M if m != model_name]
    other_data = [all_data[m] for m in other_names]

    # Find switches and label them
    switches = find_switches(x_data['y_pred_bin'])
    oracle_trans = find_transitions(x_data['y_true'])
    switch_labels = label_switches(switches, oracle_trans)

    n_true = switch_labels.sum()
    n_false = (~switch_labels).sum()

    print(f"\n{'='*80}")
    print(f"MODEL: {model_name}")
    print(f"{'='*80}")
    print(f"  Total switches: {len(switches):,} ({n_true:,} true + {n_false:,} false)")
    print(f"  Oracle transitions: {len(oracle_trans):,}")
    print(f"  True rate: {n_true/len(switches)*100:.1f}%")

    # Compute features
    features = compute_switch_features(switches, switch_labels, x_data, other_data, other_names)

    # Step 3: Distribution comparison
    print(f"\n  --- Feature Distributions (False vs True switches) ---")
    print(f"    {'Feature':<35} | {'False':>10} | {'True':>10} | {'Gap':>8}")
    print(f"    {'-'*75}")

    for other in other_names:
        print_distribution_comparison(features, f'agree_before_{other}', binary=True)
        print_distribution_comparison(features, f'agree_after_{other}', binary=True)
        print_distribution_comparison(features, f'stability_{other}', binary=False)
        print_distribution_comparison(features, f'prob_{other}', binary=False)
        print_distribution_comparison(features, f'prob_mean_{other}', binary=False)
        print()

    # Step 4: Test filtering rules
    rules = test_filtering_rules(features, other_names)

    print(f"  --- Filtering Rules ---")
    print(f"  Total: {n_false:,} false + {n_true:,} true switches")
    print(f"  {'Rule':<35} | {'FalseF%':>8} | {'TrueF%':>8} | {'Ratio':>6} | {'TrueLost':>8}")
    print(f"  {'-'*80}")

    for r in rules:
        ratio_str = f"{r['ratio']:.1f}x" if r['ratio'] < 100 else "INF"
        print(f"  {r['name']:<35} | {r['pct_false']:>7.1f}% | {r['pct_true']:>7.1f}% | "
              f"{ratio_str:>6} | {r['n_true_lost']:>8,}")

    # Best rule
    # Filter out R5 (it's a keep rule, not a filter rule)
    filter_rules = [r for r in rules if 'CONFIRM' not in r['name']]
    if filter_rules:
        # Best = highest ratio with at least 20% false filtered
        viable = [r for r in filter_rules if r['pct_false'] > 20]
        if viable:
            best = max(viable, key=lambda r: r['ratio'])
            print(f"\n  ★ Best rule: {best['name']}")
            print(f"    Filters {best['pct_false']:.1f}% false, loses {best['pct_true']:.1f}% true "
                  f"(ratio {best['ratio']:.1f}x)")
            print(f"    Would reduce false switches from {n_false:,} to {n_false - best['false_filtered']:,}")
            print(f"    Would lose {best['n_true_lost']:,} true switches out of {n_true:,}")

    return {
        'model': model_name,
        'n_switches': len(switches),
        'n_true': int(n_true),
        'n_false': int(n_false),
        'rules': [{k: v for k, v in r.items()} for r in rules],
    }


def main():
    print("=" * 80)
    print("SWITCH DISCRIMINATION ANALYSIS — Cross-model signals")
    print("=" * 80)

    # Load all 30m models
    all_data = {}
    for model_name in MODELS_30M:
        data = load_model_data(model_name)
        if data is None:
            print(f"  SKIP {model_name}: NPZ not found")
            continue
        all_data[model_name] = data
        print(f"  Loaded {model_name}: {len(data['y_true']):,} test samples")

    if len(all_data) < 3:
        print("Need all 3 models loaded!")
        return

    # Verify all test sets have same length
    lengths = {k: len(v['y_true']) for k, v in all_data.items()}
    if len(set(lengths.values())) > 1:
        print(f"  WARNING: Different test set lengths: {lengths}")
        # Truncate to minimum
        min_len = min(lengths.values())
        for k in all_data:
            all_data[k] = {
                'y_true': all_data[k]['y_true'][:min_len],
                'y_pred_prob': all_data[k]['y_pred_prob'][:min_len],
                'y_pred_bin': all_data[k]['y_pred_bin'][:min_len],
            }
        print(f"  Truncated all to {min_len:,} samples")

    # Analyze each model
    results = []
    for model_name in MODELS_30M:
        if model_name in all_data:
            r = analyze_one_model(model_name, all_data)
            results.append(r)

    # Cross-model summary
    print(f"\n{'='*80}")
    print(f"CROSS-MODEL SUMMARY")
    print(f"{'='*80}")

    for r in results:
        filter_rules = [rule for rule in r['rules'] if 'CONFIRM' not in rule['name']]
        viable = [rule for rule in filter_rules if rule['pct_false'] > 20]
        if viable:
            best = max(viable, key=lambda rule: rule['ratio'])
            net_reduction = r['n_false'] * best['pct_false'] / 100
            net_true_lost = best['n_true_lost']
            print(f"  {r['model']}: Best rule '{best['name']}' — "
                  f"removes {net_reduction:.0f}/{r['n_false']} false ({best['pct_false']:.1f}%), "
                  f"loses {net_true_lost}/{r['n_true']} true ({best['pct_true']:.1f}%), "
                  f"ratio={best['ratio']:.1f}x")

    # Save
    json_path = 'models/switch_discrimination.json'
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {json_path}")


if __name__ == '__main__':
    main()
