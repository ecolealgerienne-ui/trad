#!/usr/bin/env python3
"""
Cross-architecture switch discrimination: can combining CNN-LSTM, CNN-GRU,
and TCN predictions reduce false switches?

Hypothesis: different architectures may make different errors. If all 3
agree on a switch, it's more likely to be real.

Usage:
    python src/analyze_cross_arch_switches.py
"""

import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from constants import PREPARED_DATA_DIR

TRUE_PROX = 3
FALSE_DIST = 20

ARCHS = {
    'CNN-LSTM': 'macd_30m_crossfeat_dataset.npz',
    'CNN-GRU': 'macd_30m_crossfeat_cnngru_dataset.npz',
    'TCN': 'macd_30m_crossfeat_tcn_dataset.npz',
}


def find_switches(preds):
    return np.where(np.diff(preds) != 0)[0] + 1


def find_transitions(labels):
    return np.where(np.diff(labels) != 0)[0] + 1


def has_switch_nearby(switches, t, proximity=3):
    """Check if any switch exists within ±proximity of t."""
    if len(switches) == 0:
        return False
    return np.abs(switches.astype(int) - int(t)).min() <= proximity


def main():
    print("=" * 80)
    print("CROSS-ARCHITECTURE SWITCH DISCRIMINATION")
    print("=" * 80)

    # Load all 3 architectures
    arch_data = {}
    for name, filename in ARCHS.items():
        path = f'{PREPARED_DATA_DIR}/{filename}'
        if not Path(path).exists():
            print(f"  SKIP {name}: {path} not found")
            continue
        data = np.load(path, allow_pickle=True)
        pred_bin = (data['y_test_pred'] > 0.5).astype(int)
        arch_data[name] = {
            'y_test': data['y_test'],
            'y_pred': pred_bin,
            'switches': find_switches(pred_bin),
        }
        print(f"  Loaded {name}: {len(pred_bin):,} samples, {len(arch_data[name]['switches']):,} switches")

    if len(arch_data) < 3:
        print("Need all 3 architectures!")
        return

    # Align lengths
    min_n = min(len(d['y_test']) for d in arch_data.values())
    oracle = list(arch_data.values())[0]['y_test'][:min_n]
    oracle_trans = find_transitions(oracle)

    for name in arch_data:
        arch_data[name]['y_pred'] = arch_data[name]['y_pred'][:min_n]
        arch_data[name]['switches'] = find_switches(arch_data[name]['y_pred'])

    print(f"\n  Aligned on {min_n:,} samples")
    print(f"  Oracle transitions: {len(oracle_trans):,}")

    arch_names = list(arch_data.keys())

    # =========================================================================
    # ANALYSIS 1 — Error correlation between architectures
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"ANALYSIS 1 — ERROR CORRELATION")
    print(f"{'='*80}")

    errors = {}
    for name in arch_names:
        errors[name] = (arch_data[name]['y_pred'] != oracle).astype(int)

    print(f"\n  Error rates:")
    for name in arch_names:
        print(f"    {name}: {errors[name].mean()*100:.2f}%")

    print(f"\n  Error overlap (% of errors shared between pairs):")
    for i, n1 in enumerate(arch_names):
        for n2 in arch_names[i+1:]:
            both_wrong = ((errors[n1] == 1) & (errors[n2] == 1)).sum()
            either_wrong = ((errors[n1] == 1) | (errors[n2] == 1)).sum()
            overlap = both_wrong / either_wrong * 100 if either_wrong > 0 else 0
            print(f"    {n1} ∩ {n2}: {overlap:.1f}% overlap ({both_wrong:,} / {either_wrong:,})")

    all_wrong = ((errors[arch_names[0]] == 1) &
                 (errors[arch_names[1]] == 1) &
                 (errors[arch_names[2]] == 1)).sum()
    any_wrong = ((errors[arch_names[0]] == 1) |
                 (errors[arch_names[1]] == 1) |
                 (errors[arch_names[2]] == 1)).sum()
    print(f"\n  All 3 wrong simultaneously: {all_wrong:,} ({all_wrong/min_n*100:.2f}%)")
    print(f"  At least 1 wrong: {any_wrong:,} ({any_wrong/min_n*100:.2f}%)")
    print(f"  Jaccard overlap (3-way): {all_wrong/any_wrong*100:.1f}%")

    # =========================================================================
    # ANALYSIS 2 — Switch agreement
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"ANALYSIS 2 — SWITCH AGREEMENT")
    print(f"{'='*80}")

    # For each CNN-LSTM switch, check if GRU and TCN also switch nearby
    primary = 'CNN-LSTM'
    others = [n for n in arch_names if n != primary]
    primary_switches = arch_data[primary]['switches']

    agree_counts = {0: [], 1: [], 2: []}  # 0 others agree, 1 agrees, 2 agree

    for s in primary_switches:
        n_agree = sum(1 for other in others
                     if has_switch_nearby(arch_data[other]['switches'], s, proximity=3))

        # Classify as true or false
        if len(oracle_trans) > 0:
            min_dist = np.abs(oracle_trans.astype(int) - int(s)).min()
        else:
            min_dist = 999999

        is_true = min_dist <= TRUE_PROX
        is_false = min_dist > FALSE_DIST

        agree_counts[n_agree].append({
            'idx': int(s),
            'is_true': is_true,
            'is_false': is_false,
            'n_agree': n_agree,
        })

    print(f"\n  {primary} switches: {len(primary_switches):,}")
    print(f"\n  {'Agreement':<20} {'Total':>8} {'True':>8} {'False':>8} {'True%':>8} {'False%':>8}")
    print(f"  {'-'*60}")

    for n_agree in [0, 1, 2]:
        items = agree_counts[n_agree]
        n_total = len(items)
        n_true = sum(1 for x in items if x['is_true'])
        n_false = sum(1 for x in items if x['is_false'])
        pct_true = n_true / n_total * 100 if n_total > 0 else 0
        pct_false = n_false / n_total * 100 if n_total > 0 else 0
        label = f"{n_agree}/2 others agree"
        print(f"  {label:<20} {n_total:>8,} {n_true:>8,} {n_false:>8,} {pct_true:>7.1f}% {pct_false:>7.1f}%")

    # =========================================================================
    # ANALYSIS 3 — Voting rules
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"ANALYSIS 3 — VOTING RULES")
    print(f"{'='*80}")

    # Build voted predictions
    all_preds = np.stack([arch_data[n]['y_pred'] for n in arch_names], axis=0)  # (3, n)

    rules_results = []

    for vote_threshold, rule_name in [(1, 'Any 1/3 switches'),
                                       (2, 'Majority 2/3'),
                                       (3, 'Unanimous 3/3')]:
        # Majority vote at each step
        vote_sum = all_preds.sum(axis=0)  # 0, 1, 2, or 3
        voted_pred = (vote_sum >= 2).astype(int)  # majority

        # For voting on switches: only switch if N architectures switch nearby
        # Build filtered prediction: hold direction unless vote_threshold architectures switch
        filtered = np.zeros(min_n, dtype=int)
        filtered[0] = voted_pred[0]

        for i in range(1, min_n):
            # Count how many architectures switch at this step
            n_switching = 0
            for name in arch_names:
                if arch_data[name]['y_pred'][i] != arch_data[name]['y_pred'][i-1]:
                    n_switching += 1

            if n_switching >= vote_threshold:
                filtered[i] = voted_pred[i]
            else:
                filtered[i] = filtered[i-1]

        switches = find_switches(filtered)
        n_switches = len(switches)
        ratio = n_switches / len(oracle_trans) if len(oracle_trans) > 0 else 0

        # Classify switches
        justified = 0; spurious = 0
        for s in switches:
            if len(oracle_trans) > 0:
                d = np.abs(oracle_trans.astype(int) - int(s)).min()
                if d <= 6: justified += 1
                elif d > 20: spurious += 1

        pct_just = justified / n_switches * 100 if n_switches > 0 else 0
        pct_spur = spurious / n_switches * 100 if n_switches > 0 else 0

        # Detection
        detected = 0
        for t in oracle_trans:
            target = oracle[t]
            for i in range(t, min(t + 7, min_n)):
                if filtered[i] == target:
                    detected += 1
                    break
        pct_det = detected / len(oracle_trans) * 100

        rules_results.append({
            'rule': rule_name,
            'vote_threshold': vote_threshold,
            'switches': n_switches,
            'ratio': ratio,
            'pct_justified': pct_just,
            'pct_spurious': pct_spur,
            'pct_detect_6': pct_det,
        })

    # Add single-arch baselines
    for name in arch_names:
        sw = arch_data[name]['switches']
        n_sw = len(sw)
        justified = 0; spurious = 0
        for s in sw:
            if len(oracle_trans) > 0:
                d = np.abs(oracle_trans.astype(int) - int(s)).min()
                if d <= 6: justified += 1
                elif d > 20: spurious += 1

        detected = 0
        for t in oracle_trans:
            target = oracle[t]
            for i in range(t, min(t + 7, min_n)):
                if arch_data[name]['y_pred'][i] == target:
                    detected += 1
                    break

        rules_results.append({
            'rule': f'{name} (baseline)',
            'vote_threshold': 0,
            'switches': n_sw,
            'ratio': n_sw / len(oracle_trans),
            'pct_justified': justified / n_sw * 100 if n_sw > 0 else 0,
            'pct_spurious': spurious / n_sw * 100 if n_sw > 0 else 0,
            'pct_detect_6': detected / len(oracle_trans) * 100,
        })

    print(f"\n  {'Rule':<25} {'Switchs':>8} {'Ratio':>7} {'Justif%':>8} {'Spur%':>7} {'Det<6%':>7}")
    print(f"  {'-'*70}")

    for r in rules_results:
        print(f"  {r['rule']:<25} {r['switches']:>8,} {r['ratio']:>6.1f}x "
              f"{r['pct_justified']:>7.1f}% {r['pct_spurious']:>6.1f}% {r['pct_detect_6']:>6.1f}%")

    # =========================================================================
    # ANALYSIS 4 — Discrimination ratio for filtering
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"ANALYSIS 4 — FILTER RATIO (using agreement to filter CNN-LSTM)")
    print(f"{'='*80}")

    # For each CNN-LSTM switch, filter if fewer than N others agree
    for min_agree, rule_name in [(1, 'Keep if ≥1 other agrees'),
                                  (2, 'Keep if both others agree')]:
        true_kept = 0; true_total = 0
        false_kept = 0; false_total = 0

        for s in primary_switches:
            n_agree = sum(1 for other in others
                         if has_switch_nearby(arch_data[other]['switches'], s, proximity=3))

            if len(oracle_trans) > 0:
                min_dist = np.abs(oracle_trans.astype(int) - int(s)).min()
            else:
                min_dist = 999999

            is_true = min_dist <= TRUE_PROX
            is_false = min_dist > FALSE_DIST
            kept = n_agree >= min_agree

            if is_true:
                true_total += 1
                if kept: true_kept += 1
            if is_false:
                false_total += 1
                if kept: false_kept += 1

        pct_true_kept = true_kept / true_total * 100 if true_total > 0 else 0
        pct_false_kept = false_kept / false_total * 100 if false_total > 0 else 0
        pct_true_filt = 100 - pct_true_kept
        pct_false_filt = 100 - pct_false_kept
        ratio = pct_false_filt / pct_true_filt if pct_true_filt > 0.1 else float('inf')

        print(f"\n  {rule_name}:")
        print(f"    True:  {true_kept}/{true_total} kept ({pct_true_kept:.1f}%), {pct_true_filt:.1f}% filtered")
        print(f"    False: {false_kept}/{false_total} kept ({pct_false_kept:.1f}%), {pct_false_filt:.1f}% filtered")
        print(f"    Ratio: {ratio:.1f}x {'✅ ≥5x' if ratio >= 5 else '❌ <5x'}")

    # =========================================================================
    # VERDICT
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"VERDICT")
    print(f"{'='*80}")

    # Check error overlap
    jaccard = all_wrong / any_wrong * 100 if any_wrong > 0 else 0

    if jaccard > 80:
        print(f"  ❌ Errors highly correlated ({jaccard:.0f}% Jaccard) — architectures make SAME errors")
    elif jaccard > 50:
        print(f"  ⚠️  Moderate error correlation ({jaccard:.0f}% Jaccard)")
    else:
        print(f"  ✅ Low error correlation ({jaccard:.0f}% Jaccard) — diversity useful")

    # Best voting rule
    voting_only = [r for r in rules_results if r['vote_threshold'] > 0]
    if voting_only:
        best = min(voting_only, key=lambda r: r['ratio'])
        print(f"\n  Best voting rule: {best['rule']}")
        print(f"    Ratio: {best['ratio']:.1f}x, Spurious: {best['pct_spurious']:.1f}%, Det<6: {best['pct_detect_6']:.1f}%")

    # Save
    output = {
        'error_jaccard_3way': float(jaccard),
        'rules': rules_results,
    }
    json_path = 'models/cross_arch_analysis.json'
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {json_path}")


if __name__ == '__main__':
    main()
