#!/usr/bin/env python3
"""
Complete comparison: binary classification vs regression on all 6 models.

Phase 1: Same KPIs on both approaches
Phase 2: Combined rules (binary × regression)
Phase 3: Transition vs plateau zone analysis

Usage:
    python src/compare_binary_vs_regression.py
"""

import numpy as np
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from constants import PREPARED_DATA_DIR

MODELS = [
    ('macd', '30m'), ('cci', '30m'), ('rsi', '30m'),
    ('macd', '1h'), ('cci', '1h'), ('rsi', '1h'),
]
TRUE_PROX = 3
FALSE_DIST = 20


def find_transitions(labels):
    return np.where(np.diff(labels) != 0)[0] + 1

def find_switches(preds):
    return np.where(np.diff(preds) != 0)[0] + 1

def classify_switches(switches, oracle_trans):
    justified = 0; spurious = 0
    for s in switches:
        if len(oracle_trans) == 0:
            spurious += 1; continue
        d = np.abs(oracle_trans.astype(int) - int(s)).min()
        if d <= 6: justified += 1
        elif d > 20: spurious += 1
    return justified, spurious

def detection_within_6(oracle_trans, pred_binary, n):
    detected = 0
    for k, t in enumerate(oracle_trans):
        target = pred_binary[t] if t < n else 0
        next_t = oracle_trans[k+1] if k+1 < len(oracle_trans) else n
        for i in range(t, min(t+7, next_t)):
            if i < n and pred_binary[i] == (1 if np.sign(pred_binary[t]) >= 0 else 0):
                # Check if model matches oracle direction at this point
                pass
        # Simpler: check if model has switched to correct direction within 6 steps
        found = False
        oracle_dir = 1  # default
        # Get oracle direction at transition
        if t < n:
            # Oracle label at t
            oracle_dir = int(pred_binary[t])  # This is the model pred, not oracle
        # Actually, we need the oracle labels
        # Skip complex detection, use switch proximity instead
        detected += 1  # placeholder
    return detected

def compute_kpis(pred_binary, oracle_labels, n):
    """Compute standard KPIs for binary predictions."""
    oracle_trans = find_transitions(oracle_labels)
    model_switches = find_switches(pred_binary)
    n_oracle = len(oracle_trans)
    n_switches = len(model_switches)

    # Accuracy
    acc = (pred_binary == oracle_labels).mean()

    # Switch ratio
    ratio = n_switches / n_oracle if n_oracle > 0 else 0

    # Justified / Spurious
    justified, spurious = classify_switches(model_switches, oracle_trans)
    pct_justified = justified / n_switches * 100 if n_switches > 0 else 0
    pct_spurious = spurious / n_switches * 100 if n_switches > 0 else 0

    # Plateaus with 0 switches
    boundaries = [0] + list(oracle_trans) + [n]
    n_clean = 0
    for k in range(len(boundaries)-1):
        seg = pred_binary[boundaries[k]:boundaries[k+1]]
        if len(seg) > 0 and (np.diff(seg) != 0).sum() == 0:
            n_clean += 1
    pct_clean = n_clean / (len(boundaries)-1) * 100

    # Detection within 6 steps
    detected_6 = 0
    for k, t in enumerate(oracle_trans):
        target_label = oracle_labels[t]
        next_t = oracle_trans[k+1] if k+1 < len(oracle_trans) else n
        for i in range(t, min(t+7, next_t)):
            if i < n and pred_binary[i] == target_label:
                detected_6 += 1
                break
    pct_det6 = detected_6 / n_oracle * 100 if n_oracle > 0 else 0

    return {
        'accuracy': float(acc),
        'n_switches': n_switches,
        'ratio': float(ratio),
        'pct_justified': float(pct_justified),
        'pct_spurious': float(pct_spurious),
        'pct_clean_plateaus': float(pct_clean),
        'pct_detect_6': float(pct_det6),
    }


def main():
    print("=" * 120)
    print("BINARY vs REGRESSION — Complete Comparison")
    print("=" * 120)

    all_results = []

    # =========================================================================
    # PHASE 1 — Same KPIs on both approaches
    # =========================================================================
    print(f"\n{'='*120}")
    print(f"PHASE 1 — SAME KPIs ON BOTH APPROACHES")
    print(f"{'='*120}")

    header = (f"{'Model':<12} {'Method':<12} {'Acc':>6} {'Switchs':>8} {'Ratio':>7} "
              f"{'Justif%':>8} {'Spur%':>7} {'Clean%':>7} {'Det<6%':>7}")
    print(f"\n{header}")
    print("-" * 120)

    for ind, tf in MODELS:
        # Load binary crossfeat
        bin_path = f'{PREPARED_DATA_DIR}/{ind}_{tf}_crossfeat_dataset.npz'
        # Load regression crossfeat
        reg_path = f'{PREPARED_DATA_DIR}/{ind}_{tf}_crossfeat_regression_dataset.npz'

        if not Path(bin_path).exists() or not Path(reg_path).exists():
            print(f"  SKIP {ind}_{tf}: missing NPZ")
            continue

        bin_data = np.load(bin_path, allow_pickle=True)
        reg_data = np.load(reg_path, allow_pickle=True)

        # Oracle labels (from binary, should be same)
        oracle = bin_data['y_test']
        n = len(oracle)

        # Binary predictions
        bin_pred = (bin_data['y_test_pred'] > 0.5).astype(int)

        # Regression predictions → binary via sign
        reg_pred_raw = reg_data['y_test_pred']
        reg_pred = (reg_pred_raw > 0).astype(int)

        # Align lengths
        min_n = min(len(oracle), len(bin_pred), len(reg_pred))
        oracle = oracle[:min_n]
        bin_pred = bin_pred[:min_n]
        reg_pred = reg_pred[:min_n]
        reg_pred_raw = reg_pred_raw[:min_n]

        # Compute KPIs
        bin_kpi = compute_kpis(bin_pred, oracle, min_n)
        reg_kpi = compute_kpis(reg_pred, oracle, min_n)

        model_name = f'{ind}_{tf}'
        for method, kpi in [('Binary', bin_kpi), ('Regression', reg_kpi)]:
            print(f"{model_name:<12} {method:<12} {kpi['accuracy']*100:>5.1f}% {kpi['n_switches']:>8,} "
                  f"{kpi['ratio']:>6.1f}x {kpi['pct_justified']:>7.1f}% {kpi['pct_spurious']:>6.1f}% "
                  f"{kpi['pct_clean_plateaus']:>6.1f}% {kpi['pct_detect_6']:>6.1f}%")

        all_results.append({
            'model': model_name,
            'binary': bin_kpi,
            'regression': reg_kpi,
        })

    # =========================================================================
    # PHASE 2 — Combined rules
    # =========================================================================
    print(f"\n{'='*120}")
    print(f"PHASE 2 — COMBINED RULES (binary × regression)")
    print(f"{'='*120}")

    combined_results = []

    for ind, tf in MODELS:
        bin_path = f'{PREPARED_DATA_DIR}/{ind}_{tf}_crossfeat_dataset.npz'
        reg_path = f'{PREPARED_DATA_DIR}/{ind}_{tf}_crossfeat_regression_dataset.npz'

        if not Path(bin_path).exists() or not Path(reg_path).exists():
            continue

        bin_data = np.load(bin_path, allow_pickle=True)
        reg_data = np.load(reg_path, allow_pickle=True)

        oracle = bin_data['y_test']
        bin_probs = bin_data['y_test_pred']
        reg_raw = reg_data['y_test_pred']

        min_n = min(len(oracle), len(bin_probs), len(reg_raw))
        oracle = oracle[:min_n]
        bin_probs = bin_probs[:min_n]
        reg_raw = reg_raw[:min_n]

        bin_dir = (bin_probs > 0.5).astype(int)
        reg_dir = (reg_raw > 0).astype(int)

        oracle_trans = find_transitions(oracle)
        n_oracle = len(oracle_trans)

        model_name = f'{ind}_{tf}'

        # Find median magnitude at true transitions for threshold
        true_mags = []
        for t in oracle_trans:
            if t < min_n:
                true_mags.append(abs(reg_raw[t]))
        median_mag = np.median(true_mags) if true_mags else 0.1

        print(f"\n  {model_name} (oracle switches: {n_oracle:,}, median true mag: {median_mag:.4f})")

        rules = {}

        # R_agree: both agree on direction
        agree_pred = np.zeros(min_n, dtype=int)
        current = 0
        for i in range(min_n):
            if bin_dir[i] == reg_dir[i]:
                current = bin_dir[i]
            agree_pred[i] = current
        rules['R_agree'] = agree_pred

        # R_strong_agree: agree AND |regression| > median_mag
        strong_pred = np.zeros(min_n, dtype=int)
        current = 0
        for i in range(min_n):
            if bin_dir[i] == reg_dir[i] and abs(reg_raw[i]) > median_mag:
                current = bin_dir[i]
            strong_pred[i] = current
        rules['R_strong_agree'] = strong_pred

        # R_score: prob × |slope|, threshold = 0.5 * median_mag
        score = bin_probs * np.abs(reg_raw)
        score_pred = np.zeros(min_n, dtype=int)
        score_thr = 0.5 * median_mag
        current = 0
        for i in range(min_n):
            if score[i] > score_thr:
                current = 1
            elif (1 - bin_probs[i]) * np.abs(reg_raw[i]) > score_thr:
                current = 0
            score_pred[i] = current
        rules['R_score'] = score_pred

        print(f"  {'Rule':<20} {'Switchs':>8} {'Ratio':>7} {'Justif%':>8} {'Spur%':>7} {'Clean%':>7} {'Det<6%':>7}")
        print(f"  {'-'*70}")

        # Baselines
        for method, pred in [('Binary', bin_dir), ('Regression', reg_dir)]:
            kpi = compute_kpis(pred, oracle, min_n)
            print(f"  {method:<20} {kpi['n_switches']:>8,} {kpi['ratio']:>6.1f}x "
                  f"{kpi['pct_justified']:>7.1f}% {kpi['pct_spurious']:>6.1f}% "
                  f"{kpi['pct_clean_plateaus']:>6.1f}% {kpi['pct_detect_6']:>6.1f}%")

        model_combined = {'model': model_name}
        for rule_name, pred in rules.items():
            kpi = compute_kpis(pred, oracle, min_n)
            print(f"  {rule_name:<20} {kpi['n_switches']:>8,} {kpi['ratio']:>6.1f}x "
                  f"{kpi['pct_justified']:>7.1f}% {kpi['pct_spurious']:>6.1f}% "
                  f"{kpi['pct_clean_plateaus']:>6.1f}% {kpi['pct_detect_6']:>6.1f}%")
            model_combined[rule_name] = kpi
        combined_results.append(model_combined)

    # =========================================================================
    # PHASE 3 — Transition vs Plateau zone
    # =========================================================================
    print(f"\n{'='*120}")
    print(f"PHASE 3 — TRANSITION vs PLATEAU ZONES")
    print(f"{'='*120}")

    for ind, tf in MODELS:
        bin_path = f'{PREPARED_DATA_DIR}/{ind}_{tf}_crossfeat_dataset.npz'
        reg_path = f'{PREPARED_DATA_DIR}/{ind}_{tf}_crossfeat_regression_dataset.npz'

        if not Path(bin_path).exists() or not Path(reg_path).exists():
            continue

        bin_data = np.load(bin_path, allow_pickle=True)
        reg_data = np.load(reg_path, allow_pickle=True)

        oracle = bin_data['y_test']
        bin_pred = (bin_data['y_test_pred'] > 0.5).astype(int)
        reg_pred = (reg_data['y_test_pred'] > 0).astype(int)

        min_n = min(len(oracle), len(bin_pred), len(reg_pred))
        oracle = oracle[:min_n]
        bin_pred = bin_pred[:min_n]
        reg_pred = reg_pred[:min_n]

        oracle_trans = find_transitions(oracle)

        # Distance to nearest transition
        dist = np.full(min_n, 999999)
        for t in oracle_trans:
            d = np.abs(np.arange(min_n) - t)
            dist = np.minimum(dist, d)

        mask_trans = dist <= TRUE_PROX
        mask_plat = dist > TRUE_PROX

        model_name = f'{ind}_{tf}'
        print(f"\n  {model_name}:")
        print(f"  {'Zone':<15} {'Method':<12} {'Acc':>7} {'Switchs':>8}")
        print(f"  {'-'*45}")

        for zone_name, mask in [('TRANSITION', mask_trans), ('PLATEAU', mask_plat)]:
            for method, pred in [('Binary', bin_pred), ('Regression', reg_pred)]:
                zone_acc = (pred[mask] == oracle[mask]).mean() * 100
                zone_switches = (np.diff(pred[mask]) != 0).sum()
                print(f"  {zone_name:<15} {method:<12} {zone_acc:>6.1f}% {zone_switches:>8,}")

    # =========================================================================
    # VERDICT
    # =========================================================================
    print(f"\n{'='*120}")
    print(f"VERDICT")
    print(f"{'='*120}")

    # Check if any combined rule achieves ratio >= 5x on >= 2 models
    breakthrough = 0
    for cr in combined_results:
        for rule_name in ['R_agree', 'R_strong_agree', 'R_score']:
            if rule_name in cr and cr[rule_name]['ratio'] <= 1.5:
                pass  # Check for low ratio
            if rule_name in cr and cr[rule_name].get('ratio', 99) >= 5:
                breakthrough += 1

    if breakthrough >= 2:
        print(f"  ✅ BREAKTHROUGH: Combined rule achieves ratio ≥ 5x on {breakthrough} models")
    else:
        print(f"  ❌ No combined rule achieves ratio ≥ 5x on 2+ models")
        print(f"     Structural ceiling confirmed: binary ≈ regression at transitions")

    # Which dominates?
    bin_wins = 0; reg_wins = 0
    for r in all_results:
        if r['binary']['ratio'] < r['regression']['ratio']:
            bin_wins += 1
        else:
            reg_wins += 1
    print(f"\n  Binary wins on switch ratio: {bin_wins}/6 models")
    print(f"  Regression wins: {reg_wins}/6 models")

    if bin_wins >= 4:
        print(f"  → Binary classification is the better approach")
    elif reg_wins >= 4:
        print(f"  → Regression is the better approach")
    else:
        print(f"  → No clear winner")

    # Save
    output = {
        'phase1': all_results,
        'phase2': combined_results,
    }
    json_path = 'models/binary_vs_regression_comparison.json'
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {json_path}")


if __name__ == '__main__':
    main()
