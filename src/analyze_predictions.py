#!/usr/bin/env python3
"""
Post-training analysis of Net_macd_30m / Net_macd_1h predictions.

KPI oriented toward trading decisions, not just accuracy:
  KPI 1 — Detection latency (how fast does the model detect real transitions?)
  KPI 2 — Plateau oscillations (how many false switches during stable periods?)
  KPI 3 — Switch precision (are model switches near real transitions or spurious?)
  KPI 4 — Probability distribution (confidence profile)

Usage:
    python src/analyze_predictions.py --indicator macd --timeframe 30m
    python src/analyze_predictions.py --indicator macd --timeframe 30m --threshold 0.6
"""

import numpy as np
import pandas as pd
import json
import argparse
import logging
import sys
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from constants import PREPARED_DATA_DIR


def find_transitions(labels):
    """Find indices where the label changes. Returns array of indices."""
    changes = np.where(np.diff(labels) != 0)[0] + 1  # +1: index of the NEW label
    return changes


def find_model_switches(preds):
    """Find indices where model prediction switches."""
    return np.where(np.diff(preds) != 0)[0] + 1


# =============================================================================
# KPI 1 — DETECTION LATENCY
# =============================================================================

def compute_latency(y_true, y_pred_binary):
    """
    For each oracle transition, find the first step where the model
    switches to the correct direction AFTER the transition.

    Returns:
        list of dicts with {oracle_idx, latency, detected}
        latency = -1 means never detected before next transition
    """
    oracle_transitions = find_transitions(y_true)
    n = len(y_true)
    results = []

    for k, t_idx in enumerate(oracle_transitions):
        target_label = y_true[t_idx]

        # Find next oracle transition (boundary for search)
        if k + 1 < len(oracle_transitions):
            next_trans = oracle_transitions[k + 1]
        else:
            next_trans = n

        # Search for first model switch to correct direction
        latency = -1
        for i in range(t_idx, next_trans):
            if y_pred_binary[i] == target_label:
                latency = i - t_idx
                break

        results.append({
            'oracle_idx': int(t_idx),
            'target': int(target_label),
            'latency': latency,
            'detected': latency >= 0,
        })

    return results


def report_latency(latency_results):
    """Print KPI 1 report."""
    latencies = [r['latency'] for r in latency_results if r['detected']]
    n_total = len(latency_results)
    n_detected = len(latencies)
    n_missed = n_total - n_detected

    print(f"\n{'='*60}")
    print(f"KPI 1 — DETECTION LATENCY")
    print(f"{'='*60}")
    print(f"  Total oracle transitions: {n_total:,}")
    print(f"  Detected: {n_detected:,} ({n_detected/n_total*100:.1f}%)")
    print(f"  Never detected: {n_missed:,} ({n_missed/n_total*100:.1f}%)")

    if latencies:
        arr = np.array(latencies)
        print(f"\n  Latency distribution (in 5min steps):")
        print(f"    Mean:   {arr.mean():.1f}")
        print(f"    Median: {np.median(arr):.1f}")
        print(f"    P10:    {np.percentile(arr, 10):.0f}")
        print(f"    P25:    {np.percentile(arr, 25):.0f}")
        print(f"    P75:    {np.percentile(arr, 75):.0f}")
        print(f"    P90:    {np.percentile(arr, 90):.0f}")
        print(f"    P95:    {np.percentile(arr, 95):.0f}")

        # Histogram buckets
        buckets = [(0, 0, '0 (instant)'), (1, 3, '1-3'), (4, 6, '4-6'),
                    (7, 12, '7-12'), (13, 30, '13-30'), (31, 999999, '30+')]
        print(f"\n  Histogram:")
        for lo, hi, label in buckets:
            count = ((arr >= lo) & (arr <= hi)).sum()
            print(f"    {label:>12}: {count:>6,} ({count/len(arr)*100:>5.1f}%)")

        # % detected within 6 steps (< 1 candle 30min)
        within_6 = (arr <= 6).sum()
        print(f"\n  Within 6 steps (<30min): {within_6:,} ({within_6/n_total*100:.1f}% of all transitions)")

    return {
        'n_total': n_total,
        'n_detected': n_detected,
        'n_missed': n_missed,
        'latencies': latencies,
        'mean': float(np.mean(latencies)) if latencies else None,
        'median': float(np.median(latencies)) if latencies else None,
    }


# =============================================================================
# KPI 2 — PLATEAU OSCILLATIONS (FALSE SWITCHES)
# =============================================================================

def compute_plateau_oscillations(y_true, y_pred_binary):
    """
    For each oracle plateau (period between two transitions),
    count model switches.
    """
    oracle_transitions = find_transitions(y_true)
    n = len(y_true)

    # Build plateau boundaries: [0, t1), [t1, t2), [t2, t3), ..., [tk, n)
    boundaries = [0] + list(oracle_transitions) + [n]
    plateaus = []

    for k in range(len(boundaries) - 1):
        start = boundaries[k]
        end = boundaries[k + 1]
        length = end - start

        # Count model switches within this plateau
        pred_segment = y_pred_binary[start:end]
        n_switches = (np.diff(pred_segment) != 0).sum()

        plateaus.append({
            'start': int(start),
            'end': int(end),
            'length': int(length),
            'n_switches': int(n_switches),
            'oracle_label': int(y_true[start]),
        })

    return plateaus


def report_plateaus(plateaus, y_true, y_pred_binary):
    """Print KPI 2 report."""
    n_switches_list = [p['n_switches'] for p in plateaus]
    lengths = [p['length'] for p in plateaus]

    total_model_switches = find_model_switches(y_pred_binary).shape[0]
    total_oracle_switches = find_transitions(y_true).shape[0]

    # Model switch interval
    if total_model_switches > 0:
        avg_model_interval = len(y_pred_binary) / total_model_switches
    else:
        avg_model_interval = float('inf')

    avg_oracle_interval = len(y_true) / total_oracle_switches if total_oracle_switches > 0 else float('inf')

    print(f"\n{'='*60}")
    print(f"KPI 2 — PLATEAU OSCILLATIONS (FALSE SWITCHES)")
    print(f"{'='*60}")
    print(f"  Total plateaus: {len(plateaus):,}")
    print(f"  Avg plateau length: {np.mean(lengths):.1f} steps ({np.mean(lengths)*5:.0f} min)")
    print(f"  Total model switches: {total_model_switches:,}")
    print(f"  Total oracle switches: {total_oracle_switches:,}")
    print(f"  Ratio model/oracle: {total_model_switches/total_oracle_switches:.1f}x")
    print(f"  Avg interval between model switches: {avg_model_interval:.1f} steps ({avg_model_interval*5:.0f} min)")
    print(f"  Avg interval between oracle switches: {avg_oracle_interval:.1f} steps ({avg_oracle_interval*5:.0f} min)")

    # Distribution of switches per plateau
    sw_arr = np.array(n_switches_list)
    print(f"\n  Switches per plateau:")
    for n_sw in [0, 1, 2, 3]:
        count = (sw_arr == n_sw).sum()
        print(f"    {n_sw} switches: {count:>6,} ({count/len(sw_arr)*100:>5.1f}%)")
    count_4plus = (sw_arr >= 4).sum()
    print(f"    4+ switches: {count_4plus:>6,} ({count_4plus/len(sw_arr)*100:>5.1f}%)")
    print(f"\n  Plateaus with 0 switches (ideal): {(sw_arr==0).sum():,} ({(sw_arr==0).mean()*100:.1f}%)")

    return {
        'total_model_switches': total_model_switches,
        'total_oracle_switches': total_oracle_switches,
        'ratio': total_model_switches / total_oracle_switches if total_oracle_switches > 0 else 0,
        'pct_clean_plateaus': float((sw_arr == 0).mean()),
        'avg_model_interval': float(avg_model_interval),
        'avg_oracle_interval': float(avg_oracle_interval),
    }


# =============================================================================
# KPI 3 — SWITCH PRECISION
# =============================================================================

def compute_switch_precision(y_true, y_pred_binary):
    """
    For each model switch, find distance to nearest oracle transition.
    """
    model_switches = find_model_switches(y_pred_binary)
    oracle_transitions = find_transitions(y_true)

    if len(oracle_transitions) == 0 or len(model_switches) == 0:
        return []

    results = []
    for ms in model_switches:
        distances = np.abs(oracle_transitions.astype(int) - int(ms))
        min_dist = distances.min()
        results.append(int(min_dist))

    return results


def report_switch_precision(distances):
    """Print KPI 3 report."""
    print(f"\n{'='*60}")
    print(f"KPI 3 — SWITCH PRECISION")
    print(f"{'='*60}")

    if not distances:
        print(f"  No model switches to analyze")
        return {}

    arr = np.array(distances)
    n = len(arr)

    within_6 = (arr <= 6).sum()
    spurious = (arr > 20).sum()

    print(f"  Total model switches: {n:,}")
    print(f"  Within ±6 steps of real transition: {within_6:,} ({within_6/n*100:.1f}%) — JUSTIFIED")
    print(f"  >20 steps from any transition: {spurious:,} ({spurious/n*100:.1f}%) — SPURIOUS")
    print(f"\n  Distance distribution:")
    print(f"    Mean: {arr.mean():.1f} steps")
    print(f"    Median: {np.median(arr):.1f} steps")

    buckets = [(0, 0, '0 (exact)'), (1, 3, '1-3'), (4, 6, '4-6'),
                (7, 12, '7-12'), (13, 20, '13-20'), (21, 999999, '20+')]
    for lo, hi, label in buckets:
        count = ((arr >= lo) & (arr <= hi)).sum()
        print(f"    {label:>12}: {count:>6,} ({count/n*100:>5.1f}%)")

    return {
        'n_switches': n,
        'pct_justified': float(within_6 / n),
        'pct_spurious': float(spurious / n),
        'mean_distance': float(arr.mean()),
        'median_distance': float(np.median(arr)),
    }


# =============================================================================
# KPI 4 — PROBABILITY DISTRIBUTION
# =============================================================================

def report_probability_distribution(y_true, y_pred_probs):
    """Print KPI 4 report."""
    print(f"\n{'='*60}")
    print(f"KPI 4 — PROBABILITY DISTRIBUTION")
    print(f"{'='*60}")

    # Histogram
    bins = np.arange(0, 1.1, 0.1)
    hist, _ = np.histogram(y_pred_probs, bins=bins)
    n = len(y_pred_probs)

    print(f"\n  Probability histogram:")
    for i in range(len(hist)):
        bar = '█' * int(hist[i] / n * 200)
        print(f"    [{bins[i]:.1f}, {bins[i+1]:.1f}): {hist[i]:>8,} ({hist[i]/n*100:>5.1f}%) {bar}")

    # Grey zone
    grey = ((y_pred_probs >= 0.4) & (y_pred_probs <= 0.6)).sum()
    print(f"\n  Grey zone [0.4, 0.6]: {grey:,} ({grey/n*100:.1f}%)")

    # Proba around transitions
    oracle_transitions = find_transitions(y_true)
    if len(oracle_transitions) > 0:
        # 10 steps before transition
        before_probs = []
        for t in oracle_transitions:
            start = max(0, t - 10)
            if start < t:
                before_probs.extend(y_pred_probs[start:t])

        # Mid-plateau probs
        n_total = len(y_true)
        boundaries = [0] + list(oracle_transitions) + [n_total]
        mid_probs = []
        for k in range(len(boundaries) - 1):
            s = boundaries[k]
            e = boundaries[k + 1]
            length = e - s
            if length > 20:
                mid_start = s + length // 3
                mid_end = s + 2 * length // 3
                mid_probs.extend(y_pred_probs[mid_start:mid_end])

        if before_probs:
            print(f"  Mean prob 10 steps BEFORE transition: {np.mean(before_probs):.4f}")
        if mid_probs:
            print(f"  Mean prob MID-plateau (stable):       {np.mean(mid_probs):.4f}")
        if before_probs and mid_probs:
            diff = abs(np.mean(before_probs) - np.mean(mid_probs))
            print(f"  Difference: {diff:.4f} {'(model distinguishes!)' if diff > 0.05 else '(no distinction)'}")

    return {
        'pct_grey_zone': float(grey / n),
        'mean_before_transition': float(np.mean(before_probs)) if before_probs else None,
        'mean_mid_plateau': float(np.mean(mid_probs)) if mid_probs else None,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze model predictions (trading KPIs)')
    parser.add_argument('--indicator', default='macd', choices=['macd', 'rsi', 'cci'])
    parser.add_argument('--timeframe', default='30m', choices=['30m', '1h'])
    parser.add_argument('--threshold', type=float, default=0.5, help='Decision threshold')
    args = parser.parse_args()

    model_name = f'{args.indicator}_{args.timeframe}'
    npz_path = f'{PREPARED_DATA_DIR}/{model_name}_dataset.npz'

    if not Path(npz_path).exists():
        logger.error(f"NPZ not found: {npz_path}")
        return

    data = np.load(npz_path, allow_pickle=True)
    y_test = data['y_test']
    y_test_pred = data['y_test_pred']
    y_pred_binary = (y_test_pred > args.threshold).astype(int)

    print(f"\n{'='*60}")
    print(f"PREDICTION ANALYSIS — Net_{model_name} (threshold={args.threshold})")
    print(f"{'='*60}")
    print(f"  Test samples: {len(y_test):,}")
    print(f"  Oracle transitions: {len(find_transitions(y_test)):,}")
    print(f"  Model switches: {len(find_model_switches(y_pred_binary)):,}")

    # KPI 1
    latency_results = compute_latency(y_test, y_pred_binary)
    kpi1 = report_latency(latency_results)

    # KPI 2
    plateaus = compute_plateau_oscillations(y_test, y_pred_binary)
    kpi2 = report_plateaus(plateaus, y_test, y_pred_binary)

    # KPI 3
    distances = compute_switch_precision(y_test, y_pred_binary)
    kpi3 = report_switch_precision(distances)

    # KPI 4
    kpi4 = report_probability_distribution(y_test, y_test_pred)

    # Save JSON
    kpi_all = {'kpi1_latency': kpi1, 'kpi2_plateaus': kpi2,
               'kpi3_precision': kpi3, 'kpi4_probs': kpi4,
               'model': model_name, 'threshold': args.threshold}

    # Remove non-serializable
    if 'latencies' in kpi_all['kpi1_latency']:
        kpi_all['kpi1_latency']['latencies'] = None  # too large for JSON

    json_path = f'models/kpi_{model_name}.json'
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(kpi_all, f, indent=2, default=str)
    print(f"\n  KPIs saved: {json_path}")


if __name__ == '__main__':
    main()
