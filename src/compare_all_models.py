#!/usr/bin/env python3
"""
Run analyze_predictions.py for all 6 models and produce a comparison table.

Usage:
    python src/compare_all_models.py
"""

import numpy as np
import json
import sys
from pathlib import Path
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from constants import PREPARED_DATA_DIR


def find_transitions(labels):
    return np.where(np.diff(labels) != 0)[0] + 1

def find_switches(preds):
    return np.where(np.diff(preds) != 0)[0] + 1


def analyze_model(indicator, timeframe, threshold=0.5):
    """Run full KPI analysis for one model. Returns dict of metrics."""
    model_name = f'{indicator}_{timeframe}'
    npz_path = f'{PREPARED_DATA_DIR}/{model_name}_dataset.npz'

    if not Path(npz_path).exists():
        return None

    data = np.load(npz_path, allow_pickle=True)
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    y_pred = data['y_test_pred']
    y_bin = (y_pred > threshold).astype(int)

    n_test = len(y_test)
    oracle_trans = find_transitions(y_test)
    model_switches = find_switches(y_bin)
    n_oracle = len(oracle_trans)
    n_model_sw = len(model_switches)

    # Persistence baseline
    persistence = np.zeros_like(y_test)
    persistence[0] = y_test[0]
    persistence[1:] = y_test[:-1]
    persistence_acc = (persistence == y_test).mean()

    # Val accuracy (from training — approximate from test pred)
    val_acc = (y_bin == y_test).mean()  # test accuracy as proxy

    # AUC
    try:
        auc = roc_auc_score(y_test, y_pred)
    except:
        auc = 0.0

    # Grey zone
    grey = ((y_pred >= 0.4) & (y_pred <= 0.6)).mean()

    # --- KPI 1: Latency ---
    latencies = []
    for k, t_idx in enumerate(oracle_trans):
        target = y_test[t_idx]
        next_trans = oracle_trans[k + 1] if k + 1 < len(oracle_trans) else n_test
        lat = -1
        for i in range(t_idx, next_trans):
            if y_bin[i] == target:
                lat = i - t_idx
                break
        if lat >= 0:
            latencies.append(lat)

    lat_arr = np.array(latencies) if latencies else np.array([999])
    within_6 = (lat_arr <= 6).sum() / n_oracle * 100 if n_oracle > 0 else 0

    # --- KPI 2: Plateau oscillations ---
    boundaries = [0] + list(oracle_trans) + [n_test]
    n_clean = 0
    for k in range(len(boundaries) - 1):
        seg = y_bin[boundaries[k]:boundaries[k + 1]]
        if (np.diff(seg) != 0).sum() == 0:
            n_clean += 1
    pct_clean = n_clean / (len(boundaries) - 1) * 100

    # --- KPI 3: Switch precision ---
    if n_model_sw > 0 and n_oracle > 0:
        distances = []
        for ms in model_switches:
            d = np.abs(oracle_trans.astype(int) - int(ms)).min()
            distances.append(d)
        dist_arr = np.array(distances)
        pct_spurious = (dist_arr > 20).mean() * 100
    else:
        pct_spurious = 0

    # Transition accuracy
    trans_mask = np.zeros(n_test, dtype=bool)
    trans_mask[oracle_trans] = True  # only at transition points
    # Include the step after transition too (model might detect 1 step late)
    if n_oracle > 0:
        trans_acc = (y_bin[trans_mask] == y_test[trans_mask]).mean() * 100
    else:
        trans_acc = 0

    return {
        'model': model_name,
        'indicator': indicator.upper(),
        'timeframe': timeframe,
        'val_acc': val_acc * 100,
        'n_transitions': n_oracle,
        'persistence_acc': persistence_acc * 100,
        'transition_acc': trans_acc,
        'latency_median': float(np.median(lat_arr)),
        'latency_p90': float(np.percentile(lat_arr, 90)),
        'pct_within_6': within_6,
        'ratio_switches': n_model_sw / n_oracle if n_oracle > 0 else 0,
        'pct_clean_plateaus': pct_clean,
        'pct_spurious': pct_spurious,
        'auc': auc,
        'pct_grey': grey * 100,
    }


def main():
    models = [
        ('macd', '30m'), ('cci', '30m'), ('rsi', '30m'),
        ('macd', '1h'), ('cci', '1h'), ('rsi', '1h'),
    ]

    results = []
    for ind, tf in models:
        r = analyze_model(ind, tf)
        if r:
            results.append(r)
        else:
            print(f"  SKIP {ind}_{tf}: NPZ not found")

    if not results:
        print("No models found!")
        return

    # Print table
    print(f"\n{'='*130}")
    print(f"ALL MODELS COMPARISON — Signal Quality KPIs (BTC test set)")
    print(f"{'='*130}")

    header = (f"{'Model':<12} {'Acc%':>5} {'Pers%':>6} {'Trans%':>6} {'AUC':>6} "
              f"{'N_trans':>7} {'Lat_med':>7} {'Lat_p90':>7} {'<6stp%':>6} "
              f"{'Sw_ratio':>8} {'Clean%':>6} {'Spur%':>6} {'Grey%':>5}")
    print(header)
    print('-' * 130)

    for r in results:
        line = (f"{r['model']:<12} {r['val_acc']:>5.1f} {r['persistence_acc']:>6.1f} "
                f"{r['transition_acc']:>6.1f} {r['auc']:>6.4f} "
                f"{r['n_transitions']:>7,} {r['latency_median']:>7.1f} {r['latency_p90']:>7.1f} "
                f"{r['pct_within_6']:>6.1f} "
                f"{r['ratio_switches']:>8.1f}x {r['pct_clean_plateaus']:>6.1f} "
                f"{r['pct_spurious']:>6.1f} {r['pct_grey']:>5.1f}")
        print(line)

    print('-' * 130)

    # Signal quality ranking (composite: transition_acc + within_6 - spurious - ratio)
    print(f"\n{'='*80}")
    print(f"SIGNAL QUALITY RANKING (not accuracy!)")
    print(f"{'='*80}")
    print(f"  Ranking by: transition_acc × pct_within_6 / (ratio_switches × (1 + pct_spurious/100))")
    print()

    for r in results:
        r['signal_score'] = (r['transition_acc'] * r['pct_within_6'] /
                            (r['ratio_switches'] * (1 + r['pct_spurious'] / 100)))

    ranked = sorted(results, key=lambda x: x['signal_score'], reverse=True)

    for i, r in enumerate(ranked):
        marker = ' ★' if i == 0 else ''
        print(f"  {i+1}. {r['model']:<12} score={r['signal_score']:>8.1f}  "
              f"(trans_acc={r['transition_acc']:.1f}%, within_6={r['pct_within_6']:.1f}%, "
              f"ratio={r['ratio_switches']:.1f}x, spurious={r['pct_spurious']:.1f}%){marker}")

    # Key insights
    print(f"\n{'='*80}")
    print(f"KEY INSIGHTS")
    print(f"{'='*80}")

    best = ranked[0]
    worst = ranked[-1]

    # Best by each metric
    best_trans = max(results, key=lambda x: x['transition_acc'])
    best_latency = min(results, key=lambda x: x['latency_median'])
    least_spurious = min(results, key=lambda x: x['pct_spurious'])
    cleanest = max(results, key=lambda x: x['pct_clean_plateaus'])
    best_auc = max(results, key=lambda x: x['auc'])

    print(f"  Best transition accuracy:  {best_trans['model']} ({best_trans['transition_acc']:.1f}%)")
    print(f"  Fastest detection:         {best_latency['model']} (median={best_latency['latency_median']:.1f} steps)")
    print(f"  Least spurious switches:   {least_spurious['model']} ({least_spurious['pct_spurious']:.1f}%)")
    print(f"  Cleanest plateaus:         {cleanest['model']} ({cleanest['pct_clean_plateaus']:.1f}% clean)")
    print(f"  Best AUC:                  {best_auc['model']} ({best_auc['auc']:.4f})")
    print(f"  Best overall signal:       {best['model']} (score={best['signal_score']:.1f})")

    # Does accuracy correlate with signal quality?
    acc_rank = sorted(results, key=lambda x: x['val_acc'], reverse=True)
    print(f"\n  Accuracy ranking:      {' > '.join(r['model'] for r in acc_rank)}")
    print(f"  Signal quality ranking: {' > '.join(r['model'] for r in ranked)}")
    if [r['model'] for r in acc_rank] == [r['model'] for r in ranked]:
        print(f"  → Rankings MATCH (accuracy correlates with signal quality)")
    else:
        print(f"  → Rankings DIFFER (accuracy ≠ signal quality)")

    # Save JSON
    json_path = 'models/kpi_all_models_comparison.json'
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {json_path}")


if __name__ == '__main__':
    main()
