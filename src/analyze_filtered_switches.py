#!/usr/bin/env python3
"""
Filtered Switches Analysis — seuils sur velocity et macd_live
=============================================================

Applique des filtres sur les prédictions du modèle pour réduire
les faux switches. Le modèle ne switch que si :
  |velocity| > vel_threshold  ET  |macd_live| > macd_threshold

Teste un grid de seuils et compare : ratio switchs, justified%, spurious%.

Usage:
    python src/analyze_filtered_switches.py --indicator macd --timeframe 30m
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from constants import PREPARED_DATA_DIR

ASSET_CSV_MAP = {'BTC': 'BTCUSD'}
NEAR_THRESHOLD = 6
FAR_THRESHOLD = 20


def find_switches(labels):
    switches = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            switches.append(i)
    return switches


def classify_switches(model_switch_indices, oracle_switch_indices):
    """Returns (n_justified, n_spurious, n_total)."""
    if len(oracle_switch_indices) == 0:
        return 0, len(model_switch_indices), len(model_switch_indices)

    oracle_arr = np.array(oracle_switch_indices)
    n_justified = 0
    n_spurious = 0
    for m_idx in model_switch_indices:
        min_dist = np.min(np.abs(oracle_arr - m_idx))
        if min_dist <= NEAR_THRESHOLD:
            n_justified += 1
        elif min_dist > FAR_THRESHOLD:
            n_spurious += 1
    return n_justified, n_spurious, len(model_switch_indices)


def apply_filter(y_pred_proba, vel_values, macd_values, pred_threshold,
                 vel_thr, macd_thr):
    """
    Apply switch filter: model switches only if conditions are met.
    Otherwise, keep previous prediction.

    Returns filtered binary predictions.
    """
    n = len(y_pred_proba)
    y_raw = (y_pred_proba > pred_threshold).astype(int)
    y_filtered = np.copy(y_raw)

    for i in range(1, n):
        if y_raw[i] != y_filtered[i - 1]:
            # Model wants to switch — check conditions
            vel_ok = abs(vel_values[i]) > vel_thr if vel_thr > 0 else True
            macd_ok = abs(macd_values[i]) > macd_thr if macd_thr > 0 else True

            if vel_ok and macd_ok:
                y_filtered[i] = y_raw[i]  # allow switch
            else:
                y_filtered[i] = y_filtered[i - 1]  # block switch, keep previous
        else:
            y_filtered[i] = y_raw[i]

    return y_filtered


def main():
    parser = argparse.ArgumentParser(description='Filtered switches analysis')
    parser.add_argument('--indicator', default='macd')
    parser.add_argument('--timeframe', default='30m')
    parser.add_argument('--pred-threshold', type=float, default=0.5)
    args = parser.parse_args()

    # Load NPZ
    npz_path = f'{PREPARED_DATA_DIR}/{args.indicator}_{args.timeframe}_dataset.npz'
    data = np.load(npz_path, allow_pickle=True)
    if 'y_test' in data:
        y_test = data['y_test']
        y_pred_proba = data['y_test_pred']
    else:
        y_test = data['test_labels']
        y_pred_proba = data['test_preds']

    # Load CSV for features
    base = ASSET_CSV_MAP['BTC']
    csv_path = f'{PREPARED_DATA_DIR}/{base}_multitf_macd_rsi_cci.csv'
    if not Path(csv_path).exists():
        logger.error(f"CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path, parse_dates=['datetime']).set_index('datetime').sort_index()

    ind = args.indicator
    tf = args.timeframe
    vel_col = f'{ind}_{tf}_velocity'
    live_col = f'{ind}_{tf}_live'

    # Get test portion features (aligned with predictions)
    # Same alignment as training: test = last portion
    n_test = len(y_test)
    df_clean = df.dropna(subset=[vel_col, live_col])
    vel_all = df_clean[vel_col].values
    macd_all = df_clean[live_col].values

    # Test = last n_test samples (after sequence offset = 25-1=24)
    # The predictions correspond to the label at the LAST step of each 25-step window
    # So prediction[i] corresponds to roughly df row [val_end + 24 + i]
    vel_test = vel_all[-n_test:]
    macd_test = macd_all[-n_test:]

    # Oracle switches
    oracle_switches = find_switches(y_test)
    n_oracle = len(oracle_switches)

    # Baseline (no filter)
    y_baseline = (y_pred_proba > args.pred_threshold).astype(int)
    baseline_switches = find_switches(y_baseline)
    b_just, b_spur, b_total = classify_switches(baseline_switches, oracle_switches)

    print(f"\n{'=' * 85}")
    print(f"  FILTERED SWITCHES — {ind.upper()}_{tf}")
    print(f"  Test samples: {n_test:,}  |  Oracle switches: {n_oracle:,}")
    print(f"{'=' * 85}")
    print(f"\n  Baseline (no filter): {b_total:,} switches, "
          f"ratio={b_total/n_oracle:.1f}×, "
          f"justified={b_just/b_total*100:.1f}%, "
          f"spurious={b_spur/b_total*100:.1f}%")

    # Compute percentiles for threshold calibration
    vel_abs = np.abs(vel_test[~np.isnan(vel_test)])
    macd_abs = np.abs(macd_test[~np.isnan(macd_test)])

    vel_p50 = np.percentile(vel_abs, 50)
    vel_p75 = np.percentile(vel_abs, 75)
    macd_p25 = np.percentile(macd_abs, 25)
    macd_p50 = np.percentile(macd_abs, 50)

    print(f"\n  Feature percentiles:")
    print(f"    |velocity|: P50={vel_p50:.2f}, P75={vel_p75:.2f}")
    print(f"    |macd_live|: P25={macd_p25:.2f}, P50={macd_p50:.2f}")

    # Grid search
    vel_thresholds = [0, round(vel_p50 * 0.5, 2), round(vel_p50, 2), round(vel_p75, 2)]
    macd_thresholds = [0, round(macd_p25 * 0.5, 2), round(macd_p25, 2), round(macd_p50, 2)]

    print(f"\n  Grid: vel_thr={vel_thresholds} × macd_thr={macd_thresholds}")
    print(f"\n  {'vel_thr':>8} {'macd_thr':>9} │ {'Switches':>9} {'Ratio':>6} "
          f"{'Justified':>10} {'Spurious':>9} {'Detected':>9}")
    print(f"  {'-' * 75}")

    best_score = -1
    best_cfg = ""

    for v_thr in vel_thresholds:
        for m_thr in macd_thresholds:
            y_filt = apply_filter(y_pred_proba, vel_test, macd_test,
                                  args.pred_threshold, v_thr, m_thr)
            switches = find_switches(y_filt)
            n_sw = len(switches)
            if n_sw == 0:
                continue

            n_just, n_spur, _ = classify_switches(switches, oracle_switches)
            ratio = n_sw / n_oracle
            pct_just = n_just / n_sw * 100
            pct_spur = n_spur / n_sw * 100

            # Detection: how many oracle transitions have a model switch nearby?
            oracle_arr = np.array(oracle_switches)
            sw_arr = np.array(switches)
            n_detected = 0
            for o_idx in oracle_switches:
                if len(sw_arr) > 0 and np.min(np.abs(sw_arr - o_idx)) <= NEAR_THRESHOLD:
                    n_detected += 1
            pct_detected = n_detected / n_oracle * 100

            marker = ""
            # Score: maximize justified%, minimize ratio, keep detection >80%
            if pct_detected > 80:
                score = pct_just / ratio
                if score > best_score:
                    best_score = score
                    best_cfg = f"vel={v_thr}, macd={m_thr}"
                    marker = " ← BEST"

            print(f"  {v_thr:>8.2f} {m_thr:>9.2f} │ {n_sw:>9,} {ratio:>5.1f}× "
                  f"{pct_just:>9.1f}% {pct_spur:>8.1f}% {pct_detected:>8.1f}%{marker}")

    print(f"  {'-' * 75}")
    if best_cfg:
        print(f"\n  BEST: {best_cfg} (score={best_score:.1f})")
    print(f"{'=' * 85}")
    print("Done.")


if __name__ == '__main__':
    main()
