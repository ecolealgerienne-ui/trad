#!/usr/bin/env python3
"""
Validate that flks_vs_filter.py's Kalman forward matches pykalman exactly.

Compares position and velocity arrays on the first 100 samples of BTC MACD.
Reports max absolute error. If > 1e-6, the custom implementation is wrong.

Usage:
    python src/signal_processing/validate_kalman.py --csv data_trad/BTCUSD_all_5m.csv
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from flks_vs_filter import (
    kalman_filter_forward, compute_indicator, load_csv,
    A, H, Q, R, KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR,
)


def run_pykalman_filter(z: np.ndarray):
    """Run pykalman's filter with identical parameters."""
    from pykalman import KalmanFilter as KF

    kf = KF(
        transition_matrices=[[1, 1], [0, 1]],
        observation_matrices=[[1, 0]],
        initial_state_mean=[z[0], 0.0],
        initial_state_covariance=np.eye(2),
        observation_covariance=KALMAN_MEASURE_VAR,
        transition_covariance=np.eye(2) * KALMAN_PROCESS_VAR,
    )
    state_means, state_covs = kf.filter(z)
    return state_means, state_covs


def main():
    parser = argparse.ArgumentParser(description='Validate custom Kalman vs pykalman')
    parser.add_argument('--csv', type=str, default='data_trad/BTCUSD_all_5m.csv')
    parser.add_argument('--indicator', type=str, default='macd')
    parser.add_argument('--n-samples', type=int, default=100)
    args = parser.parse_args()

    # Load data
    print(f"Loading {args.csv} ...")
    df = load_csv(args.csv)
    indicator_raw = compute_indicator(df, args.indicator)

    # Take first N valid samples
    first_valid = 0
    for i in range(len(indicator_raw)):
        if not np.isnan(indicator_raw[i]):
            first_valid = i
            break
    z = indicator_raw[first_valid:first_valid + args.n_samples].astype(np.float64)
    T = len(z)
    print(f"  Testing on {T} samples of {args.indicator.upper()}")

    # --- Run custom implementation ---
    x_custom, P_custom, _, _ = kalman_filter_forward(z)

    # --- Run pykalman ---
    x_pykalman, P_pykalman = run_pykalman_filter(z)

    # --- Compare ---
    pos_err = np.abs(x_custom[:, 0] - x_pykalman[:, 0])
    vel_err = np.abs(x_custom[:, 1] - x_pykalman[:, 1])
    cov_err = np.abs(P_custom - P_pykalman).max(axis=(1, 2))

    print(f"\n{'='*60}")
    print(f"  Kalman Forward Validation: custom vs pykalman")
    print(f"  Samples: {T} | Indicator: {args.indicator.upper()}")
    print(f"  Q={KALMAN_PROCESS_VAR}, R={KALMAN_MEASURE_VAR}")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'Max Error':>12} {'Mean Error':>12}")
    print(f"  {'-'*51}")
    print(f"  {'Position (state[0])':<25} {pos_err.max():>12.2e} {pos_err.mean():>12.2e}")
    print(f"  {'Velocity (state[1])':<25} {vel_err.max():>12.2e} {vel_err.mean():>12.2e}")
    print(f"  {'Covariance (P)':<25} {cov_err.max():>12.2e} {cov_err.mean():>12.2e}")
    print(f"  {'-'*51}")

    # Detail first 5 samples
    print(f"\n  First 5 samples — Position:")
    print(f"  {'t':>4} {'Custom':>14} {'pykalman':>14} {'Error':>12}")
    for t in range(min(5, T)):
        print(f"  {t:>4} {x_custom[t,0]:>14.6f} {x_pykalman[t,0]:>14.6f} {pos_err[t]:>12.2e}")

    print(f"\n  First 5 samples — Velocity:")
    print(f"  {'t':>4} {'Custom':>14} {'pykalman':>14} {'Error':>12}")
    for t in range(min(5, T)):
        print(f"  {t:>4} {x_custom[t,1]:>14.6f} {x_pykalman[t,1]:>14.6f} {vel_err[t]:>12.2e}")

    # Verdict
    max_state_err = max(pos_err.max(), vel_err.max())
    THRESHOLD = 1e-6
    print(f"\n  Max state error: {max_state_err:.2e}")
    if max_state_err < THRESHOLD:
        print(f"  PASS — error < {THRESHOLD:.0e}")
    else:
        print(f"  FAIL — error >= {THRESHOLD:.0e}")
        print(f"  The custom implementation does NOT match pykalman.")
        print(f"  Likely cause: t=0 initialization (pykalman does predict+update,")
        print(f"  custom skips update). This propagates to all subsequent steps.")
        sys.exit(1)


if __name__ == '__main__':
    main()
