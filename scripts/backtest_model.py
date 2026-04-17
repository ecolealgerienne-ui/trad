#!/usr/bin/env python3
"""
Backtest d'un modèle entraîné (depuis le NPZ produit par train_model.py).

Pipeline:
  1. Charge NPZ prédictions + CSV 5m + CSV TF téléchargé
  2. Reconstruit slopes_from_preds alignées sur df_tf.index
     (0 pour bougies avant test ou hors prédictions, ±1 selon threshold)
  3. Reconstruit slopes_oracle via compute_oracle_labels
  4. backtest_5m sur:
       - Modèle (slopes_from_preds)
       - Oracle (slopes_oracle)
       - Buy & Hold

Le threshold est paramétrable (--threshold) → on peut tester plusieurs seuils
sans retrain.

Usage:
    python scripts/backtest_model.py
    python scripts/backtest_model.py --threshold 0.6
    python scripts/backtest_model.py --indicator rsi --tf 60
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import (
    load_csv, group_per_candle,
    buy_and_hold,
)

DATA_DIR = Path('data/raw')
PREP_DIR = Path('data/prepared')


SOURCE_PATHS = {
    '3months': {
        5: DATA_DIR / 'BTCUSD_3months_5m.csv',
        30: DATA_DIR / 'BTCUSD_3months_30m.csv',
        60: DATA_DIR / 'BTCUSD_3months_1h.csv',
    },
    'full': {
        5: Path('data_trad/BTCUSD_all_5m.csv'),
        30: DATA_DIR / 'BTCUSD_full_30m.csv',
    },
}


def drop_incomplete_last(df_tf, df_5m, tf_minutes):
    expected = tf_minutes // 5
    drop_count = 0
    for ts in reversed(df_tf.index):
        end = ts + pd.Timedelta(minutes=tf_minutes)
        mask = (df_5m.index >= ts) & (df_5m.index < end)
        if mask.sum() < expected:
            drop_count += 1
        else:
            break
    if drop_count > 0:
        df_tf = df_tf.iloc[:-drop_count]
    return df_tf, drop_count


def compute_stats(slopes, closes_5m_per_candle, k_substep, start, end, fees,
                    holding_min=0):
    """Backtest manuel avec stats détaillées. Threshold déjà appliqué en amont."""
    trades = []
    position = 0
    entry_price = 0.0
    entry_t = -holding_min
    for t in range(start, end):
        if np.isnan(slopes[t]) or slopes[t] == 0:
            continue
        target = 1 if slopes[t] > 0 else -1
        if position == target:
            continue
        if position != 0 and (t - entry_t) < holding_min:
            continue
        if t + 1 >= len(closes_5m_per_candle):
            continue
        closes_5m = closes_5m_per_candle[t + 1]
        step_idx = k_substep - 1
        if step_idx >= len(closes_5m):
            continue
        exec_price = closes_5m[step_idx]
        if np.isnan(exec_price):
            continue
        if position != 0:
            pnl = (exec_price - entry_price) / entry_price if position == 1 \
                  else (entry_price - exec_price) / entry_price
            pnl -= 2 * fees
            trades.append({'entry_t': entry_t, 'exit_t': t,
                           'duration': t - entry_t, 'pnl': pnl,
                           'position': position})
        entry_price = exec_price
        position = target
        entry_t = t
    # Close last
    if position != 0:
        last_candle = min(end, len(closes_5m_per_candle) - 1)
        closes_last = closes_5m_per_candle[last_candle]
        if len(closes_last) > 0 and not np.isnan(closes_last[-1]):
            exit_price = closes_last[-1]
            pnl = (exit_price - entry_price) / entry_price if position == 1 \
                  else (entry_price - exit_price) / entry_price
            pnl -= 2 * fees
            trades.append({'entry_t': entry_t, 'exit_t': last_candle,
                           'duration': last_candle - entry_t,
                           'pnl': pnl, 'position': position})
    if not trades:
        return dict(n_trades=0, pnl_pct=0.0, win_rate=0.0, profit_factor=0.0,
                    sharpe=0.0, avg_duration=0.0, n_long=0, n_short=0)
    pnls = np.array([t['pnl'] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    pnl_total = pnls.sum() * 100
    wr = len(wins) / len(pnls) * 100
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else np.inf
    mu, sd = pnls.mean(), pnls.std()
    sharpe = mu / sd if sd > 1e-10 else 0.0
    avg_dur = np.mean([t['duration'] for t in trades])
    n_long = sum(1 for t in trades if t['position'] == 1)
    n_short = sum(1 for t in trades if t['position'] == -1)
    return dict(n_trades=len(trades), pnl_pct=pnl_total, win_rate=wr,
                profit_factor=pf, sharpe=sharpe, avg_duration=avg_dur,
                n_long=n_long, n_short=n_short)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indicator', default='macd',
                        choices=['macd', 'rsi', 'cci'])
    parser.add_argument('--tf', type=int, default=30, choices=[30, 60])
    parser.add_argument('--source', default='3months',
                        choices=['3months', 'full'])
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold pour binariser pred_proba (default 0.5)')
    parser.add_argument('--k', type=int, default=6,
                        help='k substep pour backtest_5m (default 6)')
    parser.add_argument('--fees', type=float, default=0.001)
    parser.add_argument('--holding-min', type=int, default=0,
                        help='Min bougies entre 2 trades (réduit overtrading)')
    args = parser.parse_args()

    tf_label = f'{args.tf}m' if args.tf < 60 else '1h'
    print("=" * 80)
    print(f"BACKTEST modèle — {args.indicator.upper()} × {tf_label}  "
          f"source={args.source}  "
          f"(threshold={args.threshold}, fees={args.fees*100:.2f}%, "
          f"holding_min={args.holding_min})")
    print("=" * 80)

    # [1] Charger NPZ preds + dataset (contient slopes_oracle précalculées)
    preds_path = PREP_DIR / f'preds_{args.indicator}_{tf_label}_{args.source}.npz'
    dataset_path = PREP_DIR / f'dataset_{args.indicator}_{tf_label}_{args.source}.npz'
    if not preds_path.exists():
        print(f"❌ NPZ predictions non trouvé: {preds_path}")
        print(f"   Lance d'abord: python scripts/train_model.py "
              f"--indicator {args.indicator} --tf {args.tf} --source {args.source}")
        return
    if not dataset_path.exists():
        print(f"❌ NPZ dataset non trouvé: {dataset_path}")
        return

    preds = np.load(preds_path, allow_pickle=True)
    ds = np.load(dataset_path, allow_pickle=True)
    test_preds_proba = preds['test_preds_proba']
    test_y_true = preds['test_y_true']
    test_indices = preds['test_indices']
    print(f"\n✅ NPZ preds:   {preds_path}")
    print(f"✅ NPZ dataset: {dataset_path}")
    print(f"   {len(test_preds_proba):,} predictions sur test set")

    # Classification metrics pour info
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    y_pred_bin = (test_preds_proba > args.threshold).astype(int)
    acc = accuracy_score(test_y_true, y_pred_bin)
    f1 = f1_score(test_y_true, y_pred_bin)
    auc = roc_auc_score(test_y_true, test_preds_proba)
    print(f"\nClassification test: acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  "
          f"(threshold={args.threshold})")

    # [2] slopes_oracle + df_tf_closes : directement depuis le dataset NPZ
    slopes_oracle = ds['oracle_slopes_full']
    df_tf_dates = pd.to_datetime(ds['df_tf_dates'])
    df_tf_closes = ds['df_tf_closes']
    n_tf = len(df_tf_dates)
    print(f"\n✅ Oracle slopes + df_tf chargés depuis dataset  ({n_tf:,} bougies)")

    # Reconstruire df_tf DataFrame minimal pour group_per_candle
    df_tf = pd.DataFrame({'close': df_tf_closes}, index=pd.DatetimeIndex(df_tf_dates))

    # [3] Reconstruire slopes_from_preds aligné sur df_tf.index
    slopes_model = np.zeros(n_tf)
    for i, idx_tf in enumerate(test_indices):
        if idx_tf < n_tf:
            slopes_model[idx_tf] = 1.0 if test_preds_proba[i] > args.threshold else -1.0

    start = int(test_indices.min())
    end = int(test_indices.max()) + 1
    n_backtest = end - start
    print(f"\nBacktest range: [{start}, {end}) = {n_backtest:,} bougies TF "
          f"({test_dates_min_max(preds)})")

    # [4] closes_5m_per_candle (group_per_candle vectorisé maintenant)
    print(f"\nLoad 5m + group_per_candle ...")
    paths = SOURCE_PATHS[args.source]
    df_5m = load_csv(paths[5])
    closes_5m_per_candle = group_per_candle(df_5m, df_tf, df_5m['close'].values)
    print(f"   5m: {len(df_5m):,}  |  closes_5m_per_candle: {len(closes_5m_per_candle):,}")

    # [5] Backtests
    res_model = compute_stats(slopes_model, closes_5m_per_candle, args.k,
                                start, end, args.fees, args.holding_min)
    res_oracle = compute_stats(slopes_oracle, closes_5m_per_candle, args.k,
                                 start, end, args.fees, args.holding_min)
    bh_pnl = buy_and_hold(df_tf['close'].values, start, end)

    # Affichage
    print(f"\n{'Stratégie':<25} {'PnL %':>10} {'Trades':>8} {'WR %':>8} "
          f"{'PF':>7} {'Sharpe':>8} {'AvgDur':>8} {'L/S':>10}")
    print("-" * 95)

    def print_row(name, r):
        print(f"{name:<25} {r['pnl_pct']:>+10.2f} {r['n_trades']:>8} "
              f"{r['win_rate']:>7.1f}% {r['profit_factor']:>7.2f} "
              f"{r['sharpe']:>8.3f} {r['avg_duration']:>8.1f} "
              f"{r['n_long']}/{r['n_short']:<6}")

    print_row("Oracle (slope)", res_oracle)
    print_row(f"Model (thr={args.threshold})", res_model)
    print(f"{'Buy & Hold':<25} {bh_pnl:>+10.2f}")
    print("-" * 95)

    if abs(res_oracle['pnl_pct']) > 1e-6:
        capture = res_model['pnl_pct'] / res_oracle['pnl_pct'] * 100
        print(f"\nCapture ratio (Model / Oracle): {capture:.1f}%")
    alpha = res_model['pnl_pct'] - bh_pnl
    print(f"Alpha vs Buy & Hold: {alpha:+.2f}%")
    fees_tot = res_model['n_trades'] * 2 * args.fees * 100
    print(f"Fees totaux: {fees_tot:.2f}% ({res_model['n_trades']} trades × "
          f"2 × {args.fees*100:.2f}%)")


def test_dates_min_max(npz):
    if 'test_dates' in npz:
        d = npz['test_dates']
        return f"{pd.Timestamp(d[0])} → {pd.Timestamp(d[-1])}"
    return ""


if __name__ == '__main__':
    main()
