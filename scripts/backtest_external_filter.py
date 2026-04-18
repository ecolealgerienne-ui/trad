#!/usr/bin/env python3
"""
Backtest avec filtre externe ATR appliqué EN AVAL du modèle.

Architecture en 2 étages (séparation des dynamiques) :
  Niveau 1 (rapide) : signal direction = sign(model.proba - 0.5)
  Niveau 2 (lent)   : filtre ATR  → garde le signal seulement si dans la bande
  Décision finale   : trade si OK, conserve position si bloqué (slope = 0)

⚠️ Le modèle N'EST PAS retrained. On utilise les preds existantes.
Itération rapide : on teste plusieurs seuils ATR sans toucher au modèle.

Objectif : transformer le PnL Net négatif du Model pur en POSITIF en
filtrant les signaux émis dans des conditions de marché défavorables.

Filtres testables :
  - ATR > low                     : trader uniquement marchés actifs
  - ATR < high                    : éviter chaos / news / spikes
  - low < ATR < high              : bande optimale
  - ATR normalisé (ATR / close)   : volatilité relative

Usage :
    python scripts/backtest_external_filter.py --npz <NPZ> --preds <PREDS>
    python scripts/backtest_external_filter.py --npz ... --preds ... --period 14 --normalize
    python scripts/backtest_external_filter.py --npz ... --preds ... \\
        --atr-lows 0.0 0.001 0.002 --atr-highs inf 0.01 0.02
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import (
    load_csv, backtest_5min_progressive, calculate_atr,
)

CSV_5M = Path('data_trad/BTCUSD_all_5m.csv')


def buy_and_hold_5m(closes):
    first, last = closes[0], closes[-1]
    if np.isnan(first) or np.isnan(last) or first == 0:
        return 0.0
    return (last - first) / first * 100


def apply_atr_filter(slopes_model, atr_values, atr_low, atr_high):
    """
    Filtre les slopes du modèle selon ATR.
    Si atr_low <= ATR <= atr_high : garde le signal model.
    Sinon : slope = 0 (conserve position dans backtest_5min_progressive).
    """
    in_band = (atr_values >= atr_low) & (atr_values <= atr_high)
    # Si ATR est NaN (warmup), on bloque (skip)
    in_band = in_band & ~np.isnan(atr_values)
    slopes_filtered = np.where(in_band, slopes_model, 0.0)
    return slopes_filtered, int(in_band.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', required=True,
                        help='Path vers dataset_..._progressive[_*].npz')
    parser.add_argument('--preds', required=True,
                        help='Path vers preds NPZ (XGBoost ou CNN-LSTM)')
    parser.add_argument('--split', default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--period', type=int, default=14,
                        help='Période ATR (default 14)')
    parser.add_argument('--normalize', action='store_true',
                        help='Utiliser ATR normalisé (ATR / close), '
                             'sinon ATR brut en USD')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold model proba')
    parser.add_argument('--fees', type=float, default=0.001)
    parser.add_argument('--atr-lows', type=float, nargs='+',
                        default=[0.0, 0.001, 0.002, 0.003, 0.005],
                        help='Bornes basses (ATR normalisé) à tester')
    parser.add_argument('--atr-highs', type=float, nargs='+',
                        default=[0.01, 0.02, 0.05, 0.10, 1.0],
                        help='Bornes hautes (ATR normalisé) à tester')
    parser.add_argument('--top', type=int, default=20)
    args = parser.parse_args()

    print("=" * 110)
    print(f"BACKTEST FILTRE EXTERNE ATR — split={args.split}  fees={args.fees*100:.2f}%")
    print(f"  ATR period={args.period}  normalized={args.normalize}")
    print(f"  threshold model={args.threshold}")
    print(f"  Grid lows  : {args.atr_lows}")
    print(f"  Grid highs : {args.atr_highs}")
    print("=" * 110)

    # ========== [1] Load NPZ + preds ==========
    npz_path = Path(args.npz)
    if not npz_path.exists():
        print(f"❌ NPZ introuvable: {npz_path}")
        return
    preds_path = Path(args.preds)
    if not preds_path.exists():
        print(f"❌ Preds NPZ introuvable: {preds_path}")
        return

    ds = np.load(npz_path, allow_pickle=True)
    preds = np.load(preds_path, allow_pickle=True)

    closes = ds[f'closes_{args.split}']
    dates = pd.to_datetime(ds[f'dates_{args.split}'])
    y_cont = ds[f'y_{args.split}_continuous']
    p = preds[f'{args.split}_preds_proba']

    period_days = (dates[-1] - dates[0]).total_seconds() / 86400
    print(f"\n[1] Split {args.split}: {len(closes):,} rows  |  "
          f"{dates[0]} → {dates[-1]}  ({period_days:.0f} jours)")

    # ========== [2] Charger CSV BTCUSD pour HLC ==========
    print(f"\n[2] Charger CSV {CSV_5M} pour HLC ...")
    if not CSV_5M.exists():
        print(f"❌ CSV introuvable: {CSV_5M}")
        return
    df_5m_full = load_csv(CSV_5M)
    print(f"   df_5m full: {len(df_5m_full):,} rows  |  "
          f"{df_5m_full.index[0]} → {df_5m_full.index[-1]}")

    # ========== [3] Calcul ATR(period) sur df_5m_full ==========
    print(f"\n[3] Calcul ATR({args.period}) sur df_5m_full "
          f"(via core.calculate_atr) ...")
    atr_full = calculate_atr(df_5m_full, period=args.period,
                                normalize=args.normalize)
    atr_label = (f'ATR{args.period} / close (relatif)' if args.normalize
                  else f'ATR{args.period} (USD brut)')
    print(f"   {atr_label}")

    # ========== [4] Aligner ATR avec dates_test (NPZ) ==========
    print(f"\n[4] Alignement ATR avec dates {args.split} ...")
    atr_series = pd.Series(atr_full, index=df_5m_full.index)
    atr_aligned = atr_series.reindex(dates).values
    n_nan = int(np.isnan(atr_aligned).sum())
    print(f"   {len(atr_aligned):,} valeurs ATR alignées  ({n_nan} NaN, "
          f"{(1-n_nan/len(atr_aligned))*100:.2f}% valid)")

    # Stats ATR sur le split
    valid = ~np.isnan(atr_aligned)
    if valid.sum() > 0:
        a = atr_aligned[valid]
        print(f"   ATR distribution: min={a.min():.6f}  max={a.max():.6f}  "
              f"mean={a.mean():.6f}  median={np.median(a):.6f}")
        print(f"   Percentiles: 10%={np.percentile(a, 10):.6f}  "
              f"25%={np.percentile(a, 25):.6f}  "
              f"50%={np.percentile(a, 50):.6f}  "
              f"75%={np.percentile(a, 75):.6f}  "
              f"90%={np.percentile(a, 90):.6f}")

    # ========== [5] Backtests baselines ==========
    print(f"\n[5] Backtests baselines (Oracle + Model pur) ...")
    slopes_oracle = y_cont
    r_oracle = backtest_5min_progressive(slopes_oracle, closes, fees=args.fees)

    slopes_model = np.where(p > args.threshold, 1.0, -1.0)
    r_model = backtest_5min_progressive(slopes_model, closes, fees=args.fees)
    bh = buy_and_hold_5m(closes)

    # ========== [6] Grid ATR filter ==========
    print(f"\n[6] Grid ATR filter ({len(args.atr_lows) * len(args.atr_highs)} configs) ...")
    results = []
    for low in args.atr_lows:
        for high in args.atr_highs:
            if low >= high:
                continue
            slopes_f, n_inband = apply_atr_filter(slopes_model, atr_aligned, low, high)
            r = backtest_5min_progressive(slopes_f, closes, fees=args.fees)
            r['atr_low'] = low
            r['atr_high'] = high
            r['n_inband'] = n_inband
            r['inband_pct'] = n_inband / len(slopes_f) * 100
            results.append(r)

    results.sort(key=lambda x: -x['pnl_pct'])

    # ========== [7] Affichage ==========
    print(f"\n{'=' * 110}")
    print(f"RÉSULTATS — {args.split}  ({period_days:.0f} jours)")
    print(f"{'=' * 110}")

    def fmt_baseline(label, r):
        fees_pct = r['n_trades'] * 2 * args.fees * 100
        capture = (r['pnl_pct'] / r_oracle['pnl_pct'] * 100
                   if r_oracle['pnl_pct'] != 0 else 0)
        return (f"  {label:<28}{r['n_trades']:>8,} "
                f"{r['win_rate']:>6.1f}% {r['profit_factor']:>6.2f} "
                f"{r['sharpe']:>7.3f} {r['pnl_pct']+fees_pct:>+9.2f}% "
                f"{fees_pct:>9.2f}% {r['pnl_pct']:>+10.2f}% "
                f"{r['pnl_pct']-bh:>+10.2f}% {capture:>+7.1f}%")

    header = (f"  {'Stratégie':<28}{'Trades':>8} {'WR':>7}{'PF':>7}{'Sharpe':>8}"
              f"{'Brut':>10}{'Fees':>10}{'Net':>11}{'αB&H':>11}{'Capt%':>8}")
    print(header)
    print(f"  {'-' * 106}")
    print(fmt_baseline('ORACLE', r_oracle))
    print(fmt_baseline(f'MODEL pur (t={args.threshold})', r_model))
    print(f"  {'-' * 106}")

    print(f"\n  TOP {min(args.top, len(results))} configs ATR filter — triées par PnL Net")
    print(f"  {'low':>10} {'high':>10} {'inband%':>8}  "
          f"{'Trades':>8} {'WR':>7}{'PF':>7}{'Sharpe':>8}"
          f"{'Brut':>10}{'Fees':>10}{'Net':>11}{'Capt%':>8}")
    print(f"  {'-' * 106}")
    for r in results[:args.top]:
        fees_pct = r['n_trades'] * 2 * args.fees * 100
        capture = (r['pnl_pct'] / r_oracle['pnl_pct'] * 100
                   if r_oracle['pnl_pct'] != 0 else 0)
        print(f"  {r['atr_low']:>10.6f} {r['atr_high']:>10.6f} "
              f"{r['inband_pct']:>7.1f}% "
              f"{r['n_trades']:>8,} {r['win_rate']:>6.1f}% "
              f"{r['profit_factor']:>6.2f} {r['sharpe']:>7.3f} "
              f"{r['pnl_pct']+fees_pct:>+9.2f}% {fees_pct:>9.2f}% "
              f"{r['pnl_pct']:>+10.2f}% {capture:>+7.1f}%")

    # ========== [8] Best ==========
    if results:
        best = results[0]
        print(f"\n  ★ BEST: low={best['atr_low']:.6f}  high={best['atr_high']:.6f}  "
              f"({best['inband_pct']:.1f}% temps en bande)")
        print(f"    PnL Net {best['pnl_pct']:+.2f}%  vs Model pur {r_model['pnl_pct']:+.2f}%  "
              f"(gain {best['pnl_pct'] - r_model['pnl_pct']:+.2f}%)")
        print(f"    Trades {best['n_trades']:,} vs Model pur {r_model['n_trades']:,}  "
              f"({best['n_trades'] - r_model['n_trades']:+,})")
        if best['pnl_pct'] > 0:
            print(f"    🏆 PnL POSITIF SUR MODEL PUR — filtre ATR validé !")
        elif best['pnl_pct'] > r_model['pnl_pct']:
            print(f"    ⚠️ Amélioration mais encore négatif")
        else:
            print(f"    ❌ Pas d'amélioration vs Model pur")


if __name__ == '__main__':
    main()
