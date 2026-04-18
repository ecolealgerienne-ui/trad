#!/usr/bin/env python3
"""
Extrait la liste détaillée des trades Oracle (et optionnellement Model)
depuis un NPZ progressif. Sauvegarde en CSV pour analyse downstream
(meta-classifier, pattern mining, etc.).

Pour chaque trade :
  - entry_dt, exit_dt   : datetimes 5min
  - entry_price, exit_price
  - position            : LONG (+1) ou SHORT (-1)
  - pnl_brut, pnl_net   : avant/après fees
  - duration_5m         : durée en rows 5min
  - duration_min        : durée en minutes
  - is_winner           : True si pnl_net > 0

Stats descriptives :
  - Distribution PnL (mean, std, percentiles)
  - Distribution durée
  - Win rate
  - Profit factor
  - Décomposition LONG/SHORT
  - Distribution des trades dans le temps (par mois, par heure)

Usage :
    python scripts/extract_oracle_trades.py --npz <NPZ> --split test
    python scripts/extract_oracle_trades.py --npz <NPZ> --preds <PREDS> --split test
        # → extrait Oracle ET Model trades
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import extract_trades_5min_progressive


def trades_to_df(trades, dates_5m):
    """Convertit list[dict] en DataFrame avec datetimes."""
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    df['entry_dt'] = pd.to_datetime(dates_5m[df['entry_i'].values])
    df['exit_dt'] = pd.to_datetime(dates_5m[df['exit_i'].values])
    df['duration_min'] = df['duration_5m'] * 5
    df['is_winner'] = df['pnl_net'] > 0
    df['position_label'] = df['position'].map({1: 'LONG', -1: 'SHORT'})
    cols = ['entry_dt', 'exit_dt', 'entry_i', 'exit_i', 'position', 'position_label',
            'entry_price', 'exit_price', 'pnl_brut', 'pnl_net',
            'duration_5m', 'duration_min', 'is_winner']
    return df[cols]


def print_stats(label, df, fees):
    if df.empty:
        print(f"\n  {label}: aucun trade extrait")
        return
    print(f"\n{'=' * 100}")
    print(f"  {label}  —  {len(df):,} trades")
    print(f"{'=' * 100}")

    # Périodicité
    period_days = (df['exit_dt'].max() - df['entry_dt'].min()).total_seconds() / 86400
    print(f"  Période          : {df['entry_dt'].min()} → {df['exit_dt'].max()}  "
          f"({period_days:.0f} jours)")
    print(f"  Trades/jour      : {len(df) / period_days:.2f}")

    # PnL
    pnl_brut_total = df['pnl_brut'].sum() * 100
    pnl_net_total = df['pnl_net'].sum() * 100
    fees_total = len(df) * 2 * fees * 100
    print(f"  PnL Brut total   : {pnl_brut_total:+.2f}%")
    print(f"  Fees total       : {fees_total:.2f}%  ({len(df)} trades × 2 × {fees*100:.2f}%)")
    print(f"  PnL Net total    : {pnl_net_total:+.2f}%")

    # Stats par trade
    print(f"\n  PnL Net par trade :")
    print(f"    Mean      : {df['pnl_net'].mean()*100:+.4f}%")
    print(f"    Std       : {df['pnl_net'].std()*100:.4f}%")
    print(f"    Min       : {df['pnl_net'].min()*100:+.4f}%")
    p10, p25, p50, p75, p90 = np.percentile(df['pnl_net'].values * 100,
                                              [10, 25, 50, 75, 90])
    print(f"    P10/P25   : {p10:+.4f}% / {p25:+.4f}%")
    print(f"    P50 (med) : {p50:+.4f}%")
    print(f"    P75/P90   : {p75:+.4f}% / {p90:+.4f}%")
    print(f"    Max       : {df['pnl_net'].max()*100:+.4f}%")

    # Win rate
    n_win = int(df['is_winner'].sum())
    wr = n_win / len(df) * 100
    print(f"\n  Win rate         : {wr:.2f}%  ({n_win:,} winners, {len(df)-n_win:,} losers)")

    # Profit factor
    pos_pnl = df.loc[df['pnl_net'] > 0, 'pnl_net'].sum()
    neg_pnl = df.loc[df['pnl_net'] < 0, 'pnl_net'].sum()
    pf = (pos_pnl / abs(neg_pnl)) if neg_pnl != 0 else np.inf
    print(f"  Profit factor    : {pf:.3f}")

    # LONG vs SHORT
    n_long = int((df['position'] == 1).sum())
    n_short = int((df['position'] == -1).sum())
    long_pnl = df.loc[df['position'] == 1, 'pnl_net'].sum() * 100
    short_pnl = df.loc[df['position'] == -1, 'pnl_net'].sum() * 100
    print(f"\n  LONG  : {n_long:,} trades, PnL Net {long_pnl:+.2f}%")
    print(f"  SHORT : {n_short:,} trades, PnL Net {short_pnl:+.2f}%")

    # Durée
    print(f"\n  Durée :")
    print(f"    Mean      : {df['duration_min'].mean():.1f} min  "
          f"({df['duration_5m'].mean():.2f} rows 5min)")
    print(f"    Median    : {df['duration_min'].median():.0f} min")
    p25, p75 = np.percentile(df['duration_min'].values, [25, 75])
    print(f"    P25/P75   : {p25:.0f} / {p75:.0f} min")
    print(f"    Min/Max   : {df['duration_min'].min():.0f} / "
          f"{df['duration_min'].max():.0f} min")

    # Distribution mensuelle
    print(f"\n  Distribution mensuelle :")
    df_m = df.copy()
    df_m['month'] = df_m['entry_dt'].dt.to_period('M')
    monthly = df_m.groupby('month').agg(
        n_trades=('pnl_net', 'count'),
        pnl_net_pct=('pnl_net', lambda x: x.sum() * 100),
        wr_pct=('is_winner', lambda x: x.mean() * 100),
    )
    for m, row in monthly.iterrows():
        bar = '█' * int(row['n_trades'] / monthly['n_trades'].max() * 30)
        print(f"    {m}  : {int(row['n_trades']):>4} trades  "
              f"PnL Net {row['pnl_net_pct']:+8.2f}%  "
              f"WR {row['wr_pct']:5.1f}%  {bar}")

    # Distribution heure (UTC)
    print(f"\n  Distribution par heure (UTC) :")
    df_h = df.copy()
    df_h['hour'] = df_h['entry_dt'].dt.hour
    hourly = df_h.groupby('hour').agg(
        n_trades=('pnl_net', 'count'),
        pnl_net_pct=('pnl_net', lambda x: x.sum() * 100),
        wr_pct=('is_winner', lambda x: x.mean() * 100),
    )
    for h, row in hourly.iterrows():
        bar = '█' * int(row['n_trades'] / hourly['n_trades'].max() * 30)
        print(f"    {int(h):02d}h UTC : {int(row['n_trades']):>4} trades  "
              f"PnL Net {row['pnl_net_pct']:+8.2f}%  "
              f"WR {row['wr_pct']:5.1f}%  {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', required=True,
                        help='Path NPZ progressif')
    parser.add_argument('--preds', default=None,
                        help='Path preds NPZ (optionnel, ajoute Model trades)')
    parser.add_argument('--split', default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold proba pour Model (default 0.5)')
    parser.add_argument('--fees', type=float, default=0.001)
    parser.add_argument('--output-dir', default='results/trades',
                        help='Dossier de sortie pour CSV')
    args = parser.parse_args()

    print("=" * 100)
    print(f"EXTRACT TRADES — split={args.split}  fees={args.fees*100:.2f}%")
    print("=" * 100)

    # Load NPZ
    npz_path = Path(args.npz)
    if not npz_path.exists():
        print(f"❌ NPZ introuvable: {npz_path}")
        return
    ds = np.load(npz_path, allow_pickle=True)
    print(f"\n✅ NPZ chargé: {npz_path}")

    closes = ds[f'closes_{args.split}']
    dates_5m = ds[f'dates_{args.split}']
    y_cont = ds[f'y_{args.split}_continuous']
    print(f"   {len(closes):,} rows  |  "
          f"{pd.Timestamp(dates_5m[0])} → {pd.Timestamp(dates_5m[-1])}")

    # Output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tag du dataset (extrait du nom NPZ)
    tag = npz_path.stem.replace('dataset_', '')

    # Extract Oracle trades
    print(f"\n→ Extract Oracle trades (signal = y_{args.split}_continuous)")
    oracle_trades = extract_trades_5min_progressive(y_cont, closes, fees=args.fees)
    df_oracle = trades_to_df(oracle_trades, dates_5m)
    out_oracle = out_dir / f'trades_oracle_{tag}_{args.split}.csv'
    df_oracle.to_csv(out_oracle, index=False)
    print(f"   Sauvegardé: {out_oracle}  ({out_oracle.stat().st_size/1024:.1f} KB)")
    print_stats(f'ORACLE TRADES ({tag}, {args.split})', df_oracle, args.fees)

    # Extract Model trades (optionnel)
    if args.preds:
        preds_path = Path(args.preds)
        if not preds_path.exists():
            print(f"\n❌ Preds NPZ introuvable: {preds_path}")
            return
        preds = np.load(preds_path, allow_pickle=True)
        p = preds[f'{args.split}_preds_proba']
        slopes_model = np.where(p > args.threshold, 1.0, -1.0)

        print(f"\n→ Extract Model trades (threshold={args.threshold})")
        model_trades = extract_trades_5min_progressive(slopes_model, closes,
                                                          fees=args.fees)
        df_model = trades_to_df(model_trades, dates_5m)
        model_tag = preds_path.stem.replace('preds_', '').replace('_progressive', '')
        out_model = out_dir / f'trades_model_{model_tag}_{args.split}.csv'
        df_model.to_csv(out_model, index=False)
        print(f"   Sauvegardé: {out_model}  ({out_model.stat().st_size/1024:.1f} KB)")
        print_stats(f'MODEL TRADES ({model_tag}, {args.split})', df_model, args.fees)


if __name__ == '__main__':
    main()
