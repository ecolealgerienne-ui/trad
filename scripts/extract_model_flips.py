#!/usr/bin/env python3
"""
Extrait les FLIPS du modèle (changements de décision sign(proba)[t] != sign[t-1])
avec leurs features contextuelles, et label "good_flip" basé sur l'accord
avec l'oracle au même instant.

Idée (validée 2026-04-18) :
  À chaque moment où le modèle CHANGE d'avis, le seul instant où on prend
  une vraie décision active, on capture les conditions de marché et on note
  si ce flip était fondé (oracle d'accord) ou parasite (oracle contraire).

Sortie : 2 CSV séparés par direction du flip
  - flips_to_long_<config>.csv  : flips où nouveau_signal = +1
  - flips_to_short_<config>.csv : flips où nouveau_signal = -1

Features contextuelles (causales, calculées à l'instant t du flip) :
  - hour_utc, dayofweek, month
  - atr_14_norm     : ATR(14) / close
  - atr_ratio_sl    : ATR(14) / ATR(48)        (court/long terme)
  - close_slope_1h  : (close[t] - close[t-12]) / close[t-12]
  - close_slope_4h  : (close[t] - close[t-48]) / close[t-48]
  - distance_to_ma20 : (close - MA20) / close
  - distance_to_ma60 : (close - MA60) / close
  - volume_relative  : vol[t] / mean(vol[t-288..t])    (24h)
  - range_vs_atr     : (high - low)[t] / ATR(14)[t]
  - time_since_last_flip : rows 5min depuis le flip précédent
  - model_proba         : proba brute (raw output modèle)
  - model_proba_strength : |proba - 0.5| (force du signal)

Labels :
  - new_signal_model    : +1 (flip vers LONG) ou -1 (flip vers SHORT)
  - oracle_now          : sign(y_continuous[t])  → +1 / 0 / -1
  - is_good_flip        : 1 si new_signal_model == oracle_now, 0 sinon

Usage :
    python scripts/extract_model_flips.py --npz <NPZ> --preds <PREDS> --split test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.signal_processing.core import load_csv, calculate_atr

CSV_5M = Path('data_trad/BTCUSD_all_5m.csv')


def compute_features_full(df_5m_full, atr_period_short=14, atr_period_long=48,
                            ma_short=20, ma_long=60, slope_short=12, slope_long=48,
                            vol_window=288):
    """Calcule toutes les features contextuelles causales sur tout df_5m."""
    out = pd.DataFrame(index=df_5m_full.index)

    close = df_5m_full['close']
    high = df_5m_full['high']
    low = df_5m_full['low']
    vol = df_5m_full['volume'] if 'volume' in df_5m_full.columns else None

    # Temporel (causal par construction)
    out['hour_utc'] = df_5m_full.index.hour
    out['dayofweek'] = df_5m_full.index.dayofweek
    out['month'] = df_5m_full.index.month

    # ATR (causal)
    atr_short = calculate_atr(df_5m_full, period=atr_period_short, normalize=False)
    atr_long = calculate_atr(df_5m_full, period=atr_period_long, normalize=False)
    out['atr_14_norm'] = atr_short / close.values
    out['atr_ratio_sl'] = atr_short / np.where(atr_long > 0, atr_long, np.nan)

    # Slopes (causales : pct_change backward)
    out['close_slope_1h'] = close.pct_change(slope_short).values
    out['close_slope_4h'] = close.pct_change(slope_long).values

    # Distance à moyenne mobile (causale : rolling backward)
    ma20 = close.rolling(ma_short).mean()
    ma60 = close.rolling(ma_long).mean()
    out['distance_to_ma20'] = ((close - ma20) / close).values
    out['distance_to_ma60'] = ((close - ma60) / close).values

    # Volume relatif (causal)
    if vol is not None:
        vol_avg = vol.rolling(vol_window).mean()
        out['volume_relative'] = (vol / vol_avg).values
    else:
        out['volume_relative'] = np.nan

    # Range / ATR
    out['range_vs_atr'] = ((high - low).values
                            / np.where(atr_short > 0, atr_short, np.nan))

    return out


def detect_flips(proba, threshold=0.5):
    """Détecte les indices où sign(proba - threshold) change."""
    sig = np.where(proba > threshold, 1, -1)
    # flip à i : sig[i] != sig[i-1]
    diff = np.diff(sig)
    flip_mask = np.concatenate([[False], diff != 0])
    flip_indices = np.where(flip_mask)[0]
    return flip_indices, sig


def stats_per_direction(label, df_dir):
    """Affiche stats pour un dataset (LONG ou SHORT flips)."""
    n = len(df_dir)
    if n == 0:
        print(f"\n  {label}: aucun flip extrait")
        return
    n_good = int(df_dir['is_good_flip'].sum())
    n_bad = n - n_good
    wr = n_good / n * 100
    print(f"\n  {label} : {n:,} flips  |  good={n_good:,} bad={n_bad:,}  "
          f"good rate={wr:.2f}%")

    # Stats par feature : différence good vs bad
    feat_cols = [c for c in df_dir.columns
                  if c not in ('flip_dt', 'flip_i', 'new_signal_model',
                               'oracle_now', 'is_good_flip', 'model_proba',
                               'model_proba_strength', 'hour_utc', 'dayofweek',
                               'month')]
    print(f"  Features (mean good vs mean bad) :")
    for c in feat_cols:
        vals_good = df_dir.loc[df_dir['is_good_flip'] == 1, c].dropna()
        vals_bad = df_dir.loc[df_dir['is_good_flip'] == 0, c].dropna()
        if len(vals_good) == 0 or len(vals_bad) == 0:
            continue
        m_g, m_b = vals_good.mean(), vals_bad.mean()
        s_g, s_b = vals_good.std(), vals_bad.std()
        # Cohen's d (effect size)
        s_pool = np.sqrt((s_g ** 2 + s_b ** 2) / 2) if (s_g + s_b) > 0 else 0
        d = (m_g - m_b) / s_pool if s_pool > 0 else 0
        marker = '🔥' if abs(d) > 0.2 else ''
        print(f"    {c:<22} good={m_g:+.6f}  bad={m_b:+.6f}  "
              f"d={d:+.3f}  {marker}")

    # Heure UTC : good rate par heure (signal fort observé sur Oracle)
    print(f"  Good rate par heure UTC :")
    by_hour = df_dir.groupby('hour_utc').agg(
        n=('is_good_flip', 'count'),
        gr=('is_good_flip', 'mean'),
    )
    for h, row in by_hour.iterrows():
        bar = '█' * int(row['gr'] * 30)
        print(f"    {int(h):02d}h : {int(row['n']):>4} flips  "
              f"good rate={row['gr']*100:5.1f}%  {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', required=True)
    parser.add_argument('--preds', required=True)
    parser.add_argument('--split', default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output-dir', default='results/flips')
    args = parser.parse_args()

    print("=" * 100)
    print(f"EXTRACT MODEL FLIPS — split={args.split}  threshold={args.threshold}")
    print("=" * 100)

    # Load NPZ + preds
    npz_path = Path(args.npz)
    preds_path = Path(args.preds)
    if not npz_path.exists() or not preds_path.exists():
        print(f"❌ NPZ ou preds introuvable")
        return
    ds = np.load(npz_path, allow_pickle=True)
    preds = np.load(preds_path, allow_pickle=True)

    closes = ds[f'closes_{args.split}']
    dates_5m = pd.to_datetime(ds[f'dates_{args.split}'])
    y_cont = ds[f'y_{args.split}_continuous']
    p = preds[f'{args.split}_preds_proba']
    print(f"\n[1] Split {args.split}: {len(closes):,} rows  |  "
          f"{dates_5m[0]} → {dates_5m[-1]}")

    # Charger CSV pour HLCV complet
    print(f"\n[2] Charger CSV {CSV_5M} pour HLCV ...")
    if not CSV_5M.exists():
        print(f"❌ CSV introuvable")
        return
    df_5m_full = load_csv(CSV_5M)
    has_volume = 'volume' in df_5m_full.columns
    print(f"   {len(df_5m_full):,} rows  |  volume={'OK' if has_volume else 'ABSENT'}")

    # Calculer toutes les features sur df_5m_full
    print(f"\n[3] Calcul features contextuelles sur df_5m_full ...")
    feat_full = compute_features_full(df_5m_full)
    print(f"   {feat_full.shape[1]} features calculées")

    # Aligner sur dates_5m du split
    feat_aligned = feat_full.reindex(dates_5m)
    n_nan = int(feat_aligned.isna().any(axis=1).sum())
    print(f"   {len(feat_aligned):,} rows alignées  ({n_nan} avec ≥1 NaN)")

    # Détecter les flips
    print(f"\n[4] Détection des flips du modèle ...")
    flip_indices, sig = detect_flips(p, threshold=args.threshold)
    print(f"   {len(flip_indices):,} flips détectés "
          f"({len(flip_indices) / len(p) * 100:.2f}% des rows)")

    # Pour chaque flip, construire la row du dataset
    print(f"\n[5] Construction dataset (1 row par flip) ...")
    rows = []
    last_flip_i = None
    for fi in flip_indices:
        new_signal = int(sig[fi])
        oracle_val = y_cont[fi]
        oracle_sign = 0 if oracle_val == 0 else int(np.sign(oracle_val))
        is_good = int(new_signal == oracle_sign and oracle_sign != 0)

        time_since_last = (fi - last_flip_i) if last_flip_i is not None else np.nan
        last_flip_i = fi

        feats = feat_aligned.iloc[fi]
        row = {
            'flip_i': fi,
            'flip_dt': dates_5m[fi],
            'new_signal_model': new_signal,
            'oracle_now': oracle_sign,
            'is_good_flip': is_good,
            'model_proba': float(p[fi]),
            'model_proba_strength': float(abs(p[fi] - 0.5)),
            'time_since_last_flip': time_since_last,
            **feats.to_dict(),
        }
        rows.append(row)

    df_flips = pd.DataFrame(rows)
    print(f"   Dataset shape: {df_flips.shape}")
    print(f"   Bons flips: {int(df_flips['is_good_flip'].sum()):,}  "
          f"({df_flips['is_good_flip'].mean() * 100:.2f}%)")

    # Séparer par direction
    df_long = df_flips[df_flips['new_signal_model'] == 1].copy()
    df_short = df_flips[df_flips['new_signal_model'] == -1].copy()
    print(f"   Flips → LONG: {len(df_long):,}  |  Flips → SHORT: {len(df_short):,}")

    # Sauvegarder
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = preds_path.stem.replace('preds_', '').replace('_progressive', '')
    out_long = out_dir / f'flips_to_long_{tag}_{args.split}.csv'
    out_short = out_dir / f'flips_to_short_{tag}_{args.split}.csv'
    df_long.to_csv(out_long, index=False)
    df_short.to_csv(out_short, index=False)
    print(f"\n[6] Sauvegardé:")
    print(f"   {out_long}  ({out_long.stat().st_size / 1024:.1f} KB)")
    print(f"   {out_short}  ({out_short.stat().st_size / 1024:.1f} KB)")

    # Stats par direction
    print(f"\n{'=' * 100}")
    print(f"STATS — discrimination good vs bad par feature (Cohen's d)")
    print(f"{'=' * 100}")
    stats_per_direction('FLIPS → LONG', df_long)
    stats_per_direction('FLIPS → SHORT', df_short)

    print(f"\n{'=' * 100}")
    print(f"PROCHAINE ÉTAPE :")
    print(f"  Entraîner 2 classifiers (XGBoost) sur ces datasets pour")
    print(f"  prédire is_good_flip à partir des features contextuelles.")
    print(f"{'=' * 100}")


if __name__ == '__main__':
    main()
