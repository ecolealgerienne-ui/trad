#!/usr/bin/env python3
"""
Extrait les FLIPS du modèle (changements de décision sign(proba)[t] != sign[t-1])
avec leurs features contextuelles, et 2 labels :
  - is_good_flip       : flip d'accord avec l'oracle au même instant (instantané)
  - is_profitable_flip : trade qui suit ce flip est net positif (PROFITABILITÉ)

Idée (validée 2026-04-18) :
  À chaque moment où le modèle CHANGE d'avis, capturer les conditions
  de marché et noter si ce flip mène à un trade rentable.

Sortie : 2 CSV séparés par direction du flip
  - flips_to_long_<config>.csv  : flips où nouveau_signal = +1
  - flips_to_short_<config>.csv : flips où nouveau_signal = -1

Features contextuelles (causales) :
  Marché (causales depuis CSV) :
    hour_utc, dayofweek, month
    atr_14_norm     : ATR(14) / close
    atr_ratio_sl    : ATR(14) / ATR(48)        (court/long terme)
    close_slope_1h  : (close[t] - close[t-12]) / close[t-12]
    close_slope_4h  : (close[t] - close[t-48]) / close[t-48]
    distance_to_ma20 : (close - MA20) / close
    distance_to_ma60 : (close - MA60) / close
    volume_relative  : vol[t] / mean(vol[t-288..t])    (24h)

  État interne du modèle (catégorie 1 — recommandation expert) :
    proba_std_12rows         : std(proba) sur 12 dernières rows
                                → modèle hésitant si std élevée
    recent_flip_count_1h     : nb flips dans les 12 rows précédentes
                                → si déjà flippé bcp, suspect
    proba_distance_to_extreme : min(proba, 1-proba)
                                → inverse de la confiance bimodale
    proba_trend_3rows        : (proba[t] - proba[t-3]) / 3
                                → direction du changement de proba

Labels :
  new_signal_model     : +1 (flip vers LONG) ou -1 (flip vers SHORT)
  oracle_now           : sign(y_continuous[t])  → +1 / 0 / -1
  is_good_flip         : 1 si new_signal_model == oracle_now (legacy)
  is_profitable_flip   : 1 si trade qui suit (jusqu'au prochain flip) > 0 net

Features ÉLIMINÉES par rapport à v1 (Cohen's d <0.1) :
  - time_since_last_flip (d=0.07)
  - range_vs_atr (d=0.04)

Usage :
    python scripts/extract_model_flips.py --npz <NPZ> --preds <PREDS> --split test
    python scripts/extract_model_flips.py --npz ... --preds ... --label profitable
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

    # NB: range_vs_atr supprimé (Cohen's d=0.04 = inutile)
    return out


def detect_flips(proba, threshold=0.5):
    """Détecte les indices où sign(proba - threshold) change."""
    sig = np.where(proba > threshold, 1, -1)
    diff = np.diff(sig)
    flip_mask = np.concatenate([[False], diff != 0])
    flip_indices = np.where(flip_mask)[0]
    return flip_indices, sig


def compute_proba_features(proba, flip_indices, std_window=12,
                              flip_count_window=12, trend_lag=3):
    """
    Catégorie 1 — État interne du modèle (recommandation expert) :
      - proba_std_<W>           : std(proba) sur W rows précédentes
      - proba_distance_to_extreme : min(proba, 1-proba)
      - proba_trend_<L>         : (proba[t] - proba[t-L]) / L
      - recent_flip_count_<W>   : nb flips dans les W rows précédentes (excl. t)
    """
    n = len(proba)
    p_series = pd.Series(proba)

    # Volatilité de la proba (modèle hésitant)
    proba_std = p_series.rolling(std_window, min_periods=2).std().values

    # Distance à l'extrême (inverse de bimodalité)
    proba_dist_extreme = np.minimum(proba, 1.0 - proba)

    # Trend de la proba (direction du changement)
    proba_trend = np.zeros(n)
    if n > trend_lag:
        proba_trend[trend_lag:] = (proba[trend_lag:] - proba[:-trend_lag]) / float(trend_lag)

    # Nb flips récents (sur les W rows AVANT t, pas inclus t)
    flip_mask = np.zeros(n, dtype=int)
    flip_mask[flip_indices] = 1
    flip_mask_shifted = np.concatenate([[0], flip_mask[:-1]])
    recent_flip_count = (pd.Series(flip_mask_shifted)
                          .rolling(flip_count_window, min_periods=1).sum().values)

    return {
        'proba_std_12rows': proba_std,
        'proba_distance_to_extreme': proba_dist_extreme,
        'proba_trend_3rows': proba_trend,
        'recent_flip_count_1h': recent_flip_count,
    }


def compute_profitable_flip_labels(flip_indices, sig, closes, fees=0.001):
    """
    Pour chaque flip, simule le trade qui suit (jusqu'au prochain flip)
    et retourne is_profitable_flip (1 si pnl_net > 0) + pnl_net brut.

    Conventions identiques à backtest_5min_progressive :
      entry_price = closes[flip_i + 1]   (exec lag 1 tick)
      exit_price  = closes[next_flip_i + 1]   (ou closes[-1] si dernier)
      direction = sig[flip_i] (+1 LONG, -1 SHORT)
      pnl_net = pnl_brut - 2 * fees
    """
    n = len(closes)
    n_flips = len(flip_indices)
    pnl_net_arr = np.full(n_flips, np.nan)
    duration_arr = np.full(n_flips, np.nan)

    for k, fi in enumerate(flip_indices):
        if fi + 1 >= n:
            continue
        entry_price = closes[fi + 1]
        if k + 1 < n_flips:
            next_fi = flip_indices[k + 1]
            exit_idx = min(next_fi + 1, n - 1)
        else:
            exit_idx = n - 1
        exit_price = closes[exit_idx]
        if np.isnan(entry_price) or np.isnan(exit_price) or entry_price == 0:
            continue
        direction = sig[fi]
        if direction == 1:
            pnl_brut = (exit_price - entry_price) / entry_price
        else:
            pnl_brut = (entry_price - exit_price) / entry_price
        pnl_net_arr[k] = pnl_brut - 2 * fees
        duration_arr[k] = exit_idx - (fi + 1)

    is_profitable = (pnl_net_arr > 0).astype(int)
    # NaN → considéré comme non-profitable (0)
    is_profitable = np.where(np.isnan(pnl_net_arr), 0, is_profitable)
    return is_profitable, pnl_net_arr, duration_arr


def stats_per_direction(label, df_dir, label_col='is_good_flip'):
    """Affiche stats pour un dataset (LONG ou SHORT flips).

    label_col : 'is_good_flip' (instantané oracle) ou 'is_profitable_flip' (PnL).
    """
    n = len(df_dir)
    if n == 0:
        print(f"\n  {label}: aucun flip extrait")
        return
    n_pos = int(df_dir[label_col].sum())
    n_neg = n - n_pos
    rate = n_pos / n * 100
    print(f"\n  {label} : {n:,} flips  |  positive={n_pos:,} negative={n_neg:,}  "
          f"rate ({label_col})={rate:.2f}%")
    # PnL moyen si dispo
    if 'pnl_net_flip' in df_dir.columns:
        pnl_total = df_dir['pnl_net_flip'].sum() * 100
        pnl_mean = df_dir['pnl_net_flip'].mean() * 100
        print(f"  PnL Net total des trades suivants: {pnl_total:+.2f}%  "
              f"(mean={pnl_mean:+.4f}%/flip)")

    # Stats par feature : différence positive vs negative
    excluded = {'flip_dt', 'flip_i', 'new_signal_model', 'oracle_now',
                'is_good_flip', 'is_profitable_flip', 'pnl_net_flip',
                'duration_flip', 'model_proba', 'hour_utc', 'dayofweek',
                'month'}
    feat_cols = [c for c in df_dir.columns if c not in excluded]
    print(f"  Features (mean positive vs mean negative) — labelled by '{label_col}' :")
    for c in feat_cols:
        vals_pos = df_dir.loc[df_dir[label_col] == 1, c].dropna()
        vals_neg = df_dir.loc[df_dir[label_col] == 0, c].dropna()
        if len(vals_pos) == 0 or len(vals_neg) == 0:
            continue
        m_p, m_n = vals_pos.mean(), vals_neg.mean()
        s_p, s_n = vals_pos.std(), vals_neg.std()
        s_pool = np.sqrt((s_p ** 2 + s_n ** 2) / 2) if (s_p + s_n) > 0 else 0
        d = (m_p - m_n) / s_pool if s_pool > 0 else 0
        marker = '🔥' if abs(d) > 0.2 else ('•' if abs(d) > 0.1 else '')
        print(f"    {c:<28} pos={m_p:+.6f}  neg={m_n:+.6f}  "
              f"d={d:+.3f}  {marker}")

    # Heure UTC : rate par heure
    print(f"  Rate par heure UTC ({label_col}) :")
    by_hour = df_dir.groupby('hour_utc').agg(
        n=(label_col, 'count'),
        rate=(label_col, 'mean'),
    )
    for h, row in by_hour.iterrows():
        bar = '█' * int(row['rate'] * 30)
        print(f"    {int(h):02d}h : {int(row['n']):>4} flips  "
              f"rate={row['rate']*100:5.1f}%  {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', required=True)
    parser.add_argument('--preds', required=True)
    parser.add_argument('--split', default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--fees', type=float, default=0.001,
                        help='Fees par côté (pour is_profitable_flip)')
    parser.add_argument('--label', default='profitable',
                        choices=['good', 'profitable'],
                        help='Label affiché dans les stats Cohen d (CSV contient les 2)')
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

    # Catégorie 1 : features état interne du modèle (pré-calculées sur tout p)
    print(f"\n[4.5] Calcul features modèle (proba_std, distance_extreme, "
          f"trend, recent_flips) ...")
    proba_features = compute_proba_features(p, flip_indices)

    # Label is_profitable_flip via simulation des trades
    print(f"\n[4.6] Simulation trades pour is_profitable_flip "
          f"(fees={args.fees*100:.2f}% par côté) ...")
    is_profit_arr, pnl_net_arr, dur_arr = compute_profitable_flip_labels(
        flip_indices, sig, closes, fees=args.fees)
    print(f"   profitable rate: {is_profit_arr.mean()*100:.2f}%  "
          f"PnL Net total des flips-trades: {pnl_net_arr[~np.isnan(pnl_net_arr)].sum()*100:+.2f}%")

    # Pour chaque flip, construire la row du dataset
    print(f"\n[5] Construction dataset (1 row par flip) ...")
    rows = []
    for k, fi in enumerate(flip_indices):
        new_signal = int(sig[fi])
        oracle_val = y_cont[fi]
        oracle_sign = 0 if oracle_val == 0 else int(np.sign(oracle_val))
        is_good = int(new_signal == oracle_sign and oracle_sign != 0)

        feats = feat_aligned.iloc[fi]
        row = {
            'flip_i': fi,
            'flip_dt': dates_5m[fi],
            'new_signal_model': new_signal,
            'oracle_now': oracle_sign,
            'is_good_flip': is_good,
            'is_profitable_flip': int(is_profit_arr[k]),
            'pnl_net_flip': float(pnl_net_arr[k]) if not np.isnan(pnl_net_arr[k]) else 0.0,
            'duration_flip': float(dur_arr[k]) if not np.isnan(dur_arr[k]) else 0.0,
            'model_proba': float(p[fi]),
            'proba_std_12rows': float(proba_features['proba_std_12rows'][fi]),
            'proba_distance_to_extreme': float(proba_features['proba_distance_to_extreme'][fi]),
            'proba_trend_3rows': float(proba_features['proba_trend_3rows'][fi]),
            'recent_flip_count_1h': float(proba_features['recent_flip_count_1h'][fi]),
            **feats.to_dict(),
        }
        rows.append(row)

    df_flips = pd.DataFrame(rows)
    print(f"   Dataset shape: {df_flips.shape}")
    print(f"   is_good_flip       : {int(df_flips['is_good_flip'].sum()):,}/{len(df_flips):,}  "
          f"({df_flips['is_good_flip'].mean() * 100:.2f}%)  [oracle instantané]")
    print(f"   is_profitable_flip : {int(df_flips['is_profitable_flip'].sum()):,}/{len(df_flips):,}  "
          f"({df_flips['is_profitable_flip'].mean() * 100:.2f}%)  [PnL > 0 du trade]")

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
    label_col = f'is_{args.label}_flip'
    print(f"\n{'=' * 100}")
    print(f"STATS — discrimination par feature (Cohen's d) avec label='{label_col}'")
    print(f"{'=' * 100}")
    stats_per_direction('FLIPS → LONG', df_long, label_col=label_col)
    stats_per_direction('FLIPS → SHORT', df_short, label_col=label_col)

    print(f"\n{'=' * 100}")
    print(f"PROCHAINE ÉTAPE :")
    print(f"  Entraîner 2 classifiers (XGBoost) sur ces datasets pour")
    print(f"  prédire is_good_flip à partir des features contextuelles.")
    print(f"{'=' * 100}")


if __name__ == '__main__':
    main()
