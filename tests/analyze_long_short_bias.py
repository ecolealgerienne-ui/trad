#!/usr/bin/env python3
"""
Analyser le biais LONG vs SHORT dans les meta-probs.

Analyse:
1. Distribution des meta-labels (profitabilité) par direction
2. Distribution des meta-probs prédites par direction
3. Recommandation de thresholds asymétriques si biais détecté
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import joblib
import argparse


def analyze_bias(indicator: str = 'macd', filter_type: str = 'kalman', split: str = 'test'):
    """Analyse le biais LONG vs SHORT."""

    # Charger meta-labels data
    data_path = Path(f'data/prepared/meta_labels_{indicator}_{filter_type}_{split}_aligned.npz')
    if not data_path.exists():
        print(f"❌ Fichier introuvable: {data_path}")
        return

    data = np.load(data_path, allow_pickle=True)
    predictions_primary = data[f'predictions_{indicator}']
    meta_labels = data['meta_labels']

    # Charger XGBoost meta-model
    model_path = Path(f'models/meta_model/meta_model_xgboost_{filter_type}_aligned.pkl')
    if not model_path.exists():
        print(f"❌ Meta-model introuvable: {model_path}")
        return

    print(f"\n{'='*80}")
    print(f"ANALYSE BIAIS LONG vs SHORT")
    print(f"{'='*80}")
    print(f"Indicateur: {indicator}")
    print(f"Filter: {filter_type}")
    print(f"Split: {split}")

    # Binariser predictions primaires
    directions = (predictions_primary > 0.5).astype(int)  # 1=UP/LONG, 0=DOWN/SHORT

    # Filtrer samples ignored (-1)
    mask = meta_labels != -1
    directions_filtered = directions[mask]
    meta_labels_filtered = meta_labels[mask]

    # Séparer LONG et SHORT
    long_mask = directions_filtered == 1
    short_mask = directions_filtered == 0

    # === ANALYSE 1: Distribution des LABELS (ground truth) ===
    print(f"\n{'='*80}")
    print("1. DISTRIBUTION META-LABELS (Ground Truth)")
    print(f"{'='*80}")

    long_total = long_mask.sum()
    long_profitable = (meta_labels_filtered[long_mask] == 1).sum()
    long_rate = long_profitable / long_total if long_total > 0 else 0

    short_total = short_mask.sum()
    short_profitable = (meta_labels_filtered[short_mask] == 1).sum()
    short_rate = short_profitable / short_total if short_total > 0 else 0

    print(f"\nLONG (direction=1, UP):")
    print(f"  Total samples: {long_total:,}")
    print(f"  Profitable (meta_label=1): {long_profitable:,} ({100*long_rate:.1f}%)")
    print(f"  Unprofitable (meta_label=0): {long_total - long_profitable:,} ({100*(1-long_rate):.1f}%)")

    print(f"\nSHORT (direction=0, DOWN):")
    print(f"  Total samples: {short_total:,}")
    print(f"  Profitable (meta_label=1): {short_profitable:,} ({100*short_rate:.1f}%)")
    print(f"  Unprofitable (meta_label=0): {short_total - short_profitable:,} ({100*(1-short_rate):.1f}%)")

    print(f"\n{'='*80}")
    print(f"PROFITABILITY RATE:")
    print(f"  LONG:  {long_rate:.3f} ({100*long_rate:.1f}%)")
    print(f"  SHORT: {short_rate:.3f} ({100*short_rate:.1f}%)")
    print(f"  Ratio LONG/SHORT: {long_rate/short_rate:.2f}x")

    if long_rate > short_rate * 1.1:
        print(f"\n⚠️  BIAIS LABELS DÉTECTÉ: LONG {long_rate/short_rate:.2f}x plus profitable que SHORT!")
        print(f"    → Marché en tendance haussière (crypto 2017-2026)")
    elif short_rate > long_rate * 1.1:
        print(f"\n⚠️  BIAIS LABELS DÉTECTÉ: SHORT {short_rate/long_rate:.2f}x plus profitable que LONG!")
    else:
        print(f"\n✅ Distribution labels relativement équilibrée")

    # === ANALYSE 2: Charger meta-model et prédire ===
    print(f"\n{'='*80}")
    print("2. DISTRIBUTION META-PROBS (Prédictions du modèle)")
    print(f"{'='*80}")

    meta_model = joblib.load(model_path)

    # Reconstruire features meta (simplifié - même logique que backtest)
    # On charge les 3 indicateurs pour calculer confidence
    try:
        pred_macd = data['predictions_macd']
        pred_rsi = data['predictions_rsi']
        pred_cci = data['predictions_cci']

        # Confidence features
        probs_stack = np.stack([pred_macd, pred_rsi, pred_cci], axis=1)
        confidence_spread = probs_stack.max(axis=1) - probs_stack.min(axis=1)
        confidence_mean = probs_stack.mean(axis=1)

        # ATR (simplifié - on met des valeurs moyennes)
        volatility_atr = np.ones(len(pred_macd)) * 0.5

        # Features meta
        meta_features = np.column_stack([
            pred_macd,
            pred_rsi,
            pred_cci,
            confidence_spread,
            confidence_mean,
            volatility_atr
        ])

        # Prédire meta-probs
        meta_probs = meta_model.predict_proba(meta_features)[:, 1]

        # Filtrer et séparer par direction
        meta_probs_filtered = meta_probs[mask]
        meta_probs_long = meta_probs_filtered[long_mask]
        meta_probs_short = meta_probs_filtered[short_mask]

        print(f"\nLONG (direction=1):")
        print(f"  Meta-prob moyenne: {meta_probs_long.mean():.3f}")
        print(f"  Meta-prob médiane: {np.median(meta_probs_long):.3f}")
        print(f"  Meta-prob max: {meta_probs_long.max():.3f}")
        print(f"  Samples avec prob ≥ 0.7: {(meta_probs_long >= 0.7).sum():,} ({100*(meta_probs_long >= 0.7).mean():.1f}%)")
        print(f"  Samples avec prob ≥ 0.8: {(meta_probs_long >= 0.8).sum():,} ({100*(meta_probs_long >= 0.8).mean():.1f}%)")

        print(f"\nSHORT (direction=0):")
        print(f"  Meta-prob moyenne: {meta_probs_short.mean():.3f}")
        print(f"  Meta-prob médiane: {np.median(meta_probs_short):.3f}")
        print(f"  Meta-prob max: {meta_probs_short.max():.3f}")
        print(f"  Samples avec prob ≥ 0.7: {(meta_probs_short >= 0.7).sum():,} ({100*(meta_probs_short >= 0.7).mean():.1f}%)")
        print(f"  Samples avec prob ≥ 0.8: {(meta_probs_short >= 0.8).sum():,} ({100*(meta_probs_short >= 0.8).mean():.1f}%)")

        print(f"\n{'='*80}")
        print(f"RATIO META-PROBS:")
        print(f"  LONG moyenne:  {meta_probs_long.mean():.3f}")
        print(f"  SHORT moyenne: {meta_probs_short.mean():.3f}")
        print(f"  Ratio LONG/SHORT: {meta_probs_long.mean()/meta_probs_short.mean():.2f}x")

        if meta_probs_long.mean() > meta_probs_short.mean() * 1.1:
            print(f"\n⚠️  BIAIS PRÉDICTIONS DÉTECTÉ: Meta-probs LONG {meta_probs_long.mean()/meta_probs_short.mean():.2f}x plus élevées que SHORT!")
            print(f"    → Le modèle a appris que LONG est plus profitable")

        # === RECOMMANDATIONS ===
        print(f"\n{'='*80}")
        print("3. RECOMMANDATIONS THRESHOLDS ASYMÉTRIQUES")
        print(f"{'='*80}")

        # Trouver thresholds donnant ~500 trades par direction
        from scipy.stats import scoreatpercentile

        # Threshold LONG pour avoir ~500 trades (sur test set ~640k samples)
        target_long_samples = 500
        percentile_long = 100 * (1 - target_long_samples / len(meta_probs_long))
        threshold_long = scoreatpercentile(meta_probs_long, percentile_long)

        target_short_samples = 500
        percentile_short = 100 * (1 - target_short_samples / len(meta_probs_short))
        threshold_short = scoreatpercentile(meta_probs_short, percentile_short)

        print(f"\nPour obtenir ~{target_long_samples} trades LONG:")
        print(f"  Threshold LONG recommandé: {threshold_long:.3f}")
        print(f"  Samples ≥ threshold: {(meta_probs_long >= threshold_long).sum():,}")

        print(f"\nPour obtenir ~{target_short_samples} trades SHORT:")
        print(f"  Threshold SHORT recommandé: {threshold_short:.3f}")
        print(f"  Samples ≥ threshold: {(meta_probs_short >= threshold_short).sum():,}")

        print(f"\n{'='*80}")
        print(f"CONFIGURATION RECOMMANDÉE:")
        print(f"{'='*80}")
        print(f"Option A - Thresholds symétriques (actuel):")
        print(f"  Threshold unique: 0.7-0.8")
        print(f"  → Simple mais biais LONG")

        print(f"\nOption B - Thresholds asymétriques (recommandé):")
        print(f"  Threshold LONG:  {threshold_long:.2f}")
        print(f"  Threshold SHORT: {threshold_short:.2f}")
        print(f"  → Compense le biais, équilibre LONG/SHORT")

    except Exception as e:
        print(f"\n⚠️  Erreur calcul meta-probs: {e}")


def main():
    parser = argparse.ArgumentParser(description='Analyze LONG vs SHORT bias in meta-model')
    parser.add_argument('--indicator', type=str, default='macd', choices=['macd', 'rsi', 'cci'])
    parser.add_argument('--filter', type=str, default='kalman', choices=['kalman', 'octave'])
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    args = parser.parse_args()

    analyze_bias(args.indicator, args.filter, args.split)


if __name__ == '__main__':
    main()
