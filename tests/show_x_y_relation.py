#!/usr/bin/env python3
"""
Affiche la correspondance entre X (séquence) et Y (label) de manière claire.
"""

import sys
import numpy as np
from datetime import datetime

def show_x_y_relationship(npz_path, idx=None):
    """
    Montre clairement:
    1. X = séquence de 25 timesteps avec features
    2. Y = UN SEUL label pour le dernier timestep
    3. Comment macd_direction dans Y correspond à la séquence X
    """

    data = np.load(npz_path, allow_pickle=True)

    X = data['X_train']
    Y = data['Y_train']
    OHLCV = data['OHLCV_train']

    if idx is None:
        idx = X.shape[0] // 2

    print("=" * 100)
    print("🔍 RELATION X (SÉQUENCE) → Y (LABEL)")
    print("=" * 100)

    x_sample = X[idx]  # (25, 22)
    y_sample = Y[idx]  # (13,)
    ohlcv_sample = OHLCV[idx]  # (7,)

    print(f"\n📌 Échantillon #{idx}")
    print(f"\n📊 OHLCV[{idx}] - RÉFÉRENCE PRIX:")
    print(f"   timestamp:  {ohlcv_sample[0]:.0f} ({datetime.fromtimestamp(int(ohlcv_sample[0])).strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"   asset_id:   {int(ohlcv_sample[1])} (ADA)")
    print(f"   Open:       ${ohlcv_sample[2]:.6f}")
    print(f"   High:       ${ohlcv_sample[3]:.6f}")
    print(f"   Low:        ${ohlcv_sample[4]:.6f}")
    print(f"   Close:      ${ohlcv_sample[5]:.6f}")
    print(f"   Volume:     {ohlcv_sample[6]:.2f}")

    print(f"\n" + "=" * 100)
    print(f"📥 X[{idx}] - SÉQUENCE D'ENTRÉE (25 timesteps)")
    print("=" * 100)
    print(f"\nShape: {x_sample.shape}  # (25 timesteps, 22 features)")
    print(f"\n⏱️  SÉQUENCE TEMPORELLE:")

    # Afficher quelques timesteps clés
    for t in [0, 12, 24]:
        ts = x_sample[t, 0]
        dt = datetime.fromtimestamp(int(ts)).strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n   Timestep {t:2d} | {dt}")
        print(f"      timestamp:    {ts:.0f}")
        print(f"      asset_id:     {int(x_sample[t, 1])}")

        # Afficher quelques features (colonnes 2-21)
        feat_names = [
            "ma20_slope", "ma50_slope", "regression_slope", "regression_r2",
            "adx", "macd_hist_norm", "hurst", "atr_norm", "bb_upper", "bb_middle"
        ]
        print(f"      features[2-11] (premiers 10):")
        for i, name in enumerate(feat_names):
            if i + 2 < x_sample.shape[1]:
                print(f"         [{i+2:2d}] {name:20s}: {x_sample[t, i+2]:8.4f}")

    print(f"\n" + "=" * 100)
    print(f"📤 Y[{idx}] - LABEL DE SORTIE (UN SEUL VECTEUR)")
    print("=" * 100)
    print(f"\nShape: {y_sample.shape}  # (13,) - UN SEUL vecteur de labels")
    print(f"\n⚠️  CE N'EST PAS UNE SÉQUENCE! C'est le label pour le DERNIER timestep (t=24)")
    print(f"\n📋 Contenu Y:")

    timestamp_y = y_sample[0]
    dt_y = datetime.fromtimestamp(int(timestamp_y)).strftime('%Y-%m-%d %H:%M:%S')

    print(f"\n   [0] timestamp:       {timestamp_y:.0f} ({dt_y})")
    print(f"   [1] asset_id:        {int(y_sample[1])}")
    print(f"   [2] regime:          {int(y_sample[2])} (RANGE_LOW/HIGH, TREND_LOW/HIGH)")
    print(f"   [3] trend_strength:  {y_sample[3]:.4f}")
    print(f"   [4] volatility:      {y_sample[4]:.4f}")
    print(f"\n   ━━━━ CIBLES POUR ENTRAÎNEMENT ━━━━")
    print(f"   [5] macd_direction:  {int(y_sample[5])} ({'UP' if y_sample[5] == 1 else 'DOWN'})")
    print(f"   [6] rsi_direction:   {int(y_sample[6])} ({'UP' if y_sample[6] == 1 else 'DOWN'})")
    print(f"   [7] cci_direction:   {int(y_sample[7])} ({'UP' if y_sample[7] == 1 else 'DOWN'})")

    if len(y_sample) == 13:
        print(f"\n   ━━━━ ENRICHISSEMENT (RÉGIME PROBS) ━━━━")
        print(f"   [8]  regime_prob_0:  {y_sample[8]:.4f}")
        print(f"   [9]  regime_prob_1:  {y_sample[9]:.4f}")
        print(f"   [10] regime_prob_2:  {y_sample[10]:.4f}")
        print(f"   [11] regime_prob_3:  {y_sample[11]:.4f}")
        print(f"   [12] regime_pred:    {int(y_sample[12])}")

    print(f"\n" + "=" * 100)
    print(f"🎯 OBJECTIF D'ENTRAÎNEMENT")
    print("=" * 100)

    print(f"""
Le modèle CNN-LSTM doit apprendre:

INPUT:  X = Séquence de 25 timesteps avec ~20 features de régime par timestep
        Shape: (batch, 25, 22)

OUTPUT: Y[:, 5] = macd_direction pour le dernier timestep (t=24)
        Shape: (batch,) - binaire 0/1

ATTENTION:
- X N'EST PAS c_ret! Ce sont ~20 features complexes (trend, vol, volume)
- Si tu veux c_ret uniquement, utilise dataset_*_macd_direction_only_kalman.npz
- Le dataset regime est pour CLASSIFIER les régimes, pas pour prédire direction simple
""")

    # Vérifier synchronisation timestamp
    print(f"\n" + "=" * 100)
    print(f"🔍 VÉRIFICATION SYNCHRONISATION")
    print("=" * 100)

    last_timestamp_x = x_sample[-1, 0]
    timestamp_y_val = y_sample[0]
    timestamp_ohlcv = ohlcv_sample[0]

    print(f"\n   X[{idx}] dernier timestamp (t=24): {last_timestamp_x:.0f}")
    print(f"   Y[{idx}] timestamp:                 {timestamp_y_val:.0f}")
    print(f"   OHLCV[{idx}] timestamp:             {timestamp_ohlcv:.0f}")

    if last_timestamp_x == timestamp_y_val == timestamp_ohlcv:
        print(f"\n   ✅ SYNCHRONISÉS - X, Y, OHLCV ont le même timestamp final")
    else:
        print(f"\n   ❌ DÉSYNCHRONISÉS!")

    print(f"\n" + "=" * 100)
    print(f"💡 TYPE DE DATASET")
    print("=" * 100)

    print(f"""
Ce dataset est: data/prepared/dataset_btc_eth_bnb_ada_ltc_REGIME.npz

FEATURES:
  ~20 features de régime (trend, volatility, volume indicators)
  PAS de simples returns (c_ret, h_ret, etc.)

CIBLES:
  - regime (0-3)
  - macd_direction, rsi_direction, cci_direction (0/1)

USAGE:
  1. train_regime_classifier.py: Prédire regime (0-3) → enrichir Y avec probs
  2. train.py --indicator macd: Prédire macd_direction (0/1) en utilisant features régime

SI TU VEUX UNIQUEMENT c_ret POUR PRÉDIRE DIRECTION:
  Utilise dataset_*_macd_direction_only_kalman.npz (généré par prepare_data_direction_only.py)
  - X: (25, 1) avec seulement c_ret
  - Y: (1,) avec macd_direction
""")

    print(f"\n" + "=" * 100)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Affiche relation X-Y clairement')
    parser.add_argument('--data', type=str,
                       default='data/prepared/dataset_btc_eth_bnb_ada_ltc_regime.npz',
                       help='Chemin du fichier NPZ')
    parser.add_argument('--index', type=int, default=None,
                       help='Index spécifique (défaut: milieu du dataset)')

    args = parser.parse_args()

    show_x_y_relationship(args.data, args.index)
