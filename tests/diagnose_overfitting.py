#!/usr/bin/env python3
"""
Script de diagnostic pour comprendre l'overfitting du modèle large.

Analyse:
1. Distribution labels train/val/test
2. Périodes temporelles de chaque split
3. Volatilité et difficulté de chaque période
4. Vérification du calcul accuracy val vs test
"""

import numpy as np
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Diagnostiquer l'overfitting")
    parser.add_argument('--data', type=str, required=True,
                        help='Fichier .npz avec prédictions')
    args = parser.parse_args()

    print("=" * 80)
    print("DIAGNOSTIC OVERFITTING - Modèle Large")
    print("=" * 80)

    # Charger données
    data = np.load(args.data, allow_pickle=True)

    # Extraire
    Y_train = data['Y_train'][:, 2:3] if data['Y_train'].ndim == 2 else data['Y_train']
    Y_val = data['Y_val'][:, 2:3] if data['Y_val'].ndim == 2 else data['Y_val']
    Y_test = data['Y_test'][:, 2:3] if data['Y_test'].ndim == 2 else data['Y_test']

    Y_train_pred = data['Y_train_pred']
    Y_val_pred = data['Y_val_pred']
    Y_test_pred = data['Y_test_pred']

    T_train = data.get('T_train')
    T_val = data.get('T_val')
    T_test = data.get('T_test')

    if T_train is not None and T_train.ndim == 2:
        T_train = T_train[:, 2:3]
        T_val = T_val[:, 2:3]
        T_test = T_test[:, 2:3]

    OHLCV_train = data['OHLCV_train']
    OHLCV_val = data['OHLCV_val']
    OHLCV_test = data['OHLCV_test']

    print("\n1. DISTRIBUTION LABELS")
    print("-" * 80)

    def analyze_labels(Y, name):
        up_pct = (Y == 1).mean() * 100
        down_pct = (Y == 0).mean() * 100
        print(f"{name:8} UP: {up_pct:5.2f}% | DOWN: {down_pct:5.2f}% | Total: {len(Y):,}")

    analyze_labels(Y_train, "Train")
    analyze_labels(Y_val, "Val")
    analyze_labels(Y_test, "Test")

    print("\n2. PÉRIODES TEMPORELLES")
    print("-" * 80)

    def get_period(OHLCV):
        timestamps = OHLCV[:, 0]
        start = np.min(timestamps)
        end = np.max(timestamps)
        # Convert nanoseconds to readable date
        from datetime import datetime
        start_date = datetime.fromtimestamp(start / 1e9).strftime('%Y-%m-%d')
        end_date = datetime.fromtimestamp(end / 1e9).strftime('%Y-%m-%d')
        duration_days = (end - start) / 1e9 / 86400
        return start_date, end_date, duration_days

    train_start, train_end, train_days = get_period(OHLCV_train)
    val_start, val_end, val_days = get_period(OHLCV_val)
    test_start, test_end, test_days = get_period(OHLCV_test)

    print(f"Train: {train_start} → {train_end} ({train_days:.1f} jours)")
    print(f"Val:   {val_start} → {val_end} ({val_days:.1f} jours)")
    print(f"Test:  {test_start} → {test_end} ({test_days:.1f} jours)")

    print("\n3. TRANSITIONS (si weighted loss utilisé)")
    print("-" * 80)

    if T_train is not None:
        def analyze_transitions(T, name):
            trans_pct = (T == 1).mean() * 100
            cont_pct = (T == 0).mean() * 100
            print(f"{name:8} Transitions: {trans_pct:5.2f}% | Continuations: {cont_pct:5.2f}%")

        analyze_transitions(T_train, "Train")
        analyze_transitions(T_val, "Val")
        analyze_transitions(T_test, "Test")
    else:
        print("Pas de transitions (T_* absent)")

    print("\n4. ACCURACY RECALCULÉE (vérification)")
    print("-" * 80)

    def recalc_accuracy(Y_true, Y_pred, name):
        preds_binary = (Y_pred >= 0.5).astype(float)
        accuracy = (preds_binary == Y_true).mean()

        # Par classe
        up_mask = (Y_true == 1)
        down_mask = (Y_true == 0)

        up_acc = (preds_binary[up_mask] == Y_true[up_mask]).mean() if up_mask.sum() > 0 else 0
        down_acc = (preds_binary[down_mask] == Y_true[down_mask]).mean() if down_mask.sum() > 0 else 0

        print(f"{name:8} Acc: {accuracy*100:5.2f}% | UP Acc: {up_acc*100:5.2f}% | DOWN Acc: {down_acc*100:5.2f}%")
        return accuracy

    train_acc = recalc_accuracy(Y_train, Y_train_pred, "Train")
    val_acc = recalc_accuracy(Y_val, Y_val_pred, "Val")
    test_acc = recalc_accuracy(Y_test, Y_test_pred, "Test")

    print("\n5. GAPS")
    print("-" * 80)
    gap_train_val = (train_acc - val_acc) * 100
    gap_val_test = (test_acc - val_acc) * 100

    print(f"Gap Train/Val:  {gap_train_val:+6.2f}%  {'❌ OVERFITTING' if gap_train_val > 10 else '✅ OK'}")
    print(f"Gap Val/Test:   {gap_val_test:+6.2f}%  {'⚠️ ANOMALIE' if abs(gap_val_test) > 10 else '✅ OK'}")

    print("\n6. VOLATILITÉ PAR SPLIT (difficulté)")
    print("-" * 80)

    def calc_volatility(OHLCV):
        close = OHLCV[:, 5]
        returns = np.diff(close) / close[:-1]
        volatility = np.std(returns)
        return volatility * 100

    train_vol = calc_volatility(OHLCV_train)
    val_vol = calc_volatility(OHLCV_val)
    test_vol = calc_volatility(OHLCV_test)

    print(f"Train Vol: {train_vol:.4f}%")
    print(f"Val Vol:   {val_vol:.4f}%  {'(+' if val_vol > train_vol else '('}{(val_vol/train_vol - 1)*100:+.1f}%)")
    print(f"Test Vol:  {test_vol:.4f}%  {'(+' if test_vol > train_vol else '('}{(test_vol/train_vol - 1)*100:+.1f}%)")

    if val_vol > train_vol * 1.2:
        print("\n⚠️  Val set est 20%+ plus volatil → période plus difficile")

    print("\n7. DIAGNOSTIC FINAL")
    print("=" * 80)

    if gap_train_val > 20:
        print("❌ OVERFITTING SÉVÈRE (gap train/val > 20%)")
        print("\n   Causes possibles:")
        print("   - Modèle trop grand pour la quantité de données")
        print("   - Dropout insuffisant")
        print("   - Weighted transitions trop agressif")

        print("\n   Solutions:")
        print("   1. Revenir au modèle baseline (64/64/2)")
        print("   2. Augmenter dropout (--lstm-dropout 0.35 --dense-dropout 0.4)")
        print("   3. Réduire taille (96/96/2 au lieu de 128/128/3)")
        print("   4. Désactiver weighted transitions si activé")

    if abs(gap_val_test) > 15:
        print("\n⚠️  ANOMALIE: Test beaucoup meilleur que Val")
        print("\n   Explications possibles:")
        print("   - Val set d'une période exceptionnellement difficile")
        print("   - Test set d'une période plus facile")
        print("   - Split temporel biaisé")

    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
