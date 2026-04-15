#!/usr/bin/env python3
"""
Evaluation script for Net_macd_30m / Net_macd_1h pilot models.

Computes metrics that matter for trading, not just accuracy:
- Label distribution (detect class imbalance)
- Persistence baseline (label[t] = label[t-1])
- Majority class baseline
- Transition accuracy (the metric that matters for trading)
- AUC ROC (robust to imbalance)
- Confusion matrix on transitions

Usage:
    python tests/eval_multitf_model.py --indicator macd --timeframe 30m
    python tests/eval_multitf_model.py --indicator macd --timeframe 1h
"""

import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from constants import PREPARED_DATA_DIR


def main():
    parser = argparse.ArgumentParser(description='Evaluate multitf pilot model')
    parser.add_argument('--indicator', default='macd', choices=['macd', 'rsi', 'cci'])
    parser.add_argument('--timeframe', default='30m', choices=['30m', '1h'])
    parser.add_argument('--crossfeat', action='store_true', help='Evaluate crossfeat model')
    args = parser.parse_args()

    suffix = '_crossfeat' if args.crossfeat else ''
    model_name = f'{args.indicator}_{args.timeframe}{suffix}'
    npz_path = f'{PREPARED_DATA_DIR}/{args.indicator}_{args.timeframe}{suffix}_dataset.npz'

    if not Path(npz_path).exists():
        logger.error(f"NPZ not found: {npz_path}")
        logger.error(f"Run: python src/train_multitf.py --indicator {args.indicator} --timeframe {args.timeframe} {'--crossfeat' if args.crossfeat else ''}")
        return

    data = np.load(npz_path, allow_pickle=True)
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    y_test_pred = data['y_test_pred']  # probabilities [0,1]

    y_test_binary = (y_test_pred > 0.5).astype(int)

    print(f"\n{'='*70}")
    print(f"EVALUATION — Net_{model_name}")
    print(f"{'='*70}")

    # =========================================================================
    # 1. Label distribution
    # =========================================================================
    print(f"\n--- 1. LABEL DISTRIBUTION ---")
    for name, y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        n_up = (y == 1).sum()
        n_down = (y == 0).sum()
        n = len(y)
        print(f"  {name:5s}: {n:>10,} samples | UP={n_up:,} ({n_up/n*100:.1f}%) | DOWN={n_down:,} ({n_down/n*100:.1f}%)")

    # =========================================================================
    # 2. Baseline: persistence (label[t] = label[t-1])
    # =========================================================================
    persistence_pred = np.zeros_like(y_test)
    persistence_pred[0] = y_test[0]
    persistence_pred[1:] = y_test[:-1]
    persistence_acc = (persistence_pred == y_test).mean()

    print(f"\n--- 2. PERSISTENCE BASELINE ---")
    print(f"  label_pred[t] = label_true[t-1]")
    print(f"  Accuracy: {persistence_acc*100:.2f}%")

    # =========================================================================
    # 3. Baseline: majority class
    # =========================================================================
    majority_class = 1 if y_train.mean() > 0.5 else 0
    majority_acc = (y_test == majority_class).mean()

    print(f"\n--- 3. MAJORITY CLASS BASELINE ---")
    print(f"  Majority class (from train): {majority_class} ({'UP' if majority_class==1 else 'DOWN'})")
    print(f"  Accuracy: {majority_acc*100:.2f}%")

    # =========================================================================
    # 4. Model accuracy (overall)
    # =========================================================================
    model_acc = (y_test_binary == y_test).mean()

    print(f"\n--- 4. MODEL ACCURACY ---")
    print(f"  Overall: {model_acc*100:.2f}%")
    print(f"  vs Persistence: {(model_acc - persistence_acc)*100:+.2f}%")
    print(f"  vs Majority:    {(model_acc - majority_acc)*100:+.2f}%")

    # =========================================================================
    # 5. Transition accuracy (THE metric that matters)
    # =========================================================================
    # Transitions = positions where label changes
    transitions = np.zeros(len(y_test), dtype=bool)
    transitions[1:] = y_test[1:] != y_test[:-1]
    n_transitions = transitions.sum()
    n_continuations = (~transitions).sum()

    # Accuracy on transitions only
    if n_transitions > 0:
        trans_acc = (y_test_binary[transitions] == y_test[transitions]).mean()
    else:
        trans_acc = 0

    # Accuracy on continuations only
    if n_continuations > 0:
        cont_acc = (y_test_binary[~transitions] == y_test[~transitions]).mean()
    else:
        cont_acc = 0

    # Persistence on transitions (always wrong by definition)
    persistence_trans_acc = 0.0  # persistence always gets transitions wrong

    print(f"\n--- 5. TRANSITION ACCURACY (key metric) ---")
    print(f"  Total transitions: {n_transitions:,} / {len(y_test):,} ({n_transitions/len(y_test)*100:.1f}%)")
    print(f"  Total continuations: {n_continuations:,} ({n_continuations/len(y_test)*100:.1f}%)")
    print(f"")
    print(f"  {'Metric':<30} {'Model':>10} {'Persistence':>12}")
    print(f"  {'-'*55}")
    print(f"  {'Accuracy on transitions':<30} {trans_acc*100:>9.2f}% {persistence_trans_acc*100:>11.2f}%")
    print(f"  {'Accuracy on continuations':<30} {cont_acc*100:>9.2f}% {100.0:>11.2f}%")
    print(f"  {'Overall accuracy':<30} {model_acc*100:>9.2f}% {persistence_acc*100:>11.2f}%")

    # =========================================================================
    # 6. AUC ROC
    # =========================================================================
    try:
        auc = roc_auc_score(y_test, y_test_pred)
    except ValueError:
        auc = 0.0

    print(f"\n--- 6. AUC ROC ---")
    print(f"  AUC: {auc:.4f}")
    if auc > 0.85:
        print(f"  ✅ Strong signal (AUC > 0.85)")
    elif auc > 0.70:
        print(f"  ⚠️  Moderate signal (0.70 < AUC < 0.85)")
    else:
        print(f"  ❌ Weak signal (AUC < 0.70)")

    # =========================================================================
    # 7. Confusion matrix on transitions
    # =========================================================================
    if n_transitions > 0:
        # Categorize transitions
        up_to_down = transitions & (y_test == 0)  # was UP (prev=1), now DOWN (curr=0)
        down_to_up = transitions & (y_test == 1)  # was DOWN (prev=0), now UP (curr=1)

        n_up_to_down = up_to_down.sum()
        n_down_to_up = down_to_up.sum()

        # Model correct on each type
        u2d_correct = (y_test_binary[up_to_down] == 0).sum() if n_up_to_down > 0 else 0
        d2u_correct = (y_test_binary[down_to_up] == 1).sum() if n_down_to_up > 0 else 0

        print(f"\n--- 7. CONFUSION ON TRANSITIONS ---")
        print(f"  {'Transition type':<20} {'Total':>8} {'Correct':>8} {'Rate':>8}")
        print(f"  {'-'*48}")
        print(f"  {'UP → DOWN':<20} {n_up_to_down:>8,} {u2d_correct:>8,} {u2d_correct/n_up_to_down*100 if n_up_to_down else 0:>7.1f}%")
        print(f"  {'DOWN → UP':<20} {n_down_to_up:>8,} {d2u_correct:>8,} {d2u_correct/n_down_to_up*100 if n_down_to_up else 0:>7.1f}%")
        print(f"  {'ALL transitions':<20} {n_transitions:>8,} {u2d_correct+d2u_correct:>8,} {(u2d_correct+d2u_correct)/n_transitions*100:>7.1f}%")

    # =========================================================================
    # 8. Full confusion matrix
    # =========================================================================
    cm = confusion_matrix(y_test, y_test_binary)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n--- 8. FULL CONFUSION MATRIX ---")
    print(f"                  Predicted DOWN  Predicted UP")
    print(f"  Actual DOWN     {tn:>12,}    {fp:>12,}")
    print(f"  Actual UP       {fn:>12,}    {tp:>12,}")
    print(f"")
    print(f"  Precision (UP):  {tp/(tp+fp)*100:.1f}%")
    print(f"  Recall (UP):     {tp/(tp+fn)*100:.1f}%")
    print(f"  Precision (DOWN): {tn/(tn+fn)*100:.1f}%")
    print(f"  Recall (DOWN):    {tn/(tn+fp)*100:.1f}%")

    # =========================================================================
    # Summary verdict
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    issues = []
    if model_acc - persistence_acc < 1.0:
        issues.append(f"Model barely beats persistence ({(model_acc-persistence_acc)*100:+.2f}%)")
    if trans_acc < 50:
        issues.append(f"Transition accuracy below 50% ({trans_acc*100:.1f}%)")
    if auc < 0.70:
        issues.append(f"AUC too low ({auc:.4f})")

    if not issues:
        print(f"  ✅ Model shows genuine signal")
        print(f"     Transition accuracy: {trans_acc*100:.1f}%")
        print(f"     AUC: {auc:.4f}")
        print(f"     Beats persistence by {(model_acc-persistence_acc)*100:+.2f}%")
    else:
        print(f"  ⚠️  Issues detected:")
        for issue in issues:
            print(f"     - {issue}")


if __name__ == '__main__':
    main()
