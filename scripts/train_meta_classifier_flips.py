#!/usr/bin/env python3
"""
Meta-classifier XGBoost pour filtrer les flips parasites du modèle.

Étape 2.C de la pipeline meta (validée par utilisateur 2026-04-18).

Architecture en 2 étages :
  Étage 1 : modèle direction (CNN-LSTM) → signe à chaque 5min
  Étage 2 : meta-classifier (XGBoost)   → "ce flip vaut-il la peine ?"
              → 2 classifiers spécialisés (LONG, SHORT)
              → label = is_profitable_flip (PnL > 0 du trade qui suit)

Méthodologie :
  - Charge les CSV `flips_to_long_*.csv` et `flips_to_short_*.csv`
    générés par scripts/extract_model_flips.py
  - Split chronologique 70/15/15 (train/val/test du META)
  - XGBoost avec scale_pos_weight (déséquilibre ~85/15)
  - Early stopping sur val
  - Calibration du seuil sur val (max F1 ou max precision)
  - Évaluation sur test : AUC, accuracy, precision/recall, log_loss
  - Feature importance (gain) + analyse SHAP (top features)
  - Sauvegarde modèles pickle + preds NPZ + importances JSON

Usage :
    python scripts/train_meta_classifier_flips.py \\
        --long-csv results/flips/flips_to_long_<TAG>_test.csv \\
        --short-csv results/flips/flips_to_short_<TAG>_test.csv
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import xgboost as xgb
except ImportError:
    print("❌ xgboost non installé. pip install xgboost")
    sys.exit(1)

from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                log_loss, precision_score, recall_score,
                                confusion_matrix)

MODELS_DIR = Path('models/meta_flips')
RESULTS_DIR = Path('results/meta_flips')

# Colonnes à exclure des features (labels, métadonnées, simulations)
EXCLUDE_COLS = {
    'flip_dt', 'flip_i', 'new_signal_model', 'oracle_now',
    'is_good_flip', 'is_profitable_flip',
    'pnl_net_flip', 'duration_flip',
}


def chronological_split(df, train_ratio=0.70, val_ratio=0.15):
    """Split chronologique strict (df doit être trié par flip_dt)."""
    df = df.sort_values('flip_dt').reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def feature_cols(df):
    """Retourne la liste des colonnes utilisables comme features."""
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def best_threshold_f1(y_true, proba, grid=None):
    """Trouve le seuil qui maximise F1 sur (y_true, proba)."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best_f1, best_thr = 0.0, 0.5
    for thr in grid:
        y_pred = (proba > thr).astype(int)
        if y_pred.sum() == 0 or y_pred.sum() == len(y_pred):
            continue
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


def best_threshold_precision(y_true, proba, min_recall=0.10, grid=None):
    """Trouve le seuil qui maximise precision avec recall >= min_recall."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best_prec, best_thr = 0.0, 0.5
    for thr in grid:
        y_pred = (proba > thr).astype(int)
        if y_pred.sum() == 0:
            continue
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        if rec >= min_recall and prec > best_prec:
            best_prec, best_thr = prec, thr
    return best_thr, best_prec


def evaluate(label, y_true, proba, threshold):
    """Évalue prédictions binaires + tableau métriques."""
    y_pred = (proba > threshold).astype(int)
    n = len(y_true)
    n_pos = int(y_true.sum())
    n_pred_pos = int(y_pred.sum())
    auc = roc_auc_score(y_true, proba) if 0 < n_pos < n else float('nan')
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ll = log_loss(y_true, np.clip(proba, 1e-15, 1 - 1e-15)) if 0 < n_pos < n else float('nan')
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  {label}  (n={n:,}, positives={n_pos}, threshold={threshold:.3f})")
    print(f"    AUC={auc:.4f}  Acc={acc*100:.2f}%  "
          f"Prec={prec*100:.2f}%  Rec={rec*100:.2f}%  F1={f1*100:.2f}%  LogLoss={ll:.4f}")
    print(f"    Confusion: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}  "
          f"(predicted positives = {n_pred_pos:,}, base rate = {n_pos/n*100:.2f}%)")
    return {'auc': auc, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
            'log_loss': ll, 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'n': n, 'n_pos': n_pos, 'n_pred_pos': int(n_pred_pos)}


def train_one_classifier(direction, df_train_pool, df_test_external=None,
                            label_col='is_profitable_flip',
                            n_estimators=500, max_depth=4, lr=0.05,
                            early_stop=20, seed=42):
    """Train + evaluate XGBoost pour une direction (LONG ou SHORT).

    Modes :
      df_test_external=None : split chronologique 70/15/15 sur df_train_pool
                               (legacy, in-sample partiel)
      df_test_external=df   : split chronologique 85/15 sur df_train_pool pour
                               train/val (early stop), df_test_external pour
                               évaluation finale (OOB rigoureux)
    """
    print(f"\n{'=' * 100}")
    print(f"TRAIN META-CLASSIFIER {direction}  "
          f"({len(df_train_pool):,} flips train pool"
          + (f", {len(df_test_external):,} flips test external" if df_test_external is not None else "")
          + ")")
    print(f"{'=' * 100}")

    if df_test_external is not None:
        # OOB mode : train/val split interne, test externe
        df_train_pool = df_train_pool.sort_values('flip_dt').reset_index(drop=True)
        n = len(df_train_pool)
        cut = int(n * 0.85)
        df_tr = df_train_pool.iloc[:cut]
        df_va = df_train_pool.iloc[cut:]
        df_te = df_test_external.sort_values('flip_dt').reset_index(drop=True)
        mode = 'OOB (rigoureux)'
    else:
        df_tr, df_va, df_te = chronological_split(df_train_pool)
        mode = 'in-sample partiel (legacy 70/15/15)'

    print(f"  Mode : {mode}")
    print(f"  Split : train={len(df_tr):,}  val={len(df_va):,}  test={len(df_te):,}")
    print(f"  Périodes : train {df_tr['flip_dt'].min()} → {df_tr['flip_dt'].max()}")
    print(f"             val   {df_va['flip_dt'].min()} → {df_va['flip_dt'].max()}")
    print(f"             test  {df_te['flip_dt'].min()} → {df_te['flip_dt'].max()}")

    # Base rates
    for name, d in [('train', df_tr), ('val', df_va), ('test', df_te)]:
        rate = d[label_col].mean() * 100
        print(f"  {name:<5}: positive rate ({label_col}) = {rate:.2f}%  "
              f"(n_pos={int(d[label_col].sum())}/{len(d)})")

    feat_cols = feature_cols(df_tr)
    print(f"\n  Features ({len(feat_cols)}) : {feat_cols}")

    X_tr = df_tr[feat_cols].values
    y_tr = df_tr[label_col].values
    X_va = df_va[feat_cols].values
    y_va = df_va[label_col].values
    X_te = df_te[feat_cols].values
    y_te = df_te[label_col].values

    # Class imbalance handling
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    spw = n_neg / max(n_pos, 1)
    print(f"\n  scale_pos_weight = {n_neg}/{n_pos} = {spw:.3f}")

    # Train XGBoost
    print(f"\n  XGBoost: n_est={n_estimators}  max_depth={max_depth}  lr={lr}  "
          f"early_stop={early_stop}")
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=spw,
        tree_method='hist',
        random_state=seed,
        n_jobs=-1,
        early_stopping_rounds=early_stop,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=50)
    best_iter = getattr(model, 'best_iteration', n_estimators)
    print(f"  best_iteration: {best_iter}")

    # Predictions
    p_tr = model.predict_proba(X_tr)[:, 1]
    p_va = model.predict_proba(X_va)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]

    # Threshold calibration sur val
    thr_f1, val_f1 = best_threshold_f1(y_va, p_va)
    thr_prec, val_prec = best_threshold_precision(y_va, p_va, min_recall=0.10)
    print(f"\n  Threshold calibration sur val :")
    print(f"    Best F1     : threshold={thr_f1:.3f}  F1={val_f1:.4f}")
    print(f"    Best Prec   : threshold={thr_prec:.3f}  Prec={val_prec:.4f} "
          f"(avec recall≥10%)")

    # Évaluation sur les 3 splits avec threshold F1-optimal
    print(f"\n  ÉVALUATION (threshold F1-optimal = {thr_f1:.3f})")
    res = {
        'train': evaluate('train', y_tr, p_tr, thr_f1),
        'val':   evaluate('val',   y_va, p_va, thr_f1),
        'test':  evaluate('test',  y_te, p_te, thr_f1),
    }

    # Évaluation aussi avec threshold haute-precision
    print(f"\n  ÉVALUATION ALTERNATIVE (threshold high-precision = {thr_prec:.3f})")
    res_prec = {
        'val':   evaluate('val',   y_va, p_va, thr_prec),
        'test':  evaluate('test',  y_te, p_te, thr_prec),
    }

    # Feature importance (gain)
    print(f"\n  Feature importance (gain) :")
    importances = model.get_booster().get_score(importance_type='gain')
    # XGBoost retourne 'f0', 'f1', etc. → map vers noms
    imp_named = {feat_cols[int(k[1:])]: v for k, v in importances.items()}
    sorted_imp = sorted(imp_named.items(), key=lambda x: -x[1])
    for name, gain in sorted_imp:
        bar = '█' * int(gain / sorted_imp[0][1] * 30)
        print(f"    {name:<28} {gain:>10.4f}  {bar}")

    return {
        'model': model,
        'feat_cols': feat_cols,
        'best_iteration': int(best_iter),
        'scale_pos_weight': spw,
        'threshold_f1': float(thr_f1),
        'threshold_precision': float(thr_prec),
        'metrics_f1': res,
        'metrics_precision': res_prec,
        'feature_importance': {k: float(v) for k, v in imp_named.items()},
        'preds': {
            'train': p_tr.astype(np.float32),
            'val':   p_va.astype(np.float32),
            'test':  p_te.astype(np.float32),
        },
        'splits': {
            'train_indices': df_tr['flip_i'].values.astype(np.int64),
            'val_indices':   df_va['flip_i'].values.astype(np.int64),
            'test_indices':  df_te['flip_i'].values.astype(np.int64),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--long-csv', required=True,
                        help='Path CSV flips_to_long_*.csv (train pool)')
    parser.add_argument('--short-csv', required=True,
                        help='Path CSV flips_to_short_*.csv (train pool)')
    parser.add_argument('--long-test-csv', default=None,
                        help='Path CSV flips_to_long_*.csv pour test OOB '
                             '(si fourni → mode OOB rigoureux)')
    parser.add_argument('--short-test-csv', default=None,
                        help='Path CSV flips_to_short_*.csv pour test OOB')
    parser.add_argument('--label', default='is_profitable_flip',
                        choices=['is_profitable_flip', 'is_good_flip'])
    parser.add_argument('--n-estimators', type=int, default=500)
    parser.add_argument('--max-depth', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--early-stop', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-suffix', default='',
                        help='Suffixe pour fichiers sauvés (ex: _oob)')
    args = parser.parse_args()

    oob_mode = args.long_test_csv is not None and args.short_test_csv is not None
    if oob_mode and args.out_suffix == '':
        args.out_suffix = '_oob'

    print("=" * 100)
    print(f"META-CLASSIFIER FLIPS — label={args.label}  "
          f"mode={'OOB rigoureux' if oob_mode else 'in-sample partiel (legacy)'}")
    print("=" * 100)

    long_csv = Path(args.long_csv)
    short_csv = Path(args.short_csv)
    if not long_csv.exists() or not short_csv.exists():
        print(f"❌ CSV introuvable")
        return

    df_long = pd.read_csv(long_csv, parse_dates=['flip_dt'])
    df_short = pd.read_csv(short_csv, parse_dates=['flip_dt'])
    print(f"\n✅ Train pool loaded:")
    print(f"   LONG  : {long_csv}  ({len(df_long):,} flips)")
    print(f"   SHORT : {short_csv}  ({len(df_short):,} flips)")

    df_long_test = None
    df_short_test = None
    if oob_mode:
        long_test_csv = Path(args.long_test_csv)
        short_test_csv = Path(args.short_test_csv)
        if not long_test_csv.exists() or not short_test_csv.exists():
            print(f"❌ Test CSV introuvable")
            return
        df_long_test = pd.read_csv(long_test_csv, parse_dates=['flip_dt'])
        df_short_test = pd.read_csv(short_test_csv, parse_dates=['flip_dt'])
        print(f"\n✅ Test external (OOB) loaded:")
        print(f"   LONG  : {long_test_csv}  ({len(df_long_test):,} flips)")
        print(f"   SHORT : {short_test_csv}  ({len(df_short_test):,} flips)")

    # Tag pour sauvegarde — basé sur le test_csv si OOB (pour cohérence avec backtest)
    if oob_mode:
        long_tag = long_test_csv.stem.replace('flips_to_long_', '')
        short_tag = short_test_csv.stem.replace('flips_to_short_', '')
    else:
        long_tag = long_csv.stem.replace('flips_to_long_', '')
        short_tag = short_csv.stem.replace('flips_to_short_', '')
    assert long_tag == short_tag, f"Mismatch tags: {long_tag} vs {short_tag}"
    tag = long_tag + args.out_suffix

    # Train les 2 classifiers
    res_long = train_one_classifier('LONG', df_long, df_long_test,
                                       label_col=args.label,
                                       n_estimators=args.n_estimators,
                                       max_depth=args.max_depth,
                                       lr=args.learning_rate,
                                       early_stop=args.early_stop,
                                       seed=args.seed)
    res_short = train_one_classifier('SHORT', df_short, df_short_test,
                                        label_col=args.label,
                                        n_estimators=args.n_estimators,
                                        max_depth=args.max_depth,
                                        lr=args.learning_rate,
                                        early_stop=args.early_stop,
                                        seed=args.seed)

    # Sauvegarde
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for direction, res in [('long', res_long), ('short', res_short)]:
        # Modèle pickle
        model_path = MODELS_DIR / f'meta_{direction}_{tag}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': res['model'],
                'feat_cols': res['feat_cols'],
                'best_iteration': res['best_iteration'],
                'scale_pos_weight': res['scale_pos_weight'],
                'threshold_f1': res['threshold_f1'],
                'threshold_precision': res['threshold_precision'],
                'direction': direction,
                'label': args.label,
                'tag': tag,
            }, f)
        print(f"\n  ✅ Modèle {direction.upper()} sauvé: {model_path}  "
              f"({model_path.stat().st_size / 1024:.1f} KB)")

        # Preds NPZ
        preds_path = RESULTS_DIR / f'meta_{direction}_preds_{tag}.npz'
        np.savez(preds_path,
                  train_proba=res['preds']['train'],
                  val_proba=res['preds']['val'],
                  test_proba=res['preds']['test'],
                  train_indices=res['splits']['train_indices'],
                  val_indices=res['splits']['val_indices'],
                  test_indices=res['splits']['test_indices'],
                  feat_cols=np.array(res['feat_cols']),
                  threshold_f1=res['threshold_f1'],
                  threshold_precision=res['threshold_precision'],
                  )
        print(f"  ✅ Preds {direction.upper()} sauvées: {preds_path}  "
              f"({preds_path.stat().st_size / 1024:.1f} KB)")

    # Métriques JSON
    metrics_path = RESULTS_DIR / f'meta_metrics_{tag}.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'tag': tag, 'label': args.label,
            'long': {
                'metrics_f1': res_long['metrics_f1'],
                'metrics_precision': res_long['metrics_precision'],
                'feature_importance': res_long['feature_importance'],
                'threshold_f1': res_long['threshold_f1'],
                'threshold_precision': res_long['threshold_precision'],
            },
            'short': {
                'metrics_f1': res_short['metrics_f1'],
                'metrics_precision': res_short['metrics_precision'],
                'feature_importance': res_short['feature_importance'],
                'threshold_f1': res_short['threshold_f1'],
                'threshold_precision': res_short['threshold_precision'],
            },
        }, f, indent=2, default=float)
    print(f"\n  ✅ Métriques JSON: {metrics_path}")

    # Synthèse comparative
    print(f"\n{'=' * 100}")
    print("SYNTHÈSE — Test set (out-of-sample)")
    print(f"{'=' * 100}")
    for direction, res in [('LONG', res_long), ('SHORT', res_short)]:
        m = res['metrics_f1']['test']
        m_p = res['metrics_precision']['test']
        print(f"\n  {direction}:")
        print(f"    AUC test         : {m['auc']:.4f}")
        print(f"    Base rate (test) : {m['n_pos']/m['n']*100:.2f}%")
        print(f"    F1-optimal thr   : {res['threshold_f1']:.3f}  "
              f"→ Prec={m['prec']*100:.2f}%  Rec={m['rec']*100:.2f}%  F1={m['f1']*100:.2f}%")
        print(f"    High-precision thr: {res['threshold_precision']:.3f}  "
              f"→ Prec={m_p['prec']*100:.2f}%  Rec={m_p['rec']*100:.2f}%")

    print(f"\n  → Si AUC test > 0.60 → piste validée, étape 2.D = backtest")
    print(f"  → Si AUC test < 0.55 → pas assez de signal, pivoter")


if __name__ == '__main__':
    main()
