"""Train meta-classifier sur dataset Triple Barrier (Étape 3).

Pipeline :
  1. Charge meta dataset (X_meta_train/test, y_meta_train/test, feature_names)
  2. Filtre les samples avec direction = 0 (pas de trade) — ils n'ont pas de label TB pertinent
  3. Train Logistic Regression baseline avec class_weight='balanced'
     → Lit les coefficients (interprétabilité)
  4. Train XGBoost avec scale_pos_weight pour imbalance
     → Feature importances + (optionnel) SHAP values
  5. Sauve modèles + scores + importances

Usage :
    python experiments/foundation_finetune/train_meta_classifier.py \\
        --data data/foundation/meta_btc_close_kalman.npz \\
        --output-dir models/meta_classifier
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def metrics_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    """Precision / Recall / F1 / Coverage à un seuil donné."""
    y_pred = (y_proba >= threshold).astype(np.int8)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    n_pred_pos = tp + fp
    n_pos = tp + fn
    n = len(y_true)
    precision = tp / max(n_pred_pos, 1)
    recall = tp / max(n_pos, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "n_pred_pos": n_pred_pos,
        "coverage_pct": float(n_pred_pos / n * 100),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def coverage_precision_table(y_true: np.ndarray, y_proba: np.ndarray,
                               coverages: list) -> list:
    """Pour chaque coverage cible, trouve le seuil qui sélectionne ce % de top
    et retourne les métriques."""
    n = len(y_true)
    rows = []
    for cov in coverages:
        k = max(1, int(n * cov / 100))
        # Top-k par proba descendante
        idx_sorted = np.argsort(y_proba)[::-1]
        top_idx = idx_sorted[:k]
        threshold = float(y_proba[idx_sorted[k - 1]])
        y_pred = np.zeros_like(y_true)
        y_pred[top_idx] = 1
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = k - tp
        recall = tp / max(int(y_true.sum()), 1)
        precision = tp / max(k, 1)
        rows.append({
            "coverage_pct": cov,
            "n_selected": k,
            "threshold": threshold,
            "precision": float(precision),
            "recall": float(recall),
            "tp": tp, "fp": fp,
        })
    return rows


def stratified_metrics(y_true: np.ndarray, y_proba: np.ndarray,
                        regime: np.ndarray, threshold: float) -> dict:
    """Précision/recall stratifié par régime à un seuil donné."""
    out = {}
    for r_id, r_name in [(0, "RANGE_LOW_VOL"), (1, "RANGE_HIGH_VOL"), (2, "TREND")]:
        mask = regime == r_id
        if mask.sum() == 0:
            continue
        m = metrics_at_threshold(y_true[mask], y_proba[mask], threshold)
        m["n_total"] = int(mask.sum())
        out[r_name] = m
    return out


def fmt_table(rows: list, columns: list, label: str = "") -> str:
    if label:
        out = [f"\n--- {label} ---"]
    else:
        out = []
    headers = columns
    widths = [max(len(h), 12) for h in headers]
    out.append("  ".join(f"{h:>{w}}" for h, w in zip(headers, widths)))
    out.append("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows:
        out.append("  ".join(
            f"{row.get(h, ''):>{w}.4f}" if isinstance(row.get(h), float) else f"{str(row.get(h, '')):>{w}}"
            for h, w in zip(headers, widths)
        ))
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True, help="Meta dataset npz")
    parser.add_argument("--output-dir", default=str(ROOT / "models" / "meta_classifier"))
    parser.add_argument("--use-xgboost", action="store_true", help="Entraîne aussi XGBoost (lent)")
    parser.add_argument("--xgb-rounds", type=int, default=200)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-eta", type=float, default=0.1)
    args = parser.parse_args()

    print(f"Loading {args.data}...")
    data = np.load(args.data, allow_pickle=True)
    feature_names = list(data["feature_names"])

    X_train = data["X_meta_train"]
    y_train = data["y_meta_train"]
    X_test = data["X_meta_test"]
    y_test = data["y_meta_test"]
    direction_train = data["mean_direction_train"]
    direction_test = data["mean_direction_test"]
    regime_train = data["regime_train"]
    regime_test = data["regime_test"]

    print(f"  Train: {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
    print(f"  Test : {X_test.shape[0]:,} samples")

    # Filtre direction != 0 (pas de trade = pas de label TB significatif)
    mask_train = direction_train != 0
    mask_test = direction_test != 0
    X_train_f = X_train[mask_train]
    y_train_f = y_train[mask_train]
    X_test_f = X_test[mask_test]
    y_test_f = y_test[mask_test]
    regime_test_f = regime_test[mask_test]

    print(f"  Direction != 0: train {mask_train.sum():,}/{len(mask_train):,}, "
          f"test {mask_test.sum():,}/{len(mask_test):,}")
    print(f"  TB positive rate: train {y_train_f.mean():.3f}, test {y_test_f.mean():.3f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"feature_names": feature_names, "models": {}}

    # =========================================================================
    # MODEL 1: Logistic Regression baseline
    # =========================================================================
    print("\n=== Logistic Regression (baseline) ===")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_f)
    X_test_scaled = scaler.transform(X_test_f)

    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        C=1.0,
        random_state=42,
    )
    lr.fit(X_train_scaled, y_train_f)
    proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

    print("  Coefficients (sorted by abs value) :")
    coefs = lr.coef_[0]
    coef_pairs = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
    for name, c in coef_pairs:
        bar = "█" * min(int(abs(c) * 20), 30)
        sign = "+" if c >= 0 else "-"
        print(f"    {name:<24} {sign}{abs(c):.4f}  {bar}")
    print(f"  Intercept: {lr.intercept_[0]:+.4f}")

    # Coverage-precision curves
    coverages = [1, 2, 5, 10, 20, 30, 50, 70, 100]
    cov_table_lr = coverage_precision_table(y_test_f, proba_lr, coverages)
    print("\n  Coverage-Precision table (test):")
    print(fmt_table(cov_table_lr,
                    ["coverage_pct", "n_selected", "threshold", "precision", "recall", "tp", "fp"]))

    # Stratification par régime au seuil 0.7 (haute précision attendue)
    cov_for_strata = 10
    k_strata = max(1, int(len(y_test_f) * cov_for_strata / 100))
    threshold_strata = float(np.sort(proba_lr)[::-1][k_strata - 1])
    strata_lr = stratified_metrics(y_test_f, proba_lr, regime_test_f, threshold_strata)
    print(f"\n  Stratification par régime @ top {cov_for_strata}% (threshold={threshold_strata:.4f}):")
    for r_name, m in strata_lr.items():
        print(f"    {r_name:<18} N={m['n_total']:>6}  "
              f"selected={m['n_pred_pos']:>5}  "
              f"prec={m['precision']:.4f}  rec={m['recall']:.4f}")

    # Save LR
    import pickle
    lr_path = output_dir / "logistic_regression.pkl"
    with open(lr_path, "wb") as f:
        pickle.dump({"model": lr, "scaler": scaler,
                     "feature_names": feature_names}, f)
    print(f"  Saved → {lr_path}")

    summary["models"]["logistic_regression"] = {
        "coefficients": {n: float(c) for n, c in zip(feature_names, coefs)},
        "intercept": float(lr.intercept_[0]),
        "coverage_precision": cov_table_lr,
        "stratified_top_10pct": {n: m for n, m in strata_lr.items()},
    }

    # =========================================================================
    # MODEL 2: XGBoost (optional)
    # =========================================================================
    if args.use_xgboost:
        print("\n=== XGBoost ===")
        try:
            import xgboost as xgb
        except ImportError:
            print("  XGBoost not installed (pip install xgboost). Skipping.")
        else:
            n_pos = int(y_train_f.sum())
            n_neg = len(y_train_f) - n_pos
            scale_pos_weight = n_neg / max(n_pos, 1)
            print(f"  scale_pos_weight = {scale_pos_weight:.3f}")

            dtrain = xgb.DMatrix(X_train_f, label=y_train_f, feature_names=feature_names)
            dtest = xgb.DMatrix(X_test_f, label=y_test_f, feature_names=feature_names)
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "max_depth": args.xgb_max_depth,
                "eta": args.xgb_eta,
                "scale_pos_weight": scale_pos_weight,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "seed": 42,
                "verbosity": 1,
            }
            bst = xgb.train(
                params, dtrain, num_boost_round=args.xgb_rounds,
                evals=[(dtrain, "train"), (dtest, "test")],
                verbose_eval=20,
            )
            proba_xgb = bst.predict(dtest)

            # Feature importances (gain)
            importance = bst.get_score(importance_type="gain")
            imp_pairs = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            print("\n  Feature importances (gain) :")
            for name, score in imp_pairs:
                bar = "█" * min(int(score / max(imp_pairs[0][1], 1) * 30), 30)
                print(f"    {name:<24} {score:>10.2f}  {bar}")

            cov_table_xgb = coverage_precision_table(y_test_f, proba_xgb, coverages)
            print("\n  Coverage-Precision table (test):")
            print(fmt_table(cov_table_xgb,
                            ["coverage_pct", "n_selected", "threshold", "precision", "recall", "tp", "fp"]))

            threshold_strata_xgb = float(np.sort(proba_xgb)[::-1][k_strata - 1])
            strata_xgb = stratified_metrics(y_test_f, proba_xgb, regime_test_f, threshold_strata_xgb)
            print(f"\n  Stratification par régime @ top {cov_for_strata}% (threshold={threshold_strata_xgb:.4f}):")
            for r_name, m in strata_xgb.items():
                print(f"    {r_name:<18} N={m['n_total']:>6}  "
                      f"selected={m['n_pred_pos']:>5}  "
                      f"prec={m['precision']:.4f}  rec={m['recall']:.4f}")

            xgb_path = output_dir / "xgboost.json"
            bst.save_model(str(xgb_path))
            print(f"  Saved → {xgb_path}")

            summary["models"]["xgboost"] = {
                "params": params,
                "num_rounds": args.xgb_rounds,
                "scale_pos_weight": float(scale_pos_weight),
                "importance_gain": {n: float(s) for n, s in imp_pairs},
                "coverage_precision": cov_table_xgb,
                "stratified_top_10pct": {n: m for n, m in strata_xgb.items()},
            }

    # Save summary
    summary["created_at"] = datetime.now(timezone.utc).isoformat()
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n✓ Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
