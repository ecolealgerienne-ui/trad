"""
diagnose_label_separability.py — Diagnostic features vs labels.

Question centrale: les features portent-elles l'info qui permet à l'Oracle
(+198%/an sur test) de distinguer Label=1 (TP) de Label=0 (SL) ?

3 analyses:
1. Per channel × per timestep: mean/std comparison, KS test, Cohen's d
2. Univariate AUC per channel × timestep
3. Baseline logistic regression sur features à l'event (dernier timestep)

3 verdicts possibles:
- Aucune feature discriminante (Cohen's d < 0.1, KS < 0.05, univariate AUC < 0.52)
  → Info pas dans les features individuelles → architecture/data shift problème
- Features discriminantes (Cohen's d > 0.2 ou univariate AUC > 0.55)
  → Info présente, modèle échoue à l'extraire → architecture inadaptée
- Logistic baseline AUC > PatchTST AUC
  → Le modèle complexe overfit, modèle simple suffit

Usage:
    python -m experiments.patchtst_v5.diagnose_label_separability \\
        --data-dir data/patchtst_v5_rr2_v54_indicators/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

logger = logging.getLogger("patchtst_v5.diagnose_label_separability")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_split(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=False)
    return data["X"].astype("float32"), data["y"].astype("int8")


def load_channel_names(metadata_path: Path) -> list[str]:
    metadata = json.loads(metadata_path.read_text())
    return metadata["channels"]


# ---------------------------------------------------------------------------
# Analyse 1: per (channel, timestep) — séparabilité des distributions
# ---------------------------------------------------------------------------

def per_feature_separability(X: np.ndarray, y: np.ndarray,
                              channel_names: list[str]) -> pd.DataFrame:
    """Pour chaque (channel, timestep), comparer Label=1 vs Label=0."""
    n, T, C = X.shape
    pos_mask = (y == 1)
    neg_mask = (y == 0)

    if pos_mask.sum() < 100 or neg_mask.sum() < 100:
        raise ValueError("Pas assez d'événements par classe (besoin ≥100 chacune)")

    results = []
    for c in range(C):
        for t in range(T):
            vals_pos = X[pos_mask, t, c]
            vals_neg = X[neg_mask, t, c]

            # Skip features sans variance
            std_pos = vals_pos.std()
            std_neg = vals_neg.std()
            if std_pos < 1e-9 and std_neg < 1e-9:
                continue

            mean_pos = vals_pos.mean()
            mean_neg = vals_neg.mean()
            mean_diff = mean_pos - mean_neg

            # KS test
            try:
                ks_stat, ks_p = ks_2samp(vals_pos, vals_neg)
            except Exception:
                continue

            # Cohen's d (standardized effect size)
            pooled_std = np.sqrt((std_pos ** 2 + std_neg ** 2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 1e-9 else 0.0

            results.append({
                "channel": channel_names[c],
                "timestep": t,
                "mean_pos": mean_pos,
                "mean_neg": mean_neg,
                "mean_diff": mean_diff,
                "cohens_d": cohens_d,
                "abs_cohens_d": abs(cohens_d),
                "ks_stat": ks_stat,
                "ks_p": ks_p,
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Analyse 2: AUC univariée per (channel, timestep)
# ---------------------------------------------------------------------------

def univariate_auc(X: np.ndarray, y: np.ndarray, channel_names: list[str]) -> pd.DataFrame:
    """AUC univariée pour chaque (channel, timestep). AUC > 0.55 = discriminant."""
    n, T, C = X.shape
    results = []
    for c in range(C):
        for t in range(T):
            try:
                values = X[:, t, c]
                if np.std(values) < 1e-9:
                    continue
                auc = roc_auc_score(y, values)
                # Normaliser: AUC < 0.5 = anti-corrélé, on prend la version "informative"
                discriminative_auc = max(auc, 1 - auc)
                results.append({
                    "channel": channel_names[c],
                    "timestep": t,
                    "raw_auc": auc,
                    "discriminative_auc": discriminative_auc,
                    "is_anti_correlated": auc < 0.5,
                })
            except Exception:
                continue
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Analyse 3: Logistic regression baseline sur features à l'event
# ---------------------------------------------------------------------------

def baseline_logistic(X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      channel_names: list[str]) -> dict:
    """Régression logistique sur features à l'event time (dernier timestep)."""
    # Last timestep only
    X_tr = X_train[:, -1, :]
    X_va = X_val[:, -1, :]
    X_te = X_test[:, -1, :]

    # Standardize (fit train only)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    # L2 logistic with class_weight balanced
    model = LogisticRegression(
        max_iter=2000, class_weight="balanced", C=1.0, solver="lbfgs"
    ).fit(X_tr_s, y_train)

    train_auc = roc_auc_score(y_train, model.predict_proba(X_tr_s)[:, 1])
    val_auc = roc_auc_score(y_val, model.predict_proba(X_va_s)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_te_s)[:, 1])

    # Top-K precision
    test_scores = model.predict_proba(X_te_s)[:, 1]
    sorted_idx = np.argsort(-test_scores)
    top_k = {}
    for k in [1, 5, 10, 25]:
        n_top = max(1, int(len(y_test) * k / 100))
        top_k[f"precision_top_{k}pct"] = float(y_test[sorted_idx[:n_top]].mean())

    coef_df = pd.DataFrame({
        "channel": channel_names,
        "coefficient": model.coef_[0],
        "abs_coef": np.abs(model.coef_[0]),
    }).sort_values("abs_coef", ascending=False)

    return {
        "train_auc": train_auc,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "top_k_precision": top_k,
        "coefficients": coef_df,
    }


# ---------------------------------------------------------------------------
# Sweep multi-timestep logistic (pour voir si timesteps profonds aident)
# ---------------------------------------------------------------------------

def sweep_logistic_timesteps(X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray,
                              channel_names: list[str]) -> pd.DataFrame:
    """Logistic AUC en utilisant features de plus en plus de timesteps avant l'event.
    Permet de voir si le signal est dans le récent ou diffus dans le passé."""
    n, T, C = X_train.shape
    results = []
    for n_steps in [1, 6, 12, 24, 48, 96]:
        if n_steps > T:
            continue
        # Use last n_steps timesteps, flatten
        X_tr = X_train[:, -n_steps:, :].reshape(len(X_train), -1)
        X_va = X_val[:, -n_steps:, :].reshape(len(X_val), -1)

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_va_s = scaler.transform(X_va)

        model = LogisticRegression(max_iter=1500, class_weight="balanced",
                                    C=0.1, solver="lbfgs").fit(X_tr_s, y_train)
        train_auc = roc_auc_score(y_train, model.predict_proba(X_tr_s)[:, 1])
        val_auc = roc_auc_score(y_val, model.predict_proba(X_va_s)[:, 1])
        results.append({
            "n_timesteps": n_steps,
            "n_features_total": n_steps * C,
            "train_auc": train_auc,
            "val_auc": val_auc,
            "gap": train_auc - val_auc,
        })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summary_per_channel(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Synthèse par channel: max effet observé sur tous timesteps."""
    summary = detail_df.groupby("channel").agg(
        max_abs_cohens_d=("abs_cohens_d", "max"),
        max_ks_stat=("ks_stat", "max"),
        max_abs_mean_diff=("mean_diff", lambda x: x.abs().max()),
        n_significant_ks=("ks_p", lambda x: (x < 0.001).sum()),
    ).reset_index().sort_values("max_abs_cohens_d", ascending=False)
    return summary


def summary_univariate(univariate_df: pd.DataFrame) -> pd.DataFrame:
    """Synthèse AUC univariée par channel."""
    return univariate_df.groupby("channel").agg(
        max_discriminative_auc=("discriminative_auc", "max"),
        mean_discriminative_auc=("discriminative_auc", "mean"),
    ).reset_index().sort_values("max_discriminative_auc", ascending=False)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--metadata", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    metadata_path = args.metadata or (args.data_dir / "dataset_metadata.json")
    channel_names = load_channel_names(metadata_path)
    logger.info("Channels: %d", len(channel_names))

    X_train, y_train = load_split(args.data_dir / "train.npz")
    X_val, y_val = load_split(args.data_dir / "val.npz")
    X_test, y_test = load_split(args.data_dir / "test.npz")
    logger.info("Train: %d (Class1=%.1f%%) | Val: %d | Test: %d",
                len(y_train), 100 * y_train.mean(), len(y_val), len(y_test))

    # Analyse 1: per (channel, timestep) separability sur train
    logger.info("Computing per (channel, timestep) separability stats ...")
    sep_df = per_feature_separability(X_train, y_train, channel_names)
    sep_summary = summary_per_channel(sep_df)

    logger.info("=" * 110)
    logger.info("1. SÉPARABILITÉ PAR CHANNEL (max sur tous les timesteps)")
    logger.info("=" * 110)
    fmt = sep_summary.copy()
    for col in ["max_abs_cohens_d", "max_ks_stat", "max_abs_mean_diff"]:
        fmt[col] = fmt[col].round(4)
    logger.info(fmt.to_string(index=False))
    logger.info("")

    # Top discriminating (channel, timestep) cells
    logger.info("=" * 110)
    logger.info("TOP %d (channel × timestep) les plus discriminants (par |Cohen's d|)", args.top_k)
    logger.info("=" * 110)
    top_cells = sep_df.nlargest(args.top_k, "abs_cohens_d")[
        ["channel", "timestep", "mean_pos", "mean_neg", "cohens_d", "ks_stat", "ks_p"]
    ].copy()
    for col in ["mean_pos", "mean_neg", "cohens_d", "ks_stat", "ks_p"]:
        top_cells[col] = top_cells[col].round(5)
    logger.info(top_cells.to_string(index=False))
    logger.info("")

    # Analyse 2: AUC univariée
    logger.info("Computing univariate AUC ...")
    auc_df = univariate_auc(X_train, y_train, channel_names)
    auc_summary = summary_univariate(auc_df)
    logger.info("=" * 110)
    logger.info("2. AUC UNIVARIÉE PAR CHANNEL (max et mean sur timesteps)")
    logger.info("=" * 110)
    fmt = auc_summary.copy()
    for col in ["max_discriminative_auc", "mean_discriminative_auc"]:
        fmt[col] = fmt[col].round(4)
    logger.info(fmt.to_string(index=False))
    logger.info("")

    # Analyse 3: Logistic regression baseline
    logger.info("Fitting baseline logistic regression on event-time features ...")
    logit_result = baseline_logistic(X_train, y_train, X_val, y_val, X_test, y_test, channel_names)
    logger.info("=" * 110)
    logger.info("3. LOGISTIC REGRESSION BASELINE (last timestep features uniquement)")
    logger.info("=" * 110)
    logger.info("Train AUC: %.4f | Val AUC: %.4f | Test AUC: %.4f",
                logit_result["train_auc"], logit_result["val_auc"], logit_result["test_auc"])
    logger.info("Top-K precision sur test:")
    for k, v in logit_result["top_k_precision"].items():
        logger.info("  %s : %.4f", k, v)
    logger.info("")
    logger.info("Coefficients (top 10 par |coef|):")
    coef_top = logit_result["coefficients"].head(10).copy()
    coef_top["coefficient"] = coef_top["coefficient"].round(4)
    coef_top["abs_coef"] = coef_top["abs_coef"].round(4)
    logger.info(coef_top.to_string(index=False))
    logger.info("")

    # Sweep multi-timesteps
    logger.info("Sweeping logistic with increasing temporal context ...")
    sweep_df = sweep_logistic_timesteps(X_train, y_train, X_val, y_val, channel_names)
    logger.info("=" * 110)
    logger.info("4. LOGISTIC AUC vs CONTEXTE TEMPOREL")
    logger.info("=" * 110)
    fmt = sweep_df.copy()
    for col in ["train_auc", "val_auc", "gap"]:
        fmt[col] = fmt[col].round(4)
    logger.info(fmt.to_string(index=False))
    logger.info("")

    # Synthèse / verdict
    logger.info("=" * 110)
    logger.info("SYNTHESE — VERDICT DIAGNOSTIQUE")
    logger.info("=" * 110)
    max_d = sep_summary["max_abs_cohens_d"].max()
    max_auc = auc_summary["max_discriminative_auc"].max()
    n_sig_ks = sep_df["ks_p"].lt(0.001).sum()
    n_total = len(sep_df)
    logit_test_auc = logit_result["test_auc"]

    logger.info("Max |Cohen's d| observé        : %.3f", max_d)
    logger.info("Max AUC univariée observée    : %.3f", max_auc)
    logger.info("Cellules avec KS p<0.001       : %d / %d (%.1f%%)",
                n_sig_ks, n_total, 100 * n_sig_ks / n_total)
    logger.info("Logistic test AUC              : %.4f", logit_test_auc)
    logger.info("")

    if max_d < 0.1 and max_auc < 0.52 and logit_test_auc < 0.52:
        logger.info("⚠️  HYPOTHÈSE A — Les features ne portent AUCUN signal marginal")
        logger.info("    discriminant entre Label=1 et Label=0.")
        logger.info("    → Architecture/data shift n'est PAS la cause.")
        logger.info("    → L'info nécessaire pour battre l'Oracle n'est pas dans ces features.")
    elif max_d > 0.2 or max_auc > 0.55:
        logger.info("✅ HYPOTHÈSE B — Les features portent du signal discriminant.")
        logger.info("    → Si PatchTST AUC reste à 0.50, c'est un problème d'architecture/training.")
        logger.info("    → Tester XGBoost, MLP simple, ou augmenter capacité PatchTST.")
    else:
        logger.info("⚠️  ZONE GRISE — Signal marginal très faible (Cohen's d ∈ [0.1, 0.2]).")
        logger.info("    → Possiblement signal dans interactions non-linéaires (hypothèse C).")
        logger.info("    → Tester modèles non-linéaires (XGBoost, gradient boosting).")
    logger.info("=" * 110)

    # Save JSON
    output_path = args.output or (args.data_dir / "label_separability_diagnostic.json")
    out = {
        "n_train": int(len(y_train)),
        "class_1_ratio_train": float(y_train.mean()),
        "summary_per_channel": sep_summary.to_dict(orient="records"),
        "top_discriminating_cells": top_cells.to_dict(orient="records"),
        "univariate_auc_summary": auc_summary.to_dict(orient="records"),
        "logistic_baseline": {
            "train_auc": logit_result["train_auc"],
            "val_auc": logit_result["val_auc"],
            "test_auc": logit_result["test_auc"],
            "top_k_precision": logit_result["top_k_precision"],
            "coefficients": logit_result["coefficients"].to_dict(orient="records"),
        },
        "logistic_sweep_temporal": sweep_df.to_dict(orient="records"),
    }
    output_path.write_text(json.dumps(out, indent=2, default=str))
    logger.info("JSON saved: %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
