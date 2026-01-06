#!/usr/bin/env python3
"""
Analyse des Améliorations du Stacking par Indicateur

Objectif: Comparer les prédictions de chaque indicateur AVANT et APRÈS Stacking.

Questions:
1. Le Stacking améliore-t-il RSI/CCI (baseline 87.4%/89.3%) ?
2. Sur QUELS samples le Stacking corrige les erreurs ?
3. POURQUOI ces samples ? (patterns détectés par MACD que RSI/CCI ratent)

Vision: Utiliser ces insights pour le relabeling intelligent.

Usage:
  python src/analyze_stacking_improvements.py --model logistic --split test
"""

import sys
import numpy as np
from pathlib import Path
import logging
import argparse
from typing import Dict, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DATASET_PATHS = {
    'macd': 'data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_kalman.npz',
    'rsi': 'data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_dual_binary_kalman.npz',
    'cci': 'data/prepared/dataset_btc_eth_bnb_ada_ltc_cci_dual_binary_kalman.npz',
}

BASELINE_ACCURACIES = {
    'macd': 92.4,
    'rsi': 87.4,
    'cci': 89.3,
}


def load_predictions_from_npz(split: str = 'train') -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Charge les prédictions des 3 modèles depuis les .npz.

    Returns:
        X_meta: (n, 6) - Prédictions [macd_dir, macd_force, rsi_dir, rsi_force, cci_dir, cci_force]
        Y_meta: (n, 1) - Direction Kalman
        raw_preds: dict avec Y_pred de chaque indicateur
    """
    logger.info(f"📂 Chargement prédictions split '{split}'...")

    predictions = {}
    raw_preds = {}
    Y_meta = None

    for indicator in ['macd', 'rsi', 'cci']:
        path = DATASET_PATHS[indicator]
        data = np.load(path, allow_pickle=True)

        Y_pred = data[f'Y_{split}_pred']  # Shape: (n, 2) - [direction, force]
        Y = data[f'Y_{split}']            # Shape: (n, 2) - [direction, force]

        predictions[indicator] = Y_pred
        raw_preds[indicator] = {
            'Y_pred': Y_pred,
            'Y_true': Y,
        }

        if Y_meta is None:
            Y_meta = Y[:, 0:1]  # Direction uniquement

    # Concaténer prédictions (6 features)
    X_meta = np.concatenate([
        predictions['macd'],
        predictions['rsi'],
        predictions['cci'],
    ], axis=1)

    return X_meta, Y_meta, raw_preds


def train_stacking_model(X_train, Y_train, model_type='logistic'):
    """Entraîne le meta-modèle."""
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Model type inconnu: {model_type}")

    model.fit(X_train, Y_train.ravel())
    return model


def analyze_individual_improvements(
    meta_model,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    raw_preds: Dict
) -> Dict:
    """
    Analyse les améliorations par indicateur.

    Compare:
    - Baseline: Y_pred de l'indicateur seul (déjà calculé)
    - Stacking: Prédiction du meta-modèle EN SE CONCENTRANT sur cet indicateur

    Args:
        meta_model: Modèle Stacking entraîné
        X_test: Méta-features (n, 6)
        Y_test: Vérité terrain (n, 1)
        raw_preds: Prédictions baseline de chaque indicateur

    Returns:
        dict avec résultats par indicateur
    """
    results = {}

    # Prédictions Stacking globales
    Y_stacking_pred = meta_model.predict(X_test)
    stacking_acc = accuracy_score(Y_test.ravel(), Y_stacking_pred) * 100

    logger.info(f"\n" + "="*80)
    logger.info(f"📊 COMPARAISON BASELINE vs STACKING")
    logger.info(f"="*80)

    for i, indicator in enumerate(['macd', 'rsi', 'cci']):
        # Baseline: prédictions de l'indicateur seul
        Y_baseline_pred_proba = raw_preds[indicator]['Y_pred'][:, 0]  # Direction probabilités
        Y_baseline_pred = (Y_baseline_pred_proba > 0.5).astype(int)  # Convertir en binaire
        Y_true = raw_preds[indicator]['Y_true'][:, 0].astype(int)

        baseline_acc = accuracy_score(Y_true, Y_baseline_pred) * 100

        # Analyser où Stacking corrige les erreurs de baseline
        baseline_errors = (Y_baseline_pred != Y_true)
        stacking_correct = (Y_stacking_pred == Y_true.ravel())

        # Samples où Stacking corrige baseline
        corrected_by_stacking = baseline_errors & stacking_correct
        n_corrected = corrected_by_stacking.sum()
        pct_corrected = (n_corrected / baseline_errors.sum()) * 100 if baseline_errors.sum() > 0 else 0

        # Samples où Stacking fait de nouvelles erreurs
        baseline_correct = ~baseline_errors
        stacking_errors = ~stacking_correct
        new_errors = baseline_correct & stacking_errors
        n_new_errors = new_errors.sum()

        # Gain net
        net_gain = n_corrected - n_new_errors
        net_gain_pct = (net_gain / len(Y_true)) * 100

        results[indicator] = {
            'baseline_acc': baseline_acc,
            'stacking_acc': stacking_acc,
            'delta': stacking_acc - baseline_acc,
            'n_corrected': n_corrected,
            'pct_corrected': pct_corrected,
            'n_new_errors': n_new_errors,
            'net_gain': net_gain,
            'net_gain_pct': net_gain_pct,
            'corrected_indices': np.where(corrected_by_stacking)[0],
        }

        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 {indicator.upper()}")
        logger.info(f"{'='*80}")
        logger.info(f"   Baseline Accuracy:  {baseline_acc:.2f}%")
        logger.info(f"   Stacking Accuracy:  {stacking_acc:.2f}%")
        logger.info(f"   Delta:              {stacking_acc - baseline_acc:+.2f}%")
        logger.info(f"\n   Erreurs Baseline corrigées par Stacking:")
        logger.info(f"      Total corrigé:   {n_corrected:,} samples ({pct_corrected:.1f}% des erreurs)")
        logger.info(f"      Nouvelles erreurs: {n_new_errors:,} samples")
        logger.info(f"      Gain net:        {net_gain:+,} samples ({net_gain_pct:+.3f}%)")

        # Verdict
        if net_gain_pct > 0.5:
            verdict = "✅ AMÉLIORATION SIGNIFICATIVE"
        elif net_gain_pct > 0:
            verdict = "✅ Légère amélioration"
        elif net_gain_pct == 0:
            verdict = "⚪ Neutre"
        else:
            verdict = "❌ Dégradation"

        logger.info(f"\n   Verdict: {verdict}")

    return results


def analyze_correction_patterns(
    results: Dict,
    X_test: np.ndarray,
    raw_preds: Dict,
    indicator: str
):
    """
    Analyse POURQUOI le Stacking corrige les erreurs de cet indicateur.

    Hypothèse: MACD détecte des patterns que RSI/CCI ratent.
    """
    if results[indicator]['n_corrected'] == 0:
        logger.info(f"\n⚠️  Aucune correction pour {indicator.upper()} - Analyse impossible")
        return

    logger.info(f"\n" + "="*80)
    logger.info(f"🔬 ANALYSE DES PATTERNS - {indicator.upper()}")
    logger.info(f"="*80)

    corrected_idx = results[indicator]['corrected_indices']

    # Extraire features des samples corrigés
    X_corrected = X_test[corrected_idx]

    # Analyser les prédictions MACD/RSI/CCI sur ces samples
    macd_dir = X_corrected[:, 0]  # MACD Direction
    rsi_dir = X_corrected[:, 2]   # RSI Direction
    cci_dir = X_corrected[:, 4]   # CCI Direction

    macd_force = X_corrected[:, 1]
    rsi_force = X_corrected[:, 3]
    cci_force = X_corrected[:, 5]

    logger.info(f"\n📊 Caractéristiques des {len(corrected_idx):,} samples corrigés:")

    # Pattern 1: Accord MACD avec les autres
    if indicator == 'rsi':
        macd_agrees = (macd_dir == cci_dir)
        pct_macd_cci_agree = (macd_agrees.sum() / len(corrected_idx)) * 100
        logger.info(f"\n   Pattern 1: MACD + CCI d'accord (corrigent RSI)")
        logger.info(f"      Fréquence: {pct_macd_cci_agree:.1f}%")

    elif indicator == 'cci':
        macd_agrees = (macd_dir == rsi_dir)
        pct_macd_rsi_agree = (macd_agrees.sum() / len(corrected_idx)) * 100
        logger.info(f"\n   Pattern 1: MACD + RSI d'accord (corrigent CCI)")
        logger.info(f"      Fréquence: {pct_macd_rsi_agree:.1f}%")

    elif indicator == 'macd':
        rsi_cci_agrees = (rsi_dir == cci_dir)
        pct_rsi_cci_agree = (rsi_cci_agrees.sum() / len(corrected_idx)) * 100
        logger.info(f"\n   Pattern 1: RSI + CCI d'accord (corrigent MACD)")
        logger.info(f"      Fréquence: {pct_rsi_cci_agree:.1f}%")

    # Pattern 2: Force des signaux
    target_force = X_corrected[:, [1, 3, 5][['macd', 'rsi', 'cci'].index(indicator)]]
    strong_signals = (target_force > 0.5).sum()
    pct_strong = (strong_signals / len(corrected_idx)) * 100

    logger.info(f"\n   Pattern 2: Force du signal {indicator.upper()}")
    logger.info(f"      STRONG: {pct_strong:.1f}%")
    logger.info(f"      WEAK:   {100-pct_strong:.1f}%")

    # Pattern 3: Consensus des 3 indicateurs
    all_agree = (macd_dir == rsi_dir) & (rsi_dir == cci_dir)
    pct_consensus = (all_agree.sum() / len(corrected_idx)) * 100

    logger.info(f"\n   Pattern 3: Consensus 3 indicateurs")
    logger.info(f"      Tous d'accord: {pct_consensus:.1f}%")
    logger.info(f"      Désaccord:     {100-pct_consensus:.1f}%")

    # Insight pour relabeling
    logger.info(f"\n💡 INSIGHT POUR RELABELING:")

    if indicator == 'rsi' and pct_macd_cci_agree > 70:
        logger.info(f"   → RSI fait des erreurs quand MACD+CCI sont d'accord ({pct_macd_cci_agree:.1f}%)")
        logger.info(f"   → Relabeling: Utiliser (MACD+CCI) comme vérité pour corriger labels RSI")

    elif indicator == 'cci' and pct_macd_rsi_agree > 70:
        logger.info(f"   → CCI fait des erreurs quand MACD+RSI sont d'accord ({pct_macd_rsi_agree:.1f}%)")
        logger.info(f"   → Relabeling: Utiliser (MACD+RSI) comme vérité pour corriger labels CCI")

    elif indicator == 'macd':
        logger.info(f"   → MACD déjà excellent (92.4%) - Stacking n'améliore pas significativement")

    if pct_strong < 40:
        logger.info(f"   → Les erreurs surviennent surtout sur signaux WEAK ({100-pct_strong:.1f}%)")
        logger.info(f"   → Relabeling: Filtrer/relabeler les WEAK discordants")


def main():
    parser = argparse.ArgumentParser(description='Analyse Stacking - Améliorations par Indicateur')
    parser.add_argument(
        '--model',
        type=str,
        choices=['logistic', 'rf'],
        default='logistic',
        help="Modèle Stacking à analyser (défaut: logistic)"
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test'],
        default='test',
        help="Split à analyser (défaut: test)"
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("🔬 ANALYSE STACKING - Améliorations par Indicateur")
    logger.info("="*80)
    logger.info(f"\nModèle: {args.model}")
    logger.info(f"Split: {args.split}")

    # Charger données
    X_train, Y_train, _ = load_predictions_from_npz('train')
    X_test, Y_test, raw_preds_test = load_predictions_from_npz(args.split)

    # Entraîner meta-modèle
    logger.info(f"\n⏳ Entraînement {args.model}...")
    meta_model = train_stacking_model(X_train, Y_train, args.model)

    # Analyser améliorations
    results = analyze_individual_improvements(meta_model, X_test, Y_test, raw_preds_test)

    # Analyser patterns de correction pour RSI et CCI
    logger.info(f"\n" + "="*80)
    logger.info(f"🔬 ANALYSE DES PATTERNS DE CORRECTION")
    logger.info(f"="*80)

    for indicator in ['rsi', 'cci', 'macd']:
        if results[indicator]['n_corrected'] > 0:
            analyze_correction_patterns(results, X_test, raw_preds_test, indicator)

    # Résumé final
    logger.info(f"\n" + "="*80)
    logger.info(f"📋 RÉSUMÉ FINAL")
    logger.info(f"="*80)

    for indicator in ['macd', 'rsi', 'cci']:
        delta = results[indicator]['delta']
        net_gain_pct = results[indicator]['net_gain_pct']

        if delta > 0.5:
            status = "🏆 AMÉLIORATION"
        elif delta > 0:
            status = "✅ Légère amélioration"
        elif delta == 0:
            status = "⚪ Neutre"
        else:
            status = "❌ Dégradation"

        logger.info(f"\n{indicator.upper():4s}: Baseline {results[indicator]['baseline_acc']:.2f}% → Stacking {results[indicator]['stacking_acc']:.2f}% ({delta:+.2f}%) {status}")

    logger.info(f"\n" + "="*80)
    logger.info(f"🎯 RECOMMANDATIONS POUR RELABELING")
    logger.info(f"="*80)

    # Recommandations basées sur les résultats
    best_improvement = max(results.items(), key=lambda x: x[1]['net_gain_pct'])
    indicator_improved, stats = best_improvement

    if stats['net_gain_pct'] > 0.5:
        logger.info(f"\n✅ {indicator_improved.upper()} montre le meilleur potentiel d'amélioration (+{stats['net_gain_pct']:.3f}%)")
        logger.info(f"\n📌 Stratégie de Relabeling Recommandée:")
        logger.info(f"   1. Identifier les {stats['n_corrected']:,} samples où Stacking corrige {indicator_improved.upper()}")
        logger.info(f"   2. Analyser les prédictions MACD/RSI/CCI sur ces samples")
        logger.info(f"   3. Relabeler selon consensus des 2 autres indicateurs")
        logger.info(f"   4. Réentraîner {indicator_improved.upper()} sur données relabelées")
    else:
        logger.info(f"\n⚠️  Stacking n'améliore pas significativement les indicateurs individuels")
        logger.info(f"   → Les 3 modèles sont déjà bien optimisés indépendamment")
        logger.info(f"   → Le relabeling basé sur Stacking aura un impact limité")
        logger.info(f"\n💡 Alternatives:")
        logger.info(f"   - Profitability Relabeling (testé: +8% Win Rate MACD)")
        logger.info(f"   - Features additionnelles (volatilité, volume)")

    logger.info(f"\n" + "="*80)


if __name__ == '__main__':
    sys.exit(main())
