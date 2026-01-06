#!/usr/bin/env python3
"""
Test Meta-Modèles Spécifiques par Indicateur (Option B)

Objectif: Créer des meta-modèles spécifiques pour améliorer chaque indicateur
en utilisant les AUTRES indicateurs comme features.

Exemples:
  - meta-RSI: Utilise CCI (ou CCI+MACD) pour prédire Y_true_RSI
  - meta-CCI: Utilise RSI (ou RSI+MACD) pour prédire Y_true_CCI
  - meta-MACD: Utilise RSI+CCI pour prédire Y_true_MACD

Hypothèse: Un indicateur peut aider à corriger les erreurs d'un autre.

Usage:
  # Test RSI avec CCI seul
  python src/train_meta_models_per_indicator.py --target rsi --use-indicators cci

  # Test RSI avec CCI+MACD
  python src/train_meta_models_per_indicator.py --target rsi --use-indicators cci macd

  # Test CCI avec RSI seul
  python src/train_meta_models_per_indicator.py --target cci --use-indicators rsi

  # Test MACD avec RSI+CCI
  python src/train_meta_models_per_indicator.py --target macd --use-indicators rsi cci
"""

import sys
import numpy as np
from pathlib import Path
import logging
import argparse
from typing import Dict, List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DATASET_PATHS = {
    'macd': 'data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_kalman.npz',
    'rsi': 'data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_dual_binary_kalman.npz',
    'cci': 'data/prepared/dataset_btc_eth_bnb_ada_ltc_cci_dual_binary_kalman.npz',
}


def load_indicator_predictions(indicator: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge les prédictions d'un indicateur.

    Returns:
        Y_pred: (n, 2) - [direction_proba, force_proba]
        Y_true: (n, 2) - [direction, force]
    """
    path = DATASET_PATHS[indicator]
    data = np.load(path, allow_pickle=True)

    Y_pred = data[f'Y_{split}_pred']  # Probabilities
    Y_true = data[f'Y_{split}']       # Ground truth

    return Y_pred, Y_true


def build_meta_features(
    target_indicator: str,
    use_indicators: List[str],
    split: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construit les méta-features pour un indicateur cible.

    Args:
        target_indicator: Indicateur à améliorer (ex: 'rsi')
        use_indicators: Indicateurs à utiliser comme features (ex: ['cci'] ou ['cci', 'macd'])
        split: 'train', 'val', ou 'test'

    Returns:
        X_meta: Features (n, k) où k = len(use_indicators) * 2
        Y_meta: Cible (n, 1) - Direction de target_indicator
    """
    logger.info(f"\n📦 Construction méta-features pour {target_indicator.upper()}")
    logger.info(f"   Using: {', '.join([i.upper() for i in use_indicators])}")

    # Charger target (pour Y_true)
    _, Y_target = load_indicator_predictions(target_indicator, split)
    Y_meta = Y_target[:, 0:1]  # Direction uniquement

    # Charger features (prédictions des autres indicateurs)
    features = []
    for indicator in use_indicators:
        Y_pred, _ = load_indicator_predictions(indicator, split)
        features.append(Y_pred)  # (n, 2) - [dir_proba, force_proba]
        logger.info(f"   ✅ {indicator.upper()}: {Y_pred.shape}")

    X_meta = np.concatenate(features, axis=1)  # (n, k)

    logger.info(f"\n   X_meta shape: {X_meta.shape}")
    logger.info(f"   Y_meta shape: {Y_meta.shape}")

    return X_meta, Y_meta


def train_and_evaluate_meta_model(
    target_indicator: str,
    use_indicators: List[str],
    model_type: str = 'logistic'
) -> Dict:
    """
    Entraîne et évalue un meta-modèle pour un indicateur.

    Args:
        target_indicator: Indicateur à améliorer
        use_indicators: Indicateurs utilisés comme features
        model_type: 'logistic' ou 'rf'

    Returns:
        dict avec résultats
    """
    logger.info("="*80)
    logger.info(f"🎯 META-MODÈLE POUR {target_indicator.upper()}")
    logger.info("="*80)
    logger.info(f"   Target: Y_true_{target_indicator.upper()} (Direction)")
    logger.info(f"   Features: {', '.join([f'{i.upper()}_pred' for i in use_indicators])}")

    # Charger données
    X_train, Y_train = build_meta_features(target_indicator, use_indicators, 'train')
    X_val, Y_val = build_meta_features(target_indicator, use_indicators, 'val')
    X_test, Y_test = build_meta_features(target_indicator, use_indicators, 'test')

    # Charger baseline (prédictions du modèle target seul)
    Y_baseline_pred, Y_baseline_true = load_indicator_predictions(target_indicator, 'test')
    Y_baseline_pred_binary = (Y_baseline_pred[:, 0] > 0.5).astype(int)
    Y_baseline_true_binary = Y_baseline_true[:, 0].astype(int)

    baseline_acc = accuracy_score(Y_baseline_true_binary, Y_baseline_pred_binary) * 100

    logger.info(f"\n📊 Baseline {target_indicator.upper()}: {baseline_acc:.2f}%")

    # Entraîner meta-modèle
    logger.info(f"\n⏳ Entraînement meta-modèle ({model_type})...")

    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Model type inconnu: {model_type}")

    model.fit(X_train, Y_train.ravel())

    # Prédictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Métriques
    train_acc = accuracy_score(Y_train.ravel(), y_train_pred) * 100
    val_acc = accuracy_score(Y_val.ravel(), y_val_pred) * 100
    test_acc = accuracy_score(Y_test.ravel(), y_test_pred) * 100

    logger.info(f"\n📈 Résultats Meta-Modèle:")
    logger.info(f"   Train Accuracy: {train_acc:.2f}%")
    logger.info(f"   Val Accuracy:   {val_acc:.2f}%")
    logger.info(f"   Test Accuracy:  {test_acc:.2f}%")

    delta = test_acc - baseline_acc
    gap_train_val = abs(train_acc - val_acc)
    gap_val_test = abs(val_acc - test_acc)

    logger.info(f"\n🎯 Comparaison:")
    logger.info(f"   Baseline:   {baseline_acc:.2f}%")
    logger.info(f"   Meta-Model: {test_acc:.2f}%")
    logger.info(f"   Delta:      {delta:+.2f}%")

    logger.info(f"\n📊 Généralisation:")
    logger.info(f"   Gap Train/Val: {gap_train_val:.2f}%")
    logger.info(f"   Gap Val/Test:  {gap_val_test:.2f}%")

    # Interprétabilité (Logistic seulement)
    if model_type == 'logistic':
        logger.info(f"\n🔍 Poids des features:")
        feature_names = []
        for indicator in use_indicators:
            feature_names.extend([f'{indicator.upper()}_dir', f'{indicator.upper()}_force'])

        for name, weight in zip(feature_names, model.coef_[0]):
            logger.info(f"     {name:12s}: {weight:+.4f}")

    # Verdict
    logger.info(f"\n" + "="*80)
    if delta > 1.0:
        verdict = "🏆 AMÉLIORATION SIGNIFICATIVE"
        logger.info(f"✅ {verdict}")
        logger.info(f"   → {', '.join([i.upper() for i in use_indicators])} aide {target_indicator.upper()} (+{delta:.2f}%)")
    elif delta > 0.3:
        verdict = "✅ Amélioration modérée"
        logger.info(f"{verdict}")
        logger.info(f"   → Gain marginal avec {', '.join([i.upper() for i in use_indicators])}")
    elif delta >= 0:
        verdict = "⚪ Neutre"
        logger.info(f"{verdict}")
        logger.info(f"   → Pas d'amélioration significative")
    else:
        verdict = "❌ Dégradation"
        logger.info(f"{verdict}")
        logger.info(f"   → {', '.join([i.upper() for i in use_indicators])} nuit à {target_indicator.upper()}")

    logger.info("="*80)

    return {
        'target': target_indicator,
        'use_indicators': use_indicators,
        'baseline_acc': baseline_acc,
        'test_acc': test_acc,
        'delta': delta,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'gap_train_val': gap_train_val,
        'gap_val_test': gap_val_test,
        'verdict': verdict,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Test Meta-Modèles Spécifiques par Indicateur (Option B)'
    )
    parser.add_argument(
        '--target',
        type=str,
        required=True,
        choices=['rsi', 'cci', 'macd'],
        help="Indicateur à améliorer (cible)"
    )
    parser.add_argument(
        '--use-indicators',
        type=str,
        nargs='+',
        required=True,
        choices=['rsi', 'cci', 'macd'],
        help="Indicateurs à utiliser comme features (ex: cci, ou cci macd)"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='logistic',
        choices=['logistic', 'rf'],
        help="Type de meta-modèle (défaut: logistic)"
    )

    args = parser.parse_args()

    # Validation: target ne doit pas être dans use_indicators
    if args.target in args.use_indicators:
        logger.error(f"❌ Erreur: target '{args.target}' ne peut pas être dans use_indicators")
        logger.error(f"   Utilisation correcte: --target rsi --use-indicators cci macd")
        sys.exit(1)

    logger.info("="*80)
    logger.info("🧪 TEST META-MODÈLES SPÉCIFIQUES PAR INDICATEUR (Option B)")
    logger.info("="*80)

    # Entraîner et évaluer
    results = train_and_evaluate_meta_model(
        target_indicator=args.target,
        use_indicators=args.use_indicators,
        model_type=args.model
    )

    # Résumé final
    logger.info(f"\n" + "="*80)
    logger.info(f"📋 RÉSUMÉ FINAL")
    logger.info(f"="*80)
    logger.info(f"\n   Target: {results['target'].upper()}")
    logger.info(f"   Features: {', '.join([i.upper() for i in results['use_indicators']])}")
    logger.info(f"   Baseline: {results['baseline_acc']:.2f}%")
    logger.info(f"   Meta-Model: {results['test_acc']:.2f}%")
    logger.info(f"   Delta: {results['delta']:+.2f}%")
    logger.info(f"   Verdict: {results['verdict']}")

    logger.info(f"\n💡 Prochaines étapes:")
    if results['delta'] > 1.0:
        logger.info(f"   ✅ Utiliser ce meta-modèle en production")
        logger.info(f"   ✅ Tester en backtest pour mesurer impact Win Rate")
    elif results['delta'] > 0.3:
        logger.info(f"   → Tester avec d'autres combinaisons d'indicateurs")
        logger.info(f"   → Essayer Random Forest si Logistic utilisé")
    else:
        logger.info(f"   → Pas d'amélioration significative")
        logger.info(f"   → Essayer d'autres combinaisons ou revenir à Profitability Relabeling")

    logger.info(f"\n" + "="*80)


if __name__ == '__main__':
    sys.exit(main())
