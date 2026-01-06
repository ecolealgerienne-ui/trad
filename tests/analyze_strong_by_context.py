#!/usr/bin/env python3
"""
ANALYSE CONTEXTE - Quand les indicateurs STRONG sont-ils prédictifs ?

Segmente les données par 4 dimensions :
1. Volatilité (Q1, Q2, Q3, Q4)
2. Régime de marché (Trend vs Range)
3. Densité de retournements (Low churn vs High churn)
4. Durée STRONG actuelle (Nouveau vs Établi)

Pour chaque contexte, mesure :
- N samples
- Oracle STRONG : accuracy future, corrélation
- IA STRONG : accuracy future, corrélation
- Delta Oracle vs IA

Objectif : Découvrir quels contextes sont TRADABLES et lesquels sont NOISE.
"""

import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_dataset(indicator: str, split: str):
    """Charge le dataset."""
    dataset_path = Path(f"data/prepared/dataset_btc_eth_bnb_ada_ltc_{indicator}_dual_binary_kalman.npz")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)

    return {
        'X': data[f'X_{split}'],
        'Y': data[f'Y_{split}'],
        'Y_pred': data.get(f'Y_{split}_pred', None),
    }


def extract_c_ret(X: np.ndarray, indicator: str) -> np.ndarray:
    """Extrait c_ret de X."""
    if indicator in ['rsi', 'macd']:
        c_ret = X[:, -1, 0]
    elif indicator == 'cci':
        c_ret = X[:, -1, 2]
    else:
        raise ValueError(f"Indicateur inconnu: {indicator}")

    return c_ret


def compute_directional_accuracy(direction_labels, future_returns):
    """Calcule accuracy directionnelle."""
    predicted_up = (direction_labels == 1)
    actual_up = (future_returns > 0)
    correct = (predicted_up == actual_up)
    return correct.mean()


def compute_correlation(direction_labels, future_returns):
    """Calcule corrélation."""
    direction_signed = (direction_labels * 2) - 1  # 0→-1, 1→+1
    if len(direction_signed) == 0 or len(future_returns) == 0:
        return 0.0
    correlation = np.corrcoef(direction_signed, future_returns)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0


def compute_volatility_rolling(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Calcule volatilité rolling (abs returns)."""
    vol = np.zeros(len(returns))
    for i in range(window, len(returns)):
        vol[i] = np.abs(returns[i-window:i]).mean()
    return vol


def compute_regime(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Calcule régime de marché.
    Trend: cumul returns fort (momentum)
    Range: cumul returns faible (oscillation)
    """
    regime = np.zeros(len(returns))
    for i in range(window, len(returns)):
        cumul = returns[i-window:i].sum()
        # Trend si cumul > 1% ou < -1%
        regime[i] = 1 if abs(cumul) > 0.01 else 0
    return regime  # 1=Trend, 0=Range


def compute_churn(direction_labels: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Calcule densité de retournements (churn).
    Compte transitions Direction sur fenêtre rolling.
    """
    churn = np.zeros(len(direction_labels))
    for i in range(window, len(direction_labels)):
        transitions = (direction_labels[i-window:i-1] != direction_labels[i-window+1:i]).sum()
        churn[i] = transitions
    return churn


def compute_strong_duration(force_labels: np.ndarray) -> np.ndarray:
    """
    Calcule durée STRONG actuelle.
    Nombre de périodes consécutives STRONG jusqu'à maintenant.
    """
    duration = np.zeros(len(force_labels))
    current_duration = 0

    for i in range(len(force_labels)):
        if force_labels[i] == 1:
            current_duration += 1
        else:
            current_duration = 0
        duration[i] = current_duration

    return duration


def analyze_context(
    mask: np.ndarray,
    oracle_dir: np.ndarray,
    oracle_force: np.ndarray,
    pred_dir: np.ndarray,
    pred_force: np.ndarray,
    future_returns: np.ndarray,
    context_name: str
) -> Dict:
    """Analyse un contexte spécifique."""
    n_total = mask.sum()

    if n_total == 0:
        return {
            'context': context_name,
            'n_total': 0,
            'pct_total': 0.0,
            'oracle_strong_pct': 0.0,
            'ia_strong_pct': 0.0,
            'oracle_acc': 0.0,
            'oracle_corr': 0.0,
            'ia_acc': 0.0,
            'ia_corr': 0.0,
            'delta_acc': 0.0,
            'delta_corr': 0.0,
        }

    # Statistiques générales
    pct_total = n_total / len(mask) * 100

    # Oracle STRONG dans ce contexte
    oracle_strong_mask = mask & (oracle_force == 1)
    n_oracle_strong = oracle_strong_mask.sum()
    oracle_strong_pct = n_oracle_strong / n_total * 100 if n_total > 0 else 0

    # IA STRONG dans ce contexte
    ia_strong_mask = mask & (pred_force == 1)
    n_ia_strong = ia_strong_mask.sum()
    ia_strong_pct = n_ia_strong / n_total * 100 if n_total > 0 else 0

    # Prédictivité Oracle STRONG
    if n_oracle_strong > 0:
        oracle_acc = compute_directional_accuracy(
            oracle_dir[oracle_strong_mask],
            future_returns[oracle_strong_mask]
        )
        oracle_corr = compute_correlation(
            oracle_dir[oracle_strong_mask],
            future_returns[oracle_strong_mask]
        )
    else:
        oracle_acc = 0.0
        oracle_corr = 0.0

    # Prédictivité IA STRONG
    if n_ia_strong > 0:
        ia_acc = compute_directional_accuracy(
            pred_dir[ia_strong_mask],
            future_returns[ia_strong_mask]
        )
        ia_corr = compute_correlation(
            pred_dir[ia_strong_mask],
            future_returns[ia_strong_mask]
        )
    else:
        ia_acc = 0.0
        ia_corr = 0.0

    return {
        'context': context_name,
        'n_total': n_total,
        'pct_total': pct_total,
        'n_oracle_strong': n_oracle_strong,
        'oracle_strong_pct': oracle_strong_pct,
        'n_ia_strong': n_ia_strong,
        'ia_strong_pct': ia_strong_pct,
        'oracle_acc': oracle_acc,
        'oracle_corr': oracle_corr,
        'ia_acc': ia_acc,
        'ia_corr': ia_corr,
        'delta_acc': ia_acc - oracle_acc,
        'delta_corr': ia_corr - oracle_corr,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyse contexte - Quand STRONG est-il prédictif ?")
    parser.add_argument('--indicator', type=str, required=True, choices=['macd', 'rsi', 'cci'])
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])

    args = parser.parse_args()

    # Charger données
    data = load_dataset(args.indicator, args.split)
    returns = extract_c_ret(data['X'], args.indicator)

    Y_oracle = data['Y']
    Y_pred = data['Y_pred']

    if Y_pred is None:
        logger.error("❌ Prédictions non disponibles. Exécuter train.py + evaluate.py d'abord.")
        return

    # Futurs returns
    future_returns = returns[1:]
    Y_oracle = Y_oracle[:-1]
    Y_pred = Y_pred[:-1]
    returns = returns[:-1]  # Aligner

    n_samples = len(future_returns)

    # Extraire labels
    oracle_dir = Y_oracle[:, 0]
    oracle_force = Y_oracle[:, 1]
    pred_dir = (Y_pred[:, 0] > 0.5).astype(int)
    pred_force = (Y_pred[:, 1] > 0.5).astype(int)

    logger.info("=" * 80)
    logger.info(f"🔬 ANALYSE CONTEXTE - {args.indicator.upper()} ({args.split})")
    logger.info("=" * 80)
    logger.info(f"\n📊 Dataset: {n_samples:,} samples")

    # ============================================
    # 1. SEGMENTATION PAR VOLATILITÉ
    # ============================================
    logger.info(f"\n{'=' * 80}")
    logger.info(f"📊 SEGMENTATION 1 : VOLATILITÉ")
    logger.info("=" * 80)

    vol = compute_volatility_rolling(returns, window=20)
    vol_quantiles = np.percentile(vol[vol > 0], [0, 25, 50, 75, 100])

    logger.info(f"\nQuantiles volatilité: {vol_quantiles}")

    vol_results = []
    for i in range(len(vol_quantiles) - 1):
        q_low, q_high = vol_quantiles[i], vol_quantiles[i + 1]
        mask = (vol >= q_low) & (vol < q_high)

        context_name = f"Vol Q{i+1} [{q_low*100:.2f}%, {q_high*100:.2f}%)"
        result = analyze_context(mask, oracle_dir, oracle_force, pred_dir, pred_force, future_returns, context_name)
        vol_results.append(result)

    # Afficher résultats
    for res in vol_results:
        logger.info(f"\n{res['context']}:")
        logger.info(f"   Samples: {res['n_total']:,} ({res['pct_total']:.1f}%)")
        logger.info(f"   Oracle STRONG: {res['n_oracle_strong']:,} ({res['oracle_strong_pct']:.1f}%)")
        logger.info(f"   IA STRONG: {res['n_ia_strong']:,} ({res['ia_strong_pct']:.1f}%)")
        logger.info(f"   Oracle: Acc={res['oracle_acc']*100:.2f}%, Corr={res['oracle_corr']:.4f}")
        logger.info(f"   IA:     Acc={res['ia_acc']*100:.2f}%, Corr={res['ia_corr']:.4f}")
        logger.info(f"   Delta:  Acc={res['delta_acc']*100:+.2f}%, Corr={res['delta_corr']:+.4f}")

    # ============================================
    # 2. SEGMENTATION PAR RÉGIME
    # ============================================
    logger.info(f"\n{'=' * 80}")
    logger.info(f"📊 SEGMENTATION 2 : RÉGIME DE MARCHÉ")
    logger.info("=" * 80)

    regime = compute_regime(returns, window=20)

    regime_results = []
    for regime_val, regime_name in [(1, "Trend (momentum fort)"), (0, "Range (oscillation)")]:
        mask = (regime == regime_val)
        result = analyze_context(mask, oracle_dir, oracle_force, pred_dir, pred_force, future_returns, regime_name)
        regime_results.append(result)

    for res in regime_results:
        logger.info(f"\n{res['context']}:")
        logger.info(f"   Samples: {res['n_total']:,} ({res['pct_total']:.1f}%)")
        logger.info(f"   Oracle STRONG: {res['n_oracle_strong']:,} ({res['oracle_strong_pct']:.1f}%)")
        logger.info(f"   IA STRONG: {res['n_ia_strong']:,} ({res['ia_strong_pct']:.1f}%)")
        logger.info(f"   Oracle: Acc={res['oracle_acc']*100:.2f}%, Corr={res['oracle_corr']:.4f}")
        logger.info(f"   IA:     Acc={res['ia_acc']*100:.2f}%, Corr={res['ia_corr']:.4f}")
        logger.info(f"   Delta:  Acc={res['delta_acc']*100:+.2f}%, Corr={res['delta_corr']:+.4f}")

    # ============================================
    # 3. SEGMENTATION PAR CHURN
    # ============================================
    logger.info(f"\n{'=' * 80}")
    logger.info(f"📊 SEGMENTATION 3 : DENSITÉ DE RETOURNEMENTS (CHURN)")
    logger.info("=" * 80)

    churn = compute_churn(oracle_dir, window=20)

    churn_results = []
    for threshold, churn_name in [(5, "Low churn (0-5 transitions)"), (100, "High churn (5+ transitions)")]:
        if threshold == 5:
            mask = (churn <= 5)
        else:
            mask = (churn > 5)

        result = analyze_context(mask, oracle_dir, oracle_force, pred_dir, pred_force, future_returns, churn_name)
        churn_results.append(result)

    for res in churn_results:
        logger.info(f"\n{res['context']}:")
        logger.info(f"   Samples: {res['n_total']:,} ({res['pct_total']:.1f}%)")
        logger.info(f"   Oracle STRONG: {res['n_oracle_strong']:,} ({res['oracle_strong_pct']:.1f}%)")
        logger.info(f"   IA STRONG: {res['n_ia_strong']:,} ({res['ia_strong_pct']:.1f}%)")
        logger.info(f"   Oracle: Acc={res['oracle_acc']*100:.2f}%, Corr={res['oracle_corr']:.4f}")
        logger.info(f"   IA:     Acc={res['ia_acc']*100:.2f}%, Corr={res['ia_corr']:.4f}")
        logger.info(f"   Delta:  Acc={res['delta_acc']*100:+.2f}%, Corr={res['delta_corr']:+.4f}")

    # ============================================
    # 4. SEGMENTATION PAR DURÉE STRONG
    # ============================================
    logger.info(f"\n{'=' * 80}")
    logger.info(f"📊 SEGMENTATION 4 : DURÉE STRONG ACTUELLE")
    logger.info("=" * 80)

    duration = compute_strong_duration(oracle_force)

    duration_results = []
    for threshold, duration_name in [(1, "Nouveau STRONG (1-2 périodes)"),
                                      (3, "Court STRONG (3-5 périodes)"),
                                      (6, "Établi STRONG (6+ périodes)")]:
        if threshold == 1:
            mask = (duration >= 1) & (duration <= 2) & (oracle_force == 1)
        elif threshold == 3:
            mask = (duration >= 3) & (duration <= 5) & (oracle_force == 1)
        else:
            mask = (duration >= 6) & (oracle_force == 1)

        result = analyze_context(mask, oracle_dir, oracle_force, pred_dir, pred_force, future_returns, duration_name)
        duration_results.append(result)

    for res in duration_results:
        logger.info(f"\n{res['context']}:")
        logger.info(f"   Samples: {res['n_total']:,} ({res['pct_total']:.1f}%)")
        logger.info(f"   IA détecte: {res['n_ia_strong']:,} ({res['ia_strong_pct']:.1f}%)")
        logger.info(f"   Oracle: Acc={res['oracle_acc']*100:.2f}%, Corr={res['oracle_corr']:.4f}")
        logger.info(f"   IA:     Acc={res['ia_acc']*100:.2f}%, Corr={res['ia_corr']:.4f}")
        logger.info(f"   Delta:  Acc={res['delta_acc']*100:+.2f}%, Corr={res['delta_corr']:+.4f}")

    # ============================================
    # SYNTHÈSE ET RECOMMANDATIONS
    # ============================================
    logger.info(f"\n{'=' * 80}")
    logger.info(f"💡 SYNTHÈSE ET RECOMMANDATIONS")
    logger.info("=" * 80)

    # Trouver meilleurs contextes
    all_results = vol_results + regime_results + churn_results + duration_results

    # Trier par Oracle accuracy
    sorted_by_oracle = sorted([r for r in all_results if r['oracle_acc'] > 0],
                              key=lambda x: x['oracle_acc'], reverse=True)

    logger.info(f"\n📊 TOP 5 CONTEXTES (Oracle STRONG le plus prédictif):")
    for i, res in enumerate(sorted_by_oracle[:5], 1):
        logger.info(f"\n{i}. {res['context']}")
        logger.info(f"   Oracle Acc: {res['oracle_acc']*100:.2f}%, Corr: {res['oracle_corr']:.4f}")
        logger.info(f"   Samples: {res['n_oracle_strong']:,}")

    # Trouver pires contextes
    sorted_by_oracle_worst = sorted([r for r in all_results if r['oracle_acc'] > 0],
                                    key=lambda x: x['oracle_acc'])

    logger.info(f"\n📊 BOTTOM 5 CONTEXTES (Oracle STRONG le moins prédictif = NOISE):")
    for i, res in enumerate(sorted_by_oracle_worst[:5], 1):
        logger.info(f"\n{i}. {res['context']}")
        logger.info(f"   Oracle Acc: {res['oracle_acc']*100:.2f}%, Corr: {res['oracle_corr']:.4f}")
        logger.info(f"   Samples: {res['n_oracle_strong']:,}")
        if res['oracle_acc'] < 0.52:
            logger.info(f"   ⚠️  NON TRADABLE (accuracy ≈ hasard)")

    # Recommandations nettoyage
    logger.info(f"\n💊 RECOMMANDATIONS NETTOYAGE STRUCTUREL:")

    # Volatilité minimale
    worst_vol = vol_results[0]
    if worst_vol['oracle_acc'] < 0.52:
        logger.info(f"\n   1. RETIRER volatilité < {vol_quantiles[1]*100:.2f}%")
        logger.info(f"      → {worst_vol['n_total']:,} samples ({worst_vol['pct_total']:.1f}%)")
        logger.info(f"      → Oracle Acc = {worst_vol['oracle_acc']*100:.2f}% (non prédictif)")

    # Durée minimale
    worst_duration = duration_results[0]
    if worst_duration['oracle_acc'] < sorted_by_oracle[0]['oracle_acc'] - 0.10:
        logger.info(f"\n   2. RETIRER STRONG durée < 3 périodes")
        logger.info(f"      → {worst_duration['n_total']:,} samples")
        logger.info(f"      → Oracle Acc = {worst_duration['oracle_acc']*100:.2f}% vs {sorted_by_oracle[0]['oracle_acc']*100:.2f}% (meilleur)")

    # High churn
    if len(churn_results) > 1:
        high_churn = churn_results[1]
        low_churn = churn_results[0]
        if high_churn['oracle_acc'] < low_churn['oracle_acc'] - 0.05:
            logger.info(f"\n   3. CONSIDÉRER retirer high churn (5+ transitions/20 périodes)")
            logger.info(f"      → {high_churn['n_total']:,} samples ({high_churn['pct_total']:.1f}%)")
            logger.info(f"      → Oracle Acc = {high_churn['oracle_acc']*100:.2f}% vs {low_churn['oracle_acc']*100:.2f}% (low churn)")

    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    main()
