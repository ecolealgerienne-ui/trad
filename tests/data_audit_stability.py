#!/usr/bin/env python3
"""
DATA AUDIT - Validation Stabilité Temporelle des Patterns

Objectif : Vérifier que les découvertes ne sont PAS accidentelles (data snooping).

Tests de stabilité :
1. Nouveau STRONG reste dominant sur toutes les périodes ?
2. Vol faible > Vol haute stable dans le temps ?
3. RSI > MACD consistant sur toutes les périodes ?
4. Court STRONG (3-5) reste le pire sur toutes les périodes ?

Méthode : Walk-forward analysis
- Découper dataset en périodes de 3 mois
- Mesurer patterns clés dans chaque période
- Vérifier consistance

Si patterns instables → data snooping → NE PAS utiliser
Si patterns stables → validation empirique → GO implémentation
"""

import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

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


def compute_volatility_rolling(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Calcule volatilité rolling."""
    vol = np.zeros(len(returns))
    for i in range(window, len(returns)):
        vol[i] = np.abs(returns[i-window:i]).mean()
    return vol


def compute_strong_duration(force_labels: np.ndarray) -> np.ndarray:
    """Calcule durée STRONG actuelle."""
    duration = np.zeros(len(force_labels))
    current_duration = 0

    for i in range(len(force_labels)):
        if force_labels[i] == 1:
            current_duration += 1
        else:
            current_duration = 0
        duration[i] = current_duration

    return duration


def compute_directional_accuracy(direction_labels, future_returns):
    """Calcule accuracy directionnelle."""
    if len(direction_labels) == 0:
        return 0.0
    predicted_up = (direction_labels == 1)
    actual_up = (future_returns > 0)
    correct = (predicted_up == actual_up)
    return correct.mean()


def split_by_periods(n_samples: int, period_size: int = 36000):
    """
    Découpe dataset en périodes temporelles.

    Args:
        n_samples: Nombre total de samples
        period_size: Taille période en samples (défaut: 36000 = ~125 jours à 5min)

    Returns:
        List[(start, end)] indices pour chaque période
    """
    periods = []
    start = 0
    while start < n_samples:
        end = min(start + period_size, n_samples)
        if end - start > period_size // 2:  # Au moins 50% d'une période
            periods.append((start, end))
        start += period_size

    return periods


def analyze_period(
    oracle_dir: np.ndarray,
    oracle_force: np.ndarray,
    pred_dir: np.ndarray,
    pred_force: np.ndarray,
    future_returns: np.ndarray,
    returns: np.ndarray,
    period_name: str
) -> Dict:
    """
    Analyse patterns clés sur une période.

    Returns:
        Dict avec métriques de stabilité
    """
    result = {
        'period': period_name,
        'n_samples': len(oracle_dir),
    }

    # Calculer contextes
    vol = compute_volatility_rolling(returns, window=20)
    strong_duration = compute_strong_duration(oracle_force)

    # Pattern 1 : Nouveau STRONG vs Court STRONG
    nouveau_mask = ((strong_duration >= 1) & (strong_duration <= 2) & (oracle_force == 1))
    court_mask = ((strong_duration >= 3) & (strong_duration <= 5) & (oracle_force == 1))

    if nouveau_mask.sum() > 100 and court_mask.sum() > 100:
        nouveau_acc = compute_directional_accuracy(
            oracle_dir[nouveau_mask],
            future_returns[nouveau_mask]
        )
        court_acc = compute_directional_accuracy(
            oracle_dir[court_mask],
            future_returns[court_mask]
        )
        result['nouveau_acc'] = nouveau_acc
        result['court_acc'] = court_acc
        result['nouveau_vs_court'] = nouveau_acc - court_acc
        result['nouveau_better'] = nouveau_acc > court_acc
    else:
        result['nouveau_acc'] = None
        result['court_acc'] = None
        result['nouveau_vs_court'] = None
        result['nouveau_better'] = None

    # Pattern 2 : Vol faible vs Vol haute
    vol_valid = vol[vol > 0]
    if len(vol_valid) > 100:
        vol_q1 = np.percentile(vol_valid, 25)
        vol_q4_threshold = np.percentile(vol_valid, 75)

        vol_low_mask = ((vol <= vol_q1) & (oracle_force == 1))
        vol_high_mask = ((vol >= vol_q4_threshold) & (oracle_force == 1))

        if vol_low_mask.sum() > 100 and vol_high_mask.sum() > 100:
            vol_low_acc = compute_directional_accuracy(
                oracle_dir[vol_low_mask],
                future_returns[vol_low_mask]
            )
            vol_high_acc = compute_directional_accuracy(
                oracle_dir[vol_high_mask],
                future_returns[vol_high_mask]
            )
            result['vol_low_acc'] = vol_low_acc
            result['vol_high_acc'] = vol_high_acc
            result['vol_low_vs_high'] = vol_low_acc - vol_high_acc
            result['vol_low_better'] = vol_low_acc > vol_high_acc
        else:
            result['vol_low_acc'] = None
            result['vol_high_acc'] = None
            result['vol_low_vs_high'] = None
            result['vol_low_better'] = None
    else:
        result['vol_low_acc'] = None
        result['vol_high_acc'] = None
        result['vol_low_vs_high'] = None
        result['vol_low_better'] = None

    # Pattern 3 : Oracle vs IA accuracy (RSI > MACD sera testé entre indicateurs)
    oracle_strong_mask = (oracle_force == 1)
    ia_strong_mask = (pred_force == 1)

    if oracle_strong_mask.sum() > 100:
        oracle_acc = compute_directional_accuracy(
            oracle_dir[oracle_strong_mask],
            future_returns[oracle_strong_mask]
        )
        result['oracle_acc'] = oracle_acc
    else:
        result['oracle_acc'] = None

    if ia_strong_mask.sum() > 100:
        ia_acc = compute_directional_accuracy(
            pred_dir[ia_strong_mask],
            future_returns[ia_strong_mask]
        )
        result['ia_acc'] = ia_acc
    else:
        result['ia_acc'] = None

    if result['oracle_acc'] is not None and result['ia_acc'] is not None:
        result['oracle_vs_ia'] = result['oracle_acc'] - result['ia_acc']
        result['oracle_better'] = result['oracle_acc'] > result['ia_acc']
    else:
        result['oracle_vs_ia'] = None
        result['oracle_better'] = None

    return result


def main():
    parser = argparse.ArgumentParser(description="Data Audit - Validation stabilité temporelle patterns")
    parser.add_argument('--indicator', type=str, required=True, choices=['macd', 'rsi', 'cci'])
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--period-size', type=int, default=36000,
                       help="Taille période en samples (défaut: 36000 = ~125 jours)")

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
    returns = returns[:-1]

    n_samples = len(future_returns)

    # Extraire labels
    oracle_dir = Y_oracle[:, 0]
    oracle_force = Y_oracle[:, 1]
    pred_dir = (Y_pred[:, 0] > 0.5).astype(int)
    pred_force = (Y_pred[:, 1] > 0.5).astype(int)

    logger.info("=" * 80)
    logger.info(f"🔬 DATA AUDIT - STABILITÉ TEMPORELLE - {args.indicator.upper()} ({args.split})")
    logger.info("=" * 80)
    logger.info(f"\n📊 Dataset:")
    logger.info(f"   Total samples: {n_samples:,}")
    logger.info(f"   Période size: {args.period_size:,} samples (~{args.period_size*5/1440:.0f} jours)")

    # Découper en périodes
    periods = split_by_periods(n_samples, args.period_size)
    logger.info(f"   Nombre de périodes: {len(periods)}")

    # Analyser chaque période
    logger.info(f"\n{'=' * 80}")
    logger.info(f"📊 ANALYSE PAR PÉRIODE")
    logger.info("=" * 80)

    results = []
    for i, (start, end) in enumerate(periods, 1):
        period_name = f"Période {i} (samples {start:,}-{end:,})"

        result = analyze_period(
            oracle_dir[start:end],
            oracle_force[start:end],
            pred_dir[start:end],
            pred_force[start:end],
            future_returns[start:end],
            returns[start:end],
            period_name
        )
        results.append(result)

        # Afficher résultats période
        logger.info(f"\n{period_name}:")
        logger.info(f"   Samples: {result['n_samples']:,}")

        if result['nouveau_acc'] is not None:
            logger.info(f"   Nouveau STRONG: {result['nouveau_acc']*100:.2f}%")
            logger.info(f"   Court STRONG:   {result['court_acc']*100:.2f}%")
            logger.info(f"   Delta:          {result['nouveau_vs_court']*100:+.2f}% {'✅' if result['nouveau_better'] else '❌'}")
        else:
            logger.info(f"   Nouveau vs Court: ⚠️ Pas assez de samples")

        if result['vol_low_acc'] is not None:
            logger.info(f"   Vol faible:     {result['vol_low_acc']*100:.2f}%")
            logger.info(f"   Vol haute:      {result['vol_high_acc']*100:.2f}%")
            logger.info(f"   Delta:          {result['vol_low_vs_high']*100:+.2f}% {'✅' if result['vol_low_better'] else '❌'}")
        else:
            logger.info(f"   Vol faible vs haute: ⚠️ Pas assez de samples")

        if result['oracle_acc'] is not None:
            logger.info(f"   Oracle acc:     {result['oracle_acc']*100:.2f}%")
            logger.info(f"   IA acc:         {result['ia_acc']*100:.2f}%")
            logger.info(f"   Delta:          {result['oracle_vs_ia']*100:+.2f}% {'✅' if result['oracle_better'] else '❌'}")
        else:
            logger.info(f"   Oracle vs IA: ⚠️ Pas assez de samples")

    # Synthèse stabilité
    logger.info(f"\n{'=' * 80}")
    logger.info(f"📊 SYNTHÈSE STABILITÉ")
    logger.info("=" * 80)

    # Pattern 1 : Nouveau > Court
    nouveau_vs_court_results = [r for r in results if r['nouveau_better'] is not None]
    if nouveau_vs_court_results:
        nouveau_better_count = sum([1 for r in nouveau_vs_court_results if r['nouveau_better']])
        nouveau_stability = nouveau_better_count / len(nouveau_vs_court_results)

        logger.info(f"\n1️⃣  NOUVEAU STRONG > COURT STRONG:")
        logger.info(f"   Périodes validées: {nouveau_better_count}/{len(nouveau_vs_court_results)} ({nouveau_stability*100:.1f}%)")

        deltas = [r['nouveau_vs_court'] for r in nouveau_vs_court_results]
        logger.info(f"   Delta moyen:       {np.mean(deltas)*100:+.2f}%")
        logger.info(f"   Delta min:         {np.min(deltas)*100:+.2f}%")
        logger.info(f"   Delta max:         {np.max(deltas)*100:+.2f}%")
        logger.info(f"   Écart-type:        {np.std(deltas)*100:.2f}%")

        if nouveau_stability >= 0.8:
            logger.info(f"   ✅ PATTERN STABLE (≥80% périodes)")
        elif nouveau_stability >= 0.6:
            logger.info(f"   ⚠️  PATTERN MODÉRÉ (60-80% périodes)")
        else:
            logger.info(f"   ❌ PATTERN INSTABLE (<60% périodes) - DATA SNOOPING ?")
    else:
        logger.info(f"\n1️⃣  NOUVEAU STRONG > COURT STRONG:")
        logger.info(f"   ⚠️  Pas assez de données pour valider")

    # Pattern 2 : Vol faible > Vol haute
    vol_results = [r for r in results if r['vol_low_better'] is not None]
    if vol_results:
        vol_better_count = sum([1 for r in vol_results if r['vol_low_better']])
        vol_stability = vol_better_count / len(vol_results)

        logger.info(f"\n2️⃣  VOL FAIBLE > VOL HAUTE:")
        logger.info(f"   Périodes validées: {vol_better_count}/{len(vol_results)} ({vol_stability*100:.1f}%)")

        deltas = [r['vol_low_vs_high'] for r in vol_results]
        logger.info(f"   Delta moyen:       {np.mean(deltas)*100:+.2f}%")
        logger.info(f"   Delta min:         {np.min(deltas)*100:+.2f}%")
        logger.info(f"   Delta max:         {np.max(deltas)*100:+.2f}%")
        logger.info(f"   Écart-type:        {np.std(deltas)*100:.2f}%")

        if vol_stability >= 0.8:
            logger.info(f"   ✅ PATTERN STABLE (≥80% périodes)")
        elif vol_stability >= 0.6:
            logger.info(f"   ⚠️  PATTERN MODÉRÉ (60-80% périodes)")
        else:
            logger.info(f"   ❌ PATTERN INSTABLE (<60% périodes) - DATA SNOOPING ?")
    else:
        logger.info(f"\n2️⃣  VOL FAIBLE > VOL HAUTE:")
        logger.info(f"   ⚠️  Pas assez de données pour valider")

    # Pattern 3 : Oracle > IA
    oracle_results = [r for r in results if r['oracle_better'] is not None]
    if oracle_results:
        oracle_better_count = sum([1 for r in oracle_results if r['oracle_better']])
        oracle_stability = oracle_better_count / len(oracle_results)

        logger.info(f"\n3️⃣  ORACLE > IA:")
        logger.info(f"   Périodes validées: {oracle_better_count}/{len(oracle_results)} ({oracle_stability*100:.1f}%)")

        deltas = [r['oracle_vs_ia'] for r in oracle_results]
        logger.info(f"   Delta moyen:       {np.mean(deltas)*100:+.2f}%")
        logger.info(f"   Delta min:         {np.min(deltas)*100:+.2f}%")
        logger.info(f"   Delta max:         {np.max(deltas)*100:+.2f}%")
        logger.info(f"   Écart-type:        {np.std(deltas)*100:.2f}%")

        if oracle_stability >= 0.8:
            logger.info(f"   ✅ PATTERN STABLE (≥80% périodes)")
        elif oracle_stability >= 0.6:
            logger.info(f"   ⚠️  PATTERN MODÉRÉ (60-80% périodes)")
        else:
            logger.info(f"   ❌ PATTERN INSTABLE (<60% périodes) - DATA SNOOPING ?")
    else:
        logger.info(f"\n3️⃣  ORACLE > IA:")
        logger.info(f"   ⚠️  Pas assez de données pour valider")

    # Verdict final
    logger.info(f"\n{'=' * 80}")
    logger.info(f"💡 VERDICT FINAL")
    logger.info("=" * 80)

    stable_patterns = 0
    total_patterns = 0

    if nouveau_vs_court_results:
        total_patterns += 1
        if nouveau_stability >= 0.6:
            stable_patterns += 1

    if vol_results:
        total_patterns += 1
        if vol_stability >= 0.6:
            stable_patterns += 1

    if oracle_results:
        total_patterns += 1
        if oracle_stability >= 0.6:
            stable_patterns += 1

    if total_patterns > 0:
        overall_stability = stable_patterns / total_patterns

        logger.info(f"\n   Patterns stables (≥60%): {stable_patterns}/{total_patterns} ({overall_stability*100:.1f}%)")

        if overall_stability >= 0.8:
            logger.info(f"\n   ✅ VALIDATION GLOBALE : Patterns STABLES")
            logger.info(f"      → Les découvertes sont ROBUSTES temporellement")
            logger.info(f"      → GO pour implémentation (nettoyage + meta-modèle)")
        elif overall_stability >= 0.5:
            logger.info(f"\n   ⚠️  VALIDATION PARTIELLE : Patterns MODÉRÉS")
            logger.info(f"      → Certains patterns sont stables, d'autres incertains")
            logger.info(f"      → PRUDENCE : Utiliser uniquement patterns stables")
        else:
            logger.info(f"\n   ❌ ÉCHEC VALIDATION : Patterns INSTABLES")
            logger.info(f"      → Risque élevé de DATA SNOOPING")
            logger.info(f"      → NE PAS implémenter sans investigation approfondie")
    else:
        logger.info(f"\n   ⚠️  Pas assez de données pour validation globale")

    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    main()
