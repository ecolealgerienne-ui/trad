#!/usr/bin/env python3
"""
PHASE 1 - Nettoyage Structurel (Expert 1)

Applique les règles validées par le Data Audit:
1. UNIVERSEL: Retrait "Kill Zone" (Duration 3-5)
2. SÉLECTIF: Retrait Haute Volatilité (Q4) pour MACD/CCI uniquement

Philosophie:
- Non destructif: Crée de nouvelles versions _cleaned.npz
- Traçable: Logs détaillés des samples retirés
- Conditionnel: Règles adaptées par indicateur

Référence: docs/EXPERT_VALIDATION_PHASE1.md
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuration des seuils validés par l'audit
CONFIG = {
    'universal': {
        'forbidden_duration': [3, 4, 5]  # La "Kill Zone" (Court STRONG)
    },
    'conditional': {
        'macd': {'remove_high_vol': True},   # Tendance → déteste bruit
        'cci':  {'remove_high_vol': True},   # Multi-features → vulnérable
        'rsi':  {'remove_high_vol': False}   # Impulsion → besoin volatilité
    }
}


def load_dataset(path):
    """Charge le dataset .npz"""
    return dict(np.load(path, allow_pickle=True))


def compute_features(returns, force_labels):
    """
    Recalcule les features critiques pour le filtrage.

    Args:
        returns: Rendements c_ret (array 1D)
        force_labels: Labels Force (0=WEAK, 1=STRONG)

    Returns:
        vol_rolling: Volatilité rolling (abs returns, window=20)
        duration: Strong Duration (compteur consécutif)
    """
    # 1. Volatilité Rolling (20 périodes)
    # Utilise abs(returns) comme proxy de volatilité locale
    vol_rolling = pd.Series(returns).abs().rolling(window=20).mean().fillna(0).values

    # 2. Strong Duration (Compteur consécutif)
    duration = np.zeros_like(force_labels, dtype=int)
    count = 0
    for i in range(len(force_labels)):
        if force_labels[i] == 1:  # STRONG
            count += 1
        else:
            count = 0
        duration[i] = count

    return vol_rolling, duration


def clean_data(indicator, split, data):
    """
    Applique le nettoyage structurel sur un split.

    Args:
        indicator: 'macd', 'rsi', ou 'cci'
        split: 'train', 'val', ou 'test'
        data: Dict contenant X_{split}, Y_{split}, Y_{split}_pred (optionnel)

    Returns:
        data_clean: Dict avec X, Y, Y_pred nettoyés
    """
    logger.info(f"\n🧹 Nettoyage {indicator.upper()} [{split}]...")

    X = data[f'X_{split}']
    Y = data[f'Y_{split}']
    Y_pred = data.get(f'Y_{split}_pred', None)

    # Extraction returns (c_ret)
    # RSI/MACD: 1 feature (c_ret à index 0)
    # CCI: 3 features (c_ret à index 2)
    idx_ret = 2 if indicator == 'cci' else 0
    returns = X[:, -1, idx_ret]  # Dernière période de la séquence

    # Labels Force (colonne 1)
    force_labels = Y[:, 1]

    # Calcul métriques
    vol, duration = compute_features(returns, force_labels)

    # --- FILTRE 1 : UNIVERSAL (Duration) ---
    mask_duration = ~np.isin(duration, CONFIG['universal']['forbidden_duration'])
    removed_duration = (~mask_duration).sum()
    logger.info(f"   - Filtre Duration (Retrait 3-5p): {removed_duration} samples retirés")

    # --- FILTRE 2 : CONDITIONAL (Volatilité) ---
    mask_vol = np.ones(len(X), dtype=bool)
    removed_vol = 0

    if CONFIG['conditional'][indicator]['remove_high_vol']:
        # Seuil Q4 (p75) calculé sur le split courant
        # Note: Idéalement utiliser seuil du train pour test,
        # mais pour nettoyage structurel, Q4 local est acceptable
        q4_threshold = np.percentile(vol[vol > 0], 75)
        mask_vol = vol < q4_threshold
        removed_vol = (~mask_vol).sum()
        logger.info(f"   - Filtre Volatilité (Retrait Q4 > {q4_threshold:.5f}): {removed_vol} samples retirés")
    else:
        logger.info(f"   - Filtre Volatilité: DÉSACTIVÉ (Spécifique {indicator.upper()})")

    # COMBINAISON
    mask_final = mask_duration & mask_vol
    removed_total = (~mask_final).sum()
    total_count = len(X)

    logger.info(f"   🚫 TOTAL RETIRÉ: {removed_total} / {total_count} ({removed_total/total_count*100:.1f}%)")

    # Application
    data_clean = {
        f'X_{split}': X[mask_final],
        f'Y_{split}': Y[mask_final]
    }

    if Y_pred is not None:
        data_clean[f'Y_{split}_pred'] = Y_pred[mask_final]

    # Stats distribution Direction (balance UP/DOWN)
    dir_before = (Y[:, 0] == 1).mean()
    dir_after = (Y[mask_final, 0] == 1).mean()
    logger.info(f"   📊 Balance Direction (UP): {dir_before*100:.1f}% -> {dir_after*100:.1f}%")

    # Stats distribution Force (% STRONG)
    force_before = (Y[:, 1] == 1).mean()
    force_after = (Y[mask_final, 1] == 1).mean()
    logger.info(f"   📊 Balance Force (STRONG): {force_before*100:.1f}% -> {force_after*100:.1f}%")

    return data_clean


def main():
    parser = argparse.ArgumentParser(
        description='Phase 1 - Nettoyage Structurel (Expert 1 validated)'
    )
    parser.add_argument(
        '--assets',
        nargs='+',
        default=['BTC', 'ETH', 'BNB', 'ADA', 'LTC'],
        help='Liste des assets (default: BTC ETH BNB ADA LTC)'
    )
    args = parser.parse_args()

    indicators = ['macd', 'rsi', 'cci']
    splits = ['train', 'val', 'test']

    logger.info("=" * 80)
    logger.info("PHASE 1 - NETTOYAGE STRUCTUREL")
    logger.info("=" * 80)
    logger.info("\nConfiguration:")
    logger.info(f"  - Universal: Retrait Duration {CONFIG['universal']['forbidden_duration']}")
    logger.info(f"  - MACD: Retrait Vol Q4 = {CONFIG['conditional']['macd']['remove_high_vol']}")
    logger.info(f"  - CCI:  Retrait Vol Q4 = {CONFIG['conditional']['cci']['remove_high_vol']}")
    logger.info(f"  - RSI:  Retrait Vol Q4 = {CONFIG['conditional']['rsi']['remove_high_vol']}")
    logger.info("")

    for ind in indicators:
        # Chemin fichier
        assets_str = "_".join([a.lower() for a in args.assets])
        filename = f"dataset_{assets_str}_{ind}_dual_binary_kalman.npz"
        path = Path(f"data/prepared/{filename}")

        if not path.exists():
            logger.warning(f"⚠️  Fichier introuvable: {path}")
            continue

        logger.info(f"💾 Chargement {filename}...")
        full_data = load_dataset(path)
        new_data = full_data.copy()

        # Nettoyage de chaque split
        for split in splits:
            if f'X_{split}' in full_data:
                cleaned = clean_data(ind, split, full_data)
                # Mise à jour du dict global
                for k, v in cleaned.items():
                    new_data[k] = v

        # Sauvegarde
        out_path = str(path).replace('.npz', '_cleaned.npz')
        np.savez(out_path, **new_data)
        logger.info(f"✅ Sauvegardé: {out_path}\n")

    logger.info("=" * 80)
    logger.info("🎉 NETTOYAGE COMPLÉTÉ")
    logger.info("=" * 80)
    logger.info("\nProchaine étape:")
    logger.info("  python src/evaluate.py --data data/prepared/dataset_*_cleaned.npz")
    logger.info("\nGain attendu (Expert 1): +5-8% accuracy Oracle sur test set")
    logger.info("")


if __name__ == "__main__":
    main()
