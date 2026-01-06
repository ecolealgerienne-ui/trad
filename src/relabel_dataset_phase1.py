#!/usr/bin/env python3
"""
PHASE 1 - Target Correction (RELABELING)

APPROCHE CORRIGÉE: Au lieu de SUPPRIMER les données difficiles,
on RELABELE Force=STRONG → Force=WEAK pour les "pièges".

Principe (Hard Negative Mining):
- Le modèle VOIT les configurations pièges
- Il APPREND à les reconnaître comme WEAK (pas STRONG)
- En prod, il DÉTECTE ces patterns et prédit correctement WEAK

Pièges identifiés par Data Audit:
1. UNIVERSEL: Duration 3-5 ("Kill Zone" - Bull Traps)
2. CONDITIONNEL: Vol > Q4 pour MACD/CCI (bruit déstabilisant)

Au lieu de:
  SUPPRESSION → Modèle ne voit jamais → Tombe dedans en prod ❌

On fait:
  RELABELING → Modèle apprend que c'est WEAK → Détecte en prod ✅

Référence: Target Correction / Hard Negative Mining (ML classique)
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
        'trap_duration': [3, 4, 5]  # "Kill Zone" - Faux STRONG
    },
    'conditional': {
        'macd': {'relabel_high_vol': True},   # Tendance → bruit=piège
        'cci':  {'relabel_high_vol': True},   # Multi-features → vulnérable
        'rsi':  {'relabel_high_vol': False}   # Impulsion → besoin vol!
    }
}


def load_dataset(path):
    """Charge le dataset .npz"""
    return dict(np.load(path, allow_pickle=True))


def compute_features(returns, force_labels):
    """
    Recalcule les features critiques pour le relabeling.

    Args:
        returns: Rendements c_ret (array 1D)
        force_labels: Labels Force (0=WEAK, 1=STRONG)

    Returns:
        vol_rolling: Volatilité rolling (abs returns, window=20)
        duration: Strong Duration (compteur consécutif)
    """
    # 1. Volatilité Rolling (20 périodes)
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


def relabel_data(indicator, split, data):
    """
    Applique le RELABELING (Target Correction) sur un split.

    Au lieu de supprimer, on change Force=1 → Force=0 pour les pièges.

    Args:
        indicator: 'macd', 'rsi', ou 'cci'
        split: 'train', 'val', ou 'test'
        data: Dict contenant X_{split}, Y_{split}, Y_{split}_pred (optionnel)

    Returns:
        data_relabeled: Dict avec Y relabelé (X inchangé)
    """
    logger.info(f"\n🎯 Relabeling {indicator.upper()} [{split}]...")

    X = data[f'X_{split}']
    Y = data[f'Y_{split}'].copy()  # IMPORTANT: copie pour ne pas modifier l'original
    Y_pred = data.get(f'Y_{split}_pred', None)

    # Extraction returns (c_ret)
    idx_ret = 2 if indicator == 'cci' else 0
    returns = X[:, -1, idx_ret]

    # Labels Force AVANT relabeling
    force_labels = Y[:, 1]

    # Calcul métriques
    vol, duration = compute_features(returns, force_labels)

    # --- MASQUE 1 : UNIVERSAL (Duration Trap) ---
    mask_duration_trap = np.isin(duration, CONFIG['universal']['trap_duration'])
    trap_duration_count = mask_duration_trap.sum()
    logger.info(f"   - Pièges Duration (3-5p): {trap_duration_count} samples identifiés")

    # --- MASQUE 2 : CONDITIONAL (Vol Trap) ---
    mask_vol_trap = np.zeros(len(X), dtype=bool)
    trap_vol_count = 0

    if CONFIG['conditional'][indicator]['relabel_high_vol']:
        q4_threshold = np.percentile(vol[vol > 0], 75)
        mask_vol_trap = vol > q4_threshold
        trap_vol_count = mask_vol_trap.sum()
        logger.info(f"   - Pièges Volatilité (Q4 > {q4_threshold:.5f}): {trap_vol_count} samples identifiés")
    else:
        logger.info(f"   - Pièges Volatilité: DÉSACTIVÉ (Spécifique {indicator.upper()})")

    # MASQUE COMBINÉ des pièges
    mask_trap = mask_duration_trap | mask_vol_trap
    total_traps = mask_trap.sum()

    # --- RELABELING (PAS DE SUPPRESSION!) ---
    # Forcer Force=0 (WEAK) pour les pièges identifiés
    relabeled_count = 0
    for i in np.where(mask_trap)[0]:
        if Y[i, 1] == 1:  # Si c'était STRONG
            Y[i, 1] = 0   # → Forcer WEAK
            relabeled_count += 1

    logger.info(f"   🔄 RELABELING effectué: {relabeled_count} labels Force 1→0")
    logger.info(f"   📊 Samples totaux: {len(X)} (AUCUN supprimé)")

    # Stats distribution Force AVANT/APRÈS
    force_before = force_labels.mean()
    force_after = Y[:, 1].mean()
    delta = force_after - force_before
    logger.info(f"   📊 Force STRONG: {force_before*100:.1f}% → {force_after*100:.1f}% ({delta*100:+.1f}%)")

    # Création du dict de sortie
    data_relabeled = {
        f'X_{split}': X,           # X INCHANGÉ
        f'Y_{split}': Y            # Y RELABELÉ
    }

    if Y_pred is not None:
        # Y_pred reste inchangé (ce sont les anciennes prédictions)
        # Elles seront recalculées après réentraînement
        data_relabeled[f'Y_{split}_pred'] = Y_pred

    return data_relabeled


def main():
    parser = argparse.ArgumentParser(
        description='Phase 1 - Target Correction (RELABELING - pas suppression!)'
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
    logger.info("PHASE 1 - TARGET CORRECTION (RELABELING)")
    logger.info("=" * 80)
    logger.info("\n🎯 Principe: Hard Negative Mining")
    logger.info("   Au lieu de SUPPRIMER les pièges → Le modèle ne les voit jamais ❌")
    logger.info("   On RELABELE Force=1 → Force=0 → Le modèle APPREND à les détecter ✅")
    logger.info("\nConfiguration:")
    logger.info(f"  - Universal: Relabel Duration {CONFIG['universal']['trap_duration']} → WEAK")
    logger.info(f"  - MACD: Relabel Vol Q4 = {CONFIG['conditional']['macd']['relabel_high_vol']}")
    logger.info(f"  - CCI:  Relabel Vol Q4 = {CONFIG['conditional']['cci']['relabel_high_vol']}")
    logger.info(f"  - RSI:  Relabel Vol Q4 = {CONFIG['conditional']['rsi']['relabel_high_vol']}")
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

        # Relabeling de chaque split
        for split in splits:
            if f'X_{split}' in full_data:
                relabeled = relabel_data(ind, split, full_data)
                # Mise à jour du dict global
                for k, v in relabeled.items():
                    new_data[k] = v

        # Sauvegarde
        out_path = str(path).replace('.npz', '_relabeled.npz')
        np.savez(out_path, **new_data)
        logger.info(f"✅ Sauvegardé: {out_path}\n")

    logger.info("=" * 80)
    logger.info("🎉 RELABELING COMPLÉTÉ")
    logger.info("=" * 80)
    logger.info("\nProchaine étape:")
    logger.info("  1. Réentraîner les modèles sur datasets _relabeled.npz")
    logger.info("  2. Le modèle VERRA les pièges et APPRENDRA qu'ils sont WEAK")
    logger.info("  3. En prod, il DÉTECTERA ces patterns → Moins de faux STRONG")
    logger.info("\nGain attendu:")
    logger.info("  - Accuracy Force: Montée (le modèle apprend à reconnaître les pièges)")
    logger.info("  - Win Rate: Montée (moins de trades sur faux STRONG)")
    logger.info("  - Généralisation: Améliorée (le modèle sait gérer les cas difficiles)")
    logger.info("")


if __name__ == "__main__":
    main()
