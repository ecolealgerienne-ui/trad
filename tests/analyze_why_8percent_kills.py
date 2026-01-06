#!/usr/bin/env python3
"""
Analyse en profondeur : Pourquoi 8% d'erreurs génèrent des résultats catastrophiques ?

Question centrale : Un modèle avec 92% accuracy Direction génère un Win Rate de 14-29%
et un PnL Brut négatif. POURQUOI ?

Analyses implémentées :
1. Erreurs vs Amplitude des mouvements (les erreurs tombent-elles sur les gros moves ?)
2. Transitions vs Continuations (trade-t-on aux pires moments ?)
3. Performance par Régime de Volatilité (catastrophe en haute volatilité ?)
4. Calibration des Probabilités (probabilités sous/sur-confiantes ?)

Usage:
    python tests/analyze_why_8percent_kills.py --indicator macd --split test
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import logging
from dataclasses import dataclass
from typing import Dict, Tuple
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_data(indicator: str, split: str) -> Dict:
    """Charge les données préparées."""
    data_dir = project_root / 'data' / 'prepared'

    # Mapping indicateurs
    file_mapping = {
        'rsi': 'dataset_btc_eth_bnb_ada_ltc_rsi_dual_binary_kalman.npz',
        'macd': 'dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_kalman.npz',
        'cci': 'dataset_btc_eth_bnb_ada_ltc_cci_dual_binary_kalman.npz',
    }

    if indicator not in file_mapping:
        raise ValueError(f"Indicateur inconnu: {indicator}")

    file_path = data_dir / file_mapping[indicator]

    if not file_path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {file_path}")

    logger.info(f"📂 Chargement: {file_path}")

    data = np.load(file_path, allow_pickle=True)

    # Vérifier que Y_pred existe
    if 'Y_pred' not in data:
        raise ValueError(
            f"Y_pred absent dans {file_path}.\n"
            "   Générer d'abord les prédictions:\n"
            "   python src/train.py --data <dataset> --epochs 50\n"
            "   python src/evaluate.py --data <dataset>"
        )

    # Extraire le split
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    Y_train, Y_val, Y_test = data['Y_train'], data['Y_val'], data['Y_test']
    Y_pred_train = data['Y_pred_train']
    Y_pred_val = data['Y_pred_val']
    Y_pred_test = data['Y_pred_test']

    split_data = {
        'train': (X_train, Y_train, Y_pred_train),
        'val': (X_val, Y_val, Y_pred_val),
        'test': (X_test, Y_test, Y_pred_test)
    }

    if split not in split_data:
        raise ValueError(f"Split inconnu: {split}")

    X, Y, Y_pred = split_data[split]

    logger.info(f"  ✅ Chargé: {indicator.upper()}")
    logger.info(f"     X shape: {X.shape}")
    logger.info(f"     Y shape: {Y.shape}")
    logger.info(f"     Y_pred shape: {Y_pred.shape}")

    # Extraire c_ret (returns)
    if indicator in ['rsi', 'macd']:
        # 1 feature (c_ret) → index 0
        returns = X[:, -1, 0]
    elif indicator == 'cci':
        # 3 features (h_ret, l_ret, c_ret) → c_ret à index 2
        returns = X[:, -1, 2]

    return {
        'X': X,
        'Y': Y,
        'Y_pred': Y_pred,
        'returns': returns
    }


def analyze_errors_vs_amplitude(Y: np.ndarray, Y_pred: np.ndarray, returns: np.ndarray) -> pd.DataFrame:
    """
    Analyse 1 : Les erreurs tombent-elles sur les gros mouvements ?

    Hypothèse : 92% accuracy sur petits mouvements (±0.1%), mais 60% sur gros (±1%)
    """
    logger.info("\n" + "="*70)
    logger.info("📊 ANALYSE 1 : ERREURS vs AMPLITUDE DES MOUVEMENTS")
    logger.info("="*70)

    # Convertir probabilités en prédictions binaires
    direction_pred = (Y_pred[:, 0] > 0.5).astype(int)
    direction_true = Y[:, 0].astype(int)

    # Calculer erreurs
    errors = (direction_pred != direction_true)

    # Amplitude absolue des mouvements
    returns_abs = np.abs(returns)

    # Définir bins d'amplitude
    bins = [0, 0.002, 0.005, 0.01, 0.02, np.inf]
    bin_labels = ['0-0.2%', '0.2-0.5%', '0.5-1%', '1-2%', '>2%']

    results = []

    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (returns_abs >= low) & (returns_abs < high)
        n_samples = mask.sum()

        if n_samples < 10:
            continue

        # Accuracy dans ce bin
        accuracy = (~errors[mask]).mean()

        # Distribution des samples
        pct_samples = n_samples / len(returns) * 100

        # PnL moyen si on tradait ce bin
        # Simuler : si Direction correcte → gain return, sinon perte
        correct_trades = ~errors[mask]
        pnl_per_trade = np.where(correct_trades, returns_abs[mask], -returns_abs[mask])
        avg_pnl = pnl_per_trade.mean() * 100

        results.append({
            'Amplitude': bin_labels[i],
            'N_Samples': n_samples,
            'Pct_Samples': pct_samples,
            'Accuracy': accuracy * 100,
            'Avg_PnL_per_trade': avg_pnl
        })

        logger.info(f"\n📍 Amplitude {bin_labels[i]}:")
        logger.info(f"   Samples:      {n_samples:,} ({pct_samples:.1f}%)")
        logger.info(f"   Accuracy:     {accuracy*100:.2f}%")
        logger.info(f"   Avg PnL/trade: {avg_pnl:+.3f}%")

    df = pd.DataFrame(results)

    # Diagnostics
    logger.info("\n🔍 DIAGNOSTIC:")

    # Comparer accuracy petits vs gros mouvements
    small_moves = df[df['Amplitude'].isin(['0-0.2%', '0.2-0.5%'])]['Accuracy'].mean()
    big_moves = df[df['Amplitude'].isin(['1-2%', '>2%'])]['Accuracy'].mean()
    gap = small_moves - big_moves

    logger.info(f"   Accuracy Petits Mouvements (<0.5%): {small_moves:.1f}%")
    logger.info(f"   Accuracy Gros Mouvements (>1%):     {big_moves:.1f}%")
    logger.info(f"   Gap:                                {gap:.1f} points")

    if gap > 15:
        logger.info(f"\n⚠️  PROBLÈME IDENTIFIÉ : Accuracy chute de {gap:.1f} points sur gros mouvements")
        logger.info("   → Le modèle rate ce qui COMPTE (les gros moves)")
        logger.info("   → Solution : Filtrer les gros mouvements ou améliorer le modèle sur eux")
    elif gap < 5:
        logger.info("\n✅ Accuracy stable sur toutes amplitudes")

    return df


def analyze_transitions_vs_continuations(Y: np.ndarray, Y_pred: np.ndarray, returns: np.ndarray) -> Dict:
    """
    Analyse 2 : Trade-t-on aux pires moments (transitions) ?

    Hypothèse : Accuracy 95% sur continuations, mais 60% sur transitions (tops/bottoms)
    """
    logger.info("\n" + "="*70)
    logger.info("📊 ANALYSE 2 : TRANSITIONS vs CONTINUATIONS")
    logger.info("="*70)

    direction_pred = (Y_pred[:, 0] > 0.5).astype(int)
    direction_true = Y[:, 0].astype(int)
    force_true = Y[:, 1].astype(int)

    errors = (direction_pred != direction_true)

    # Identifier transitions (changement de direction)
    direction_changes = np.zeros(len(direction_true), dtype=bool)
    direction_changes[1:] = (direction_true[1:] != direction_true[:-1])

    # Transitions = changements avec Force STRONG (moments où on traderait)
    transitions = direction_changes & (force_true == 1)
    continuations = (~direction_changes) & (force_true == 1)

    # Accuracy sur chaque catégorie
    n_transitions = transitions.sum()
    n_continuations = continuations.sum()

    if n_transitions < 10 or n_continuations < 10:
        logger.warning("⚠️  Pas assez de samples pour cette analyse")
        return {}

    acc_transitions = (~errors[transitions]).mean() * 100
    acc_continuations = (~errors[continuations]).mean() * 100
    gap = acc_continuations - acc_transitions

    # PnL moyen par catégorie
    correct_trans = ~errors[transitions]
    correct_cont = ~errors[continuations]

    pnl_transitions = np.where(correct_trans, np.abs(returns[transitions]), -np.abs(returns[transitions])).mean() * 100
    pnl_continuations = np.where(correct_cont, np.abs(returns[continuations]), -np.abs(returns[continuations])).mean() * 100

    logger.info(f"\n📍 TRANSITIONS (Force STRONG + Direction change):")
    logger.info(f"   Samples:      {n_transitions:,}")
    logger.info(f"   Accuracy:     {acc_transitions:.2f}%")
    logger.info(f"   Avg PnL/trade: {pnl_transitions:+.3f}%")

    logger.info(f"\n📍 CONTINUATIONS (Force STRONG + Direction stable):")
    logger.info(f"   Samples:      {n_continuations:,}")
    logger.info(f"   Accuracy:     {acc_continuations:.2f}%")
    logger.info(f"   Avg PnL/trade: {pnl_continuations:+.3f}%")

    logger.info(f"\n🔍 DIAGNOSTIC:")
    logger.info(f"   Gap Accuracy: {gap:+.1f} points")

    if gap > 15:
        logger.info(f"\n⚠️  PROBLÈME IDENTIFIÉ : Accuracy {gap:.1f} points plus faible sur transitions")
        logger.info("   → On trade aux PIRES moments (tops/bottoms)")
        logger.info("   → Solutions :")
        logger.info("      1. Attendre N périodes APRÈS le changement de Direction")
        logger.info("      2. Timeframe plus long (15min/30min) - transitions plus claires")
        logger.info("      3. Ne trader QUE les continuations (abandonner les transitions)")
    elif gap < 5:
        logger.info("\n✅ Accuracy similaire sur transitions et continuations")

    return {
        'n_transitions': n_transitions,
        'n_continuations': n_continuations,
        'acc_transitions': acc_transitions,
        'acc_continuations': acc_continuations,
        'gap': gap,
        'pnl_transitions': pnl_transitions,
        'pnl_continuations': pnl_continuations
    }


def analyze_volatility_regimes(Y: np.ndarray, Y_pred: np.ndarray, returns: np.ndarray) -> pd.DataFrame:
    """
    Analyse 3 : Performance catastrophique en haute volatilité ?

    Hypothèse : Accuracy 95% en volatilité normale, 55% en haute volatilité
    """
    logger.info("\n" + "="*70)
    logger.info("📊 ANALYSE 3 : PERFORMANCE PAR RÉGIME DE VOLATILITÉ")
    logger.info("="*70)

    direction_pred = (Y_pred[:, 0] > 0.5).astype(int)
    direction_true = Y[:, 0].astype(int)

    errors = (direction_pred != direction_true)

    # Calculer volatilité rolling (50 périodes = ~4h pour 5min)
    volatility = pd.Series(returns).rolling(window=50, min_periods=20).std().values

    # Définir régimes par quantiles
    vol_quantiles = np.nanquantile(volatility, [0.33, 0.67])

    regimes = np.full(len(volatility), 'MED')
    regimes[volatility < vol_quantiles[0]] = 'LOW'
    regimes[volatility >= vol_quantiles[1]] = 'HIGH'

    results = []

    for regime in ['LOW', 'MED', 'HIGH']:
        mask = (regimes == regime) & ~np.isnan(volatility)
        n_samples = mask.sum()

        if n_samples < 10:
            continue

        accuracy = (~errors[mask]).mean() * 100
        pct_samples = n_samples / len(returns) * 100

        # PnL moyen
        correct = ~errors[mask]
        pnl_per_trade = np.where(correct, np.abs(returns[mask]), -np.abs(returns[mask])).mean() * 100

        # Volatilité moyenne du régime
        avg_vol = np.mean(volatility[mask]) * 100

        results.append({
            'Regime': regime,
            'N_Samples': n_samples,
            'Pct_Samples': pct_samples,
            'Avg_Volatility': avg_vol,
            'Accuracy': accuracy,
            'Avg_PnL_per_trade': pnl_per_trade
        })

        logger.info(f"\n📍 Régime {regime} Volatility:")
        logger.info(f"   Samples:       {n_samples:,} ({pct_samples:.1f}%)")
        logger.info(f"   Avg Vol:       {avg_vol:.3f}%")
        logger.info(f"   Accuracy:      {accuracy:.2f}%")
        logger.info(f"   Avg PnL/trade: {pnl_per_trade:+.3f}%")

    df = pd.DataFrame(results)

    # Diagnostic
    logger.info("\n🔍 DIAGNOSTIC:")

    if len(df) >= 2:
        acc_low = df[df['Regime'] == 'LOW']['Accuracy'].values[0] if 'LOW' in df['Regime'].values else None
        acc_high = df[df['Regime'] == 'HIGH']['Accuracy'].values[0] if 'HIGH' in df['Regime'].values else None

        if acc_low is not None and acc_high is not None:
            gap = acc_low - acc_high
            logger.info(f"   Accuracy LOW Vol:  {acc_low:.1f}%")
            logger.info(f"   Accuracy HIGH Vol: {acc_high:.1f}%")
            logger.info(f"   Gap:               {gap:.1f} points")

            if gap > 15:
                logger.info(f"\n⚠️  PROBLÈME IDENTIFIÉ : Accuracy chute de {gap:.1f} points en haute volatilité")
                logger.info("   → Le modèle n'a pas de features pour gérer les market regimes")
                logger.info("   → Solutions :")
                logger.info("      1. Ajouter volatility features (ATR, rolling std)")
                logger.info("      2. Filtrer : ne pas trader si volatility > quantile 0.8")
                logger.info("      3. Modèle séparé pour haute volatilité")
            elif gap < 5:
                logger.info("\n✅ Accuracy stable sur tous régimes de volatilité")

    return df


def analyze_probability_calibration(Y: np.ndarray, Y_pred: np.ndarray) -> pd.DataFrame:
    """
    Analyse 4 : Les probabilités sont-elles bien calibrées ?

    Hypothèse : Le modèle dit "prob 0.7" mais c'est en fait 55% ou 85%
    """
    logger.info("\n" + "="*70)
    logger.info("📊 ANALYSE 4 : CALIBRATION DES PROBABILITÉS")
    logger.info("="*70)

    direction_pred_prob = Y_pred[:, 0]
    direction_true = Y[:, 0].astype(int)

    # Bins de probabilités
    prob_bins = np.arange(0.5, 1.0, 0.05)

    results = []

    logger.info("\n📍 Courbe de Calibration (Direction):")
    logger.info("   Si bien calibré : prob 0.7 → accuracy 70%\n")

    for prob in prob_bins:
        # Échantillons dans ce bin de probabilité
        mask = (direction_pred_prob >= prob) & (direction_pred_prob < prob+0.05)
        n_samples = mask.sum()

        if n_samples < 50:  # Pas assez de samples
            continue

        # Accuracy réelle dans ce bin
        predictions_binary = (direction_pred_prob[mask] > 0.5).astype(int)
        actual_accuracy = (predictions_binary == direction_true[mask]).mean() * 100

        # Probabilité moyenne du bin
        avg_prob = direction_pred_prob[mask].mean() * 100

        # Écart calibration
        calibration_error = actual_accuracy - avg_prob

        results.append({
            'Prob_Bin': f"{prob:.2f}-{prob+0.05:.2f}",
            'Avg_Prob': avg_prob,
            'N_Samples': n_samples,
            'True_Accuracy': actual_accuracy,
            'Calibration_Error': calibration_error
        })

        logger.info(f"   Prob {prob:.2f}-{prob+0.05:.2f}: "
                   f"N={n_samples:5,}, "
                   f"True Acc={actual_accuracy:5.1f}%, "
                   f"Error={calibration_error:+5.1f}%")

    df = pd.DataFrame(results)

    # Diagnostic
    logger.info("\n🔍 DIAGNOSTIC:")

    if len(df) > 0:
        avg_error = df['Calibration_Error'].abs().mean()
        max_error = df['Calibration_Error'].abs().max()

        logger.info(f"   Erreur Moyenne:  {avg_error:.1f}%")
        logger.info(f"   Erreur Maximale: {max_error:.1f}%")

        if avg_error > 10:
            logger.info(f"\n⚠️  PROBLÈME IDENTIFIÉ : Probabilités MAL calibrées (erreur {avg_error:.1f}%)")
            logger.info("   → Le modèle est sur-confiant ou sous-confiant")
            logger.info("   → Solutions :")
            logger.info("      1. Calibration isotonique ou Platt scaling")
            logger.info("      2. Augmenter threshold (ex: 0.7 au lieu de 0.5)")
            logger.info("      3. Temperature scaling dans la loss")
        elif avg_error < 5:
            logger.info("\n✅ Probabilités bien calibrées")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Analyse : Pourquoi 8% d'erreurs génèrent des résultats catastrophiques ?"
    )
    parser.add_argument(
        '--indicator',
        required=True,
        choices=['rsi', 'macd', 'cci'],
        help="Indicateur à analyser"
    )
    parser.add_argument(
        '--split',
        default='test',
        choices=['train', 'val', 'test'],
        help="Split à analyser (défaut: test)"
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help="Sauvegarder les graphiques"
    )

    args = parser.parse_args()

    logger.info("="*70)
    logger.info(f"🔬 ANALYSE APPROFONDIE : {args.indicator.upper()} ({args.split})")
    logger.info("="*70)
    logger.info("\nQuestion : Pourquoi un modèle à 92% accuracy génère-t-il des")
    logger.info("           résultats catastrophiques (Win Rate 14-29%, PnL négatif) ?")

    # Charger données
    data = load_data(args.indicator, args.split)
    Y = data['Y']
    Y_pred = data['Y_pred']
    returns = data['returns']

    # Accuracy globale
    direction_pred = (Y_pred[:, 0] > 0.5).astype(int)
    direction_true = Y[:, 0].astype(int)
    global_accuracy = (direction_pred == direction_true).mean() * 100

    logger.info(f"\n📊 Accuracy Globale Direction: {global_accuracy:.2f}%")

    # Analyse 1 : Erreurs vs Amplitude
    df_amplitude = analyze_errors_vs_amplitude(Y, Y_pred, returns)

    # Analyse 2 : Transitions vs Continuations
    results_transitions = analyze_transitions_vs_continuations(Y, Y_pred, returns)

    # Analyse 3 : Régimes de Volatilité
    df_volatility = analyze_volatility_regimes(Y, Y_pred, returns)

    # Analyse 4 : Calibration
    df_calibration = analyze_probability_calibration(Y, Y_pred)

    # Résumé final
    logger.info("\n" + "="*70)
    logger.info("📋 RÉSUMÉ DES ANALYSES")
    logger.info("="*70)

    logger.info("\n1️⃣  ERREURS vs AMPLITUDE:")
    logger.info(f"   → Voir tableau ci-dessus")

    if results_transitions:
        logger.info("\n2️⃣  TRANSITIONS vs CONTINUATIONS:")
        logger.info(f"   → Gap: {results_transitions['gap']:+.1f} points")

    logger.info("\n3️⃣  RÉGIMES DE VOLATILITÉ:")
    logger.info(f"   → Voir tableau ci-dessus")

    logger.info("\n4️⃣  CALIBRATION:")
    logger.info(f"   → Voir tableau ci-dessus")

    logger.info("\n" + "="*70)
    logger.info("✅ Analyse terminée")
    logger.info("="*70)


if __name__ == '__main__':
    main()
