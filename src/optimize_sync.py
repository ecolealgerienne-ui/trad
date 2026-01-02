"""
Optimisation de la synchronisation des indicateurs.

Ce script trouve les parametres optimaux pour chaque indicateur
afin de synchroniser leurs signaux avec la direction du Close.

Metriques utilisees:
1. Concordance (30%): % de labels identiques entre indicateur et close
2. Anticipation (40%): lag temporel (negatif = indicateur en avance = mieux)
3. Pivot Accuracy (30%): % de match sur les points de retournement

Score = 0.3*Concordance + 0.4*Anticipation + 0.3*PivotAccuracy

Usage:
    python src/optimize_sync.py --assets BTC ETH BNB --val-assets ADA LTC
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Imports locaux
from constants import (
    AVAILABLE_ASSETS_5M,
    TRIM_EDGES,
    KALMAN_PROCESS_VAR,
    KALMAN_MEASURE_VAR,
)
from filters import kalman_filter
from indicators import (
    calculate_rsi,
    calculate_cci,
    calculate_bollinger_bands,
    calculate_macd,
    normalize_cci,
    normalize_macd_histogram,
)


# =============================================================================
# GRILLES DE PARAMETRES A TESTER
# =============================================================================

PARAM_GRIDS = {
    'RSI': {
        'period': [14, 12, 10, 8, 7, 6, 5, 4, 3]
    },
    'CCI': {
        'period': [20, 15, 12, 10, 8, 7, 6, 5]
    },
    'BOL': {
        'period': [20, 15, 12, 10, 8, 6]
    },
    'MACD': {
        'fast': [12, 10, 8, 6, 5, 4, 3],
        'slow': [26, 20, 16, 13, 10, 8],
    }
}


@dataclass
class SyncResult:
    """Resultat de synchronisation pour un jeu de parametres."""
    indicator: str
    params: Dict
    concordance: float       # % de labels identiques (stabilite)
    anticipation: float      # Lag (negatif = en avance = mieux)
    pivot_accuracy: float    # % de match sur les pivots (pertinence)
    composite_score: float   # Score composite 30/40/30
    n_samples: int


def load_asset_data(asset: str) -> pd.DataFrame:
    """Charge les donnees d'un asset."""
    file_path = AVAILABLE_ASSETS_5M.get(asset)
    if not file_path:
        raise ValueError(f"Asset {asset} non disponible. Choix: {list(AVAILABLE_ASSETS_5M.keys())}")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier non trouve: {path}")

    df = pd.read_csv(path)

    # Normaliser les noms de colonnes
    df.columns = df.columns.str.lower()

    # Trim edges
    if TRIM_EDGES > 0:
        df = df.iloc[TRIM_EDGES:-TRIM_EDGES].reset_index(drop=True)

    logger.info(f"  {asset}: {len(df):,} bougies chargees")
    return df


def calculate_slope_labels(signal: np.ndarray) -> np.ndarray:
    """
    Calcule les labels binaires a partir de la pente du signal.

    Label[t] = 1 si signal[t-1] > signal[t-2] (pente positive)
    Label[t] = 0 si signal[t-1] <= signal[t-2] (pente negative ou nulle)

    IMPORTANT: On compare t-1 vs t-2 car a l'instant t, la bougie t
    n'est pas encore fermee. On utilise donc les 2 dernieres bougies
    fermees pour determiner la direction.

    Le trade est execute a Open[t+1].
    """
    labels = np.zeros(len(signal), dtype=int)

    # A partir de t=2 (besoin de t-1 et t-2)
    for t in range(2, len(signal)):
        if signal[t-1] > signal[t-2]:
            labels[t] = 1  # Pente haussiere
        else:
            labels[t] = 0  # Pente baissiere

    return labels


def calculate_reference_labels(close: np.ndarray,
                               process_var: float = KALMAN_PROCESS_VAR,
                               measure_var: float = KALMAN_MEASURE_VAR) -> np.ndarray:
    """
    Calcule les labels de reference a partir du Close filtre par Kalman.
    """
    filtered_close = kalman_filter(close, process_var, measure_var)
    return calculate_slope_labels(filtered_close)


def calculate_indicator_labels(df: pd.DataFrame,
                               indicator: str,
                               params: Dict,
                               process_var: float = KALMAN_PROCESS_VAR,
                               measure_var: float = KALMAN_MEASURE_VAR) -> np.ndarray:
    """
    Calcule les labels pour un indicateur avec des parametres donnes.
    """
    if indicator == 'RSI':
        values = calculate_rsi(df['close'], period=params['period'])

    elif indicator == 'CCI':
        values = calculate_cci(df['high'], df['low'], df['close'], period=params['period'])
        values = normalize_cci(values)

    elif indicator == 'BOL':
        bb = calculate_bollinger_bands(df['close'], period=params['period'])
        # %B = position du prix dans les bandes
        percent_b = (df['close'].values - bb['lower']) / (bb['upper'] - bb['lower'])
        values = np.clip(percent_b * 100, 0, 100)

    elif indicator == 'MACD':
        macd = calculate_macd(df['close'],
                              fast_period=params['fast'],
                              slow_period=params['slow'],
                              signal_period=9)
        values = normalize_macd_histogram(macd['histogram'])
    else:
        raise ValueError(f"Indicateur inconnu: {indicator}")

    # Filtrer avec Kalman et calculer les labels
    # Gerer les NaN
    values = pd.Series(values).ffill().fillna(50).values
    filtered = kalman_filter(values, process_var, measure_var)
    return calculate_slope_labels(filtered)


def calculate_concordance(labels1: np.ndarray, labels2: np.ndarray) -> float:
    """Calcule le taux de concordance (% de labels identiques)."""
    # Ignorer les premiers elements (warm-up)
    start_idx = 50
    l1 = labels1[start_idx:]
    l2 = labels2[start_idx:]
    return np.mean(l1 == l2)


def calculate_lag(labels1: np.ndarray, labels2: np.ndarray, max_lag: int = 10) -> int:
    """
    Calcule le lag optimal entre deux series de labels.

    Retourne:
        lag negatif = labels1 en avance sur labels2
        lag positif = labels1 en retard sur labels2
    """
    start_idx = 50
    l1 = labels1[start_idx:].astype(float)
    l2 = labels2[start_idx:].astype(float)

    # Centrer les series
    l1 = l1 - l1.mean()
    l2 = l2 - l2.mean()

    best_lag = 0
    best_corr = -1

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = np.corrcoef(l1[-lag:], l2[:lag])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(l1[:-lag], l2[lag:])[0, 1]
        else:
            corr = np.corrcoef(l1, l2)[0, 1]

        if not np.isnan(corr) and corr > best_corr:
            best_corr = corr
            best_lag = lag

    return best_lag


def calculate_pivot_accuracy(labels_ind: np.ndarray, labels_ref: np.ndarray) -> float:
    """
    Calcule le % de match sur les PIVOTS (changements de direction).

    Les pivots sont les points ou on gagne/perd de l'argent.
    Un indicateur avec 95% de concordance globale mais qui rate les pivots
    est moins utile qu'un indicateur a 80% qui capte les retournements.

    IMPORTANT: np.diff donne l'indice AVANT le changement.
    Si le prix change a l'indice T, diff le marque a T-1.
    On compare donc a pivots_ref + 1 (premiere bougie nouvelle direction).
    """
    start_idx = 50
    l_ind = labels_ind[start_idx:]
    l_ref = labels_ref[start_idx:]

    # Detecter les pivots dans la reference (Close)
    # np.diff != 0 indique un changement de direction
    pivot_indices = np.where(np.diff(l_ref) != 0)[0]

    if len(pivot_indices) == 0:
        return 0.5  # Pas de pivot = score neutre

    # Comparer a l'indice + 1 (premiere bougie de la nouvelle direction)
    # C'est a cet instant que l'ordre est execute a Open(t+1)
    valid_pivots = pivot_indices[pivot_indices + 1 < len(l_ref)]

    if len(valid_pivots) == 0:
        return 0.5

    # Accuracy sur les pivots
    matches = l_ind[valid_pivots + 1] == l_ref[valid_pivots + 1]
    return np.mean(matches)


def calculate_composite_score(concordance: float,
                              anticipation: int,
                              pivot_accuracy: float,
                              w_concordance: float = 0.3,
                              w_anticipation: float = 0.4,
                              w_pivot: float = 0.3) -> float:
    """
    Calcule le score composite (30/40/30).

    - concordance (30%): Stabilite - l'indicateur ne devient pas "fou"
    - anticipation (40%): Vitesse - privilegie les indicateurs qui sentent le mouvement
    - pivot_accuracy (30%): Pertinence - match sur les points de retournement

    Un indicateur ideal:
    - Est en avance sur le Close (anticipation negative)
    - Match bien sur les pivots (la ou on gagne l'argent)
    - Reste stable entre les pivots (pas de bruit)
    """
    # Normaliser anticipation: -10 -> 1.0, 0 -> 0.5, +10 -> 0.0
    # On veut que les indicateurs en avance (lag negatif) aient un meilleur score
    anticipation_score = max(0, min(1, 0.5 - anticipation / 20))

    score = (
        w_concordance * concordance +
        w_anticipation * anticipation_score +
        w_pivot * pivot_accuracy
    )
    return score


def evaluate_params(df: pd.DataFrame,
                    ref_labels: np.ndarray,
                    indicator: str,
                    params: Dict) -> SyncResult:
    """Evalue un jeu de parametres pour un indicateur."""

    # Calculer les labels pour cet indicateur
    try:
        ind_labels = calculate_indicator_labels(df, indicator, params)
    except Exception as e:
        logger.warning(f"Erreur pour {indicator} avec {params}: {e}")
        return SyncResult(
            indicator=indicator,
            params=params,
            concordance=0.0,
            anticipation=0,
            pivot_accuracy=0.0,
            composite_score=0.0,
            n_samples=0
        )

    # Calculer les 3 metriques
    concordance = calculate_concordance(ind_labels, ref_labels)
    anticipation = calculate_lag(ind_labels, ref_labels)
    pivot_acc = calculate_pivot_accuracy(ind_labels, ref_labels)
    score = calculate_composite_score(concordance, anticipation, pivot_acc)

    return SyncResult(
        indicator=indicator,
        params=params,
        concordance=concordance,
        anticipation=anticipation,
        pivot_accuracy=pivot_acc,
        composite_score=score,
        n_samples=len(df)
    )


def optimize_indicator(dfs: Dict[str, pd.DataFrame],
                       ref_labels_dict: Dict[str, np.ndarray],
                       indicator: str) -> List[SyncResult]:
    """
    Optimise les parametres d'un indicateur sur plusieurs assets.

    Retourne la liste de tous les resultats tries par score.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Optimisation {indicator}")
    logger.info(f"{'='*60}")

    param_grid = PARAM_GRIDS[indicator]
    all_results = []

    # Generer toutes les combinaisons de parametres
    if indicator == 'MACD':
        param_combinations = []
        for fast in param_grid['fast']:
            for slow in param_grid['slow']:
                if fast < slow:  # MACD fast doit etre < slow
                    param_combinations.append({'fast': fast, 'slow': slow})
    else:
        param_combinations = [{'period': p} for p in param_grid['period']]

    # Tester chaque combinaison
    for params in param_combinations:
        scores_per_asset = []

        for asset, df in dfs.items():
            ref_labels = ref_labels_dict[asset]
            result = evaluate_params(df, ref_labels, indicator, params)
            scores_per_asset.append(result)

        # Moyenne des scores sur tous les assets
        avg_concordance = np.mean([r.concordance for r in scores_per_asset])
        avg_anticipation = np.mean([r.anticipation for r in scores_per_asset])
        avg_pivot_acc = np.mean([r.pivot_accuracy for r in scores_per_asset])
        avg_score = np.mean([r.composite_score for r in scores_per_asset])
        total_samples = sum([r.n_samples for r in scores_per_asset])

        avg_result = SyncResult(
            indicator=indicator,
            params=params,
            concordance=avg_concordance,
            anticipation=avg_anticipation,
            pivot_accuracy=avg_pivot_acc,
            composite_score=avg_score,
            n_samples=total_samples
        )
        all_results.append(avg_result)

        # Afficher progression
        if indicator == 'MACD':
            params_str = f"fast={params['fast']}, slow={params['slow']}"
        else:
            params_str = f"period={params['period']}"
        logger.info(f"  {params_str:20s} | Conc: {avg_concordance:.3f} | "
                   f"Lag: {avg_anticipation:+3.0f} | Pivot: {avg_pivot_acc:.3f} | "
                   f"Score: {avg_score:.3f}")

    # Trier par score decroissant
    all_results.sort(key=lambda x: x.composite_score, reverse=True)

    return all_results


def validate_on_assets(optimal_params: Dict[str, Dict],
                       val_assets: List[str]) -> Dict[str, float]:
    """
    Valide les parametres optimaux sur des assets non vus.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Validation sur: {val_assets}")
    logger.info(f"{'='*60}")

    # Charger les assets de validation
    val_dfs = {}
    val_ref_labels = {}

    for asset in val_assets:
        df = load_asset_data(asset)
        val_dfs[asset] = df
        val_ref_labels[asset] = calculate_reference_labels(df['close'].values)

    # Evaluer chaque indicateur avec ses parametres optimaux
    validation_scores = {}

    for indicator, params in optimal_params.items():
        scores = []
        for asset, df in val_dfs.items():
            ref_labels = val_ref_labels[asset]
            result = evaluate_params(df, ref_labels, indicator, params)
            scores.append(result.composite_score)
            logger.info(f"  {indicator} sur {asset}: "
                       f"Conc={result.concordance:.3f}, "
                       f"Lag={result.anticipation:+d}, "
                       f"Pivot={result.pivot_accuracy:.3f}, "
                       f"Score={result.composite_score:.3f}")

        validation_scores[indicator] = np.mean(scores)

    return validation_scores


def parse_args():
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(
        description='Optimisation de la synchronisation des indicateurs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--assets', '-a', nargs='+',
                        default=['BTC', 'ETH', 'BNB'],
                        help='Assets pour optimisation')

    parser.add_argument('--val-assets', '-v', nargs='+',
                        default=['ADA', 'LTC'],
                        help='Assets pour validation (cross-validation)')

    parser.add_argument('--output', '-o', type=str,
                        default='results/sync_optimization.json',
                        help='Fichier de sortie pour les resultats')

    parser.add_argument('--indicators', '-i', nargs='+',
                        default=['RSI', 'CCI', 'BOL', 'MACD'],
                        choices=['RSI', 'CCI', 'BOL', 'MACD'],
                        help='Indicateurs a optimiser')

    return parser.parse_args()


def main():
    """Pipeline principal d'optimisation."""
    args = parse_args()

    logger.info("="*80)
    logger.info("OPTIMISATION DE LA SYNCHRONISATION DES INDICATEURS")
    logger.info("="*80)
    logger.info(f"Assets optimisation: {args.assets}")
    logger.info(f"Assets validation:   {args.val_assets}")
    logger.info(f"Indicateurs:         {args.indicators}")

    # =========================================================================
    # 1. CHARGER LES DONNEES
    # =========================================================================
    logger.info("\n1. Chargement des donnees...")

    dfs = {}
    ref_labels_dict = {}

    for asset in args.assets:
        df = load_asset_data(asset)
        dfs[asset] = df
        ref_labels_dict[asset] = calculate_reference_labels(df['close'].values)

        # Stats des labels
        buy_pct = ref_labels_dict[asset].mean() * 100
        logger.info(f"     Labels {asset}: {buy_pct:.1f}% UP / {100-buy_pct:.1f}% DOWN")

    # =========================================================================
    # 2. OPTIMISER CHAQUE INDICATEUR
    # =========================================================================
    logger.info("\n2. Optimisation des parametres...")

    optimal_params = {}
    best_results = {}

    for indicator in args.indicators:
        results = optimize_indicator(dfs, ref_labels_dict, indicator)

        # Garder le meilleur
        best = results[0]
        optimal_params[indicator] = best.params
        best_results[indicator] = {
            'params': best.params,
            'concordance': best.concordance,
            'anticipation': best.anticipation,
            'pivot_accuracy': best.pivot_accuracy,
            'composite_score': best.composite_score
        }

        logger.info(f"\n  MEILLEUR {indicator}: {best.params}")
        logger.info(f"    Concordance:    {best.concordance:.3f} (30%)")
        logger.info(f"    Anticipation:   {best.anticipation:+d} steps (40%)")
        logger.info(f"    Pivot Accuracy: {best.pivot_accuracy:.3f} (30%)")
        logger.info(f"    Score Total:    {best.composite_score:.3f}")

    # =========================================================================
    # 3. VALIDATION CROISEE
    # =========================================================================
    if args.val_assets:
        validation_scores = validate_on_assets(optimal_params, args.val_assets)

        logger.info("\n" + "="*60)
        logger.info("RESUME VALIDATION")
        logger.info("="*60)
        for indicator, score in validation_scores.items():
            train_score = best_results[indicator]['composite_score']
            gap = train_score - score
            status = "OK" if gap < 0.05 else "ATTENTION"
            logger.info(f"  {indicator}: Train={train_score:.3f} | Val={score:.3f} | "
                       f"Gap={gap:+.3f} [{status}]")

    # =========================================================================
    # 4. RESUME FINAL
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PARAMETRES OPTIMAUX")
    logger.info("="*80)

    print("\n# Copier dans constants.py:\n")
    for indicator, params in optimal_params.items():
        if indicator == 'RSI':
            print(f"RSI_PERIOD = {params['period']}")
        elif indicator == 'CCI':
            print(f"CCI_PERIOD = {params['period']}")
        elif indicator == 'BOL':
            print(f"BOL_PERIOD = {params['period']}")
        elif indicator == 'MACD':
            print(f"MACD_FAST = {params['fast']}")
            print(f"MACD_SLOW = {params['slow']}")

    # =========================================================================
    # 5. SAUVEGARDER RESULTATS
    # =========================================================================
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'timestamp': datetime.now().isoformat(),
        'optimization_assets': args.assets,
        'validation_assets': args.val_assets,
        'optimal_params': optimal_params,
        'detailed_results': best_results,
        'validation_scores': validation_scores if args.val_assets else None
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResultats sauvegardes: {output_path}")

    logger.info("\n" + "="*80)
    logger.info("OPTIMISATION TERMINEE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
