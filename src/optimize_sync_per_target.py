"""
Optimisation de la synchronisation par indicateur cible.

Ce script trouve les parametres optimaux pour les indicateurs FEATURES
afin de synchroniser leurs signaux avec un indicateur CIBLE.

Exemple: --target rsi
  - Reference = Kalman(RSI) avec parametres par defaut
  - Optimise CCI et MACD pour synchroniser avec Kalman(RSI)

Usage:
    python src/optimize_sync_per_target.py --target rsi --assets BTC ETH BNB
    python src/optimize_sync_per_target.py --target cci --assets BTC ETH BNB
    python src/optimize_sync_per_target.py --target macd --assets BTC ETH BNB
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
import torch

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device global
DEVICE = torch.device('cpu')

# Imports locaux
from constants import (
    AVAILABLE_ASSETS_5M,
    TRIM_EDGES,
    KALMAN_PROCESS_VAR,
    KALMAN_MEASURE_VAR,
    RSI_PERIOD,
    CCI_PERIOD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
)
from filters import kalman_filter
from indicators import (
    calculate_rsi,
    calculate_cci,
    calculate_macd,
    normalize_cci,
    normalize_macd_histogram,
)


# =============================================================================
# GRILLES DE PARAMETRES A TESTER
# =============================================================================

# Grilles avec pas de ~20%, limite ±60% (3 pas) autour du defaut
PARAM_GRIDS = {
    'RSI': {
        # Defaut 22: ±60% → [9, 35], 3 pas de 20%
        'period': [35, 26, 22, 18, 9]
    },
    'CCI': {
        # Defaut 32: ±60% → [13, 51], 3 pas de 20%
        'period': [51, 38, 32, 26, 13]
    },
    'MACD': {
        # Defaut fast=8: ±60% → [3, 13]
        'fast': [13, 10, 8, 6, 3],
        # Defaut slow=42: ±60% → [17, 67]
        'slow': [67, 50, 42, 34, 17],
    }
}

# Plage de lag reduite: 3 pas arriere, 2 pas avant
LAG_RANGE = (-3, 3)  # range(-3, 3) = [-3, -2, -1, 0, 1, 2]

# Parametres par defaut pour l'indicateur cible
DEFAULT_PARAMS = {
    'RSI': {'period': RSI_PERIOD},
    'CCI': {'period': CCI_PERIOD},
    'MACD': {'fast': MACD_FAST, 'slow': MACD_SLOW},
}


@dataclass
class SyncResult:
    """Resultat de synchronisation pour un jeu de parametres."""
    indicator: str
    params: Dict
    concordance: float
    anticipation: float
    pivot_accuracy: float
    composite_score: float
    n_samples: int


def load_asset_data(asset: str, max_samples: int = None, step: int = 1) -> pd.DataFrame:
    """Charge les donnees d'un asset."""
    file_path = AVAILABLE_ASSETS_5M.get(asset)
    if not file_path:
        raise ValueError(f"Asset {asset} non disponible. Choix: {list(AVAILABLE_ASSETS_5M.keys())}")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier non trouve: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()

    if TRIM_EDGES > 0:
        df = df.iloc[TRIM_EDGES:-TRIM_EDGES].reset_index(drop=True)

    if max_samples and len(df) > max_samples:
        df = df.iloc[-max_samples:].reset_index(drop=True)

    if step > 1:
        df = df.iloc[::step].reset_index(drop=True)

    logger.info(f"  {asset}: {len(df):,} bougies chargees")
    return df


def calculate_indicator_values(df: pd.DataFrame,
                                indicator: str,
                                params: Dict) -> np.ndarray:
    """Calcule les valeurs brutes d'un indicateur."""
    if indicator == 'RSI':
        values = calculate_rsi(df['close'], period=params['period'])

    elif indicator == 'CCI':
        values = calculate_cci(df['high'], df['low'], df['close'], period=params['period'])
        values = normalize_cci(values)

    elif indicator == 'MACD':
        macd = calculate_macd(df['close'],
                              fast_period=params['fast'],
                              slow_period=params['slow'],
                              signal_period=MACD_SIGNAL)
        values = normalize_macd_histogram(macd['histogram'])
    else:
        raise ValueError(f"Indicateur inconnu: {indicator}")

    values = pd.Series(values).ffill().fillna(50).values
    return values


def to_gpu_tensor(data: np.ndarray, start_idx: int = 50, dtype=torch.float32) -> torch.Tensor:
    """Convertit un array numpy en tensor GPU."""
    return torch.tensor(data[start_idx:], device=DEVICE, dtype=dtype)


def calculate_slope_labels_gpu(signal_gpu: torch.Tensor) -> torch.Tensor:
    """Calcule les labels binaires sur GPU."""
    labels = torch.zeros(len(signal_gpu), device=DEVICE, dtype=torch.int32)
    if len(signal_gpu) > 2:
        labels[2:] = (signal_gpu[1:-1] > signal_gpu[:-2]).int()
    return labels


def calculate_composite_score(concordance: float,
                              anticipation: int,
                              pivot_accuracy: float,
                              w_concordance: float = 0.3,
                              w_anticipation: float = 0.4,
                              w_pivot: float = 0.3) -> float:
    """
    Score pour PREDICTION (pas trading).

    Score = Concordance si Lag == 0
    Score = 0 si Lag != 0 (desynchronise = inutilisable)
    """
    if anticipation != 0:
        return 0.0  # Desynchronise, disqualifie
    return concordance


def parse_args():
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(
        description='Optimisation de la synchronisation par indicateur cible',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--target', '-t', type=str, required=True,
                        choices=['rsi', 'cci', 'macd'],
                        help='Indicateur cible (reference pour synchronisation)')

    parser.add_argument('--assets', '-a', nargs='+',
                        default=['BTC', 'ETH', 'BNB'],
                        help='Assets pour optimisation')

    parser.add_argument('--val-assets', '-v', nargs='+',
                        default=['ADA', 'LTC'],
                        help='Assets pour validation')

    parser.add_argument('--output', '-o', type=str,
                        default=None,
                        help='Fichier de sortie (defaut: results/sync_target_<target>.json)')

    parser.add_argument('--device', '-D', type=str,
                        default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device pour les calculs')

    parser.add_argument('--max-samples', '-n', type=int,
                        default=50000,
                        help='Nombre max de bougies par asset')

    parser.add_argument('--step', '-s', type=int,
                        default=1,
                        help='Echantillonnage (1=toutes, 2=1sur2, etc.)')

    return parser.parse_args()


def main():
    """Pipeline principal d'optimisation par cible."""
    global DEVICE

    args = parse_args()

    # Configurer le device
    if args.device == 'auto':
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = torch.device(args.device)

    target = args.target.upper()

    # Determiner les indicateurs a optimiser (tous sauf la cible)
    all_indicators = ['RSI', 'CCI', 'MACD']
    features_to_optimize = [ind for ind in all_indicators if ind != target]

    # Output path
    if args.output is None:
        args.output = f'results/sync_target_{args.target}.json'

    logger.info("="*80)
    logger.info(f"OPTIMISATION POUR CIBLE: {target}")
    logger.info("="*80)
    logger.info(f"Device:              {DEVICE}")
    logger.info(f"Indicateur cible:    {target} (reference)")
    logger.info(f"Indicateurs features: {features_to_optimize} (a optimiser)")
    logger.info(f"Assets optimisation: {args.assets}")
    logger.info(f"Assets validation:   {args.val_assets}")

    all_assets = list(set(args.assets + args.val_assets))

    # =========================================================================
    # 1. CHARGER DONNEES
    # =========================================================================
    logger.info(f"\n1. Chargement des donnees...")

    dfs = {}
    for asset in all_assets:
        df = load_asset_data(asset, max_samples=args.max_samples, step=args.step)
        dfs[asset] = df

    # =========================================================================
    # 2. CALCULER REFERENCE = Kalman(TARGET) avec params par defaut
    # =========================================================================
    logger.info(f"\n2. Calcul reference Kalman({target})...")

    target_params = DEFAULT_PARAMS[target]
    ref_filtered_dict = {}

    for asset in all_assets:
        values = calculate_indicator_values(dfs[asset], target, target_params)
        filtered = kalman_filter(values, KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR)
        ref_filtered_dict[asset] = filtered
        logger.info(f"  {asset}: Kalman({target}) OK")

    # =========================================================================
    # 3. KALMAN POUR TOUS LES FEATURES
    # =========================================================================
    logger.info(f"\n3. Kalman pour features {features_to_optimize}...")

    ind_filtered_dict = {}

    for indicator in features_to_optimize:
        ind_filtered_dict[indicator] = {}
        param_grid = PARAM_GRIDS[indicator]

        if indicator == 'MACD':
            combinations = [{'fast': f, 'slow': s}
                           for f in param_grid['fast']
                           for s in param_grid['slow'] if f < s]
        else:
            combinations = [{'period': p} for p in param_grid['period']]

        for params in combinations:
            params_key = str(params)
            ind_filtered_dict[indicator][params_key] = {}

            for asset in all_assets:
                values = calculate_indicator_values(dfs[asset], indicator, params)
                filtered = kalman_filter(values, KALMAN_PROCESS_VAR, KALMAN_MEASURE_VAR)
                ind_filtered_dict[indicator][params_key][asset] = filtered

        logger.info(f"  {indicator}: {len(combinations)} combinaisons OK")

    # =========================================================================
    # 4. TRANSFERT GPU + CALCUL LABELS
    # =========================================================================
    logger.info(f"\n4. Transfert vers {DEVICE}...")

    # Reference
    ref_labels_gpu_dict = {}
    for asset in all_assets:
        values_gpu = to_gpu_tensor(ref_filtered_dict[asset])
        ref_labels_gpu_dict[asset] = calculate_slope_labels_gpu(values_gpu)

    # Features
    ind_labels_gpu_dict = {}
    for indicator in features_to_optimize:
        ind_labels_gpu_dict[indicator] = {}
        for params_key in ind_filtered_dict[indicator]:
            ind_labels_gpu_dict[indicator][params_key] = {}
            for asset in all_assets:
                values_gpu = to_gpu_tensor(ind_filtered_dict[indicator][params_key][asset])
                ind_labels_gpu_dict[indicator][params_key][asset] = calculate_slope_labels_gpu(values_gpu)

    # =========================================================================
    # 5. OPTIMISATION
    # =========================================================================
    logger.info(f"\n5. Optimisation des features pour cible {target}...")

    optimal_params = {}
    best_results = {}

    for indicator in features_to_optimize:
        logger.info(f"\n{'='*60}")
        logger.info(f"Optimisation {indicator} -> {target}")
        logger.info(f"{'='*60}")

        all_results = []
        param_grid = PARAM_GRIDS[indicator]

        if indicator == 'MACD':
            combinations = [{'fast': f, 'slow': s}
                           for f in param_grid['fast']
                           for s in param_grid['slow'] if f < s]
        else:
            combinations = [{'period': p} for p in param_grid['period']]

        for params in combinations:
            params_key = str(params)
            scores_per_asset = []

            for asset in args.assets:
                ind_gpu = ind_labels_gpu_dict[indicator][params_key][asset]
                ref_gpu = ref_labels_gpu_dict[asset]

                # Concordance
                concordance = (ind_gpu == ref_gpu).float().mean().item()

                # Lag
                l1 = ind_gpu.float() - ind_gpu.float().mean()
                l2 = ref_gpu.float() - ref_gpu.float().mean()
                best_lag, best_corr = 0, -1.0
                for lag in range(LAG_RANGE[0], LAG_RANGE[1]):
                    if lag < 0:
                        a, b = l1[-lag:], l2[:lag]
                    elif lag > 0:
                        a, b = l1[:-lag], l2[lag:]
                    else:
                        a, b = l1, l2
                    if len(a) > 1:
                        cov = ((a - a.mean()) * (b - b.mean())).mean()
                        std_a, std_b = a.std(), b.std()
                        if std_a > 0 and std_b > 0:
                            corr = (cov / (std_a * std_b)).item()
                            if not np.isnan(corr) and corr > best_corr:
                                best_corr, best_lag = corr, lag

                # Pivot accuracy
                diff_ref = torch.diff(ref_gpu)
                pivot_indices = torch.where(diff_ref != 0)[0]
                if len(pivot_indices) > 0:
                    valid_pivots = pivot_indices[(pivot_indices + 1) < len(ref_gpu)]
                    if len(valid_pivots) > 0:
                        matches = ind_gpu[valid_pivots + 1] == ref_gpu[valid_pivots + 1]
                        pivot_acc = matches.float().mean().item()
                    else:
                        pivot_acc = 0.5
                else:
                    pivot_acc = 0.5

                score = calculate_composite_score(concordance, best_lag, pivot_acc)
                scores_per_asset.append({
                    'concordance': concordance,
                    'anticipation': best_lag,
                    'pivot_accuracy': pivot_acc,
                    'score': score
                })

            # Moyenne
            avg_concordance = np.mean([s['concordance'] for s in scores_per_asset])
            avg_anticipation = np.mean([s['anticipation'] for s in scores_per_asset])
            avg_pivot_acc = np.mean([s['pivot_accuracy'] for s in scores_per_asset])
            avg_score = np.mean([s['score'] for s in scores_per_asset])

            all_results.append(SyncResult(
                indicator=indicator,
                params=params,
                concordance=avg_concordance,
                anticipation=avg_anticipation,
                pivot_accuracy=avg_pivot_acc,
                composite_score=avg_score,
                n_samples=0
            ))

            # Log
            if indicator == 'MACD':
                params_str = f"fast={params['fast']}, slow={params['slow']}"
            else:
                params_str = f"period={params['period']}"

            # Marquer si desynchronise
            sync_status = "✓" if avg_anticipation == 0 else "✗"
            logger.info(f"  {params_str:20s} | Conc: {avg_concordance:.3f} | "
                       f"Pivot: {avg_pivot_acc:.3f} | "
                       f"Lag: {avg_anticipation:+3.0f} {sync_status} | "
                       f"Score: {avg_score:.3f}")

        # Meilleur
        all_results.sort(key=lambda x: x.composite_score, reverse=True)
        best = all_results[0]
        optimal_params[indicator] = best.params
        best_results[indicator] = {
            'params': best.params,
            'concordance': best.concordance,
            'anticipation': best.anticipation,
            'pivot_accuracy': best.pivot_accuracy,
            'composite_score': best.composite_score
        }

        logger.info(f"\n  MEILLEUR {indicator} pour {target}: {best.params}")
        logger.info(f"    Concordance:    {best.concordance:.1%}")
        logger.info(f"    Pivot Accuracy: {best.pivot_accuracy:.1%}")
        logger.info(f"    Lag:            {best.anticipation:+.0f} (0 = synchronise)")

    # =========================================================================
    # 6. RESUME FINAL
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info(f"PARAMETRES OPTIMAUX POUR CIBLE {target}")
    logger.info("="*80)

    print(f"\n# Parametres optimises pour predire {target}:\n")
    print(f"# Cible: {target} avec params {target_params}")
    print(f"# Score = Concordance (Lag=0 requis)")
    print(f"#")
    print(f"# Features optimisees:")
    for indicator, params in optimal_params.items():
        conc = best_results[indicator]['concordance']
        pivot = best_results[indicator]['pivot_accuracy']
        delta = conc - pivot  # Difference globale vs pivots
        if indicator == 'RSI':
            print(f"#   RSI_PERIOD_{target} = {params['period']:3d}  (Conc: {conc:.1%}, Pivot: {pivot:.1%}, Delta: {delta:+.1%})")
        elif indicator == 'CCI':
            print(f"#   CCI_PERIOD_{target} = {params['period']:3d}  (Conc: {conc:.1%}, Pivot: {pivot:.1%}, Delta: {delta:+.1%})")
        elif indicator == 'MACD':
            print(f"#   MACD_FAST_{target} = {params['fast']:3d}  (Conc: {conc:.1%}, Pivot: {pivot:.1%}, Delta: {delta:+.1%})")
            print(f"#   MACD_SLOW_{target} = {params['slow']:3d}")

    # =========================================================================
    # 7. SAUVEGARDER
    # =========================================================================
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'timestamp': datetime.now().isoformat(),
        'target_indicator': target,
        'target_params': target_params,
        'optimization_assets': args.assets,
        'validation_assets': args.val_assets,
        'optimal_params': optimal_params,
        'detailed_results': best_results,
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResultats sauvegardes: {output_path}")
    logger.info("\n" + "="*80)
    logger.info("OPTIMISATION TERMINEE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
