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
import torch

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device global (sera configure dans main)
DEVICE = torch.device('cpu')

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
                               params: Dict) -> np.ndarray:
    """
    Calcule les labels pour un indicateur avec des parametres donnes.

    NOTE: Pas de filtre Kalman sur l'indicateur - on compare la pente BRUTE
    de l'indicateur avec la pente lissee du Close. Cela permet de detecter
    si l'indicateur anticipe le mouvement du prix.
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

    # Gerer les NaN et calculer les labels directement (sans Kalman)
    values = pd.Series(values).ffill().fillna(50).values
    return calculate_slope_labels(values)


def to_gpu_tensor(labels: np.ndarray, start_idx: int = 50) -> torch.Tensor:
    """Convertit un array numpy en tensor GPU (cache-friendly)."""
    return torch.tensor(labels[start_idx:], device=DEVICE, dtype=torch.int32)


def calculate_concordance(labels1: np.ndarray, labels2: np.ndarray,
                          l1_gpu: Optional[torch.Tensor] = None,
                          l2_gpu: Optional[torch.Tensor] = None) -> float:
    """
    Calcule le taux de concordance (% de labels identiques) - GPU accelere.

    Si l1_gpu/l2_gpu sont fournis, évite la conversion CPU→GPU.
    """
    start_idx = 50

    # Utiliser tensors pré-chargés si disponibles
    if l1_gpu is None:
        l1_gpu = torch.tensor(labels1[start_idx:], device=DEVICE, dtype=torch.int32)
    if l2_gpu is None:
        l2_gpu = torch.tensor(labels2[start_idx:], device=DEVICE, dtype=torch.int32)

    # Calcul GPU
    concordance = (l1_gpu == l2_gpu).float().mean().item()
    return concordance


def calculate_lag(labels1: np.ndarray, labels2: np.ndarray, max_lag: int = 10,
                  l1_gpu: Optional[torch.Tensor] = None,
                  l2_gpu: Optional[torch.Tensor] = None) -> int:
    """
    Calcule le lag optimal entre deux series de labels - GPU accelere.

    Retourne:
        lag negatif = labels1 en avance sur labels2
        lag positif = labels1 en retard sur labels2

    Si l1_gpu/l2_gpu sont fournis, évite la conversion CPU→GPU.
    """
    start_idx = 50

    # Utiliser tensors pré-chargés si disponibles
    if l1_gpu is None:
        l1 = torch.tensor(labels1[start_idx:], device=DEVICE, dtype=torch.float32)
    else:
        l1 = l1_gpu.float()
    if l2_gpu is None:
        l2 = torch.tensor(labels2[start_idx:], device=DEVICE, dtype=torch.float32)
    else:
        l2 = l2_gpu.float()

    # Centrer les series
    l1 = l1 - l1.mean()
    l2 = l2 - l2.mean()

    best_lag = 0
    best_corr = -1.0

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = l1[-lag:], l2[:lag]
        elif lag > 0:
            a, b = l1[:-lag], l2[lag:]
        else:
            a, b = l1, l2

        # Correlation sur GPU
        if len(a) > 1:
            cov = ((a - a.mean()) * (b - b.mean())).mean()
            std_a, std_b = a.std(), b.std()
            if std_a > 0 and std_b > 0:
                corr = (cov / (std_a * std_b)).item()
                if not np.isnan(corr) and corr > best_corr:
                    best_corr = corr
                    best_lag = lag

    return best_lag


def calculate_pivot_accuracy(labels_ind: np.ndarray, labels_ref: np.ndarray,
                              l_ind_gpu: Optional[torch.Tensor] = None,
                              l_ref_gpu: Optional[torch.Tensor] = None) -> float:
    """
    Calcule le % de match sur les PIVOTS (changements de direction) - GPU accelere.

    Les pivots sont les points ou on gagne/perd de l'argent.
    Un indicateur avec 95% de concordance globale mais qui rate les pivots
    est moins utile qu'un indicateur a 80% qui capte les retournements.

    IMPORTANT: np.diff donne l'indice AVANT le changement.
    Si le prix change a l'indice T, diff le marque a T-1.
    On compare donc a pivots_ref + 1 (premiere bougie nouvelle direction).

    Si l_ind_gpu/l_ref_gpu sont fournis, évite la conversion CPU→GPU.
    """
    start_idx = 50

    # Utiliser tensors pré-chargés si disponibles
    if l_ind_gpu is None:
        l_ind = torch.tensor(labels_ind[start_idx:], device=DEVICE, dtype=torch.int32)
    else:
        l_ind = l_ind_gpu
    if l_ref_gpu is None:
        l_ref = torch.tensor(labels_ref[start_idx:], device=DEVICE, dtype=torch.int32)
    else:
        l_ref = l_ref_gpu

    # Detecter les pivots dans la reference (Close)
    diff_ref = torch.diff(l_ref)
    pivot_indices = torch.where(diff_ref != 0)[0]

    if len(pivot_indices) == 0:
        return 0.5  # Pas de pivot = score neutre

    # Comparer a l'indice + 1 (premiere bougie de la nouvelle direction)
    valid_mask = (pivot_indices + 1) < len(l_ref)
    valid_pivots = pivot_indices[valid_mask]

    if len(valid_pivots) == 0:
        return 0.5

    # Accuracy sur les pivots (GPU)
    matches = l_ind[valid_pivots + 1] == l_ref[valid_pivots + 1]
    return matches.float().mean().item()


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
                    params: Dict,
                    ref_labels_gpu: Optional[torch.Tensor] = None) -> SyncResult:
    """
    Evalue un jeu de parametres pour un indicateur.

    Si ref_labels_gpu est fourni, évite les transferts CPU→GPU répétés.
    """

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

    # Pré-charger ind_labels sur GPU une fois pour les 3 métriques
    ind_labels_gpu = to_gpu_tensor(ind_labels)

    # Calculer les 3 metriques avec tensors GPU pré-chargés
    concordance = calculate_concordance(ind_labels, ref_labels,
                                        l1_gpu=ind_labels_gpu, l2_gpu=ref_labels_gpu)
    anticipation = calculate_lag(ind_labels, ref_labels,
                                 l1_gpu=ind_labels_gpu, l2_gpu=ref_labels_gpu)
    pivot_acc = calculate_pivot_accuracy(ind_labels, ref_labels,
                                         l_ind_gpu=ind_labels_gpu, l_ref_gpu=ref_labels_gpu)
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
                       ref_labels_gpu_dict: Dict[str, torch.Tensor],
                       indicator: str) -> List[SyncResult]:
    """
    Optimise les parametres d'un indicateur sur plusieurs assets.

    Args:
        dfs: DataFrames par asset
        ref_labels_dict: Labels numpy par asset (pour compatibilité)
        ref_labels_gpu_dict: Labels pré-chargés sur GPU par asset
        indicator: Nom de l'indicateur

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
            ref_labels_gpu = ref_labels_gpu_dict[asset]
            result = evaluate_params(df, ref_labels, indicator, params,
                                     ref_labels_gpu=ref_labels_gpu)
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
                       val_assets: List[str],
                       dfs: Dict[str, pd.DataFrame],
                       ref_labels_dict: Dict[str, np.ndarray],
                       ref_labels_gpu_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Valide les parametres optimaux sur des assets non vus.

    Utilise les donnees deja chargees et pre-transferees sur GPU.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Validation sur: {val_assets}")
    logger.info(f"{'='*60}")

    # Evaluer chaque indicateur avec ses parametres optimaux
    validation_scores = {}

    for indicator, params in optimal_params.items():
        scores = []
        for asset in val_assets:
            df = dfs[asset]
            ref_labels = ref_labels_dict[asset]
            ref_labels_gpu = ref_labels_gpu_dict[asset]
            result = evaluate_params(df, ref_labels, indicator, params,
                                     ref_labels_gpu=ref_labels_gpu)
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

    parser.add_argument('--device', '-D', type=str,
                        default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device pour les calculs (auto=cuda si disponible)')

    return parser.parse_args()


def main():
    """Pipeline principal d'optimisation."""
    global DEVICE

    args = parse_args()

    # Configurer le device (GPU/CPU)
    if args.device == 'auto':
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = torch.device(args.device)

    logger.info("="*80)
    logger.info("OPTIMISATION DE LA SYNCHRONISATION DES INDICATEURS")
    logger.info("="*80)
    logger.info(f"Device:              {DEVICE}")
    logger.info(f"Assets optimisation: {args.assets}")
    logger.info(f"Assets validation:   {args.val_assets}")
    logger.info(f"Indicateurs:         {args.indicators}")

    # =========================================================================
    # 1. CHARGER TOUTES LES DONNEES ET CALCULER KALMAN (CPU)
    # =========================================================================
    logger.info("\n1. Chargement des donnees et calcul Kalman (CPU)...")
    logger.info("   (Cette etape est lente car Kalman ne supporte pas GPU)")

    all_assets = list(set(args.assets + args.val_assets))
    dfs = {}
    ref_labels_dict = {}

    for asset in all_assets:
        df = load_asset_data(asset)
        dfs[asset] = df
        ref_labels_dict[asset] = calculate_reference_labels(df['close'].values)

        # Stats des labels
        buy_pct = ref_labels_dict[asset].mean() * 100
        role = "OPT" if asset in args.assets else "VAL"
        logger.info(f"     [{role}] {asset}: {buy_pct:.1f}% UP / {100-buy_pct:.1f}% DOWN")

    # =========================================================================
    # 2. TRANSFERT VERS GPU (une seule fois pour tous les assets)
    # =========================================================================
    logger.info(f"\n2. Transfert des {len(all_assets)} assets vers {DEVICE}...")

    ref_labels_gpu_dict = {}
    for asset in all_assets:
        ref_labels_gpu_dict[asset] = to_gpu_tensor(ref_labels_dict[asset])

    logger.info(f"  -> {len(all_assets)} assets pre-charges sur {DEVICE}")

    # =========================================================================
    # 3. OPTIMISER CHAQUE INDICATEUR (GPU)
    # =========================================================================
    logger.info("\n3. Optimisation des parametres (GPU)...")

    optimal_params = {}
    best_results = {}

    for indicator in args.indicators:
        results = optimize_indicator(dfs, ref_labels_dict, ref_labels_gpu_dict, indicator)

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
    # 4. VALIDATION CROISEE (GPU - donnees deja chargees)
    # =========================================================================
    if args.val_assets:
        validation_scores = validate_on_assets(
            optimal_params, args.val_assets,
            dfs, ref_labels_dict, ref_labels_gpu_dict
        )

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
    # 5. RESUME FINAL
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
    # 6. SAUVEGARDER RESULTATS
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
