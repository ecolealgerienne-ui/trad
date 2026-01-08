#!/usr/bin/env python3
"""
Test Oracle avec Filtre Kalman (Sliding Window ou Global) ou Octave
====================================================================

Phase 2.11 Analysis - Test du signal maximum avec labels parfaits (Oracle).

Ce script teste le potentiel maximum du signal en appliquant un filtre
(Kalman GLOBAL, Kalman Sliding, ou Octave Sliding) et en tradant avec les labels parfaits.

Pipeline CORRECT:
-----------------
1. Charger CSV brut (OHLC)
2. Calculer indicateur technique (RSI/MACD/CCI) - valeurs BRUTES
3. Appliquer filtre:
   - Kalman GLOBAL: filtre sur tout l'historique (validé Phase 2.10: +6,644%)
   - Kalman Sliding: filtre sur fenêtre glissante (échec: -19% à -30%)
   - Octave Sliding: filtre Butterworth sur fenêtre glissante (catastrophe: -37% à -116%)
4. Calculer labels Oracle
5. Extraire returns (close.pct_change())
6. Backtest

Tests:
------
1. Test 1: Label = filtered[t-2] > filtered[t-3] (défaut)
2. Test 2: Label = filtered[t-3] > filtered[t-4]
3. Custom: --lag1 N --lag2 M (ex: t-3 vs t-4)

Usage:
------
# Test Kalman GLOBAL (RECOMMANDÉ - Phase 2.10 validé)
python tests/test_oracle_sliding_window.py --indicator macd --mode global --n-samples 1000

# Test Kalman GLOBAL avec lags différents (t-3 vs t-4)
python tests/test_oracle_sliding_window.py --indicator macd --mode global --lag1 3 --lag2 4

# Test Kalman Sliding Window (pour comparaison)
python tests/test_oracle_sliding_window.py --indicator macd --mode sliding --window 100 --n-samples 1000

# Test Octave Sliding Window
python tests/test_oracle_sliding_window.py --indicator macd --filter-type octave --mode sliding --n-samples 1000
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.signal
from pykalman import KalmanFilter

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from constants import AVAILABLE_ASSETS_5M, RSI_PERIOD, CCI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL
from indicators import calculate_rsi, calculate_cci, calculate_macd
from backtest_utils import backtest_simple_labels, BacktestResult as BT_Result

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & DATACLASSES
# =============================================================================

class Position(Enum):
    """Position trading."""
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass
class Trade:
    """Trade individuel."""
    start: int
    end: int
    duration: int
    position: str
    pnl: float
    pnl_after_fees: float


@dataclass
class OracleResult:
    """Résultat backtest Oracle."""
    name: str
    lag1: int
    lag2: int
    n_samples: int
    n_trades: int
    n_long: int
    n_short: int
    total_pnl: float
    total_pnl_after_fees: float
    total_fees: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_duration: float
    sharpe_ratio: float
    trades: List[Trade]


# =============================================================================
# CHARGEMENT DONNÉES BRUTES
# =============================================================================

def load_csv_data(asset: str = 'BTC') -> pd.DataFrame:
    """
    Charge CSV brut (OHLC).

    Args:
        asset: Asset à charger (BTC, ETH, etc.)

    Returns:
        DataFrame avec colonnes: timestamp, open, high, low, close, volume
    """
    if asset not in AVAILABLE_ASSETS_5M:
        raise ValueError(f"Asset inconnu: {asset}. Disponibles: {list(AVAILABLE_ASSETS_5M.keys())}")

    csv_path = AVAILABLE_ASSETS_5M[asset]
    logger.info(f"\n📂 Chargement CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    # Standardiser les noms de colonnes (minuscules)
    df.columns = df.columns.str.lower()

    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Colonnes: {list(df.columns)}")
    logger.info(f"  Période: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

    return df


def calculate_indicator(df: pd.DataFrame, indicator: str) -> np.ndarray:
    """
    Calcule indicateur technique à partir du DataFrame OHLC.

    Args:
        df: DataFrame avec OHLC
        indicator: 'rsi', 'macd', ou 'cci'

    Returns:
        Valeurs de l'indicateur (shape: n,)
    """
    logger.info(f"\n🔧 Calcul indicateur: {indicator.upper()}")

    if indicator == 'rsi':
        values = calculate_rsi(df['close'], period=RSI_PERIOD)
        logger.info(f"  Période: {RSI_PERIOD}")
        logger.info(f"  Min: {np.nanmin(values):.2f}")
        logger.info(f"  Max: {np.nanmax(values):.2f}")
        logger.info(f"  Mean: {np.nanmean(values):.2f}")

    elif indicator == 'macd':
        macd_data = calculate_macd(df['close'], fast_period=MACD_FAST, slow_period=MACD_SLOW, signal_period=MACD_SIGNAL)
        values = macd_data['macd']  # MACD line (pas histogram)
        logger.info(f"  Périodes: {MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL}")
        logger.info(f"  Min: {np.nanmin(values):.4f}")
        logger.info(f"  Max: {np.nanmax(values):.4f}")
        logger.info(f"  Mean: {np.nanmean(values):.4f}")

    elif indicator == 'cci':
        values = calculate_cci(df['high'], df['low'], df['close'], period=CCI_PERIOD)
        logger.info(f"  Période: {CCI_PERIOD}")
        logger.info(f"  Min: {np.nanmin(values):.2f}")
        logger.info(f"  Max: {np.nanmax(values):.2f}")
        logger.info(f"  Mean: {np.nanmean(values):.2f}")

    else:
        raise ValueError(f"Indicateur inconnu: {indicator}")

    # Vérifier NaN
    n_nan = np.isnan(values).sum()
    logger.info(f"  NaN: {n_nan} ({n_nan/len(values)*100:.1f}%)\n")

    return values


# =============================================================================
# KALMAN SLIDING WINDOW
# =============================================================================

def apply_sliding_kalman(
    values: np.ndarray,
    window: int,
    start_idx: int,
    n_samples: int
) -> np.ndarray:
    """
    Applique filtre Kalman sur fenêtre glissante.

    Args:
        values: Valeurs indicateur (shape: n,)
        window: Taille fenêtre Kalman
        start_idx: Index de départ
        n_samples: Nombre de samples à traiter

    Returns:
        Valeurs filtrées (shape: n_samples,)
    """
    logger.info(f"🔧 Application Kalman glissant:")
    logger.info(f"  Window: {window}")
    logger.info(f"  Start index: {start_idx}")
    logger.info(f"  N samples: {n_samples}")
    logger.info(f"  End index: {start_idx + n_samples}")

    # Vérifier qu'on a assez de données
    required_length = start_idx + n_samples
    if len(values) < required_length:
        raise ValueError(
            f"Pas assez de données! Requis: {required_length}, "
            f"Disponible: {len(values)}"
        )

    # Vérifier NaN dans la zone d'intérêt
    segment = values[start_idx - window + 1 : start_idx + n_samples]
    n_nan = np.isnan(segment).sum()
    if n_nan > 0:
        logger.warning(f"  ⚠️  {n_nan} NaN détectés dans le segment!")
        logger.info(f"  → Remplissage par forward-fill\n")
        # Forward fill des NaN
        df_temp = pd.DataFrame({'values': values})
        df_temp['values'] = df_temp['values'].fillna(method='ffill').fillna(method='bfill')
        values = df_temp['values'].values

    # Initialiser Kalman (config standard du projet)
    # On prend la première valeur non-NaN comme initial state
    first_valid_idx = np.where(~np.isnan(values))[0][0]
    initial_state = values[first_valid_idx]

    kf = KalmanFilter(
        transition_matrices=[[1]],
        observation_matrices=[[1]],
        initial_state_mean=initial_state,
        initial_state_covariance=1.0,
        observation_covariance=1.0,
        transition_covariance=1e-5
    )

    filtered = np.zeros(n_samples)

    # Appliquer sur chaque fenêtre glissante
    for i in range(n_samples):
        global_idx = start_idx + i

        # Fenêtre: [global_idx - window + 1 : global_idx + 1]
        window_start = max(0, global_idx - window + 1)
        window_end = global_idx + 1

        window_data = values[window_start:window_end]

        # Appliquer Kalman sur cette fenêtre
        state_means, _ = kf.filter(window_data)

        # Prendre dernière valeur filtrée
        filtered[i] = state_means[-1, 0]

    logger.info(f"✅ Kalman appliqué sur {n_samples} fenêtres")
    logger.info(f"  Filtered min: {filtered.min():.4f}")
    logger.info(f"  Filtered max: {filtered.max():.4f}")
    logger.info(f"  Filtered mean: {filtered.mean():.4f}\n")

    return filtered


# =============================================================================
# KALMAN GLOBAL (TOUT L'HISTORIQUE)
# =============================================================================

def apply_global_kalman(
    values: np.ndarray,
    start_idx: int,
    n_samples: int
) -> np.ndarray:
    """
    Applique filtre Kalman sur TOUT l'historique (mode GLOBAL).

    Phase 2.10 validé: +6,644% Oracle PnL (vs -19% à -30% sliding window).

    Args:
        values: Valeurs indicateur complètes (shape: n,)
        start_idx: Index de départ pour extraction segment
        n_samples: Nombre de samples à extraire

    Returns:
        Segment filtré (shape: n_samples,)
    """
    logger.info(f"🔧 Application Kalman GLOBAL:")
    logger.info(f"  Dataset complet: {len(values)} samples")
    logger.info(f"  Extraction segment: [{start_idx}:{start_idx + n_samples}]")

    # Vérifier qu'on a assez de données
    required_length = start_idx + n_samples
    if len(values) < required_length:
        raise ValueError(
            f"Pas assez de données! Requis: {required_length}, "
            f"Disponible: {len(values)}"
        )

    # Vérifier NaN dans TOUT le dataset
    n_nan = np.isnan(values).sum()
    if n_nan > 0:
        logger.warning(f"  ⚠️  {n_nan} NaN détectés dans le dataset complet!")
        logger.info(f"  → Remplissage par forward-fill\n")
        # Forward fill des NaN
        df_temp = pd.DataFrame({'values': values})
        df_temp['values'] = df_temp['values'].fillna(method='ffill').fillna(method='bfill')
        values = df_temp['values'].values

    # Initialiser Kalman (config standard du projet)
    first_valid_idx = np.where(~np.isnan(values))[0][0]
    initial_state = values[first_valid_idx]

    kf = KalmanFilter(
        transition_matrices=[[1]],
        observation_matrices=[[1]],
        initial_state_mean=initial_state,
        initial_state_covariance=1.0,
        observation_covariance=1.0,
        transition_covariance=1e-5
    )

    # ✅ FILTRE GLOBAL: Appliquer sur TOUT l'historique (une seule fois)
    logger.info(f"  Filtrage de {len(values)} samples (GLOBAL)...")
    state_means, _ = kf.filter(values)
    filtered_full = state_means[:, 0]

    # Extraire segment d'intérêt
    filtered_segment = filtered_full[start_idx:start_idx + n_samples]

    logger.info(f"✅ Kalman GLOBAL appliqué")
    logger.info(f"  Segment extrait: {len(filtered_segment)} samples")
    logger.info(f"  Filtered min: {filtered_segment.min():.4f}")
    logger.info(f"  Filtered max: {filtered_segment.max():.4f}")
    logger.info(f"  Filtered mean: {filtered_segment.mean():.4f}\n")

    return filtered_segment


# =============================================================================
# OCTAVE SLIDING WINDOW
# =============================================================================

def apply_sliding_octave(
    values: np.ndarray,
    window: int,
    start_idx: int,
    n_samples: int,
    step: float = 0.25,
    order: int = 3
) -> np.ndarray:
    """
    Applique filtre Octave (Butterworth + filtfilt) sur fenêtre glissante.

    Args:
        values: Valeurs indicateur (shape: n,)
        window: Taille fenêtre Octave
        start_idx: Index de départ
        n_samples: Nombre de samples à traiter
        step: Paramètre de lissage Butterworth (0.0-1.0, défaut: 0.25)
        order: Ordre du filtre Butterworth (défaut: 3)

    Returns:
        Valeurs filtrées (shape: n_samples,)
    """
    logger.info(f"🔧 Application Octave glissant:")
    logger.info(f"  Window: {window}")
    logger.info(f"  Start index: {start_idx}")
    logger.info(f"  N samples: {n_samples}")
    logger.info(f"  End index: {start_idx + n_samples}")
    logger.info(f"  Step (smoothing): {step}")
    logger.info(f"  Order: {order}")

    # Vérifier qu'on a assez de données
    required_length = start_idx + n_samples
    if len(values) < required_length:
        raise ValueError(
            f"Pas assez de données! Requis: {required_length}, "
            f"Disponible: {len(values)}"
        )

    # Vérifier NaN dans la zone d'intérêt
    segment = values[start_idx - window + 1 : start_idx + n_samples]
    n_nan = np.isnan(segment).sum()
    if n_nan > 0:
        logger.warning(f"  ⚠️  {n_nan} NaN détectés dans le segment!")
        logger.info(f"  → Remplissage par forward-fill\n")
        # Forward fill des NaN
        df_temp = pd.DataFrame({'values': values})
        df_temp['values'] = df_temp['values'].fillna(method='ffill').fillna(method='bfill')
        values = df_temp['values'].values

    # Créer le filtre Butterworth (une seule fois)
    B, A = scipy.signal.butter(order, step, output='ba')

    filtered = np.zeros(n_samples)

    # Appliquer sur chaque fenêtre glissante
    for i in range(n_samples):
        global_idx = start_idx + i

        # Fenêtre: [global_idx - window + 1 : global_idx + 1]
        window_start = max(0, global_idx - window + 1)
        window_end = global_idx + 1

        window_data = values[window_start:window_end]

        # Vérifier taille minimale pour filtfilt
        if len(window_data) < 3 * order:
            # Pas assez de données pour filtfilt, utiliser la valeur brute
            filtered[i] = window_data[-1] if len(window_data) > 0 else 0.0
        else:
            # Appliquer filtfilt (bidirectionnel, non-causal)
            try:
                filtered_data = scipy.signal.filtfilt(B, A, window_data)
                # Prendre dernière valeur filtrée
                filtered[i] = filtered_data[-1]
            except Exception as e:
                logger.warning(f"  ⚠️  Erreur filtfilt à i={i}: {e}, utilisation valeur brute")
                filtered[i] = window_data[-1]

    logger.info(f"✅ Octave appliqué sur {n_samples} fenêtres")
    logger.info(f"  Filtered min: {filtered.min():.4f}")
    logger.info(f"  Filtered max: {filtered.max():.4f}")
    logger.info(f"  Filtered mean: {filtered.mean():.4f}\n")

    return filtered


# =============================================================================
# CALCUL LABELS
# =============================================================================

def compute_labels(
    filtered: np.ndarray,
    lag1: int,
    lag2: int
) -> np.ndarray:
    """
    Calcule labels: filtered[t-lag1] > filtered[t-lag2]

    Args:
        filtered: Valeurs filtrées (shape: n,)
        lag1: Premier lag (ex: 2)
        lag2: Deuxième lag (ex: 3)

    Returns:
        Labels binaires (shape: n,) - 1=UP, 0=DOWN
    """
    n = len(filtered)
    labels = np.zeros(n, dtype=int)

    # Calculer labels à partir de max(lag1, lag2)
    max_lag = max(lag1, lag2)

    for i in range(max_lag, n):
        labels[i] = 1 if filtered[i - lag1] > filtered[i - lag2] else 0

    # Samples valides (où on peut calculer le label)
    n_valid = n - max_lag
    n_up = labels[max_lag:].sum()
    n_down = n_valid - n_up

    logger.info(f"📊 Labels calculés (lag {lag1} vs {lag2}):")
    logger.info(f"  Samples valides: {n_valid}/{n}")
    logger.info(f"  UP: {n_up} ({n_up/n_valid*100:.1f}%)")
    logger.info(f"  DOWN: {n_down} ({n_down/n_valid*100:.1f}%)\n")

    return labels


# =============================================================================
# BACKTEST ORACLE (WRAPPER pour backtest_simple_labels)
# =============================================================================

def backtest_oracle(
    labels: np.ndarray,
    df_segment: pd.DataFrame,
    fees: float,
    lag1: int,
    lag2: int
) -> OracleResult:
    """
    Backtest avec labels Oracle parfaits.

    MÉTHODE CAUSALE CORRECTE (via backtest_simple_labels):
    - Achète à Open[i+1] quand signal à timestep i
    - PnL calculé sur les Opens, pas les Closes

    Args:
        labels: Labels Oracle (1=UP, 0=DOWN)
        df_segment: DataFrame avec colonnes 'open', 'close' (segment backtest)
        fees: Frais par side
        lag1, lag2: Lags utilisés pour le nom

    Returns:
        OracleResult
    """
    # Utiliser la fonction commune de backtest
    bt_result = backtest_simple_labels(labels, df_segment, fees)

    # Convertir BT_Result → OracleResult
    return OracleResult(
        name=f"Oracle (t-{lag1} vs t-{lag2})",
        lag1=lag1,
        lag2=lag2,
        n_samples=len(labels),
        n_trades=bt_result.n_trades,
        n_long=bt_result.n_long,
        n_short=bt_result.n_short,
        total_pnl=bt_result.total_pnl,
        total_pnl_after_fees=bt_result.total_pnl_after_fees,
        total_fees=bt_result.total_fees,
        win_rate=bt_result.win_rate,
        profit_factor=bt_result.profit_factor,
        avg_win=bt_result.avg_win,
        avg_loss=bt_result.avg_loss,
        avg_duration=bt_result.avg_duration,
        sharpe_ratio=bt_result.sharpe_ratio,
        trades=bt_result.trades
    )




# =============================================================================
# AFFICHAGE
# =============================================================================

def print_results(results: List[OracleResult], filter_type: str, mode: str):
    """Affiche résultats comparatifs."""
    mode_display = "GLOBAL" if mode == "global" else "SLIDING WINDOW"
    logger.info("\n" + "="*100)
    logger.info(f"RÉSULTATS TESTS ORACLE - {filter_type.upper()} {mode_display}")
    logger.info("="*100)
    logger.info(
        f"{'Test':<25} {'Trades':>8} {'Win Rate':>9} "
        f"{'PnL Net':>10} {'Sharpe':>8} {'Avg Dur':>9}"
    )
    logger.info("-"*100)

    for r in results:
        logger.info(
            f"{r.name:<25} {r.n_trades:>8} {r.win_rate*100:>8.2f}% "
            f"{r.total_pnl_after_fees*100:>9.2f}% {r.sharpe_ratio:>8.3f} "
            f"{r.avg_duration:>8.1f}p"
        )

    logger.info("\n📊 DÉTAILS PAR TEST:")

    for r in results:
        logger.info(f"\n{r.name}")
        logger.info(f"  Samples: {r.n_samples}")
        logger.info(f"  Trades: {r.n_trades} (LONG: {r.n_long}, SHORT: {r.n_short})")
        logger.info(f"  Win Rate: {r.win_rate*100:.2f}%")
        logger.info(f"  Profit Factor: {r.profit_factor:.3f}")
        logger.info(f"  PnL Brut: {r.total_pnl*100:+.2f}%")
        logger.info(f"  PnL Net: {r.total_pnl_after_fees*100:+.2f}%")
        logger.info(f"  Frais Total: {r.total_fees*100:.2f}%")
        logger.info(f"  Avg Win: {r.avg_win*100:+.3f}%")
        logger.info(f"  Avg Loss: {r.avg_loss*100:+.3f}%")
        logger.info(f"  Avg Duration: {r.avg_duration:.1f} périodes")
        logger.info(f"  Sharpe Ratio: {r.sharpe_ratio:.3f}")

    # Comparaison Test 1 vs Test 2
    if len(results) == 2:
        logger.info("\n📈 COMPARAISON:")
        r1, r2 = results[0], results[1]

        delta_pnl = r2.total_pnl_after_fees - r1.total_pnl_after_fees
        delta_wr = r2.win_rate - r1.win_rate
        delta_sharpe = r2.sharpe_ratio - r1.sharpe_ratio

        logger.info(f"  Test 2 vs Test 1:")
        logger.info(f"    PnL Net: {delta_pnl*100:+.2f}% ({r2.total_pnl_after_fees*100:.2f}% vs {r1.total_pnl_after_fees*100:.2f}%)")
        logger.info(f"    Win Rate: {delta_wr*100:+.2f}% ({r2.win_rate*100:.2f}% vs {r1.win_rate*100:.2f}%)")
        logger.info(f"    Sharpe: {delta_sharpe:+.3f} ({r2.sharpe_ratio:.3f} vs {r1.sharpe_ratio:.3f})")

        if delta_pnl > 0:
            logger.info(f"\n✅ Test 2 (t-{r2.lag1} vs t-{r2.lag2}) MEILLEUR que Test 1 (t-{r1.lag1} vs t-{r1.lag2})")
        else:
            logger.info(f"\n✅ Test 1 (t-{r1.lag1} vs t-{r1.lag2}) MEILLEUR que Test 2 (t-{r2.lag1} vs t-{r2.lag2})")

    logger.info("\n" + "="*100 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Test Oracle avec filtre en fenêtre glissante (Kalman ou Octave)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--indicator', type=str, default='macd',
                        choices=['rsi', 'macd', 'cci'],
                        help='Indicateur (défaut: macd)')
    parser.add_argument('--asset', type=str, default='BTC',
                        choices=list(AVAILABLE_ASSETS_5M.keys()),
                        help='Asset (défaut: BTC)')
    parser.add_argument('--mode', type=str, default='sliding',
                        choices=['sliding', 'global'],
                        help='Mode filtre: sliding (fenêtre glissante) ou global (tout historique) - défaut: sliding')
    parser.add_argument('--filter-type', type=str, default='kalman',
                        choices=['kalman', 'octave'],
                        help='Type de filtre (défaut: kalman)')
    parser.add_argument('--window', type=int, default=100,
                        help='Taille fenêtre pour mode sliding (défaut: 100, ignoré si mode=global)')
    parser.add_argument('--start-idx', type=int, default=100,
                        help='Index de départ (défaut: 100)')
    parser.add_argument('--n-samples', type=int, default=1000,
                        help='Nombre de samples (défaut: 1000)')
    parser.add_argument('--fees', type=float, default=0.001,
                        help='Frais par side (défaut: 0.001)')

    # Paramètres Octave (optionnels)
    parser.add_argument('--octave-step', type=float, default=0.25,
                        help='Octave smoothing step (0.0-1.0, défaut: 0.25)')
    parser.add_argument('--octave-order', type=int, default=3,
                        help='Octave filter order (défaut: 3)')

    # Tests prédéfinis
    parser.add_argument('--test1-only', action='store_true',
                        help='Exécuter Test 1 uniquement (t-2 vs t-3)')
    parser.add_argument('--test2-only', action='store_true',
                        help='Exécuter Test 2 uniquement (t-3 vs t-4)')

    # Ou custom lags
    parser.add_argument('--lag1', type=int, default=None,
                        help='Lag 1 custom (override tests prédéfinis)')
    parser.add_argument('--lag2', type=int, default=None,
                        help='Lag 2 custom (override tests prédéfinis)')

    args = parser.parse_args()

    # Validation: Octave n'est disponible qu'en mode sliding
    if args.filter_type == 'octave' and args.mode == 'global':
        logger.error("❌ ERREUR: Octave n'est disponible qu'en mode SLIDING!")
        logger.error("  → Utilisez --mode sliding avec --filter-type octave")
        logger.error("  → OU utilisez --mode global avec --filter-type kalman (recommandé)")
        sys.exit(1)

    mode_display = "GLOBAL" if args.mode == "global" else "SLIDING WINDOW"
    logger.info("="*100)
    logger.info(f"TEST ORACLE {mode_display} - {args.indicator.upper()} ({args.asset}) - {args.filter_type.upper()}")
    logger.info("="*100)
    logger.info(f"Indicateur: {args.indicator}")
    logger.info(f"Asset: {args.asset}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Filtre: {args.filter_type}")
    if args.filter_type == 'octave':
        logger.info(f"  Octave step: {args.octave_step}")
        logger.info(f"  Octave order: {args.octave_order}")
    if args.mode == 'sliding':
        logger.info(f"Window: {args.window}")
    else:
        logger.info(f"Window: N/A (mode GLOBAL, filtre sur tout l'historique)")
    logger.info(f"Start index: {args.start_idx}")
    logger.info(f"N samples: {args.n_samples}")
    logger.info(f"Fees: {args.fees*100:.2f}% par side ({args.fees*2*100:.2f}% round-trip)")
    logger.info("="*100)

    # 1. Charger CSV brut
    df = load_csv_data(args.asset)

    # 2. Calculer indicateur
    indicator_values = calculate_indicator(df, args.indicator)

    # 4. Appliquer filtre (Global ou Sliding)
    if args.mode == 'global':
        # Mode GLOBAL: Kalman uniquement (validé Phase 2.10)
        if args.filter_type != 'kalman':
            raise ValueError(f"Mode GLOBAL supporté uniquement avec Kalman (filter_type=kalman)")

        filtered = apply_global_kalman(
            indicator_values,
            args.start_idx,
            args.n_samples
        )

    elif args.mode == 'sliding':
        # Mode SLIDING: Kalman ou Octave
        if args.filter_type == 'kalman':
            filtered = apply_sliding_kalman(
                indicator_values,
                args.window,
                args.start_idx,
                args.n_samples
            )
        elif args.filter_type == 'octave':
            filtered = apply_sliding_octave(
                indicator_values,
                args.window,
                args.start_idx,
                args.n_samples,
                step=args.octave_step,
                order=args.octave_order
            )
        else:
            raise ValueError(f"Type de filtre inconnu: {args.filter_type}")
    else:
        raise ValueError(f"Mode inconnu: {args.mode}")

    # 5. Extraire segment DataFrame pour backtest
    # IMPORTANT: Besoin de start_idx + n_samples + 1 pour avoir Open[i+1]
    df_segment = df.iloc[args.start_idx:args.start_idx + args.n_samples + 1].reset_index(drop=True)
    logger.info(f"📊 DataFrame segment pour backtest:")
    logger.info(f"  Lignes: {len(df_segment)} (n_samples + 1 pour Open[i+1])")
    logger.info(f"  Colonnes: {list(df_segment.columns)}\n")

    # Déterminer tests à exécuter
    if args.lag1 is not None and args.lag2 is not None:
        tests = [(args.lag1, args.lag2)]
        logger.info(f"🔧 Test custom: lag {args.lag1} vs {args.lag2}\n")
    elif args.test1_only:
        tests = [(2, 3)]
        logger.info(f"🔧 Test 1 uniquement: t-2 vs t-3\n")
    elif args.test2_only:
        tests = [(3, 4)]
        logger.info(f"🔧 Test 2 uniquement: t-3 vs t-4\n")
    else:
        tests = [(2, 3), (3, 4)]
        logger.info(f"🔧 Exécution des 2 tests:\n")
        logger.info(f"  Test 1: t-2 vs t-3")
        logger.info(f"  Test 2: t-3 vs t-4\n")

    # Exécuter tests
    results = []

    for lag1, lag2 in tests:
        logger.info(f"\n{'='*100}")
        logger.info(f"TEST: t-{lag1} vs t-{lag2}")
        logger.info(f"{'='*100}\n")

        # Calculer labels
        labels = compute_labels(filtered, lag1, lag2)

        # Backtest (achète à Open[i+1], pas Close[i])
        result = backtest_oracle(labels, df_segment, args.fees, lag1, lag2)
        results.append(result)

    # Afficher résultats
    print_results(results, args.filter_type, args.mode)


if __name__ == '__main__':
    main()
