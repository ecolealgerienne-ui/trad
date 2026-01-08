#!/usr/bin/env python3
"""
Test Stratégie ATR Structural Filter - Filtrer trades par régime de volatilité.

HYPOTHÈSE (López de Prado 2018):
Ne trader QUE dans les régimes de volatilité "sains" (ni trop basse, ni trop haute).
- ATR < Q20: Marché trop calme (ranging, signaux faibles)
- Q20 < ATR < Q80: Volatilité saine (conditions optimales)
- ATR > Q80: Volatilité extrême (gaps, slippage élevé)

PRINCIPE:
- Entrée: MACD Direction=UP ou DOWN (Direction-Only)
- Filtre: Trade UNIQUEMENT si Q20 < ATR < Q80
- Sortie: Direction flip

OBJECTIF Phase 2.8:
- Baseline: 30,876 trades, +110.89% PnL Brut, -2,976% PnL Net
- Cible: ~15,000 trades (-50%), PnL Net POSITIF (+100-200%)

TESTS:
- Percentiles: (Q10, Q90), (Q20, Q80), (Q30, Q70)
- Indicateur: MACD Direction-Only Kalman (92.5% accuracy)
- Dataset: 5 assets (BTC, ETH, BNB, ADA, LTC), Test Set

Usage:
    python tests/test_atr_structural_filter.py --split test --fees 0.003

Référence littérature:
    Marcos López de Prado (2018) - "Advances in Financial ML"
    Chapitre 18: Structural Breaks and Microstructure Noise
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import argparse
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

# Importer fonctions validées (réutilisation!)
from structural_filters import calculate_atr_normalized

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTES
# =============================================================================

# Percentiles à tester (low, high)
PERCENTILE_CONFIGS = [
    (10, 90, "Large"),   # Large (garde 80%)
    (20, 80, "Standard"), # Standard (garde 60%)
    (30, 70, "Strict"),   # Strict (garde 40%)
]

# Mapping assets vers CSV (RÉUTILISER config existante!)
AVAILABLE_ASSETS = {
    'BTC': 'data_trad/BTCUSD_all_5m.csv',
    'ETH': 'data_trad/ETHUSD_all_5m.csv',
    'BNB': 'data_trad/BNBUSD_all_5m.csv',
    'ADA': 'data_trad/ADAUSD_all_5m.csv',
    'LTC': 'data_trad/LTCUSD_all_5m.csv',
}


# =============================================================================
# DATACLASSES (RÉUTILISÉES de test_holding_strategy.py)
# =============================================================================

class Position(Enum):
    """Positions possibles."""
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Trade:
    """Enregistrement d'un trade."""
    start: int
    end: int
    duration: int
    position: str
    pnl: float
    pnl_after_fees: float
    exit_reason: str  # "DIRECTION_FLIP", "END"


@dataclass
class StrategyResult:
    """Résultats d'une stratégie."""
    name: str
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
    # Métriques filtre ATR
    atr_coverage: float  # % samples où filtre autorise trade
    atr_q_low: float
    atr_q_high: float


# =============================================================================
# CHARGEMENT DONNÉES
# =============================================================================

def load_dataset_direction_only(split: str = 'test') -> Dict:
    """
    Charge dataset MACD Direction-Only Kalman.

    RÉUTILISE: Logique de test_holding_strategy.py mais adapté Direction-Only.
    """
    path = f'data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_direction_only_kalman.npz'

    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset introuvable: {path}")

    logger.info(f"📂 Chargement: {path}")
    data = np.load(path, allow_pickle=True)

    # Vérifier shape Y (doit être (n,1) pour Direction-Only)
    Y_split = data[f'Y_{split}']
    if Y_split.ndim == 2 and Y_split.shape[1] == 1:
        logger.info(f"✅ Direction-Only détecté: Y shape = {Y_split.shape}")
    else:
        raise ValueError(f"Y shape incorrect (attendu (n,1)): {Y_split.shape}")

    return {
        'X': data[f'X_{split}'],
        'Y': data[f'Y_{split}'],
        'Y_pred': data.get(f'Y_{split}_pred', None),
        'metadata': data.get('metadata', None),
    }


def load_ohlcv_data() -> Dict[str, pd.DataFrame]:
    """
    Charge données OHLCV pour tous les assets (pour calcul ATR).

    RÉUTILISE: Structure AVAILABLE_ASSETS de constants.py.
    """
    logger.info(f"📂 Chargement OHLCV ({len(AVAILABLE_ASSETS)} assets)")

    dataframes = {}
    for asset, path in AVAILABLE_ASSETS.items():
        if not Path(path).exists():
            logger.warning(f"⚠️ CSV introuvable: {path} (asset {asset} ignoré)")
            continue

        df = pd.read_csv(path)

        # Vérifier colonnes requises
        required = ['high', 'low', 'close', 'timestamp']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Asset {asset}: colonnes manquantes {missing}")

        # Calculer ATR normalisé (RÉUTILISE fonction validée!)
        df['atr_norm'] = calculate_atr_normalized(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            window=14
        )

        dataframes[asset] = df
        logger.info(f"   {asset}: {len(df)} samples, ATR mean={df['atr_norm'].mean():.4f}")

    logger.info(f"✅ OHLCV chargé pour {len(dataframes)} assets")
    return dataframes


def extract_c_ret(X: np.ndarray) -> np.ndarray:
    """
    Extrait c_ret des features.

    RÉUTILISE: Logique exacte de test_holding_strategy.py.
    """
    # MACD: c_ret est feature 0
    c_ret = X[:, :, 0]
    return c_ret[:, -1]  # Dernier timestep


def compute_atr_from_ohlcv(
    ohlcv_dfs: Dict[str, pd.DataFrame],
    n_samples_total: int
) -> np.ndarray:
    """
    Calcule ATR depuis OHLCV (approche simplifiée sans alignement parfait).

    APPROCHE SIMPLIFIÉE (sans metadata par sample):
    - Concatène tous les assets (ordre: BTC → ETH → BNB → ADA → LTC)
    - Calcule ATR globalement
    - Mapping indices: samples ML correspondent aux derniers SEQUENCE_LENGTH steps

    NOTE: Moins précis que l'alignement parfait, mais suffisant pour valider hypothèse.
    Si test prometteur → améliorer alignement dans version ultérieure.

    Args:
        ohlcv_dfs: DataFrames OHLCV par asset
        n_samples_total: Nombre de samples dans le dataset ML

    Returns:
        ATR array (n_samples_total,)
    """
    logger.info(f"🔗 Calcul ATR depuis OHLCV (approche simplifiée)")

    # Concaténer tous les assets dans l'ordre (comme prepare_data_direction_only.py)
    asset_order = ['BTC', 'ETH', 'BNB', 'ADA', 'LTC']
    atr_all = []

    for asset in asset_order:
        if asset in ohlcv_dfs:
            df = ohlcv_dfs[asset]
            atr_all.append(df['atr_norm'].values)
            logger.info(f"   {asset}: {len(df)} samples, ATR mean={df['atr_norm'].mean():.4f}")
        else:
            logger.warning(f"⚠️ Asset {asset} non trouvé dans OHLCV")

    # Concaténer
    atr_concat = np.concatenate(atr_all)
    logger.info(f"   ATR total: {len(atr_concat)} samples")

    # Prendre les derniers n_samples_total (après TRIM edges + sequences)
    # NOTE: Approximation! Assume que samples ML = dernières sequences créées
    if len(atr_concat) >= n_samples_total:
        atr_subset = atr_concat[-n_samples_total:]
        logger.info(f"   ATR subset: {len(atr_subset)} samples (tail du dataset)")
    else:
        logger.warning(f"⚠️ ATR concat ({len(atr_concat)}) < samples ML ({n_samples_total})")
        # Remplir avec médiane si insuffisant
        atr_subset = np.full(n_samples_total, np.median(atr_concat))
        atr_subset[:len(atr_concat)] = atr_concat

    logger.info(f"✅ ATR calculé: mean={atr_subset.mean():.4f}, median={np.median(atr_subset):.4f}")

    return atr_subset


# =============================================================================
# BACKTEST AVEC FILTRE ATR
# =============================================================================

def backtest_atr_filter(
    pred: np.ndarray,
    returns: np.ndarray,
    atr: np.ndarray,
    fees: float,
    atr_q_low: float = 20.0,
    atr_q_high: float = 80.0
) -> StrategyResult:
    """
    Backtest avec filtre ATR percentile.

    RÉUTILISE: Logique calcul PnL de test_holding_strategy.py (PROUVÉE!).
    ADAPTE: Direction-Only (1 output) au lieu de Dual-Binary (2 outputs).
    AJOUTE: Filtre ATR Q20-Q80.

    Args:
        pred: Prédictions (n_samples, 1) - [Direction uniquement]
        returns: Returns (n_samples,)
        atr: ATR normalisé (n_samples,)
        fees: Frais par side (ex: 0.003 = 0.3%)
        atr_q_low: Percentile bas (défaut: 20)
        atr_q_high: Percentile haut (défaut: 80)

    Returns:
        StrategyResult
    """
    # Convertir en binaire
    pred_bin = (pred > 0.5).astype(int)  # Shape (n,1)
    pred_direction = pred_bin[:, 0]       # Shape (n,)

    # Calculer percentiles ATR
    q_low_val = np.percentile(atr, atr_q_low)
    q_high_val = np.percentile(atr, atr_q_high)

    # Masque ATR (Q20 <= ATR <= Q80)
    # NOTE: Utiliser <= pour inclure les bornes (important pour baseline Q0-Q100)
    if atr_q_low == 0.0 and atr_q_high == 100.0:
        # Baseline: tout autoriser
        atr_mask = np.ones(len(atr), dtype=bool)
    else:
        # Filtre: bornes inclusives
        atr_mask = (atr >= q_low_val) & (atr <= q_high_val)

    atr_coverage = atr_mask.mean()

    logger.info(f"📊 ATR Percentiles: Q{atr_q_low}={q_low_val:.4f}, Q{atr_q_high}={q_high_val:.4f}")
    logger.info(f"📊 ATR Coverage: {atr_coverage*100:.1f}% samples autorisés")

    n_samples = len(pred_direction)
    trades = []

    position = Position.FLAT
    entry_time = 0
    current_pnl = 0.0

    n_long = 0
    n_short = 0

    for i in range(n_samples):
        direction = int(pred_direction[i])  # 0=DOWN, 1=UP
        atr_ok = atr_mask[i]                # True si ATR dans range
        ret = returns[i]

        # Accumuler PnL (RÉUTILISE logique PROUVÉE!)
        if position != Position.FLAT:
            if position == Position.LONG:
                current_pnl += ret
            else:  # SHORT
                current_pnl -= ret

        # Decision Matrix (SIMPLIFIÉ vs Dual-Binary)
        if direction == 1 and atr_ok:
            target = Position.LONG
        elif direction == 0 and atr_ok:
            target = Position.SHORT
        else:
            target = Position.FLAT  # Filtre ATR bloque ou pas de signal

        # LOGIQUE SORTIE (SIMPLIFIÉ: Flip uniquement)
        exit_signal = False
        exit_reason = None

        if position != Position.FLAT:
            # Direction flip (RÉUTILISE logique commit e51a691!)
            if target != Position.FLAT and target != position:
                exit_signal = True
                exit_reason = "DIRECTION_FLIP"

            # Sortie si ATR sort du range (force exit)
            elif not atr_ok:
                exit_signal = True
                exit_reason = "ATR_OUT_OF_RANGE"

        # Exécuter sortie si nécessaire
        if exit_signal:
            trade_fees = 2 * fees  # Round-trip
            pnl_after_fees = current_pnl - trade_fees

            trades.append(Trade(
                start=entry_time,
                end=i,
                duration=i - entry_time,
                position=position.value,
                pnl=current_pnl,
                pnl_after_fees=pnl_after_fees,
                exit_reason=exit_reason
            ))

            # Flip immédiat si DIRECTION_FLIP (RÉUTILISE logique PROUVÉE!)
            if exit_reason == "DIRECTION_FLIP":
                position = target
                entry_time = i
                current_pnl = 0.0
                if target == Position.LONG:
                    n_long += 1
                else:
                    n_short += 1
            else:
                # Sortie complète (ATR out of range)
                position = Position.FLAT
                current_pnl = 0.0

        # Nouvelle entrée si FLAT
        elif position == Position.FLAT and target != Position.FLAT:
            position = target
            entry_time = i
            current_pnl = 0.0
            if target == Position.LONG:
                n_long += 1
            else:
                n_short += 1

    # Clôturer position finale si ouverte
    if position != Position.FLAT:
        trade_fees = 2 * fees
        pnl_after_fees = current_pnl - trade_fees

        trades.append(Trade(
            start=entry_time,
            end=n_samples - 1,
            duration=n_samples - 1 - entry_time,
            position=position.value,
            pnl=current_pnl,
            pnl_after_fees=pnl_after_fees,
            exit_reason="END"
        ))

    # Calculer métriques (RÉUTILISE formules de test_holding_strategy.py)
    if len(trades) == 0:
        return StrategyResult(
            name=f"ATR_Q{int(atr_q_low)}-Q{int(atr_q_high)}",
            n_trades=0, n_long=0, n_short=0,
            total_pnl=0.0, total_pnl_after_fees=0.0, total_fees=0.0,
            win_rate=0.0, profit_factor=0.0,
            avg_win=0.0, avg_loss=0.0, avg_duration=0.0,
            sharpe_ratio=0.0, trades=[],
            atr_coverage=atr_coverage,
            atr_q_low=q_low_val,
            atr_q_high=q_high_val
        )

    pnl_brut = sum(t.pnl for t in trades)
    pnl_net = sum(t.pnl_after_fees for t in trades)
    total_fees = pnl_brut - pnl_net

    wins = [t for t in trades if t.pnl_after_fees > 0]
    losses = [t for t in trades if t.pnl_after_fees <= 0]

    win_rate = len(wins) / len(trades) if trades else 0.0

    total_wins = sum(t.pnl_after_fees for t in wins)
    total_losses = abs(sum(t.pnl_after_fees for t in losses))
    profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

    avg_win = np.mean([t.pnl_after_fees for t in wins]) if wins else 0.0
    avg_loss = np.mean([t.pnl_after_fees for t in losses]) if losses else 0.0
    avg_duration = np.mean([t.duration for t in trades])

    # Sharpe Ratio
    trade_returns = np.array([t.pnl_after_fees for t in trades])
    sharpe = (trade_returns.mean() / trade_returns.std()) if trade_returns.std() > 0 else 0.0

    return StrategyResult(
        name=f"ATR_Q{int(atr_q_low)}-Q{int(atr_q_high)}",
        n_trades=len(trades),
        n_long=n_long,
        n_short=n_short,
        total_pnl=pnl_brut,
        total_pnl_after_fees=pnl_net,
        total_fees=total_fees,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_duration=avg_duration,
        sharpe_ratio=sharpe,
        trades=trades,
        atr_coverage=atr_coverage,
        atr_q_low=q_low_val,
        atr_q_high=q_high_val
    )


# =============================================================================
# AFFICHAGE RÉSULTATS
# =============================================================================

def print_comparison_table(results: List[StrategyResult], baseline: StrategyResult):
    """
    Affiche tableau comparatif (RÉUTILISE style de test_holding_strategy.py).
    """
    print("\n" + "="*100)
    print("RÉSULTATS ATR STRUCTURAL FILTER".center(100))
    print("="*100)

    # Header
    header = (
        f"{'Config':<15} | "
        f"{'Trades':>8} | {'Réduc':>7} | "
        f"{'WR':>6} | {'Δ WR':>7} | "
        f"{'PnL Brut':>10} | {'PnL Net':>10} | "
        f"{'Coverage':>9} | {'Verdict':<10}"
    )
    print(header)
    print("-"*100)

    # Baseline
    print(
        f"{'Baseline':<15} | "
        f"{baseline.n_trades:>8,} | {'-':>7} | "
        f"{baseline.win_rate*100:>5.2f}% | {'-':>7} | "
        f"{baseline.total_pnl:>+9.2f}% | {baseline.total_pnl_after_fees:>+9.2f}% | "
        f"{'-':>9} | {'Référence':<10}"
    )
    print("-"*100)

    # Autres configs
    for res in results:
        if res.name == "Baseline":
            continue

        # Calculer réduction (avec protection division par zéro)
        if baseline.n_trades > 0:
            reduction = (1 - res.n_trades / baseline.n_trades) * 100
        else:
            reduction = 0.0

        delta_wr = (res.win_rate - baseline.win_rate) * 100

        # Verdict
        if res.total_pnl_after_fees > 0:
            verdict = "✅ POSITIF"
        elif res.total_pnl_after_fees > baseline.total_pnl_after_fees:
            verdict = "⚠️ Mieux"
        else:
            verdict = "❌ Pire"

        print(
            f"{res.name:<15} | "
            f"{res.n_trades:>8,} | {reduction:>+6.1f}% | "
            f"{res.win_rate*100:>5.2f}% | {delta_wr:>+6.2f}% | "
            f"{res.total_pnl:>+9.2f}% | {res.total_pnl_after_fees:>+9.2f}% | "
            f"{res.atr_coverage*100:>8.1f}% | {verdict:<10}"
        )

    print("="*100)

    # Métriques détaillées du meilleur
    # Filtrer baseline pour trouver le meilleur
    non_baseline_results = [r for r in results if r.name != "Baseline"]

    if non_baseline_results:
        best = max(non_baseline_results, key=lambda r: r.total_pnl_after_fees)
        print(f"\n🏆 MEILLEURE CONFIG: {best.name}")
        print(f"   Trades: {best.n_trades:,} ({best.n_long:,} LONG + {best.n_short:,} SHORT)")
        print(f"   Win Rate: {best.win_rate*100:.2f}%")
        print(f"   PnL Brut: {best.total_pnl:+.2f}%")
        print(f"   PnL Net: {best.total_pnl_after_fees:+.2f}%")
        print(f"   Frais: {best.total_fees:.2f}%")
        print(f"   Profit Factor: {best.profit_factor:.2f}")
        print(f"   Sharpe Ratio: {best.sharpe_ratio:.2f}")
        print(f"   Avg Duration: {best.avg_duration:.1f} périodes")
        print(f"   ATR Coverage: {best.atr_coverage*100:.1f}%")
        print(f"   ATR Range: [{best.atr_q_low:.4f}, {best.atr_q_high:.4f}]")
    else:
        print(f"\n⚠️ Aucune config ATR testée (baseline uniquement)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test ATR Structural Filter")
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Split à tester (défaut: test)')
    parser.add_argument('--fees', type=float, default=0.003,
                        help='Frais par side (défaut: 0.003 = 0.3%%)')

    args = parser.parse_args()

    logger.info(f"🚀 Test ATR Structural Filter")
    logger.info(f"   Split: {args.split}")
    logger.info(f"   Frais: {args.fees*100:.2f}% par side ({args.fees*2*100:.2f}% round-trip)")

    # Charger dataset Direction-Only
    data = load_dataset_direction_only(args.split)

    X = data['X']
    Y = data['Y']
    Y_pred = data['Y_pred']
    metadata = data['metadata']

    if Y_pred is None:
        raise ValueError("Prédictions manquantes! Utiliser src/train.py avec --save-predictions")

    logger.info(f"✅ Dataset: X={X.shape}, Y={Y.shape}, Y_pred={Y_pred.shape}")

    # Extraire returns (RÉUTILISE logique validée!)
    returns = extract_c_ret(X)
    logger.info(f"✅ Returns extraits: {returns.shape}")

    # Charger OHLCV et calculer ATR
    ohlcv_dfs = load_ohlcv_data()

    # Calculer ATR (approche simplifiée sans metadata par sample)
    atr = compute_atr_from_ohlcv(ohlcv_dfs, len(Y_pred))

    # Tester configs ATR
    results = []

    # Baseline (sans filtre ATR)
    logger.info(f"\n{'='*60}\nBASELINE (sans filtre ATR)\n{'='*60}")

    # Pour baseline: utiliser ATR réel mais avec percentiles Q0-Q100 (tout autorisé)
    baseline = backtest_atr_filter(
        pred=Y_pred,
        returns=returns,
        atr=atr,  # ATR réel
        fees=args.fees,
        atr_q_low=0.0,   # Q0 = minimum absolu
        atr_q_high=100.0  # Q100 = maximum absolu
    )
    baseline.name = "Baseline"
    results.append(baseline)

    logger.info(f"   Trades: {baseline.n_trades:,}")
    logger.info(f"   Win Rate: {baseline.win_rate*100:.2f}%")
    logger.info(f"   PnL Brut: {baseline.total_pnl:+.2f}%")
    logger.info(f"   PnL Net: {baseline.total_pnl_after_fees:+.2f}%")

    # Tester configs ATR
    for q_low, q_high, name in PERCENTILE_CONFIGS:
        logger.info(f"\n{'='*60}\nATR Filter: Q{q_low}-Q{q_high} ({name})\n{'='*60}")

        res = backtest_atr_filter(
            pred=Y_pred,
            returns=returns,
            atr=atr,
            fees=args.fees,
            atr_q_low=q_low,
            atr_q_high=q_high
        )

        results.append(res)

        # Calculer réduction (avec protection division par zéro)
        if baseline.n_trades > 0:
            reduction_pct = (1 - res.n_trades / baseline.n_trades) * 100
            logger.info(f"   Trades: {res.n_trades:,} ({reduction_pct:+.1f}%)")
        else:
            logger.info(f"   Trades: {res.n_trades:,} (N/A)")

        logger.info(f"   Win Rate: {res.win_rate*100:.2f}% ({(res.win_rate-baseline.win_rate)*100:+.2f}%)")
        logger.info(f"   PnL Net: {res.total_pnl_after_fees:+.2f}%")
        logger.info(f"   Coverage: {res.atr_coverage*100:.1f}%")

    # Afficher tableau comparatif
    print_comparison_table(results, baseline)

    logger.info(f"\n✅ Tests terminés!")


if __name__ == "__main__":
    main()
