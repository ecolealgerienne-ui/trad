#!/usr/bin/env python3
"""
Backtesting de la stratégie Dual-Binary (Direction + Force).

ARCHITECTURE DUAL-BINARY:
    Chaque indicateur prédit 2 outputs:
    - Label 1 (Direction): UP (1) ou DOWN (0)
    - Label 2 (Force): STRONG (1) ou WEAK (0)

STRATÉGIE SIMPLE (Decision Matrix):
    if Direction == UP and Force == STRONG:
        → LONG
    elif Direction == DOWN and Force == STRONG:
        → SHORT
    else:
        → HOLD (filtrer signaux faibles)

DONNÉES D'ENTRÉE:
    Fichiers .npz générés par prepare_data_purified_dual_binary.py:
    - dataset_btc_eth_bnb_ada_ltc_rsi_dual_binary_kalman.npz
    - dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_kalman.npz
    - dataset_btc_eth_bnb_ada_ltc_cci_dual_binary_kalman.npz

    Shape Y: (n_samples, 2) → [direction, force]
    Shape X: (n_samples, 25, n_features)
        - RSI/MACD: n_features = 1 (c_ret)
        - CCI: n_features = 3 (h_ret, l_ret, c_ret)

CALCUL DU RENDEMENT:
    Utilise c_ret (Close return) extrait de X:
    - c_ret = (Close[t] - Close[t-1]) / Close[t-1]
    - Position LONG: PnL += c_ret
    - Position SHORT: PnL -= c_ret

FEES ET SLIPPAGE (APPROCHE CONSERVATRICE):
    Simulation pessimiste pour éviter les mauvaises surprises en production.

    Frais par trade (un côté: entrée OU sortie):
    - Binance Standard: 0.1% (0.001)
    - Slippage estimé: 0.05% (0.0005) - décalage prix signal → exécution
    - TOTAL PAR SIDE: 0.15% (0.0015)

    Frais par trade complet (aller-retour: entrée + sortie):
    - TOTAL ROUND-TRIP: 0.3% (0.003)

    Philosophie: Si rentable avec 0.3% de frais simulés, ce sera une
    "machine de guerre" avec les vrais frais Binance (0.1-0.2%).

    Paramètre --fees:
    - --fees 0.15 → 0.15% par side → 0.3% total (RECOMMANDÉ CONSERVATEUR)
    - --fees 0.1  → 0.1% par side → 0.2% total (Binance sans slippage)
    - --fees 0.02 → 0.02% par side → 0.04% total (Maker fees optimiste)

MÉTRIQUES:
    - Total Trades
    - Win Rate (% trades positifs)
    - Profit Factor (sum(wins) / abs(sum(losses)))
    - Total PnL (brut et net de frais)
    - Avg Duration (périodes par trade)

Usage:
    # Test MACD (recommandé: meilleur indicateur 86.9%)
    # Par défaut: fees 0.15% (conservateur avec slippage)
    python tests/test_dual_binary_trading.py \\
        --indicator macd \\
        --split test

    # Test avec prédictions modèle (si disponibles)
    python tests/test_dual_binary_trading.py \\
        --indicator macd \\
        --split test \\
        --use-predictions

    # Test avec fees personnalisés
    python tests/test_dual_binary_trading.py --indicator macd --split test --fees 0.1  # Sans slippage
    python tests/test_dual_binary_trading.py --indicator macd --split test --fees 0.02 # Maker fees optimiste

    # Test RSI ou CCI
    python tests/test_dual_binary_trading.py --indicator rsi --split test
    python tests/test_dual_binary_trading.py --indicator cci --split test
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import argparse
from typing import Tuple, List, Dict
from enum import Enum
from dataclasses import dataclass, field
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTES
# =============================================================================

# Mapping indicateurs → fichiers .npz
DATASET_PATHS = {
    'rsi': 'data/prepared/dataset_btc_eth_bnb_ada_ltc_rsi_dual_binary_kalman.npz',
    'macd': 'data/prepared/dataset_btc_eth_bnb_ada_ltc_macd_dual_binary_kalman.npz',
    'cci': 'data/prepared/dataset_btc_eth_bnb_ada_ltc_cci_dual_binary_kalman.npz',
}

# Index des features dans X (shape: n_samples, 25, n_features)
# RSI/MACD: n_features = 1 → c_ret à index 0
# CCI: n_features = 3 → h_ret (0), l_ret (1), c_ret (2)
FEATURE_INDICES = {
    'rsi': {'c_ret': 0},
    'macd': {'c_ret': 0},
    'cci': {'h_ret': 0, 'l_ret': 1, 'c_ret': 2},
}


# =============================================================================
# DATACLASSES
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
    end: int = 0
    duration: int = 0
    position: str = ""
    pnl: float = 0.0
    pnl_after_fees: float = 0.0
    direction_at_entry: int = 0  # 0=DOWN, 1=UP
    force_at_entry: int = 0      # 0=WEAK, 1=STRONG


@dataclass
class Context:
    """État du contexte de trading."""
    position: Position = Position.FLAT
    entry_time: int = 0
    trades: List[Trade] = field(default_factory=list)
    current_pnl: float = 0.0
    direction_at_entry: int = 0
    force_at_entry: int = 0
    # Confirmation temporelle
    prev_target: Position = Position.FLAT
    confirmation_count: int = 0
    # Délai post-transition (éviter faux tops/bottoms)
    prev_direction: int = -1  # Direction de la période précédente
    periods_since_transition: int = 999  # Périodes depuis dernier changement Direction
    transitions_blocked: int = 0  # Compteur de transitions bloquées


# =============================================================================
# FONCTIONS PRINCIPALES
# =============================================================================

def load_dataset(indicator: str, split: str = 'test') -> Dict:
    """
    Charge le dataset .npz pour l'indicateur spécifié.

    Args:
        indicator: 'rsi', 'macd', ou 'cci'
        split: 'train', 'val', ou 'test'

    Returns:
        dict avec X, Y, Y_pred (si disponible), metadata
    """
    if indicator not in DATASET_PATHS:
        raise ValueError(f"Indicateur inconnu: {indicator}. Choix: {list(DATASET_PATHS.keys())}")

    path = DATASET_PATHS[indicator]
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset introuvable: {path}\n"
            f"Exécuter d'abord: python src/prepare_data_purified_dual_binary.py --assets BTC ETH BNB ADA LTC"
        )

    logger.info(f"📂 Chargement: {path}")
    data = np.load(path, allow_pickle=True)

    result = {
        'X': data[f'X_{split}'],
        'Y': data[f'Y_{split}'],
        'Y_pred': data.get(f'Y_{split}_pred', None),
        'indicator': indicator,
    }

    # Charger metadata si disponible
    if 'metadata' in data:
        import json
        metadata_str = str(data['metadata'])
        result['metadata'] = json.loads(metadata_str)
    else:
        result['metadata'] = {}

    logger.info(f"  ✅ Chargé: {indicator.upper()}")
    logger.info(f"     X shape: {result['X'].shape}")
    logger.info(f"     Y shape: {result['Y'].shape}")
    if result['Y_pred'] is not None:
        logger.info(f"     Y_pred shape: {result['Y_pred'].shape}")
    else:
        logger.info(f"     ⚠️  Prédictions non disponibles (utiliser --use-predictions après entraînement)")

    return result


def extract_c_ret(X: np.ndarray, indicator: str) -> np.ndarray:
    """
    Extrait c_ret (Close return) de X.

    Args:
        X: Features (n_samples, sequence_length, n_features)
        indicator: 'rsi', 'macd', ou 'cci'

    Returns:
        c_ret pour chaque sample (n_samples,)
        Utilise la dernière valeur de la séquence (t=-1)
    """
    if indicator in ['rsi', 'macd']:
        # 1 feature (c_ret) → index 0
        c_ret = X[:, -1, 0]  # Shape: (n_samples,)
    elif indicator == 'cci':
        # 3 features (h_ret, l_ret, c_ret) → c_ret à index 2
        c_ret = X[:, -1, 2]  # Shape: (n_samples,)
    else:
        raise ValueError(f"Indicateur inconnu: {indicator}")

    return c_ret


def _convert_to_binary_labels(
    signals: np.ndarray,
    mode: str,
    threshold_direction: float = 0.5,
    threshold_force: float = 0.5
) -> np.ndarray:
    """
    Convertit probabilités en labels binaires si nécessaire.

    Args:
        signals: Array (n_samples, 2) - labels ou probabilités
        mode: 'Oracle' ou 'Prédictions' (pour logging)
        threshold_direction: Seuil pour Direction (défaut: 0.5)
        threshold_force: Seuil pour Force (défaut: 0.5)

    Returns:
        Binary labels {0, 1}
    """
    # Vérifier si conversion nécessaire
    if signals.max() <= 1.0 and signals.min() >= 0.0:
        unique_vals = np.unique(signals)
        if len(unique_vals) > 2:  # Plus de 2 valeurs → probabilités continues
            # Appliquer seuils séparés pour Direction et Force
            direction = (signals[:, 0] > threshold_direction).astype(int)
            force = (signals[:, 1] > threshold_force).astype(int)
            signals = np.column_stack([direction, force])

            logger.info(f"   📊 {mode} converties - Direction seuil={threshold_direction}, Force seuil={threshold_force}")
            # Afficher distribution
            dir_up = (signals[:, 0] == 1).sum()
            force_strong = (signals[:, 1] == 1).sum()
            logger.info(f"   📊 Distribution: Direction UP={dir_up/len(signals)*100:.1f}%, Force STRONG={force_strong/len(signals)*100:.1f}%")
        else:
            # Déjà binaire, afficher juste la distribution
            dir_up = (signals[:, 0] == 1).sum()
            force_strong = (signals[:, 1] == 1).sum()
            logger.info(f"   📊 Distribution: Direction UP={dir_up/len(signals)*100:.1f}%, Force STRONG={force_strong/len(signals)*100:.1f}%")

    return signals


def run_dual_binary_strategy(
    Y: np.ndarray,
    returns: np.ndarray,
    fees: float = 0.0,
    use_predictions: bool = False,
    Y_pred: np.ndarray = None,
    threshold_force: float = 0.5,
    min_confirmation: int = 1,
    transition_delay: int = 0,
    continuations_only: bool = False,
    oracle_transition_filter: bool = False,
    require_oracle_agreement: bool = False,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Stratégie Dual-Binary simple (Decision Matrix) avec confirmation temporelle.

    Args:
        Y: Labels (n_samples, 2) → [direction, force]
           OU prédictions si use_predictions=True
        returns: Rendements c_ret (n_samples,)
        fees: Frais par trade (ex: 0.001 = 0.1%)
        use_predictions: Si True, utiliser Y_pred au lieu de Y
        Y_pred: Prédictions (si use_predictions=True)
        threshold_force: Seuil pour Force (défaut: 0.5)
        min_confirmation: Nombre de périodes de signal stable requis avant d'agir (défaut: 1)
        transition_delay: Périodes d'attente après changement Direction (défaut: 0 = désactivé)
        continuations_only: Si True, trader SEULEMENT les continuations (Direction stable 3+ périodes)
        oracle_transition_filter: TEST ORACLE - Bloquer sorties IA si transition Oracle détectée
        require_oracle_agreement: TEST TIMING - Trader SEULEMENT si IA et Oracle d'accord au même moment

    Returns:
        positions: Array des positions (n_samples,)
        stats: Statistiques de trading
    """
    if use_predictions:
        if Y_pred is None:
            raise ValueError("use_predictions=True mais Y_pred est None")
        signals = Y_pred
        logger.info("🎯 Mode: Prédictions modèle")
        signals = _convert_to_binary_labels(signals, "Prédictions", threshold_force=threshold_force)
    else:
        signals = Y
        logger.info("🎯 Mode: Labels Oracle (monde parfait)")
        signals = _convert_to_binary_labels(signals, "Labels Oracle", threshold_force=threshold_force)

    n_samples = len(signals)
    positions = np.zeros(n_samples, dtype=int)
    ctx = Context()

    # Stats
    stats = {
        'n_trades': 0,
        'n_long': 0,
        'n_short': 0,
        'n_hold': 0,
        'total_pnl': 0.0,
        'total_pnl_after_fees': 0.0,
        'total_fees': 0.0,
        'fees_per_trade': fees,
        'transitions_blocked': 0,  # Compteur entrées bloquées par transition_delay
        'transitions_ignored': 0,  # Compteur transitions ignorées (continuations_only)
        'oracle_transitions_blocked': 0,  # TEST ORACLE: Sorties bloquées car transition Oracle
        'oracle_disagreements': 0,  # TEST TIMING: Entrées bloquées car IA ≠ Oracle
    }

    for i in range(n_samples):
        direction = int(signals[i, 0])  # 0=DOWN, 1=UP
        force = int(signals[i, 1])      # 0=WEAK, 1=STRONG
        ret = returns[i]

        # Accumuler PnL si en position
        if ctx.position != Position.FLAT:
            if ctx.position == Position.LONG:
                ctx.current_pnl += ret
            else:  # SHORT
                ctx.current_pnl -= ret

        # ============================================
        # DÉTECTION DE TRANSITION (éviter faux tops/bottoms)
        # ============================================
        if transition_delay > 0:
            # Détecter changement de Direction
            if ctx.prev_direction != -1 and direction != ctx.prev_direction:
                # Transition détectée !
                ctx.periods_since_transition = 0
            else:
                # Pas de transition, incrémenter compteur
                ctx.periods_since_transition += 1

            # Mettre à jour Direction précédente
            ctx.prev_direction = direction

        # ============================================
        # DECISION MATRIX (Stratégie Simple - ORIGINALE)
        # ============================================
        # LOGIQUE VALIDÉE : Exit on Force WEAK (réactif)
        # - Crypto 5min : Trends courtes (~45min), Force WEAK = affaiblissement réel
        # - PnL Brut +49.84% avec cette logique (vs -271% avec Hold on WEAK)
        # ============================================

        if direction == 1 and force == 1:
            # UP + STRONG → LONG
            target_position = Position.LONG
        elif direction == 0 and force == 1:
            # DOWN + STRONG → SHORT
            target_position = Position.SHORT
        else:
            # Autres (signaux WEAK) → HOLD (sortir de position)
            target_position = Position.FLAT
            stats['n_hold'] += 1

        # ============================================
        # FILTRAGE CONTINUATIONS UNIQUEMENT (Solution 2)
        # ============================================
        is_continuation = True  # Par défaut, autoriser

        if continuations_only and i >= 2:
            # Vérifier si Direction est STABLE depuis 3 périodes
            # Continuation = direction[i] == direction[i-1] == direction[i-2]
            prev_dir_1 = int(signals[i-1, 0])
            prev_dir_2 = int(signals[i-2, 0])

            is_continuation = (direction == prev_dir_1 == prev_dir_2)

            if not is_continuation and target_position != Position.FLAT:
                # C'est une transition (Direction a changé) - IGNORER
                stats['transitions_ignored'] += 1
                # Forcer FLAT pour ne pas entrer
                target_position = Position.FLAT

        # ============================================
        # CONFIRMATION TEMPORELLE
        # ============================================

        # Mettre à jour compteur de confirmation
        if target_position == ctx.prev_target:
            ctx.confirmation_count += 1
        else:
            ctx.prev_target = target_position
            ctx.confirmation_count = 1

        # Agir seulement si signal confirmé pendant min_confirmation périodes
        confirmed = (ctx.confirmation_count >= min_confirmation)

        # ============================================
        # TEST ORACLE: Bloquer sorties sur transitions Oracle
        # ============================================
        if oracle_transition_filter and i >= 1 and ctx.position != Position.FLAT:
            # On est en position et IA veut sortir/inverser
            if target_position != ctx.position:
                # Vérifier si c'est une transition ORACLE (vraies labels)
                oracle_dir_current = int(Y[i, 0])
                oracle_dir_previous = int(Y[i-1, 0])
                oracle_transition = (oracle_dir_current != oracle_dir_previous)

                if oracle_transition:
                    # TRANSITION ORACLE détectée → BLOQUER la sortie IA
                    # Forcer à garder la position actuelle
                    target_position = ctx.position
                    stats['oracle_transitions_blocked'] += 1

        # ============================================
        # TEST TIMING: Trader SEULEMENT si IA et Oracle d'accord
        # ============================================
        if require_oracle_agreement and target_position != Position.FLAT:
            # Vérifier si IA et Oracle sont d'accord sur Direction
            oracle_dir = int(Y[i, 0])  # 0=DOWN, 1=UP
            ia_dir = direction  # From predictions

            if oracle_dir != ia_dir:
                # Désaccord IA/Oracle → BLOQUER l'entrée
                target_position = Position.FLAT
                stats['oracle_disagreements'] += 1

        # ============================================
        # LOGIQUE DE TRADING (avec confirmation)
        # ============================================

        # Sortie si target_position = FLAT et on est en position ET confirmé
        if confirmed and target_position == Position.FLAT and ctx.position != Position.FLAT:
            # Sortir de position (signal faible)
            trade_fees = 2 * fees  # Entrée + sortie
            pnl_after_fees = ctx.current_pnl - trade_fees

            trade = Trade(
                start=ctx.entry_time,
                end=i,
                duration=i - ctx.entry_time,
                position=ctx.position.value,
                pnl=ctx.current_pnl,
                pnl_after_fees=pnl_after_fees,
                direction_at_entry=ctx.direction_at_entry,
                force_at_entry=ctx.force_at_entry
            )
            ctx.trades.append(trade)

            stats['n_trades'] += 1
            stats['total_pnl'] += ctx.current_pnl
            stats['total_pnl_after_fees'] += pnl_after_fees
            stats['total_fees'] += trade_fees

            ctx.position = Position.FLAT
            ctx.current_pnl = 0.0

        # Entrée ou changement de direction (seulement si confirmé)
        elif confirmed and target_position != Position.FLAT:
            # BLOQUER ENTRÉES pendant transition_delay (éviter faux tops/bottoms)
            if transition_delay > 0 and ctx.periods_since_transition < transition_delay:
                # Trop tôt après transition - NE PAS ENTRER
                ctx.transitions_blocked += 1
                pass  # Garder position actuelle (FLAT ou existante)

            elif ctx.position == Position.FLAT:
                # Nouvelle entrée
                ctx.position = target_position
                ctx.entry_time = i
                ctx.current_pnl = 0.0
                ctx.direction_at_entry = direction
                ctx.force_at_entry = force

                if target_position == Position.LONG:
                    stats['n_long'] += 1
                else:
                    stats['n_short'] += 1

            elif ctx.position != target_position:
                # Changement de direction = sortie + nouvelle entrée
                trade_fees = 2 * fees
                pnl_after_fees = ctx.current_pnl - trade_fees

                trade = Trade(
                    start=ctx.entry_time,
                    end=i,
                    duration=i - ctx.entry_time,
                    position=ctx.position.value,
                    pnl=ctx.current_pnl,
                    pnl_after_fees=pnl_after_fees,
                    direction_at_entry=ctx.direction_at_entry,
                    force_at_entry=ctx.force_at_entry
                )
                ctx.trades.append(trade)

                stats['n_trades'] += 1
                stats['total_pnl'] += ctx.current_pnl
                stats['total_pnl_after_fees'] += pnl_after_fees
                stats['total_fees'] += trade_fees

                # Nouvelle entrée dans l'autre sens
                ctx.position = target_position
                ctx.entry_time = i
                ctx.current_pnl = 0.0
                ctx.direction_at_entry = direction
                ctx.force_at_entry = force

                if target_position == Position.LONG:
                    stats['n_long'] += 1
                else:
                    stats['n_short'] += 1

        # Enregistrer position courante
        if ctx.position == Position.LONG:
            positions[i] = 1
        elif ctx.position == Position.SHORT:
            positions[i] = -1
        else:
            positions[i] = 0

    # Fermer position ouverte à la fin
    if ctx.position != Position.FLAT:
        trade_fees = 2 * fees
        pnl_after_fees = ctx.current_pnl - trade_fees

        trade = Trade(
            start=ctx.entry_time,
            end=n_samples - 1,
            duration=n_samples - 1 - ctx.entry_time,
            position=ctx.position.value,
            pnl=ctx.current_pnl,
            pnl_after_fees=pnl_after_fees,
            direction_at_entry=ctx.direction_at_entry,
            force_at_entry=ctx.force_at_entry
        )
        ctx.trades.append(trade)

        stats['n_trades'] += 1
        stats['total_pnl'] += ctx.current_pnl
        stats['total_pnl_after_fees'] += pnl_after_fees
        stats['total_fees'] += trade_fees

    # Calculer métriques finales
    if ctx.trades:
        pnls = [t.pnl for t in ctx.trades]
        pnls_after_fees = [t.pnl_after_fees for t in ctx.trades]
        wins = [p for p in pnls_after_fees if p > 0]
        losses = [p for p in pnls_after_fees if p <= 0]

        stats['win_rate'] = len(wins) / len(pnls) if pnls else 0
        stats['avg_win'] = np.mean(wins) if wins else 0
        stats['avg_loss'] = np.mean(losses) if losses else 0
        stats['profit_factor'] = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float('inf')
        stats['avg_duration'] = np.mean([t.duration for t in ctx.trades])
        stats['trades'] = ctx.trades
    else:
        stats['win_rate'] = 0
        stats['avg_win'] = 0
        stats['avg_loss'] = 0
        stats['profit_factor'] = 0
        stats['avg_duration'] = 0
        stats['trades'] = []

    # Copier compteurs
    stats['transitions_blocked'] = ctx.transitions_blocked
    # transitions_ignored déjà dans stats (compteur direct)

    return positions, stats


def print_results(stats: Dict, indicator: str, split: str, use_predictions: bool, n_samples: int = 0, n_assets: int = 5):
    """Affiche les résultats du backtest."""
    mode = "Prédictions" if use_predictions else "Oracle"

    # Calculer métriques temporelles
    periods_per_asset = n_samples // n_assets if n_assets > 0 else n_samples
    minutes_total = periods_per_asset * 5  # Périodes de 5 min
    days_total = minutes_total / (60 * 24)
    months_total = days_total / 30.0

    logger.info("\n" + "=" * 70)
    logger.info(f"📊 RÉSULTATS BACKTEST - {indicator.upper()} ({mode})")
    logger.info("=" * 70)

    logger.info(f"\n📅 Période:")
    logger.info(f"   Total samples:    {n_samples:,}")
    logger.info(f"   Assets:           {n_assets}")
    logger.info(f"   Samples/asset:    {periods_per_asset:,}")
    logger.info(f"   Durée/asset:      {days_total:.0f} jours (~{months_total:.1f} mois)")

    logger.info(f"\n📈 Trades:")
    logger.info(f"  Total Trades:     {stats['n_trades']:,}")
    logger.info(f"  LONG:             {stats['n_long']:,}")
    logger.info(f"  SHORT:            {stats['n_short']:,}")
    logger.info(f"  HOLD (filtered):  {stats['n_hold']:,}")
    if stats.get('transitions_blocked', 0) > 0:
        logger.info(f"  Transitions bloquées: {stats['transitions_blocked']:,} (délai post-transition)")
    if stats.get('transitions_ignored', 0) > 0:
        logger.info(f"  Transitions ignorées: {stats['transitions_ignored']:,} (continuations uniquement)")
    if stats.get('oracle_transitions_blocked', 0) > 0:
        logger.info(f"  🎯 Sorties bloquées: {stats['oracle_transitions_blocked']:,} (TEST ORACLE: transitions détectées)")
    if stats.get('oracle_disagreements', 0) > 0:
        logger.info(f"  ⏰ Entrées bloquées: {stats['oracle_disagreements']:,} (TEST TIMING: IA ≠ Oracle)")
    logger.info(f"  Avg Duration:     {stats['avg_duration']:.1f} périodes")

    logger.info(f"\n💰 Performance:")
    logger.info(f"  Win Rate:         {stats['win_rate']*100:.2f}%")
    logger.info(f"  Profit Factor:    {stats['profit_factor']:.3f}")
    logger.info(f"  Avg Win:          {stats['avg_win']*100:+.3f}%")
    logger.info(f"  Avg Loss:         {stats['avg_loss']*100:+.3f}%")

    logger.info(f"\n💵 PnL:")
    logger.info(f"  PnL Brut:         {stats['total_pnl']*100:+.2f}%")
    logger.info(f"  Frais Totaux:     {stats['total_fees']*100:.2f}%")
    logger.info(f"  PnL Net:          {stats['total_pnl_after_fees']*100:+.2f}%")

    # Métriques par asset et par mois
    if n_assets > 0 and months_total > 0:
        pnl_per_asset = (stats['total_pnl_after_fees'] * 100) / n_assets
        pnl_per_month = (stats['total_pnl_after_fees'] * 100) / months_total
        logger.info(f"\n📊 Performance Normalisée:")
        logger.info(f"  PnL Net/asset:    {pnl_per_asset:+.2f}%")
        logger.info(f"  PnL Net/mois:     {pnl_per_month:+.2f}%")

    logger.info("\n" + "=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Backtest Stratégie Dual-Binary (Direction + Force)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--indicator',
        type=str,
        required=True,
        choices=['rsi', 'macd', 'cci'],
        help="Indicateur à tester (recommandé: macd)"
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help="Split à tester (défaut: test)"
    )
    parser.add_argument(
        '--fees',
        type=float,
        default=0.15,
        help="Frais par side en %% (défaut: 0.15%% = Binance 0.1%% + Slippage 0.05%%)"
    )
    parser.add_argument(
        '--use-predictions',
        action='store_true',
        help="Utiliser prédictions modèle au lieu de labels Oracle"
    )
    parser.add_argument(
        '--threshold-force',
        type=float,
        default=0.5,
        help="Seuil pour Force (défaut: 0.5). Baisser à 0.3-0.4 pour augmenter Recall STRONG"
    )
    parser.add_argument(
        '--min-confirmation',
        type=int,
        default=1,
        help="Périodes de signal stable requis avant d'agir (défaut: 1). 2-3 réduit flickering."
    )
    parser.add_argument(
        '--transition-delay',
        type=int,
        default=0,
        help="Périodes d'attente après changement Direction avant d'entrer (défaut: 0 = désactivé). 3-5 évite faux tops/bottoms."
    )
    parser.add_argument(
        '--continuations-only',
        action='store_true',
        help="SOLUTION 2: Trader SEULEMENT les continuations (Direction stable 3+ périodes). Abandonne toutes les transitions."
    )
    parser.add_argument(
        '--oracle-transition-filter',
        action='store_true',
        help="TEST ORACLE: Bloquer sorties IA si transition Oracle détectée. Isole l'impact des transitions."
    )
    parser.add_argument(
        '--require-oracle-agreement',
        action='store_true',
        help="TEST TIMING: Trader SEULEMENT si IA et Oracle d'accord au même moment. Teste timing/amplitude."
    )

    args = parser.parse_args()

    # Convertir fees en décimal
    fees_decimal = args.fees / 100.0

    # Charger données
    data = load_dataset(args.indicator, args.split)

    # Extraire c_ret
    returns = extract_c_ret(data['X'], args.indicator)

    # Vérifier prédictions si demandées
    if args.use_predictions and data['Y_pred'] is None:
        logger.error(
            "❌ Prédictions non disponibles dans le fichier .npz\n"
            "   Exécuter d'abord: python src/train.py --data <dataset> --indicator <indicator>\n"
            "   Puis: python src/evaluate.py --data <dataset> (sauvegarde Y_pred dans .npz)"
        )
        sys.exit(1)

    # Run backtest
    logger.info(f"\n🚀 Lancement backtest: {args.indicator.upper()} ({args.split})")
    logger.info(f"   Samples: {len(data['Y']):,}")
    logger.info(f"   Frais: {args.fees}% par side → {args.fees*2:.2f}% total aller-retour")
    if args.threshold_force != 0.5:
        logger.info(f"   ⚙️  Seuil Force personnalisé: {args.threshold_force}")
    if args.min_confirmation > 1:
        logger.info(f"   ⏱️  Confirmation temporelle: {args.min_confirmation} périodes")
    if args.transition_delay > 0:
        logger.info(f"   🚦 Délai post-transition: {args.transition_delay} périodes (évite faux tops/bottoms)")
    if args.continuations_only:
        logger.info(f"   ⚡ SOLUTION 2: Continuations UNIQUEMENT (Direction stable 3+ périodes)")
    if args.oracle_transition_filter:
        logger.info(f"   🎯 TEST ORACLE: Bloquer sorties si transition Oracle détectée")
    if args.require_oracle_agreement:
        logger.info(f"   ⏰ TEST TIMING: Trader SEULEMENT si IA et Oracle d'accord (teste timing/amplitude)")

    positions, stats = run_dual_binary_strategy(
        Y=data['Y'],
        returns=returns,
        fees=fees_decimal,
        use_predictions=args.use_predictions,
        Y_pred=data['Y_pred'],
        threshold_force=args.threshold_force,
        min_confirmation=args.min_confirmation,
        transition_delay=args.transition_delay,
        continuations_only=args.continuations_only,
        oracle_transition_filter=args.oracle_transition_filter,
        require_oracle_agreement=args.require_oracle_agreement
    )

    # Afficher résultats
    print_results(stats, args.indicator, args.split, args.use_predictions, n_samples=len(data['Y']), n_assets=5)

    logger.info("\n✅ Backtest terminé")


if __name__ == '__main__':
    main()
