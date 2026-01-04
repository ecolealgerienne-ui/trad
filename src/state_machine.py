"""
State Machine pour le trading basée sur les prédictions ML.

Architecture validée par expert (2026-01-04):
- MACD = pivot (décide la direction)
- RSI/CCI = modulateurs
- Octave/Kalman = confiance structurelle

Mode TRANSITION-ONLY (nouveau - 2026-01-04):
- Entrer SEULEMENT sur CHANGEMENT vers TOTAL (pas continuation)
- Signal doit être stable pendant min_stability périodes
- Objectif: Réduire trades de 80-90% pour rendre les frais viables
- Usage: --transition-only --min-stability 2

Mode STRICT (recommandé - validé empiriquement):
- Seul l'accord TOTAL autorise les entrées
- PARTIEL et FORT = FLAT (pas de trade)
- Résultat test: +1300% (TOTAL) vs -286% (PARTIEL)

Mode NORMAL (déprécié):
- TOTAL = entrée immédiate
- PARTIEL = entrée après 2 confirmations
- FORT = bloqué

Usage:
    # Mode Transition-Only (recommandé pour frais réels)
    python src/state_machine.py \
        --rsi-octave data/prepared/dataset_..._rsi_octave20.npz \
        --cci-octave data/prepared/dataset_..._cci_octave20.npz \
        --macd-octave data/prepared/dataset_..._macd_octave20.npz \
        --rsi-kalman data/prepared/dataset_..._rsi_kalman.npz \
        --cci-kalman data/prepared/dataset_..._cci_kalman.npz \
        --macd-kalman data/prepared/dataset_..._macd_kalman.npz \
        --split test --transition-only --min-stability 2 --fees 0.1

    # Mode Strict (sans frais)
    python src/state_machine.py \
        ... --split test --strict
"""

import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum


class Position(Enum):
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


class Agreement(Enum):
    TOTAL = "TOTAL"      # Tous d'accord → agir vite
    PARTIEL = "PARTIEL"  # Désaccord partiel → confirmation requise
    FORT = "FORT"        # Désaccord fort → ne rien faire (sauf sortie)


@dataclass
class Context:
    """État du contexte de trading."""
    position: Position = Position.FLAT
    entry_time: int = 0
    entry_price: float = 0.0
    last_transition: int = 0
    confirmation_count: int = 0
    exit_delay_count: int = 0
    prev_macd: int = -1  # -1 = pas de valeur précédente

    # Pour Transition-Only mode
    prev_agreement: Agreement = None
    signal_stability: int = 0  # Nombre de périodes avec même signal
    prev_direction: int = -1   # Direction précédente (0=SHORT, 1=LONG)

    # Statistiques
    trades: List = field(default_factory=list)
    current_trade_start: int = 0


def load_dataset(path: str, split: str = 'test') -> dict:
    """
    Charge un dataset et retourne les données pour le split demandé.
    """
    data = np.load(path, allow_pickle=True)

    result = {
        'X': data[f'X_{split}'],
        'Y': data[f'Y_{split}'],
        'Y_pred': data.get(f'Y_{split}_pred', None),
        'metadata': None,
        'assets': None,
        'samples_per_asset': None
    }

    # Charger les prédictions si disponibles
    pred_key = f'Y_{split}_pred'
    if pred_key in data:
        result['Y_pred'] = data[pred_key]

    # Charger métadonnées
    if 'metadata' in data:
        try:
            meta = json.loads(str(data['metadata']))
            result['metadata'] = meta
            # Extraire les assets et samples par asset
            if 'assets' in meta:
                result['assets'] = meta['assets']
            if f'samples_per_asset_{split}' in meta:
                result['samples_per_asset'] = meta[f'samples_per_asset_{split}']
        except:
            pass

    return result


def validate_indices_sync(datasets: dict, verbose: bool = True) -> bool:
    """
    Vérifie que tous les datasets sont synchronisés (mêmes indices/features).

    CRITIQUE: Les features OHLC doivent être IDENTIQUES entre tous les datasets.
    Seuls les labels et prédictions diffèrent.

    Args:
        datasets: Dict avec clés = nom, valeurs = dict avec X, Y, Y_pred
        verbose: Afficher les détails

    Returns:
        True si synchronisés, False sinon
    """
    if verbose:
        print("\n" + "="*80)
        print("VALIDATION DE LA SYNCHRONISATION DES INDICES")
        print("="*80)

    names = list(datasets.keys())
    if len(names) < 2:
        if verbose:
            print("   ⚠️ Moins de 2 datasets, pas de validation nécessaire")
        return True

    reference_name = names[0]
    reference = datasets[reference_name]
    X_ref = reference['X']

    all_synced = True

    for name in names[1:]:
        data = datasets[name]
        X = data['X']

        # Vérifier les shapes
        if X.shape != X_ref.shape:
            if verbose:
                print(f"   ❌ {name}: Shape différente {X.shape} vs {X_ref.shape}")
            all_synced = False
            continue

        # Vérifier que les features sont identiques
        if not np.allclose(X, X_ref, rtol=1e-10, atol=1e-10):
            max_diff = np.max(np.abs(X - X_ref))
            if verbose:
                print(f"   ❌ {name}: Features différentes (max_diff={max_diff:.2e})")
            all_synced = False
        else:
            if verbose:
                print(f"   ✅ {name}: Features identiques à {reference_name}")

    if verbose:
        if all_synced:
            print(f"\n✅ Tous les datasets sont synchronisés ({len(names)} datasets)")
        else:
            print(f"\n❌ ERREUR: Datasets non synchronisés!")

    return all_synced


def get_agreement_level(
    macd_pred: int,
    rsi_pred: int,
    cci_pred: int,
    octave_dir: int,
    kalman_dir: int
) -> Agreement:
    """
    Retourne le niveau d'accord des signaux.

    Args:
        macd_pred, rsi_pred, cci_pred: Prédictions des indicateurs (0 ou 1)
        octave_dir, kalman_dir: Direction des filtres (0 ou 1)

    Returns:
        Agreement level
    """
    indicators_agree = (macd_pred == rsi_pred == cci_pred)
    filters_agree = (octave_dir == kalman_dir)

    if indicators_agree and filters_agree:
        return Agreement.TOTAL
    elif not indicators_agree and not filters_agree:
        return Agreement.FORT
    else:
        return Agreement.PARTIEL


def update_confirmation(
    macd_pred: int,
    agreement: Agreement,
    ctx: Context
) -> None:
    """
    Met à jour le compteur de confirmation de manière directionnelle.

    La confirmation doit être:
    - Directionnelle (MACD stable)
    - Cohérente (pas de désaccord fort)
    - Réinitialisable (reset si contradiction)
    """
    macd_stable = (macd_pred == ctx.prev_macd) or (ctx.prev_macd == -1)

    if macd_stable and agreement != Agreement.FORT:
        ctx.confirmation_count += 1
    else:
        ctx.confirmation_count = 0  # RESET obligatoire

    # Reset aussi le délai de sortie si direction change
    if not macd_stable:
        ctx.exit_delay_count = 0

    # Mettre à jour prev_macd
    ctx.prev_macd = macd_pred


def should_enter(
    macd_pred: int,
    rsi_pred: int,
    cci_pred: int,
    octave_dir: int,
    kalman_dir: int,
    ctx: Context,
    current_time: int,
    strict: bool = False,
    transition_only: bool = False,
    min_stability: int = 2
) -> Optional[Position]:
    """
    Décide si on doit entrer en position.

    Args:
        strict: Si True, seul TOTAL autorise l'entrée (PARTIEL bloqué)
        transition_only: Si True, entrer seulement sur CHANGEMENT vers TOTAL
        min_stability: Nombre de périodes de stabilité du signal avant entrée

    Returns:
        Position.LONG, Position.SHORT, ou None si pas d'entrée
    """
    if ctx.position != Position.FLAT:
        return None

    agreement = get_agreement_level(macd_pred, rsi_pred, cci_pred, octave_dir, kalman_dir)
    time_since_transition = current_time - ctx.last_transition

    # Règle 1: MACD décide la direction
    direction = Position.LONG if macd_pred == 1 else Position.SHORT

    # Mode TRANSITION-ONLY (prioritaire sur STRICT)
    if transition_only:
        # Condition 1: TRANSITION vers TOTAL (pas continuation)
        is_transition = (ctx.prev_agreement != Agreement.TOTAL and agreement == Agreement.TOTAL)
        if not is_transition:
            return None

        # Condition 2: Signal stable pendant min_stability périodes
        if ctx.signal_stability < min_stability:
            return None

        return direction

    # Mode STRICT: seul TOTAL autorise l'entrée
    if strict:
        if agreement != Agreement.TOTAL:
            return None
        return direction

    # Mode NORMAL (déprécié)
    # Règle 2: Confirmation conditionnelle
    if agreement == Agreement.FORT:
        return None  # Aucune action
    elif agreement == Agreement.PARTIEL:
        if ctx.confirmation_count < 2:
            return None  # Attendre confirmation
    # agreement == TOTAL → pas de confirmation requise

    # Règle 3: Délai post-transition MACD
    if agreement != Agreement.TOTAL and time_since_transition < 1:
        return None

    return direction


def should_exit(
    macd_pred: int,
    rsi_pred: int,
    cci_pred: int,
    octave_dir: int,
    kalman_dir: int,
    ctx: Context
) -> bool:
    """
    Décide si on doit sortir de position.

    RÈGLE CRITIQUE: Ne JAMAIS bloquer une sortie MACD indéfiniment.

    Returns:
        True si sortie, False sinon
    """
    if ctx.position == Position.FLAT:
        return False

    # Signal opposé à la position?
    if ctx.position == Position.LONG and macd_pred == 0:
        exit_signal = True
    elif ctx.position == Position.SHORT and macd_pred == 1:
        exit_signal = True
    else:
        exit_signal = False

    if not exit_signal:
        return False

    agreement = get_agreement_level(macd_pred, rsi_pred, cci_pred, octave_dir, kalman_dir)

    # CORRECTION EXPERT: Sortie TOUJOURS possible si MACD change
    # - TOTAL: sortie immédiate
    # - PARTIEL: sortie après 1 confirmation
    # - FORT: sortie après 1 période max (JAMAIS bloquer)
    if agreement == Agreement.TOTAL:
        return True
    elif agreement == Agreement.PARTIEL and ctx.confirmation_count >= 1:
        return True
    elif agreement == Agreement.FORT:
        # Délai max 1 période, puis sortie forcée
        if ctx.exit_delay_count >= 1:
            return True  # Sortie forcée pour protéger le capital
        ctx.exit_delay_count += 1
        return False

    return False


def run_state_machine(
    rsi_pred: np.ndarray,
    cci_pred: np.ndarray,
    macd_pred: np.ndarray,
    rsi_octave: np.ndarray,
    cci_octave: np.ndarray,
    macd_octave: np.ndarray,
    rsi_kalman: np.ndarray,
    cci_kalman: np.ndarray,
    macd_kalman: np.ndarray,
    returns: np.ndarray = None,
    strict: bool = False,
    fees: float = 0.0,
    asset_indices: np.ndarray = None,
    assets: List[str] = None,
    transition_only: bool = False,
    min_stability: int = 2,
    verbose: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Exécute la state machine sur les prédictions.

    Args:
        *_pred: Prédictions du modèle (0 ou 1)
        *_octave: Labels Octave (direction)
        *_kalman: Labels Kalman (direction)
        returns: Rendements (c_ret) pour calcul PnL (optionnel)
        strict: Si True, seul TOTAL autorise les entrées (recommandé)
        fees: Frais par trade (ex: 0.001 = 0.1%)
        asset_indices: Index de l'asset pour chaque sample
        assets: Liste des noms d'assets
        transition_only: Si True, entrer seulement sur CHANGEMENT vers TOTAL
        min_stability: Nombre de périodes de stabilité du signal avant entrée

    Returns:
        positions: Array des positions (0=FLAT, 1=LONG, -1=SHORT)
        stats: Statistiques
    """
    n_samples = len(macd_pred)
    positions = np.zeros(n_samples, dtype=int)
    ctx = Context()

    # Initialiser PnL par asset
    pnl_by_asset = {}
    trades_by_asset = {}
    if assets:
        for asset in assets:
            pnl_by_asset[asset] = []
            trades_by_asset[asset] = 0

    # Statistiques
    stats = {
        'n_trades': 0,
        'n_long': 0,
        'n_short': 0,
        'entries_total': 0,
        'entries_partiel': 0,
        'exits_total': 0,
        'exits_partiel': 0,
        'exits_fort_forced': 0,
        'blocked_by_fort': 0,
        'blocked_by_partiel': 0,  # Pour mode strict
        'blocked_by_no_transition': 0,  # Pour mode transition-only
        'blocked_by_low_stability': 0,  # Pour mode transition-only
        'agreement_counts': {'TOTAL': 0, 'PARTIEL': 0, 'FORT': 0},
        # PnL par état d'entrée
        'pnl_by_entry_state': {'TOTAL': [], 'PARTIEL': []},
        'total_pnl': 0.0,
        'total_pnl_after_fees': 0.0,
        'total_fees': 0.0,
        'pnl_by_asset': pnl_by_asset,
        'trades_by_asset': trades_by_asset,
        'strict_mode': strict,
        'transition_only_mode': transition_only,
        'min_stability': min_stability,
        'fees_rate': fees
    }

    # Variables pour tracker le trade en cours
    current_entry_agreement = None
    current_trade_pnl = 0.0
    current_entry_asset = None

    for i in range(n_samples):
        # Récupérer les signaux
        m_pred = int(macd_pred[i])
        r_pred = int(rsi_pred[i])
        c_pred = int(cci_pred[i])

        # Direction des filtres (basée sur les labels = pente filtrée)
        octave_dir = int(macd_octave[i])  # MACD comme référence principale
        kalman_dir = int(macd_kalman[i])

        # Déterminer l'asset courant
        current_asset = None
        if asset_indices is not None and assets:
            asset_idx = int(asset_indices[i])
            if 0 <= asset_idx < len(assets):
                current_asset = assets[asset_idx]

        # Calculer l'accord
        agreement = get_agreement_level(m_pred, r_pred, c_pred, octave_dir, kalman_dir)
        stats['agreement_counts'][agreement.value] += 1

        # Mettre à jour signal_stability (basé sur direction MACD)
        current_direction = m_pred
        if ctx.prev_direction == -1:
            # Premier sample
            ctx.signal_stability = 1
        elif current_direction == ctx.prev_direction:
            # Direction stable → incrémenter
            ctx.signal_stability += 1
        else:
            # Direction a changé → reset
            ctx.signal_stability = 1

        # Mettre à jour la confirmation
        update_confirmation(m_pred, agreement, ctx)

        # Accumuler le PnL si en position
        if ctx.position != Position.FLAT and returns is not None:
            period_return = returns[i]
            if ctx.position == Position.LONG:
                current_trade_pnl += period_return
            else:  # SHORT
                current_trade_pnl -= period_return

        # Vérifier sortie
        if ctx.position != Position.FLAT:
            if should_exit(m_pred, r_pred, c_pred, octave_dir, kalman_dir, ctx):
                # Calculer les frais (entrée + sortie = 2x fees)
                trade_fees = 2 * fees
                pnl_after_fees = current_trade_pnl - trade_fees

                # Enregistrer le trade
                trade_duration = i - ctx.current_trade_start
                ctx.trades.append({
                    'start': ctx.current_trade_start,
                    'end': i,
                    'duration': trade_duration,
                    'type': ctx.position.value,
                    'entry_agreement': current_entry_agreement,
                    'pnl': current_trade_pnl,
                    'pnl_after_fees': pnl_after_fees,
                    'fees': trade_fees,
                    'asset': current_entry_asset
                })
                stats['n_trades'] += 1
                stats['total_pnl'] += current_trade_pnl
                stats['total_pnl_after_fees'] += pnl_after_fees
                stats['total_fees'] += trade_fees

                # PnL par état d'entrée
                if current_entry_agreement and current_entry_agreement in stats['pnl_by_entry_state']:
                    stats['pnl_by_entry_state'][current_entry_agreement].append(current_trade_pnl)

                # PnL par asset
                if current_entry_asset and current_entry_asset in pnl_by_asset:
                    pnl_by_asset[current_entry_asset].append(pnl_after_fees)
                    trades_by_asset[current_entry_asset] += 1

                # Stats par type de sortie
                if agreement == Agreement.TOTAL:
                    stats['exits_total'] += 1
                elif agreement == Agreement.PARTIEL:
                    stats['exits_partiel'] += 1
                else:
                    stats['exits_fort_forced'] += 1

                # Reset
                ctx.position = Position.FLAT
                ctx.confirmation_count = 0
                ctx.exit_delay_count = 0
                current_trade_pnl = 0.0
                current_entry_agreement = None
                current_entry_asset = None

        # Vérifier entrée
        if ctx.position == Position.FLAT:
            new_position = should_enter(
                m_pred, r_pred, c_pred, octave_dir, kalman_dir, ctx, i,
                strict=strict, transition_only=transition_only, min_stability=min_stability
            )
            if new_position:
                ctx.position = new_position
                ctx.current_trade_start = i
                ctx.last_transition = i
                ctx.confirmation_count = 0
                current_entry_asset = current_asset  # Stocker l'asset d'entrée

                if new_position == Position.LONG:
                    stats['n_long'] += 1
                else:
                    stats['n_short'] += 1

                # Stats par type d'entrée et enregistrer l'état d'entrée
                if agreement == Agreement.TOTAL:
                    stats['entries_total'] += 1
                    current_entry_agreement = 'TOTAL'
                else:
                    stats['entries_partiel'] += 1
                    current_entry_agreement = 'PARTIEL'
            else:
                # Entrée refusée - identifier la raison
                if transition_only:
                    # Mode transition-only
                    is_transition = (ctx.prev_agreement != Agreement.TOTAL and agreement == Agreement.TOTAL)
                    if agreement == Agreement.TOTAL and not is_transition:
                        stats['blocked_by_no_transition'] += 1
                    elif agreement == Agreement.TOTAL and ctx.signal_stability < min_stability:
                        stats['blocked_by_low_stability'] += 1
                    elif agreement == Agreement.FORT:
                        stats['blocked_by_fort'] += 1
                    elif agreement == Agreement.PARTIEL:
                        stats['blocked_by_partiel'] += 1
                else:
                    # Mode strict ou normal
                    if agreement == Agreement.FORT:
                        stats['blocked_by_fort'] += 1
                    elif agreement == Agreement.PARTIEL and strict:
                        stats['blocked_by_partiel'] += 1

        # Enregistrer la position
        if ctx.position == Position.LONG:
            positions[i] = 1
        elif ctx.position == Position.SHORT:
            positions[i] = -1
        else:
            positions[i] = 0

        # Mettre à jour prev_agreement et prev_direction pour le prochain cycle
        ctx.prev_agreement = agreement
        ctx.prev_direction = current_direction

    if verbose:
        print("\n" + "="*80)
        print("RÉSULTATS STATE MACHINE")
        print("="*80)
        if transition_only:
            mode_str = f"TRANSITION-ONLY (min_stability={min_stability})"
        elif strict:
            mode_str = "STRICT (TOTAL only)"
        else:
            mode_str = "NORMAL (déprécié)"
        print(f"\n⚙️ Mode: {mode_str}")
        if fees > 0:
            print(f"   Frais: {fees*100:.2f}% par trade (entrée + sortie)")

        # Calculer la période de test
        n_assets = len(assets) if assets else 1
        samples_per_asset = n_samples // n_assets
        minutes = samples_per_asset * 5
        days = minutes / 60 / 24
        months = days / 30

        print(f"\n📊 Statistiques globales:")
        print(f"   Samples: {n_samples:,} ({n_assets} assets × {samples_per_asset:,})")
        print(f"   Période: {days:.0f} jours (~{months:.1f} mois par asset)")
        print(f"   Trades: {stats['n_trades']}")
        print(f"   LONG: {stats['n_long']}, SHORT: {stats['n_short']}")

        print(f"\n🔀 Niveaux d'accord:")
        for level, count in stats['agreement_counts'].items():
            pct = count / n_samples * 100
            print(f"   {level}: {count:,} ({pct:.1f}%)")

        print(f"\n📈 Entrées:")
        print(f"   Via TOTAL: {stats['entries_total']}")
        if not strict and not transition_only:
            print(f"   Via PARTIEL: {stats['entries_partiel']}")
        print(f"   Bloquées par FORT: {stats['blocked_by_fort']}")
        if strict or transition_only:
            print(f"   Bloquées par PARTIEL: {stats['blocked_by_partiel']}")
        if transition_only:
            print(f"   Bloquées (pas de transition): {stats['blocked_by_no_transition']}")
            print(f"   Bloquées (stabilité < {min_stability}): {stats['blocked_by_low_stability']}")

        print(f"\n📉 Sorties:")
        print(f"   Via TOTAL: {stats['exits_total']}")
        print(f"   Via PARTIEL: {stats['exits_partiel']}")
        print(f"   Forcées (FORT): {stats['exits_fort_forced']}")

        if ctx.trades:
            durations = [t['duration'] for t in ctx.trades]
            print(f"\n⏱️ Durée des trades:")
            print(f"   Moyenne: {np.mean(durations):.1f} périodes (~{np.mean(durations)*5:.0f} min)")
            print(f"   Médiane: {np.median(durations):.1f} périodes (~{np.median(durations)*5:.0f} min)")
            print(f"   Max: {max(durations)} périodes (~{max(durations)*5:.0f} min)")

        # Statistiques PnL
        if returns is not None:
            print(f"\n💰 Performance Globale:")
            print(f"   PnL Brut: {stats['total_pnl']*100:+.2f}%")
            if fees > 0:
                print(f"   Frais totaux: {stats['total_fees']*100:.2f}% ({stats['n_trades']} × {fees*200:.2f}%)")
                print(f"   PnL Net: {stats['total_pnl_after_fees']*100:+.2f}%")
                print(f"   Par mois: {stats['total_pnl_after_fees']/months*100:+.1f}%")
            else:
                print(f"   Par mois: {stats['total_pnl']/months*100:+.1f}%")

            # Stats par état d'entrée
            for state in ['TOTAL', 'PARTIEL']:
                pnls = stats['pnl_by_entry_state'][state]
                if pnls:
                    total = sum(pnls)
                    avg = np.mean(pnls)
                    n_win = sum(1 for p in pnls if p > 0)
                    win_rate = n_win / len(pnls) * 100
                    print(f"   {state}: {total*100:+.2f}% ({len(pnls)} trades, WR={win_rate:.1f}%)")

        # Résultats par asset
        if assets and any(pnl_by_asset.values()):
            print(f"\n📈 Performance par Asset:")
            for asset in assets:
                pnls = pnl_by_asset.get(asset, [])
                n_trades = trades_by_asset.get(asset, 0)
                if pnls:
                    total_pnl = sum(pnls)
                    avg_pnl = np.mean(pnls)
                    n_win = sum(1 for p in pnls if p > 0)
                    win_rate = n_win / len(pnls) * 100 if pnls else 0
                    print(f"   {asset}: {total_pnl*100:+.2f}% ({n_trades} trades, WR={win_rate:.1f}%, avg={avg_pnl*100:+.4f}%)")
                else:
                    print(f"   {asset}: pas de trades")

    return positions, stats


def main():
    parser = argparse.ArgumentParser(
        description="State Machine pour le trading basée sur les prédictions ML",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Datasets Octave20
    parser.add_argument('--rsi-octave', type=str, required=True,
                        help='Dataset RSI Octave20 (.npz)')
    parser.add_argument('--cci-octave', type=str, required=True,
                        help='Dataset CCI Octave20 (.npz)')
    parser.add_argument('--macd-octave', type=str, required=True,
                        help='Dataset MACD Octave20 (.npz)')

    # Datasets Kalman
    parser.add_argument('--rsi-kalman', type=str, required=True,
                        help='Dataset RSI Kalman (.npz)')
    parser.add_argument('--cci-kalman', type=str, required=True,
                        help='Dataset CCI Kalman (.npz)')
    parser.add_argument('--macd-kalman', type=str, required=True,
                        help='Dataset MACD Kalman (.npz)')

    # Options
    parser.add_argument('--split', '-s', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Split à utiliser')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Fichier de sortie pour les positions (.npy)')
    parser.add_argument('--strict', action='store_true',
                        help='Mode strict: seul TOTAL autorise les entrées (recommandé)')
    parser.add_argument('--fees', '-f', type=float, default=0.0,
                        help='Frais par trade en %% (ex: 0.1 = 0.1%% par trade)')
    parser.add_argument('--transition-only', action='store_true',
                        help='Mode Transition-Only: entrer seulement sur CHANGEMENT vers TOTAL')
    parser.add_argument('--min-stability', type=int, default=2,
                        help='Nombre de périodes de stabilité du signal avant entrée')

    args = parser.parse_args()

    # Convertir les frais en décimal
    fees = args.fees / 100 if args.fees > 0 else 0.0

    print("="*80)
    print("STATE MACHINE - Trading basé sur ML")
    print("="*80)

    # Charger tous les datasets
    print(f"\n📂 Chargement des datasets ({args.split})...")

    datasets = {}

    # Octave20
    print("   Loading RSI Octave20...")
    datasets['rsi_octave'] = load_dataset(args.rsi_octave, args.split)
    print("   Loading CCI Octave20...")
    datasets['cci_octave'] = load_dataset(args.cci_octave, args.split)
    print("   Loading MACD Octave20...")
    datasets['macd_octave'] = load_dataset(args.macd_octave, args.split)

    # Kalman
    print("   Loading RSI Kalman...")
    datasets['rsi_kalman'] = load_dataset(args.rsi_kalman, args.split)
    print("   Loading CCI Kalman...")
    datasets['cci_kalman'] = load_dataset(args.cci_kalman, args.split)
    print("   Loading MACD Kalman...")
    datasets['macd_kalman'] = load_dataset(args.macd_kalman, args.split)

    # Valider la synchronisation
    if not validate_indices_sync(datasets):
        print("\n❌ ERREUR CRITIQUE: Les datasets ne sont pas synchronisés!")
        print("   Vérifiez que tous les datasets ont été préparés avec les mêmes paramètres.")
        return

    # Vérifier que les prédictions existent
    for name, data in datasets.items():
        if 'octave' in name and data['Y_pred'] is None:
            print(f"\n❌ ERREUR: {name} n'a pas de prédictions!")
            print(f"   Exécutez d'abord train.py pour générer les prédictions.")
            return

    # Extraire les données
    rsi_pred = datasets['rsi_octave']['Y_pred'].flatten()
    cci_pred = datasets['cci_octave']['Y_pred'].flatten()
    macd_pred = datasets['macd_octave']['Y_pred'].flatten()

    rsi_octave = datasets['rsi_octave']['Y'].flatten()
    cci_octave = datasets['cci_octave']['Y'].flatten()
    macd_octave = datasets['macd_octave']['Y'].flatten()

    rsi_kalman = datasets['rsi_kalman']['Y'].flatten()
    cci_kalman = datasets['cci_kalman']['Y'].flatten()
    macd_kalman = datasets['macd_kalman']['Y'].flatten()

    # Extraire les returns (c_ret = index 3) du dernier timestep
    # Features OHLC: [O_ret, H_ret, L_ret, C_ret, Range_ret]
    X = datasets['macd_octave']['X']  # Shape: (n_samples, seq_len, 5)
    returns = X[:, -1, 3]  # c_ret du dernier timestep
    n_samples = len(macd_pred)

    # Extraire les assets et créer les indices
    assets = datasets['macd_octave'].get('assets', None)
    samples_per_asset = datasets['macd_octave'].get('samples_per_asset', None)

    asset_indices = None
    if assets and samples_per_asset:
        # Créer un array d'indices d'assets
        asset_indices = np.zeros(n_samples, dtype=int)
        offset = 0
        for i, (asset, count) in enumerate(zip(assets, samples_per_asset)):
            asset_indices[offset:offset + count] = i
            offset += count
        print(f"\n📊 Assets détectés: {', '.join(assets)}")
    elif assets:
        # Estimer samples_per_asset si non disponible
        n_assets = len(assets)
        samples_per_asset_est = n_samples // n_assets
        asset_indices = np.zeros(n_samples, dtype=int)
        for i in range(n_assets):
            start = i * samples_per_asset_est
            end = (i + 1) * samples_per_asset_est if i < n_assets - 1 else n_samples
            asset_indices[start:end] = i
        print(f"\n📊 Assets détectés: {', '.join(assets)} (estimation: {samples_per_asset_est:,} samples chacun)")

    print(f"\n📊 Données chargées:")
    print(f"   Samples: {n_samples:,}")
    print(f"   RSI pred mean: {rsi_pred.mean():.3f}")
    print(f"   CCI pred mean: {cci_pred.mean():.3f}")
    print(f"   MACD pred mean: {macd_pred.mean():.3f}")
    print(f"   Returns mean: {returns.mean()*100:.4f}%, std: {returns.std()*100:.4f}%")
    if fees > 0:
        print(f"   Frais: {fees*100:.2f}% par trade")

    # Exécuter la state machine
    positions, stats = run_state_machine(
        rsi_pred, cci_pred, macd_pred,
        rsi_octave, cci_octave, macd_octave,
        rsi_kalman, cci_kalman, macd_kalman,
        returns=returns,
        strict=args.strict,
        fees=fees,
        asset_indices=asset_indices,
        assets=assets,
        transition_only=args.transition_only,
        min_stability=args.min_stability
    )

    # Sauvegarder si demandé
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, positions)
        print(f"\n💾 Positions sauvegardées: {output_path}")

    # Résumé final
    print("\n" + "="*80)
    print("RÉSUMÉ FINAL")
    print("="*80)

    # Calculer métriques finales
    total_pnl = stats['total_pnl']
    total_pnl_net = stats['total_pnl_after_fees']
    total_fees = stats['total_fees']
    n_trades = stats['n_trades']

    # Période en mois
    n_assets = len(assets) if assets else 1
    samples_per_asset = n_samples // n_assets
    months = (samples_per_asset * 5) / 60 / 24 / 30

    print(f"\n💰 Performance:")
    print(f"   PnL Brut: {total_pnl*100:+.2f}%")
    if fees > 0:
        print(f"   Frais ({fees*100:.2f}% × 2 × {n_trades}): -{total_fees*100:.2f}%")
        print(f"   PnL Net: {total_pnl_net*100:+.2f}%")
        print(f"   Par mois (net): {total_pnl_net/months*100:+.1f}%")
    else:
        print(f"   Par mois: {total_pnl/months*100:+.1f}%")

    # Win rate global
    total_pnls = stats['pnl_by_entry_state']['TOTAL']
    if total_pnls:
        n_win = sum(1 for p in total_pnls if p > 0)
        n_loss = sum(1 for p in total_pnls if p < 0)
        win_rate = n_win / len(total_pnls) * 100
        avg_win = np.mean([p for p in total_pnls if p > 0]) if n_win > 0 else 0
        avg_loss = np.mean([p for p in total_pnls if p < 0]) if n_loss > 0 else 0
        profit_factor = (n_win * avg_win) / (n_loss * abs(avg_loss)) if n_loss > 0 and avg_loss != 0 else float('inf')

        print(f"\n📊 Métriques:")
        print(f"   Win Rate: {win_rate:.1f}% ({n_win}W / {n_loss}L)")
        print(f"   Avg Win: {avg_win*100:+.4f}%")
        print(f"   Avg Loss: {avg_loss*100:+.4f}%")
        print(f"   Profit Factor: {profit_factor:.2f}")

    print("\n" + "="*80)
    print("✅ STATE MACHINE TERMINÉE")
    print("="*80)


if __name__ == '__main__':
    main()
