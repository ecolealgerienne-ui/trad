"""
State Machine V2 - Architecture simplifiée validée par CART.

Architecture 3 niveaux:
    NIVEAU 1 — RÉGIME ÉCONOMIQUE (CART validé)
    "Le marché paye-t-il l'action?"
    → Volatilité gate (seuil ~0.13% = P35-40)

    NIVEAU 2 — DIRECTION (ML validé)
    "Dans quel sens?"
    → MACD prob > 0.5 = LONG, sinon SHORT

    NIVEAU 3 — SÉCURITÉ (Optionnel)
    "Garde-fous, pas décideurs"
    → RSI/CCI extrêmes, filtres en désaccord

Découvertes CART:
- Volatilité = 100% de l'importance pour décider SI on agit
- MACD/RSI/CCI = utiles pour DIRECTION, pas pour décider d'agir
- Seuil volatilité ~0.13% sépare HOLD de AGIR

Usage:
    python src/state_machine_v2.py \
        --macd-data data/prepared/dataset_..._macd_octave20.npz \
        --split test --vol-threshold 0.0013 --fees 0.1
"""

import numpy as np
import argparse
from pathlib import Path
from typing import Tuple, List
from enum import Enum
from dataclasses import dataclass, field


class Position(Enum):
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
    entry_volatility: float = 0.0


@dataclass
class Context:
    """État du contexte de trading."""
    position: Position = Position.FLAT
    entry_time: int = 0
    entry_volatility: float = 0.0
    trades: List = field(default_factory=list)
    current_pnl: float = 0.0


def load_dataset(path: str, split: str = 'test') -> dict:
    """Charge un dataset."""
    data = np.load(path, allow_pickle=True)

    result = {
        'X': data[f'X_{split}'],
        'Y': data[f'Y_{split}'],
        'Y_pred': data.get(f'Y_{split}_pred', None),
        'assets': None,
        'samples_per_asset': None
    }

    # Charger les métadonnées
    if 'assets' in data:
        result['assets'] = list(data['assets'])
    if 'samples_per_asset' in data:
        result['samples_per_asset'] = list(data['samples_per_asset'])

    return result


def run_state_machine_v2(
    macd_prob: np.ndarray,
    volatility: np.ndarray,
    returns: np.ndarray,
    vol_threshold: float = 0.0013,
    vol_min: float = 0.0008,
    direction_threshold: float = 0.5,
    fees: float = 0.0,
    asset_indices: np.ndarray = None,
    assets: List[str] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    State Machine V2 - Architecture simplifiée.

    Niveau 1: Volatilité gate
    Niveau 2: Direction MACD
    Niveau 3: (désactivé dans cette version)

    Args:
        macd_prob: Probabilités MACD [0,1]
        volatility: Volatilité (range_ret)
        returns: Rendements (c_ret)
        vol_threshold: Seuil volatilité pour AGIR (défaut: 0.13%)
        vol_min: Seuil min (en dessous = bruit, ignorer)
        direction_threshold: Seuil MACD pour direction
        fees: Frais par trade (ex: 0.001 = 0.1%)

    Returns:
        positions: Array des positions
        stats: Statistiques
    """
    n_samples = len(macd_prob)
    positions = np.zeros(n_samples, dtype=int)
    ctx = Context()

    # Stats
    stats = {
        'n_trades': 0,
        'n_long': 0,
        'n_short': 0,
        'blocked_by_low_vol': 0,
        'blocked_by_noise': 0,
        'total_pnl': 0.0,
        'total_pnl_after_fees': 0.0,
        'total_fees': 0.0,
        'vol_threshold': vol_threshold,
        'vol_min': vol_min,
    }

    # PnL par asset
    pnl_by_asset = {a: [] for a in (assets or [])}
    trades_by_asset = {a: 0 for a in (assets or [])}

    for i in range(n_samples):
        vol = volatility[i]
        prob = macd_prob[i]
        ret = returns[i] if returns is not None else 0.0

        # Asset courant
        current_asset = None
        if asset_indices is not None and assets:
            idx = int(asset_indices[i])
            if 0 <= idx < len(assets):
                current_asset = assets[idx]

        # Accumuler PnL si en position
        if ctx.position != Position.FLAT and returns is not None:
            if ctx.position == Position.LONG:
                ctx.current_pnl += ret
            else:
                ctx.current_pnl -= ret

        # ============================================
        # NIVEAU 1: GATE ÉCONOMIQUE (Volatilité)
        # ============================================

        # Régime "bruit" - volatilité trop basse = ignorer
        if vol < vol_min:
            stats['blocked_by_noise'] += 1
            should_act = False
        # Régime "faible" - volatilité insuffisante
        elif vol < vol_threshold:
            stats['blocked_by_low_vol'] += 1
            should_act = False
        else:
            should_act = True

        # ============================================
        # NIVEAU 2: DIRECTION (MACD)
        # ============================================

        if prob > direction_threshold:
            target_direction = Position.LONG
        else:
            target_direction = Position.SHORT

        # ============================================
        # LOGIQUE DE TRADING
        # ============================================

        # Sortie si on ne devrait pas agir ET on est en position
        if not should_act and ctx.position != Position.FLAT:
            # Sortir de position
            trade_fees = 2 * fees
            pnl_after_fees = ctx.current_pnl - trade_fees

            trade = Trade(
                start=ctx.entry_time,
                end=i,
                duration=i - ctx.entry_time,
                position=ctx.position.value,
                pnl=ctx.current_pnl,
                pnl_after_fees=pnl_after_fees,
                entry_volatility=ctx.entry_volatility
            )
            ctx.trades.append(trade)

            stats['n_trades'] += 1
            stats['total_pnl'] += ctx.current_pnl
            stats['total_pnl_after_fees'] += pnl_after_fees
            stats['total_fees'] += trade_fees

            if current_asset and current_asset in pnl_by_asset:
                pnl_by_asset[current_asset].append(pnl_after_fees)
                trades_by_asset[current_asset] += 1

            ctx.position = Position.FLAT
            ctx.current_pnl = 0.0

        # Entrée ou changement de direction
        elif should_act:
            if ctx.position == Position.FLAT:
                # Nouvelle entrée
                ctx.position = target_direction
                ctx.entry_time = i
                ctx.entry_volatility = vol
                ctx.current_pnl = 0.0

                if target_direction == Position.LONG:
                    stats['n_long'] += 1
                else:
                    stats['n_short'] += 1

            elif ctx.position != target_direction:
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
                    entry_volatility=ctx.entry_volatility
                )
                ctx.trades.append(trade)

                stats['n_trades'] += 1
                stats['total_pnl'] += ctx.current_pnl
                stats['total_pnl_after_fees'] += pnl_after_fees
                stats['total_fees'] += trade_fees

                if current_asset and current_asset in pnl_by_asset:
                    pnl_by_asset[current_asset].append(pnl_after_fees)
                    trades_by_asset[current_asset] += 1

                # Nouvelle entrée dans l'autre sens
                ctx.position = target_direction
                ctx.entry_time = i
                ctx.entry_volatility = vol
                ctx.current_pnl = 0.0

                if target_direction == Position.LONG:
                    stats['n_long'] += 1
                else:
                    stats['n_short'] += 1

        # Enregistrer position
        if ctx.position == Position.LONG:
            positions[i] = 1
        elif ctx.position == Position.SHORT:
            positions[i] = -1
        else:
            positions[i] = 0

    # Fermer position ouverte
    if ctx.position != Position.FLAT:
        trade_fees = 2 * fees
        pnl_after_fees = ctx.current_pnl - trade_fees

        trade = Trade(
            start=ctx.entry_time,
            end=n_samples-1,
            duration=n_samples-1 - ctx.entry_time,
            position=ctx.position.value,
            pnl=ctx.current_pnl,
            pnl_after_fees=pnl_after_fees,
            entry_volatility=ctx.entry_volatility
        )
        ctx.trades.append(trade)

        stats['n_trades'] += 1
        stats['total_pnl'] += ctx.current_pnl
        stats['total_pnl_after_fees'] += pnl_after_fees
        stats['total_fees'] += trade_fees

    # Calculer métriques
    if ctx.trades:
        pnls = [t.pnl for t in ctx.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        stats['win_rate'] = len(wins) / len(pnls) if pnls else 0
        stats['avg_win'] = np.mean(wins) if wins else 0
        stats['avg_loss'] = np.mean(losses) if losses else 0
        stats['profit_factor'] = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float('inf')
        stats['avg_duration'] = np.mean([t.duration for t in ctx.trades])
        stats['trades'] = ctx.trades

    stats['pnl_by_asset'] = pnl_by_asset
    stats['trades_by_asset'] = trades_by_asset

    # Affichage
    if verbose:
        print("\n" + "="*80)
        print("STATE MACHINE V2 - Architecture CART")
        print("="*80)

        print(f"\n⚙️ Configuration:")
        print(f"   Seuil volatilité: {vol_threshold*100:.2f}% (AGIR si >)")
        print(f"   Seuil bruit: {vol_min*100:.2f}% (ignorer si <)")
        print(f"   Seuil direction: {direction_threshold:.2f}")
        if fees > 0:
            print(f"   Frais: {fees*100:.2f}% par trade")

        n_assets = len(assets) if assets else 1
        samples_per_asset = n_samples // n_assets
        days = samples_per_asset * 5 / 60 / 24

        print(f"\n📊 Statistiques:")
        print(f"   Samples: {n_samples:,} ({n_assets} assets × {samples_per_asset:,})")
        print(f"   Période: {days:.0f} jours par asset")
        print(f"   Trades: {stats['n_trades']:,}")
        print(f"   LONG: {stats['n_long']:,}, SHORT: {stats['n_short']:,}")

        print(f"\n🚫 Blocages:")
        print(f"   Par bruit (vol < {vol_min*100:.2f}%): {stats['blocked_by_noise']:,}")
        print(f"   Par faible vol (< {vol_threshold*100:.2f}%): {stats['blocked_by_low_vol']:,}")

        if ctx.trades:
            print(f"\n📈 Performance:")
            print(f"   PnL Brut: {stats['total_pnl']*100:+.2f}%")
            if fees > 0:
                print(f"   Frais: -{stats['total_fees']*100:.2f}%")
                print(f"   PnL Net: {stats['total_pnl_after_fees']*100:+.2f}%")
            print(f"   Win Rate: {stats['win_rate']*100:.1f}%")
            print(f"   Avg Win: {stats['avg_win']*100:+.4f}%")
            print(f"   Avg Loss: {stats['avg_loss']*100:+.4f}%")
            print(f"   Profit Factor: {stats['profit_factor']:.2f}")
            print(f"   Durée moyenne: {stats['avg_duration']:.1f} périodes")

        if assets and any(pnl_by_asset.values()):
            print(f"\n📊 Par Asset:")
            for asset in assets:
                pnls = pnl_by_asset.get(asset, [])
                n_trades = trades_by_asset.get(asset, 0)
                if pnls:
                    total = sum(pnls)
                    wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
                    print(f"   {asset}: {total*100:+.2f}% ({n_trades} trades, WR={wr:.1f}%)")

    return positions, stats


def main():
    parser = argparse.ArgumentParser(
        description="State Machine V2 - Architecture simplifiée CART",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset
    parser.add_argument('--macd-data', type=str, required=True,
                        help='Dataset MACD avec prédictions (.npz)')

    # Options
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--vol-threshold', type=float, default=0.0013,
                        help='Seuil volatilité pour AGIR (0.0013 = 0.13%%)')
    parser.add_argument('--vol-min', type=float, default=0.0008,
                        help='Seuil min (bruit à ignorer)')
    parser.add_argument('--direction-threshold', type=float, default=0.5,
                        help='Seuil MACD pour direction')
    parser.add_argument('--fees', type=float, default=0.0,
                        help='Frais par trade en %% (0.1 = 0.1%%)')

    args = parser.parse_args()

    # Convertir frais
    fees = args.fees / 100 if args.fees > 0 else 0.0

    print("="*80)
    print("STATE MACHINE V2 - CART Validated Architecture")
    print("="*80)

    # Charger dataset
    print(f"\n📂 Chargement {args.macd_data} ({args.split})...")
    data = load_dataset(args.macd_data, args.split)

    if data['Y_pred'] is None:
        print("❌ ERREUR: Pas de prédictions dans le dataset!")
        print("   Exécutez train.py d'abord.")
        return

    # Extraire données
    macd_prob = data['Y_pred'].flatten()
    X = data['X']
    returns = X[:, -1, 3]      # c_ret
    volatility = X[:, -1, 4]   # range_ret

    n_samples = len(macd_prob)
    assets = data.get('assets')
    samples_per_asset = data.get('samples_per_asset')

    # Créer indices assets
    asset_indices = None
    if assets and samples_per_asset:
        asset_indices = np.zeros(n_samples, dtype=int)
        offset = 0
        for i, count in enumerate(samples_per_asset):
            asset_indices[offset:offset+count] = i
            offset += count
    elif assets:
        n_assets = len(assets)
        est = n_samples // n_assets
        asset_indices = np.zeros(n_samples, dtype=int)
        for i in range(n_assets):
            start = i * est
            end = (i+1) * est if i < n_assets-1 else n_samples
            asset_indices[start:end] = i

    print(f"\n📊 Données:")
    print(f"   Samples: {n_samples:,}")
    print(f"   MACD prob: mean={macd_prob.mean():.3f}, std={macd_prob.std():.3f}")
    print(f"   Volatilité: mean={volatility.mean()*100:.4f}%, median={np.median(volatility)*100:.4f}%")
    print(f"   P35 volatilité: {np.percentile(volatility, 35)*100:.4f}%")
    print(f"   P50 volatilité: {np.percentile(volatility, 50)*100:.4f}%")

    # Exécuter
    positions, stats = run_state_machine_v2(
        macd_prob=macd_prob,
        volatility=volatility,
        returns=returns,
        vol_threshold=args.vol_threshold,
        vol_min=args.vol_min,
        direction_threshold=args.direction_threshold,
        fees=fees,
        asset_indices=asset_indices,
        assets=assets
    )

    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ")
    print("="*80)

    print(f"\n🎯 Architecture CART validée:")
    print(f"   Niveau 1: Volatilité > {args.vol_threshold*100:.2f}% → AGIR")
    print(f"   Niveau 2: MACD > 0.5 → LONG, sinon SHORT")
    print(f"   Niveau 3: (garde-fous désactivés)")

    if stats['n_trades'] > 0:
        print(f"\n💰 Résultat final:")
        print(f"   Trades: {stats['n_trades']:,}")
        print(f"   PnL Net: {stats['total_pnl_after_fees']*100:+.2f}%")
        print(f"   Win Rate: {stats['win_rate']*100:.1f}%")
        print(f"   Profit Factor: {stats['profit_factor']:.2f}")


if __name__ == '__main__':
    main()
