"""
State Machine avec filtre Cross-Asset.

Principe:
- Calcule un "market_score" = somme des prédictions MACD sur les 5 cryptos
- N'entre en position que si le marché confirme (score >= seuil)

Usage:
    python src/state_machine_cross_asset.py \
        --rsi-octave data/prepared/dataset_..._rsi_octave20.npz \
        --cci-octave data/prepared/dataset_..._cci_octave20.npz \
        --macd-octave data/prepared/dataset_..._macd_octave20.npz \
        --rsi-kalman data/prepared/dataset_..._rsi_kalman.npz \
        --cci-kalman data/prepared/dataset_..._cci_kalman.npz \
        --macd-kalman data/prepared/dataset_..._macd_kalman.npz \
        --split test --strict --fees 0.1 --market-threshold 4
"""

import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum


class Position(Enum):
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


class Agreement(Enum):
    TOTAL = "TOTAL"
    PARTIEL = "PARTIEL"
    FORT = "FORT"


@dataclass
class Context:
    """État du contexte de trading."""
    position: Position = Position.FLAT
    entry_time: int = 0
    current_trade_start: int = 0
    last_transition: int = 0
    confirmation_count: int = 0
    exit_delay_count: int = 0
    prev_macd: int = -1
    trades: List = field(default_factory=list)


def load_dataset(path: str, split: str = 'test') -> dict:
    """Charge un dataset."""
    data = np.load(path, allow_pickle=True)

    result = {
        'X': data[f'X_{split}'],
        'Y': data[f'Y_{split}'],
        'Y_pred': data.get(f'Y_{split}_pred', None),
        'metadata': None,
        'assets': None,
        'samples_per_asset': None
    }

    pred_key = f'Y_{split}_pred'
    if pred_key in data:
        result['Y_pred'] = data[pred_key]

    if 'metadata' in data:
        try:
            meta = json.loads(str(data['metadata']))
            result['metadata'] = meta
            if 'assets' in meta:
                result['assets'] = meta['assets']
            if f'samples_per_asset_{split}' in meta:
                result['samples_per_asset'] = meta[f'samples_per_asset_{split}']
        except:
            pass

    return result


def restructure_to_parallel(
    predictions: np.ndarray,
    labels: np.ndarray,
    returns: np.ndarray,
    assets: List[str],
    n_samples: int
) -> pd.DataFrame:
    """
    Restructure les données séquentielles en données parallèles.

    Input:
        [BTC_0, BTC_1, ..., BTC_n, ETH_0, ETH_1, ..., ETH_n, ...]

    Output:
        DataFrame avec colonnes: [BTC_pred, ETH_pred, ..., BTC_ret, ETH_ret, ...]
        Index: timestamps alignés
    """
    n_assets = len(assets)
    samples_per_asset = n_samples // n_assets

    print(f"\n🔄 Restructuration des données:")
    print(f"   {n_assets} assets × {samples_per_asset:,} samples = {n_samples:,} total")

    # Créer le DataFrame
    data = {}

    for i, asset in enumerate(assets):
        start = i * samples_per_asset
        end = (i + 1) * samples_per_asset

        data[f'{asset}_pred'] = predictions[start:end]
        data[f'{asset}_label'] = labels[start:end]
        data[f'{asset}_ret'] = returns[start:end]

    df = pd.DataFrame(data)

    # Calculer le market_score (somme des prédictions MACD)
    pred_cols = [f'{asset}_pred' for asset in assets]
    df['market_score'] = df[pred_cols].sum(axis=1)

    # Calculer la direction majoritaire
    df['market_bullish'] = (df['market_score'] >= 3).astype(int)  # 3+ sur 5
    df['market_bearish'] = (df['market_score'] <= 2).astype(int)  # 2- sur 5

    print(f"   Market Score distribution:")
    for score in range(n_assets + 1):
        count = (df['market_score'] == score).sum()
        pct = count / len(df) * 100
        print(f"      Score {score}: {count:,} ({pct:.1f}%)")

    return df


def get_agreement_level(
    macd_pred: int,
    rsi_pred: int,
    cci_pred: int,
    octave_dir: int,
    kalman_dir: int
) -> Agreement:
    """Retourne le niveau d'accord des signaux."""
    indicators_agree = (macd_pred == rsi_pred == cci_pred)
    filters_agree = (octave_dir == kalman_dir)

    if indicators_agree and filters_agree:
        return Agreement.TOTAL
    elif not indicators_agree and not filters_agree:
        return Agreement.FORT
    else:
        return Agreement.PARTIEL


def run_state_machine_cross_asset(
    df: pd.DataFrame,
    assets: List[str],
    rsi_preds: Dict[str, np.ndarray],
    cci_preds: Dict[str, np.ndarray],
    octave_labels: Dict[str, np.ndarray],
    kalman_labels: Dict[str, np.ndarray],
    market_threshold_long: int = 4,
    market_threshold_short: int = 1,
    strict: bool = True,
    fees: float = 0.0,
    verbose: bool = True
) -> Tuple[Dict[str, np.ndarray], dict]:
    """
    Exécute la state machine avec filtre cross-asset.

    Args:
        df: DataFrame avec market_score pré-calculé
        assets: Liste des assets
        market_threshold_long: Score minimum pour LONG (ex: 4 = 4+ cryptos bullish)
        market_threshold_short: Score maximum pour SHORT (ex: 1 = 1- cryptos bullish)

    Returns:
        positions: Dict[asset] -> array des positions
        stats: Statistiques globales
    """
    n_samples = len(df)

    # Statistiques globales
    stats = {
        'n_trades': 0,
        'n_long': 0,
        'n_short': 0,
        'trades_by_asset': {asset: 0 for asset in assets},
        'pnl_by_asset': {asset: [] for asset in assets},
        'total_pnl': 0.0,
        'total_pnl_after_fees': 0.0,
        'total_fees': 0.0,
        'blocked_by_market': 0,
        'blocked_by_agreement': 0,
    }

    positions = {asset: np.zeros(n_samples, dtype=int) for asset in assets}
    contexts = {asset: Context() for asset in assets}

    # Variables pour tracker les trades en cours
    current_trade_pnl = {asset: 0.0 for asset in assets}

    for t in range(n_samples):
        market_score = df['market_score'].iloc[t]

        for asset in assets:
            ctx = contexts[asset]

            # Récupérer les signaux pour cet asset
            macd_pred = int(df[f'{asset}_pred'].iloc[t])
            rsi_pred = int(rsi_preds[asset][t])
            cci_pred = int(cci_preds[asset][t])
            octave_dir = int(octave_labels[asset][t])
            kalman_dir = int(kalman_labels[asset][t])
            ret = df[f'{asset}_ret'].iloc[t]

            # Calculer l'accord
            agreement = get_agreement_level(macd_pred, rsi_pred, cci_pred, octave_dir, kalman_dir)

            # Accumuler le PnL si en position
            if ctx.position != Position.FLAT:
                if ctx.position == Position.LONG:
                    current_trade_pnl[asset] += ret
                else:
                    current_trade_pnl[asset] -= ret

            # Vérifier sortie
            if ctx.position != Position.FLAT:
                should_exit = False

                # Sortie si signal inverse
                if ctx.position == Position.LONG and macd_pred == 0:
                    should_exit = True
                elif ctx.position == Position.SHORT and macd_pred == 1:
                    should_exit = True

                # Sortie si marché inverse (cross-asset)
                if ctx.position == Position.LONG and market_score <= market_threshold_short:
                    should_exit = True
                elif ctx.position == Position.SHORT and market_score >= market_threshold_long:
                    should_exit = True

                if should_exit:
                    # Calculer les frais
                    trade_fees = 2 * fees
                    pnl_after_fees = current_trade_pnl[asset] - trade_fees

                    # Enregistrer
                    trade_duration = t - ctx.current_trade_start
                    ctx.trades.append({
                        'start': ctx.current_trade_start,
                        'end': t,
                        'duration': trade_duration,
                        'type': ctx.position.value,
                        'pnl': current_trade_pnl[asset],
                        'pnl_after_fees': pnl_after_fees,
                    })

                    stats['n_trades'] += 1
                    stats['trades_by_asset'][asset] += 1
                    stats['total_pnl'] += current_trade_pnl[asset]
                    stats['total_pnl_after_fees'] += pnl_after_fees
                    stats['total_fees'] += trade_fees
                    stats['pnl_by_asset'][asset].append(pnl_after_fees)

                    # Reset
                    ctx.position = Position.FLAT
                    current_trade_pnl[asset] = 0.0

            # Vérifier entrée
            if ctx.position == Position.FLAT:
                can_enter = False
                direction = None

                # Condition 1: Accord TOTAL (mode strict)
                if strict and agreement != Agreement.TOTAL:
                    stats['blocked_by_agreement'] += 1
                elif not strict or agreement == Agreement.TOTAL:

                    # Condition 2: Filtre cross-asset
                    if macd_pred == 1 and market_score >= market_threshold_long:
                        can_enter = True
                        direction = Position.LONG
                    elif macd_pred == 0 and market_score <= market_threshold_short:
                        can_enter = True
                        direction = Position.SHORT
                    else:
                        stats['blocked_by_market'] += 1

                if can_enter and direction:
                    ctx.position = direction
                    ctx.current_trade_start = t

                    if direction == Position.LONG:
                        stats['n_long'] += 1
                    else:
                        stats['n_short'] += 1

            # Enregistrer la position
            if ctx.position == Position.LONG:
                positions[asset][t] = 1
            elif ctx.position == Position.SHORT:
                positions[asset][t] = -1

    if verbose:
        print("\n" + "="*80)
        print("RÉSULTATS STATE MACHINE CROSS-ASSET")
        print("="*80)

        print(f"\n⚙️ Configuration:")
        print(f"   Mode: {'STRICT' if strict else 'NORMAL'}")
        print(f"   Seuil LONG: market_score >= {market_threshold_long}")
        print(f"   Seuil SHORT: market_score <= {market_threshold_short}")
        if fees > 0:
            print(f"   Frais: {fees*100:.2f}% par trade")

        samples_per_asset = n_samples
        days = (samples_per_asset * 5) / 60 / 24
        months = days / 30

        print(f"\n📊 Statistiques globales:")
        print(f"   Période: {days:.0f} jours (~{months:.1f} mois)")
        print(f"   Trades: {stats['n_trades']}")
        print(f"   LONG: {stats['n_long']}, SHORT: {stats['n_short']}")
        print(f"   Bloqués (market): {stats['blocked_by_market']:,}")
        print(f"   Bloqués (agreement): {stats['blocked_by_agreement']:,}")

        print(f"\n💰 Performance Globale:")
        print(f"   PnL Brut: {stats['total_pnl']*100:+.2f}%")
        if fees > 0:
            print(f"   Frais totaux: {stats['total_fees']*100:.2f}%")
            print(f"   PnL Net: {stats['total_pnl_after_fees']*100:+.2f}%")
            print(f"   Par mois: {stats['total_pnl_after_fees']/months*100:+.1f}%")

        print(f"\n📈 Performance par Asset:")
        for asset in assets:
            pnls = stats['pnl_by_asset'][asset]
            n_trades = stats['trades_by_asset'][asset]
            if pnls:
                total_pnl = sum(pnls)
                n_win = sum(1 for p in pnls if p > 0)
                win_rate = n_win / len(pnls) * 100
                print(f"   {asset}: {total_pnl*100:+.2f}% ({n_trades} trades, WR={win_rate:.1f}%)")
            else:
                print(f"   {asset}: pas de trades")

        # Métriques globales
        all_pnls = []
        for asset in assets:
            all_pnls.extend(stats['pnl_by_asset'][asset])

        if all_pnls:
            n_win = sum(1 for p in all_pnls if p > 0)
            n_loss = sum(1 for p in all_pnls if p < 0)
            win_rate = n_win / len(all_pnls) * 100 if all_pnls else 0
            avg_win = np.mean([p for p in all_pnls if p > 0]) if n_win > 0 else 0
            avg_loss = np.mean([p for p in all_pnls if p < 0]) if n_loss > 0 else 0
            pf = (n_win * avg_win) / (n_loss * abs(avg_loss)) if n_loss > 0 and avg_loss != 0 else float('inf')

            print(f"\n📊 Métriques:")
            print(f"   Win Rate: {win_rate:.1f}% ({n_win}W / {n_loss}L)")
            print(f"   Avg Win: {avg_win*100:+.4f}%")
            print(f"   Avg Loss: {avg_loss*100:+.4f}%")
            print(f"   Profit Factor: {pf:.2f}")

    return positions, stats


def main():
    parser = argparse.ArgumentParser(
        description="State Machine avec filtre Cross-Asset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Datasets
    parser.add_argument('--rsi-octave', type=str, required=True)
    parser.add_argument('--cci-octave', type=str, required=True)
    parser.add_argument('--macd-octave', type=str, required=True)
    parser.add_argument('--rsi-kalman', type=str, required=True)
    parser.add_argument('--cci-kalman', type=str, required=True)
    parser.add_argument('--macd-kalman', type=str, required=True)

    # Options
    parser.add_argument('--split', '-s', type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--strict', action='store_true',
                        help='Mode strict: seul TOTAL autorise les entrées')
    parser.add_argument('--fees', '-f', type=float, default=0.0,
                        help='Frais par trade en %% (ex: 0.1)')
    parser.add_argument('--market-threshold-long', type=int, default=4,
                        help='Score minimum pour LONG (0-5)')
    parser.add_argument('--market-threshold-short', type=int, default=1,
                        help='Score maximum pour SHORT (0-5)')

    args = parser.parse_args()
    fees = args.fees / 100 if args.fees > 0 else 0.0

    print("="*80)
    print("STATE MACHINE CROSS-ASSET")
    print("="*80)

    # Charger les datasets
    print(f"\n📂 Chargement des datasets ({args.split})...")

    datasets = {
        'rsi_octave': load_dataset(args.rsi_octave, args.split),
        'cci_octave': load_dataset(args.cci_octave, args.split),
        'macd_octave': load_dataset(args.macd_octave, args.split),
        'rsi_kalman': load_dataset(args.rsi_kalman, args.split),
        'cci_kalman': load_dataset(args.cci_kalman, args.split),
        'macd_kalman': load_dataset(args.macd_kalman, args.split),
    }

    # Extraire les assets
    assets = datasets['macd_octave'].get('assets', ['BTC', 'ETH', 'BNB', 'ADA', 'LTC'])
    n_samples = len(datasets['macd_octave']['Y_pred'].flatten())
    n_assets = len(assets)
    samples_per_asset = n_samples // n_assets

    print(f"   Assets: {', '.join(assets)}")
    print(f"   Samples: {n_samples:,} ({n_assets} × {samples_per_asset:,})")

    # Extraire les données
    macd_pred = datasets['macd_octave']['Y_pred'].flatten()
    X = datasets['macd_octave']['X']
    returns = X[:, -1, 3]  # c_ret
    macd_labels = datasets['macd_octave']['Y'].flatten()

    # Restructurer en parallèle
    df = restructure_to_parallel(
        predictions=macd_pred,
        labels=macd_labels,
        returns=returns,
        assets=assets,
        n_samples=n_samples
    )

    # Préparer les prédictions par asset
    rsi_pred_all = datasets['rsi_octave']['Y_pred'].flatten()
    cci_pred_all = datasets['cci_octave']['Y_pred'].flatten()

    rsi_octave_all = datasets['rsi_octave']['Y'].flatten()
    cci_octave_all = datasets['cci_octave']['Y'].flatten()
    macd_octave_all = datasets['macd_octave']['Y'].flatten()

    rsi_kalman_all = datasets['rsi_kalman']['Y'].flatten()
    cci_kalman_all = datasets['cci_kalman']['Y'].flatten()
    macd_kalman_all = datasets['macd_kalman']['Y'].flatten()

    # Restructurer par asset
    rsi_preds = {}
    cci_preds = {}
    octave_labels = {}
    kalman_labels = {}

    for i, asset in enumerate(assets):
        start = i * samples_per_asset
        end = (i + 1) * samples_per_asset

        rsi_preds[asset] = rsi_pred_all[start:end]
        cci_preds[asset] = cci_pred_all[start:end]

        # Pour octave/kalman, on utilise MACD comme référence
        octave_labels[asset] = macd_octave_all[start:end]
        kalman_labels[asset] = macd_kalman_all[start:end]

    # Exécuter la state machine
    positions, stats = run_state_machine_cross_asset(
        df=df,
        assets=assets,
        rsi_preds=rsi_preds,
        cci_preds=cci_preds,
        octave_labels=octave_labels,
        kalman_labels=kalman_labels,
        market_threshold_long=args.market_threshold_long,
        market_threshold_short=args.market_threshold_short,
        strict=args.strict,
        fees=fees,
    )

    print("\n" + "="*80)
    print("✅ STATE MACHINE CROSS-ASSET TERMINÉE")
    print("="*80)


if __name__ == '__main__':
    main()
