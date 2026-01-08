#!/usr/bin/env python3
"""
Test ATR ML-Aware Filter (Entry-Focused Strategy)

Principe:
- ATR pondéré par désaccord Kalman/Octave (difficulté de prédiction ML)
- Filtrer les zones où le ML est incertain (désaccords)
- Objectif: Réduire trades 30k → 15k, améliorer Win Rate

Différence vs Structural ATR:
- Structural ATR: filtre par volatilité brute (Q20-Q80)
- ML-Aware ATR: pondère ATR par difficulté ML (désaccords filtres)

Formule:
    TR = True Range standard
    difficulty = (Kalman_dir != Octave_dir) + prolonged_disagreement
    w = 1 + lambda * difficulty
    ATR_ML = EMA(TR * w, n)

Stratégie Entry-Focused:
- ML responsable: Qualité des ENTRÉES
- Portfolio Management: Gestion des SORTIES

Usage:
    python tests/test_atr_ml_aware_filter.py --indicator macd --split test --n 6 --lambda-w 1.0 --atr-q-low 25 --atr-q-high 70
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass


# =============================================================================
# RÉSULTATS BACKTEST
# =============================================================================

@dataclass
class BacktestResult:
    """Résultats d'un backtest."""
    n_trades: int
    win_rate: float
    pnl_gross: float
    pnl_net: float
    avg_duration: float
    avg_confidence: float
    n_long: int
    n_short: int
    pnl_long: float
    pnl_short: float
    n_direction_flip: int
    n_time_exit: int
    n_stop_loss: int
    atr_coverage: float  # % samples où ATR dans range
    avg_atr: float
    n_atr_window: int
    lambda_w: float


# =============================================================================
# CALCUL ATR ML-AWARE
# =============================================================================

def compute_ml_aware_atr(
    df_ohlcv: pd.DataFrame,
    kalman_dir: np.ndarray,
    octave_dir: np.ndarray,
    n: int = 6,
    lambda_w: float = 1.0
) -> np.ndarray:
    """
    Calcule ATR ML-Aware pondéré par désaccord Kalman/Octave.

    Args:
        df_ohlcv: DataFrame OHLCV (high, low, close)
        kalman_dir: Direction Kalman (0/1)
        octave_dir: Direction Octave (0/1)
        n: Fenêtre EMA pour ATR (5-8 périodes = 25-40min)
        lambda_w: Poids du désaccord (0.5-1.5)

    Returns:
        ATR ML-Aware (même longueur que df_ohlcv)

    Formule:
        TR = max(H-L, abs(H-C_prev), abs(L-C_prev))
        difficulty = (K != O) + prolonged_disagreement
        w = 1 + lambda_w * difficulty
        ATR_ML = EMA(TR * w, n)
    """
    # True Range standard
    high = df_ohlcv['high'].values
    low = df_ohlcv['low'].values
    close = df_ohlcv['close'].values
    close_prev = np.roll(close, 1)
    close_prev[0] = close[0]  # Éviter NaN

    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - close_prev),
            np.abs(low - close_prev)
        )
    )

    # Difficulty = désaccord Kalman/Octave
    difficulty = (kalman_dir != octave_dir).astype(float)

    # Désaccord prolongé (blocs ≥2 périodes consécutives)
    disagreement_series = pd.Series(difficulty)
    prolonged = (disagreement_series.rolling(2).sum() == 2).astype(float).values
    difficulty += prolonged

    # Poids ML-aware
    w = 1.0 + lambda_w * difficulty

    # ATR pondéré (EMA)
    tr_weighted = tr * w
    atr_ml = pd.Series(tr_weighted).ewm(span=n, adjust=False).mean().values

    return atr_ml


# =============================================================================
# BACKTEST ATR ML-AWARE
# =============================================================================

def backtest_atr_ml_aware(
    predictions: np.ndarray,
    returns: np.ndarray,
    atr_ml: np.ndarray,
    fees: float,
    atr_q_low: float,
    atr_q_high: float,
    exit_mode: str = 'time',
    exit_time: int = 20,
    exit_stop_loss: float = 0.02,
    n_atr_window: int = 6,
    lambda_w: float = 1.0
) -> BacktestResult:
    """
    Backtest avec filtre ATR ML-Aware sur les ENTRÉES.

    Args:
        predictions: Probabilités Direction ML (0-1)
        returns: Returns réels (%)
        atr_ml: ATR ML-Aware
        fees: Frais par trade (0.001 = 0.1%)
        atr_q_low: Percentile bas ATR (ex: 25)
        atr_q_high: Percentile haut ATR (ex: 70)
        exit_mode: 'time' ou 'stop_loss'
        exit_time: Durée max trade si mode time (périodes)
        exit_stop_loss: Stop loss si mode stop_loss (%)
        n_atr_window: Fenêtre ATR (pour info)
        lambda_w: Lambda (pour info)

    Returns:
        BacktestResult avec métriques complètes

    Stratégie:
        - ENTRÉE: Direction ML + ATR dans [Q_low, Q_high]
        - SORTIE: TIME (20p) ou STOP_LOSS (-2%) ou DIRECTION_FLIP
    """
    n_samples = len(predictions)

    # Masque ATR (filtre les entrées)
    if atr_q_low == 0.0 and atr_q_high == 100.0:
        atr_mask = np.ones(n_samples, dtype=bool)
    else:
        q_low_val = np.percentile(atr_ml, atr_q_low)
        q_high_val = np.percentile(atr_ml, atr_q_high)
        atr_mask = (atr_ml >= q_low_val) & (atr_ml <= q_high_val)

    atr_coverage = atr_mask.mean() * 100
    avg_atr = atr_ml.mean()

    # Direction prédite
    direction = (predictions > 0.5).astype(int)

    # Confiance
    confidence = np.abs(predictions - 0.5) * 2.0

    # Tracking
    position = 0  # 0=FLAT, 1=LONG, -1=SHORT
    entry_price_idx = -1
    entry_time = -1
    current_pnl = 0.0

    trades = []
    confidences = []
    durations = []
    exit_reasons = {'DIRECTION_FLIP': 0, 'TIME': 0, 'STOP_LOSS': 0}

    for i in range(n_samples):
        # === GESTION SORTIE ===
        if position != 0:
            trade_duration = i - entry_time

            # Sortie TIME
            if exit_mode == 'time' and trade_duration >= exit_time:
                exit_return = returns[i]
                final_pnl = (current_pnl + exit_return) * position - fees

                trades.append({
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'pnl': final_pnl,
                    'duration': trade_duration,
                    'exit_reason': 'TIME'
                })
                confidences.append(confidence[entry_price_idx])
                durations.append(trade_duration)
                exit_reasons['TIME'] += 1

                position = 0
                current_pnl = 0.0
                continue

            # Sortie STOP_LOSS
            if exit_mode == 'stop_loss':
                drawdown = (current_pnl + returns[i]) * position
                if drawdown <= -exit_stop_loss:
                    final_pnl = drawdown - fees

                    trades.append({
                        'direction': 'LONG' if position == 1 else 'SHORT',
                        'pnl': final_pnl,
                        'duration': trade_duration,
                        'exit_reason': 'STOP_LOSS'
                    })
                    confidences.append(confidence[entry_price_idx])
                    durations.append(trade_duration)
                    exit_reasons['STOP_LOSS'] += 1

                    position = 0
                    current_pnl = 0.0
                    continue

            # Sortie DIRECTION_FLIP
            current_dir = direction[i]
            target_position = 1 if current_dir == 1 else -1

            if target_position != position:
                exit_return = returns[i]
                final_pnl = (current_pnl + exit_return) * position - fees

                trades.append({
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'pnl': final_pnl,
                    'duration': trade_duration,
                    'exit_reason': 'DIRECTION_FLIP'
                })
                confidences.append(confidence[entry_price_idx])
                durations.append(trade_duration)
                exit_reasons['DIRECTION_FLIP'] += 1

                # Flip immédiat (réutilise logique test_holding_strategy.py)
                position = target_position
                entry_price_idx = i
                entry_time = i
                current_pnl = 0.0
                continue

            # Continuer trade
            current_pnl += returns[i]

        # === ENTRÉE (uniquement si FLAT + ATR OK) ===
        if position == 0 and atr_mask[i]:
            current_dir = direction[i]
            position = 1 if current_dir == 1 else -1
            entry_price_idx = i
            entry_time = i
            current_pnl = 0.0

    # Clôture position finale
    if position != 0:
        final_pnl = current_pnl * position - fees
        trades.append({
            'direction': 'LONG' if position == 1 else 'SHORT',
            'pnl': final_pnl,
            'duration': n_samples - entry_time,
            'exit_reason': 'END'
        })
        confidences.append(confidence[entry_price_idx])
        durations.append(n_samples - entry_time)

    # Métriques
    if len(trades) == 0:
        return BacktestResult(
            n_trades=0, win_rate=0.0, pnl_gross=0.0, pnl_net=0.0,
            avg_duration=0.0, avg_confidence=0.0,
            n_long=0, n_short=0, pnl_long=0.0, pnl_short=0.0,
            n_direction_flip=0, n_time_exit=0, n_stop_loss=0,
            atr_coverage=atr_coverage, avg_atr=avg_atr,
            n_atr_window=n_atr_window, lambda_w=lambda_w
        )

    df_trades = pd.DataFrame(trades)

    n_trades = len(df_trades)
    win_rate = (df_trades['pnl'] > 0).mean() * 100
    pnl_gross = df_trades['pnl'].sum() + (n_trades * fees)
    pnl_net = df_trades['pnl'].sum()
    avg_duration = np.mean(durations)
    avg_confidence = np.mean(confidences)

    longs = df_trades[df_trades['direction'] == 'LONG']
    shorts = df_trades[df_trades['direction'] == 'SHORT']

    n_long = len(longs)
    n_short = len(shorts)
    pnl_long = longs['pnl'].sum() if n_long > 0 else 0.0
    pnl_short = shorts['pnl'].sum() if n_short > 0 else 0.0

    return BacktestResult(
        n_trades=n_trades,
        win_rate=win_rate,
        pnl_gross=pnl_gross,
        pnl_net=pnl_net,
        avg_duration=avg_duration,
        avg_confidence=avg_confidence,
        n_long=n_long,
        n_short=n_short,
        pnl_long=pnl_long,
        pnl_short=pnl_short,
        n_direction_flip=exit_reasons['DIRECTION_FLIP'],
        n_time_exit=exit_reasons['TIME'],
        n_stop_loss=exit_reasons['STOP_LOSS'],
        atr_coverage=atr_coverage,
        avg_atr=avg_atr,
        n_atr_window=n_atr_window,
        lambda_w=lambda_w
    )


# =============================================================================
# CHARGEMENT DONNÉES
# =============================================================================

def load_predictions_and_ohlcv(
    indicator: str,
    filter_type: str,
    split: str = 'test'
) -> Tuple[np.ndarray, np.ndarray, Dict[str, pd.DataFrame]]:
    """
    Charge prédictions Direction et DataFrames OHLCV.

    Args:
        indicator: 'macd', 'rsi', 'cci'
        filter_type: 'kalman' ou 'octave20'
        split: 'train', 'val', 'test'

    Returns:
        (predictions, returns, ohlcv_dfs)
        - predictions: Probabilités Direction (0-1)
        - returns: Returns réels (%)
        - ohlcv_dfs: Dict {asset: DataFrame OHLCV}
    """
    dataset_pattern = f"dataset_btc_eth_bnb_ada_ltc_{indicator}_direction_only_{filter_type}.npz"
    dataset_path = Path("data/prepared") / dataset_pattern

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)

    # Sélection split
    if split == 'train':
        X = data['X_train']
        Y = data['Y_train']
        metadata = data['metadata_train']
    elif split == 'val':
        X = data['X_val']
        Y = data['Y_val']
        metadata = data['metadata_val']
    else:
        X = data['X_test']
        Y = data['Y_test']
        metadata = data['metadata_test']

    # Charger modèle et prédire
    model_name = f"best_model_{indicator}_{filter_type}_direction_only.pth"
    model_path = Path("models") / model_name

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")

    import torch
    from model import MultiOutputCNNLSTM

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_features = X.shape[2]
    model = MultiOutputCNNLSTM(n_features=n_features, n_outputs=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        outputs = model(X_tensor).cpu().numpy().flatten()

    predictions = outputs  # Déjà en [0,1] (sigmoid dans forward)

    # Calculer returns
    returns = []
    for i in range(len(X)):
        seq = X[i]
        last_close_ret = seq[-1, 0]  # c_ret (dernière période)
        returns.append(last_close_ret)
    returns = np.array(returns)

    # Charger OHLCV
    ohlcv_dfs = {}
    assets = ['BTC', 'ETH', 'BNB', 'ADA', 'LTC']
    for asset in assets:
        csv_path = Path(f"data_trad/{asset}USD_all_5m.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            ohlcv_dfs[asset] = df

    return predictions, returns, ohlcv_dfs


def align_predictions_with_ohlcv(
    metadata: np.ndarray,
    ohlcv_dfs: Dict[str, pd.DataFrame],
    n_samples_total: int
) -> pd.DataFrame:
    """
    Aligne les prédictions avec les données OHLCV complètes.

    Simplifié: Concatène tous les assets sans métadonnées par sample.

    Args:
        metadata: Métadonnées samples (pas utilisées ici)
        ohlcv_dfs: Dict {asset: DataFrame OHLCV}
        n_samples_total: Nombre total de samples à extraire

    Returns:
        DataFrame OHLCV global (n_samples_total lignes)
    """
    # Concaténer tous les assets
    all_dfs = []
    for asset in sorted(ohlcv_dfs.keys()):
        df = ohlcv_dfs[asset].copy()
        df = df[['timestamp', 'high', 'low', 'close']].copy()
        all_dfs.append(df)

    df_global = pd.concat(all_dfs, ignore_index=True)
    df_global = df_global.sort_values('timestamp').reset_index(drop=True)

    # Prendre les n_samples_total dernières lignes
    df_global = df_global.tail(n_samples_total).reset_index(drop=True)

    return df_global


# =============================================================================
# TESTS CONFIGURATIONS
# =============================================================================

def run_atr_ml_aware_tests(
    indicator: str,
    split: str,
    fees: float,
    exit_mode: str,
    exit_time: int,
    exit_stop_loss: float
):
    """
    Teste plusieurs configurations ATR ML-Aware.

    Configurations:
        - n (window ATR): 5, 6, 7, 8
        - lambda_w: 0.5, 1.0, 1.5
        - Percentiles: (Q25-Q70), (Q20-Q80), (Q30-Q70)
    """
    print("=" * 80)
    print(f"TEST ATR ML-AWARE - {indicator.upper()} - Split: {split}")
    print("=" * 80)
    print(f"Stratégie: Entry-Focused (ML filtre entrées, exits={exit_mode})")
    print(f"Exit mode: {exit_mode}, time={exit_time}p, stop_loss={exit_stop_loss*100:.1f}%")
    print()

    # Charger Kalman
    print("Chargement prédictions Kalman...")
    pred_kalman, returns, ohlcv_dfs = load_predictions_and_ohlcv(indicator, 'kalman', split)

    # Charger Octave
    print("Chargement prédictions Octave...")
    pred_octave, _, _ = load_predictions_and_ohlcv(indicator, 'octave20', split)

    # Vérifier longueurs
    if len(pred_kalman) != len(pred_octave):
        raise ValueError(f"Longueurs différentes: Kalman {len(pred_kalman)}, Octave {len(pred_octave)}")

    n_samples = len(pred_kalman)
    print(f"Samples: {n_samples:,}")
    print()

    # Aligner OHLCV
    print("Alignement OHLCV...")
    df_ohlcv = align_predictions_with_ohlcv(None, ohlcv_dfs, n_samples)
    print(f"OHLCV shape: {df_ohlcv.shape}")
    print()

    # Directions
    kalman_dir = (pred_kalman > 0.5).astype(int)
    octave_dir = (pred_octave > 0.5).astype(int)

    # Baseline (utilise Kalman, pas de filtre ATR)
    print("Baseline (Kalman, sans filtre ATR)...")
    baseline = backtest_atr_ml_aware(
        predictions=pred_kalman,
        returns=returns,
        atr_ml=np.ones(n_samples),  # ATR dummy
        fees=fees,
        atr_q_low=0.0,
        atr_q_high=100.0,
        exit_mode=exit_mode,
        exit_time=exit_time,
        exit_stop_loss=exit_stop_loss,
        n_atr_window=0,
        lambda_w=0.0
    )

    print(f"  Trades: {baseline.n_trades:,}")
    print(f"  Win Rate: {baseline.win_rate:.2f}%")
    print(f"  PnL Brut: {baseline.pnl_gross:+.2f}%")
    print(f"  PnL Net: {baseline.pnl_net:+.2f}%")
    print(f"  Avg Duration: {baseline.avg_duration:.1f}p")
    print(f"  Avg Confidence: {baseline.avg_confidence:.3f}")
    print(f"  Direction Flips: {baseline.n_direction_flip} ({baseline.n_direction_flip/baseline.n_trades*100:.1f}%)")
    print()

    # Configurations à tester
    n_windows = [5, 6, 7, 8]
    lambdas = [0.5, 1.0, 1.5]
    percentile_ranges = [
        (25, 70),
        (20, 80),
        (30, 70)
    ]

    results = []

    print("Tests ATR ML-Aware...")
    print("-" * 80)

    for n in n_windows:
        for lambda_w in lambdas:
            # Calculer ATR ML-Aware
            atr_ml = compute_ml_aware_atr(df_ohlcv, kalman_dir, octave_dir, n, lambda_w)

            for q_low, q_high in percentile_ranges:
                res = backtest_atr_ml_aware(
                    predictions=pred_kalman,
                    returns=returns,
                    atr_ml=atr_ml,
                    fees=fees,
                    atr_q_low=q_low,
                    atr_q_high=q_high,
                    exit_mode=exit_mode,
                    exit_time=exit_time,
                    exit_stop_loss=exit_stop_loss,
                    n_atr_window=n,
                    lambda_w=lambda_w
                )

                results.append(res)

                # Calcul delta vs baseline
                if baseline.n_trades > 0:
                    reduction = (1 - res.n_trades / baseline.n_trades) * 100
                else:
                    reduction = 0.0

                wr_delta = res.win_rate - baseline.win_rate

                print(f"n={n}, λ={lambda_w:.1f}, Q{q_low}-Q{q_high}:")
                print(f"  Trades: {res.n_trades:,} ({reduction:+.1f}%)")
                print(f"  Win Rate: {res.win_rate:.2f}% (Δ{wr_delta:+.2f}%)")
                print(f"  PnL Net: {res.pnl_net:+.2f}%")
                print(f"  ATR Coverage: {res.atr_coverage:.1f}%")
                print(f"  Avg ATR: {res.avg_atr:.4f}")

    print()
    print("=" * 80)
    print("RÉSUMÉ")
    print("=" * 80)

    # Meilleure config (Win Rate max)
    best_wr = max(results, key=lambda r: r.win_rate)
    print(f"Meilleur Win Rate: {best_wr.win_rate:.2f}%")
    print(f"  n={best_wr.n_atr_window}, λ={best_wr.lambda_w:.1f}")
    print(f"  Trades: {best_wr.n_trades:,}")
    print(f"  PnL Net: {best_wr.pnl_net:+.2f}%")
    print()

    # Meilleure config (PnL Net max)
    best_pnl = max(results, key=lambda r: r.pnl_net)
    print(f"Meilleur PnL Net: {best_pnl.pnl_net:+.2f}%")
    print(f"  n={best_pnl.n_atr_window}, λ={best_pnl.lambda_w:.1f}")
    print(f"  Trades: {best_pnl.n_trades:,}")
    print(f"  Win Rate: {best_pnl.win_rate:.2f}%")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test ATR ML-Aware Filter (Entry-Focused)")
    parser.add_argument('--indicator', type=str, default='macd', choices=['macd', 'rsi', 'cci'],
                        help="Indicateur à tester (défaut: macd)")
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help="Split à tester (défaut: test)")
    parser.add_argument('--fees', type=float, default=0.001,
                        help="Frais par trade (défaut: 0.001 = 0.1%)")
    parser.add_argument('--exit-mode', type=str, default='time', choices=['time', 'stop_loss'],
                        help="Mode de sortie (défaut: time)")
    parser.add_argument('--exit-time', type=int, default=20,
                        help="Durée max trade si exit-mode=time (défaut: 20 périodes)")
    parser.add_argument('--exit-stop-loss', type=float, default=0.02,
                        help="Stop loss si exit-mode=stop_loss (défaut: 0.02 = 2%%)")

    args = parser.parse_args()

    run_atr_ml_aware_tests(
        indicator=args.indicator,
        split=args.split,
        fees=args.fees,
        exit_mode=args.exit_mode,
        exit_time=args.exit_time,
        exit_stop_loss=args.exit_stop_loss
    )


if __name__ == '__main__':
    main()
