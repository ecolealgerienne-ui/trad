#!/usr/bin/env python3
"""
Fonctions communes de backtest réutilisables.

Principe: Acheter à Open(t+1), pas à Close(t) pour être causalement correct.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List
from enum import Enum


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
    position: str  # 'LONG' ou 'SHORT'
    pnl: float
    pnl_after_fees: float


@dataclass
class BacktestResult:
    """Résultat backtest."""
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


def backtest_simple_labels(
    labels: np.ndarray,
    df: pd.DataFrame,
    fees: float = 0.001
) -> BacktestResult:
    """
    Backtest simple avec labels binaires (0=DOWN, 1=UP).

    Méthode CAUSALE CORRECTE:
    - À timestep i, on lit label[i]
    - Si label[i] != position actuelle → on change
    - **On entre à Open[i+1]** (pas à Close[i])
    - Le PnL est calculé sur les Opens

    Args:
        labels: Labels binaires (n_samples,) - 1=UP, 0=DOWN
        df: DataFrame avec colonnes 'open', 'close' (doit avoir au moins n_samples lignes)
        fees: Frais par side (défaut: 0.001 = 0.1%)

    Returns:
        BacktestResult

    Exemple:
        ```python
        # Calculer labels
        filtered = apply_kalman_global(indicator_values)
        labels = (filtered[:-1] > np.roll(filtered, 1)[:-1]).astype(int)

        # Backtest
        result = backtest_simple_labels(labels, df, fees=0.001)
        print(f"PnL Net: {result.total_pnl_after_fees*100:.2f}%")
        ```
    """
    n_samples = len(labels)

    # Vérifier qu'on a assez de données (besoin de i+1 pour l'Open)
    if len(df) < n_samples + 1:
        raise ValueError(
            f"DataFrame trop court! Besoin de {n_samples + 1} lignes, "
            f"disponible: {len(df)}"
        )

    trades = []
    position = Position.FLAT
    entry_time = 0
    entry_price = 0.0

    n_long = 0
    n_short = 0

    for i in range(n_samples):
        label = labels[i]

        # Target position basé sur label
        target = Position.LONG if label == 1 else Position.SHORT

        # Changement de position?
        if position != target:
            # Fermer position actuelle
            if position != Position.FLAT:
                # Sortie à Open[i+1]
                exit_price = df['open'].iloc[i + 1]

                # Calculer PnL
                if position == Position.LONG:
                    pnl = (exit_price - entry_price) / entry_price
                else:  # SHORT
                    pnl = (entry_price - exit_price) / entry_price

                # Frais: 2× (entrée + sortie)
                trade_fees = 2 * fees
                pnl_after_fees = pnl - trade_fees

                trades.append(Trade(
                    start=entry_time,
                    end=i,
                    duration=i - entry_time,
                    position='LONG' if position == Position.LONG else 'SHORT',
                    pnl=pnl,
                    pnl_after_fees=pnl_after_fees
                ))

            # Ouvrir nouvelle position à Open[i+1]
            position = target
            entry_time = i
            entry_price = df['open'].iloc[i + 1]

            if target == Position.LONG:
                n_long += 1
            else:
                n_short += 1

    # Fermer position finale
    if position != Position.FLAT:
        # Sortie à Open[n_samples] (dernière valeur disponible)
        exit_price = df['open'].iloc[n_samples]

        if position == Position.LONG:
            pnl = (exit_price - entry_price) / entry_price
        else:  # SHORT
            pnl = (entry_price - exit_price) / entry_price

        trade_fees = 2 * fees
        pnl_after_fees = pnl - trade_fees

        trades.append(Trade(
            start=entry_time,
            end=n_samples - 1,
            duration=n_samples - 1 - entry_time,
            position='LONG' if position == Position.LONG else 'SHORT',
            pnl=pnl,
            pnl_after_fees=pnl_after_fees
        ))

    # Calculer statistiques
    return _compute_stats(trades, n_long, n_short, n_samples)


def _compute_stats(
    trades: List[Trade],
    n_long: int,
    n_short: int,
    n_samples: int
) -> BacktestResult:
    """Calcule statistiques backtest."""
    if len(trades) == 0:
        return BacktestResult(
            n_trades=0,
            n_long=0,
            n_short=0,
            total_pnl=0.0,
            total_pnl_after_fees=0.0,
            total_fees=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_duration=0.0,
            sharpe_ratio=0.0,
            trades=[]
        )

    pnls = np.array([t.pnl for t in trades])
    pnls_after_fees = np.array([t.pnl_after_fees for t in trades])
    durations = np.array([t.duration for t in trades])

    total_pnl = pnls.sum()
    total_pnl_after_fees = pnls_after_fees.sum()
    total_fees = total_pnl - total_pnl_after_fees

    wins = pnls_after_fees > 0
    losses = pnls_after_fees < 0

    win_rate = wins.mean() if len(trades) > 0 else 0.0

    sum_wins = pnls_after_fees[wins].sum() if wins.any() else 0.0
    sum_losses = abs(pnls_after_fees[losses].sum()) if losses.any() else 0.0
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else 0.0

    avg_win = pnls_after_fees[wins].mean() if wins.any() else 0.0
    avg_loss = pnls_after_fees[losses].mean() if losses.any() else 0.0

    avg_duration = durations.mean()

    # Sharpe Ratio (annualisé, 5min = 288 périodes/jour)
    if len(pnls_after_fees) > 1:
        returns_mean = pnls_after_fees.mean()
        returns_std = pnls_after_fees.std()
        if returns_std > 0:
            sharpe = (returns_mean / returns_std) * np.sqrt(288 * 365)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    return BacktestResult(
        n_trades=len(trades),
        n_long=n_long,
        n_short=n_short,
        total_pnl=total_pnl,
        total_pnl_after_fees=total_pnl_after_fees,
        total_fees=total_fees,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_duration=avg_duration,
        sharpe_ratio=sharpe,
        trades=trades
    )
