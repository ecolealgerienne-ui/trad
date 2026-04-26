"""
backtest_realistic.py — Backtest event-driven avec frais et métriques de PnL (étape 8 v5.0).

Charge predictions_test.npz et applique différentes stratégies de filtrage pour
mesurer la viabilité réelle :
  - Threshold-based: trade si score >= 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
  - Top-K%%: trade le top 1, 2, 5, 10, 25, 50% des events les plus confiants

Pour chaque stratégie:
  - n_trades, win_rate, avg PnL gross/net per-trade, cumul PnL
  - Sharpe ratio annualisé
  - Max drawdown sur equity curve cumulative
  - Calmar ratio

Output:
  - backtest_summary.json (table comparative des stratégies)
  - equity_curves.csv (per-trade equity timeline pour chaque stratégie)

Frais : déjà appliqués dans pnl_after_fees_pct de predictions_test.npz
       (default 0.04% × 2 round-trip = 0.08%, configurable via --extra-slippage-pct
       pour ajouter slippage).

Usage:
    python -m experiments.patchtst_v5.backtest_realistic \\
        --predictions models/patchtst_v5/predictions_test.npz \\
        --output-dir models/patchtst_v5/

Voir STATUS_v5.0.md et experiments/patchtst_v5/README.md.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger("patchtst_v5.backtest_realistic")

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
TOP_K_PCTS = [1, 2, 5, 10, 25, 50]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class StrategyResult:
    name: str
    n_trades: int
    n_wins: int
    win_rate: float
    avg_pnl_gross_pct: float
    avg_pnl_net_pct: float
    cumul_pnl_net_pct: float
    annualized_return_pct: float
    annualized_std_pct: float
    sharpe_annualized: float
    max_drawdown_pct: float
    calmar: float
    span_days: float


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def load_predictions(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {
        "scores": data["scores"].astype("float64"),
        "y_true": data["y_true"].astype("int8"),
        "direction": data["direction"].astype("int8"),
        "timestamp": data["timestamp"],
        "pnl_after_fees_pct": data["pnl_after_fees_pct"].astype("float64"),
    }


def equity_metrics(pnl_pct: np.ndarray, span_days: float) -> dict[str, float]:
    """Compute Sharpe, max drawdown, etc. from a series of per-trade returns."""
    n = len(pnl_pct)
    if n == 0:
        return {
            "annualized_return_pct": 0.0,
            "annualized_std_pct": 0.0,
            "sharpe_annualized": float("nan"),
            "max_drawdown_pct": 0.0,
        }
    mean_per_trade = float(pnl_pct.mean())
    std_per_trade = float(pnl_pct.std(ddof=0))
    trades_per_year = n / max(span_days / 365.25, 1e-6)
    annualized_return = mean_per_trade * trades_per_year
    annualized_std = std_per_trade * np.sqrt(trades_per_year)
    sharpe = annualized_return / annualized_std if annualized_std > 1e-9 else float("nan")

    equity = np.cumsum(pnl_pct)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    return {
        "annualized_return_pct": annualized_return,
        "annualized_std_pct": annualized_std,
        "sharpe_annualized": sharpe,
        "max_drawdown_pct": max_dd,
    }


def evaluate_strategy(
    name: str,
    mask: np.ndarray,
    scores: np.ndarray,
    y: np.ndarray,
    pnl: np.ndarray,
    extra_slippage_pct: float,
    span_days: float,
) -> tuple[StrategyResult, np.ndarray, np.ndarray]:
    """Apply selection mask and compute strategy metrics + equity curve."""
    sel_y = y[mask]
    sel_pnl = pnl[mask] - extra_slippage_pct  # additional friction beyond train fees
    n = int(mask.sum())
    n_wins = int(sel_y.sum())
    win_rate = n_wins / n if n else float("nan")

    if n == 0:
        equity = np.array([])
        eq_metrics = equity_metrics(np.array([]), span_days)
    else:
        # gross = pnl before fees: we don't have it directly; we know pnl_net = gross - 2*fees(0.04% default)
        # For analysis purposes, expose net only — gross was reported in pivot_labeler stage.
        equity = np.cumsum(sel_pnl)
        eq_metrics = equity_metrics(sel_pnl, span_days)

    cumul = float(sel_pnl.sum()) if n else 0.0
    avg_net = float(sel_pnl.mean()) if n else 0.0
    avg_gross = avg_net + extra_slippage_pct + 0.08  # rough reconstruct (default fees were 2×0.04 = 0.08)
    calmar = (eq_metrics["annualized_return_pct"] / abs(eq_metrics["max_drawdown_pct"])
              if eq_metrics["max_drawdown_pct"] < -1e-6 else float("nan"))

    result = StrategyResult(
        name=name,
        n_trades=n,
        n_wins=n_wins,
        win_rate=win_rate,
        avg_pnl_gross_pct=avg_gross,
        avg_pnl_net_pct=avg_net,
        cumul_pnl_net_pct=cumul,
        annualized_return_pct=eq_metrics["annualized_return_pct"],
        annualized_std_pct=eq_metrics["annualized_std_pct"],
        sharpe_annualized=eq_metrics["sharpe_annualized"],
        max_drawdown_pct=eq_metrics["max_drawdown_pct"],
        calmar=calmar,
        span_days=span_days,
    )
    return result, equity, sel_pnl


def run_strategies(
    scores: np.ndarray,
    y: np.ndarray,
    pnl: np.ndarray,
    timestamps: np.ndarray,
    extra_slippage_pct: float,
) -> tuple[list[StrategyResult], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Run threshold + top-K%% strategies, return results + equity curves + per-trade returns."""
    ts_pd = pd.to_datetime(timestamps)
    span_days = float((ts_pd.max() - ts_pd.min()).total_seconds() / 86400)
    logger.info("Backtest span: %.1f days (%.2f years)", span_days, span_days / 365.25)
    logger.info("Total events available: %d", len(scores))
    logger.info("Extra slippage applied: %.4f%% per trade (in addition to %.2f%% fees from labeler)",
                extra_slippage_pct, 0.08)

    results: list[StrategyResult] = []
    equity_curves: dict[str, np.ndarray] = {}
    per_trade_returns: dict[str, np.ndarray] = {}

    # Baseline: every event traded
    mask_all = np.ones(len(scores), dtype=bool)
    res_all, eq_all, ret_all = evaluate_strategy("all_events", mask_all, scores, y, pnl,
                                                  extra_slippage_pct, span_days)
    results.append(res_all)
    equity_curves["all_events"] = eq_all
    per_trade_returns["all_events"] = ret_all

    # Threshold strategies
    for t in THRESHOLDS:
        mask = scores >= t
        name = f"threshold_{t:.2f}"
        res, eq, ret = evaluate_strategy(name, mask, scores, y, pnl, extra_slippage_pct, span_days)
        results.append(res)
        equity_curves[name] = eq
        per_trade_returns[name] = ret

    # Top-K%% strategies
    sorted_idx = np.argsort(-scores)
    n_total = len(scores)
    for k in TOP_K_PCTS:
        n_top = max(1, int(n_total * k / 100))
        mask = np.zeros(n_total, dtype=bool)
        mask[sorted_idx[:n_top]] = True
        name = f"top_{k}pct"
        res, eq, ret = evaluate_strategy(name, mask, scores, y, pnl, extra_slippage_pct, span_days)
        results.append(res)
        equity_curves[name] = eq
        per_trade_returns[name] = ret

    return results, equity_curves, per_trade_returns


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary_table(results: list[StrategyResult]) -> None:
    logger.info("=" * 110)
    logger.info("BACKTEST SUMMARY — sorted by Sharpe annualized")
    logger.info("=" * 110)
    header = (
        f"{'Strategy':<20} {'Trades':>7} {'WR':>6} "
        f"{'AvgNet%':>9} {'CumulNet%':>10} {'AnnRet%':>9} {'AnnStd%':>9} "
        f"{'Sharpe':>7} {'MaxDD%':>9} {'Calmar':>8}"
    )
    logger.info(header)
    logger.info("-" * 110)
    sorted_results = sorted(
        results,
        key=lambda r: (r.sharpe_annualized if not np.isnan(r.sharpe_annualized) else -1e9),
        reverse=True,
    )
    for r in sorted_results:
        sharpe_str = f"{r.sharpe_annualized:.2f}" if not np.isnan(r.sharpe_annualized) else "  nan "
        calmar_str = f"{r.calmar:.2f}" if not np.isnan(r.calmar) else "  nan "
        wr_str = f"{r.win_rate:.3f}" if not np.isnan(r.win_rate) else "  nan"
        logger.info(
            f"{r.name:<20} {r.n_trades:>7d} {wr_str:>6} "
            f"{r.avg_pnl_net_pct:>+9.4f} {r.cumul_pnl_net_pct:>+10.2f} "
            f"{r.annualized_return_pct:>+9.2f} {r.annualized_std_pct:>9.2f} "
            f"{sharpe_str:>7} {r.max_drawdown_pct:>+9.2f} {calmar_str:>8}"
        )
    logger.info("=" * 110)


def save_equity_curves(curves: dict[str, np.ndarray], output_path: Path) -> None:
    max_len = max(len(c) for c in curves.values()) if curves else 0
    df = pd.DataFrame()
    for name, c in curves.items():
        padded = np.full(max_len, np.nan)
        padded[: len(c)] = c
        df[name] = padded
    df.index.name = "trade_idx"
    df.to_csv(output_path)
    logger.info("Equity curves saved: %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, default=Path("models/patchtst_v5/predictions_test.npz"))
    p.add_argument("--output-dir", type=Path, default=Path("models/patchtst_v5/"))
    p.add_argument("--extra-slippage-pct", type=float, default=0.0,
                   help="Additional friction per trade in %% (default 0.0). Cumulative with the "
                        "0.08%% round-trip fees already in pnl_after_fees_pct.")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    logger.info("Loading predictions: %s", args.predictions)
    pred = load_predictions(args.predictions)

    results, equity_curves, _ = run_strategies(
        scores=pred["scores"],
        y=pred["y_true"],
        pnl=pred["pnl_after_fees_pct"],
        timestamps=pred["timestamp"],
        extra_slippage_pct=args.extra_slippage_pct,
    )

    print_summary_table(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / "backtest_summary.json"
    summary_path.write_text(json.dumps([asdict(r) for r in results], indent=2, default=str))
    logger.info("Summary saved: %s", summary_path)

    equity_path = args.output_dir / "equity_curves.csv"
    save_equity_curves(equity_curves, equity_path)

    # Highlight key conclusions
    by_sharpe = sorted(
        results,
        key=lambda r: (r.sharpe_annualized if not np.isnan(r.sharpe_annualized) else -1e9),
        reverse=True,
    )
    best = by_sharpe[0]
    logger.info("Best strategy by Sharpe : %s (Sharpe=%.2f, AnnRet=%+.2f%%, MaxDD=%+.2f%%, Trades=%d)",
                best.name, best.sharpe_annualized, best.annualized_return_pct,
                best.max_drawdown_pct, best.n_trades)
    return 0


if __name__ == "__main__":
    sys.exit(main())
