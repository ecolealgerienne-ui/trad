"""
Backtest engine — replay Qwen decisions from JSONL logs.

Reads pre-generated decisions (test_run.py output), replays them against
real OHLCV data with strict anti-look-ahead, position sizing, stop/TP,
and forced 22h close.

Usage:
    python -m src.backtest_engine --jsonl logs/test_run_XXXX.jsonl --data-dir src/data_trad
"""

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.feature_engineering import (
    SYMBOLS,
    TIMEFRAMES,
    filter_closed_candles,
    load_all_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------

ANON_TO_REAL = {
    "ASSET_A": "BTCUSDT",
    "ASSET_B": "ETHUSDT",
    "ASSET_C": "SOLUSDT",
    "ASSET_D": "XRPUSDT",
    "ASSET_E": "BNBUSDT",
}

SESSION_START_HOUR = 8
SESSION_END_HOUR = 22

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Position:
    symbol: str
    entry_price: float
    entry_time: pd.Timestamp
    qty: float          # units of the asset
    size_usd: float     # qty * entry_price
    stop_price: float
    tp_price: float
    entry_fee_usd: float


@dataclass
class Trade:
    symbol: str
    entry_price: float
    entry_time: pd.Timestamp
    exit_price: float
    exit_time: pd.Timestamp
    qty: float
    size_usd: float
    pnl_usd: float
    pnl_pct: float
    entry_fee_usd: float
    exit_fee_usd: float
    exit_reason: str    # stop_loss | take_profit | qwen_close | forced_22h
    duration_minutes: int


# ---------------------------------------------------------------------------
# Decision loading
# ---------------------------------------------------------------------------


def load_decisions(jsonl_path: str) -> Dict[str, dict]:
    """Load JSONL, index by as_of timestamp. Only keep success=True cycles."""
    decisions = {}
    total = 0
    skipped = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            rec = json.loads(line.strip())
            if not rec.get("success"):
                skipped += 1
                continue
            ts_str = rec.get("as_of")
            if ts_str:
                decisions[ts_str] = rec

    logger.info(
        "Loaded %d decisions from %d records (%d skipped non-success)",
        len(decisions), total, skipped,
    )
    return decisions


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


def calculate_position_size(
    capital: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    fee_rate: float,
    cash: float,
    sizing_mode: str = "risk",
) -> Optional[Tuple[float, float, float]]:
    """Calculate position size.

    Two modes:
    - "risk": risk_amount = capital * risk_pct, qty = risk_amount / stop_distance
      (can create huge positions on tight stops)
    - "fixed_pct": size_usd = capital * risk_pct (direct % of capital per trade)

    Returns (qty, size_usd, entry_fee_usd) or None if can't afford.
    """
    if sizing_mode == "fixed_pct":
        size_usd = capital * risk_pct
        qty = size_usd / entry_price
        entry_fee_usd = size_usd * fee_rate
    else:
        stop_distance = entry_price - stop_price
        if stop_distance <= 0:
            logger.warning("Invalid stop: entry=%.4f stop=%.4f", entry_price, stop_price)
            return None
        risk_amount_usd = capital * risk_pct
        qty = risk_amount_usd / stop_distance
        size_usd = qty * entry_price
        entry_fee_usd = size_usd * fee_rate

    # Check cash
    total_cost = size_usd + entry_fee_usd
    if total_cost > cash:
        max_size = cash / (1 + fee_rate)
        qty = max_size / entry_price
        size_usd = qty * entry_price
        entry_fee_usd = size_usd * fee_rate
        if size_usd < 1.0:
            return None

    return qty, size_usd, entry_fee_usd


# ---------------------------------------------------------------------------
# Exit processing
# ---------------------------------------------------------------------------


def process_exits(
    positions: List[Position],
    data_sources: dict,
    qwen_assets: List[dict],
    as_of: pd.Timestamp,
    fee_rate: float,
) -> Tuple[List[Trade], List[Position]]:
    """Check exits for all open positions. Order: stop → tp → qwen_close → 22h.

    CONVENTION: If a candle has low <= stop AND high >= tp, stop is hit first
    (conservative backtest assumption).

    Returns (closed_trades, remaining_positions).
    """
    closed = []
    remaining = []

    # Build qwen close set (symbols Qwen wants to close this cycle)
    qwen_close_symbols = set()
    for a in qwen_assets:
        anon = a.get("symbol", "")
        real = ANON_TO_REAL.get(anon, anon)
        if a.get("action") == "close":
            qwen_close_symbols.add(real)

    is_22h = as_of.hour == SESSION_END_HOUR and as_of.minute == 0

    for pos in positions:
        # Get the closed 15m candle at this cycle
        key = (pos.symbol, "15m")
        if key not in data_sources:
            remaining.append(pos)
            continue

        try:
            df_closed = filter_closed_candles(data_sources[key], 15, as_of)
        except ValueError:
            remaining.append(pos)
            continue

        candle = df_closed.iloc[-1]
        c_high = float(candle["high"])
        c_low = float(candle["low"])
        c_close = float(candle["close"])

        exit_price = None
        exit_reason = None

        # Order: stop → tp → qwen_close → 22h
        if c_low <= pos.stop_price:
            exit_price = pos.stop_price
            exit_reason = "stop_loss"
        elif c_high >= pos.tp_price:
            exit_price = pos.tp_price
            exit_reason = "take_profit"
        elif pos.symbol in qwen_close_symbols:
            exit_price = c_close
            exit_reason = "qwen_close"
        elif is_22h:
            exit_price = c_close
            exit_reason = "forced_22h"

        if exit_price is not None:
            exit_fee_usd = pos.qty * exit_price * fee_rate
            pnl_usd = pos.qty * (exit_price - pos.entry_price) - pos.entry_fee_usd - exit_fee_usd
            pnl_pct = pnl_usd / pos.size_usd * 100
            duration = int((as_of - pos.entry_time).total_seconds() / 60)

            closed.append(Trade(
                symbol=pos.symbol,
                entry_price=pos.entry_price,
                entry_time=pos.entry_time,
                exit_price=exit_price,
                exit_time=as_of,
                qty=pos.qty,
                size_usd=pos.size_usd,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                entry_fee_usd=pos.entry_fee_usd,
                exit_fee_usd=exit_fee_usd,
                exit_reason=exit_reason,
                duration_minutes=duration,
            ))
        else:
            remaining.append(pos)

    return closed, remaining


# ---------------------------------------------------------------------------
# Entry processing
# ---------------------------------------------------------------------------


def process_entries(
    qwen_decision: dict,
    features_snapshot: dict,
    data_sources: dict,
    positions: List[Position],
    capital: float,
    cash: float,
    as_of: pd.Timestamp,
    fee_rate: float,
    risk_pct: float,
    funnel: dict,
    sizing_mode: str = "risk",
) -> Tuple[List[Position], float]:
    """Process buy signals from Qwen. No Python-side filters on conviction,
    market_mode, or max_positions — trust Qwen's decisions directly.

    Only filters: already_in_position, insufficient_cash, missing ATR/multipliers.
    Returns (new_positions, updated_cash).
    """
    new_positions = []
    assets = qwen_decision.get("assets", [])
    current_symbols = {p.symbol for p in positions}

    for asset_dec in assets:
        if asset_dec.get("action") != "buy":
            continue

        funnel["total_buys"] += 1
        anon_sym = asset_dec.get("symbol", "")
        real_sym = ANON_TO_REAL.get(anon_sym, anon_sym)

        # Only filter: already in position on this symbol
        if real_sym in current_symbols:
            funnel["filtered_already_in_pos"] += 1
            continue

        # Get entry price from last closed candle
        key = (real_sym, "15m")
        if key not in data_sources:
            funnel["filtered_no_data"] += 1
            continue
        try:
            df_closed = filter_closed_candles(data_sources[key], 15, as_of)
        except ValueError:
            funnel["filtered_no_data"] += 1
            continue
        entry_price = float(df_closed.iloc[-1]["close"])

        # Get ATR from features_snapshot
        snap = features_snapshot.get(anon_sym, {})
        atr = snap.get("atr_15m_abs")
        if atr is None or atr <= 0:
            funnel["filtered_no_atr"] += 1
            continue

        # Calculate stop/TP from ATR multipliers
        stop_mult = asset_dec.get("atr_stop_multiplier")
        tp_mult = asset_dec.get("atr_tp_multiplier")
        if stop_mult is None or tp_mult is None:
            funnel["filtered_no_multipliers"] += 1
            continue

        stop_price = entry_price - atr * stop_mult
        tp_price = entry_price + atr * tp_mult

        if stop_price <= 0:
            funnel["filtered_invalid_stop"] += 1
            continue

        # Size position
        result = calculate_position_size(
            capital, risk_pct, entry_price, stop_price, fee_rate, cash, sizing_mode
        )
        if result is None:
            funnel["filtered_insufficient_cash"] += 1
            continue

        qty, size_usd, entry_fee_usd = result

        pos = Position(
            symbol=real_sym,
            entry_price=entry_price,
            entry_time=as_of,
            qty=qty,
            size_usd=size_usd,
            stop_price=stop_price,
            tp_price=tp_price,
            entry_fee_usd=entry_fee_usd,
        )
        new_positions.append(pos)
        cash -= (size_usd + entry_fee_usd)
        current_symbols.add(real_sym)
        funnel["executed"] += 1

        logger.debug(
            "ENTRY %s @ %.2f | qty=%.6f size=$%.2f | stop=%.2f tp=%.2f",
            real_sym, entry_price, qty, size_usd, stop_price, tp_price,
        )

    return new_positions, cash


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------


def run_backtest(
    data_sources: dict,
    decisions: Dict[str, dict],
    capital: float = 10000.0,
    fee_rate: float = 0.001,
    risk_pct: float = 0.02,
    sizing_mode: str = "risk",
) -> Tuple[List[Trade], List[dict], dict]:
    """Main backtest loop. Returns (all_trades, equity_curve, execution_funnel)."""
    all_trades: List[Trade] = []
    funnel = {
        "total_buys": 0,
        "filtered_already_in_pos": 0,
        "filtered_insufficient_cash": 0,
        "filtered_no_data": 0,
        "filtered_no_atr": 0,
        "filtered_no_multipliers": 0,
        "filtered_invalid_stop": 0,
        "executed": 0,
    }
    equity_curve: List[dict] = []
    positions: List[Position] = []
    cash = capital

    # Sort timestamps
    timestamps = sorted(decisions.keys())
    if not timestamps:
        logger.error("No decisions to replay.")
        return all_trades, equity_curve, funnel

    logger.info(
        "Running backtest: %d cycles, capital=$%.2f, fee=%.2f%%, risk=%.1f%%",
        len(timestamps), capital, fee_rate * 100, risk_pct * 100,
    )

    for ts_str in timestamps:
        as_of = pd.Timestamp(ts_str)
        rec = decisions[ts_str]
        parsed = rec.get("parsed", {})
        snapshot = rec.get("features_snapshot", {})

        if not parsed:
            continue

        qwen_assets = parsed.get("assets", [])

        # --- 1. Process exits ---
        closed, positions = process_exits(
            positions, data_sources, qwen_assets, as_of, fee_rate
        )
        for t in closed:
            all_trades.append(t)
            cash += t.qty * t.exit_price - t.exit_fee_usd

        # --- 2. Process entries ---
        new_pos, cash = process_entries(
            parsed, snapshot, data_sources, positions,
            capital, cash, as_of, fee_rate, risk_pct, funnel, sizing_mode,
        )
        positions.extend(new_pos)

        # --- 3. Record equity ---
        # Unrealized P&L of open positions
        unrealized = 0.0
        for pos in positions:
            key = (pos.symbol, "15m")
            try:
                df_closed = filter_closed_candles(data_sources[key], 15, as_of)
                current_price = float(df_closed.iloc[-1]["close"])
                unrealized += pos.qty * (current_price - pos.entry_price)
            except (ValueError, KeyError):
                pass

        equity = cash + sum(p.size_usd for p in positions) + unrealized
        equity_curve.append({
            "timestamp": ts_str,
            "equity": round(equity, 2),
            "cash": round(cash, 2),
            "open_positions": len(positions),
            "unrealized_pnl": round(unrealized, 2),
        })

    # Close any remaining positions at last available price
    if positions:
        last_ts = pd.Timestamp(timestamps[-1])
        logger.info("Force-closing %d positions at end of backtest", len(positions))
        for pos in positions:
            key = (pos.symbol, "15m")
            try:
                df_closed = filter_closed_candles(data_sources[key], 15, last_ts)
                exit_price = float(df_closed.iloc[-1]["close"])
            except (ValueError, KeyError):
                exit_price = pos.entry_price

            exit_fee = pos.qty * exit_price * fee_rate
            pnl_usd = pos.qty * (exit_price - pos.entry_price) - pos.entry_fee_usd - exit_fee
            pnl_pct = pnl_usd / pos.size_usd * 100 if pos.size_usd > 0 else 0
            duration = int((last_ts - pos.entry_time).total_seconds() / 60)

            all_trades.append(Trade(
                symbol=pos.symbol,
                entry_price=pos.entry_price,
                entry_time=pos.entry_time,
                exit_price=exit_price,
                exit_time=last_ts,
                qty=pos.qty,
                size_usd=pos.size_usd,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                entry_fee_usd=pos.entry_fee_usd,
                exit_fee_usd=exit_fee,
                exit_reason="end_of_backtest",
                duration_minutes=duration,
            ))

    return all_trades, equity_curve, funnel


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def compute_report(
    trades: List[Trade],
    equity_curve: List[dict],
    data_sources: dict,
    capital: float,
    funnel: Optional[dict] = None,
) -> dict:
    """Compute aggregate stats."""
    report: Dict[str, Any] = {
        "capital_initial": capital,
        "total_trades": len(trades),
    }

    if not trades:
        report["error"] = "No trades executed"
        return report

    # Basic P&L
    total_pnl = sum(t.pnl_usd for t in trades)
    report["pnl_total_usd"] = round(total_pnl, 2)
    report["pnl_total_pct"] = round(total_pnl / capital * 100, 2)

    # Final equity
    if equity_curve:
        report["equity_final"] = equity_curve[-1]["equity"]
    else:
        report["equity_final"] = capital + total_pnl

    # Win rate
    winners = [t for t in trades if t.pnl_usd > 0]
    losers = [t for t in trades if t.pnl_usd <= 0]
    report["win_rate_pct"] = round(len(winners) / len(trades) * 100, 1)
    report["winners"] = len(winners)
    report["losers"] = len(losers)

    # Profit factor
    gross_profit = sum(t.pnl_usd for t in winners) if winners else 0
    gross_loss = abs(sum(t.pnl_usd for t in losers)) if losers else 0
    report["profit_factor"] = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

    # Avg P&L per trade
    report["avg_pnl_usd"] = round(total_pnl / len(trades), 2)
    report["avg_pnl_pct"] = round(sum(t.pnl_pct for t in trades) / len(trades), 2)

    # Duration
    durations = [t.duration_minutes for t in trades]
    report["avg_duration_minutes"] = round(np.mean(durations), 1)
    report["median_duration_minutes"] = round(np.median(durations), 1)

    # Fees
    total_fees = sum(t.entry_fee_usd + t.exit_fee_usd for t in trades)
    report["total_fees_usd"] = round(total_fees, 2)

    # Max drawdown from equity curve
    if equity_curve:
        equities = [e["equity"] for e in equity_curve]
        peak = equities[0]
        max_dd = 0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd
        report["max_drawdown_pct"] = round(max_dd, 2)

    # Sharpe (annualized, assuming 15min returns)
    if len(equity_curve) > 1:
        equities = np.array([e["equity"] for e in equity_curve])
        returns = np.diff(equities) / equities[:-1]
        if np.std(returns) > 0:
            periods_per_year = 4 * 14 * 252  # 15min bars, 14h/day, 252 days
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)
            report["sharpe_annualized"] = round(sharpe, 2)

    # Stats by exit reason
    reasons = {}
    for t in trades:
        r = t.exit_reason
        if r not in reasons:
            reasons[r] = {"count": 0, "pnl_usd": 0, "winners": 0}
        reasons[r]["count"] += 1
        reasons[r]["pnl_usd"] += t.pnl_usd
        if t.pnl_usd > 0:
            reasons[r]["winners"] += 1
    for r in reasons:
        reasons[r]["pnl_usd"] = round(reasons[r]["pnl_usd"], 2)
        reasons[r]["win_rate_pct"] = round(
            reasons[r]["winners"] / reasons[r]["count"] * 100, 1
        ) if reasons[r]["count"] > 0 else 0
    report["by_exit_reason"] = reasons

    # Stats by symbol
    symbols = {}
    for t in trades:
        s = t.symbol
        if s not in symbols:
            symbols[s] = {"count": 0, "pnl_usd": 0, "winners": 0}
        symbols[s]["count"] += 1
        symbols[s]["pnl_usd"] += t.pnl_usd
        if t.pnl_usd > 0:
            symbols[s]["winners"] += 1
    for s in symbols:
        symbols[s]["pnl_usd"] = round(symbols[s]["pnl_usd"], 2)
        symbols[s]["win_rate_pct"] = round(
            symbols[s]["winners"] / symbols[s]["count"] * 100, 1
        ) if symbols[s]["count"] > 0 else 0
    report["by_symbol"] = symbols

    # Execution funnel
    if funnel:
        report["execution_funnel"] = funnel

    # Buy & Hold benchmark (equi-weighted 20% each)
    if equity_curve:
        first_ts = pd.Timestamp(equity_curve[0]["timestamp"])
        last_ts = pd.Timestamp(equity_curve[-1]["timestamp"])
        bh_pnl = 0.0
        alloc_per_asset = capital / 5
        for symbol in SYMBOLS:
            key = (symbol, "15m")
            if key not in data_sources:
                continue
            try:
                df = data_sources[key]
                df_start = filter_closed_candles(df, 15, first_ts)
                df_end = filter_closed_candles(df, 15, last_ts)
                p_start = float(df_start.iloc[-1]["close"])
                p_end = float(df_end.iloc[-1]["close"])
                bh_pnl += alloc_per_asset * (p_end - p_start) / p_start
            except (ValueError, KeyError):
                pass
        report["buy_hold_pnl_usd"] = round(bh_pnl, 2)
        report["buy_hold_pnl_pct"] = round(bh_pnl / capital * 100, 2)

    return report


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_outputs(
    trades: List[Trade],
    equity_curve: List[dict],
    report: dict,
    output_dir: str,
):
    """Write trades.csv, equity_curve.csv, report.json, summary.txt."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # trades.csv
    trades_path = out / "trades.csv"
    if trades:
        fieldnames = [
            "symbol", "entry_price", "entry_time", "exit_price", "exit_time",
            "qty", "size_usd", "pnl_usd", "pnl_pct", "entry_fee_usd",
            "exit_fee_usd", "exit_reason", "duration_minutes",
        ]
        with open(trades_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in trades:
                writer.writerow({
                    "symbol": t.symbol,
                    "entry_price": round(t.entry_price, 6),
                    "entry_time": t.entry_time,
                    "exit_price": round(t.exit_price, 6),
                    "exit_time": t.exit_time,
                    "qty": round(t.qty, 8),
                    "size_usd": round(t.size_usd, 2),
                    "pnl_usd": round(t.pnl_usd, 2),
                    "pnl_pct": round(t.pnl_pct, 2),
                    "entry_fee_usd": round(t.entry_fee_usd, 2),
                    "exit_fee_usd": round(t.exit_fee_usd, 2),
                    "exit_reason": t.exit_reason,
                    "duration_minutes": t.duration_minutes,
                })
    logger.info("Trades written to %s (%d trades)", trades_path, len(trades))

    # equity_curve.csv
    eq_path = out / "equity_curve.csv"
    if equity_curve:
        with open(eq_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=equity_curve[0].keys())
            writer.writeheader()
            writer.writerows(equity_curve)
    logger.info("Equity curve written to %s (%d points)", eq_path, len(equity_curve))

    # report.json
    report_path = out / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report written to %s", report_path)

    # summary.txt
    summary_path = out / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  BACKTEST SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"  Capital:           ${report.get('capital_initial', 0):,.2f}\n")
        f.write(f"  Final equity:      ${report.get('equity_final', 0):,.2f}\n")
        f.write(f"  P&L:               ${report.get('pnl_total_usd', 0):,.2f} ({report.get('pnl_total_pct', 0):.1f}%)\n")
        f.write(f"  Total trades:      {report.get('total_trades', 0)}\n")
        f.write(f"  Win rate:          {report.get('win_rate_pct', 0):.1f}%\n")
        f.write(f"  Profit factor:     {report.get('profit_factor', 0):.2f}\n")
        f.write(f"  Avg P&L/trade:     ${report.get('avg_pnl_usd', 0):.2f} ({report.get('avg_pnl_pct', 0):.2f}%)\n")
        f.write(f"  Avg duration:      {report.get('avg_duration_minutes', 0):.0f} min\n")
        f.write(f"  Max drawdown:      {report.get('max_drawdown_pct', 0):.1f}%\n")
        f.write(f"  Total fees:        ${report.get('total_fees_usd', 0):.2f}\n")

        if "sharpe_annualized" in report:
            f.write(f"  Sharpe (ann.):     {report['sharpe_annualized']:.2f}\n")

        if "buy_hold_pnl_usd" in report:
            f.write(f"\n  Buy & Hold:        ${report['buy_hold_pnl_usd']:,.2f} ({report.get('buy_hold_pnl_pct', 0):.1f}%)\n")

        # By exit reason
        by_reason = report.get("by_exit_reason", {})
        if by_reason:
            f.write(f"\n  By exit reason:\n")
            for reason, stats in sorted(by_reason.items()):
                f.write(f"    {reason:15s}: {stats['count']:3d} trades, ${stats['pnl_usd']:8.2f}, WR={stats['win_rate_pct']:.0f}%\n")

        # By symbol
        by_sym = report.get("by_symbol", {})
        if by_sym:
            f.write(f"\n  By symbol:\n")
            for sym, stats in sorted(by_sym.items()):
                f.write(f"    {sym:10s}: {stats['count']:3d} trades, ${stats['pnl_usd']:8.2f}, WR={stats['win_rate_pct']:.0f}%\n")

        # Execution funnel
        ef = report.get("execution_funnel", {})
        if ef:
            f.write(f"\n  Execution funnel:\n")
            f.write(f"    Total buy signals from Qwen:  {ef.get('total_buys', 0)}\n")
            f.write(f"    Filtered already_in_pos:      {ef.get('filtered_already_in_pos', 0)}\n")
            f.write(f"    Filtered insufficient_cash:   {ef.get('filtered_insufficient_cash', 0)}\n")
            f.write(f"    Filtered no_atr:              {ef.get('filtered_no_atr', 0)}\n")
            f.write(f"    Filtered no_multipliers:      {ef.get('filtered_no_multipliers', 0)}\n")
            f.write(f"    Filtered invalid_stop:        {ef.get('filtered_invalid_stop', 0)}\n")
            f.write(f"    Filtered no_data:             {ef.get('filtered_no_data', 0)}\n")
            f.write(f"    Executed:                     {ef.get('executed', 0)}\n")

        f.write("\n" + "=" * 60 + "\n")

    logger.info("Summary written to %s", summary_path)

    # Print summary to stdout
    with open(summary_path, "r") as f:
        print(f.read())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Backtest: replay Qwen decisions from JSONL")
    parser.add_argument("--jsonl", required=True, help="Path to test_run JSONL log")
    parser.add_argument("--data-dir", default="src/data_trad", help="OHLCV data directory")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital (USD)")
    parser.add_argument("--fee-rate", type=float, default=0.001, help="Fee rate per side (default 0.1%%)")
    parser.add_argument("--risk-pct", type=float, default=0.10, help="Risk/size per trade (default 10%%)")
    parser.add_argument("--sizing-mode", choices=["risk", "fixed_pct"], default="fixed_pct",
                        help="risk=2%%risk/stop_distance, fixed_pct=%%capital per trade (default: fixed_pct)")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    # Load OHLCV data
    logger.info("Loading OHLCV data from %s ...", args.data_dir)
    data_sources = load_all_data(args.data_dir)
    logger.info("Loaded %d datasets.", len(data_sources))

    # Load decisions
    decisions = load_decisions(args.jsonl)
    if not decisions:
        logger.error("No valid decisions found in %s", args.jsonl)
        sys.exit(1)

    # Run backtest
    trades, equity_curve, funnel = run_backtest(
        data_sources, decisions, args.capital, args.fee_rate, args.risk_pct,
        args.sizing_mode,
    )

    # Report
    report = compute_report(trades, equity_curve, data_sources, args.capital, funnel)

    # Write outputs
    write_outputs(trades, equity_curve, report, args.output_dir)


if __name__ == "__main__":
    main()
