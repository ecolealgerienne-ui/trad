"""
Backtest engine v2 — live mode with Qwen calls + portfolio context.

Two modes:
- live: calls Qwen at each cycle with full context (portfolio, trades, session)
- replay: reads decisions from JSONL (legacy, for quick iteration)

Supports --resume to restart from last logged cycle after a crash.

Usage:
    python -m src.backtest_engine --mode live --data-dir src/data_trad --start 2026-03-01 --end 2026-04-01
    python -m src.backtest_engine --mode live --resume logs/backtest_XXXX.jsonl
    python -m src.backtest_engine --mode replay --jsonl logs/test_run_XXXX.jsonl
"""

import argparse
import csv
import json
import logging
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.feature_engineering import (
    SYMBOLS,
    anonymize_and_format,
    compute_features,
    filter_closed_candles,
    load_all_data,
)
from src.context_formatter import format_user_message
from src.llm_client import call_gemma, load_system_prompt, ping_ollama

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANON_MAPPING = {
    "BTCUSDT": "ASSET_A", "ETHUSDT": "ASSET_B", "SOLUSDT": "ASSET_C",
    "XRPUSDT": "ASSET_D", "BNBUSDT": "ASSET_E",
}
ANON_TO_REAL = {v: k for k, v in ANON_MAPPING.items()}

SESSION_START_HOUR = 8
SESSION_END_HOUR = 22


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Position:
    symbol: str
    anon_symbol: str
    entry_price: float
    entry_time: pd.Timestamp
    qty: float
    size_usd: float
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
    exit_reason: str
    duration_minutes: int

MAX_POSITION_PCT = 0.20  # 20% of capital per position max


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def calculate_position_size(
    capital: float, risk_pct: float, entry_price: float,
    stop_price: float, fee_rate: float, cash: float,
    funnel: Optional[dict] = None,
) -> Optional[Tuple[float, float, float]]:
    """Size based on risk with cap at MAX_POSITION_PCT of capital.

    1. qty = (capital * risk_pct) / stop_distance  (risk-based)
    2. Cap size_usd at capital * MAX_POSITION_PCT   (exposure limit)
    3. Check cash available

    Returns (qty, size_usd, entry_fee_usd) or None if can't afford.
    """
    stop_distance = entry_price - stop_price
    if stop_distance <= 0:
        return None

    # Risk-based sizing
    risk_amount = capital * risk_pct
    qty = risk_amount / stop_distance
    size_usd = qty * entry_price

    # Cap at max position size
    max_size_usd = capital * MAX_POSITION_PCT
    if size_usd > max_size_usd:
        qty = max_size_usd / entry_price
        size_usd = max_size_usd
        if funnel is not None:
            funnel["positions_capped_at_max"] = funnel.get("positions_capped_at_max", 0) + 1

    entry_fee = size_usd * fee_rate

    # Check cash
    required = size_usd * (1 + fee_rate)
    if required > cash:
        return None

    return qty, size_usd, entry_fee


# ---------------------------------------------------------------------------
# Exit processing
# ---------------------------------------------------------------------------

def process_exits(
    positions: List[Position], data_sources: dict,
    qwen_assets: List[dict], as_of: pd.Timestamp, fee_rate: float,
) -> Tuple[List[Trade], List[Position]]:
    """Check exits: stop → tp → qwen_close → 22h.
    If low <= stop AND high >= tp, stop wins (conservative).
    """
    closed = []
    remaining = []

    qwen_close_syms = set()
    for a in qwen_assets:
        real = ANON_TO_REAL.get(a.get("symbol", ""), "")
        if a.get("action") == "close":
            qwen_close_syms.add(real)

    is_22h = as_of.hour == SESSION_END_HOUR and as_of.minute == 0

    for pos in positions:
        key = (pos.symbol, "15m")
        try:
            df = filter_closed_candles(data_sources[key], 15, as_of)
        except (ValueError, KeyError):
            remaining.append(pos)
            continue

        candle = df.iloc[-1]
        h, l, c = float(candle["high"]), float(candle["low"]), float(candle["close"])

        exit_price, reason = None, None
        if l <= pos.stop_price:
            exit_price, reason = pos.stop_price, "stop_loss"
        elif h >= pos.tp_price:
            exit_price, reason = pos.tp_price, "take_profit"
        elif pos.symbol in qwen_close_syms:
            exit_price, reason = c, "qwen_close"
        elif is_22h:
            exit_price, reason = c, "forced_22h"

        if exit_price:
            exit_fee = pos.qty * exit_price * fee_rate
            pnl = pos.qty * (exit_price - pos.entry_price) - pos.entry_fee_usd - exit_fee
            pnl_pct = pnl / pos.size_usd * 100 if pos.size_usd > 0 else 0
            dur = int((as_of - pos.entry_time).total_seconds() / 60)
            closed.append(Trade(
                pos.symbol, pos.entry_price, pos.entry_time, exit_price, as_of,
                pos.qty, pos.size_usd, pnl, pnl_pct,
                pos.entry_fee_usd, exit_fee, reason, dur,
            ))
        else:
            remaining.append(pos)

    return closed, remaining


# ---------------------------------------------------------------------------
# Entry processing — Qwen decides, Python executes
# ---------------------------------------------------------------------------

def process_entries(
    qwen_assets: List[dict], features_snapshot: dict,
    data_sources: dict, positions: List[Position],
    capital: float, cash: float, as_of: pd.Timestamp,
    fee_rate: float, risk_pct: float, funnel: dict,
) -> Tuple[List[Position], float]:
    """Execute buy signals. No Python filters on conviction/mode/max_pos."""
    new_positions = []
    current_syms = {p.symbol for p in positions}

    for a in qwen_assets:
        if a.get("action") != "buy":
            continue
        funnel["total_buys"] += 1

        anon = a.get("symbol", "")
        real = ANON_TO_REAL.get(anon, anon)

        if real in current_syms:
            funnel["filtered_already_in_pos"] += 1
            continue

        # Entry price from last closed 15m candle
        try:
            df = filter_closed_candles(data_sources[(real, "15m")], 15, as_of)
        except (ValueError, KeyError):
            funnel["filtered_no_data"] += 1
            continue
        entry_price = float(df.iloc[-1]["close"])

        # ATR from snapshot
        snap = features_snapshot.get(anon, {})
        atr = snap.get("atr_15m_abs")
        if not atr or atr <= 0:
            funnel["filtered_no_atr"] += 1
            continue

        stop_mult = a.get("stop_mult")
        tp_mult = a.get("tp_mult")
        if stop_mult is None or tp_mult is None:
            funnel["filtered_no_multipliers"] += 1
            continue

        stop_price = entry_price - atr * stop_mult
        tp_price = entry_price + atr * tp_mult
        if stop_price <= 0:
            funnel["filtered_invalid_stop"] += 1
            continue

        result = calculate_position_size(capital, risk_pct, entry_price, stop_price, fee_rate, cash, funnel)
        if result is None:
            funnel["filtered_insufficient_cash"] += 1
            continue

        qty, size_usd, entry_fee = result
        new_positions.append(Position(
            real, anon, entry_price, as_of, qty, size_usd, stop_price, tp_price, entry_fee,
        ))
        cash -= (size_usd + entry_fee)
        current_syms.add(real)
        funnel["executed"] += 1

    return new_positions, cash


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def build_portfolio_state(
    positions: List[Position], data_sources: dict,
    as_of: pd.Timestamp, cash: float, equity: float,
) -> dict:
    """Build portfolio snapshot for context_formatter."""
    pos_list = []
    for p in positions:
        try:
            df = filter_closed_candles(data_sources[(p.symbol, "15m")], 15, as_of)
            current = float(df.iloc[-1]["close"])
        except (ValueError, KeyError):
            current = p.entry_price
        pnl_pct = (current - p.entry_price) / p.entry_price * 100
        age = int((as_of - p.entry_time).total_seconds() / 60)
        pos_list.append({
            "symbol": p.anon_symbol,
            "entry_price": p.entry_price,
            "current_price": current,
            "pnl_pct": round(pnl_pct, 2),
            "stop_price": p.stop_price,
            "tp_price": p.tp_price,
            "age_minutes": age,
        })

    exposure = sum(p.size_usd for p in positions) / equity * 100 if equity > 0 else 0
    return {
        "positions": pos_list,
        "cash": round(cash, 2),
        "exposure_pct": round(exposure, 1),
        "equity": round(equity, 2),
    }


def build_features_snapshot(features: dict) -> dict:
    """Extract price + ATR per asset for backtest sizing."""
    snap = {}
    for a in features.get("assets", []):
        sym = a.get("_symbol", "")
        anon = ANON_MAPPING.get(sym, sym)
        snap[anon] = {
            "real_symbol": sym,
            "price": a.get("price"),
            "atr_15m_abs": (a.get("volatility") or {}).get("atr_15m_abs"),
        }
    return snap


class SessionStats:
    """Track daily session stats. Resets at SESSION_START_HOUR."""

    def __init__(self):
        self.current_day = None
        self.pnl_usd = 0.0
        self.capital_start = 0.0
        self.n_closed = 0
        self.n_winners = 0

    def maybe_reset(self, as_of: pd.Timestamp, equity: float):
        day = as_of.normalize()
        if as_of.hour < SESSION_START_HOUR:
            day -= pd.Timedelta(days=1)
        if day != self.current_day:
            self.current_day = day
            self.pnl_usd = 0.0
            self.capital_start = equity
            self.n_closed = 0
            self.n_winners = 0

    def record_trade(self, trade: Trade):
        self.pnl_usd += trade.pnl_usd
        self.n_closed += 1
        if trade.pnl_usd > 0:
            self.n_winners += 1

    def to_dict(self) -> dict:
        pnl_pct = self.pnl_usd / self.capital_start * 100 if self.capital_start > 0 else 0
        wr = self.n_winners / self.n_closed * 100 if self.n_closed > 0 else 0
        return {
            "pnl_pct": round(pnl_pct, 2),
            "n_closed": self.n_closed,
            "win_rate_pct": round(wr, 1),
        }


# ---------------------------------------------------------------------------
# Timestamp generation
# ---------------------------------------------------------------------------

def generate_timestamps(data_sources: dict, start: str, end: str) -> List[pd.Timestamp]:
    """Generate 15-min timestamps between start and end, 08:00-22:00 UTC."""
    t_start = pd.Timestamp(start)
    t_end = pd.Timestamp(end)
    all_ts = pd.date_range(t_start, t_end, freq="15min")
    filtered = [ts for ts in all_ts if SESSION_START_HOUR <= ts.hour < SESSION_END_HOUR]

    # Ensure data available (need 24h+ of 15m before first cycle)
    btc = data_sources[("BTCUSDT", "15m")]
    data_start = btc.index[0] + pd.Timedelta(hours=24)
    data_end = btc.index[-1] + pd.Timedelta(minutes=15)
    filtered = [ts for ts in filtered if data_start <= ts <= data_end]
    return filtered


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_resume_state(jsonl_path: str) -> Tuple[str, List[dict]]:
    """Load existing JSONL, find last as_of, return (last_as_of, all_records)."""
    records = []
    last_as_of = None
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            records.append(rec)
            last_as_of = rec.get("as_of")
    return last_as_of, records


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def run_backtest_live(
    data_sources: dict, timestamps: List[pd.Timestamp],
    capital: float, fee_rate: float, risk_pct: float,
    model: str, system_prompt: str, log_path: Path,
    resume_records: Optional[List[dict]] = None,
) -> Tuple[List[Trade], List[dict], dict]:
    """Live backtest: call Qwen each cycle with full context."""
    all_trades: List[Trade] = []
    equity_curve: List[dict] = []
    positions: List[Position] = []
    cash = capital
    recent_trades: deque = deque(maxlen=5)
    session = SessionStats()
    funnel = {
        "total_buys": 0, "filtered_already_in_pos": 0,
        "filtered_insufficient_cash": 0, "filtered_no_data": 0,
        "filtered_no_atr": 0, "filtered_no_multipliers": 0,
        "filtered_invalid_stop": 0, "executed": 0,
    }

    interrupted = False
    def sig_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        logger.warning("Ctrl+C — flushing and printing stats...")
    signal.signal(signal.SIGINT, sig_handler)

    context_base = {
        "positions_open_count": 0, "total_exposure_pct": 0.0,
        "btc_dominance": {"value": 54.0, "chg_24h_pct": 0.0},
        "funding_rates": {s: 0.0 for s in SYMBOLS},
    }

    total = len(timestamps)
    logger.info("Backtest: %d cycles, capital=$%.0f, fee=%.2f%%, risk=%.1f%%",
                total, capital, fee_rate * 100, risk_pct * 100)

    with open(log_path, "a", encoding="utf-8") as logf:
        for i, as_of in enumerate(timestamps):
            if interrupted:
                break

            record = {
                "as_of": str(as_of), "success": False, "latency_sec": 0,
                "parsed": None, "error": None, "features_snapshot": None,
            }
            t0 = time.perf_counter()

            try:
                # Compute equity
                unrealized = 0.0
                for p in positions:
                    try:
                        df = filter_closed_candles(data_sources[(p.symbol, "15m")], 15, as_of)
                        unrealized += p.qty * (float(df.iloc[-1]["close"]) - p.entry_price)
                    except (ValueError, KeyError):
                        pass
                equity = cash + sum(p.size_usd for p in positions) + unrealized

                # Session reset
                session.maybe_reset(as_of, equity)

                # Features
                features = compute_features(data_sources, as_of, context_base)
                anon_features = anonymize_and_format(features, ANON_MAPPING)
                snapshot = build_features_snapshot(features)
                record["features_snapshot"] = snapshot

                # Portfolio state
                portfolio = build_portfolio_state(positions, data_sources, as_of, cash, equity)

                # Format user message with full context
                user_msg = format_user_message(
                    anon_features, portfolio, list(recent_trades), session.to_dict(),
                )

                # Call Qwen
                llm_result = call_gemma(system_prompt, anon_features, model=model,
                                        _override_user_content=user_msg)
                record["success"] = llm_result["success"]
                record["latency_sec"] = llm_result["latency_sec"]
                record["parsed"] = llm_result["parsed"]
                record["thinking"] = llm_result.get("thinking")

                if not llm_result["success"]:
                    record["error"] = str(llm_result.get("validation_errors", []))
                    logf.write(json.dumps(record, default=str) + "\n")
                    logf.flush()
                    continue

                parsed = llm_result["parsed"]
                qwen_assets = parsed.get("assets", [])

                # Exits
                closed, positions = process_exits(
                    positions, data_sources, qwen_assets, as_of, fee_rate,
                )
                for t in closed:
                    all_trades.append(t)
                    cash += t.qty * t.exit_price - t.exit_fee_usd
                    session.record_trade(t)
                    recent_trades.append({
                        "symbol": ANON_MAPPING.get(t.symbol, t.symbol),
                        "exit_reason": t.exit_reason,
                        "pnl_pct": round(t.pnl_pct, 2),
                        "duration_min": t.duration_minutes,
                    })

                # Entries
                new_pos, cash = process_entries(
                    qwen_assets, snapshot, data_sources, positions,
                    capital, cash, as_of, fee_rate, risk_pct, funnel,
                )
                positions.extend(new_pos)

                # Equity curve
                unrealized = 0.0
                for p in positions:
                    try:
                        df = filter_closed_candles(data_sources[(p.symbol, "15m")], 15, as_of)
                        unrealized += p.qty * (float(df.iloc[-1]["close"]) - p.entry_price)
                    except (ValueError, KeyError):
                        pass
                equity = cash + sum(p.size_usd for p in positions) + unrealized
                equity_curve.append({
                    "timestamp": str(as_of), "equity": round(equity, 2),
                    "cash": round(cash, 2), "open_positions": len(positions),
                })

            except Exception as e:
                record["error"] = str(e)
                logger.error("Cycle %s error: %s", as_of, e)

            record["latency_sec"] = round(time.perf_counter() - t0, 2)
            logf.write(json.dumps(record, default=str) + "\n")
            logf.flush()

            # Progress every 10 cycles
            if (i + 1) % 10 == 0:
                pct = (i + 1) / total * 100
                lat = record["latency_sec"]
                status = "OK" if record["success"] else "FAIL"
                logger.info(
                    "[%d/%d %.0f%%] %s | lat=%.1fs | equity=$%.0f | pos=%d | trades=%d",
                    i + 1, total, pct, status, lat, equity, len(positions), len(all_trades),
                )

            # Intermediate report every 500 cycles (~12h market data)
            if (i + 1) % 500 == 0:
                n_trades = len(all_trades)
                total_pnl = sum(t.pnl_usd for t in all_trades)
                n_win = sum(1 for t in all_trades if t.pnl_usd > 0)
                wr = n_win / n_trades * 100 if n_trades > 0 else 0
                syms = {}
                for t in all_trades:
                    syms[t.symbol] = syms.get(t.symbol, 0) + 1
                reasons = {}
                for t in all_trades:
                    reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
                logger.info(
                    "\n" + "=" * 50 +
                    f"\n  INTERMEDIATE REPORT @ cycle {i+1}/{total}"
                    f"\n  Equity: ${equity:,.2f} | P&L: ${total_pnl:+,.2f} ({total_pnl/capital*100:+.1f}%)"
                    f"\n  Trades: {n_trades} | WR: {wr:.0f}%"
                    f"\n  By symbol: {syms}"
                    f"\n  By exit: {reasons}"
                    f"\n  Open positions: {len(positions)}"
                    f"\n" + "=" * 50
                )

    # Force-close remaining
    if positions:
        last_ts = timestamps[-1] if timestamps else pd.Timestamp.now()
        logger.info("Force-closing %d positions at end", len(positions))
        for p in positions:
            try:
                df = filter_closed_candles(data_sources[(p.symbol, "15m")], 15, last_ts)
                ep = float(df.iloc[-1]["close"])
            except (ValueError, KeyError):
                ep = p.entry_price
            ef = p.qty * ep * fee_rate
            pnl = p.qty * (ep - p.entry_price) - p.entry_fee_usd - ef
            pnl_pct = pnl / p.size_usd * 100 if p.size_usd > 0 else 0
            dur = int((last_ts - p.entry_time).total_seconds() / 60)
            all_trades.append(Trade(
                p.symbol, p.entry_price, p.entry_time, ep, last_ts,
                p.qty, p.size_usd, pnl, pnl_pct, p.entry_fee_usd, ef,
                "end_of_backtest", dur,
            ))

    return all_trades, equity_curve, funnel


# ---------------------------------------------------------------------------
# Replay mode (legacy — reads from JSONL)
# ---------------------------------------------------------------------------

def run_backtest_replay(
    data_sources: dict, jsonl_path: str,
    capital: float, fee_rate: float, risk_pct: float,
) -> Tuple[List[Trade], List[dict], dict]:
    """Replay decisions from existing JSONL without calling Qwen."""
    decisions = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec.get("success") and rec.get("parsed"):
                decisions[rec["as_of"]] = rec
    logger.info("Loaded %d valid decisions from %s", len(decisions), jsonl_path)

    all_trades: List[Trade] = []
    equity_curve: List[dict] = []
    positions: List[Position] = []
    cash = capital
    funnel = {
        "total_buys": 0, "filtered_already_in_pos": 0,
        "filtered_insufficient_cash": 0, "filtered_no_data": 0,
        "filtered_no_atr": 0, "filtered_no_multipliers": 0,
        "filtered_invalid_stop": 0, "executed": 0,
    }

    for ts_str in sorted(decisions):
        as_of = pd.Timestamp(ts_str)
        rec = decisions[ts_str]
        parsed = rec["parsed"]
        snapshot = rec.get("features_snapshot", {})
        qwen_assets = parsed.get("assets", [])

        closed, positions = process_exits(positions, data_sources, qwen_assets, as_of, fee_rate)
        for t in closed:
            all_trades.append(t)
            cash += t.qty * t.exit_price - t.exit_fee_usd

        new_pos, cash = process_entries(
            qwen_assets, snapshot, data_sources, positions,
            capital, cash, as_of, fee_rate, risk_pct, funnel,
        )
        positions.extend(new_pos)

        unrealized = 0.0
        for p in positions:
            try:
                df = filter_closed_candles(data_sources[(p.symbol, "15m")], 15, as_of)
                unrealized += p.qty * (float(df.iloc[-1]["close"]) - p.entry_price)
            except (ValueError, KeyError):
                pass
        equity = cash + sum(p.size_usd for p in positions) + unrealized
        equity_curve.append({"timestamp": ts_str, "equity": round(equity, 2),
                             "cash": round(cash, 2), "open_positions": len(positions)})

    # Force-close
    if positions:
        last_ts = pd.Timestamp(sorted(decisions)[-1])
        for p in positions:
            try:
                df = filter_closed_candles(data_sources[(p.symbol, "15m")], 15, last_ts)
                ep = float(df.iloc[-1]["close"])
            except (ValueError, KeyError):
                ep = p.entry_price
            ef = p.qty * ep * fee_rate
            pnl = p.qty * (ep - p.entry_price) - p.entry_fee_usd - ef
            pnl_pct = pnl / p.size_usd * 100 if p.size_usd > 0 else 0
            dur = int((last_ts - p.entry_time).total_seconds() / 60)
            all_trades.append(Trade(
                p.symbol, p.entry_price, p.entry_time, ep, last_ts,
                p.qty, p.size_usd, pnl, pnl_pct, p.entry_fee_usd, ef,
                "end_of_backtest", dur,
            ))

    return all_trades, equity_curve, funnel


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def compute_report(trades: List[Trade], equity_curve: List[dict],
                   data_sources: dict, capital: float, funnel: dict) -> dict:
    report: Dict[str, Any] = {"capital_initial": capital, "total_trades": len(trades)}
    if not trades:
        report["error"] = "No trades"
        return report

    total_pnl = sum(t.pnl_usd for t in trades)
    report["pnl_total_usd"] = round(total_pnl, 2)
    report["pnl_total_pct"] = round(total_pnl / capital * 100, 2)
    report["equity_final"] = equity_curve[-1]["equity"] if equity_curve else capital + total_pnl

    winners = [t for t in trades if t.pnl_usd > 0]
    losers = [t for t in trades if t.pnl_usd <= 0]
    report["win_rate_pct"] = round(len(winners) / len(trades) * 100, 1)
    report["winners"] = len(winners)
    report["losers"] = len(losers)

    gp = sum(t.pnl_usd for t in winners) if winners else 0
    gl = abs(sum(t.pnl_usd for t in losers)) if losers else 0
    report["profit_factor"] = round(gp / gl, 2) if gl > 0 else float("inf")
    report["avg_pnl_usd"] = round(total_pnl / len(trades), 2)
    report["avg_pnl_pct"] = round(sum(t.pnl_pct for t in trades) / len(trades), 2)

    durs = [t.duration_minutes for t in trades]
    report["avg_duration_min"] = round(np.mean(durs), 1)
    report["total_fees_usd"] = round(sum(t.entry_fee_usd + t.exit_fee_usd for t in trades), 2)

    if equity_curve:
        eqs = [e["equity"] for e in equity_curve]
        peak, max_dd = eqs[0], 0
        for eq in eqs:
            if eq > peak: peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd: max_dd = dd
        report["max_drawdown_pct"] = round(max_dd, 2)

    if len(equity_curve) > 1:
        eqs = np.array([e["equity"] for e in equity_curve])
        rets = np.diff(eqs) / eqs[:-1]
        if np.std(rets) > 0:
            report["sharpe_ann"] = round(np.mean(rets) / np.std(rets) * np.sqrt(4 * 14 * 252), 2)

    # By exit reason
    reasons = {}
    for t in trades:
        r = t.exit_reason
        reasons.setdefault(r, {"count": 0, "pnl_usd": 0, "winners": 0})
        reasons[r]["count"] += 1
        reasons[r]["pnl_usd"] += t.pnl_usd
        if t.pnl_usd > 0: reasons[r]["winners"] += 1
    for r in reasons:
        reasons[r]["pnl_usd"] = round(reasons[r]["pnl_usd"], 2)
        reasons[r]["wr"] = round(reasons[r]["winners"] / reasons[r]["count"] * 100, 1)
    report["by_exit_reason"] = reasons

    # By symbol
    syms = {}
    for t in trades:
        syms.setdefault(t.symbol, {"count": 0, "pnl_usd": 0, "winners": 0})
        syms[t.symbol]["count"] += 1
        syms[t.symbol]["pnl_usd"] += t.pnl_usd
        if t.pnl_usd > 0: syms[t.symbol]["winners"] += 1
    for s in syms:
        syms[s]["pnl_usd"] = round(syms[s]["pnl_usd"], 2)
        syms[s]["wr"] = round(syms[s]["winners"] / syms[s]["count"] * 100, 1)
    report["by_symbol"] = syms

    report["execution_funnel"] = funnel

    # Buy & Hold
    if equity_curve:
        t0 = pd.Timestamp(equity_curve[0]["timestamp"])
        tN = pd.Timestamp(equity_curve[-1]["timestamp"])
        bh = 0.0
        alloc = capital / 5
        for sym in SYMBOLS:
            try:
                d0 = filter_closed_candles(data_sources[(sym, "15m")], 15, t0)
                dN = filter_closed_candles(data_sources[(sym, "15m")], 15, tN)
                bh += alloc * (float(dN.iloc[-1]["close"]) - float(d0.iloc[-1]["close"])) / float(d0.iloc[-1]["close"])
            except (ValueError, KeyError):
                pass
        report["buy_hold_usd"] = round(bh, 2)
        report["buy_hold_pct"] = round(bh / capital * 100, 2)

    return report


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_outputs(trades: List[Trade], equity_curve: List[dict],
                  report: dict, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # trades.csv
    if trades:
        with open(out / "trades.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["symbol","entry_price","entry_time","exit_price","exit_time",
                         "qty","size_usd","pnl_usd","pnl_pct","exit_reason","duration_min"])
            for t in trades:
                w.writerow([t.symbol, f"{t.entry_price:.6f}", t.entry_time,
                            f"{t.exit_price:.6f}", t.exit_time, f"{t.qty:.8f}",
                            f"{t.size_usd:.2f}", f"{t.pnl_usd:.2f}", f"{t.pnl_pct:.2f}",
                            t.exit_reason, t.duration_minutes])

    # equity_curve.csv
    if equity_curve:
        with open(out / "equity_curve.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=equity_curve[0].keys())
            w.writeheader()
            w.writerows(equity_curve)

    # report.json
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # summary.txt
    with open(out / "summary.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n  BACKTEST SUMMARY\n" + "=" * 60 + "\n\n")
        f.write(f"  Capital:           ${report.get('capital_initial',0):,.2f}\n")
        f.write(f"  Final equity:      ${report.get('equity_final',0):,.2f}\n")
        f.write(f"  P&L:               ${report.get('pnl_total_usd',0):,.2f} ({report.get('pnl_total_pct',0):.1f}%)\n")
        f.write(f"  Total trades:      {report.get('total_trades',0)}\n")
        f.write(f"  Win rate:          {report.get('win_rate_pct',0):.1f}%\n")
        f.write(f"  Profit factor:     {report.get('profit_factor',0):.2f}\n")
        f.write(f"  Avg P&L/trade:     ${report.get('avg_pnl_usd',0):.2f} ({report.get('avg_pnl_pct',0):.2f}%)\n")
        f.write(f"  Avg duration:      {report.get('avg_duration_min',0):.0f} min\n")
        f.write(f"  Max drawdown:      {report.get('max_drawdown_pct',0):.1f}%\n")
        f.write(f"  Total fees:        ${report.get('total_fees_usd',0):.2f}\n")
        if "sharpe_ann" in report:
            f.write(f"  Sharpe (ann.):     {report['sharpe_ann']:.2f}\n")
        if "buy_hold_usd" in report:
            f.write(f"\n  Buy & Hold:        ${report['buy_hold_usd']:,.2f} ({report.get('buy_hold_pct',0):.1f}%)\n")

        for section, label in [("by_exit_reason", "By exit reason"), ("by_symbol", "By symbol")]:
            data = report.get(section, {})
            if data:
                f.write(f"\n  {label}:\n")
                for k, v in sorted(data.items()):
                    f.write(f"    {k:15s}: {v['count']:3d} trades, ${v['pnl_usd']:8.2f}, WR={v['wr']:.0f}%\n")

        ef = report.get("execution_funnel", {})
        if ef:
            f.write(f"\n  Execution funnel:\n")
            for k, v in ef.items():
                f.write(f"    {k:30s}: {v}\n")
        f.write("\n" + "=" * 60 + "\n")

    # Print
    with open(out / "summary.txt") as f:
        print(f.read())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backtest engine v2")
    parser.add_argument("--mode", choices=["live", "replay"], default="live")
    parser.add_argument("--jsonl", help="JSONL path for replay mode")
    parser.add_argument("--resume", help="JSONL path to resume live mode from")
    parser.add_argument("--data-dir", default="src/data_trad")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--risk-pct", type=float, default=0.20)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--prompt", default="gemma_system_v6.txt")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    logger.info("Loading OHLCV data from %s ...", args.data_dir)
    data_sources = load_all_data(args.data_dir)
    logger.info("Loaded %d datasets.", len(data_sources))

    if args.mode == "replay":
        if not args.jsonl:
            logger.error("--jsonl required for replay mode")
            sys.exit(1)
        trades, eq, funnel = run_backtest_replay(
            data_sources, args.jsonl, args.capital, args.fee_rate, args.risk_pct,
        )
    else:
        # Live mode
        if not ping_ollama():
            logger.error("Ollama not reachable. Start with: ollama serve")
            sys.exit(1)

        system_prompt = load_system_prompt(args.prompt)
        logger.info("Prompt loaded: %s (%d chars)", args.prompt, len(system_prompt))

        # Determine timestamps
        if args.resume:
            last_as_of, _ = load_resume_state(args.resume)
            logger.info("Resuming from %s", last_as_of)
            btc = data_sources[("BTCUSDT", "15m")]
            start = btc.index[0].strftime("%Y-%m-%d")
            end = btc.index[-1].strftime("%Y-%m-%d")
            if args.start: start = args.start
            if args.end: end = args.end
            all_ts = generate_timestamps(data_sources, start, end)
            timestamps = [ts for ts in all_ts if str(ts) > last_as_of]
            log_path = Path(args.resume)
        else:
            btc = data_sources[("BTCUSDT", "15m")]
            start = args.start or (btc.index[-1] - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            end = args.end or btc.index[-1].strftime("%Y-%m-%d")
            timestamps = generate_timestamps(data_sources, start, end)
            log_path = Path("logs") / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            log_path.parent.mkdir(exist_ok=True)

        logger.info("%d cycles: %s → %s", len(timestamps),
                    timestamps[0] if timestamps else "?", timestamps[-1] if timestamps else "?")

        trades, eq, funnel = run_backtest_live(
            data_sources, timestamps, args.capital, args.fee_rate, args.risk_pct,
            args.model, system_prompt, log_path,
        )

    report = compute_report(trades, eq, data_sources, args.capital, funnel)
    write_outputs(trades, eq, report, args.output_dir)


if __name__ == "__main__":
    main()
