"""
Post-mortem report generator for backtest results.

Produces a structured markdown report (~15-25k tokens) designed to be
analyzed by Claude for prompt improvement recommendations.

Usage:
    python -m src.postmortem_reporter --jsonl logs/backtest_XXX.jsonl --output-dir logs/
    (also called automatically at end of backtest_engine.py)
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_backtest_data(
    jsonl_path: str,
    trades_csv: str = "results/trades.csv",
    report_json: str = "results/report.json",
    equity_csv: str = "results/equity_curve.csv",
) -> Dict[str, Any]:
    """Load all backtest artifacts. Link trades to their JSONL cycles."""

    # Cycles from JSONL
    cycles = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cycles.append(json.loads(line))

    # Trades
    trades = []
    tp = Path(trades_csv)
    if tp.exists():
        with open(tp, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for k in ["entry_price", "exit_price", "qty", "size_usd", "pnl_usd", "pnl_pct"]:
                    if k in row:
                        row[k] = float(row[k])
                if "duration_min" in row:
                    row["duration_min"] = int(row["duration_min"])
                trades.append(row)

    # Report
    report = {}
    rp = Path(report_json)
    if rp.exists():
        with open(rp, "r", encoding="utf-8") as f:
            report = json.load(f)

    # Equity curve
    equity = []
    ep = Path(equity_csv)
    if ep.exists():
        with open(ep, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                equity.append(row)

    # Link trades to cycles by entry_time matching as_of
    cycle_by_time = {}
    for c in cycles:
        cycle_by_time[c.get("as_of", "")] = c

    for t in trades:
        entry_time = t.get("entry_time", "")
        t["_cycle"] = cycle_by_time.get(entry_time)

    return {
        "cycles": cycles,
        "trades": trades,
        "report": report,
        "equity": equity,
        "jsonl_path": jsonl_path,
    }


# ---------------------------------------------------------------------------
# Thinking excerpt extraction
# ---------------------------------------------------------------------------

_DECISION_KEYWORDS = re.compile(
    r'\b(buy|sell|skip|hold|close|stop|target|support|resistance|trend|'
    r'breakout|pullback|reversion|bearish|bullish|momentum|volume|'
    r'risk|conviction|entry|exit|profit|loss|overbought|oversold|'
    r'divergence|confirmation|weakness|strength)\b',
    re.IGNORECASE,
)


def extract_thinking_excerpt(thinking: str, max_words: int = 150) -> str:
    """Extract the most relevant portion of thinking.

    Strategy:
    - Always keep first sentence (context)
    - Always keep last 2 sentences (conclusion/decision)
    - Score middle sentences by decision keywords
    - Trim middle to fit max_words, never remove start/end
    """
    if not thinking or not thinking.strip():
        return "(no thinking)"

    sentences = re.split(r'(?<=[.!?])\s+', thinking.strip())
    if len(sentences) <= 3:
        text = " ".join(sentences)
        words = text.split()
        return " ".join(words[:max_words]) if len(words) > max_words else text

    first = sentences[0]
    last_two = sentences[-2:]
    middle = sentences[1:-2]

    # Score middle sentences
    scored = []
    for s in middle:
        score = len(_DECISION_KEYWORDS.findall(s))
        scored.append((score, s))
    scored.sort(key=lambda x: -x[0])

    # Build excerpt within word budget
    fixed = first + " " + " ".join(last_two)
    fixed_words = len(fixed.split())
    remaining = max_words - fixed_words

    selected_middle = []
    for score, s in scored:
        s_words = len(s.split())
        if remaining >= s_words:
            selected_middle.append(s)
            remaining -= s_words

    result = first + " " + " ".join(selected_middle) + " " + " ".join(last_two)
    return result.strip()


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------


def section_context(data: Dict) -> str:
    """Section 1: Context."""
    cycles = data["cycles"]
    report = data["report"]

    first_ts = cycles[0].get("as_of", "?") if cycles else "?"
    last_ts = cycles[-1].get("as_of", "?") if cycles else "?"
    n_cycles = len(cycles)
    n_success = sum(1 for c in cycles if c.get("success"))
    n_trades = len(data["trades"])
    latencies = [c.get("latency_sec", 0) for c in cycles if c.get("latency_sec")]
    wall_clock = sum(latencies)
    run_type = "short run" if n_trades < 30 else "full run"

    lines = [
        "## 1. Context",
        "",
        f"- **Period**: {first_ts} → {last_ts}",
        f"- **Cycles**: {n_cycles} ({n_success} successful, {n_cycles - n_success} failed)",
        f"- **Run type**: {run_type} ({n_trades} trades)",
        f"- **Model**: qwen3:8b",
        f"- **Prompt**: v6",
        f"- **Thinking**: enabled",
        f"- **Wall clock**: {wall_clock/3600:.1f} hours ({wall_clock:.0f}s)",
        f"- **Avg latency**: {np.mean(latencies):.1f}s" if latencies else "",
        "",
    ]
    return "\n".join(lines)


def section_summary(data: Dict) -> str:
    """Section 2: Executive Summary."""
    r = data["report"]
    trades = data["trades"]

    lines = [
        "## 2. Executive Summary",
        "",
        "| Metric | Value | vs Benchmark |",
        "|--------|-------|--------------|",
        f"| Final P&L | ${r.get('pnl_total_usd', 0):,.2f} ({r.get('pnl_total_pct', 0):.1f}%) | B&H: ${r.get('buy_hold_usd', 0):,.2f} ({r.get('buy_hold_pct', 0):.1f}%) |",
        f"| Sharpe (annualized) | {r.get('sharpe_ann', 'N/A')} | - |",
        f"| Max Drawdown | {r.get('max_drawdown_pct', 0):.1f}% | - |",
        f"| Win Rate | {r.get('win_rate_pct', 0):.1f}% | - |",
        f"| Profit Factor | {r.get('profit_factor', 0):.2f} | - |",
        f"| Avg P&L per trade | ${r.get('avg_pnl_usd', 0):.2f} ({r.get('avg_pnl_pct', 0):.2f}%) | - |",
        f"| Avg duration | {r.get('avg_duration_min', 0):.0f} min | - |",
        f"| Total fees | ${r.get('total_fees_usd', 0):.2f} ({r.get('total_fees_usd', 0) / r.get('capital_initial', 10000) * 100:.1f}% of capital) | - |",
        f"| Total trades | {r.get('total_trades', 0)} | - |",
        "",
    ]

    # Verdict paragraph
    pnl_pct = r.get("pnl_total_pct", 0)
    bh_pct = r.get("buy_hold_pct", 0)
    wr = r.get("win_rate_pct", 0)
    pf = r.get("profit_factor", 0)
    n = r.get("total_trades", 0)

    if pnl_pct > 0 and pnl_pct > bh_pct:
        verdict = f"**Positive result.** Strategy returned {pnl_pct:.1f}% vs B&H {bh_pct:.1f}%, outperforming by {pnl_pct - bh_pct:.1f}pp."
    elif pnl_pct > 0:
        verdict = f"**Positive but underperformed.** Strategy returned {pnl_pct:.1f}% but B&H returned {bh_pct:.1f}%."
    else:
        verdict = f"**Negative result.** Strategy lost {abs(pnl_pct):.1f}% while B&H returned {bh_pct:.1f}%."

    verdict += f" Win rate {wr:.0f}% with profit factor {pf:.2f} over {n} trades."
    lines.append(verdict)
    lines.append("")
    return "\n".join(lines)


def section_exit_reasons(data: Dict) -> str:
    """Section 3: Exit Reasons Breakdown."""
    trades = data["trades"]
    if not trades:
        return "## 3. Exit Reasons Breakdown\n\nNo trades.\n"

    reasons = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0, "durations": []})
    for t in trades:
        r = t.get("exit_reason", "unknown")
        reasons[r]["count"] += 1
        reasons[r]["pnl"] += t["pnl_usd"]
        if t["pnl_usd"] > 0:
            reasons[r]["wins"] += 1
        reasons[r]["durations"].append(t.get("duration_min", 0))

    lines = [
        "## 3. Exit Reasons Breakdown",
        "",
        "| Reason | Count | Total P&L | WR | Avg P&L | Avg Duration |",
        "|--------|-------|-----------|-----|---------|--------------|",
    ]
    for r in sorted(reasons):
        d = reasons[r]
        wr = d["wins"] / d["count"] * 100 if d["count"] > 0 else 0
        avg_pnl = d["pnl"] / d["count"]
        avg_dur = np.mean(d["durations"]) if d["durations"] else 0
        toxic = " ⚠️" if wr < 20 and d["count"] >= 3 else ""
        lines.append(
            f"| {r}{toxic} | {d['count']} | ${d['pnl']:.2f} | {wr:.0f}% | ${avg_pnl:.2f} | {avg_dur:.0f}min |"
        )
    lines.append("")
    return "\n".join(lines)


def section_per_asset(data: Dict) -> str:
    """Section 4: Per Asset Performance."""
    trades = data["trades"]
    if not trades:
        return "## 4. Per Asset Performance\n\nNo trades.\n"

    assets = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0, "convictions": [], "stops": [], "tps": []})
    for t in trades:
        s = t.get("symbol", "?")
        c = t.get("_cycle")
        assets[s]["count"] += 1
        assets[s]["pnl"] += t["pnl_usd"]
        if t["pnl_usd"] > 0:
            assets[s]["wins"] += 1
        # Extract conviction from cycle
        if c and c.get("parsed"):
            for a in c["parsed"].get("assets", []):
                if a.get("action") == "buy":
                    assets[s]["convictions"].append(a.get("conviction", 0))
                    if a.get("stop_mult"):
                        assets[s]["stops"].append(a["stop_mult"])
                    if a.get("tp_mult"):
                        assets[s]["tps"].append(a["tp_mult"])

    lines = [
        "## 4. Per Asset Performance",
        "",
        "| Asset | Trades | P&L | WR | Avg Conv | Most used stop | Most used tp |",
        "|-------|--------|-----|-----|----------|----------------|--------------|",
    ]
    for s in sorted(assets):
        d = assets[s]
        wr = d["wins"] / d["count"] * 100 if d["count"] > 0 else 0
        avg_conv = np.mean(d["convictions"]) if d["convictions"] else 0
        top_stop = Counter([round(x, 1) for x in d["stops"]]).most_common(1)
        top_tp = Counter([round(x, 1) for x in d["tps"]]).most_common(1)
        ts = f"{top_stop[0][0]}" if top_stop else "-"
        tt = f"{top_tp[0][0]}" if top_tp else "-"
        flag = " ⚠️" if d["pnl"] < 0 and d["count"] >= 5 else ""
        lines.append(f"| {s}{flag} | {d['count']} | ${d['pnl']:.2f} | {wr:.0f}% | {avg_conv:.1f} | {ts} | {tt} |")
    lines.append("")
    return "\n".join(lines)


def section_conviction(data: Dict) -> str:
    """Section 5: Conviction Analysis."""
    trades = data["trades"]
    if not trades:
        return "## 5. Conviction Analysis\n\nNo trades.\n"
    if len(trades) < 10:
        return f"## 5. Conviction Analysis\n\n*{len(trades)} trades — too few for meaningful conviction buckets. Skipped.*\n"

    buckets = {"1-3": [], "4-6": [], "7-8": [], "9-10": []}
    for t in trades:
        c = t.get("_cycle")
        conv = 5  # default
        if c and c.get("parsed"):
            sym = t.get("symbol", "")
            for a in c["parsed"].get("assets", []):
                # Match by resolving anon
                if a.get("action") == "buy":
                    conv = a.get("conviction", 5)
                    break
        if conv <= 3:
            buckets["1-3"].append(t)
        elif conv <= 6:
            buckets["4-6"].append(t)
        elif conv <= 8:
            buckets["7-8"].append(t)
        else:
            buckets["9-10"].append(t)

    lines = [
        "## 5. Conviction Analysis",
        "",
        "| Conviction | N trades | WR | Avg P&L | % of total |",
        "|------------|----------|-----|---------|------------|",
    ]
    total = len(trades)
    for bucket, ts in buckets.items():
        n = len(ts)
        if n == 0:
            lines.append(f"| {bucket} | 0 | - | - | 0% |")
            continue
        wins = sum(1 for t in ts if t["pnl_usd"] > 0)
        wr = wins / n * 100
        avg_pnl = sum(t["pnl_usd"] for t in ts) / n
        pct = n / total * 100
        lines.append(f"| {bucket} | {n} | {wr:.0f}% | ${avg_pnl:.2f} | {pct:.0f}% |")

    lines.append("")
    # Calibration check
    high_conv = buckets["7-8"] + buckets["9-10"]
    low_conv = buckets["1-3"] + buckets["4-6"]
    if high_conv and low_conv:
        wr_high = sum(1 for t in high_conv if t["pnl_usd"] > 0) / len(high_conv) * 100
        wr_low = sum(1 for t in low_conv if t["pnl_usd"] > 0) / len(low_conv) * 100
        if wr_high > wr_low:
            lines.append(f"**Calibration check**: High conviction WR ({wr_high:.0f}%) > Low conviction WR ({wr_low:.0f}%) — ✅ conviction is predictive.")
        else:
            lines.append(f"**Calibration check**: High conviction WR ({wr_high:.0f}%) ≤ Low conviction WR ({wr_low:.0f}%) — ❌ conviction is NOT predictive.")
    lines.append("")
    return "\n".join(lines)


def section_multipliers(data: Dict) -> str:
    """Section 6: ATR Multipliers Effectiveness."""
    if len(data["trades"]) < 10:
        return f"## 6. ATR Multipliers Effectiveness\n\n*{len(data['trades'])} trades — too few for multiplier analysis. Skipped.*\n"
    trades = data["trades"]
    if not trades:
        return "## 6. ATR Multipliers Effectiveness\n\nNo trades.\n"

    combos = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    for t in trades:
        c = t.get("_cycle")
        if not c or not c.get("parsed"):
            continue
        for a in c["parsed"].get("assets", []):
            if a.get("action") == "buy" and a.get("stop_mult") and a.get("tp_mult"):
                key = (round(a["stop_mult"], 1), round(a["tp_mult"], 1))
                combos[key]["count"] += 1
                combos[key]["pnl"] += t["pnl_usd"]
                if t["pnl_usd"] > 0:
                    combos[key]["wins"] += 1
                break

    lines = [
        "## 6. ATR Multipliers Effectiveness",
        "",
        "| (stop, tp) | N | WR | Total P&L | Avg P&L |",
        "|------------|---|-----|-----------|---------|",
    ]
    for key in sorted(combos, key=lambda k: -combos[k]["count"]):
        d = combos[key]
        wr = d["wins"] / d["count"] * 100 if d["count"] > 0 else 0
        avg = d["pnl"] / d["count"]
        lines.append(f"| ({key[0]}, {key[1]}) | {d['count']} | {wr:.0f}% | ${d['pnl']:.2f} | ${avg:.2f} |")
    lines.append("")
    return "\n".join(lines)


def section_setup_types(data: Dict) -> str:
    """Section 7: Setup Types Distribution (extracted from rationales)."""
    trades = data["trades"]
    if not trades:
        return "## 7. Setup Types Distribution\n\nNo trades.\n"

    patterns = {
        "breakout": re.compile(r"breakout|break.*above|clear.*resistance", re.I),
        "pullback": re.compile(r"pullback|pull.*back|retest|bounce.*support|dip.*buy", re.I),
        "mean_reversion": re.compile(r"mean.?reversion|oversold.*bounce|bb.*lower|range.*entry", re.I),
    }

    setups = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
    for t in trades:
        c = t.get("_cycle")
        rationale = ""
        if c and c.get("parsed"):
            for a in c["parsed"].get("assets", []):
                if a.get("action") == "buy":
                    rationale = a.get("rationale", "")
                    break

        setup = "other"
        for name, pat in patterns.items():
            if pat.search(rationale):
                setup = name
                break
        setups[setup]["count"] += 1
        setups[setup]["pnl"] += t["pnl_usd"]
        if t["pnl_usd"] > 0:
            setups[setup]["wins"] += 1

    lines = [
        "## 7. Setup Types Distribution",
        "",
        "| Setup | N | WR | Avg P&L |",
        "|-------|---|-----|---------|",
    ]
    for s in sorted(setups):
        d = setups[s]
        wr = d["wins"] / d["count"] * 100 if d["count"] > 0 else 0
        avg = d["pnl"] / d["count"]
        lines.append(f"| {s} | {d['count']} | {wr:.0f}% | ${avg:.2f} |")
    lines.append("")
    return "\n".join(lines)


def section_timing(data: Dict) -> str:
    """Section 8: Timing Patterns."""
    trades = data["trades"]
    if not trades:
        return "## 8. Timing Patterns\n\nNo trades.\n"

    # Duration buckets
    buckets = {"0-30min": [], "30-60min": [], "1-2h": [], "2-4h": [], "4h+": []}
    for t in trades:
        d = t.get("duration_min", 0)
        if d <= 30:
            buckets["0-30min"].append(t)
        elif d <= 60:
            buckets["30-60min"].append(t)
        elif d <= 120:
            buckets["1-2h"].append(t)
        elif d <= 240:
            buckets["2-4h"].append(t)
        else:
            buckets["4h+"].append(t)

    lines = [
        "## 8. Timing Patterns",
        "",
        "| Duration | N trades | WR | Avg P&L |",
        "|----------|----------|-----|---------|",
    ]
    for bucket, ts in buckets.items():
        n = len(ts)
        if n == 0:
            continue
        wins = sum(1 for t in ts if t["pnl_usd"] > 0)
        wr = wins / n * 100
        avg = sum(t["pnl_usd"] for t in ts) / n
        lines.append(f"| {bucket} | {n} | {wr:.0f}% | ${avg:.2f} |")

    # Winners vs losers duration
    winners = [t for t in trades if t["pnl_usd"] > 0]
    losers = [t for t in trades if t["pnl_usd"] <= 0]
    if winners and losers:
        lines.append("")
        avg_w = np.mean([t.get("duration_min", 0) for t in winners])
        avg_l = np.mean([t.get("duration_min", 0) for t in losers])
        lines.append(f"Avg duration — winners: {avg_w:.0f}min, losers: {avg_l:.0f}min")
    lines.append("")
    return "\n".join(lines)


def section_qwen_close(data: Dict) -> str:
    """Section 9: Qwen Close Deep Dive."""
    trades = data["trades"]
    qc = [t for t in trades if t.get("exit_reason") == "qwen_close"]
    if not qc:
        return "## 9. Qwen Close Deep Dive\n\nNo qwen_close trades.\n"

    lines = ["## 9. Qwen Close Deep Dive", "", f"{len(qc)} trades closed by Qwen:", ""]
    for i, t in enumerate(qc[:10]):
        lines.append(f"### qwen_close #{i+1} — {t.get('symbol', '?')}")
        lines.append(f"- Entry: {t.get('entry_time')}, ${t['entry_price']:.2f}")
        lines.append(f"- Exit: {t.get('exit_time')}, ${t['exit_price']:.2f}")
        lines.append(f"- P&L: ${t['pnl_usd']:.2f} ({t['pnl_pct']:.2f}%), duration: {t.get('duration_min', 0)}min")
        exit_time = t.get("exit_time", "")
        for c in data["cycles"]:
            if c.get("as_of") == exit_time and c.get("thinking"):
                lines.append(f"\n> **Thinking at close**: {extract_thinking_excerpt(c['thinking'], 150)}")
                break
        lines.append("")
    return "\n".join(lines)


def _format_trade_detail(t: Dict, data: Dict, rank: int) -> str:
    """Format one trade for top 10 sections."""
    lines = [f"### Trade #{rank} — {t.get('symbol', '?')}"]
    lines.append(f"- Entry: {t.get('entry_time')}, ${t['entry_price']:.4f}")
    lines.append(f"- Exit: {t.get('exit_time')}, ${t['exit_price']:.4f} ({t.get('exit_reason', '?')})")
    lines.append(f"- P&L: ${t['pnl_usd']:.2f} ({t['pnl_pct']:.2f}%) over {t.get('duration_min', 0)}min")
    c = t.get("_cycle")
    if c and c.get("parsed"):
        for a in c["parsed"].get("assets", []):
            if a.get("action") == "buy":
                lines.append(f"- Conviction: {a.get('conviction')}, Stop: {a.get('stop_mult')}, TP: {a.get('tp_mult')}")
                lines.append(f"\n> **Rationale**: {a.get('rationale', 'N/A')}")
                break
    if c and c.get("thinking"):
        lines.append(f"\n> **Thinking**: {extract_thinking_excerpt(c['thinking'], 150)}")
    lines.append("")
    return "\n".join(lines)


def section_top_trades(data: Dict, best: bool = True) -> str:
    """Section 10/11: Top 10 Best or Worst trades."""
    trades = data["trades"]
    if not trades:
        n = "10" if best else "11"
        return f"## {n}. Top 10 {'Best' if best else 'Worst'} Trades\n\nNo trades.\n"
    sorted_t = sorted(trades, key=lambda t: t["pnl_usd"], reverse=best)
    n = "10" if best else "11"
    label = "Best" if best else "Worst"
    lines = [f"## {n}. Top 10 {label} Trades", ""]
    for i, t in enumerate(sorted_t[:10]):
        lines.append(_format_trade_detail(t, data, i + 1))
    return "\n".join(lines)


def section_patterns(data: Dict) -> str:
    """Section 12: Recurring Patterns (auto-detected)."""
    trades = data["trades"]
    cycles = data["cycles"]
    patterns = []

    stops = [t for t in trades if t.get("exit_reason") == "stop_loss"]
    if stops:
        fast = [t for t in stops if t.get("duration_min", 999) < 30]
        pct = len(fast) / len(stops) * 100
        if pct > 30:
            patterns.append(f"⚠️ {pct:.0f}% of stops hit in < 30min ({len(fast)}/{len(stops)}). Stops too tight?")

    tps = [t for t in trades if t.get("exit_reason") == "take_profit"]
    if tps:
        slow = [t for t in tps if t.get("duration_min", 0) > 240]
        pct = len(slow) / len(tps) * 100
        if pct > 50:
            patterns.append(f"⚠️ {pct:.0f}% of TPs took > 4h ({len(slow)}/{len(tps)}). TPs too ambitious?")

    assets = defaultdict(lambda: {"n": 0, "pnl": 0})
    for t in trades:
        assets[t.get("symbol", "?")]["n"] += 1
        assets[t.get("symbol", "?")]["pnl"] += t["pnl_usd"]
    for s, d in assets.items():
        if d["n"] >= 5 and d["pnl"] < 0:
            patterns.append(f"⚠️ {s} systematically losing: {d['n']} trades, ${d['pnl']:.2f}.")

    uncertain_high = 0
    for c in cycles:
        th = (c.get("thinking") or "").lower()
        parsed = c.get("parsed") or {}
        if any(w in th for w in ["uncertain", "cautious", "unclear", "no clear"]):
            for a in parsed.get("assets", []):
                if a.get("conviction", 0) >= 7 and a.get("action") == "buy":
                    uncertain_high += 1
    if uncertain_high:
        patterns.append(f"⚠️ Thinking uncertain but conviction ≥ 7 in {uncertain_high} buys. Overconfidence?")

    lines = ["## 12. Recurring Patterns", ""]
    if patterns:
        for p in patterns:
            lines.append(f"- {p}")
    else:
        lines.append("No significant patterns detected.")
    lines.append("")
    return "\n".join(lines)


def section_thinking_sample(data: Dict) -> str:
    """Section 13: 5 random thinking samples."""
    cycles = [c for c in data["cycles"] if c.get("thinking") and c.get("success")]
    if not cycles:
        return "## 13. Thinking Quality Sample\n\nNo thinking data.\n"
    rng = np.random.RandomState(42)
    n = min(5, len(cycles))
    indices = rng.choice(len(cycles), size=n, replace=False)
    lines = ["## 13. Thinking Quality Sample", ""]
    for idx in sorted(indices):
        c = cycles[idx]
        lines.append(f"### Cycle {c.get('as_of', '?')}")
        th = c.get("thinking", "")
        words = th.split()
        if len(words) > 300:
            lines.append(f"\n> {' '.join(words[:300])} [...]")
        else:
            lines.append(f"\n> {th}")
        if c.get("parsed"):
            lines.append("\n**Decisions**:")
            for a in c["parsed"].get("assets", []):
                lines.append(f"- {a.get('symbol')}: {a.get('action')} (conv={a.get('conviction')})")
        lines.append("")
    return "\n".join(lines)


def section_open_questions(data: Dict) -> str:
    """Section 14: Open Questions."""
    r = data["report"]
    trades = data["trades"]
    questions = []
    by_sym = r.get("by_symbol", {})
    for s, d in by_sym.items():
        if d.get("pnl_usd", 0) < -50 and d.get("count", 0) >= 5:
            questions.append(f"Why does {s} systematically underperform ({d['count']} trades, ${d['pnl_usd']:.0f})?")
    if any(c.get("thinking") for c in data["cycles"]):
        questions.append("Is the thinking genuinely exploited, or is it superficial?")
    stops = [t for t in trades if t.get("exit_reason") == "stop_loss"]
    tps = [t for t in trades if t.get("exit_reason") == "take_profit"]
    if stops and tps:
        questions.append(f"Are ATR multipliers well-calibrated? ({len(stops)} stops vs {len(tps)} TPs)")
    wr = r.get("win_rate_pct", 0)
    if wr < 40:
        questions.append(f"Win rate is {wr:.0f}%. Is Qwen entering with insufficient edge?")
    questions.append("Are there market regimes where Qwen performs notably better or worse?")

    lines = ["## 14. Open Questions for Analysis", ""]
    for i, q in enumerate(questions[:5], 1):
        lines.append(f"{i}. {q}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def generate_report(
    jsonl_path: str,
    trades_csv: str = "results/trades.csv",
    report_json: str = "results/report.json",
    equity_csv: str = "results/equity_curve.csv",
    output_dir: str = "logs",
    prompt_version: str = "v6",
) -> str:
    """Generate full post-mortem markdown."""
    data = load_backtest_data(jsonl_path, trades_csv, report_json, equity_csv)

    sections = [
        "# Backtest Post-Mortem\n",
        section_context(data),
        section_summary(data),
        section_exit_reasons(data),
        section_per_asset(data),
        section_conviction(data),
        section_multipliers(data),
        section_setup_types(data),
        section_timing(data),
        section_qwen_close(data),
        section_top_trades(data, best=True),
        section_top_trades(data, best=False),
        section_patterns(data),
        section_thinking_sample(data),
        section_open_questions(data),
    ]

    report_md = "\n".join(sections)
    if len(report_md) > 100000:
        report_md = report_md[:100000] + "\n\n[... truncated ...]\n"

    # Compute duration label from data
    cycles = data["cycles"]
    if len(cycles) >= 2:
        first = cycles[0].get("as_of", "")
        last = cycles[-1].get("as_of", "")
        try:
            t0 = pd.Timestamp(first)
            t1 = pd.Timestamp(last)
            delta_days = (t1 - t0).days
            if delta_days <= 1:
                duration = "1day"
            elif delta_days <= 7:
                duration = f"{delta_days}days"
            else:
                duration = f"{delta_days // 30}month" if delta_days >= 28 else f"{delta_days}days"
        except Exception:
            duration = "unknown"
    else:
        duration = "unknown"

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"postmortem_{ts}_{prompt_version}_{duration}.md"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate post-mortem report")
    parser.add_argument("--jsonl", required=True, help="Backtest JSONL path")
    parser.add_argument("--trades-csv", default="results/trades.csv")
    parser.add_argument("--report-json", default="results/report.json")
    parser.add_argument("--equity-csv", default="results/equity_curve.csv")
    parser.add_argument("--output-dir", default="logs")
    parser.add_argument("--prompt-version", default="v6", help="Prompt version for filename (e.g. v6, v7)")
    args = parser.parse_args()
    path = generate_report(
        args.jsonl, args.trades_csv, args.report_json, args.equity_csv,
        args.output_dir, args.prompt_version,
    )
    print(f"Post-mortem written to: {path}")


if __name__ == "__main__":
    main()
