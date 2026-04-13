"""
Dry-run test: features → Gemma → parse → log.

No order execution, no position management. Just tests the full pipeline
over 2-3 days of historical data and logs every cycle to JSONL.

Usage:
    python src/test_run.py                          # last 3 days
    python src/test_run.py --days 1                 # last 1 day
    python src/test_run.py --start 2025-09-01 --end 2025-09-03
    python src/test_run.py --data-dir src/data_trad
"""

import argparse
import json
import logging
import signal
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.feature_engineering import (
    SYMBOLS,
    anonymize_and_format,
    compute_features,
    load_all_data,
)
from src.llm_client import (
    call_gemma,
    load_system_prompt,
    ping_ollama,
)

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
    "BTCUSDT": "ASSET_A",
    "ETHUSDT": "ASSET_B",
    "SOLUSDT": "ASSET_C",
    "XRPUSDT": "ASSET_D",
    "BNBUSDT": "ASSET_E",
}

FAKE_CONTEXT = {
    "positions_open_count": 0,
    "total_exposure_pct": 0.0,
    "btc_dominance": {"value": 54.0, "chg_24h_pct": 0.0},
    "funding_rates": {sym: 0.0 for sym in SYMBOLS},
}

SESSION_START_HOUR = 8
SESSION_END_HOUR = 22

# ---------------------------------------------------------------------------
# Timestamp generation
# ---------------------------------------------------------------------------


def generate_timestamps(
    data_sources: dict,
    days: int = 3,
    start: str = None,
    end: str = None,
) -> list:
    """Generate 15-min timestamps between 08:00 and 22:00 UTC.

    By default: last `days` days of available data.
    """
    btc_15m = data_sources[("BTCUSDT", "15m")]
    data_end = btc_15m.index[-1]

    if start and end:
        t_start = pd.Timestamp(start)
        t_end = pd.Timestamp(end)
    elif start:
        t_start = pd.Timestamp(start)
        t_end = t_start + pd.Timedelta(days=days)
    else:
        # Last N days of data, but need enough closed candles
        t_end = data_end
        t_start = t_end - pd.Timedelta(days=days)

    # Generate every 15 min
    all_ts = pd.date_range(t_start, t_end, freq="15min")

    # Keep only 08:00 to 22:00 (exclusive) — these are the as_of timestamps
    # At 08:00, the 07:45 bar just closed. At 22:00, the 21:45 bar just closed.
    filtered = [
        ts for ts in all_ts
        if SESSION_START_HOUR <= ts.hour < SESSION_END_HOUR
    ]

    # Ensure we have data for each timestamp (need at least 24h of 15m before)
    data_start = btc_15m.index[0] + pd.Timedelta(hours=24)
    filtered = [ts for ts in filtered if ts >= data_start and ts <= data_end + pd.Timedelta(minutes=15)]

    return filtered


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------


def compute_stats(records: list) -> dict:
    """Compute summary statistics from logged records."""
    total = len(records)
    if total == 0:
        return {"total_cycles": 0}

    successes_first = sum(1 for r in records if r.get("success") and not r.get("retried"))
    successes_retry = sum(1 for r in records if r.get("success") and r.get("retried"))
    failures = sum(1 for r in records if not r.get("success"))

    latencies = [r["latency_sec"] for r in records if r.get("latency_sec")]

    # Action counts per asset
    action_counts = {}
    conviction_values = []
    for r in records:
        parsed = r.get("parsed")
        if not parsed:
            continue
        for asset in parsed.get("assets", []):
            sym = asset.get("symbol", "?")
            act = asset.get("action", "?")
            if sym not in action_counts:
                action_counts[sym] = Counter()
            action_counts[sym][act] += 1
            if asset.get("conviction") is not None:
                conviction_values.append(asset["conviction"])

    # Conviction histogram
    conv_hist = Counter()
    for c in conviction_values:
        bucket = f"{(c // 3) * 3}-{min((c // 3) * 3 + 2, 10)}"
        conv_hist[bucket] += 1

    stats = {
        "total_cycles": total,
        "valid_first_attempt": successes_first,
        "valid_first_attempt_pct": round(successes_first / total * 100, 1),
        "valid_after_retry": successes_retry,
        "valid_after_retry_pct": round(successes_retry / total * 100, 1),
        "failures": failures,
        "failures_pct": round(failures / total * 100, 1),
    }

    if latencies:
        stats["latency_mean_sec"] = round(np.mean(latencies), 2)
        stats["latency_p50_sec"] = round(np.percentile(latencies, 50), 2)
        stats["latency_p95_sec"] = round(np.percentile(latencies, 95), 2)

    stats["actions_per_asset"] = {
        sym: dict(counts) for sym, counts in sorted(action_counts.items())
    }
    stats["conviction_histogram"] = dict(sorted(conv_hist.items()))

    return stats


def print_stats(stats: dict):
    """Pretty-print run statistics."""
    print("\n" + "=" * 60)
    print("  TEST RUN STATISTICS")
    print("=" * 60)

    print(f"  Total cycles:            {stats['total_cycles']}")
    print(f"  Valid 1st attempt:       {stats.get('valid_first_attempt', 0)} ({stats.get('valid_first_attempt_pct', 0)}%)")
    print(f"  Valid after retry:       {stats.get('valid_after_retry', 0)} ({stats.get('valid_after_retry_pct', 0)}%)")
    print(f"  Failures:                {stats.get('failures', 0)} ({stats.get('failures_pct', 0)}%)")

    if "latency_mean_sec" in stats:
        print(f"\n  Latency mean:            {stats['latency_mean_sec']}s")
        print(f"  Latency p50:             {stats['latency_p50_sec']}s")
        print(f"  Latency p95:             {stats['latency_p95_sec']}s")

    if stats.get("actions_per_asset"):
        print("\n  Actions per asset:")
        for sym, counts in stats["actions_per_asset"].items():
            parts = [f"{act}={n}" for act, n in sorted(counts.items())]
            print(f"    {sym}: {', '.join(parts)}")

    if stats.get("conviction_histogram"):
        print("\n  Conviction distribution:")
        for bucket, count in stats["conviction_histogram"].items():
            bar = "#" * min(count, 40)
            print(f"    [{bucket:>5s}]: {count:4d} {bar}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Dry-run: features → Gemma → log")
    parser.add_argument("--days", type=int, default=3, help="Number of days to test (default: 3)")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--data-dir", type=str, default="src/data_trad", help="Data directory")
    parser.add_argument("--model", type=str, default="qwen3:8b", help="Ollama model name")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    args = parser.parse_args()

    # --- Setup logs directory ---
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    log_filename = f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    log_path = logs_dir / log_filename

    # --- Ping Ollama ---
    logger.info("Pinging Ollama at %s ...", "http://localhost:11434")
    if not ping_ollama():
        logger.error(
            "Ollama is not reachable at http://localhost:11434. "
            "Start it with: ollama serve"
        )
        sys.exit(1)
    logger.info("Ollama OK.")

    # --- Load data ---
    logger.info("Loading data from %s ...", args.data_dir)
    data_sources = load_all_data(args.data_dir)
    logger.info("Loaded %d datasets.", len(data_sources))

    # --- Load system prompt ---
    system_prompt = load_system_prompt()
    logger.info("System prompt loaded (%d chars).", len(system_prompt))

    # --- Generate timestamps ---
    timestamps = generate_timestamps(data_sources, args.days, args.start, args.end)
    logger.info("Generated %d cycles (%s → %s).", len(timestamps), timestamps[0], timestamps[-1])

    if not timestamps:
        logger.error("No valid timestamps in range.")
        sys.exit(1)

    # --- Ctrl+C handler ---
    records = []
    interrupted = False

    def signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        logger.warning("Ctrl+C received — flushing logs and printing partial stats...")

    signal.signal(signal.SIGINT, signal_handler)

    # --- Main loop ---
    logger.info("Starting dry-run. Logging to %s", log_path)

    with open(log_path, "a", encoding="utf-8") as f:
        pbar = tqdm(timestamps, desc="Cycles", unit="cycle")

        for as_of in pbar:
            if interrupted:
                break

            record = {
                "cycle_index": None,
                "as_of": str(as_of),
                "success": False,
                "retried": False,
                "latency_sec": 0.0,
                "parsed": None,
                "validation_errors": [],
                "raw_response_first_attempt": None,
                "raw_response_retry": None,
                "error": None,
            }

            try:
                # Compute features
                features = compute_features(data_sources, as_of, FAKE_CONTEXT)
                record["cycle_index"] = features.get("cycle_index")

                # Build features_snapshot for backtest (price + ATR per asset)
                snapshot = {}
                for asset_data in features.get("assets", []):
                    sym = asset_data.get("_symbol", "")
                    anon_id = ANON_MAPPING.get(sym, sym)
                    snapshot[anon_id] = {
                        "real_symbol": sym,
                        "price": asset_data.get("price"),
                        "atr_15m_abs": (asset_data.get("volatility") or {}).get("atr_15m_abs"),
                    }
                record["features_snapshot"] = snapshot

                # Anonymize
                anon_features = anonymize_and_format(features, ANON_MAPPING)

                # Call Gemma
                llm_result = call_gemma(
                    system_prompt,
                    anon_features,
                    temperature=args.temperature,
                    model=args.model,
                )

                record["success"] = llm_result["success"]
                record["retried"] = llm_result["retried"]
                record["latency_sec"] = llm_result["latency_sec"]
                record["parsed"] = llm_result["parsed"]
                record["validation_errors"] = llm_result["validation_errors"]
                record["raw_response_first_attempt"] = llm_result["raw_response_first_attempt"]
                record["raw_response_retry"] = llm_result["raw_response_retry"]
                record["thinking"] = llm_result.get("thinking")

                # Update progress bar
                status = "OK" if llm_result["success"] else "FAIL"
                pbar.set_postfix({"status": status, "latency": f"{llm_result['latency_sec']:.1f}s"})

            except Exception as e:
                record["error"] = str(e)
                logger.error("Cycle %s error: %s", as_of, e)
                pbar.set_postfix({"status": "ERR"})

            records.append(record)

            # Write JSONL line immediately (no buffering)
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()

    # --- Stats ---
    stats = compute_stats(records)
    print_stats(stats)

    # Save stats summary
    stats_path = log_path.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info("Stats saved to %s", stats_path)
    logger.info("Full log: %s (%d records)", log_path, len(records))


if __name__ == "__main__":
    main()
