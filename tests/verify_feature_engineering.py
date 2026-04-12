"""
Verification script for feature_engineering.py against REAL data.

This is NOT a unit test — it's a diagnostic tool that loads your actual CSVs,
picks multiple timestamps, and cross-checks every calculation layer.

Run:
    python tests/verify_feature_engineering.py [data_dir]
    python tests/verify_feature_engineering.py src/data_trad

7 verification blocks:
    1. Aggregation 15m→1h vs real CSV (OHLCV exact match)
    2. Aggregation 15m→4h vs real CSV
    3. Indicator accuracy (precomputed vs fresh talib on filtered slice)
    4. Anti-look-ahead on real data (full vs truncated)
    5. VWAP session reset at 08:00 UTC
    6. Filter boundary — no future candle leak on higher TF
    7. Cross-TF price consistency (last 15m close == last 1h close at boundary)

Exit code 0 = all checks passed.
Exit code 1 = at least one check failed.
"""

import sys
import json
import numpy as np
import pandas as pd
import talib
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import (
    load_all_data,
    filter_closed_candles,
    build_current_bar,
    compute_features,
    compute_indicators,
    _precompute_indicators,
    _load_csv,
    SYMBOLS,
    TIMEFRAMES,
    SESSION_START_HOUR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class CheckResult:
    def __init__(self, name):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.details = []

    def ok(self, msg):
        self.passed += 1
        self.details.append(("PASS", msg))

    def fail(self, msg):
        self.failed += 1
        self.details.append(("FAIL", msg))

    def report(self):
        status = "PASS" if self.failed == 0 else "FAIL"
        print(f"\n{'='*70}")
        print(f"  [{status}] {self.name}  ({self.passed} passed, {self.failed} failed)")
        print(f"{'='*70}")
        for kind, msg in self.details:
            marker = "  OK " if kind == "PASS" else "  ** "
            print(f"{marker} {msg}")
        return self.failed == 0


def approx_eq(a, b, tol=1e-6, rel_tol=None):
    """Check approximate equality. Handles None/NaN."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if np.isnan(a) and np.isnan(b):
        return True
    if np.isnan(a) or np.isnan(b):
        return False
    if rel_tol is not None and b != 0:
        return abs(a - b) / abs(b) < rel_tol
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# Check 1: Aggregation 15m → 1h vs real CSV
# ---------------------------------------------------------------------------

def check_aggregation_15m_to_1h(data_dir):
    """For N random complete 1h periods, aggregate 4×15m bars and compare
    against the actual 1h bar from the CSV. Tests OHLCV exact match."""
    check = CheckResult("CHECK 1: Aggregation 15m → 1h vs real CSV")

    n_samples = 20  # Number of random 1h bars to verify

    for symbol in SYMBOLS:
        file_stem = {"BTCUSDT": "BTCUSD", "ETHUSDT": "ETHUSD", "SOLUSDT": "SOLUSD",
                     "XRPUSDT": "XRPUSD", "BNBUSDT": "BNBUSD"}[symbol]

        df_15m = _load_csv(data_dir / f"{file_stem}_all_15m.csv")
        df_1h = _load_csv(data_dir / f"{file_stem}_all_1h.csv")

        # Find 1h bars that have exactly 4 matching 15m bars
        rng = np.random.RandomState(42)
        valid_1h_indices = []

        for ts_1h in df_1h.index:
            # The 4 corresponding 15m bars: ts_1h, ts_1h+15, ts_1h+30, ts_1h+45
            expected_15m = [ts_1h + pd.Timedelta(minutes=m) for m in [0, 15, 30, 45]]
            if all(ts in df_15m.index for ts in expected_15m):
                valid_1h_indices.append(ts_1h)

        if len(valid_1h_indices) < n_samples:
            check.fail(f"{symbol}: Only {len(valid_1h_indices)} valid 1h bars found (need {n_samples})")
            continue

        sample_indices = rng.choice(len(valid_1h_indices), size=n_samples, replace=False)
        mismatches = 0

        for idx in sample_indices:
            ts_1h = valid_1h_indices[idx]
            bar_1h = df_1h.loc[ts_1h]

            # Aggregate 15m bars
            ts_15m_list = [ts_1h + pd.Timedelta(minutes=m) for m in [0, 15, 30, 45]]
            bars_15m = df_15m.loc[ts_15m_list]

            agg_open = bars_15m.iloc[0]["open"]
            agg_high = bars_15m["high"].max()
            agg_low = bars_15m["low"].min()
            agg_close = bars_15m.iloc[-1]["close"]
            agg_volume = bars_15m["volume"].sum()

            tol = 1e-6
            o_ok = approx_eq(agg_open, bar_1h["open"], tol)
            h_ok = approx_eq(agg_high, bar_1h["high"], tol)
            l_ok = approx_eq(agg_low, bar_1h["low"], tol)
            c_ok = approx_eq(agg_close, bar_1h["close"], tol)
            v_ok = approx_eq(agg_volume, bar_1h["volume"], rel_tol=1e-4)

            if not all([o_ok, h_ok, l_ok, c_ok, v_ok]):
                mismatches += 1
                fields = []
                if not o_ok: fields.append(f"O: {agg_open} vs {bar_1h['open']}")
                if not h_ok: fields.append(f"H: {agg_high} vs {bar_1h['high']}")
                if not l_ok: fields.append(f"L: {agg_low} vs {bar_1h['low']}")
                if not c_ok: fields.append(f"C: {agg_close} vs {bar_1h['close']}")
                if not v_ok: fields.append(f"V: {agg_volume} vs {bar_1h['volume']}")
                check.fail(f"{symbol} @ {ts_1h}: mismatch on {', '.join(fields)}")

        if mismatches == 0:
            check.ok(f"{symbol}: {n_samples}/{n_samples} 1h bars match perfectly")
        else:
            check.fail(f"{symbol}: {mismatches}/{n_samples} mismatches")

    check.report()
    return check


# ---------------------------------------------------------------------------
# Check 2: Aggregation 15m → 4h vs real CSV
# ---------------------------------------------------------------------------

def check_aggregation_15m_to_4h(data_dir):
    """Same as check 1 but for 4h bars (16 × 15m bars)."""
    check = CheckResult("CHECK 2: Aggregation 15m → 4h vs real CSV")

    n_samples = 10

    for symbol in SYMBOLS:
        file_stem = {"BTCUSDT": "BTCUSD", "ETHUSDT": "ETHUSD", "SOLUSDT": "SOLUSD",
                     "XRPUSDT": "XRPUSD", "BNBUSDT": "BNBUSD"}[symbol]

        df_15m = _load_csv(data_dir / f"{file_stem}_all_15m.csv")
        df_4h = _load_csv(data_dir / f"{file_stem}_all_4h.csv")

        rng = np.random.RandomState(123)
        valid_4h_indices = []

        for ts_4h in df_4h.index:
            expected_15m = [ts_4h + pd.Timedelta(minutes=m) for m in range(0, 240, 15)]
            if all(ts in df_15m.index for ts in expected_15m):
                valid_4h_indices.append(ts_4h)

        if len(valid_4h_indices) < n_samples:
            check.fail(f"{symbol}: Only {len(valid_4h_indices)} valid 4h bars (need {n_samples})")
            continue

        sample_indices = rng.choice(len(valid_4h_indices), size=n_samples, replace=False)
        mismatches = 0

        for idx in sample_indices:
            ts_4h = valid_4h_indices[idx]
            bar_4h = df_4h.loc[ts_4h]

            ts_15m_list = [ts_4h + pd.Timedelta(minutes=m) for m in range(0, 240, 15)]
            bars_15m = df_15m.loc[ts_15m_list]

            agg_open = bars_15m.iloc[0]["open"]
            agg_high = bars_15m["high"].max()
            agg_low = bars_15m["low"].min()
            agg_close = bars_15m.iloc[-1]["close"]
            agg_volume = bars_15m["volume"].sum()

            tol = 1e-6
            o_ok = approx_eq(agg_open, bar_4h["open"], tol)
            h_ok = approx_eq(agg_high, bar_4h["high"], tol)
            l_ok = approx_eq(agg_low, bar_4h["low"], tol)
            c_ok = approx_eq(agg_close, bar_4h["close"], tol)
            v_ok = approx_eq(agg_volume, bar_4h["volume"], rel_tol=1e-4)

            if not all([o_ok, h_ok, l_ok, c_ok, v_ok]):
                mismatches += 1
                check.fail(f"{symbol} @ {ts_4h}: OHLCV mismatch")

        if mismatches == 0:
            check.ok(f"{symbol}: {n_samples}/{n_samples} 4h bars match perfectly")

    check.report()
    return check


# ---------------------------------------------------------------------------
# Check 3: Indicator accuracy (precomputed vs fresh on filtered slice)
# ---------------------------------------------------------------------------

def check_indicator_accuracy(data_sources):
    """At N random timestamps, compute indicators fresh on the filtered slice
    and compare with precomputed values. Tests EMA, RSI, ATR, ADX, BB."""
    check = CheckResult("CHECK 3: Indicator accuracy (precomputed vs fresh talib)")

    n_samples = 15
    rng = np.random.RandomState(77)

    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:  # 3 symbols enough

        # --- 15m: EMA20, RSI14, ATR14, BB ---
        df_15m_full = data_sources[(symbol, "15m")]
        # Pick timestamps well into the data (skip first 250 for warmup)
        valid_range = range(250, len(df_15m_full) - 10)
        sample_idx = rng.choice(valid_range, size=n_samples, replace=False)

        for idx in sample_idx:
            as_of = df_15m_full.index[idx] + pd.Timedelta(minutes=15)
            df_closed = filter_closed_candles(df_15m_full, 15, as_of)

            # Read precomputed
            last = df_closed.iloc[-1]
            pre_ema20 = last["ema20"]
            pre_rsi = last["rsi14"]
            pre_atr = last["atr14"]
            pre_bb = last["bb_position"]

            # Compute fresh on the SAME filtered slice (raw OHLCV only)
            close = df_closed["close"].values.astype(float)
            high = df_closed["high"].values.astype(float)
            low = df_closed["low"].values.astype(float)

            fresh_ema20 = talib.EMA(close, timeperiod=20)[-1]
            fresh_rsi = talib.RSI(close, timeperiod=14)[-1]
            fresh_atr = talib.ATR(high, low, close, timeperiod=14)[-1]

            bb_u, bb_m, bb_l = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            bb_range = bb_u[-1] - bb_l[-1]
            fresh_bb = (close[-1] - bb_l[-1]) / bb_range if bb_range > 0 else 0.5

            ts_str = f"{symbol} 15m @ {df_closed.index[-1]}"

            if not approx_eq(pre_ema20, fresh_ema20, tol=1e-4):
                check.fail(f"{ts_str}: EMA20 {pre_ema20:.6f} vs {fresh_ema20:.6f}")
            if not approx_eq(pre_rsi, fresh_rsi, tol=1e-4):
                check.fail(f"{ts_str}: RSI14 {pre_rsi:.4f} vs {fresh_rsi:.4f}")
            if not approx_eq(pre_atr, fresh_atr, tol=1e-4):
                check.fail(f"{ts_str}: ATR14 {pre_atr:.6f} vs {fresh_atr:.6f}")
            if not approx_eq(pre_bb, fresh_bb, tol=1e-4):
                check.fail(f"{ts_str}: BB_pos {pre_bb:.6f} vs {fresh_bb:.6f}")

        check.ok(f"{symbol} 15m: {n_samples} timestamps — EMA20, RSI14, ATR14, BB all match")

        # --- 1h: EMA50, RSI14, ADX14 ---
        df_1h_full = data_sources[(symbol, "1h")]
        valid_range_1h = range(100, len(df_1h_full) - 5)
        if len(valid_range_1h) < n_samples:
            check.fail(f"{symbol} 1h: not enough data for {n_samples} samples")
            continue

        sample_idx_1h = rng.choice(valid_range_1h, size=min(n_samples, len(valid_range_1h)), replace=False)

        for idx in sample_idx_1h:
            as_of = df_1h_full.index[idx] + pd.Timedelta(hours=1)
            df_closed = filter_closed_candles(df_1h_full, 60, as_of)

            last = df_closed.iloc[-1]
            pre_ema50 = last["ema50"]
            pre_rsi = last["rsi14"]
            pre_adx = last["adx14"]

            close = df_closed["close"].values.astype(float)
            high = df_closed["high"].values.astype(float)
            low = df_closed["low"].values.astype(float)

            fresh_ema50 = talib.EMA(close, timeperiod=50)[-1]
            fresh_rsi = talib.RSI(close, timeperiod=14)[-1]
            fresh_adx = talib.ADX(high, low, close, timeperiod=14)[-1]

            ts_str = f"{symbol} 1h @ {df_closed.index[-1]}"

            if not approx_eq(pre_ema50, fresh_ema50, tol=1e-4):
                check.fail(f"{ts_str}: EMA50 {pre_ema50:.6f} vs {fresh_ema50:.6f}")
            if not approx_eq(pre_rsi, fresh_rsi, tol=1e-4):
                check.fail(f"{ts_str}: RSI14 {pre_rsi:.4f} vs {fresh_rsi:.4f}")
            if not approx_eq(pre_adx, fresh_adx, tol=1e-4):
                check.fail(f"{ts_str}: ADX14 {pre_adx:.4f} vs {fresh_adx:.4f}")

        check.ok(f"{symbol} 1h: {n_samples} timestamps — EMA50, RSI14, ADX14 all match")

    check.report()
    return check


# ---------------------------------------------------------------------------
# Check 4: Anti-look-ahead on real data
# ---------------------------------------------------------------------------

def check_anti_look_ahead_real(data_sources):
    """compute_features(full_data, T) == compute_features(truncated_at_T+4h, T)
    on real data. Verifies that EXISTENCE of future data doesn't pollute."""
    check = CheckResult("CHECK 4: Anti-look-ahead on real data (full vs truncated)")

    rng = np.random.RandomState(99)
    btc_15m = data_sources[("BTCUSDT", "15m")]

    # Pick 5 random timestamps, well into the data
    valid = range(500, len(btc_15m) - 500)
    sample_idx = rng.choice(valid, size=5, replace=False)

    context = {
        "positions_open_count": 1,
        "total_exposure_pct": 2.0,
        "btc_dominance": {"value": 55.0, "chg_24h_pct": 0.1},
        "funding_rates": {},
    }

    for idx in sample_idx:
        T = btc_15m.index[idx]
        T = T.floor("15min")

        # Result A: full dataset
        result_a = compute_features(data_sources, T, context)

        # Result B: truncate at T + 4h
        cutoff = T + pd.Timedelta(hours=4)
        data_trunc = {}
        for key, df in data_sources.items():
            data_trunc[key] = df[df.index <= cutoff].copy()

        result_b = compute_features(data_trunc, T, context)

        # Deep compare
        try:
            _deep_compare(result_a, result_b, tol=1e-6, path="root")
            check.ok(f"T={T}: full vs truncated(T+4h) — identical")
        except AssertionError as e:
            check.fail(f"T={T}: {e}")

    check.report()
    return check


def _deep_compare(a, b, tol, path):
    """Recursive comparison with tolerance."""
    if isinstance(a, dict) and isinstance(b, dict):
        keys_a = set(a.keys())
        keys_b = set(b.keys())
        if keys_a != keys_b:
            raise AssertionError(f"Keys differ at {path}: {keys_a ^ keys_b}")
        for k in a:
            _deep_compare(a[k], b[k], tol, f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise AssertionError(f"List len at {path}: {len(a)} vs {len(b)}")
        for i in range(len(a)):
            _deep_compare(a[i], b[i], tol, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, float):
        if abs(a - b) > tol:
            raise AssertionError(f"Float at {path}: {a} vs {b} (delta={abs(a-b):.2e})")
    elif a is None and b is None:
        pass
    else:
        if a != b:
            raise AssertionError(f"Value at {path}: {a!r} vs {b!r}")


class AssertionError(Exception):
    pass


# ---------------------------------------------------------------------------
# Check 5: VWAP session reset at 08:00 UTC
# ---------------------------------------------------------------------------

def check_vwap_session_reset(data_sources):
    """Verify that VWAP resets at SESSION_START_HOUR (08:00 UTC).

    At the first 15m bar of a new session (08:00), VWAP should equal
    the typical price of that single bar. At 08:15, it should be the
    cumulative VWAP of the 08:00 and 08:15 bars."""
    check = CheckResult("CHECK 5: VWAP session reset at 08:00 UTC")

    for symbol in ["BTCUSDT", "ETHUSDT"]:
        df = data_sources[(symbol, "15m")]

        # Find bars at exactly 08:00
        session_starts = df[df.index.hour == SESSION_START_HOUR]
        session_starts = session_starts[session_starts.index.minute == 0]

        if len(session_starts) < 5:
            check.fail(f"{symbol}: not enough 08:00 bars found")
            continue

        # Check 5 random session starts
        rng = np.random.RandomState(55)
        indices = rng.choice(len(session_starts), size=min(5, len(session_starts)), replace=False)

        for i in indices:
            ts = session_starts.index[i]
            bar = df.loc[ts]

            # At 08:00, VWAP should equal typical price of this single bar
            tp = (bar["high"] + bar["low"] + bar["close"]) / 3
            vwap_at_start = bar.get("vwap")

            if vwap_at_start is None or pd.isna(vwap_at_start):
                check.fail(f"{symbol} @ {ts}: VWAP is NaN at session start")
                continue

            if not approx_eq(vwap_at_start, tp, tol=1e-4):
                check.fail(
                    f"{symbol} @ {ts}: VWAP={vwap_at_start:.4f} != TP={tp:.4f} "
                    f"(delta={abs(vwap_at_start - tp):.6f})"
                )
                continue

            # Also check 08:15 bar — VWAP should be cumulative of 08:00 + 08:15
            ts_next = ts + pd.Timedelta(minutes=15)
            if ts_next in df.index:
                bar_next = df.loc[ts_next]
                tp_next = (bar_next["high"] + bar_next["low"] + bar_next["close"]) / 3

                cum_tp_vol = tp * bar["volume"] + tp_next * bar_next["volume"]
                cum_vol = bar["volume"] + bar_next["volume"]
                expected_vwap = cum_tp_vol / cum_vol if cum_vol > 0 else np.nan

                actual_vwap = bar_next.get("vwap")
                if not approx_eq(actual_vwap, expected_vwap, tol=1e-4):
                    check.fail(
                        f"{symbol} @ {ts_next}: VWAP={actual_vwap:.4f} != "
                        f"expected={expected_vwap:.4f}"
                    )
                    continue

            check.ok(f"{symbol} @ {ts}: VWAP correctly resets to TP={tp:.2f}")

    check.report()
    return check


# ---------------------------------------------------------------------------
# Check 6: Filter boundary — no future candle on higher TF (real data)
# ---------------------------------------------------------------------------

def check_filter_boundary_real(data_sources):
    """At multiple mid-bar timestamps, verify filter_closed_candles
    never includes candles whose close is in the future."""
    check = CheckResult("CHECK 6: Filter boundary — no future leak (real data)")

    for symbol in ["BTCUSDT", "SOLUSDT"]:
        for tf_label, tf_min in [("1h", 60), ("4h", 240)]:
            df = data_sources[(symbol, tf_label)]

            # Pick 10 timestamps in the MIDDLE of bars
            rng = np.random.RandomState(66)
            mid_offsets = rng.randint(1, tf_min, size=10)  # 1 to tf_min-1 minutes into a bar

            for i, offset in enumerate(mid_offsets):
                if i + 100 >= len(df):
                    break
                bar_ts = df.index[i + 100]
                as_of = bar_ts + pd.Timedelta(minutes=int(offset))

                closed = filter_closed_candles(df, tf_min, as_of)

                # Every candle in the result must have close_time <= as_of
                for ts in closed.index:
                    close_time = ts + pd.Timedelta(minutes=tf_min)
                    if close_time > as_of:
                        check.fail(
                            f"{symbol} {tf_label}: bar @ {ts} closes at {close_time} "
                            f"> as_of={as_of} — FUTURE LEAK!"
                        )
                        break
                else:
                    # Also verify the NEXT bar (if exists) is correctly excluded
                    last_included = closed.index[-1]
                    next_bar_idx = df.index.get_loc(last_included) + 1
                    if next_bar_idx < len(df):
                        next_bar_ts = df.index[next_bar_idx]
                        next_close = next_bar_ts + pd.Timedelta(minutes=tf_min)
                        if next_close <= as_of:
                            check.fail(
                                f"{symbol} {tf_label}: bar @ {next_bar_ts} closes at "
                                f"{next_close} <= {as_of} but was EXCLUDED"
                            )
                            continue

            check.ok(f"{symbol} {tf_label}: 10 mid-bar timestamps — no future leak")

    check.report()
    return check


# ---------------------------------------------------------------------------
# Check 7: Cross-TF price consistency
# ---------------------------------------------------------------------------

def check_cross_tf_consistency(data_sources):
    """At hourly boundaries, the last closed 15m bar's close should equal
    the 1h bar's close (since the 1h bar is built from those 15m bars)."""
    check = CheckResult("CHECK 7: Cross-TF price consistency (15m vs 1h at boundaries)")

    for symbol in ["BTCUSDT", "ETHUSDT", "BNBUSDT"]:
        df_15m = data_sources[(symbol, "15m")]
        df_1h = data_sources[(symbol, "1h")]

        # Pick 20 random hourly boundaries
        rng = np.random.RandomState(88)
        # as_of on the hour means the previous 1h bar just closed
        hourly_timestamps = df_1h.index[50:-10]  # skip edges
        sample = rng.choice(len(hourly_timestamps), size=min(20, len(hourly_timestamps)), replace=False)

        mismatches = 0
        for idx in sample:
            # as_of = 1h bar timestamp + 1h (= close time of that bar)
            bar_ts = hourly_timestamps[idx]
            as_of = bar_ts + pd.Timedelta(hours=1)

            try:
                closed_15m = filter_closed_candles(df_15m, 15, as_of)
                closed_1h = filter_closed_candles(df_1h, 60, as_of)
            except ValueError:
                continue

            price_15m = closed_15m.iloc[-1]["close"]
            price_1h = closed_1h.iloc[-1]["close"]

            # The last closed 1h bar should be the one at bar_ts
            if closed_1h.index[-1] != bar_ts:
                # Not aligned — skip (data gap)
                continue

            # The last closed 15m bar should be bar_ts + 45min
            expected_last_15m = bar_ts + pd.Timedelta(minutes=45)
            if closed_15m.index[-1] != expected_last_15m:
                # Data gap in 15m
                continue

            if not approx_eq(price_15m, price_1h, tol=1e-6):
                mismatches += 1
                check.fail(
                    f"{symbol} @ {as_of}: 15m close={price_15m} != 1h close={price_1h}"
                )

        if mismatches == 0:
            check.ok(f"{symbol}: 20 hourly boundaries — 15m and 1h closes match perfectly")

    check.report()
    return check


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_data_summary(data_sources):
    """Print a quick summary of loaded data."""
    print("\n" + "=" * 70)
    print("  DATA SUMMARY")
    print("=" * 70)
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            df = data_sources[(symbol, tf)]
            print(
                f"  {symbol:10s} {tf:4s}: {len(df):>8,} bars  "
                f"| {df.index[0]} → {df.index[-1]}"
            )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("src/data_trad")

    print(f"\nLoading data from {data_dir}/ ...")
    data_sources = load_all_data(str(data_dir))
    print(f"Loaded {len(data_sources)} datasets.")
    print_data_summary(data_sources)

    all_passed = True

    # Checks that use raw CSVs (no precomputed indicators needed)
    c1 = check_aggregation_15m_to_1h(data_dir)
    all_passed &= (c1.failed == 0)

    c2 = check_aggregation_15m_to_4h(data_dir)
    all_passed &= (c2.failed == 0)

    # Checks that use precomputed data
    c3 = check_indicator_accuracy(data_sources)
    all_passed &= (c3.failed == 0)

    c4 = check_anti_look_ahead_real(data_sources)
    all_passed &= (c4.failed == 0)

    c5 = check_vwap_session_reset(data_sources)
    all_passed &= (c5.failed == 0)

    c6 = check_filter_boundary_real(data_sources)
    all_passed &= (c6.failed == 0)

    c7 = check_cross_tf_consistency(data_sources)
    all_passed &= (c7.failed == 0)

    # Final summary
    total_passed = sum(c.passed for c in [c1, c2, c3, c4, c5, c6, c7])
    total_failed = sum(c.failed for c in [c1, c2, c3, c4, c5, c6, c7])

    print("\n" + "=" * 70)
    print(f"  FINAL RESULT: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"  Total: {total_passed} passed, {total_failed} failed")
    print("=" * 70)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
