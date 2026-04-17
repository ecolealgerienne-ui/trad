"""Diagnostic: understand why 186 buys → 9 trades. Trace every filter."""
import json
import numpy as np
from collections import Counter, defaultdict

jsonl_path = 'logs/test_run_20260412_224953.jsonl'

ANON_TO_REAL = {
    "ASSET_A": "BTCUSDT", "ASSET_B": "ETHUSDT", "ASSET_C": "SOLUSDT",
    "ASSET_D": "XRPUSDT", "ASSET_E": "BNBUSDT",
}

# Collect all data
all_buys = []  # (cycle_idx, symbol, conviction, mode, regime, max_pos, entry_zone, price, atr)
cycles_risk_on = []
cycles_all = []

with open(jsonl_path) as f:
    for line in f:
        rec = json.loads(line)
        if not rec.get('success') or not rec.get('parsed'):
            continue

        parsed = rec['parsed']
        g = parsed.get('global', {})
        mode = g.get('market_mode', 'unknown')
        regime = g.get('btc_regime', 'unknown')
        max_pos = g.get('max_concurrent_positions', 5)
        snap = rec.get('features_snapshot', {})
        as_of = rec.get('as_of', '?')

        cycles_all.append({'mode': mode, 'regime': regime, 'max_pos': max_pos, 'as_of': as_of})
        if mode == 'risk_on':
            cycles_risk_on.append(max_pos)

        for a in parsed.get('assets', []):
            if a.get('action') == 'buy':
                sym = a.get('symbol', '?')
                conv = a.get('conviction', 0)
                ez = a.get('entry_zone')
                asset_snap = snap.get(sym, {})
                price = asset_snap.get('price')
                atr = asset_snap.get('atr_15m_abs')
                all_buys.append({
                    'cycle': as_of, 'symbol': sym, 'conviction': conv,
                    'mode': mode, 'regime': regime, 'max_pos': max_pos,
                    'entry_zone': ez, 'price': price, 'atr': atr,
                })

# ==================== 1. CONVICTION DISTRIBUTION ====================

print("=" * 60)
print("  1. CONVICTION DISTRIBUTION OF ALL BUYS")
print("=" * 60)

conv_below7 = [b for b in all_buys if b['conviction'] < 7]
conv_7_8 = [b for b in all_buys if 7 <= b['conviction'] <= 8]
conv_9_10 = [b for b in all_buys if b['conviction'] >= 9]

print(f"  Total buy signals: {len(all_buys)}")
print(f"  conviction < 7:  {len(conv_below7):3d} ({len(conv_below7)/len(all_buys)*100:.1f}%) — FILTERED")
print(f"  conviction 7-8:  {len(conv_7_8):3d} ({len(conv_7_8)/len(all_buys)*100:.1f}%)")
print(f"  conviction 9-10: {len(conv_9_10):3d} ({len(conv_9_10)/len(all_buys)*100:.1f}%)")
print(f"  conv >= 7 total: {len(conv_7_8)+len(conv_9_10):3d}")

# ==================== 2. MAX_POS ON RISK_ON CYCLES ====================

print("\n" + "=" * 60)
print("  2. MAX_CONCURRENT_POSITIONS ON RISK_ON CYCLES")
print("=" * 60)

if cycles_risk_on:
    for k, v in sorted(Counter(cycles_risk_on).items()):
        print(f"  max_pos={k}: {v} cycles")
else:
    print("  No risk_on cycles!")

risk_on_count = sum(1 for c in cycles_all if c['mode'] == 'risk_on')
risk_off_count = sum(1 for c in cycles_all if c['mode'] == 'risk_off')
other_count = len(cycles_all) - risk_on_count - risk_off_count
print(f"\n  Total cycles: {len(cycles_all)}")
print(f"  risk_on:  {risk_on_count} ({risk_on_count/len(cycles_all)*100:.1f}%)")
print(f"  risk_off: {risk_off_count} ({risk_off_count/len(cycles_all)*100:.1f}%)")
print(f"  other:    {other_count} ({other_count/len(cycles_all)*100:.1f}%)")

# ==================== 3. FILTER FUNNEL ====================

print("\n" + "=" * 60)
print("  3. FILTER FUNNEL: 186 buys → 9 trades")
print("=" * 60)

# Simulate the filter chain from backtest_engine
filtered_conviction = 0
filtered_r1 = 0
filtered_max_pos = 0
filtered_already_in = 0
executed = 0
filtered_other = 0

# We need to simulate the backtest logic cycle by cycle
open_positions = set()  # symbols currently held
cycle_groups = defaultdict(list)
for b in all_buys:
    cycle_groups[b['cycle']].append(b)

# Sort cycles chronologically
sorted_cycles = sorted(cycles_all, key=lambda c: c['as_of'])
cycle_modes = {c['as_of']: c for c in sorted_cycles}

for cycle_info in sorted_cycles:
    as_of = cycle_info['as_of']
    mode = cycle_info['mode']
    regime = cycle_info['regime']
    max_pos = cycle_info['max_pos']

    buys_this_cycle = cycle_groups.get(as_of, [])
    entries_this_cycle = 0

    for b in buys_this_cycle:
        # Filter 1: conviction
        if b['conviction'] < 7:
            filtered_conviction += 1
            continue

        # Filter 2: R1 (risk_off or chaotic)
        if mode == 'risk_off' or regime == 'chaotic':
            filtered_r1 += 1
            continue

        # Filter 3: max positions
        if len(open_positions) + entries_this_cycle >= min(5, max_pos):
            filtered_max_pos += 1
            continue

        # Filter 4: already in position
        real_sym = ANON_TO_REAL.get(b['symbol'], b['symbol'])
        if real_sym in open_positions:
            filtered_already_in += 1
            continue

        # Would be executed
        executed += 1
        entries_this_cycle += 1
        open_positions.add(real_sym)

    # Simplified: assume positions close after ~3h (12 cycles)
    # This is approximate but shows the pattern
    # For exact numbers we'd need the full backtest logic

total_buys = len(all_buys)
print(f"  Total buy signals:           {total_buys}")
print(f"  Filtered conviction < 7:     {filtered_conviction:3d} ({filtered_conviction/total_buys*100:.1f}%)")
print(f"  Filtered R1 (risk_off):      {filtered_r1:3d} ({filtered_r1/total_buys*100:.1f}%)")
print(f"  Filtered max_positions:      {filtered_max_pos:3d} ({filtered_max_pos/total_buys*100:.1f}%)")
print(f"  Filtered already_in_pos:     {filtered_already_in:3d} ({filtered_already_in/total_buys*100:.1f}%)")
print(f"  Would execute (approx):      {executed:3d} ({executed/total_buys*100:.1f}%)")
print(f"\n  NOTE: 'already_in_pos' is approximate — positions never close in this")
print(f"  simulation. Real backtest has exits, so actual number differs.")

# ==================== 4. ENTRY ZONE ANALYSIS BY ASSET ====================

print("\n" + "=" * 60)
print("  4. ENTRY ZONE QUALITY BY ASSET")
print("=" * 60)

by_asset = defaultdict(lambda: {'ok': 0, 'absurd': 0, 'null': 0, 'examples_absurd': []})

for b in all_buys:
    sym = b['symbol']
    ez = b['entry_zone']
    price = b['price']

    if ez is None:
        by_asset[sym]['null'] += 1
    elif price and price > 0:
        ez_min = ez.get('min', 0)
        ez_max = ez.get('max', 0)
        dist_min = abs(ez_min - price) / price
        dist_max = abs(ez_max - price) / price
        if dist_min > 0.10 or dist_max > 0.10:
            by_asset[sym]['absurd'] += 1
            if len(by_asset[sym]['examples_absurd']) < 3:
                by_asset[sym]['examples_absurd'].append(
                    f"price={price} ez=[{ez_min}, {ez_max}]"
                )
        else:
            by_asset[sym]['ok'] += 1

for sym in sorted(by_asset):
    d = by_asset[sym]
    total = d['ok'] + d['absurd'] + d['null']
    real = ANON_TO_REAL.get(sym, sym)
    print(f"\n  {sym} ({real}):")
    print(f"    ok={d['ok']}  absurd={d['absurd']}  null={d['null']}  total={total}")
    if d['absurd'] > 0:
        pct = d['absurd'] / total * 100
        print(f"    absurd rate: {pct:.0f}%")
        for ex in d['examples_absurd']:
            print(f"      example: {ex}")
