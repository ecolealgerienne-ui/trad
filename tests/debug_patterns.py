"""Diagnostic: analyze Qwen decision patterns across all 91 valid cycles."""
import json
import numpy as np
from collections import Counter

jsonl_path = 'logs/test_run_20260412_224953.jsonl'

modes = []
regimes = []
stop_mults = []
tp_mults = []
entry_zones_status = []  # 'ok', 'absurd', 'null'
convictions_by_mode = {'risk_off': [], 'risk_on': [], 'rotation': [], 'neutral': []}
btc_prices_by_mode = {'risk_off': [], 'risk_on': [], 'rotation': [], 'neutral': []}

with open(jsonl_path) as f:
    for line in f:
        rec = json.loads(line)
        if not rec.get('success') or not rec.get('parsed'):
            continue

        parsed = rec['parsed']
        g = parsed.get('global', {})
        mode = g.get('market_mode', 'unknown')
        regime = g.get('btc_regime', 'unknown')
        modes.append(mode)
        regimes.append(regime)

        # BTC price from snapshot
        snap = rec.get('features_snapshot', {})
        btc_price = (snap.get('ASSET_A') or {}).get('price')
        if btc_price:
            btc_prices_by_mode[mode].append(btc_price)

        for a in parsed.get('assets', []):
            conv = a.get('conviction', 0)
            convictions_by_mode[mode].append(conv)

            if a.get('action') == 'buy':
                sm = a.get('atr_stop_multiplier')
                tm = a.get('atr_tp_multiplier')
                if sm is not None:
                    stop_mults.append(sm)
                if tm is not None:
                    tp_mults.append(tm)

                # Check entry_zone
                ez = a.get('entry_zone')
                if ez is None:
                    entry_zones_status.append('null')
                else:
                    anon = a.get('symbol', '')
                    asset_snap = snap.get(anon, {})
                    price = asset_snap.get('price')
                    if price and price > 0:
                        ez_min = ez.get('min', 0)
                        ez_max = ez.get('max', 0)
                        # Absurd = outside ±10% of current price
                        if abs(ez_min - price) / price > 0.10 or abs(ez_max - price) / price > 0.10:
                            entry_zones_status.append('absurd')
                        else:
                            entry_zones_status.append('ok')
                    else:
                        entry_zones_status.append('unknown')

# ==================== PRINT RESULTS ====================

print("=" * 60)
print("  1. DISTRIBUTION DES DECISIONS QWEN (91 cycles)")
print("=" * 60)

print("\n  market_mode:")
for k, v in sorted(Counter(modes).items(), key=lambda x: -x[1]):
    pct = v / len(modes) * 100
    print(f"    {k:15s}: {v:3d} ({pct:.1f}%)")

print("\n  btc_regime:")
for k, v in sorted(Counter(regimes).items(), key=lambda x: -x[1]):
    pct = v / len(regimes) * 100
    print(f"    {k:15s}: {v:3d} ({pct:.1f}%)")

print(f"\n  atr_stop_multiplier ({len(stop_mults)} buys):")
if stop_mults:
    print(f"    min={min(stop_mults):.2f}  max={max(stop_mults):.2f}  "
          f"mean={np.mean(stop_mults):.2f}  median={np.median(stop_mults):.2f}")
    print(f"    distribution: {Counter([round(x, 1) for x in stop_mults]).most_common(5)}")

print(f"\n  atr_tp_multiplier ({len(tp_mults)} buys):")
if tp_mults:
    print(f"    min={min(tp_mults):.2f}  max={max(tp_mults):.2f}  "
          f"mean={np.mean(tp_mults):.2f}  median={np.median(tp_mults):.2f}")
    print(f"    distribution: {Counter([round(x, 1) for x in tp_mults]).most_common(5)}")

print(f"\n  entry_zone quality ({len(entry_zones_status)} buys):")
for k, v in sorted(Counter(entry_zones_status).items(), key=lambda x: -x[1]):
    pct = v / len(entry_zones_status) * 100 if entry_zones_status else 0
    print(f"    {k:10s}: {v:3d} ({pct:.1f}%)")

# ==================== CORRELATION ====================

print("\n" + "=" * 60)
print("  2. RISK_OFF vs PRIX BTC (biais ou lecture correcte?)")
print("=" * 60)

for mode in ['risk_off', 'risk_on', 'rotation', 'neutral']:
    prices = btc_prices_by_mode.get(mode, [])
    if len(prices) >= 2:
        # Price trend within this mode's cycles
        changes = [(prices[i] - prices[i-1]) / prices[i-1] * 100 for i in range(1, len(prices))]
        avg_chg = np.mean(changes)
        print(f"\n  {mode} ({len(prices)} cycles):")
        print(f"    BTC price range: ${min(prices):,.0f} — ${max(prices):,.0f}")
        print(f"    Avg inter-cycle chg: {avg_chg:+.4f}%")
        print(f"    Total chg first→last: {(prices[-1]-prices[0])/prices[0]*100:+.3f}%")
    elif len(prices) == 1:
        print(f"\n  {mode} (1 cycle): BTC=${prices[0]:,.0f}")
    else:
        print(f"\n  {mode}: no data")

# Overall BTC trend
all_prices = []
for prices in btc_prices_by_mode.values():
    all_prices.extend(prices)
if all_prices:
    print(f"\n  Overall BTC: ${min(all_prices):,.0f} → ${max(all_prices):,.0f}")
