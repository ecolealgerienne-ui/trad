"""Diagnostic: extract and analyze all Qwen rationales from backtest JSONL."""
import json
import re
from collections import Counter, defaultdict

jsonl_path = 'logs/backtest_20260413_014758.jsonl'

by_action = defaultdict(list)  # action -> [(symbol, conviction, rationale)]

with open(jsonl_path) as f:
    for line in f:
        rec = json.loads(line)
        parsed = rec.get('parsed')
        if not parsed:
            continue
        for a in parsed.get('assets', []):
            action = a.get('action', '?')
            sym = a.get('symbol', '?')
            conv = a.get('conviction', 0)
            rat = a.get('rationale', '').strip()
            by_action[action].append((sym, conv, rat))

print("=" * 60)
print("  RATIONALE ANALYSIS")
print("=" * 60)

for action in ['skip', 'hold', 'buy', 'close']:
    entries = by_action.get(action, [])
    if not entries:
        continue
    print(f"\n{'='*60}")
    print(f"  {action.upper()} — {len(entries)} decisions")
    print(f"{'='*60}")

    # By symbol
    sym_counts = Counter(sym for sym, _, _ in entries)
    print(f"\n  By asset: {dict(sorted(sym_counts.items()))}")

    # Conviction distribution
    convs = [c for _, c, _ in entries]
    conv_dist = Counter(convs)
    print(f"  Convictions: {dict(sorted(conv_dist.items()))}")

    # All rationales
    rationales = [r for _, _, r in entries if r]

    # Top 10 most common exact rationales
    exact_counts = Counter(rationales).most_common(10)
    print(f"\n  Top 10 rationales:")
    for rat, count in exact_counts:
        print(f"    [{count:3d}x] {rat[:120]}")

    # Keyword analysis
    keywords = {
        'no setup': r'no\s+(clear\s+)?setup',
        'no signal': r'no\s+(clear\s+)?signal',
        'consolidation': r'consolidat',
        'range': r'\brange\b|\branging\b',
        'weak': r'\bweak\b',
        'low conviction': r'low\s+conviction',
        'wait': r'\bwait\b',
        'no edge': r'no\s+edge',
        'sideways': r'sideways',
        'bearish': r'bearish',
        'bullish': r'bullish',
        'momentum': r'momentum',
        'trend': r'trend',
        'support': r'support',
        'resistance': r'resistance',
        'volume': r'volume',
        'overbought': r'overbought',
        'oversold': r'oversold',
        'risk': r'\brisk\b',
        'pullback': r'pullback',
        'breakout': r'breakout',
        'EMA': r'\bEMA\b|\bema\b',
        'RSI': r'\bRSI\b|\brsi\b',
        'ADX': r'\bADX\b|\badx\b',
    }

    if rationales:
        print(f"\n  Keyword frequency ({len(rationales)} rationales):")
        kw_counts = {}
        for kw_name, pattern in keywords.items():
            count = sum(1 for r in rationales if re.search(pattern, r, re.IGNORECASE))
            if count > 0:
                kw_counts[kw_name] = count

        for kw, count in sorted(kw_counts.items(), key=lambda x: -x[1]):
            pct = count / len(rationales) * 100
            print(f"    {kw:20s}: {count:3d} ({pct:.0f}%)")

# Summary
print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
total = sum(len(v) for v in by_action.values())
for action in ['buy', 'hold', 'skip', 'close']:
    n = len(by_action.get(action, []))
    pct = n / total * 100 if total > 0 else 0
    print(f"  {action:6s}: {n:4d} ({pct:.1f}%)")
print(f"  total : {total}")
