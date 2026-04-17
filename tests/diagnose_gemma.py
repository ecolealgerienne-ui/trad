"""
Diagnostic: isolate why Gemma ignores the schema.

3 tests, same Ollama endpoint, same timestamp:
  TEST 1: Simple prompt + text user message (reproduces working curl)
  TEST 2: Simple prompt + JSON user message (is JSON the problem?)
  TEST 3: Full prompt v3 + JSON user message (is prompt length the problem?)
"""

import json
import sys
import time

import requests

sys.path.insert(0, ".")
from src.feature_engineering import load_all_data, filter_closed_candles, compute_features, anonymize_and_format, SYMBOLS

OLLAMA_URL = "http://localhost:11434/api/chat"

# --- Simple prompt (~800 chars, mimics working curl) ---
SIMPLE_PROMPT = """You are a crypto analyst. Analyze 5 assets and output JSON only.

Output this exact structure:
{"global":{"btc_regime":"trending_up|trending_down|range|chaotic","market_mode":"risk_on|risk_off|rotation|neutral","market_coherence":"aligned|diverging|rotating","sector_flow":"btc_led|alt_led|mixed|defensive","relative_strength_ranking":["ASSET_A","ASSET_B","ASSET_C","ASSET_D","ASSET_E"],"risk_adjustment":"normal|defensive|aggressive","max_concurrent_positions":1,"rationale":"..."},"assets":[{"symbol":"ASSET_A","regime":"range","setup":"none","action":"skip","conviction":0,"relative_strength_rank":1,"entry_zone":null,"atr_stop_multiplier":null,"atr_tp_multiplier":null,"expected_horizon_hours":null,"holistic_justification":"...","rationale":"..."}],"meta":{"analysis_confidence":0}}

All 5 assets in order A,B,C,D,E. JSON only, no other text."""

# --- Load real features for one timestamp ---
print("Loading data...")
data = load_all_data("src/data_trad")
btc_15m = data[("BTCUSDT", "15m")]
mid = len(btc_15m) // 2
as_of = btc_15m.index[mid].floor("15min")

context = {"positions_open_count": 0, "total_exposure_pct": 0,
           "btc_dominance": {"value": 54, "chg_24h_pct": 0},
           "funding_rates": {s: 0 for s in SYMBOLS}}

features = compute_features(data, as_of, context)
mapping = {"BTCUSDT": "ASSET_A", "ETHUSDT": "ASSET_B", "SOLUSDT": "ASSET_C",
           "XRPUSDT": "ASSET_D", "BNBUSDT": "ASSET_E"}
anon = anonymize_and_format(features, mapping)

# --- Build text version of features ---
def features_to_text(f):
    lines = []
    g = f["global"]
    lines.append(f"BTC: price={g['btc']['price']}, chg_1h={g['btc']['chg_1h_pct']}%, chg_24h={g['btc']['chg_24h_pct']}%")
    lines.append(f"Portfolio: {g['portfolio']['open_positions']} positions, {g['portfolio']['total_exposure_pct']}% exposure")
    lines.append(f"Minutes to close: {g['time']['minutes_to_close']}")
    lines.append("")
    for a in f["assets"]:
        parts = [f"{a['id']}: price={a['price']}"]
        s = a.get("session", {})
        if s.get("chg_pct") is not None:
            parts.append(f"session_chg={s['chg_pct']}%")
        t = a.get("trend", {})
        e15 = t.get("ema20_15m", {})
        if e15.get("slope_pct") is not None:
            parts.append(f"ema20_slope={e15['slope_pct']}%")
            parts.append(f"ema20_dist={e15['dist_pct']}%")
        m = a.get("momentum", {})
        if m.get("rsi_15m") is not None:
            parts.append(f"rsi_15m={m['rsi_15m']}")
        if m.get("rsi_1h") is not None:
            parts.append(f"rsi_1h={m['rsi_1h']}")
        r = a.get("regime", {})
        if r.get("adx_1h") is not None:
            parts.append(f"adx_1h={r['adx_1h']}")
        v = a.get("volatility", {})
        if v.get("atr_15m_pct") is not None:
            parts.append(f"atr_pct={v['atr_15m_pct']}%")
        if v.get("bb_position_15m") is not None:
            parts.append(f"bb_pos={v['bb_position_15m']}")
        vol = a.get("volume", {})
        if vol.get("vol_rel_15m") is not None:
            parts.append(f"vol_rel={vol['vol_rel_15m']}")
        c = a.get("correlation", {})
        if c.get("corr_btc_24h") is not None:
            parts.append(f"corr_btc={c['corr_btc_24h']}")
        lines.append(", ".join(parts))
    return "\n".join(lines)

text_features = features_to_text(anon)
json_features = json.dumps(anon, default=str)

# --- Full prompt v3 (from file) ---
with open("src/prompts/gemma_system_v1.txt") as f:
    full_prompt = f.read()


def call_ollama(system, user):
    payload = {
        "model": "gemma4:26b",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": 0.2},
    }
    t0 = time.perf_counter()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    latency = round(time.perf_counter() - t0, 1)
    content = resp.json().get("message", {}).get("content", "")
    return content, latency


def check_result(raw, test_name):
    start = raw.find("{")
    if start == -1:
        print(f"  NO JSON FOUND")
        return
    # Extract first object
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{": depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    d = json.loads(raw[start:i+1])
                    keys = list(d.keys())
                    has_global = "global" in keys
                    has_assets = "assets" in keys
                    has_meta = "meta" in keys
                    ok = has_global and has_assets and has_meta
                    status = "PASS" if ok else "FAIL"
                    print(f"  {status} — top keys: {keys}")
                    if has_assets and isinstance(d["assets"], list) and len(d["assets"]) > 0:
                        print(f"  assets[0] keys: {list(d['assets'][0].keys())}")
                    if not ok:
                        print(f"  Missing: global={has_global} assets={has_assets} meta={has_meta}")
                except json.JSONDecodeError as e:
                    print(f"  JSON PARSE ERROR: {e}")
                return
    print(f"  INCOMPLETE JSON")


# =====================================================================
print(f"\nTimestamp: {as_of}")
print(f"Text features: {len(text_features)} chars")
print(f"JSON features: {len(json_features)} chars")
print(f"Simple prompt: {len(SIMPLE_PROMPT)} chars")
print(f"Full prompt v3: {len(full_prompt)} chars")

# TEST 1
print(f"\n{'='*60}")
print("TEST 1: Simple prompt + TEXT features")
print(f"{'='*60}")
raw1, lat1 = call_ollama(SIMPLE_PROMPT, text_features)
print(f"  Latency: {lat1}s, Response: {len(raw1)} chars")
check_result(raw1, "TEST 1")

# TEST 2
print(f"\n{'='*60}")
print("TEST 2: Simple prompt + JSON features")
print(f"{'='*60}")
raw2, lat2 = call_ollama(SIMPLE_PROMPT, json_features)
print(f"  Latency: {lat2}s, Response: {len(raw2)} chars")
check_result(raw2, "TEST 2")

# TEST 3
print(f"\n{'='*60}")
print("TEST 3: Full prompt v3 + JSON features")
print(f"{'='*60}")
raw3, lat3 = call_ollama(full_prompt, json_features)
print(f"  Latency: {lat3}s, Response: {len(raw3)} chars")
check_result(raw3, "TEST 3")

print(f"\n{'='*60}")
print("DONE")
print(f"{'='*60}")
