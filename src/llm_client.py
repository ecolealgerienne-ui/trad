"""
Ollama wrapper for Gemma 4 26B.

Key finding: Gemma respects the output schema when features are sent as
NATURAL TEXT, not as raw JSON. JSON input "contaminates" the model and it
invents its own JSON structure. Text input keeps it focused on the schema.

Strategy:
1. Short system prompt (~850 chars) with schema example
2. Features converted to natural text before sending
3. format:"json" forces valid JSON output
4. think:false disables internal reasoning blocks
5. Retry with error feedback + original data if validation fails
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from pydantic import ValidationError

from src.schemas import GemmaOutput

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
PROMPTS_DIR = Path(__file__).parent / "prompts"


class LLMValidationError(Exception):
    """Raised when Gemma output fails validation after retry."""

    def __init__(self, message: str, raw_first: str, raw_retry: Optional[str], errors: list):
        super().__init__(message)
        self.raw_first = raw_first
        self.raw_retry = raw_retry
        self.errors = errors


def load_system_prompt(filename: str = "gemma_system_v1.txt") -> str:
    """Read the system prompt from disk."""
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"System prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def ping_ollama(base_url: str = OLLAMA_BASE_URL, timeout: float = 5.0) -> bool:
    """Check if Ollama is reachable. Returns True if OK."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=timeout)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


# ---------------------------------------------------------------------------
# Features to natural text (CRITICAL — Gemma needs text input, not JSON)
# ---------------------------------------------------------------------------


def features_to_text(features: dict) -> str:
    """Convert anonymized features dict to structured text for Gemma.

    Gemma produces correct schema output ONLY when input is text.
    JSON input contaminates the model and it invents its own structure.
    """
    lines = []

    # Cycle
    ci = features.get("cycle_index", "?")
    lines.append(f"CYCLE {ci}")

    # Global context
    g = features.get("global", {})
    btc = g.get("btc", {})
    dom = g.get("btc_dominance", {})
    t = g.get("time", {})
    pf = g.get("portfolio", {})

    lines.append("GLOBAL:")
    lines.append(f"  BTC: chg_1h={btc.get('chg_1h_pct')}%, chg_24h={btc.get('chg_24h_pct')}%")
    if isinstance(dom, dict) and dom.get("value") is not None:
        lines.append(f"  Dominance: {dom['value']}% (chg_24h={dom.get('chg_24h_pct', 0)}%)")
    lines.append(f"  Time to close: {t.get('minutes_to_close')} min")
    lines.append(f"  Portfolio: {pf.get('open_positions', 0)} positions, {pf.get('total_exposure_pct', 0)}% exposure")

    # Per-asset
    for a in features.get("assets", []):
        asset_id = a.get("id", a.get("_symbol", "?"))
        lines.append("")
        lines.append(f"{asset_id}:")
        lines.append(f"  price={a.get('price')}")

        # Session
        s = a.get("session", {})
        s_parts = []
        if s.get("chg_pct") is not None:
            s_parts.append(f"chg={s['chg_pct']}%")
        if s.get("range_position") is not None:
            s_parts.append(f"range_pos={s['range_position']}")
        if s_parts:
            lines.append(f"  session: {', '.join(s_parts)}")

        # Trend
        tr = a.get("trend", {})
        t_parts = []
        for ema_key, ema_data in tr.items():
            if isinstance(ema_data, dict) and ema_data.get("slope_pct") is not None:
                t_parts.append(
                    f"{ema_key} slope={ema_data['slope_pct']}% dist={ema_data.get('dist_pct', '?')}%"
                )
        if t_parts:
            lines.append(f"  trend: {', '.join(t_parts)}")

        # Regime
        r = a.get("regime", {})
        if r.get("adx_1h") is not None:
            lines.append(f"  regime: adx_1h={r['adx_1h']}")

        # Momentum
        m = a.get("momentum", {})
        m_parts = []
        if m.get("rsi_15m") is not None:
            m_parts.append(f"rsi_15m={m['rsi_15m']}")
        if m.get("rsi_1h") is not None:
            m_parts.append(f"rsi_1h={m['rsi_1h']}")
        if m_parts:
            lines.append(f"  momentum: {', '.join(m_parts)}")

        # Volatility
        v = a.get("volatility", {})
        v_parts = []
        if v.get("atr_15m_pct") is not None:
            v_parts.append(f"atr_pct={v['atr_15m_pct']}%")
        if v.get("atr_ratio_vs_avg50") is not None:
            v_parts.append(f"atr_ratio={v['atr_ratio_vs_avg50']}")
        if v.get("bb_position_15m") is not None:
            v_parts.append(f"bb_pos={v['bb_position_15m']}")
        if v_parts:
            lines.append(f"  volatility: {', '.join(v_parts)}")

        # Volume
        vol = a.get("volume", {})
        vol_parts = []
        if vol.get("vol_rel_15m") is not None:
            vol_parts.append(f"vol_rel_15m={vol['vol_rel_15m']}")
        if vol.get("vol_rel_1h") is not None:
            vol_parts.append(f"vol_rel_1h={vol['vol_rel_1h']}")
        if vol.get("vwap_dist_pct") is not None:
            vol_parts.append(f"vwap_dist={vol['vwap_dist_pct']}%")
        if vol_parts:
            lines.append(f"  volume: {', '.join(vol_parts)}")

        # Structure
        st = a.get("structure", {})
        sup = st.get("support", {})
        res = st.get("resistance", {})
        st_parts = []
        if sup.get("price") is not None:
            st_parts.append(f"support={sup['price']} ({sup.get('dist_pct','')}% {sup.get('type','')})")
        if res.get("price") is not None:
            st_parts.append(f"resistance={res['price']} ({res.get('dist_pct','')}% {res.get('type','')})")
        if st_parts:
            lines.append(f"  structure: {' / '.join(st_parts)}")

        # Correlation
        c = a.get("correlation", {})
        if c.get("corr_btc_24h") is not None:
            lines.append(f"  correlation: corr_btc_24h={c['corr_btc_24h']}")

        # Sentiment
        se = a.get("sentiment", {})
        if se.get("funding_rate_perp_8h") is not None:
            lines.append(f"  sentiment: funding={se['funding_rate_perp_8h']}")

        # Current bars
        cb = a.get("current_bar", {})
        for tf_key, cb_data in cb.items():
            if cb_data is not None and isinstance(cb_data, dict):
                cb_parts = []
                if cb_data.get("progress_pct") is not None:
                    cb_parts.append(f"progress={cb_data['progress_pct']}%")
                if cb_data.get("move_pct") is not None:
                    cb_parts.append(f"move={cb_data['move_pct']}%")
                if cb_data.get("range_vs_atr") is not None:
                    cb_parts.append(f"range_vs_atr={cb_data['range_vs_atr']}")
                if cb_data.get("vol_vs_expected") is not None:
                    cb_parts.append(f"vol_vs_exp={cb_data['vol_vs_expected']}")
                if cb_parts:
                    lines.append(f"  current_bar_{tf_key}: {' '.join(cb_parts)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def call_gemma(
    system_prompt: str,
    user_payload: dict,
    temperature: float = 0.2,
    model: str = "gemma4:26b",
    base_url: str = OLLAMA_BASE_URL,
    timeout: float = 300.0,
) -> Dict[str, Any]:
    """Call Gemma via Ollama and return validated output + metadata."""
    # Convert features to text — Gemma needs text input, not JSON
    user_content = features_to_text(user_payload)

    result = {
        "parsed": None,
        "raw_response_first_attempt": None,
        "raw_response_retry": None,
        "validation_errors": [],
        "latency_sec": 0.0,
        "retried": False,
        "success": False,
    }

    t_start = time.perf_counter()

    # --- First attempt ---
    raw_text = _call_ollama(system_prompt, user_content, temperature, model, base_url, timeout)
    result["raw_response_first_attempt"] = raw_text

    parsed, errors = _validate_response(raw_text)

    if parsed is not None:
        result["parsed"] = parsed
        result["success"] = True
        result["latency_sec"] = round(time.perf_counter() - t_start, 2)
        return result

    # --- Retry with error feedback + original data ---
    result["retried"] = True
    result["validation_errors"] = errors
    logger.warning("First attempt failed validation: %s — retrying", errors)

    error_list = "\n".join(f"- {err}" for err in errors[:10])
    retry_user = (
        f"Your previous JSON response was missing required fields:\n"
        f"{error_list}\n\n"
        f"Regenerate the complete JSON with ALL required fields. "
        f"Keep the same analysis, just include the missing fields. "
        f"Strict schema only.\n\n"
        f"{user_content}"
    )

    raw_retry = _call_ollama(system_prompt, retry_user, temperature, model, base_url, timeout)
    result["raw_response_retry"] = raw_retry

    parsed_retry, errors_retry = _validate_response(raw_retry)

    if parsed_retry is not None:
        result["parsed"] = parsed_retry
        result["success"] = True
        result["validation_errors"] = []
        result["latency_sec"] = round(time.perf_counter() - t_start, 2)
        return result

    # --- Both attempts failed ---
    all_errors = errors + errors_retry
    result["validation_errors"] = all_errors
    result["latency_sec"] = round(time.perf_counter() - t_start, 2)
    logger.error("Both attempts failed validation: %s", all_errors[:5])

    return result


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _call_ollama(
    system_prompt: str,
    user_content: str,
    temperature: float,
    model: str,
    base_url: str,
    timeout: float,
) -> str:
    """Raw HTTP call to Ollama chat endpoint."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": temperature},
    }

    # Debug: log payload structure
    msg_sizes = [(m["role"], len(m["content"])) for m in payload["messages"]]
    logger.info(
        "PAYLOAD: model=%s think=%s format=%s temp=%s | messages: %s | total=%d chars",
        model, payload["think"], payload["format"],
        temperature, msg_sizes, sum(s for _, s in msg_sizes)
    )

    resp = requests.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()

    data = resp.json()
    content = data.get("message", {}).get("content", "")
    return _clean_response(content)


def _clean_response(text: str) -> str:
    """Clean LLM response: strip fences, extract first JSON object."""
    text = text.strip()

    # Strip markdown code fences
    match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    # Extract first complete JSON object
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\" and in_string:
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return text


def _patch_missing_fields(data: dict) -> dict:
    """Infer missing fields from context before pydantic validation.

    Not arbitrary defaults — each inference is logically derived:
    - max_concurrent_positions: count of buy actions in the response
    - rationale: copy from holistic_justification if present
    - analysis_confidence: mean of asset convictions
    """
    g = data.get("global", {})

    # max_concurrent_positions: count buys, min 1
    if "max_concurrent_positions" not in g:
        assets = data.get("assets", [])
        n_buys = sum(1 for a in assets if a.get("action") == "buy")
        g["max_concurrent_positions"] = max(n_buys, 1)
        logger.debug("Patched global.max_concurrent_positions=%d", g["max_concurrent_positions"])

    # Asset rationale: copy from holistic_justification
    for i, asset in enumerate(data.get("assets", [])):
        if "rationale" not in asset and "holistic_justification" in asset:
            asset["rationale"] = asset["holistic_justification"]
            logger.debug("Patched assets[%d].rationale from holistic_justification", i)
        elif "rationale" not in asset:
            asset["rationale"] = f"No explicit rationale for {asset.get('symbol', '?')}"
            logger.debug("Patched assets[%d].rationale with fallback", i)

    # meta.analysis_confidence: mean of convictions
    meta = data.get("meta", {})
    if not isinstance(meta, dict):
        data["meta"] = {}
        meta = data["meta"]
    if "analysis_confidence" not in meta:
        convictions = [a.get("conviction", 5) for a in data.get("assets", []) if isinstance(a.get("conviction"), (int, float))]
        meta["analysis_confidence"] = round(sum(convictions) / len(convictions)) if convictions else 5
        logger.debug("Patched meta.analysis_confidence=%d", meta["analysis_confidence"])

    return data


def _validate_response(raw_text: str):
    """Parse JSON, patch missing fields, validate against GemmaOutput schema."""
    if not raw_text or not raw_text.strip():
        return None, ["Empty response from LLM"]

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return None, [f"JSON parse error: {e}"]

    # Patch commonly omitted fields before strict validation
    data = _patch_missing_fields(data)

    try:
        output = GemmaOutput.parse_raw_response(data)
        return output.model_dump(by_alias=True), []
    except ValidationError as e:
        error_msgs = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            error_msgs.append(f"{loc}: {err['msg']}")
        return None, error_msgs
