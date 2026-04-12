"""
Ollama wrapper for Gemma 4 26B.

Strategy for schema compliance:
1. Short system prompt (rules only, no verbose explanations)
2. JSON Schema in Ollama `format` param (structural constraint at token level)
3. Few-shot example in conversation history (format by mimicry)
4. Compact user message (data only, no schema repetition)
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


def load_system_prompt(filename: str = "gemma_system_v2.txt") -> str:
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
# JSON Schema for Ollama structured output
# ---------------------------------------------------------------------------

_OLLAMA_JSON_SCHEMA = {
    "type": "object",
    "required": ["global", "assets", "meta"],
    "properties": {
        "global": {
            "type": "object",
            "required": [
                "btc_regime", "market_mode", "market_coherence", "sector_flow",
                "relative_strength_ranking", "risk_adjustment",
                "max_concurrent_positions", "rationale"
            ],
            "properties": {
                "btc_regime": {"type": "string", "enum": ["trending_up", "trending_down", "range", "chaotic"]},
                "market_mode": {"type": "string", "enum": ["risk_on", "risk_off", "rotation", "neutral"]},
                "market_coherence": {"type": "string", "enum": ["aligned", "diverging", "rotating"]},
                "sector_flow": {"type": "string", "enum": ["btc_led", "alt_led", "mixed", "defensive"]},
                "relative_strength_ranking": {"type": "array", "items": {"type": "string"}, "minItems": 5, "maxItems": 5},
                "risk_adjustment": {"type": "string", "enum": ["normal", "defensive", "aggressive"]},
                "max_concurrent_positions": {"type": "integer"},
                "rationale": {"type": "string"},
            },
        },
        "assets": {
            "type": "array",
            "minItems": 5,
            "maxItems": 5,
            "items": {
                "type": "object",
                "required": [
                    "symbol", "regime", "setup", "action", "conviction",
                    "relative_strength_rank", "entry_zone", "atr_stop_multiplier",
                    "atr_tp_multiplier", "expected_horizon_hours",
                    "holistic_justification", "rationale"
                ],
                "properties": {
                    "symbol": {"type": "string"},
                    "regime": {"type": "string", "enum": ["trending_up", "trending_down", "range", "chaotic"]},
                    "setup": {"type": "string", "enum": ["breakout", "pullback", "mean_reversion", "none"]},
                    "action": {"type": "string", "enum": ["buy", "close", "hold", "skip"]},
                    "conviction": {"type": "integer"},
                    "relative_strength_rank": {"type": "integer"},
                    "entry_zone": {
                        "oneOf": [
                            {"type": "null"},
                            {"type": "object", "properties": {"min": {"type": "number"}, "max": {"type": "number"}}, "required": ["min", "max"]},
                        ]
                    },
                    "atr_stop_multiplier": {"oneOf": [{"type": "null"}, {"type": "number"}]},
                    "atr_tp_multiplier": {"oneOf": [{"type": "null"}, {"type": "number"}]},
                    "expected_horizon_hours": {"oneOf": [{"type": "null"}, {"type": "integer"}]},
                    "holistic_justification": {"type": "string"},
                    "rationale": {"type": "string"},
                },
            },
        },
        "meta": {
            "type": "object",
            "required": ["analysis_confidence"],
            "properties": {
                "analysis_confidence": {"type": "integer"},
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Few-shot example (conversation history teaches format by mimicry)
# ---------------------------------------------------------------------------

_FEW_SHOT_USER = '{"cycle_index":100,"timeframe_reference":"15m","global":{"btc":{"price":68000,"chg_1h_pct":0.3,"chg_24h_pct":1.2},"btc_dominance":{"value":55},"time":{"minutes_to_close":300},"portfolio":{"open_positions":0,"total_exposure_pct":0}},"assets":[{"id":"ASSET_A","price":68000,"session":{"chg_pct":0.5},"trend":{"ema20_15m":{"slope_pct":0.02,"dist_pct":0.1}},"regime":{"adx_1h":22},"momentum":{"rsi_15m":55,"rsi_1h":52},"volatility":{"atr_15m_pct":0.2,"atr_ratio_vs_avg50":1.1,"bb_position_15m":0.6},"volume":{"vol_rel_15m":1.0},"correlation":{"corr_btc_24h":null}},{"id":"ASSET_B","price":3500,"session":{"chg_pct":0.8},"trend":{"ema20_15m":{"slope_pct":0.05,"dist_pct":-0.2}},"regime":{"adx_1h":28},"momentum":{"rsi_15m":48,"rsi_1h":55},"volatility":{"atr_15m_pct":0.3,"atr_ratio_vs_avg50":1.2,"bb_position_15m":0.4},"volume":{"vol_rel_15m":0.9},"correlation":{"corr_btc_24h":0.7}},{"id":"ASSET_C","price":150,"session":{"chg_pct":-0.3},"trend":{"ema20_15m":{"slope_pct":-0.01,"dist_pct":-0.5}},"regime":{"adx_1h":18},"momentum":{"rsi_15m":42,"rsi_1h":45},"volatility":{"atr_15m_pct":0.4,"atr_ratio_vs_avg50":0.9,"bb_position_15m":0.3},"volume":{"vol_rel_15m":0.7},"correlation":{"corr_btc_24h":0.65}},{"id":"ASSET_D","price":0.6,"session":{"chg_pct":-0.8},"trend":{"ema20_15m":{"slope_pct":-0.03,"dist_pct":-1.0}},"regime":{"adx_1h":15},"momentum":{"rsi_15m":35,"rsi_1h":38},"volatility":{"atr_15m_pct":0.2,"atr_ratio_vs_avg50":0.8,"bb_position_15m":0.15},"volume":{"vol_rel_15m":0.5},"correlation":{"corr_btc_24h":0.72}},{"id":"ASSET_E","price":600,"session":{"chg_pct":-1.2},"trend":{"ema20_15m":{"slope_pct":-0.04,"dist_pct":-1.5}},"regime":{"adx_1h":30},"momentum":{"rsi_15m":30,"rsi_1h":32},"volatility":{"atr_15m_pct":0.3,"atr_ratio_vs_avg50":1.4,"bb_position_15m":0.05},"volume":{"vol_rel_15m":1.8},"correlation":{"corr_btc_24h":0.8}}]}'

_FEW_SHOT_ASSISTANT = json.dumps({
    "global": {
        "btc_regime": "range",
        "market_mode": "neutral",
        "market_coherence": "diverging",
        "sector_flow": "mixed",
        "relative_strength_ranking": ["ASSET_B", "ASSET_A", "ASSET_C", "ASSET_D", "ASSET_E"],
        "risk_adjustment": "normal",
        "max_concurrent_positions": 3,
        "rationale": "BTC ranging, B shows strongest momentum with pullback to EMA20."
    },
    "assets": [
        {"symbol": "ASSET_A", "regime": "range", "setup": "none", "action": "skip", "conviction": 4, "relative_strength_rank": 2, "entry_zone": None, "atr_stop_multiplier": None, "atr_tp_multiplier": None, "expected_horizon_hours": None, "holistic_justification": "B offers stronger setup at rank 1.", "rationale": "Sideways, no trigger."},
        {"symbol": "ASSET_B", "regime": "trending_up", "setup": "pullback", "action": "buy", "conviction": 7, "relative_strength_rank": 1, "entry_zone": {"min": 3490, "max": 3510}, "atr_stop_multiplier": 1.5, "atr_tp_multiplier": 3.0, "expected_horizon_hours": 4, "holistic_justification": "Strongest asset, BTC stable enough.", "rationale": "Pullback to EMA20 with RSI 48, volume cooling."},
        {"symbol": "ASSET_C", "regime": "range", "setup": "none", "action": "skip", "conviction": 2, "relative_strength_rank": 3, "entry_zone": None, "atr_stop_multiplier": None, "atr_tp_multiplier": None, "expected_horizon_hours": None, "holistic_justification": "Mid-pack, no edge vs B.", "rationale": "ADX 18, sideways."},
        {"symbol": "ASSET_D", "regime": "range", "setup": "none", "action": "skip", "conviction": 1, "relative_strength_rank": 4, "entry_zone": None, "atr_stop_multiplier": None, "atr_tp_multiplier": None, "expected_horizon_hours": None, "holistic_justification": "Rank 4, R2 blocks buy.", "rationale": "Weak momentum."},
        {"symbol": "ASSET_E", "regime": "trending_down", "setup": "none", "action": "skip", "conviction": 0, "relative_strength_rank": 5, "entry_zone": None, "atr_stop_multiplier": None, "atr_tp_multiplier": None, "expected_horizon_hours": None, "holistic_justification": "Weakest, high vol selloff.", "rationale": "Below all EMAs, ATR expanding."},
    ],
    "meta": {"analysis_confidence": 6}
})


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
    user_content = json.dumps(user_payload, default=str)

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

    error_msg = "; ".join(errors[:5])  # Limit error length
    retry_user = (
        f"SCHEMA ERROR: {error_msg}. "
        f"Output the JSON with keys: global, assets (5 items), meta. "
        f"Each asset needs: symbol, regime, setup, action, conviction, "
        f"relative_strength_rank, entry_zone, atr_stop_multiplier, "
        f"atr_tp_multiplier, expected_horizon_hours, holistic_justification, rationale.\n\n"
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
            {"role": "user", "content": _FEW_SHOT_USER},
            {"role": "assistant", "content": _FEW_SHOT_ASSISTANT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": temperature},
        "format": _OLLAMA_JSON_SCHEMA,
    }

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

    # Extract first complete JSON object (ignore trailing text)
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


def _validate_response(raw_text: str):
    """Parse JSON and validate against GemmaOutput schema."""
    if not raw_text or not raw_text.strip():
        return None, ["Empty response from LLM"]

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return None, [f"JSON parse error: {e}"]

    try:
        output = GemmaOutput.parse_raw_response(data)
        return output.model_dump(by_alias=True), []
    except ValidationError as e:
        error_msgs = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            error_msgs.append(f"{loc}: {err['msg']}")
        return None, error_msgs
