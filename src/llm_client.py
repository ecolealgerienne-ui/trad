"""
Ollama wrapper for Gemma 4 26B.

- Calls POST http://localhost:11434/api/chat with format:"json" + think:false
- Validates response via pydantic GemmaOutput
- Retry once on validation failure with error feedback + original data
- Returns dict with parsed output + debug metadata (latency, raw responses, errors)
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


def call_gemma(
    system_prompt: str,
    user_payload: dict,
    temperature: float = 0.2,
    model: str = "gemma4:26b",
    base_url: str = OLLAMA_BASE_URL,
    timeout: float = 300.0,
) -> Dict[str, Any]:
    """Call Gemma via Ollama and return validated output + metadata.

    Returns dict with keys:
        - parsed: dict (the validated GemmaOutput as dict), or None if failed
        - raw_response_first_attempt: str (always present)
        - raw_response_retry: str or None
        - validation_errors: list of str (empty if OK)
        - latency_sec: float (total, including retry)
        - retried: bool
        - success: bool
    """
    # Prefix user message with compact schema reminder — Gemma prioritizes
    # the last thing it sees over a long system prompt
    user_content = (
        'Analyze this market data and respond with EXACTLY this JSON structure:\n'
        '{"global": {"btc_regime": "trending_up|trending_down|range|chaotic", '
        '"market_mode": "risk_on|risk_off|rotation|neutral", '
        '"market_coherence": "aligned|diverging|rotating", '
        '"sector_flow": "btc_led|alt_led|mixed|defensive", '
        '"relative_strength_ranking": ["ASSET_X","ASSET_X","ASSET_X","ASSET_X","ASSET_X"], '
        '"risk_adjustment": "normal|defensive|aggressive", '
        '"max_concurrent_positions": 1-5, '
        '"rationale": "one sentence"}, '
        '"assets": [{"symbol": "ASSET_A", "regime": "...", "setup": "breakout|pullback|mean_reversion|none", '
        '"action": "buy|close|hold|skip", "conviction": 0-10, "relative_strength_rank": 1-5, '
        '"entry_zone": {"min": N, "max": N}|null, "atr_stop_multiplier": N|null, '
        '"atr_tp_multiplier": N|null, "expected_horizon_hours": 1-8|null, '
        '"holistic_justification": "...", "rationale": "..."}, ... for all 5 assets], '
        '"meta": {"analysis_confidence": 0-10}}\n'
        'Rules: if action="buy" then entry_zone/multipliers/horizon are REQUIRED. '
        'If action="skip"|"hold"|"close" then those fields MUST be null. '
        'All 5 assets must appear in order A,B,C,D,E.\n\n'
        'DATA:\n'
        + json.dumps(user_payload, default=str)
    )

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

    error_msg = "; ".join(errors)
    retry_user = (
        f"Your previous output violated the schema: {error_msg}.\n\n"
        f"Here is the data again. Output ONLY the JSON matching the schema "
        f"from the system prompt. No other text.\n\n"
        f"{user_content}"
    )

    raw_retry = _call_ollama(system_prompt, retry_user, temperature, model, base_url, timeout)
    result["raw_response_retry"] = raw_retry

    parsed_retry, errors_retry = _validate_response(raw_retry)

    if parsed_retry is not None:
        result["parsed"] = parsed_retry
        result["success"] = True
        result["validation_errors"] = []  # Cleared on success
        result["latency_sec"] = round(time.perf_counter() - t_start, 2)
        return result

    # --- Both attempts failed ---
    all_errors = errors + errors_retry
    result["validation_errors"] = all_errors
    result["latency_sec"] = round(time.perf_counter() - t_start, 2)
    logger.error("Both attempts failed validation: %s", all_errors)

    return result


def _call_ollama(
    system_prompt: str,
    user_content: str,
    temperature: float,
    model: str,
    base_url: str,
    timeout: float,
) -> str:
    """Raw HTTP call to Ollama chat endpoint. Returns response text."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": temperature},
        "format": "json",
    }

    resp = requests.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()

    data = resp.json()
    content = data.get("message", {}).get("content", "")

    # Strip markdown code fences if present (```json ... ```)
    content = _strip_code_fences(content)

    return content


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences that some models wrap JSON in."""
    text = text.strip()
    # Match ```json ... ``` or ``` ... ```
    match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _validate_response(raw_text: str):
    """Parse JSON and validate against GemmaOutput schema.

    Returns (parsed_dict, []) on success, or (None, [error_strings]) on failure.
    """
    if not raw_text or not raw_text.strip():
        return None, ["Empty response from LLM"]

    # Step 1: JSON parse
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return None, [f"JSON parse error: {e}"]

    # Step 2: Pydantic validation
    try:
        output = GemmaOutput.parse_raw_response(data)
        return output.model_dump(by_alias=True), []
    except ValidationError as e:
        error_msgs = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            error_msgs.append(f"{loc}: {err['msg']}")
        return None, error_msgs
