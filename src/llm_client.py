"""
LLM client with provider pattern — supports Ollama (Qwen) and Anthropic (Claude).

Providers share the same interface:
    provider.call(system_prompt, user_message) -> dict

The dict always contains: parsed, raw_response, thinking, latency_sec,
retried, success, validation_errors, parse_method, usage (anthropic only).
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from pydantic import ValidationError

from src.schemas import QwenOutput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
# Check docs.anthropic.com/en/docs/about-claude/models if 404
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
PROMPT_VERSION = "v6"
PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_system_prompt(filename: str = "gemma_system_v6.txt") -> str:
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"System prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def ping_ollama(base_url: str = OLLAMA_BASE_URL, timeout: float = 5.0) -> bool:
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=timeout)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def parse_thinking_and_json(text: str) -> Tuple[str, str, str]:
    """Extract thinking and JSON from response.

    Handles 3 formats:
    1. <thinking>...</thinking> {json}  → standard tagged format
    2. text analysis... {json}          → untagged (Claude style)
    3. {json} only                      → no thinking

    Returns (thinking, json_text, parse_method).
    """
    text = text.strip()

    # Case 1: Explicit <thinking> tags
    thinking_match = re.search(
        r'<thinking>(.*?)</thinking>', text, re.DOTALL | re.IGNORECASE
    )
    if thinking_match:
        thinking = thinking_match.group(1).strip()
        json_part = text[thinking_match.end():].strip()
        json_part = re.sub(r'^```(?:json)?\s*', '', json_part)
        json_part = re.sub(r'\s*```$', '', json_part)
        json_text = _extract_last_json_object(json_part)
        if json_text:
            return thinking, json_text, "thinking_tag_found"

    # Case 2 & 3: No tags — find the LAST complete JSON object in the text
    # Everything before it is thinking
    json_text = _extract_last_json_object(text)
    if json_text:
        json_start = text.rfind(json_text)
        thinking = text[:json_start].strip() if json_start > 0 else ""
        # Clean thinking: remove code fences, trailing punctuation
        thinking = re.sub(r'```(?:json)?', '', thinking).strip()
        method = "thinking_tag_found" if thinking else "brace_counting_fallback"
        return thinking, json_text, method

    # Nothing found
    return "", text, "brace_counting_fallback"


def _extract_last_json_object(text: str) -> str:
    """Extract the last complete top-level JSON object from text.

    Searches backwards from the last '}' to find its matching '{'.
    """
    end = text.rfind("}")
    if end == -1:
        return ""

    # Walk backwards to find the matching opening brace
    depth = 0
    in_string = False
    escape = False

    # We need to scan forward from a candidate start to validate.
    # Strategy: find the last '}', then search backwards for candidates.
    for candidate_start in range(end, -1, -1):
        if text[candidate_start] == '{':
            # Try parsing from here to end
            candidate = text[candidate_start:end + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue

    return ""
    json_text = _extract_first_json_object(json_part)
    return thinking, json_text, "brace_counting_fallback"


def _extract_first_json_object(text: str) -> str:
    """Extract first complete top-level JSON object via brace counting."""
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
                return text[start:i + 1]
    return text


def _validate_parsed(data: dict) -> Tuple[Optional[dict], List[str]]:
    """Validate parsed JSON against QwenOutput schema."""
    try:
        output = QwenOutput.parse_raw_response(data)
        return output.model_dump(by_alias=True), []
    except ValidationError as e:
        errors = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            errors.append(f"{loc}: {err['msg']}")
        return None, errors


# ---------------------------------------------------------------------------
# Features to text (shared — Qwen needs text input)
# ---------------------------------------------------------------------------

def features_to_text(features: dict) -> str:
    """Convert anonymized features dict to structured text."""
    lines = []
    ci = features.get("cycle_index", "?")
    lines.append(f"CYCLE {ci}")

    g = features.get("global", {})
    btc = g.get("btc", {})
    dom = g.get("btc_dominance", {})
    t = g.get("time", {})
    pf = g.get("portfolio", {})

    lines.append("")
    lines.append("# GLOBAL")
    lines.append(
        f"BTC: chg_1h={btc.get('chg_1h_pct')}% chg_4h={btc.get('chg_4h_pct')}% "
        f"chg_12h={btc.get('chg_12h_pct')}% chg_24h={btc.get('chg_24h_pct')}% "
        f"chg_7d={btc.get('chg_7d_pct')}%"
    )
    if isinstance(dom, dict) and dom.get("value") is not None:
        lines.append(f"Dominance: {dom['value']}% (chg_24h={dom.get('chg_24h_pct', 0)}%)")
    lines.append(f"Time to close: {t.get('minutes_to_close')} min")

    # Cross-asset overview
    lines.append("")
    lines.append("# CROSS-ASSET OVERVIEW")
    for a in features.get("assets", []):
        aid = a.get("id", a.get("_symbol", "?"))
        lines.append(f"{aid}: 1h={a.get('chg_1h_pct')}% 24h={a.get('chg_24h_pct')}% 7d={a.get('chg_7d_pct')}%")

    # Portfolio (from context_formatter if available, else minimal)
    lines.append("")
    lines.append(f"Portfolio: {pf.get('open_positions', 0)} positions, {pf.get('total_exposure_pct', 0)}% exposure")

    # Assets
    lines.append("")
    lines.append("# ASSETS")
    for a in features.get("assets", []):
        aid = a.get("id", a.get("_symbol", "?"))
        lines.append("")
        lines.append(f"{aid}:")
        lines.append(f"  price={a.get('price')}")
        s = a.get("session", {})
        if s.get("chg_pct") is not None:
            lines.append(f"  session: chg={s['chg_pct']}% range_pos={s.get('range_position')}")
        tr = a.get("trend", {})
        t_parts = []
        for ek, ed in tr.items():
            if isinstance(ed, dict) and ed.get("slope_pct") is not None:
                t_parts.append(f"{ek}_slope={ed['slope_pct']}% dist={ed.get('dist_pct', '?')}%")
        if t_parts:
            lines.append(f"  trend: {' | '.join(t_parts)}")
        r = a.get("regime", {})
        if r.get("adx_1h") is not None:
            lines.append(f"  regime: adx_1h={r['adx_1h']}")
        m = a.get("momentum", {})
        mp = []
        if m.get("rsi_15m") is not None:
            mp.append(f"rsi_15m={m['rsi_15m']}")
        if m.get("rsi_1h") is not None:
            mp.append(f"rsi_1h={m['rsi_1h']}")
        if mp:
            lines.append(f"  momentum: {' '.join(mp)}")
        v = a.get("volatility", {})
        vp = []
        if v.get("atr_15m_pct") is not None:
            vp.append(f"atr_pct={v['atr_15m_pct']}%")
        if v.get("atr_ratio_vs_avg50") is not None:
            vp.append(f"atr_ratio={v['atr_ratio_vs_avg50']}")
        if v.get("bb_position_15m") is not None:
            vp.append(f"bb_pos={v['bb_position_15m']}")
        if vp:
            lines.append(f"  volatility: {' '.join(vp)}")
        vol = a.get("volume", {})
        volp = []
        if vol.get("vol_rel_15m") is not None:
            volp.append(f"vol_rel_15m={vol['vol_rel_15m']}")
        if vol.get("vol_rel_1h") is not None:
            volp.append(f"vol_rel_1h={vol['vol_rel_1h']}")
        if vol.get("vwap_dist_pct") is not None:
            volp.append(f"vwap_dist={vol['vwap_dist_pct']}%")
        if volp:
            lines.append(f"  volume: {' '.join(volp)}")
        st = a.get("structure", {})
        sup = st.get("support", {})
        res = st.get("resistance", {})
        stp = []
        if sup.get("price") is not None:
            stp.append(f"support={sup['price']} ({sup.get('dist_pct','')}% {sup.get('type','')})")
        if res.get("price") is not None:
            stp.append(f"resistance={res['price']} ({res.get('dist_pct','')}% {res.get('type','')})")
        if stp:
            lines.append(f"  structure: {' / '.join(stp)}")
        c = a.get("correlation", {})
        if c.get("corr_btc_24h") is not None:
            lines.append(f"  correlation: corr_btc_24h={c['corr_btc_24h']}")
        cb = a.get("current_bar", {})
        for tfk, cbd in cb.items():
            if cbd and isinstance(cbd, dict):
                cbp = []
                if cbd.get("progress_pct") is not None:
                    cbp.append(f"progress={cbd['progress_pct']}%")
                if cbd.get("move_pct") is not None:
                    cbp.append(f"move={cbd['move_pct']}%")
                if cbd.get("range_vs_atr") is not None:
                    cbp.append(f"range_vs_atr={cbd['range_vs_atr']}")
                if cbp:
                    lines.append(f"  current_bar_{tfk}: {' '.join(cbp)}")
        series = a.get("series_20", {})
        if series:
            lines.append("  series_20:")
            for sk, sv in series.items():
                if sv:
                    if sk == "rsi_15m":
                        fmt = [f"{x:.0f}" if x else "?" for x in sv]
                    elif sk == "vol_rel_15m":
                        fmt = [f"{x:.1f}" if x else "?" for x in sv]
                    elif sk == "ema20_dist_15m":
                        fmt = [f"{x:.2f}" if x else "?" for x in sv]
                    else:
                        fmt = [f"{x:.0f}" if x else "?" for x in sv]
                    lines.append(f"    {sk}: {', '.join(fmt)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ollama Provider
# ---------------------------------------------------------------------------

class OllamaProvider:
    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL,
                 temperature: float = 0.2, thinking: bool = True, timeout: float = 600.0):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.thinking = thinking
        self.timeout = timeout

    def call(self, system_prompt: str, user_message: str) -> Dict[str, Any]:
        result = {
            "parsed": None, "raw_response": None, "thinking": None,
            "validation_errors": [], "latency_sec": 0.0, "retried": False,
            "success": False, "parse_method": None, "usage": None,
        }
        t0 = time.perf_counter()

        # First attempt
        raw, thinking = self._http_call(system_prompt, user_message)
        result["raw_response"] = raw
        result["thinking"] = thinking

        if not thinking:
            # Ollama thinking comes in message.thinking; if empty try parsing from content
            thinking_parsed, json_text, method = parse_thinking_and_json(raw)
            if thinking_parsed:
                result["thinking"] = thinking_parsed
        else:
            _, json_text, method = parse_thinking_and_json(raw)

        result["parse_method"] = method
        parsed_data, errors = self._try_validate(json_text)

        if parsed_data is not None:
            result["parsed"] = parsed_data
            result["success"] = True
            result["latency_sec"] = round(time.perf_counter() - t0, 2)
            return result

        # Retry
        result["retried"] = True
        result["validation_errors"] = errors
        logger.warning("First attempt failed: %s — retrying", errors[:3])

        error_list = "\n".join(f"- {e}" for e in errors[:10])
        retry_msg = (
            f"Your previous JSON was missing required fields:\n{error_list}\n\n"
            f"Regenerate with ALL required fields. Same analysis, strict JSON only.\n\n"
            f"{user_message}"
        )
        retry_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": raw},
            {"role": "user", "content": retry_msg},
        ]
        raw2, _ = self._http_call_messages(retry_messages)
        result["raw_response_retry"] = raw2
        _, json_text2, method2 = parse_thinking_and_json(raw2)
        parsed2, errors2 = self._try_validate(json_text2)

        if parsed2 is not None:
            result["parsed"] = parsed2
            result["success"] = True
            result["parse_method"] = "retry_after_failure"
            result["validation_errors"] = []
        else:
            result["validation_errors"] = errors + errors2
            result["parse_method"] = "retry_failed"
            logger.error("Both attempts failed: %s", (errors + errors2)[:5])

        result["latency_sec"] = round(time.perf_counter() - t0, 2)
        return result

    def _http_call(self, system_prompt, user_content):
        return self._http_call_messages([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ])

    def _http_call_messages(self, messages):
        payload = {
            "model": self.model, "messages": messages,
            "stream": False, "think": self.thinking,
            "format": "json", "options": {"temperature": self.temperature},
        }
        msg_sizes = [(m["role"], len(m["content"])) for m in messages]
        logger.info(
            "PAYLOAD: model=%s prompt=%s think=%s | messages: %s | total=%d chars",
            self.model, PROMPT_VERSION, self.thinking,
            msg_sizes, sum(s for _, s in msg_sizes),
        )
        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        thinking = data.get("message", {}).get("thinking", "")
        return content, thinking

    def _try_validate(self, json_text):
        if not json_text or not json_text.strip():
            return None, ["Empty response"]
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            return None, [f"JSON parse error: {e}"]
        return _validate_parsed(data)

    def get_cost_estimate(self):
        return None  # No cost for local models


# ---------------------------------------------------------------------------
# Anthropic Provider
# ---------------------------------------------------------------------------

class AnthropicProvider:
    def __init__(self, api_key: str, model: str = DEFAULT_ANTHROPIC_MODEL,
                 temperature: float = 0.2, max_tokens: int = 4096):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read = 0
        self.total_cache_write = 0

    def call(self, system_prompt: str, user_message: str) -> Dict[str, Any]:
        result = {
            "parsed": None, "raw_response": None, "thinking": None,
            "validation_errors": [], "latency_sec": 0.0, "retried": False,
            "success": False, "parse_method": None, "usage": None,
        }
        t0 = time.perf_counter()

        logger.info(
            "ANTHROPIC: model=%s | system=%d chars | user=%d chars",
            self.model, len(system_prompt), len(user_message),
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=[{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_message}],
        )

        content = response.content[0].text
        result["raw_response"] = content

        # Track tokens
        usage = response.usage
        cache_read = getattr(usage, 'cache_read_input_tokens', 0) or 0
        cache_write = getattr(usage, 'cache_creation_input_tokens', 0) or 0
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_cache_read += cache_read
        self.total_cache_write += cache_write

        result["usage"] = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": cache_read,
            "cache_creation_tokens": cache_write,
        }

        # System prompt token estimate & cache diagnostics
        # input_tokens = system_tokens + user_tokens, so system ~ input - user_estimate
        user_token_est = len(user_message) // 4  # rough: 1 token ≈ 4 chars
        system_token_est = usage.input_tokens - user_token_est
        self._call_count = getattr(self, '_call_count', 0) + 1

        if self._call_count == 1:
            self._system_hash = hash(system_prompt)
            logger.info(
                "ANTHROPIC tokens: input=%d (system~%d, user~%d) output=%d | "
                "cache_read=%d cache_write=%d | system_hash=%d",
                usage.input_tokens, system_token_est, user_token_est,
                usage.output_tokens, cache_read, cache_write, self._system_hash,
            )
            if system_token_est < 1024:
                logger.warning(
                    "System prompt ~%d tokens — below 1024 minimum for Anthropic cache!",
                    system_token_est,
                )
        else:
            current_hash = hash(system_prompt)
            if current_hash != self._system_hash:
                logger.warning("System prompt hash CHANGED between cycles! Cache will miss.")
            logger.info(
                "ANTHROPIC tokens: input=%d output=%d | cache_read=%d cache_write=%d",
                usage.input_tokens, usage.output_tokens, cache_read, cache_write,
            )

        # Parse thinking and JSON
        thinking, json_text, method = parse_thinking_and_json(content)
        result["thinking"] = thinking
        result["parse_method"] = method

        parsed, errors = self._try_validate(json_text)

        if parsed is not None:
            result["parsed"] = parsed
            result["success"] = True
            result["latency_sec"] = round(time.perf_counter() - t0, 2)
            return result

        # Retry
        result["retried"] = True
        result["validation_errors"] = errors
        logger.warning("Anthropic first attempt failed: %s — retrying", errors[:3])

        error_list = "\n".join(f"- {e}" for e in errors[:10])
        retry_response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=[{
                "type": "text", "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": content},
                {"role": "user", "content": (
                    f"Your JSON was invalid:\n{error_list}\n\n"
                    f"Regenerate with ALL required fields. Strict JSON only."
                )},
            ],
        )

        retry_content = retry_response.content[0].text
        ru = retry_response.usage
        rc_read = getattr(ru, 'cache_read_input_tokens', 0) or 0
        rc_write = getattr(ru, 'cache_creation_input_tokens', 0) or 0
        self.total_input_tokens += ru.input_tokens
        self.total_output_tokens += ru.output_tokens
        self.total_cache_read += rc_read
        self.total_cache_write += rc_write

        _, json_text2, _ = parse_thinking_and_json(retry_content)
        parsed2, errors2 = self._try_validate(json_text2)

        if parsed2 is not None:
            result["parsed"] = parsed2
            result["success"] = True
            result["parse_method"] = "retry_after_failure"
            result["validation_errors"] = []
        else:
            result["validation_errors"] = errors + errors2
            result["parse_method"] = "retry_failed"
            logger.error("Anthropic both attempts failed: %s", (errors + errors2)[:5])

        result["latency_sec"] = round(time.perf_counter() - t0, 2)
        return result

    def _try_validate(self, json_text):
        if not json_text or not json_text.strip():
            return None, ["Empty response"]
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            return None, [f"JSON parse error: {e}"]
        return _validate_parsed(data)

    def get_cost_estimate(self) -> Dict[str, Any]:
        input_cost = self.total_input_tokens * 3.0 / 1_000_000
        output_cost = self.total_output_tokens * 15.0 / 1_000_000
        cache_read_cost = self.total_cache_read * 0.30 / 1_000_000
        cache_write_cost = self.total_cache_write * 3.75 / 1_000_000
        total = input_cost + output_cost + cache_read_cost + cache_write_cost
        total_non_cached = self.total_input_tokens - self.total_cache_read
        cache_rate = self.total_cache_read / (self.total_cache_read + total_non_cached) * 100 if (self.total_cache_read + total_non_cached) > 0 else 0
        return {
            "input_usd": round(input_cost, 4),
            "output_usd": round(output_cost, 4),
            "cache_read_usd": round(cache_read_cost, 4),
            "cache_write_usd": round(cache_write_cost, 4),
            "total_usd": round(total, 4),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "cache_read_tokens": self.total_cache_read,
            "cache_write_tokens": self.total_cache_write,
            "cache_hit_rate_pct": round(cache_rate, 1),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_provider(name: str, **kwargs):
    """Create a provider by name. kwargs passed to constructor."""
    if name == "ollama":
        return OllamaProvider(**kwargs)
    elif name == "anthropic":
        return AnthropicProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {name}")


# ---------------------------------------------------------------------------
# Legacy wrapper (backward compat for test_run.py)
# ---------------------------------------------------------------------------

def call_gemma(
    system_prompt: str,
    user_payload: dict,
    temperature: float = 0.2,
    model: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    timeout: float = 600.0,
    _override_user_content: Optional[str] = None,
) -> Dict[str, Any]:
    """Legacy wrapper — uses OllamaProvider."""
    provider = OllamaProvider(model=model, base_url=base_url, temperature=temperature, timeout=timeout)
    user_msg = _override_user_content if _override_user_content else features_to_text(user_payload)
    result = provider.call(system_prompt, user_msg)
    # Map to legacy format
    return {
        "parsed": result["parsed"],
        "raw_response_first_attempt": result.get("raw_response"),
        "raw_response_retry": result.get("raw_response_retry"),
        "thinking": result.get("thinking"),
        "validation_errors": result.get("validation_errors", []),
        "latency_sec": result["latency_sec"],
        "retried": result["retried"],
        "success": result["success"],
    }
