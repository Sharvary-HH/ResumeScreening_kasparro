"""The provider adapter.

Everything provider-specific is behind chat_json(). The HF router speaks the
OpenAI chat-completions API, so pointing this at OpenAI, Together, or a local
vLLM is a base-URL and model-name change - nothing above this file moves.

Contract: chat_json() never raises into the pipeline. It returns a validated
model or None.
"""
import hashlib
import json
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from src import config

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    """Raised at startup only - a missing token is a setup problem, not a
    per-candidate failure."""


class _FatalProviderError(RuntimeError):
    """A bad key, an expired plan, or exhausted credits.

    Retrying these across 50 resumes just wastes wall-clock time and produces 50
    identical failures, so the first one trips a breaker for the rest of the run.
    """


# Set by the pipeline from the CLI flags.
_offline = False
_use_cache = True

# Tripped by the first fatal provider error; short-circuits every later call.
_breaker_lock = threading.Lock()
_breaker_reason: Optional[str] = None


def configure(*, offline: bool = False, use_cache: bool = True) -> None:
    global _offline, _use_cache, _breaker_reason
    _offline = offline
    _use_cache = use_cache
    with _breaker_lock:
        _breaker_reason = None


def _trip_breaker(reason: str) -> None:
    global _breaker_reason
    with _breaker_lock:
        if _breaker_reason is None:
            _breaker_reason = reason


def breaker_reason() -> Optional[str]:
    """The fatal provider error that stopped the run, if one occurred."""
    with _breaker_lock:
        return _breaker_reason


def require_credentials() -> None:
    """Check up front so we fail with a sentence instead of 50 tracebacks."""
    if _offline:
        return
    if not config.LLM_API_KEY:
        raise LLMUnavailable(
            "LLM_API_KEY not set. Run with --offline to replay cached results, "
            "or see README setup."
        )


# --- Cache ------------------------------------------------------------------
# Keyed on model + messages, so the committed cache replays exactly for anyone
# running the same corpus with the same model - no token required.


def _cache_key(messages: List[Dict[str, str]]) -> str:
    payload = config.LLM_MODEL + json.dumps(messages, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(key: str):
    return config.LLM_CACHE_DIR / f"{key}.json"


def _cache_read(key: str) -> Optional[str]:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))["content"]
    except (OSError, ValueError, KeyError):
        return None


def _cache_write(key: str, content: str) -> None:
    try:
        config.LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(key).write_text(
            json.dumps({"model": config.LLM_MODEL, "content": content}, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass  # A cache write failure must never break a run.


# --- Response cleanup -------------------------------------------------------


def _clean_json_text(text: str) -> str:
    """Get from "whatever the model said" to "something json.loads can eat".

    Provider behaviour varies more than the OpenAI-compatible label suggests,
    so this handles inline reasoning tags and markdown fences defensively even
    though the configured model emits neither.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^\s*```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()

    if text.startswith("{"):
        return text

    # Last resort: the outermost {...} span.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


def _extract_content(payload: Dict[str, Any]) -> Tuple[str, str]:
    """Return (content, finish_reason).

    Some providers put chain-of-thought in a separate `reasoning_content` field
    and leave `content` clean. Read `content` only - concatenating the two is
    what produces unparseable output.
    """
    if not isinstance(payload, dict):
        return "", ""

    choices = payload.get("choices") or []
    if not choices:
        return "", ""

    choice = choices[0]
    message = choice.get("message") or {}
    return (message.get("content") or "", choice.get("finish_reason") or "")


# --- HTTP -------------------------------------------------------------------


def _post(body: Dict[str, Any]) -> Dict[str, Any]:
    """POST with backoff on 429/5xx. Raises on unrecoverable failure."""
    url = f"{config.LLM_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    last_error = "unknown error"
    next_delay = 0.0
    for delay in [0] + list(config.HTTP_RETRY_DELAYS):
        # A server-supplied retry delay beats our own guess.
        wait = max(delay, next_delay)
        next_delay = 0.0
        if wait:
            time.sleep(min(wait, config.HTTP_MAX_BACKOFF))
        try:
            response = httpx.post(
                url, headers=headers, json=body, timeout=config.LLM_TIMEOUT
            )
        except httpx.HTTPError as exc:
            last_error = f"network error: {exc}"
            continue

        if response.status_code == 200:
            return response.json()

        if response.status_code == 400:
            # Surfaced to the caller so it can drop response_format and retry.
            raise _BadRequest(response.text[:300])

        if response.status_code in (401, 402, 403):
            # Bad token, no credit, or no access - no amount of retrying helps.
            raise _FatalProviderError(
                f"HTTP {response.status_code}: {_error_message(response)}"
            )

        if response.status_code == 429 or response.status_code >= 500:
            last_error = f"HTTP {response.status_code}: {_error_message(response)}"
            if response.status_code == 429:
                next_delay = _retry_after(response)
            continue

        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")

    raise RuntimeError(f"LLM request failed after retries ({last_error})")


class _BadRequest(RuntimeError):
    pass


def _retry_after(response: "httpx.Response") -> float:
    """Seconds the server asked us to wait, or 0 if it did not say.

    Free tiers are strict and usually tell you exactly how long to back off -
    honouring that is far more effective than a fixed doubling schedule.
    """
    header = response.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass

    # Google returns it in the error body as "Please retry in 13.7s" plus a
    # structured retryDelay field.
    match = re.search(r"retry(?:Delay)?\D{0,6}?(\d+(?:\.\d+)?)\s*s", response.text, re.I)
    return float(match.group(1)) if match else 0.0


def _error_message(response: "httpx.Response") -> str:
    """Pull the provider's own message out of an error body when there is one."""
    try:
        payload = response.json()
    except ValueError:
        return response.text[:200]

    # Google's compatibility layer wraps errors in a list; OpenAI does not.
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        return str(payload)[:200]

    error = payload.get("error", payload)
    if isinstance(error, dict):
        error = error.get("message", error)
    return str(error)[:200]


def _build_body(
    messages: List[Dict[str, str]], *, json_mode: bool, max_tokens: int
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": 0,  # reproducible runs
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    return body


def _call_model(messages: List[Dict[str, str]], max_tokens: int) -> Tuple[str, str]:
    try:
        payload = _post(_build_body(messages, json_mode=True, max_tokens=max_tokens))
    except _BadRequest:
        # Not every HF Inference Provider supports response_format. The schema
        # is in the system prompt too, so dropping it is safe.
        payload = _post(_build_body(messages, json_mode=False, max_tokens=max_tokens))
    return _extract_content(payload)


# --- Public API -------------------------------------------------------------


def chat_json(
    messages: List[Dict[str, str]],
    schema_model: Type[T],
    *,
    max_retries: int = 2,
    errors: Optional[List[str]] = None,
) -> Optional[T]:
    """Call the model and return a validated `schema_model`, or None.

    On a validation failure the error text is fed back to the model, which is
    far more effective than simply asking again. Pass a list as `errors` to
    collect the reason for a None return - a failure nobody can diagnose is
    barely better than a crash.
    """

    def fail(reason: str) -> None:
        if errors is not None:
            errors.append(reason)

    conversation = list(messages)
    max_tokens = config.LLM_MAX_TOKENS

    for attempt in range(max_retries + 1):
        tripped = breaker_reason()
        if tripped:
            fail(f"llm_unavailable: {tripped}")
            return None

        key = _cache_key(conversation)

        content = _cache_read(key) if _use_cache else None
        finish_reason = "cached" if content is not None else ""

        if content is None:
            if _offline:
                fail("llm_unavailable: cache miss in --offline mode")
                return None
            try:
                content, finish_reason = _call_model(conversation, max_tokens)
            except _FatalProviderError as exc:
                _trip_breaker(str(exc))
                fail(f"llm_unavailable: {exc}")
                return None
            except Exception as exc:  # noqa: BLE001 - never raise into the pipeline
                fail(f"llm_error: {exc}")
                return None

            if finish_reason == "length":
                # Truncated output is not worth parsing; ask again with room.
                max_tokens = min(max_tokens * 2, 4000)
                continue

            if _use_cache and content:
                _cache_write(key, content)

        cleaned = _clean_json_text(content)
        try:
            return schema_model.model_validate_json(cleaned)
        except ValidationError as exc:
            if attempt == max_retries or _offline:
                fail(f"llm_invalid_json: {str(exc)[:200]}")
                return None
            conversation = conversation + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        f"Your previous response was invalid: {exc}. "
                        "Return only valid JSON matching the schema."
                    ),
                },
            ]

    fail("llm_error: retries exhausted")
    return None


def schema_instructions(schema_model: Type[BaseModel]) -> str:
    """Schema text for the system prompt.

    Belt and braces alongside response_format - the field descriptions and the
    numeric bounds both carry real information the model needs.
    """
    return json.dumps(schema_model.model_json_schema(), indent=2)
