"""Small, fixed-setting LLM adapter for API-compatible services or Ollama."""

from __future__ import annotations

import json
import os
import hashlib
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


MODEL = "clinical-rag-model"
TEMPERATURE = 0.0
MAX_TOKENS = 1200
TIMEOUT_SECONDS = 30
GROQ_MAX_COMPLETION_TOKENS = 3000


class LLMError(RuntimeError):
    """Raised when the configured provider cannot produce a response."""


class LLMRateLimitError(LLMError):
    """Provider rate limit; optional server-requested retry interval."""
    def __init__(self, retry_after: str = ""):
        super().__init__("LLM provider returned HTTP 429 (rate limit)")
        try:
            self.retry_after = min(20.0, max(0.0, float(retry_after)))
        except (TypeError, ValueError):
            self.retry_after = None


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _json_request(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            delay = exc.headers.get("Retry-After", "") if exc.headers else ""
            raise LLMRateLimitError(delay) from exc
        raise LLMError(f"LLM provider returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise LLMError("LLM API transport failed or the remote connection was closed") from exc
    except json.JSONDecodeError as exc:
        raise LLMError("LLM provider returned invalid JSON") from exc


def _is_groq_endpoint(url: str) -> bool:
    return "api.groq.com" in url.lower()


def _api_text(data: dict[str, Any]) -> str:
    try:
        choices = data["choices"]
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise LLMError("API response did not contain a valid choice") from exc

    if finish_reason == "length":
        raise LLMError(
            "LLM response was truncated before completing the recommendation JSON"
        )
    try:
        text = choice["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise LLMError("API response did not contain generated message content") from exc
    if not isinstance(text, str) or not text.strip():
        raise LLMError("LLM provider returned empty message content")
    return text


def generate(prompt: str) -> str:
    """Generate text without logging the prompt, credentials, or response."""
    _load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "api").lower()
    model = os.getenv("LLM_MODEL", MODEL)
    # Same patient and evidence may be requested repeatedly during an evaluation.
    # Cache only successful outputs; do not store API keys or patient text in filenames.
    cache_path = Path(os.getenv("LLM_CACHE_DIR", ".cache/llm_responses"))
    cache_enabled = os.getenv("LLM_CACHE", "1") == "1"
    endpoint = (os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
                if provider == "api" else os.getenv("OLLAMA_URL", "http://localhost:11434"))
    cache_key = hashlib.sha256(json.dumps([provider, endpoint, model, prompt], ensure_ascii=False).encode("utf-8")).hexdigest()
    cache_file = cache_path / (cache_key + ".json")
    if cache_enabled and cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(cached, str) and cached.strip():
                return cached
        except (ValueError, OSError):
            pass
    if provider == "ollama":
        base_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
        }
        data = _json_request(f"{base_url}/api/generate", payload, {})
        text = data.get("response")
    elif provider == "api":
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise LLMError("LLM_API_KEY is required for the API provider")
        base_url = os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
        payload: dict[str, Any] = {
            "model": model,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only the requested JSON object.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        if _is_groq_endpoint(base_url):
            payload.pop("max_tokens")
            payload.update(
                {
                    "reasoning_effort": "low",
                    "response_format": {"type": "json_object"},
                    "max_completion_tokens": GROQ_MAX_COMPLETION_TOKENS,
                }
            )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "AIMScribe-ClinicalRAG/0.1",
        }
        # Do not repeatedly retry 429: capped to two retries, honour Retry-After.
        for attempt in range(3):
            try:
                data = _json_request(base_url, payload, headers)
                break
            except LLMRateLimitError as exc:
                if attempt == 2:
                    raise
                time.sleep(exc.retry_after if exc.retry_after is not None else 2 ** attempt)
        text = _api_text(data)
    else:
        raise LLMError("LLM_PROVIDER must be 'api' or 'ollama'")
    if not isinstance(text, str) or not text.strip():
        raise LLMError("LLM provider returned empty text")
    if cache_enabled:
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(text), encoding="utf-8")
        except OSError:
            pass
    return text
