"""Small, fixed-setting LLM adapter for API-compatible services or Ollama."""

from __future__ import annotations

import json
import os
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
        raise LLMError(f"LLM provider returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LLMError("LLM request failed or timed out") from exc
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
        data = _json_request(
            base_url,
            payload,
            {
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "AIMScribe-ClinicalRAG/0.1",
            },
        )
        text = _api_text(data)
    else:
        raise LLMError("LLM_PROVIDER must be 'api' or 'ollama'")
    if not isinstance(text, str) or not text.strip():
        raise LLMError("LLM provider returned empty text")
    return text
