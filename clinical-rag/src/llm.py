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
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LLMError("LLM request failed or timed out") from exc
    except json.JSONDecodeError as exc:
        raise LLMError("LLM provider returned invalid JSON") from exc


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
        payload = {
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
        data = _json_request(base_url, payload, {"Authorization": f"Bearer {api_key}"})
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("API response did not contain generated text") from exc
    else:
        raise LLMError("LLM_PROVIDER must be 'api' or 'ollama'")
    if not isinstance(text, str) or not text.strip():
        raise LLMError("LLM provider returned empty text")
    return text
