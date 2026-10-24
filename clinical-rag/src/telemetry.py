"""Opt-in privacy-safe request telemetry for clinical RAG.

Only operational metadata is recorded. No prompts, patient details, credentials,
URLs, model output, or exception messages are persisted.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlsplit


def record_llm_request(url, latency_ms, status, http_status=None, token_usage=None):
    """Append sanitized request metrics when AIMS_TELEMETRY=1."""
    if os.getenv("AIMS_TELEMETRY", "0") != "1":
        return
    try:
        host = (urlsplit(url).hostname or "").lower()
        provider = "groq" if host == "api.groq.com" else (
            "ollama" if host in {"localhost", "127.0.0.1"} else "other")
        event = {
            "provider": provider,
            "duration_ms": float(latency_ms),
            "outcome": str(status) if status in {
                "ok", "http_error", "transport_error", "invalid_json"
            } else "other_error",
            "http_status": int(http_status) if type(http_status) is int else None,
        }
        if isinstance(token_usage, dict):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                n = token_usage.get(key)
                if type(n) is int and n >= 0:
                    event[key] = n
        path = Path(os.getenv("AIMS_TELEMETRY_PATH", "outputs/llm_metrics.jsonl"))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(event, sort_keys=True) + "\n")
    except (OSError, TypeError, ValueError):
        pass  # Telemetry must never block clinical inference.
