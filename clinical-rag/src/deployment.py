"""Host startup only: stable working directory and a populated local index."""
from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

_INDEX_LOCK = Lock()
_SETTINGS = (
    "LLM_PROVIDER", "LLM_API_URL", "LLM_MODEL", "LLM_API_KEY", "OLLAMA_URL",
    "LLM_CACHE", "LLM_CACHE_DIR", "AIMS_TELEMETRY", "AIMS_TELEMETRY_PATH",
    "AIMS_LLM_TIMEOUT_SECONDS",
)


def configure_runtime(root: Path, settings) -> None:
    # Existing modules use data/... and .env relative to the working directory.
    # Both local and hosted launches must resolve them to the same app directory.
    os.chdir(root)
    for key in _SETTINGS:
        value = settings.get(key)
        if isinstance(value, (str, int, float, bool)):
            os.environ.setdefault(key, str(value))


def ensure_index(root: Path) -> int:
    """Build only a missing/empty collection; never silently replace an index."""
    import chromadb
    from chromadb.errors import NotFoundError
    from build_index import build_index

    with _INDEX_LOCK:
        path = root / "data" / "index"
        path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(path))
        try:
            count = client.get_collection("clinical_cards").count()
        except NotFoundError:
            count = 0
        if count:
            return count
        count = build_index(root / "data" / "cards.jsonl", path, client=client)
        if not count:
            raise RuntimeError("No verified evidence cards were indexed.")
        return count
