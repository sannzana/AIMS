"""Retrieve protocol cards from a local Chroma index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


QUERY_FIELDS = (
    "symptoms",
    "duration",
    "age",
    "sex",
    "history",
)


def _embedding_values(values: Any) -> Any:
    return values.tolist() if hasattr(values, "tolist") else values


def patient_query(patient: dict[str, Any]) -> str:
    """Build a query from allowed patient fields; disease is intentionally ignored."""
    parts = []
    for field in QUERY_FIELDS:
        value = patient.get(field)
        if value not in (None, "", [], {}):
            parts.append(f"{field}: {value}")
    return "; ".join(parts)


def retrieve(
    patient: dict[str, Any],
    top_k: int = 5,
    *,
    persist_dir: Path = Path("data/index"),
    model_name: str = "all-MiniLM-L6-v2",
    embedder: Any = None,
    client: Any = None,
    collection_name: str | None = None,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    if collection_name is None:
        collection_name = "clinical_cards_dev" if persist_dir.name == "dev_index" else "clinical_cards"
    if embedder is None:
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(model_name)
    if client is None:
        import chromadb

        client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(collection_name)
    query = patient_query(patient)
    query_embedding = _embedding_values(embedder.encode([query]))
    result = collection.query(
        query_embeddings=query_embedding, n_results=top_k, include=["metadatas", "distances"]
    )
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    return [
        {
            "card_id": metadata.get("card_id"),
            "similarity_score": 1.0 - float(distance),
            "source": {
                "title": metadata.get("source_title"),
                "version": metadata.get("source_version") or None,
                "pdf_pages": json.loads(metadata.get("source_pdf_pages", "[]")),
            },
            "metadata": metadata,
        }
        for metadata, distance in zip(metadatas, distances)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-json", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    import json

    results = retrieve(json.loads(args.patient_json.read_text(encoding="utf-8")), args.top_k)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
