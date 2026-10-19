"""Build a local Chroma vector index from evidence cards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from schema import Card


def load_cards(path: Path) -> list[Card]:
    cards = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cards.append(Card.model_validate_json(line) if hasattr(Card, "model_validate_json") else Card.parse_raw(line))
    return cards


def card_document(card: Card) -> str:
    evidence = " ".join(item.passage for item in card.evidence)
    return " ".join(
        part
        for part in [card.disease, card.category, card.action or "", evidence]
        if part
    )


def _embedding_values(values: Any) -> Any:
    return values.tolist() if hasattr(values, "tolist") else values


def build_index(
    cards_path: Path,
    persist_dir: Path,
    model_name: str = "all-MiniLM-L6-v2",
    *,
    dev: bool = False,
    rebuild: bool = False,
    embedder: Any = None,
    client: Any = None,
    collection_name: str | None = None,
) -> int:
    """Build or replace a Chroma collection; returns indexed card count."""
    if dev:
        cards_path = cards_path.parent / "sample_cards.jsonl"
        if persist_dir == Path("data/index"):
            persist_dir = Path("data/dev_index")
    if collection_name is None:
        collection_name = "clinical_cards_dev" if dev else "clinical_cards"
    cards = load_cards(cards_path)
    if not dev:
        cards = [card for card in cards if card.verified]
    else:
        invalid_dev = [card.card_id for card in cards if not card.card_id.startswith("DEV-")]
        if invalid_dev:
            raise ValueError(f"DEV cards must use DEV- IDs: {invalid_dev}")
    if embedder is None:
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(model_name)
    if client is None:
        import chromadb

        persist_dir.mkdir(parents=True, exist_ok=True)
        persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(persist_dir))
    if rebuild:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(collection_name)
    if not cards:
        return 0
    documents = [card_document(card) for card in cards]
    embeddings = _embedding_values(embedder.encode(documents))
    collection.upsert(
        ids=[card.card_id for card in cards],
        documents=documents,
        embeddings=embeddings,
        metadatas=[
            {
                "card_id": card.card_id,
                "disease": card.disease,
                "category": card.category,
                "source_title": card.source.title,
                "source_version": card.source.version or "",
                "source_pdf_pages": json.dumps(card.source.pdf_pages),
                "verified": str(card.verified).lower(),
                "dev_only": str(dev).lower(),
            }
            for card in cards
        ],
    )
    return len(cards)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=Path, default=Path("data/cards.jsonl"))
    parser.add_argument("--persist-dir", type=Path, default=Path("data/index"))
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    count = build_index(
        args.cards, args.persist_dir, args.model, dev=args.dev, rebuild=args.rebuild
    )
    print(f"Indexed {count} card(s){' [DEV ONLY]' if args.dev else ''}")


if __name__ == "__main__":
    main()
