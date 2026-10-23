"""Build a local Chroma vector index from evidence cards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from schema import Card

EMBED_FIELDS = (
    "disease_protocol",
    "clinical_category",
    "card_title",
    "clinical_classification_presentation",
    "relevant_symptoms_or_prerequisites",
    "conditions_decision_logic",
    "if_else_conditions_and_corresponding_actions",
    "exceptions_contraindications",
    "recommended_action",
    "table_dependent_rules",
    "exclusions_or_missing_information",
    "medication_details",
    "additional_information",
    "exact_supporting_evidence_passages",
)


def _readable(value: Any) -> str:
    if value is None or value == "" or value == [] or value == {}:
        return ""
    if isinstance(value, dict):
        return "; ".join(
            f"{key}: {text}"
            for key, item in sorted(value.items())
            if (text := _readable(item))
        )
    if isinstance(value, (list, tuple, set)):
        return "; ".join(text for item in value if (text := _readable(item)))
    return str(value)


def load_cards(path: Path) -> list[Card]:
    cards = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cards.append(Card.model_validate_json(line) if hasattr(Card, "model_validate_json") else Card.parse_raw(line))
    return cards


def card_document(card: Card) -> str:
    data = card.model_dump()
    return " ".join(
        f"{field}: {text}"
        for field in EMBED_FIELDS
        if (text := _readable(data.get(field)))
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
                "disease_protocol": card.disease_protocol,
                "clinical_category": card.clinical_category,
                "source_pdf_title": card.source_pdf_title,
                "source_version_edition": card.source_version_edition or "",
                "pdf_page_numbers": json.dumps(card.pdf_page_numbers),
                # Legacy retrieval metadata names remain scalar and readable.
                "source_title": card.source_pdf_title,
                "source_version": card.source_version_edition or "",
                "source_pdf_pages": json.dumps(card.pdf_page_numbers),
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
