"""Validate evidence cards and perform normalized source-page checks."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from schema import Card


_FONT_EXTRACTION_MAP = str.maketrans(
    {
        "Ɵ": "ti",
        "Ō": "ft",
        "Ʃ": "tt",
    }
)


def normalize_text(value: str) -> str:
    """Normalize verified extraction artifacts for evidence matching."""
    value = re.sub(r"-\s*\n\s*", "", value)
    value = unicodedata.normalize("NFKC", value)
    value = value.translate(_FONT_EXTRACTION_MAP)
    return re.sub(r"\s+", " ", value).strip().casefold()


def _documents(pages_data: Any) -> list[dict[str, Any]]:
    if isinstance(pages_data, dict):
        return pages_data.get("documents", [])
    return pages_data


def _find_page(
    documents: list[dict[str, Any]], title: str, page_number: int
) -> str | None:
    wanted = title.casefold()
    for document in documents:
        names = {
            str(document.get("title", "")).casefold(),
            Path(str(document.get("pdf", ""))).stem.casefold(),
            str(document.get("pdf", "")).casefold(),
        }
        if wanted not in names:
            continue
        for page in document.get("pages", []):
            if page.get("page") == page_number:
                return str(page.get("text", ""))
    return None


def _validate_card(
    card: Card, documents: list[dict[str, Any]], line_number: int
) -> list[str]:
    errors: list[str] = []
    declared_pages = set(card.pdf_page_numbers)
    for evidence in card.exact_supporting_evidence_passages:
        if evidence.pdf_page <= 0:
            errors.append(
                f"line {line_number} ({card.card_id}): page must be positive"
            )
        if evidence.pdf_page not in declared_pages:
            errors.append(
                f"line {line_number} ({card.card_id}): evidence page "
                f"{evidence.pdf_page} is not declared in pdf_page_numbers"
            )
        page_text = _find_page(
            documents, card.source_pdf_title, evidence.pdf_page
        )
        if page_text is None:
            errors.append(
                f"line {line_number} ({card.card_id}): source page not found: "
                f"{card.source_pdf_title!r} p.{evidence.pdf_page}"
            )
        elif normalize_text(evidence.passage) not in normalize_text(page_text):
            errors.append(
                f"line {line_number} ({card.card_id}): evidence passage "
                f"not found on p.{evidence.pdf_page}"
            )
    return errors


def validate_cards(cards_path: Path, pages_path: Path) -> tuple[int, int]:
    pages_data = json.loads(pages_path.read_text(encoding="utf-8"))
    documents = _documents(pages_data)
    errors: list[str] = []
    card_count = 0
    card_ids: set[str] = set()

    for line_number, raw_line in enumerate(
        cards_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        card_count += 1
        try:
            payload = json.loads(raw_line)
            card = Card.parse_obj(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            errors.append(f"line {line_number}: invalid card: {exc}")
            continue
        if card.card_id in card_ids:
            errors.append(f"line {line_number}: duplicate card_id: {card.card_id}")
        card_ids.add(card.card_id)
        if card.medication is not None or card.model_dump().get("medication_details"):
            print(
                f"REVIEW: line {line_number} ({card.card_id}): medication "
                "details require manual verification against the source."
            )
        errors.extend(_validate_card(card, documents, line_number))

    for error in errors:
        print(f"ERROR: {error}")
    print("manual verification still required")
    print(f"Validated {card_count} card(s); {len(errors)} error(s)")
    return card_count, len(errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=Path, default=Path("data/cards.jsonl"))
    parser.add_argument("--pages", type=Path, default=Path("data/pages.json"))
    args = parser.parse_args()
    try:
        _, errors = validate_cards(args.cards, args.pages)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not read validation inputs: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
