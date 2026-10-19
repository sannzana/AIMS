import pytest
from pydantic import ValidationError

from schema import Card


def card_payload() -> dict:
    return {
        "card_id": "dengue-1",
        "disease": "dengue",
        "category": "diagnosis",
        "conditions": {"all": [], "any": [], "exceptions": [], "required_inputs": []},
        "source": {"title": "Dengue protocol", "version": "1", "pdf_pages": [1]},
        "evidence": [{"pdf_page": 1, "passage": "Fever"}],
        "verified": False,
    }


def test_valid_card_schema_v11() -> None:
    card = Card(**card_payload())
    assert card.schema_version == "1.1"
    assert card.medication is None


def test_invalid_category_rejected() -> None:
    payload = card_payload()
    payload["category"] = "treatment"
    with pytest.raises(ValidationError):
        Card(**payload)


def test_unknown_card_field_rejected() -> None:
    payload = card_payload()
    payload["unknown_fields"] = {"source": "guess"}
    with pytest.raises(ValidationError):
        Card(**payload)
