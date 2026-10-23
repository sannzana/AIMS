import pytest
from pydantic import ValidationError

from schema import Card


def card_payload() -> dict:
    return {
        "card_id": "dengue-1",
        "disease_protocol": "dengue",
        "clinical_category": "Diagnosis",
        "source_pdf_title": "Dengue protocol",
        "source_version_edition": "5th edition",
        "pdf_page_numbers": [1],
        "exact_supporting_evidence_passages": [
            {"pdf_page": 1, "passage": "Fever"}
        ],
        "verified": False,
    }


def test_valid_card_schema_v11() -> None:
    card = Card(**card_payload())
    assert card.schema_version == "1.1"


def test_unknown_card_field_is_allowed_and_retained() -> None:
    payload = card_payload()
    payload["future_field"] = {"value": 1}
    card = Card(**payload)
    assert card.future_field == {"value": 1}


def test_invalid_category_rejected() -> None:
    payload = card_payload()
    payload["clinical_category"] = "treatment"
    with pytest.raises(ValidationError):
        Card(**payload)


def test_missing_core_field_rejected() -> None:
    payload = card_payload()
    del payload["source_pdf_title"]
    with pytest.raises(ValidationError):
        Card(**payload)


def test_missing_required_version_key_rejected() -> None:
    payload = card_payload()
    del payload["source_version_edition"]
    with pytest.raises(ValidationError):
        Card(**payload)


def test_assessment_legacy_category_is_supported() -> None:
    payload = card_payload()
    payload["clinical_category"] = "Assessment"
    assert Card(**payload).clinical_category == "Assessment"


def test_null_source_version_is_accepted_when_present() -> None:
    payload = card_payload()
    payload["source_version_edition"] = None
    assert Card(**payload).source_version_edition is None
