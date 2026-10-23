from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from app import applicability_for, patient_from_inputs
from schema import Card


def test_patient_input_adapter_keeps_case_fields_and_measurements() -> None:
    patient = patient_from_inputs(
        "fever",
        "alert",
        "BP 150/95",
        "2 days",
        "42",
        "F",
        "none",
        "none",
        "penicillin",
    )
    assert patient["age"] == 42
    assert patient["clinical_findings"] == "alert"
    assert patient["measurements"] == "BP 150/95"
    assert patient["allergies"] == "penicillin"


def test_unruled_cards_are_retained_with_unknown_applicability() -> None:
    card = Card(
        card_id="APP-1",
        disease_protocol="Dengue",
        clinical_category="Advice",
        source_pdf_title="Protocol",
        source_version_edition=None,
        pdf_page_numbers=[1],
        exact_supporting_evidence_passages=[{"pdf_page": 1, "passage": "Rest"}],
        verified=True,
    )
    assert applicability_for(card, {"symptoms": "fever"}) is None


def test_structured_rules_are_evaluated_without_filtering_the_card() -> None:
    card = Card(
        card_id="APP-2",
        disease_protocol="Hypertension",
        clinical_category="Diagnosis",
        source_pdf_title="Protocol",
        source_version_edition=None,
        pdf_page_numbers=[1],
        exact_supporting_evidence_passages=[{"pdf_page": 1, "passage": "BP"}],
        verified=True,
        structured_rules={
            "all": [{"field": "measurements.sbp", "op": ">=", "value": 140}],
            "any": [],
            "exceptions": [],
            "required_inputs": ["measurements.sbp"],
        },
    )
    result = applicability_for(card, {"measurements": {"sbp": 150}})
    assert result is not None
    assert result["status"] == "applicable"
