"""Regression tests for management evidence boundaries and source facts.

The former _category_relevance test targeted a removed heuristic ranking API.
These assertions cover the current disease-agnostic management contract instead.
"""
from pathlib import Path
from types import SimpleNamespace

from build_index import load_cards
from recommend import (
    RecommendationResponse,
    RecommendationItem,
    MedicationRecommendation,
    _structured_patient_context,
    _enforce_management_evidence_boundary,
)

CARDS = {c.card_id: c for c in load_cards(Path(__file__).parents[1] / "data" / "cards.jsonl")}


def _item(card_id):
    return SimpleNamespace(evidence=[SimpleNamespace(card_id=card_id)])


def test_management_rejects_cross_protocol_investigation():
    source = next(c for c in CARDS.values() if c.clinical_category == "Investigation")
    other = next(c for c in CARDS.values()
                 if c.clinical_category == "Investigation"
                 and c.disease_protocol != source.disease_protocol)
    response = SimpleNamespace(investigations=[_item(source.card_id), _item(other.card_id)],
                               medication=[], advice=[])
    validated, rejected = _enforce_management_evidence_boundary(
        response, {source.card_id: source, other.card_id: other}, source.disease_protocol)
    assert len(validated.investigations) == 1
    assert validated.investigations[0].evidence[0].card_id == source.card_id
    assert len(rejected) == 1


def test_management_rejects_wrong_category_even_when_same_protocol():
    source = next(c for c in CARDS.values() if c.clinical_category == "Advice")
    response = SimpleNamespace(investigations=[_item(source.card_id)], medication=[], advice=[])
    validated, rejected = _enforce_management_evidence_boundary(
        response, {source.card_id: source}, source.disease_protocol)
    assert validated.investigations == []
    assert len(rejected) == 1


def test_only_source_extracted_facts_feed_applicable_rules():
    case = {"symptoms": "A patient narrative",
            "_structured_facts": {"days_since_illness_onset": 3.0,
                                  "patient_has_fever": False, "wbc_count": 3800}}
    normalized = _structured_patient_context(case)
    assert normalized["days_since_illness_onset"] == 3.0
    assert normalized["patient_has_fever"] is False
    assert normalized["wbc_count"] == 3800


def test_no_inferred_fever_without_structured_evidence():
    assert "patient_has_fever" not in _structured_patient_context({"symptoms": "fever"})
    assert "patient_has_fever" not in _structured_patient_context({"symptoms": "no fever"})


def test_real_corpus_card_count_and_source_integrity():
    assert len(CARDS) == 108
    assert len(CARDS) == len(set(CARDS))
    assert all(c.card_id and c.disease_protocol for c in CARDS.values())


def test_unknown_medication_prerequisite_is_conditional_not_unconditional():
    card = CARDS["CHO-020"]
    response = RecommendationResponse(
        medication=[MedicationRecommendation(
            text="Administer IV dextrose now.",
            reason="Treat hypoglycaemia.",
            evidence=[{"card_id": card.card_id, "pdf_page": card.pdf_page_numbers[0]}],
        )],
        advice=[],
        investigations=[],
        probable_diagnosis=[],
    )
    validated, rejected = _enforce_management_evidence_boundary(
        response, {card.card_id: card}, "Cholera", patient_context={"age": 34}
    )
    assert not rejected
    assert validated.medication
    assert "Conditional source guidance" in validated.medication[0].reason
    assert any("blood_glucose" in item for item in validated.missing_info)


def test_unknown_discharge_prerequisite_is_conditional_not_unconditional():
    card = CARDS["CHO-012"]
    response = RecommendationResponse(
        medication=[],
        advice=[RecommendationItem(
            text="Discharge the patient.",
            reason="Discharge criteria are met.",
            evidence=[{"card_id": card.card_id, "pdf_page": card.pdf_page_numbers[0]}],
        )],
        investigations=[],
        probable_diagnosis=[],
    )
    validated, rejected = _enforce_management_evidence_boundary(
        response, {card.card_id: card}, "Cholera", patient_context={}
    )
    assert not rejected
    assert validated.advice
    assert "Conditional source guidance" in validated.advice[0].reason
    assert any("no_signs_of_dehydration" in item for item in validated.missing_info)
