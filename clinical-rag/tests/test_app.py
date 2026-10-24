from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from app import (
    _render_recommendations,
    diagnosis_from_triage,
    applicability_for,
    patient_from_inputs,
    recommendation_for_patient,
)
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


def test_recommendation_adapter_passes_selected_mode_and_patient_to_engine() -> None:
    captured = {}
    cards = {}
    patient = {"symptoms": "fever"}

    def mock_recommend(patient_arg, **kwargs):
        captured["patient"] = patient_arg
        captured.update(kwargs)
        return {
            "mode": kwargs["mode"],
            "probable_diagnosis": [],
            "investigations": [],
            "medication": [],
            "advice": [],
            "missing_info": ["history"],
        }

    result = recommendation_for_patient(
        patient,
        "improved",
        7,
        cards,
        recommend_fn=mock_recommend,
    )

    assert result["mode"] == "improved"
    assert captured["patient"] == patient
    assert captured["mode"] == "improved"
    assert captured["top_k"] == 7
    assert captured["cards"] is cards


def test_recommendation_renderer_shows_sections_citations_and_missing_info() -> None:
    class FakeStreamlit:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            def record(*args, **kwargs):
                self.calls.append((name, args, kwargs))

            return record

    st = FakeStreamlit()
    _render_recommendations(
        st,
        {
            "probable_diagnosis": [
                {
                    "text": "Probable dengue",
                    "reason": "Fever pattern",
                    "evidence": [{"card_id": "DEN-1", "pdf_page": 12}],
                }
            ],
            "investigations": [],
            "medication": [],
            "advice": [],
            "missing_info": ["Platelet count"],
        },
    )

    rendered = [args[0] for name, args, _ in st.calls if name in {"markdown", "write", "warning"}]
    assert any("Probable dengue" in text for text in rendered)
    assert any("DEN-1" in text and "PDF page 12" in text for text in rendered)
    assert "Platelet count" in rendered


def test_triage_candidate_displays_without_second_llm_call():
    card = Card(
        card_id="SRC-1", disease_protocol="Protocol Alpha",
        clinical_category="Diagnosis", card_title="Clinical Case Definition",
        source_pdf_title="Protocol", source_version_edition=None,
        pdf_page_numbers=[4],
        exact_supporting_evidence_passages=[{"pdf_page": 4, "passage": "Evidence"}],
        verified=True,
    )
    ranking = {"candidates": [{"card_id": "SRC-1", "disease": "Protocol Alpha",
             "matched_features": ["patient-reported finding"], "missing": []}]}
    result = diagnosis_from_triage(ranking, {"SRC-1": card})
    assert len(result["probable_diagnosis"]) == 1
    assert result["probable_diagnosis"][0]["evidence"] == [
        {"card_id": "SRC-1", "pdf_page": 4}]
    assert "not confirmed" in result["probable_diagnosis"][0]["text"]


def test_triage_display_rejects_invented_protocol():
    ranking = {"candidates": [{"card_id": "missing", "disease": "Invented",
              "matched_features": ["finding"]}]}
    assert diagnosis_from_triage(ranking, {})["probable_diagnosis"] == []
