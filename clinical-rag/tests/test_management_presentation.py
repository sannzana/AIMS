"""Presentation changes must retain source entries and leave decisions intact."""
import copy
import json
from pathlib import Path

import retrieve
from build_index import load_cards
from management_presentation import split_conditional_sources, conditional_message
from recommend import recommend_management

CARDS = {c.card_id: c for c in load_cards(Path("data/cards.jsonl"))}


def _result(monkeypatch, facts):
    pool = {"Investigation": [], "Medication": ["CHO-004", "CHO-005", "CHO-020", "CHO-021", "CHO-018"],
            "Advice": ["CHO-012", "CHO-010"]}
    monkeypatch.setattr(retrieve, "retrieve_category", lambda p, c, **kw: [{"card_id": cid} for cid in pool[c]])
    def model(prompt):
        return json.dumps({"facts": {}}) if prompt.startswith("Extract patient facts") else json.dumps({"missing_info": ["review"]})
    return recommend_management({"age": "37 years", "_structured_facts": facts}, cards=CARDS,
                                diseases=["Cholera"], llm_generate=model)


def test_relevant_conditional_option_and_unknown_complications_are_separate(monkeypatch):
    result = _result(monkeypatch, {"absent_or_weak_pulse": True, "patient_age_group": "adult"})
    before = copy.deepcopy(result)
    primary, additional = split_conditional_sources(result["source_review_options"], result["management_diagnostics"])
    assert "CHO-005" in {entry["card_id"] for entry in primary}
    assert {"CHO-020", "CHO-018", "CHO-012"} <= {entry["card_id"] for entry in additional}
    assert "CHO-021" not in {entry["card_id"] for entry in primary + additional}  # already excluded by existing adult facts
    assert result == before
    expected = {entry["card_id"] for entry in result["source_review_options"] if entry["conditional_rules"]}
    assert {entry["card_id"] for entry in primary + additional} == expected


def test_pending_classification_sources_remain_visible_with_consistent_message(monkeypatch):
    result = _result(monkeypatch, {})
    primary, _ = split_conditional_sources(result["source_review_options"], result["management_diagnostics"])
    assert "CHO-004" in {entry["card_id"] for entry in primary}
    old = next(text for text in result["missing_info"] if text.startswith("Not shown until "))
    rewritten = conditional_message(old)
    assert rewritten.startswith("Not selected for the main recommendation until ")
    assert "Conditional source options remain available" in rewritten
    assert old in result["missing_info"]  # backend decision/message not mutated


def test_normal_messages_and_unknown_entries_are_not_silently_removed():
    assert conditional_message("A source restriction applies.") == "A source restriction applies."
    entry = {"card_id": "SYNTHETIC", "supported_rules": [], "conditional_rules": [{"condition": None}]}
    assert split_conditional_sources([entry], {}) == ([], [entry])


def test_known_instruction_condition_can_be_grouped_without_reading_disease_names():
    entry = {"card_id": "RENAMED", "supported_rules": [], "conditional_rules": [{
        "condition": {"all": [{"field": "marker_a", "operator": "=", "value": True}]},
        "matched_fields": ["marker_a"], "prerequisites": [], "unsupported_rule": False}]}
    assert split_conditional_sources([entry], {}) == ([entry], [])


def test_improved_ui_compacts_additional_sources_and_preserves_matching_guidance(monkeypatch):
    from streamlit.testing.v1 import AppTest
    result = _result(monkeypatch, {"absent_or_weak_pulse": True})
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"))
    for key, value in {"case_patient": {"age": "37 years"}, "case_mode": "improved", "case_diseases": ["Cholera"],
        "working_disease": "Cholera", "case_management_requested": True,
        "case_results": {"Diagnosis": {"probable_diagnosis": []}, "Management": result}}.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception
    text = "\n".join(item.value for item in at.markdown)
    assert "Start intravenous fluids immediately" in text
    assert any("Additional source material" in item.label for item in at.expander)
    assert "Unselected source alternatives" not in text  # no unrelated age display change reintroduced
