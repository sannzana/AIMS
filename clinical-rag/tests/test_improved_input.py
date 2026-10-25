"""Input aliases must produce identical Improved diagnostic inputs/prompts."""
import json
from pathlib import Path

import pytest
from build_index import load_cards
from disease_triage import triage
from improved_input import normalize_improved_inputs


@pytest.mark.parametrize("age", [32, "32", "32yr", "32 yr", "32 year", "32 years", "32 YRS"])
@pytest.mark.parametrize("duration", ["12hr", "12 hr", "12hrs", "12 hours", "12 h"])
def test_spacing_and_unit_aliases_canonicalize_identically(age, duration):
    raw = {"age": age, "duration": duration, "symptoms": "unchanged", "clinical_findings": "unchanged"}
    actual = normalize_improved_inputs(raw)
    assert actual == {**raw, "age": "32 years", "duration": "12 hours"}
    assert raw["age"] == age and raw["duration"] == duration
    assert normalize_improved_inputs(actual) == actual


def test_explicit_months_do_not_become_years_and_unknown_duration_stays_unknown():
    assert normalize_improved_inputs({"age": "32month", "duration": "12"}) == {
        "age": "32 months", "duration": "12"}
    assert normalize_improved_inputs({"age": "unknown", "duration": "several days"}) == {
        "age": "unknown", "duration": "several days"}


@pytest.mark.parametrize("raw,expected", [("3d", "3 days"), ("2wk", "2 weeks"),
    ("2mos", "2 months"), ("3yrs", "3 years"), ("30min", "30 minutes"), ("45s", "45 seconds")])
def test_other_explicit_duration_units(raw, expected):
    assert normalize_improved_inputs({"duration": raw})["duration"] == expected


def test_user_spacing_variants_reach_triage_as_exactly_the_same_prompt():
    cards = {c.card_id: c for c in load_cards(Path("data/cards.jsonl"))}
    snapshots = []
    for age, duration in (("32yr", "12hr"), ("32 yr", "12 hr")):
        patient = normalize_improved_inputs({"age": age, "duration": duration,
            "symptoms": "Sudden profuse watery rice-water diarrhoea",
            "clinical_findings": "Sunken eyes, weak rapid pulse"})
        captured = []
        def search(case, _cards):
            captured.append(case)
            return ["CHO-001"]
        def model(prompt):
            captured.append(prompt)
            return json.dumps({"candidates": [], "facts": {}})
        triage(patient, cards, candidate_search=search, llm_generate=model)
        snapshots.append(captured)
    assert snapshots[0] == snapshots[1]
    assert snapshots[0][0]["age"] == "32 years"
    assert snapshots[0][0]["duration"] == "12 hours"


def test_improved_ui_normalizes_before_calling_diagnosis(monkeypatch):
    import disease_triage
    from streamlit.testing.v1 import AppTest
    captured = []
    def fake_triage(patient, cards):
        captured.append(patient)
        return {"status": "no_match", "candidates": [], "features": {}, "diagnostics": {}}
    monkeypatch.setattr(disease_triage, "triage", fake_triage)
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    at.radio[0].set_value("improved")
    for widget in at.text_input:
        if widget.label == "Age":
            widget.set_value("32yr")
        if widget.label == "Duration":
            widget.set_value("12hr")
    next(button for button in at.button if button.label == "Generate recommendation").click().run()
    assert not at.exception
    assert captured and captured[0]["age"] == "32 years" and captured[0]["duration"] == "12 hours"


def test_baseline_ui_retains_original_input_notation(monkeypatch):
    import improved_input
    from streamlit.testing.v1 import AppTest
    def forbidden(patient):
        raise AssertionError("Baseline must not normalize Improved input")
    monkeypatch.setattr(improved_input, "normalize_improved_inputs", forbidden)
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"))
    at.session_state["case_patient"] = {"age": "32yr", "duration": "12hr"}
    at.session_state["case_mode"] = "baseline"
    at.session_state["case_results"] = {}
    at.run()
    assert not at.exception
    assert at.session_state["case_patient"] == {"age": "32yr", "duration": "12hr"}
