"""Presentation must preserve instruction scope, doses, source links and status."""
import copy

import pytest
from management_display import instruction_groups, readable_instruction
from management_presentation import guidance_changes
from management_clarification import apply_answers, questions
from test_management_presentation import _result, CARDS


def test_actual_case_groups_explicitly_linked_ors_and_labels_zinc(monkeypatch):
    original = _result(monkeypatch, {"absent_or_weak_pulse": True})
    updated = apply_answers(original, CARDS, {"patient_can_drink_safely": True}, questions(original, CARDS))
    matches = [r for e in updated["source_review_options"] for r in e["supported_rules"] if r.get("source_instruction")]
    before = copy.deepcopy(matches)
    added = guidance_changes(original, updated)["added"]
    groups = instruction_groups(matches, CARDS, added)
    ors = next(g for g in groups if g["label"] == "Oral rehydration solution (ORS)")
    assert {r["card_id"] for r in ors["rules"]} == {"CHO-004", "CHO-005"}
    assert [l["text"] for l in ors["lines"]] == ["Give ORS for ongoing losses", "Give as much as wanted."]
    assert ors["timing"] == "After each loose stool"
    assert ors["updated"]
    assert any("10 years" in (l["age_label"] or "") for l in ors["lines"])
    zinc = next(g for g in groups if g["label"] == "Zinc")
    assert zinc["lines"][0]["text"] == "20 mg once per day for 10 days."
    fluids = next(g for g in groups if g["label"].startswith("Ringer"))
    assert any("30 mL/kg over 30 minutes, then 70 mL/kg over 2.5 hours" in l["text"] for l in fluids["lines"])
    assert matches == before
    assert sum(len(g["rules"]) for g in groups) == len(matches)


def test_multi_drug_card_does_not_get_a_card_wide_medicine_or_dose_label():
    from management_evidence import rule_options
    card = CARDS["CHI-014"]
    options = [r for r in rule_options(card, {"patient_has_fever": True, "patient_has_itching": True,
        "preexisting_liver_disease": False, "preexisting_kidney_disease": False}) if r["status"] == "supported"]
    groups = instruction_groups(options, CARDS)
    assert groups
    assert all(g["label"] == card.card_title for g in groups)
    assert all(g["timing"] is None for g in groups)


@pytest.mark.parametrize("text,status,expected", [
    ("for ages 10 years and older, give as much as wanted.", "matched", "Give as much as wanted."),
    ("For ages 2–9 years give 100–200 mL after each loose stool.", "matched", "Give 100–200 mL after each loose stool."),
    ("for ages 10 years and older, give as much as wanted.", "age_unknown", "For ages 10 years and older, give as much as wanted."),
    ("For ages 10 years and older if eligible give this dose.", "matched", "For ages 10 years and older if eligible give this dose."),
    ("Do not give 10 mg before assessment.", "matched", "Do not give 10 mg before assessment."),
])
def test_age_prefix_moves_only_when_matched_without_removing_other_conditions(text, status, expected):
    rule = {"source_instruction": text, "age_clause_selection": {"status": status}}
    before = copy.deepcopy(rule)
    line = readable_instruction(rule, None)
    assert line["text"] == expected
    if text != expected and status == "matched" and "if eligible" not in text:
        assert line["age_label"]
    assert rule == before


def test_source_links_not_disease_names_control_merging(monkeypatch):
    from schema import Card
    original = _result(monkeypatch, {"absent_or_weak_pulse": True})
    updated = apply_answers(original, CARDS, {"patient_can_drink_safely": True}, questions(original, CARDS))
    matches = [r for e in updated["source_review_options"] for r in e["supported_rules"]
               if r["card_id"] in {"CHO-004", "CHO-005"}]
    remap = {"CHO-004": "SOURCE-X", "CHO-005": "SOURCE-Y"}
    cards = {}
    for old_id, new_id in remap.items():
        raw = CARDS[old_id].model_dump()
        raw["card_id"] = new_id
        raw["disease_protocol"] = "Other protocol"
        raw["card_title"] = "Renamed source"
        for branch in raw["if_else_conditions_and_corresponding_actions"] or []:
            for cid, replacement in remap.items():
                branch["then"]["next_step"] = branch["then"].get("next_step", "").replace(cid, replacement)
        cards[new_id] = Card.model_validate(raw)
    renamed = copy.deepcopy(matches)
    for r in renamed:
        r["card_id"] = remap[r["card_id"]]
    groups = instruction_groups(renamed, cards)
    ors = next(g for g in groups if g["label"] == "Oral rehydration solution (ORS)")
    assert {r["card_id"] for r in ors["rules"]} == {"SOURCE-X", "SOURCE-Y"}
    # The same medicine label without a source link is insufficient to merge.
    raw = cards["SOURCE-X"].model_dump()
    raw["if_else_conditions_and_corresponding_actions"] = []
    cards["SOURCE-X"] = Card.model_validate(raw)
    groups = instruction_groups(renamed, cards)
    assert all(len({r["card_id"] for r in g["rules"]}) == 1 for g in groups)


def test_changed_badge_does_not_color_unchanged_fluid_instructions(monkeypatch):
    result = _result(monkeypatch, {"absent_or_weak_pulse": True})
    matches = [r for e in result["source_review_options"] for r in e["supported_rules"] if r.get("source_instruction")]
    assert all(not g["updated"] and not any(l["updated"] for l in g["lines"])
               for g in instruction_groups(matches, CARDS))


def test_conditional_or_restricted_rules_cannot_be_rendered_as_matches():
    rule = {"card_id": "RENAMED", "rule_id": "R", "source_instruction": "Unassessed dose"}
    for status in ("conditional", "restricted", "not_applicable"):
        assert instruction_groups([{**rule, "status": status}], {}) == []


def test_full_reference_is_loaded_only_after_explicit_toggle(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    original = _result(monkeypatch, {"absent_or_weak_pulse": True, "patient_can_drink_safely": True})
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"))
    for key, value in {"case_patient": {"age": "37 years"}, "case_mode": "improved", "case_diseases": ["Cholera"],
        "working_disease": "Cholera", "case_management_requested": True,
        "case_results": {"Diagnosis": {"probable_diagnosis": []}, "Management": original}}.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception
    assert not any("2–9 years" in item.value for item in at.markdown)
    toggle = next(t for t in at.toggle if "CHO-005" in t.key)
    toggle.set_value(True)
    at.run()
    assert not at.exception
    assert any("2–9 years" in item.value for item in at.markdown)
    assert any("not a patient-specific plan" in e.label for e in at.expander)
    assert at.session_state["case_results"]["Management"] == original


def test_selected_plan_has_explicit_reference_box_without_changing_results(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    result = _result(monkeypatch, {"absent_or_weak_pulse": True})
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"))
    for key, value in {"case_patient": {"age": "37 years"}, "case_mode": "improved", "case_diseases": ["Cholera"],
        "working_disease": "Cholera", "case_management_requested": True,
        "case_results": {"Diagnosis": {"probable_diagnosis": []}, "Management": result}}.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception
    assert any(e.label == "Plan C — view source details" for e in at.expander)
    assert not any(e.label == "Complete card data — original conditions and tables" for e in at.expander)
    next(t for t in at.toggle if t.label == "Show selected plan source cards").set_value(True)
    at.run()
    assert not at.exception
    assert any(e.label == "Complete card data — original conditions and tables" for e in at.expander)
    assert at.session_state["case_results"]["Management"] == result
