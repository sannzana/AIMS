"""Regressions for complete prerequisites, source branches and current care."""
import json
from pathlib import Path

import pytest
import retrieve
from build_index import load_cards
from management_conditions import (boolean_source_fields, age_specific_instruction,
                                   review_requirements)
from management_evidence import prerequisite_state, rule_options
from management_facts import extract_management_facts
from recommend import recommend_management

CARDS = {c.card_id: c for c in load_cards(Path("data/cards.jsonl"))}


@pytest.mark.parametrize("can_drink,status", [(None, "conditional"), (True, "supported"), (False, "not_applicable")])
def test_source_conjunction_applies_to_age_rule_too(can_drink, status):
    context = {"age_years": 38, "age_months": 456, "treatment_plan": "Plan C"}
    if can_drink is not None:
        context["patient_can_drink_safely"] = can_drink
    gates = prerequisite_state(CARDS["CHO-005"], context,
                               {"treatment_plan": {"Plan C": []}}, boolean_source_fields(CARDS.values()))
    options = rule_options(CARDS["CHO-005"], context, gates)
    older = next(o for o in options if o["rule_id"] == "CHO-C-ORS-002")
    assert older["status"] == status
    assert "2–9" not in older["source_instruction"]
    assert "10 years and older" in older["source_instruction"]
    if can_drink is None:
        assert "patient_can_drink_safely" in older["missing_inputs"]


def test_prerequisite_matching_is_not_specific_to_a_disease_or_field():
    card = CARDS["CHO-005"].model_copy(update={
        "disease_protocol": "Synthetic", "card_id": "SYN-1",
        "relevant_symptoms_or_prerequisites": "Receiving pathway Birch and can observe the marker safely.",
        "clinical_classification_presentation": "Follow-up"})
    gates = prerequisite_state(card, {"pathway": "pathway Birch"},
                               {"pathway": {"pathway Birch": []}}, {"patient_can_observe_safely"})
    assert any(g["field"] == "patient_can_observe_safely" and g["status"] == "unknown" for g in gates)


def test_age_thresholds_come_from_arbitrary_source_text_not_fixed_values():
    text = "For ages 3–7 months give source option X; for ages 8 months and older give source option Y."
    selected, unknown, fields = age_specific_instruction(text, {"age_months": 11})
    assert selected == "for ages 8 months and older give source option Y."
    assert not unknown and fields == []
    selected, unknown, fields = age_specific_instruction(text, {})
    assert selected is None and unknown and "age_months" in fields


def test_ongoing_monitoring_does_not_require_future_transition_facts():
    gates = [{"field": "treatment_plan", "required_value": "Plan C", "status": "met"}]
    options = rule_options(CARDS["CHO-010"], {"treatment_plan": "Plan C"}, gates)
    monitoring = next(o for o in options if o["rule_id"] == "source-monitoring")
    assert monitoring["status"] == "supported"
    assert "15–30 minutes" in monitoring["source_instruction"]
    assert all(o["status"] == "conditional" for o in options if o is not monitoring)


def test_monitoring_is_source_driven_and_unknown_path_cannot_authorise_it():
    card = CARDS["CHO-010"].model_copy(update={
        "disease_protocol": "Synthetic", "card_id": "SYN-2",
        "recommended_action": "During pathway Birch, monitor the stated marker every 17 minutes. If recovery is verified, change pathway.",
    })
    gate = {"field": "chosen_path", "required_value": "pathway Birch", "status": "met"}
    result = rule_options(card, {"chosen_path": "pathway Birch"}, [gate])
    assert any(o["rule_id"] == "source-monitoring" and "17 minutes" in o["source_instruction"] for o in result)
    gate["status"] = "unknown"
    assert not any(o["rule_id"] == "source-monitoring" for o in rule_options(card, {}, [gate]))


def test_inferred_sign_count_and_severity_do_not_enter_accepted_facts():
    def model(prompt):
        if prompt.startswith("Extract patient facts"):
            return json.dumps({"facts": {
                "some_dehydration_sign_count": {"value": 38, "quote": "38 years"},
                "severe_dehydration": {"value": True, "quote": "weak rapid pulse"}}})
        # Even an overconfident verifier cannot override the source assertion guard.
        return json.dumps({"decisions": {"some_dehydration_sign_count": "confirmed", "severe_dehydration": "confirmed"}})
    facts, info = extract_management_facts({"age": "38 years", "clinical_findings": "weak rapid pulse"},
                                            CARDS.values(), "Cholera", {}, model)
    assert facts == {}
    assert "some_dehydration_sign_count" in info["rejected"]
    assert "severe_dehydration" in info["rejected"]


def test_explicitly_reported_count_can_still_be_verified():
    def model(prompt):
        if prompt.startswith("Extract patient facts"):
            return json.dumps({"facts": {"some_dehydration_sign_count": {
                "value": 2, "quote": "2 signs of some dehydration"}}})
        return json.dumps({"decisions": {"some_dehydration_sign_count": "confirmed"}})
    facts, _ = extract_management_facts({"clinical_findings": "Clinician reports 2 signs of some dehydration"},
                                         CARDS.values(), "Cholera", {}, model)
    assert facts["some_dehydration_sign_count"] == 2


def test_conditional_boolean_branches_ask_one_question():
    options = rule_options(CARDS["CHO-010"], {})
    leaf = {**options[0], "condition": {"field": "some_known_predicate", "operator": "=", "value": True}}
    inverse = {**leaf, "condition": {"field": "some_known_predicate", "operator": "=", "value": False}}
    lines = review_requirements({"conditional_rules": [leaf, inverse]})
    assert lines == ["Confirm whether some known predicate."]


def test_supported_instruction_survives_other_false_rule_in_same_card(monkeypatch):
    card = CARDS["CHI-015"].model_copy(update={
        "conditions_decision_logic": [
            {"rule_id": "SYN-YES", "condition": {"field": "marker", "operator": "=", "value": True}, "result": "Source instruction for the present marker.", "source_page": 19},
            {"rule_id": "SYN-NO", "condition": {"field": "other_marker", "operator": "=", "value": True}, "result": "Unrelated instruction.", "source_page": 19}],
        "if_else_conditions_and_corresponding_actions": [],
    })
    monkeypatch.setattr(retrieve, "retrieve_category", lambda p, c, **kw: [{"card_id": card.card_id}] if c == "Advice" else [])
    def model(prompt):
        if prompt.startswith("Extract patient facts"):
            return json.dumps({"facts": {}})
        return json.dumps({"advice": [{"text": "A concise model selection.", "reason": "Matches.",
            "evidence": [{"card_id": card.card_id, "pdf_page": 19}],
            "source_rules": [{"card_id": card.card_id, "rule_id": "SYN-YES"}]}], "missing_info": []})
    result = recommend_management({"_structured_facts": {"marker": True, "other_marker": False}},
                                  cards={card.card_id: card}, diseases=[card.disease_protocol], llm_generate=model)
    item = result["advice"][0]
    assert item["review_required"] is False
    assert item["text"] == "Source instruction for the present marker."
    assert "Unrelated" not in item["text"]


def test_wrong_instruction_link_does_not_authorise_unknown_branch(monkeypatch):
    monkeypatch.setattr(retrieve, "retrieve_category", lambda p, c, **kw: [{"card_id": "CHI-015"}] if c == "Advice" else [])
    def model(prompt):
        if prompt.startswith("Extract patient facts"):
            return json.dumps({"facts": {}})
        return json.dumps({"advice": [{"text": "Specialist action now.", "reason": "Claimed.",
            "evidence": [{"card_id": "CHI-015", "pdf_page": 20}],
            "source_rules": [{"card_id": "CHI-015", "rule_id": "CHI-ADV-003"}]}], "missing_info": []})
    result = recommend_management({"_structured_facts": {"joint_pain_present": True}},
                                  cards=CARDS, diseases=["Chikungunya"], llm_generate=model)
    assert result["advice"][0]["review_required"] is True
    assert result["management_diagnostics"]["instruction_link_rejections"]


@pytest.mark.parametrize("bad_ref", [{"card_id": [], "rule_id": {}}, "invalid", {"card_id": "CHI-015", "rule_id": []}])
def test_malformed_instruction_links_fail_without_crashing(bad_ref):
    from management_links import instruction_links
    from recommend import _parse_response_partial
    raw = json.dumps({"advice": [{"text": "Example selection.", "reason": "Example reason.",
        "evidence": [{"card_id": "CHI-015", "pdf_page": 19}], "source_rules": [bad_ref]}]})
    response, _ = _parse_response_partial(raw)
    options = rule_options(CARDS["CHI-015"], {"joint_pain_present": True})
    linked, errors = instruction_links(raw, response, options, CARDS)
    assert linked == {} and errors


def test_user_case_current_options_exclude_hypothetical_complications_and_discharge(monkeypatch):
    pool = {"Investigation": [], "Medication": ["CHO-004", "CHO-005", "CHO-020", "CHO-021", "CHO-018"],
            "Advice": ["CHO-012", "CHO-022", "CHO-010"]}
    monkeypatch.setattr(retrieve, "retrieve_category", lambda p, c, **kw: [{"card_id": cid} for cid in pool[c]])
    def model(prompt):
        return json.dumps({"facts": {}}) if prompt.startswith("Extract patient facts") else json.dumps({"missing_info": ["review"]})
    result = recommend_management({"age": "38 years", "_structured_facts": {"absent_or_weak_pulse": True}},
                                  cards=CARDS, diseases=["Cholera"], llm_generate=model)
    reviews = {entry["card_id"]: entry for entry in result["source_review_options"]}
    assert reviews["CHO-004"]["current"] and reviews["CHO-005"]["current"]
    assert reviews["CHO-010"]["current"] and reviews["CHO-010"]["supported_rules"]
    assert not reviews["CHO-020"]["current"] and not reviews["CHO-021"]["current"]
    assert not reviews["CHO-012"]["current"] and not reviews["CHO-022"]["current"]
    assert not reviews["CHO-018"]["current"]
    assert "patient_can_drink_safely" in result["current_missing_inputs"]
    assert "blood_glucose_mmol_l" not in result["current_missing_inputs"]
    assert not {"CHO-020", "CHO-021", "CHO-012", "CHO-022"} & set(result["management_diagnostics"]["prompt_card_ids"])
    assert {"CHO-004", "CHO-005", "CHO-010"} <= set(result["management_diagnostics"]["prompt_card_ids"])


def test_current_monitoring_link_survives_unknown_transition_branches(monkeypatch):
    monkeypatch.setattr(retrieve, "retrieve_category", lambda p, c, **kw: [{"card_id": "CHO-010"}] if c == "Advice" else [])
    def model(prompt):
        if prompt.startswith("Extract patient facts"):
            return json.dumps({"facts": {}})
        return json.dumps({"advice": [{"text": "Monitor according to the selected source.", "reason": "Pathway selected.",
            "evidence": [{"card_id": "CHO-010", "pdf_page": 5}],
            "source_rules": [{"card_id": "CHO-010", "rule_id": "source-monitoring"}]}], "missing_info": []})
    result = recommend_management({"age": "38 years", "_structured_facts": {"absent_or_weak_pulse": True}},
                                  cards=CARDS, diseases=["Cholera"], llm_generate=model)
    item = result["advice"][0]
    assert item["review_required"] is False
    assert "15–30 minutes" in item["text"]


def test_final_ui_preserves_current_care_without_duplicate_or_wrong_age_text(monkeypatch):
    from streamlit.testing.v1 import AppTest
    pool = {"Investigation": [], "Medication": ["CHO-004", "CHO-005", "CHO-020", "CHO-021", "CHO-018"],
            "Advice": ["CHO-012", "CHO-022", "CHO-010"]}
    monkeypatch.setattr(retrieve, "retrieve_category", lambda p, c, **kw: [{"card_id": cid} for cid in pool[c]])
    def model(prompt):
        if prompt.startswith("Extract patient facts"):
            return json.dumps({"facts": {}})
        return json.dumps({"advice": [{"text": "Monitor using the selected source.", "reason": "Source pathway applies.",
            "evidence": [{"card_id": "CHO-010", "pdf_page": 5}],
            "source_rules": [{"card_id": "CHO-010", "rule_id": "source-monitoring"}]}], "missing_info": ["review"]})
    result = recommend_management({"age": "38 years", "_structured_facts": {"absent_or_weak_pulse": True}},
                                  cards=CARDS, diseases=["Cholera"], llm_generate=model)
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"))
    state = {"case_patient": {"age": "38 years"}, "case_mode": "improved", "case_diseases": ["Cholera"],
             "working_disease": "Cholera", "case_management_requested": True,
             "case_results": {"Diagnosis": {"probable_diagnosis": []}, "Management": result}}
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception
    text = "\n".join(item.value for item in at.markdown)
    assert "15–30 minutes" in text
    assert text.count("15–30 minutes") == 1
    assert "Start intravenous fluids immediately" in text
    assert "For ages 2–9" not in text and "for ages 10 years" not in text  # ORS still conditional
    assert text.count("Confirm whether patient can drink safely.") == 1
    assert any("Other source material" in item.label for item in at.expander)
