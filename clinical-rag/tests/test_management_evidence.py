"""Instruction-level outcomes, delivery budgets and unchanged frozen paths."""
import json
from pathlib import Path

import retrieve
from build_index import load_cards
from management_evidence import rule_options, prerequisite_state, source_context, condition_text
from derived_fields import derive_classifications
from recommend import recommend_management

CARDS = {c.card_id: c for c in load_cards(Path("data/cards.jsonl"))}


def test_partial_card_keeps_matching_instruction_and_unknown_alternative():
    options = rule_options(CARDS["CHI-015"], {"joint_pain_present": True})
    assert next(o for o in options if o["rule_id"] == "CHI-ADV-001")["status"] == "supported"
    other = next(o for o in options if o["rule_id"] == "CHI-ADV-003")
    assert other["status"] == "conditional"
    assert other["missing_inputs"] == ["ophthalmic_complications_present"]


def test_and_does_not_turn_partial_evidence_into_permission_and_or_can_match():
    base = CARDS["CHI-015"]
    for operation, expected in (("all", "conditional"), ("any", "supported")):
        card = base.model_copy(update={
            "conditions_decision_logic": [{"rule_id": "SYN-RULE", "condition": {
                operation: [{"field": "marker_one", "operator": "=", "value": True},
                            {"field": "marker_two", "operator": "=", "value": True}]},
                "result": "Source option."}],
            "if_else_conditions_and_corresponding_actions": [],
        })
        option = rule_options(card, {"marker_one": True})[0]
        assert option["status"] == expected
        assert option["missing_inputs"] == ["marker_two"]


def test_known_contraindication_only_restricts_affected_medication():
    options = rule_options(CARDS["CHI-014"], {
        "patient_has_fever": True, "preexisting_liver_disease": True})
    by_id = {o["rule_id"]: o for o in options}
    assert by_id["CHI-MED-001"]["status"] == "restricted"
    assert by_id["CHI-MED-002"]["status"] == "conditional"
    assert "preexisting_liver_disease" not in by_id["CHI-MED-002"]["missing_inputs"]


def test_unreadable_table_is_explicit_and_never_supported():
    options = rule_options(CARDS["CHO-018"], {})
    assert options[0]["unsupported_rule"]
    assert options[0]["status"] == "conditional"
    assert options[0]["source_instruction"] is None


def test_enum_prerequisite_comes_from_source_and_blocks_wrong_path():
    facts = {"absent_or_weak_pulse": True}
    derived, trace = derive_classifications(CARDS.values(), "Cholera", facts)
    context, choices, resolved = source_context(CARDS.values(), "Cholera", {**facts, **derived}, trace)
    assert resolved["treatment_plan"] == "Plan C"
    assert prerequisite_state(CARDS["CHO-008"], context, choices)[0]["status"] == "not_met"
    assert prerequisite_state(CARDS["CHO-005"], context, choices)[0]["status"] == "met"


def test_prerequisite_is_generic_and_does_not_match_title_alone():
    card = CARDS["CHO-008"].model_copy(update={
        "card_title": "Pathway Cedar", "relevant_symptoms_or_prerequisites": "Receiving pathway Birch",
        "clinical_classification_presentation": "Follow-up"})
    values = {"source_pathway": {"pathway Birch": [], "pathway Cedar": []}}
    gate = prerequisite_state(card, {"source_pathway": "pathway Cedar"}, values)
    assert gate == [{"field": "source_pathway", "required_value": "pathway Birch", "status": "not_met"}]
    card = card.model_copy(update={"relevant_symptoms_or_prerequisites": "Follow-up care"})
    assert prerequisite_state(card, {}, values) == []
    card = card.model_copy(update={"relevant_symptoms_or_prerequisites": "Not on pathway Birch"})
    assert prerequisite_state(card, {}, values) == []


def test_readable_conditions_preserve_and_or_and_source_thresholds():
    text = condition_text({"all": [
        {"field": "marker_present", "operator": "=", "value": True},
        {"any": [{"field": "measured_value", "operator": ">=", "value": 23},
                 {"field": "alternate_marker", "operator": "=", "value": False}]}]})
    assert " AND " in text and " OR " in text
    assert "measured value >= 23" in text and "absent/false" in text


def test_real_large_cards_deliver_all_categories_and_primary_medication(monkeypatch):
    pool = {"Investigation": ["CHI-013", "CHI-023", "CHI-022"],
            "Medication": ["CHI-018", "CHI-014", "CHI-017"],
            "Advice": ["CHI-015", "CHI-016", "CHI-024"]}
    monkeypatch.setattr(retrieve, "retrieve_category", lambda p, c, **kw:
                        [{"card_id": cid} for cid in pool[c]])
    prompts = []
    def model(prompt):
        if prompt.startswith("Extract patient facts"):
            return json.dumps({"facts": {}})
        prompts.append(prompt)
        return json.dumps({"missing_info": ["clinical review"]})
    result = recommend_management({"symptoms": "High fever, severe joint pain", "age": "32 years"},
                                  cards=CARDS, diseases=["Chikungunya"], llm_generate=model)
    d = result["management_diagnostics"]
    assert d["prompt_categories"] == ["Advice", "Investigation", "Medication"]
    assert {"CHI-014", "CHI-015", "CHI-023"} <= set(d["prompt_card_ids"])
    assert len(prompts) == 1 and len(prompts[0].encode()) <= 19000
    assert d["prompt_bytes"] == len(prompts[0].encode())
    assert result["source_review_options"]  # not lost when the model returns no items


def test_fact_collection_connects_narrative_to_indication_and_advice(monkeypatch):
    monkeypatch.setattr(retrieve, "retrieve_category", lambda p, c, **kw:
                        [{"card_id": cid} for cid in {
                            "Investigation": ["CHI-023"], "Medication": ["CHI-014"],
                            "Advice": ["CHI-015"]}[c]])
    def model(prompt):
        if prompt.startswith("Extract patient facts"):
            return json.dumps({"facts": {
                "patient_has_fever": {"value": True, "quote": "High fever"},
                "joint_pain_present": {"value": True, "quote": "severe joint pain"}}})
        if "semantic fact verification" in prompt:
            return json.dumps({"decisions": {"patient_has_fever": "confirmed", "joint_pain_present": "confirmed"}})
        return json.dumps({"missing_info": ["safety assessment"]})
    result = recommend_management({"symptoms": "High fever, severe joint pain"}, cards=CARDS,
                                  diseases=["Chikungunya"], llm_generate=model)
    assert result["management_diagnostics"]["management_fact_extraction"]["accepted"] == [
        "joint_pain_present", "patient_has_fever"]
    reviews = {o["card_id"]: o for o in result["source_review_options"]}
    assert reviews["CHI-015"]["supported_rules"]
    assert reviews["CHI-014"]["conditional_rules"]  # fever is not clearance of liver/kidney restrictions
    assert "preexisting_liver_disease" in reviews["CHI-014"]["missing_inputs"]


def test_weak_pulse_selects_matching_source_and_excludes_other_maintenance(monkeypatch):
    monkeypatch.setattr(retrieve, "retrieve_category", lambda p, c, **kw:
                        [{"card_id": cid} for cid in {
                            "Investigation": [], "Medication": ["CHO-008", "CHO-018", "CHO-020", "CHO-021"],
                            "Advice": ["CHO-012"]}[c]])
    def model(prompt):
        if prompt.startswith("Extract patient facts"):
            return json.dumps({"facts": {"absent_or_weak_pulse": {"value": True, "quote": "weak rapid pulse"}}})
        if "semantic fact verification" in prompt:
            return json.dumps({"decisions": {"absent_or_weak_pulse": "confirmed"}})
        return json.dumps({"missing_info": ["weight and other clinical checks"]})
    result = recommend_management({"clinical_findings": "weak rapid pulse, sunken eyes"},
                                  cards=CARDS, diseases=["Cholera"], llm_generate=model)
    d = result["management_diagnostics"]
    assert "CHO-004" in d["prompt_card_ids"]
    assert "CHO-008" not in d["prompt_card_ids"]
    assert "CHO-008" not in {r["card_id"] for r in result["source_review_options"]}
    assert any(r["card_id"] == "CHO-004" and r["supported_rules"] for r in result["source_review_options"])


def test_llm_failure_still_preserves_supported_and_conditional_source_options(monkeypatch):
    from llm import LLMError
    monkeypatch.setattr(retrieve, "retrieve_category", lambda p, c, **kw:
                        [{"card_id": "CHI-015"}] if c == "Advice" else [])
    def unavailable(prompt):
        raise LLMError("429")
    result = recommend_management({"_structured_facts": {"joint_pain_present": True}},
                                  cards=CARDS, diseases=["Chikungunya"], llm_generate=unavailable)
    assert result["source_review_options"][0]["supported_rules"]
    assert result["source_review_options"][0]["conditional_rules"]
    assert result["management_diagnostics"]["error_type"] == "LLMError"
