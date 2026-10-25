"""Scoped restrictions are not blanket card suppression or blanket permission."""
import copy
import json
from pathlib import Path

import pytest
import retrieve
from applicability import card_rule_applicability
from build_index import load_cards
from management_evidence import rule_options
from management_safety import (scope_records, policies, source_digest, instruction_safety,
                               safety_notices, validation_views)
from recommend import recommend_management

CARDS = {c.card_id: c for c in load_cards(Path("data/cards.jsonl"))}


def options(card_id, facts):
    return {o["rule_id"]: o for o in rule_options(CARDS[card_id], facts)}


def test_all_source_mappings_are_current_and_ambiguous_population_scopes_pending():
    assert len(scope_records()) == 14
    for record in scope_records():
        policy = policies(CARDS[record["card_id"]])[record["exception_index"]]
        assert policy["kind"] == record["kind"]
    assert all(policy["kind"] == "scope_pending" for policy in policies(CARDS["HTN-021"]))


@pytest.mark.parametrize("liver,kidney,expected", [(True, False, "restricted"), (False, True, "restricted"),
                                               (False, False, "supported"), (None, False, "conditional")])
def test_restriction_applies_to_fever_drug_not_itching(liver, kidney, expected):
    facts = {"patient_has_fever": True, "patient_has_itching": True,
             "preexisting_liver_disease": liver, "preexisting_kidney_disease": kidney}
    result = options("CHI-014", facts)
    assert result["CHI-MED-001"]["status"] == expected
    assert result["CHI-MED-002"]["status"] == "supported"
    assert result["CHI-MED-002"]["restrictions"] == []


@pytest.mark.parametrize("days,excluded,nsaid_status", [(3, False, "not_applicable"),
    (11, True, "supported"), (11, None, "conditional"), (None, None, "conditional")])
def test_timing_restriction_does_not_remove_fluids_or_infection_instruction(days, excluded, nsaid_status):
    result = options("CHI-017", {"days_since_symptom_onset": days, "dengue_adequately_excluded": excluded,
        "dehydration_present": True, "patient_age_group": "adult", "patient_is_child": False,
        "secondary_bacterial_infection_present": True})
    assert result["CHI-HOSP-MED-002"]["status"] == "supported"
    assert result["CHI-HOSP-MED-003"]["status"] == "supported"
    assert result["CHI-HOSP-MED-004"]["status"] == nsaid_status


def test_interpretation_caution_does_not_require_a_test_result_before_ordering():
    result = options("DEN-INV-001", {"suspected_dengue": True, "illness_day": 3})
    assert result["TIME-001"]["status"] == "supported"
    assert "ns1_result" not in result["TIME-001"]["missing_inputs"]
    assert result["TIME-001"]["safety_notes"]
    result = options("DEN-INV-008", {"suspected_dengue": True})
    assert result["INV-BIO-001"]["status"] == "supported"
    assert "transaminase_result_available" not in result["INV-BIO-001"]["missing_inputs"]


def test_positive_test_interpretation_caution_is_not_a_contraindication():
    result = options("DEN-INV-001", {"suspected_dengue": True, "illness_day": 3, "ns1_result": "negative"})
    assert result["TIME-001"]["status"] == "supported"


def test_warning_instruction_does_not_block_paracetamol_rule():
    result = options("DEN-MED-001", {"home_managed_dengue_with_fever": True})
    assert result["MED-001"]["status"] == "supported"
    assert result["MED-001"]["restrictions"] == []


def test_no_source_mapping_is_not_permission():
    card = CARDS["CHI-014"].model_copy(update={"card_id": "UNMAPPED", "disease_protocol": "Synthetic"})
    result = rule_options(card, {"patient_has_fever": True, "patient_has_itching": True})
    assert not any(o["status"] == "supported" for o in result)
    assert all("source_restriction_scope_review" in o["missing_inputs"] for o in result)


def test_stale_mapping_requires_review_instead_of_becoming_warning_only():
    card = CARDS["CHI-014"].model_copy(update={"relevant_symptoms_or_prerequisites": "Changed source"})
    assert policies(card)[0]["kind"] == "scope_pending"
    assert instruction_safety(card, "CHI-MED-001", {})["pending"]


def test_renamed_source_and_mapping_follow_metadata_not_python_ids(monkeypatch):
    import management_safety
    card = CARDS["CHI-014"]
    raw = json.loads(json.dumps(card.model_dump()).replace("CHI-MED-001", "RULE-ALPHA").replace("CHI-MED-002", "RULE-BETA"))
    from schema import Card
    renamed = Card.model_validate({**raw, "card_id": "SYNTHETIC-42", "disease_protocol": "Synthetic"})
    record = copy.deepcopy(next(r for r in scope_records() if r["card_id"] == card.card_id))
    record.update(card_id=renamed.card_id, scope_rule_ids=["RULE-ALPHA"], source_digest=source_digest(renamed))
    monkeypatch.setattr(management_safety, "scope_records", lambda: (record,))
    result = {o["rule_id"]: o for o in rule_options(renamed, {"patient_has_fever": True,
        "patient_has_itching": True, "preexisting_liver_disease": True})}
    assert result["RULE-ALPHA"]["status"] == "restricted"
    assert result["RULE-BETA"]["status"] == "supported"


@pytest.mark.parametrize("medicines", ["NSAID", "NSAIDs", "aspirin", "No NSAIDs"])
def test_source_medication_mention_is_flagged_without_claiming_current_use(medicines):
    notes = safety_notices(CARDS.values(), "Chikungunya", {"days_since_symptom_onset": 3}, medicines, set())
    matching = [n for n in notes if n["card_id"] == "CHI-017"]
    assert matching and matching[0]["medicine_mentioned"]
    assert matching[0]["status"] == "restriction_applies"


def test_unknown_timing_is_not_permission_and_unlisted_drug_alias_is_not_invented():
    notes = safety_notices(CARDS.values(), "Chikungunya", {}, "NSAID", set())
    assert next(n for n in notes if n["card_id"] == "CHI-017")["status"] == "conditional"
    assert safety_notices(CARDS.values(), "Chikungunya", {}, "an-unlisted-brand", set()) == []


def test_legacy_validation_keeps_mixed_card_review_only_instead_of_suppressing_everything():
    card = CARDS["CHI-017"]
    facts = {"days_since_symptom_onset": 3, "dehydration_present": True, "patient_age_group": "adult"}
    profile = rule_options(card, facts)
    assert any(o["status"] == "supported" for o in profile)
    views, context = validation_views({card.card_id: card}, profile, facts)
    assert card_rule_applicability(card, facts)["status"] == "not_applicable"
    assert card_rule_applicability(views[card.card_id], context)["status"] == "undetermined"
    assert card.model_dump() == CARDS[card.card_id].model_dump()


def test_management_keeps_matching_source_instructions_and_exposes_safety_notice(monkeypatch):
    monkeypatch.setattr(retrieve, "retrieve_category", lambda p, c, **kw: [{"card_id": "CHI-017"}] if c == "Medication" else [])
    def model(prompt):
        if prompt.startswith("Extract patient facts"):
            return json.dumps({"facts": {}})
        return json.dumps({"medication": [{"text": "Source-supported fluid option.", "reason": "Patient findings.",
            "evidence": [{"card_id": "CHI-017", "pdf_page": CARDS["CHI-017"].pdf_page_numbers[0]}]}], "missing_info": []})
    result = recommend_management({"medicines": "NSAID", "_structured_facts": {
        "days_since_symptom_onset": 3, "dehydration_present": True, "patient_age_group": "adult"}},
        cards=CARDS, diseases=["Chikungunya"], llm_generate=model)
    entry = next(e for e in result["source_review_options"] if e["card_id"] == "CHI-017")
    assert any(o["rule_id"] == "CHI-HOSP-MED-002" for o in entry["supported_rules"])
    assert any(n["status"] == "restriction_applies" for n in result["safety_notices"])
    assert result["medication"]  # retained as review-only; applicable source clause remains visible
    assert result["medication"][0]["review_required"]


@pytest.mark.parametrize("drink,expected", [(None, "conditional"), (False, "restricted"), (True, "supported")])
def test_age_match_does_not_override_explicit_drinking_prerequisite(drink, expected):
    result = options("CHO-005", {"age_years": 37, "age_months": 444, "patient_can_drink_safely": drink})
    assert result["CHO-C-ORS-002"]["status"] == expected
    assert "2–9" not in result["CHO-C-ORS-002"]["source_instruction"]
    assert result["CHO-C-ORS-001"]["status"] == "not_applicable"


def test_changed_parent_prerequisite_is_not_silently_trusted():
    card = CARDS["CHO-005"].model_copy(update={"relevant_symptoms_or_prerequisites": "Changed prerequisite"})
    profile = rule_options(card, {"age_years": 37, "age_months": 444, "patient_can_drink_safely": True})
    older = next(o for o in profile if o["rule_id"] == "CHO-C-ORS-002")
    assert older["status"] == "conditional"
    assert "source_restriction_scope_review" in older["missing_inputs"]


def test_malformed_scope_record_is_pending_not_permission():
    record = copy.deepcopy(next(r for r in scope_records() if r["card_id"] == "CHI-014"))
    record["scope_rule_ids"] = [{"invalid": "rule"}]
    assert policies(CARDS["CHI-014"], [record])[0]["kind"] == "scope_pending"


def test_adult_ui_does_not_show_child_age_clause(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(retrieve, "retrieve_category", lambda p, c, **kw: [{"card_id": "CHO-005"}] if c == "Medication" else [])
    result = recommend_management({"age": "37 years", "_structured_facts": {
        "dehydration_classification": "severe_dehydration_or_shock", "patient_can_drink_safely": True}},
        cards=CARDS, diseases=["Cholera"], llm_generate=lambda p: json.dumps({"facts": {}, "missing_info": ["review"]}))
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"))
    for key, value in {"case_patient": {"age": "37 years"}, "case_mode": "improved",
        "case_diseases": ["Cholera"], "working_disease": "Cholera", "case_management_requested": True,
        "case_results": {"Diagnosis": {"probable_diagnosis": []}, "Management": result}}.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception
    text = "\n".join(item.value for item in at.markdown)
    assert "2–9 years" not in text
    assert "10 years and older" in text
    assert any("Age-matched source clause" in caption.value for caption in at.caption)


def test_conditional_age_matched_ors_clause_is_visible_with_missing_prerequisite(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(retrieve, "retrieve_category", lambda p, c, **kw: [{"card_id": "CHO-005"}] if c == "Medication" else [])
    result = recommend_management({"age": "37 years", "_structured_facts": {
        "dehydration_classification": "severe_dehydration_or_shock"}}, cards=CARDS, diseases=["Cholera"],
        llm_generate=lambda p: json.dumps({"facts": {}, "missing_info": ["review"]}))
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"))
    for key, value in {"case_patient": {"age": "37 years"}, "case_mode": "improved",
        "case_diseases": ["Cholera"], "working_disease": "Cholera", "case_management_requested": True,
        "case_results": {"Diagnosis": {"probable_diagnosis": []}, "Management": result}}.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception
    relevant = [info.value for info in at.info if "Conditional age-matched source option" in info.value]
    assert len(relevant) == 1
    assert "10 years and older" in relevant[0] and "2–9" not in relevant[0]
    assert "ORS" in relevant[0] and "patient can drink safely" in relevant[0]
    assert "Not confirmed for this patient" in relevant[0]
    option = next(entry for entry in result["source_review_options"] if entry["card_id"] == "CHO-005")["conditional_rules"][0]
    assert option["status"] == "conditional"  # presentation did not change eligibility


def test_known_contradiction_does_not_leak_a_generated_regimen(monkeypatch):
    monkeypatch.setattr(retrieve, "retrieve_category", lambda p, c, **kw: [{"card_id": "CHI-014"}] if c == "Medication" else [])
    def model(prompt):
        if prompt.startswith("Extract patient facts"):
            return json.dumps({"facts": {}})
        return json.dumps({"medication": [{"text": "Use the standard regimen now.", "reason": "Claimed safe.",
            "evidence": [{"card_id": "CHI-014", "pdf_page": 18}]}], "missing_info": []})
    result = recommend_management({"_structured_facts": {"patient_has_fever": True,
        "patient_has_itching": False, "management_setting": "hospital", "preexisting_liver_disease": True}},
        cards=CARDS, diseases=["Chikungunya"], llm_generate=model)
    assert result["medication"] == []
    assert any(n["status"] == "restriction_applies" for n in result["safety_notices"])
    assert result["management_diagnostics"]["validation_rejections"]


def test_scope_and_age_ui_expose_safety_without_hiding_matching_fluid_guidance(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(retrieve, "retrieve_category", lambda p, c, **kw: [{"card_id": "CHI-017"}] if c == "Medication" else [])
    def model(prompt):
        return json.dumps({"facts": {}}) if prompt.startswith("Extract patient facts") else json.dumps({"missing_info": ["clinical review"]})
    result = recommend_management({"medicines": "NSAID", "_structured_facts": {
        "days_since_symptom_onset": 3, "dehydration_present": True, "patient_age_group": "adult"}},
        cards=CARDS, diseases=["Chikungunya"], llm_generate=model)
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"))
    for key, value in {"case_patient": {"age": "32 years", "medicines": "NSAID"}, "case_mode": "improved",
        "case_diseases": ["Chikungunya"], "working_disease": "Chikungunya", "case_management_requested": True,
        "case_results": {"Diagnosis": {"probable_diagnosis": []}, "Management": result}}.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception
    assert any("Restriction applies" in warning.value and "NSAIDs" in warning.value for warning in at.warning)
    assert any("Assess for dehydration and institute proper rehydration" in item.value for item in at.markdown)
    assert not any("Only when symptoms have lasted more than 10 days" in item.value for item in at.markdown)
