"""Regression cases: source-only optional clarification cannot alter diagnosis."""
import copy
import json
from pathlib import Path

import pytest
from management_clarification import questions, apply_answers
from management_evidence import rule_options, review_summary
from schema import Card
from test_management_presentation import _result, CARDS


def test_current_case_asks_drink_safely_without_reasking_verified_findings(monkeypatch):
    result = _result(monkeypatch, {"absent_or_weak_pulse": True, "sunken_eyes": True})
    before = copy.deepcopy(result)
    offered = questions(result, CARDS)
    assert "patient_can_drink_safely" in {q["field"] for q in offered}
    assert not {"absent_or_weak_pulse", "sunken_eyes", "age_months", "age_years"} & {q["field"] for q in offered}
    assert len(offered) <= 5
    assert all(q["sources"] for q in offered)
    assert result == before


@pytest.mark.parametrize("answer", [True, False])
def test_answers_reassess_only_source_guidance_with_no_llm_or_retrieval(monkeypatch, answer):
    result = _result(monkeypatch, {"absent_or_weak_pulse": True})
    before = copy.deepcopy(result)
    offered = questions(result, CARDS)
    import retrieve
    monkeypatch.setattr(retrieve, "retrieve_category", lambda *a, **k: pytest.fail("No retrieval during clarification"))
    updated = apply_answers(result, CARDS, {"patient_can_drink_safely": answer}, offered)
    assert result == before
    assert updated["management_diagnostics"]["resolved_source_context"]["treatment_plan"] == "Plan C"
    option = next(e for e in updated["source_review_options"] if e["card_id"] == "CHO-005") if answer else None
    if answer:
        assert option["supported_rules"]
        assert all("2–9" not in (r["source_instruction"] or "") for r in option["supported_rules"])
    else:
        assert not any(e["card_id"] == "CHO-005" and e["supported_rules"] for e in updated["source_review_options"])
    assert updated["management_diagnostics"]["clarification"]["llm_calls"] == 0
    assert updated["probable_diagnosis"] == result["probable_diagnosis"]
    assert not updated["medication"]  # stale LLM prose must not be mixed with new facts
    assert not questions(updated, CARDS, {q["field"] for q in offered}) or all(
        q["field"] != "patient_can_drink_safely" for q in questions(updated, CARDS))
    assert updated == apply_answers(result, CARDS, {"patient_can_drink_safely": answer}, offered)


def test_unknown_and_skip_leave_result_exactly_unchanged(monkeypatch):
    result = _result(monkeypatch, {"absent_or_weak_pulse": True})
    offered = questions(result, CARDS)
    assert apply_answers(result, CARDS, {}, offered) == result
    assert apply_answers(result, CARDS, {q["field"]: None for q in offered}, offered) == result
    assert not questions(result, CARDS, {q["field"] for q in offered})


def test_unknown_classification_can_ask_findings_but_not_computed_counts(monkeypatch):
    result = _result(monkeypatch, {})
    offered = questions(result, CARDS)
    assert offered
    assert not any(q["field"].endswith("_count") or q["field"].startswith("all_") for q in offered)
    assert any(q["field"] == "absent_or_weak_pulse" for q in offered)
    assert not any(q["field"] == "dehydration_classification" for q in offered)
    updated = apply_answers(result, CARDS, {"absent_or_weak_pulse": True}, offered)
    assert updated["management_diagnostics"]["resolved_source_context"]["treatment_plan"] == "Plan C"


def test_invalid_and_conflicting_answers_do_not_modify_original(monkeypatch):
    result = _result(monkeypatch, {"absent_or_weak_pulse": True})
    offered = questions(result, CARDS)
    before = copy.deepcopy(result)
    with pytest.raises(ValueError):
        apply_answers(result, CARDS, {"patient_can_drink_safely": "yes"}, offered)
    with pytest.raises(ValueError):
        apply_answers(result, CARDS, {"invented": True}, offered)
    updated = apply_answers(result, CARDS, {"patient_can_drink_safely": True}, offered)
    with pytest.raises(ValueError, match="conflicts"):
        apply_answers(updated, CARDS, {"patient_can_drink_safely": False}, offered)
    assert result == before


def _synthetic():
    raw = CARDS["CHO-004"].model_dump()
    raw.update(card_id="RENAMED", card_title="Unrelated source", disease_protocol="Other protocol",
               conditions_decision_logic=[{"rule_id": "R1", "condition": {"all": [
                   {"field": "marker", "operator": "=", "value": True},
                   {"field": "reading_kg", "operator": ">", "value": 7}]}, "result": "Source instruction"}],
               if_else_conditions_and_corresponding_actions=[], exceptions_contraindications=[],
               relevant_symptoms_or_prerequisites="", clinical_classification_presentation="")
    card = Card.model_validate(raw)
    facts = {"marker": True}
    result = {"_clarification_context": facts, "working_diagnosis": card.disease_protocol,
              "management_diagnostics": {}, "probable_diagnosis": [],
              "source_review_options": review_summary(rule_options(card, facts))}
    return card, result


@pytest.mark.parametrize("number", [8, 5])
def test_renamed_source_uses_its_own_numeric_threshold(number):
    card, result = _synthetic()
    offered = questions(result, {card.card_id: card})
    assert [q["field"] for q in offered] == ["reading_kg"]
    assert offered[0]["unit"] == "kg"
    updated = apply_answers(result, {card.card_id: card}, {"reading_kg": number}, offered)
    assert bool(updated["source_review_options"]) is (number > 7)


@pytest.mark.parametrize("number", [-1, float("nan"), float("inf"), True, "8"])
def test_measurements_fail_closed(number):
    card, result = _synthetic()
    with pytest.raises(ValueError):
        apply_answers(result, {card.card_id: card}, {"reading_kg": number}, questions(result, {card.card_id: card}))


def test_ui_apply_preserves_original_diagnosis_and_reset_restores_original(monkeypatch):
    from streamlit.testing.v1 import AppTest
    result = _result(monkeypatch, {"absent_or_weak_pulse": True})
    diagnosis = {"probable_diagnosis": [], "notice": "UNCHANGED DIAGNOSIS"}
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"))
    for key, value in {"case_patient": {"age": "37 years"}, "case_mode": "improved", "case_diseases": ["Cholera"],
        "working_disease": "Cholera", "case_management_requested": True,
        "case_results": {"Diagnosis": diagnosis, "Management": result}}.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception
    question = next(s for s in at.selectbox if s.label == "patient can drink safely")
    question.select("Yes")
    next(b for b in at.button if b.label == "Apply answers").click()
    at.run()
    assert not at.exception
    assert at.session_state["case_results"]["Diagnosis"] == diagnosis
    assert at.session_state["case_results"]["Management"] == result
    assert at.session_state["management_clarification"]["view"]["_clarification_answers"]["patient_can_drink_safely"] is True
    next(b for b in at.button if b.label == "Reset additional answers").click()
    at.run()
    assert not at.exception
    assert at.session_state["management_clarification"]["view"] == result


def test_medication_safety_questions_and_answers_are_instruction_scoped(monkeypatch):
    import retrieve
    from recommend import recommend_management
    pool = {"Investigation": ["CHI-023"], "Medication": ["CHI-014", "CHI-017"], "Advice": ["CHI-015"]}
    monkeypatch.setattr(retrieve, "retrieve_category", lambda p, c, **kw: [{"card_id": cid} for cid in pool[c]])
    facts = {"patient_has_fever": True, "patient_has_itching": True}
    result = recommend_management({"age": "32 years", "_structured_facts": facts}, cards=CARDS,
        diseases=["Chikungunya"], llm_generate=lambda prompt: json.dumps({"facts": {}}) if prompt.startswith("Extract patient facts") else "{}")
    offered = questions(result, CARDS)
    fields = {q["field"] for q in offered}
    assert {"preexisting_liver_disease", "preexisting_kidney_disease"} <= fields
    assert "patient_has_fever" not in fields
    updated = apply_answers(result, CARDS, {"preexisting_liver_disease": True, "preexisting_kidney_disease": False}, offered)
    medication = next(e for e in updated["source_review_options"] if e["card_id"] == "CHI-014")
    assert not any(r["rule_id"] == "CHI-MED-001" for r in medication["supported_rules"])
    assert any(r["rule_id"] == "CHI-MED-001" and r["status"] == "restricted" for r in medication["conditional_rules"])
    assert medication["supported_rules"]  # independent instructions survive
    assert updated["_clarification_provenance"]["preexisting_liver_disease"]["origin"] == "user-supplied"


def test_ui_unknown_submission_is_a_noop_and_does_not_repeat_questions(monkeypatch):
    from streamlit.testing.v1 import AppTest
    result = _result(monkeypatch, {"absent_or_weak_pulse": True})
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"))
    for key, value in {"case_patient": {"age": "37 years"}, "case_mode": "improved", "case_diseases": ["Cholera"],
        "working_disease": "Cholera", "case_management_requested": True,
        "case_results": {"Diagnosis": {"probable_diagnosis": []}, "Management": result}}.items():
        at.session_state[key] = value
    at.run()
    assert any(s.label == "patient can drink safely" for s in at.selectbox)
    next(b for b in at.button if b.label == "Apply answers").click()
    at.run()
    assert not at.exception
    assert at.session_state["management_clarification"]["view"] == result
    assert not any(s.label == "patient can drink safely" for s in at.selectbox)


def test_baseline_has_no_clarification_ui(monkeypatch):
    from streamlit.testing.v1 import AppTest
    result = _result(monkeypatch, {})
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"))
    for key, value in {"case_patient": {"age": "37 years"}, "case_mode": "baseline", "case_diseases": ["Cholera"],
        "case_management_requested": True,
        "case_results": {"Diagnosis": {"probable_diagnosis": []}, "Management": result}}.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception
    assert not any(e.label == "Improve treatment with additional information" for e in at.expander)
    assert not any(b.label == "Apply answers" for b in at.button)


@pytest.mark.parametrize("age", ["5 years", "9 years", "10 years", "37 years"])
def test_answers_keep_source_age_selection_patient_specific(monkeypatch, age):
    result = _result(monkeypatch, {"absent_or_weak_pulse": True})
    # Recreate using the same retrieval fixture with the requested patient age.
    from recommend import recommend_management
    result = recommend_management({"age": age, "_structured_facts": {"absent_or_weak_pulse": True}},
        cards=CARDS, diseases=["Cholera"], llm_generate=lambda p: json.dumps({"facts": {}}) if p.startswith("Extract patient facts") else "{}")
    updated = apply_answers(result, CARDS, {"patient_can_drink_safely": True}, questions(result, CARDS))
    entry = next(e for e in updated["source_review_options"] if e["card_id"] == "CHO-005")
    text = " ".join(r["source_instruction"] or "" for r in entry["supported_rules"])
    assert text
    if age in {"5 years", "9 years"}:
        assert "10 years" not in text
    else:
        assert "10 years" in text
        assert "2–9" not in text and "2-9" not in text


def test_question_limit_and_skipped_fields_advance_without_asking_later_stage():
    card, result = _synthetic()
    raw = card.model_dump()
    raw["conditions_decision_logic"] = [{"rule_id": "MANY", "condition": {"all": [
        {"field": "marker", "operator": "=", "value": True}] + [
        {"field": f"finding_{i}", "operator": "=", "value": True} for i in range(9)]}, "result": "Review source"}]
    card = Card.model_validate(raw)
    result["source_review_options"] = review_summary(rule_options(card, result["_clarification_context"]))
    first = questions(result, {card.card_id: card})
    assert len(first) == 5
    second = questions(result, {card.card_id: card}, {q["field"] for q in first})
    assert len(second) == 4
    assert not {q["field"] for q in first} & {q["field"] for q in second}
    card = Card.model_validate({**raw, "clinical_category": "Advice"})
    result["source_review_options"] = review_summary(rule_options(card, result["_clarification_context"]))
    assert not questions(result, {card.card_id: card})


def test_initial_provider_failure_does_not_prevent_offline_classification_questions(monkeypatch):
    from llm import LLMRateLimitError
    from recommend import recommend_management
    _result(monkeypatch, {})
    def fail(prompt):
        raise LLMRateLimitError("0")
    result = recommend_management({"age": "37 years"}, cards=CARDS, diseases=["Cholera"], llm_generate=fail)
    assert result["management_diagnostics"]["management_fact_extraction"]["error"]
    offered = questions(result, CARDS)
    assert "absent_or_weak_pulse" in {q["field"] for q in offered}
    updated = apply_answers(result, CARDS, {"absent_or_weak_pulse": True}, offered)
    assert updated["management_diagnostics"]["resolved_source_context"]["treatment_plan"] == "Plan C"
    assert result["management_diagnostics"]["management_fact_extraction"]["error"]  # original retained
