"""Offline tests for source-only triage; no live model needed."""
import json
from types import SimpleNamespace
from disease_triage import triage

def card(cid, disease, criteria):
    return SimpleNamespace(
        card_id=cid, disease_protocol=disease, clinical_category="Diagnosis",
        card_title="Case Definition", clinical_classification_presentation=None,
        model_dump=lambda: {
            "clinical_category":"Diagnosis", "card_title":"Case Definition",
            "clinical_classification_presentation":None,
            "relevant_symptoms_or_prerequisites": criteria,
            "conditions_decision_logic":[]})

def test_grounded_source_protocol_not_model_disease():
    cards={"a":card("a","Protocol Alpha","a symptom"),
           "b":card("b","Protocol Beta","another symptom")}
    mock=lambda prompt:json.dumps({"candidates":[{"card_id":"a",
        "decision":"supported","patient_quotes":["a symptom"],"missing":[]}],
        "facts":{}})
    got=triage({"symptoms":"a symptom"},cards,llm_generate=mock)
    assert got["candidates"][0]["disease"]=="Protocol Alpha"
    assert got["status"]=="clear"

def test_rejects_hallucinated_patient_quote():
    cards={"a":card("a","Protocol Alpha","a symptom")}
    mock=lambda prompt:json.dumps({"candidates":[{"card_id":"a",
        "decision":"supported","patient_quotes":["invented finding"],"missing":[]}],
        "facts":{}})
    assert triage({"symptoms":"a symptom"},cards,llm_generate=mock)["status"]=="no_match"

def test_provider_failure_has_no_keyword_fallback():
    cards={"a":card("a","Protocol Alpha","a symptom")}
    result=triage({"symptoms":"a symptom"},cards,llm_generate=lambda _: "{}" if False else "not json")
    assert result["status"]=="unavailable"
    assert result["candidates"]==[]

def test_negation_is_in_prompt_and_no_numeric_cutoffs_in_module():
    cards={"a":card("a","Protocol Alpha","a symptom")}
    seen=[]
    def mock(prompt):
        seen.append(prompt)
        return '{"candidates":[],"facts":{}}'
    triage({"symptoms":"no a symptom", "measurements":"test 17 units"}, cards, llm_generate=mock)
    assert "NEGATION" in seen[0] and "test 17 units" in seen[0]
