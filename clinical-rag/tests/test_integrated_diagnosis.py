"""Regression tests for source-grounded triage without legacy keyword ranking."""
import json
from types import SimpleNamespace

from disease_triage import triage
from patient_features import diagnostic_query, extract_patient_features


def card(cid, disease, criteria, logic=None):
    return SimpleNamespace(
        card_id=cid, disease_protocol=disease,
        clinical_category="Diagnosis", card_title="Clinical Case Definition",
        clinical_classification_presentation=None,
        model_dump=lambda: {
            "clinical_category":"Diagnosis",
            "card_title":"Clinical Case Definition",
            "clinical_classification_presentation":None,
            "relevant_symptoms_or_prerequisites":criteria,
            "conditions_decision_logic":logic or [],
        })


def test_no_fixed_disease_or_symptom_extractor():
    assert extract_patient_features({"symptoms":"joint pain and rash"}) == {}
    assert extract_patient_features({"_structured_facts":{"wbc_count":3800,"rash_present":False}}) == {
        "wbc_count":3800, "rash_present":False}


def test_measurements_preserved_for_semantic_triage():
    assert "WBC 3800" in diagnostic_query({"measurements":"WBC 3800 /uL"})


def test_model_cannot_invent_protocol():
    cards={"A":card("A","Protocol A","acute symptoms")}
    mock=lambda _:json.dumps({"candidates":[{"card_id":"unknown","decision":"supported",
        "patient_quotes":["acute symptoms"],"missing":[]}],"facts":{}})
    assert triage({"symptoms":"acute symptoms"},cards,llm_generate=mock)["candidates"]==[]


def test_model_cannot_invent_symptom_quote():
    cards={"A":card("A","Protocol A","acute symptoms")}
    mock=lambda _:json.dumps({"candidates":[{"card_id":"A","decision":"supported",
        "patient_quotes":["imaginary fever"],"missing":[]}],"facts":{}})
    assert triage({"symptoms":"acute symptoms"},cards,llm_generate=mock)["candidates"]==[]


def test_no_lexical_fallback_on_model_failure():
    cards={"A":card("A","Protocol A","fever and rash")}
    result=triage({"symptoms":"fever and rash"},cards,llm_generate=lambda _:"bad")
    assert result["status"]=="unavailable"


def test_source_rules_are_sent_without_python_threshold():
    cards={"A":card("A","Protocol A","blood pressure",[
        {"rule_id":"a1","condition":{"field":"sbp_mmhg","operator":">=","value":170}}])}
    prompts=[]
    def mock(prompt):
        prompts.append(prompt)
        return json.dumps({"candidates":[],"facts":{}})
    triage({"measurements":"BP 180/95 mmHg"},cards,llm_generate=mock)
    assert '"value": 170' in prompts[0]
    assert "BP 180/95 mmHg" in prompts[0]


def test_multiple_supported_diseases_are_preserved():
    cards={"A":card("A","Protocol A","joint pain"),"B":card("B","Protocol B","rash")}
    mock=lambda _:json.dumps({"candidates":[
        {"card_id":"A","decision":"supported","patient_quotes":["joint pain"],"missing":[]},
        {"card_id":"B","decision":"supported","patient_quotes":["rash"],"missing":[]}],"facts":{}})
    result=triage({"symptoms":"joint pain and rash"},cards,llm_generate=mock)
    assert result["status"]=="ambiguous"
    assert len(result["candidates"])==2
