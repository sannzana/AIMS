"""Offline checks for generic source facts and provenance."""
from types import SimpleNamespace

from fact_verification import rule_fields, verify_facts, verify_boolean_facts, verify_measurement_facts


def test_source_rule_fields_are_discovered_without_disease_names():
    card = SimpleNamespace(model_dump=lambda: {
        "conditions_decision_logic": [
            {"condition": {"all": [
                {"field": "laboratory_value", "operator": ">=", "value": 12},
                {"field": "finding_present", "operator": "==", "value": True}
            ]}}
        ]
    })
    assert rule_fields({"a": card}) == {"laboratory_value", "finding_present"}


def test_numeric_fact_requires_exact_source_quote_and_value():
    good = {"laboratory_value": {"value": 12, "quote": "lab result 12 units"}}
    facts, rejected = verify_facts(good, {"measurements": "lab result 12 units"}, {"laboratory_value"})
    assert facts == {"laboratory_value": 12}
    assert rejected == {}


def test_invalid_measurement_is_not_silently_accepted():
    proposed = {"laboratory_value": {"value": 55, "quote": "lab result 12 units"}}
    facts, rejected = verify_facts(proposed, {"measurements": "lab result 12 units"}, {"laboratory_value"})
    assert facts == {}
    assert "laboratory_value" in rejected


def test_no_fact_without_quote_or_evidence_rule_field():
    proposed = {"arbitrary": {"value": 1, "quote": "lab 1"},
                "laboratory_value": 1}
    facts, rejected = verify_facts(proposed, {"measurements": "lab 1"}, {"laboratory_value"})
    assert facts == {}
    assert len(rejected) == 2


def test_negated_finding_is_not_inverted_into_positive_boolean():
    result, rejected = verify_facts(
        {"finding_present": {"value": True, "quote": "no finding"}},
        {"symptoms": "no finding"}, {"finding_present"})
    assert "finding_present" not in result
    assert "finding_present" in rejected


def test_unknown_is_not_false():
    result, rejected = verify_facts(
        {"finding_present": {"value": None, "quote": "finding not measured"}},
        {"history": "finding not measured"}, {"finding_present"})
    assert result == {}
    assert rejected == {}


def test_confirmed_boolean_requires_separate_semantic_pass():
    patient = {"symptoms": "I do not have the reported clinical finding"}
    proposed = {"finding_present": {
        "value": False, "quote": "do not have the reported clinical finding"
    }}
    prompts = []
    def check(prompt):
        prompts.append(prompt)
        return '{"decisions":{"finding_present":"confirmed"}}'
    values, errors = verify_boolean_facts(proposed, patient, {"finding_present"}, check)
    assert values == {"finding_present": False}
    assert errors == {}
    assert "negation scope" in prompts[0]


def test_boolean_cannot_be_confirmed_by_quote_alone():
    patient = {"symptoms": "I do not have the reported clinical finding"}
    proposed = {"finding_present": {
        "value": True, "quote": "do not have the reported clinical finding"
    }}
    values, errors = verify_boolean_facts(
        proposed, patient, {"finding_present"},
        lambda prompt: '{"decisions":{"finding_present":"rejected"}}'
    )
    assert values == {}
    assert "finding_present" in errors


def test_boolean_verifier_unavailable_is_unknown():
    proposed = {"finding_present": {"value": True, "quote": "finding exists"}}
    def offline(_):
        raise RuntimeError("offline")
    values, errors = verify_boolean_facts(
        proposed, {"symptoms": "finding exists"}, {"finding_present"}, offline
    )
    assert values == {}
    assert "finding_present" in errors


def test_model_cannot_introduce_boolean_for_unknown_field():
    proposed = {"not_in_source": {"value": True, "quote": "finding exists"}}
    values, _ = verify_boolean_facts(
        proposed, {"symptoms": "finding exists"}, {"other_field"},
        lambda _: '{"decisions":{"not_in_source":"confirmed"}}'
    )
    assert values == {}


def test_measurement_requires_independent_context_confirmation():
    proposals = {"first_measurement": {"value": 110, "quote": "110/70"},
                 "second_measurement": {"value": 70, "quote": "110/70"}}
    patient = {"measurements": "110/70"}
    def verify(prompt):
        return '{"decisions":{"first_measurement":"confirmed","second_measurement":"uncertain"}}'
    facts, rejected = verify_measurement_facts(
        proposals, patient, {"first_measurement", "second_measurement"}, verify)
    assert facts == {"first_measurement": 110}
    assert "second_measurement" in rejected


def test_measurement_verification_failure_is_unknown():
    def unavailable(_):
        raise RuntimeError("LLM unavailable")
    facts, rejected = verify_measurement_facts(
        {"measurement": {"value": 17, "quote": "result 17"}},
        {"measurements": "result 17"}, {"measurement"}, unavailable)
    assert facts == {}
    assert "measurement" in rejected
