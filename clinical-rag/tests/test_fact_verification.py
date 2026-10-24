"""Offline checks for generic source facts and provenance."""
from types import SimpleNamespace

from fact_verification import rule_fields, verify_facts


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
