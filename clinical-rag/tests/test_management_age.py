"""Age unit normalization and exclusion of contradicted source branches."""
import json
from pathlib import Path

import pytest
import retrieve
from build_index import load_cards
from management_age import normalize_management_age
from recommend import recommend_management

CARDS = {c.card_id: c for c in load_cards(Path("data/cards.jsonl"))}


@pytest.mark.parametrize("raw", [32, 32.0, "32", "32 years", "32 year", "32 yr",
                                 "32yr", "32yrs", " 32 YR ", "32y"])
def test_year_aliases_and_default(raw):
    context, info = normalize_management_age({"age": raw}, {})
    assert context == {"age_years": 32, "age_months": 384}
    assert info["status"] == "normalized"
    assert "patient_age_group" not in context  # no invented adult/child boundary


@pytest.mark.parametrize("raw", ["32 months", "32 month", "32month", "32mo", "32 mos", "32m"])
def test_month_aliases(raw):
    context, info = normalize_management_age({"age": raw}, {})
    assert context["age_months"] == 32
    assert context["age_years"] == pytest.approx(32 / 12)
    assert info["unit"] == "months"


@pytest.mark.parametrize("raw", [True, -1, "-1 years", "unknown", "32 days", "32 or 33", float("inf")])
def test_ambiguous_or_unsupported_input_does_not_invent_age(raw):
    context, info = normalize_management_age({"age": raw}, {})
    assert context == {} and info["status"] == "invalid"


def test_existing_age_fact_conflicts_are_reported_and_preserved():
    known = {"age_years": 5, "some_other_fact": False}
    context, info = normalize_management_age({"age": "32yr"}, known)
    assert context == known and info["status"] == "conflict"
    assert known == {"age_years": 5, "some_other_fact": False}


def test_missing_and_decimal_age():
    assert normalize_management_age({}, {}) == ({}, {"status": "missing"})
    context, _ = normalize_management_age({"age": "1.5 yr"}, {})
    assert context == {"age_years": 1.5, "age_months": 18}


@pytest.mark.parametrize("age", [32, "32", "32yr", "32 years", "32 months"])
def test_older_patient_does_not_see_under_24_month_branch(monkeypatch, age):
    monkeypatch.setattr(retrieve, "retrieve_category", lambda p, c, **kw:
                        [{"card_id": "CHO-005"}] if c == "Medication" else [])
    calls = []
    def model(prompt):
        calls.append(prompt)
        return json.dumps({"facts": {}}) if prompt.startswith("Extract patient facts") else json.dumps({"missing_info": ["review"]})
    result = recommend_management({"age": age, "_structured_facts": {"dehydration_classification": "severe_dehydration_or_shock"}},
                                  cards=CARDS, diseases=["Cholera"], llm_generate=model)
    options = [o for entry in result["source_review_options"] for kind in ("supported_rules", "conditional_rules") for o in entry[kind]]
    assert not any(o["rule_id"] == "CHO-C-ORS-001" for o in options)
    assert any(o["rule_id"] == "CHO-C-ORS-002" for o in options)
    requested = result["management_diagnostics"]["management_fact_extraction"]["requested_fields"]
    assert "age_months" not in requested and "age_years" not in requested


def test_infant_keeps_correct_age_branch(monkeypatch):
    monkeypatch.setattr(retrieve, "retrieve_category", lambda p, c, **kw:
                        [{"card_id": "CHO-005"}] if c == "Medication" else [])
    result = recommend_management({"age": "12months", "_structured_facts": {"dehydration_classification": "severe_dehydration_or_shock"}},
                                  cards=CARDS, diseases=["Cholera"],
                                  llm_generate=lambda p: json.dumps({"facts": {}, "missing_info": ["review"]}))
    options = [o for entry in result["source_review_options"] for kind in ("supported_rules", "conditional_rules") for o in entry[kind]]
    assert any(o["rule_id"] == "CHO-C-ORS-001" for o in options)
    assert not any(o["rule_id"] == "CHO-C-ORS-002" for o in options)


def test_baseline_does_not_normalize_age(monkeypatch):
    monkeypatch.setattr(retrieve, "retrieve_category", lambda *a, **kw: [])
    result = recommend_management({"age": "32yr"}, mode="baseline", cards=CARDS,
                                  diseases=["Cholera"], llm_generate=lambda p: json.dumps({"missing_info": ["review"]}))
    assert "age_normalization" not in result["management_diagnostics"]
