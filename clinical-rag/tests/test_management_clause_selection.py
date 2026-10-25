"""Age selection follows explicit source bands without changing clinical gates."""
from pathlib import Path

import pytest
from build_index import load_cards
from management_clauses import patient_age_months, select_age_clauses
from management_evidence import rule_options

CARDS = {c.card_id: c for c in load_cards(Path("data/cards.jsonl"))}
TEXT = "For ages 2–9 years give option X; for ages 10 years and older give option Y."


@pytest.mark.parametrize("age,expected", [(24, "option X"), (60, "option X"), (108, "option X"),
                                        (120, "option Y"), (444, "option Y")])
def test_matching_source_band_only(age, expected):
    result = select_age_clauses(TEXT, age)
    assert result["status"] == "matched" and expected in result["text"]
    assert len(result["matched"]) == 1 and len(result["omitted"]) == 1


@pytest.mark.parametrize("age", [None, float("nan"), float("inf"), -1, True, "37"])
def test_unknown_or_invalid_age_never_selects_a_band(age):
    assert select_age_clauses(TEXT, age)["status"] == "age_unknown"


def test_fractional_boundaries_are_not_broadened_by_a_year():
    assert select_age_clauses(TEXT, 9.5 * 12)["status"] == "boundary_review"
    assert select_age_clauses("For ages 1.5–2.5 years give X.", 3 * 12)["status"] == "excluded"


@pytest.mark.parametrize("text", [
    "For ages 2–9 years give X; repeat this dose; for ages 10 years and older give Y.",
    "For ages over five years give X; for ages 10 years and older give Y.",
    "For ages 9–2 years give X.",
    "For ages 2–9 years give X, for ages 10 years and older give Y.",
])
def test_ambiguous_prose_is_not_reassociated_with_another_regimen(text):
    result = select_age_clauses(text, 444)
    assert result["status"] == "unparsed" and result["text"] == text


def test_overlapping_bands_and_unrelated_durations():
    assert select_age_clauses("For ages 2–12 years give X; for ages 10 years and older give Y.", 132)["status"] == "overlapping_bands"
    for text in ("Review after 1 month.", "IgM may persist for 1–3 months.", "Use the age-specific source schedule."):
        assert select_age_clauses(text, 444)["status"] == "no_age_clause"


def test_source_threshold_changes_are_respected_without_special_cases():
    text = "Aged under 7 months give P; aged 7 months or older give Q."
    assert "give P" in select_age_clauses(text, 6)["text"]
    assert "give Q" in select_age_clauses(text, 7)["text"]
    text = text.replace("7 months", "11 months")
    assert "give P" in select_age_clauses(text, 7)["text"]


def test_context_age_consistency():
    assert patient_age_months({"age_years": 37, "age_months": 444}) == 444
    assert patient_age_months({"age_years": 37, "age_months": 12}) is None
    assert patient_age_months({"age_years": float("nan")}) is None


def test_known_indication_does_not_authorise_unknown_age_alternatives():
    card = CARDS["CHO-005"].model_copy(update={
        "card_id": "RENAMED-1", "disease_protocol": "Synthetic",
        "conditions_decision_logic": [{"rule_id": "AGE-1", "condition": {
            "field": "known_marker", "operator": "=", "value": True}, "result": TEXT}],
        "if_else_conditions_and_corresponding_actions": []})
    option = rule_options(card, {"known_marker": True})[0]
    assert option["status"] == "conditional"
    assert "age_months" in option["missing_inputs"]


@pytest.mark.parametrize("card_id,rule_id", [("CHO-005", "CHO-C-ORS-002"), ("CHO-008", "CHO-B-ORS-002")])
def test_actual_card_keeps_only_older_age_clause(card_id, rule_id):
    option = next(o for o in rule_options(CARDS[card_id], {"age_years": 37, "age_months": 444}) if o["rule_id"] == rule_id)
    assert "2–9" not in option["source_instruction"]
    assert "10 years and older" in option["source_instruction"]


def test_unparsed_age_does_not_remain_supported():
    card = CARDS["CHO-005"].model_copy(update={"conditions_decision_logic": [{
        "rule_id": "UNPARSED", "condition": {"field": "marker", "operator": "=", "value": True},
        "result": "For ages over five years give X."}]})
    option = rule_options(card, {"marker": True, "age_months": 444})[0]
    assert option["status"] == "conditional"
    assert "source_age_clause_review" in option["missing_inputs"]
