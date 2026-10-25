"""Classification gating on the real 108 cards (retrieval and LLM are mocked)."""
import json
from pathlib import Path

import retrieve
from build_index import load_cards
from derived_fields import blocked_cards, derive_classifications
from disease_triage import triage
from recommend import recommend_management

CARDS = {c.card_id: c for c in load_cards(Path("data/cards.jsonl"))}
POOL = {"Medication": ["CHO-007", "CHO-009", "CHO-006", "CHO-013"],  # similarity prefers milder plans
        "Advice": ["CHO-011", "CHO-012"], "Investigation": []}


def _fake_retrieve(patient, category, top_k=2, **kw):
    return [{"card_id": i} for i in POOL[category][:top_k]]


def _llm(cite):
    def llm(prompt):
        card = CARDS[cite]
        return json.dumps({"medication": [{
            "text": "x", "reason": "y",
            "evidence": [{"card_id": cite, "pdf_page": card.pdf_page_numbers[0]}]}],
            "missing_info": []})
    return llm


def _run(monkeypatch, facts, cite="CHO-004"):
    monkeypatch.setattr(retrieve, "retrieve_category", _fake_retrieve)
    return recommend_management({"_structured_facts": facts, "age": 30}, mode="improved",
                                cards=CARDS, diseases=["Cholera"], llm_generate=_llm(cite))


def test_severe_signs_select_plan_c(monkeypatch):
    out = _run(monkeypatch, {"lethargic_or_unconscious": True})
    d = out["management_diagnostics"]
    assert d["derived_classification"]["fields"]["dehydration_classification"]["value"] == \
        "severe_dehydration_or_shock"
    assert "CHO-004" in d["prompt_card_ids"]
    assert not {"CHO-007", "CHO-009"} & set(d["prompt_card_ids"])
    assert "CHO-011" not in d["prompt_card_ids"]


def test_unknown_class_blocks_plan_cards_but_keeps_supportive_cards(monkeypatch):
    out = _run(monkeypatch, {"sunken_eyes": True}, cite="CHO-007")
    d = out["management_diagnostics"]
    assert set(d["blocked_card_ids"]) >= {"CHO-004", "CHO-007", "CHO-009", "CHO-011"}
    assert not set(d["blocked_card_ids"]) & set(d["prompt_card_ids"])
    # supportive cards that do not depend on the class remain available
    assert "CHO-006" in d["prompt_card_ids"] and "CHO-012" in d["prompt_card_ids"]
    # a definitive plan citing a blocked card cannot get through
    assert out["medication"] == []
    assert any("Not shown until" in m for m in out["missing_info"])
    assert any("lethargic_or_unconscious" in m for m in out["missing_info"])


def test_management_keeps_unknown_medication_and_advice_conditional(monkeypatch):
    def retrieve_management(_patient, category, top_k=2, **_kwargs):
        return [
            {"card_id": card_id}
            for card_id in {
                "Medication": ["CHO-020"],
                "Advice": ["CHO-012"],
                "Investigation": [],
            }[category][:top_k]
        ]

    monkeypatch.setattr(retrieve, "retrieve_category", retrieve_management)

    def model(_prompt):
        return json.dumps({
            "medication": [{
                "text": "Administer IV dextrose now.",
                "reason": "Treat hypoglycaemia.",
                "evidence": [{
                    "card_id": "CHO-020",
                    "pdf_page": CARDS["CHO-020"].pdf_page_numbers[0],
                }],
            }],
            "advice": [{
                "text": "Discharge the patient.",
                "reason": "Discharge criteria are met.",
                "evidence": [{
                    "card_id": "CHO-012",
                    "pdf_page": CARDS["CHO-012"].pdf_page_numbers[0],
                }],
            }],
            "missing_info": [],
        })

    out = recommend_management(
        {"age": 34, "_structured_facts": {}},
        mode="improved",
        cards=CARDS,
        diseases=["Cholera"],
        llm_generate=model,
    )

    assert out["medication"] and out["advice"]
    assert out["medication"][0]["eligibility"] == "conditional"
    assert "Conditional source guidance" in out["advice"][0]["reason"]
    assert any("blood_glucose" in item for item in out["missing_info"])


def test_clinician_supplied_class_is_used_not_overwritten(monkeypatch):
    out = _run(monkeypatch, {"dehydration_classification": "some_dehydration"}, cite="CHO-007")
    d = out["management_diagnostics"]
    assert "CHO-011" not in d["blocked_card_ids"]
    assert "CHO-007" in d["prompt_card_ids"] and "CHO-004" not in d["prompt_card_ids"]
    assert out["medication"]


def test_unknown_never_derives_and_independent_labels_not_blocked():
    derived, trace = derive_classifications(CARDS.values(), "Cholera", {})
    assert derived == {} and blocked_cards(trace)
    # Chikungunya's classification labels have no consumers and no gate: reported, blocks nothing
    derived, trace = derive_classifications(CARDS.values(), "Chikungunya", {})
    assert derived == {} and blocked_cards(trace) == {}
    assert set(trace["labels"]) == {"possible_case", "probable_case", "confirmed_case"}


def test_dependency_comes_from_the_source_not_card_ids():
    cards = dict(CARDS)
    sev = cards["CHO-002"]
    assert "CHO-003" in derive_classifications(CARDS.values(), "Cholera", {})[1]["dependencies"]
    # remove the source's next_card link: the milder labels are no longer gated
    ungated = sev.model_copy(update={"if_else_conditions_and_corresponding_actions": []})
    cards["CHO-002"] = ungated
    _, trace = derive_classifications(cards.values(), "Cholera", {})
    assert "CHO-003" not in trace["dependencies"]
    assert "note" not in trace["labels"]["some_dehydration"]


def test_exclusive_labels_true_together_are_reported_as_conflict():
    cards = dict(CARDS)
    cards["CHO-002"] = cards["CHO-002"].model_copy(update={"if_else_conditions_and_corresponding_actions": []})
    always = {"all": [{"field": "x_flag", "operator": "=", "value": True}]}
    logic = [dict(r, condition=always) for r in cards["CHO-003"].model_dump()["conditions_decision_logic"]]
    cards["CHO-003"] = cards["CHO-003"].model_copy(update={"conditions_decision_logic": logic})
    derived, trace = derive_classifications(cards.values(), "Cholera", {"x_flag": True})
    assert derived == {}
    assert trace["fields"]["dehydration_classification"]["status"] == "conflict"
    assert trace["conflicts"]


def test_disease_without_classification_cards_is_unchanged(monkeypatch):
    monkeypatch.setattr(retrieve, "retrieve_category",
                        lambda p, c, top_k=2, **kw: [{"card_id": i} for i in
                        {"Medication": ["CHI-014"], "Advice": [], "Investigation": []}[c][:top_k]])
    out = recommend_management({"age": 30}, mode="improved", cards=CARDS, diseases=["Chikungunya"],
                               llm_generate=lambda p: json.dumps({"missing_info": ["x"]}))
    d = out["management_diagnostics"]
    assert d["blocked_card_ids"] == [] and "CHI-014" in d["prompt_card_ids"]


def test_chikungunya_supportive_cards_remain_conditional_when_facts_are_unknown(monkeypatch):
    monkeypatch.setattr(
        retrieve,
        "retrieve_category",
        lambda p, c, top_k=2, **kw: [
            {"card_id": card_id}
            for card_id in {
                "Medication": ["CHI-014"],
                "Advice": ["CHI-015"],
                "Investigation": [],
            }[c][:top_k]
        ],
    )
    out = recommend_management(
        {"age": 30},
        mode="improved",
        cards=CARDS,
        diseases=["Chikungunya"],
        llm_generate=lambda p: json.dumps({"missing_info": ["clinical context needed"]}),
    )
    diagnostics = out["management_diagnostics"]

    assert {"CHI-014", "CHI-015"} <= set(diagnostics["prompt_card_ids"])
    assert diagnostics["applicability_by_card"]["CHI-014"] == "undetermined"
    assert diagnostics["applicability_by_card"]["CHI-015"] == "undetermined"
    assert not {"CHI-014", "CHI-015"} & set(diagnostics["not_applicable_card_ids"])


def test_rule_based_missing_info_survives_llm_missing_info(monkeypatch):
    monkeypatch.setattr(retrieve, "retrieve_category", _fake_retrieve)
    out = recommend_management({"_structured_facts": {}, "age": 30}, mode="improved",
                               cards=CARDS, diseases=["Cholera"],
                               llm_generate=lambda p: json.dumps({"missing_info": ["model note"]}))
    assert any("Not shown until" in m for m in out["missing_info"])
    assert "model note" in out["missing_info"]


def test_source_facts_reach_severe_classification_from_synonym_and_measurements():
    patient = {
        "clinical_findings": "weak rapid pulse, sunken eyes, dry mouth",
        "measurements": "BP 90/60 mmHg, pulse 118 bpm",
    }
    def model(prompt):
        if "separate semantic fact verification pass" in prompt:
            return json.dumps({"decisions": {
                "absent_or_weak_pulse": "confirmed",
                "sunken_eyes": "confirmed",
            }})
        return json.dumps({"candidates": [], "facts": {
            "absent_or_weak_pulse": {
                "value": True, "quote": "weak rapid pulse"
            },
            "sunken_eyes": {
                "value": True, "quote": "sunken eyes"
            },
        }})

    ranking = triage(
        patient,
        CARDS,
        llm_generate=model,
        candidate_search=lambda _patient, _cards: ["CHI-011", "DEN-DF-DX-001"],
    )
    derived, trace = derive_classifications(
        CARDS.values(), "Cholera", ranking["features"]
    )

    assert ranking["features"]["absent_or_weak_pulse"] is True
    assert ranking["features"]["sunken_eyes"] is True
    assert derived["dehydration_classification"] == "severe_dehydration_or_shock"
    assert trace["fields"]["dehydration_classification"]["status"] == "decided"


def test_cholera_case_definition_reports_missing_outbreak_status():
    patient = {
        "symptoms": (
            "sudden profuse watery rice-water diarrhoea, repeated vomiting, "
            "intense thirst, weakness, leg cramps"
        ),
        "clinical_findings": (
            "sunken eyes, dry mouth, reduced urine output, weak rapid pulse"
        ),
        "age": 34,
        "measurements": "BP 90/60 mmHg, pulse 118 bpm, temperature 37 C",
    }

    def model(prompt):
        if "separate semantic fact verification pass" in prompt:
            return json.dumps({
                "decisions": {"acute_watery_diarrhoea": "confirmed"}
            })
        return json.dumps({
            "candidates": [{
                "card_id": "CHO-001",
                "decision": "uncertain",
                "patient_quotes": ["watery rice-water diarrhoea"],
                "missing": ["cholera_outbreak_declared"],
            }],
            "facts": {
                "acute_watery_diarrhoea": {
                    "value": True,
                    "quote": "watery rice-water diarrhoea",
                },
            },
        })

    ranking = triage(
        patient,
        CARDS,
        llm_generate=model,
        candidate_search=lambda _patient, _cards: ["CHO-001"],
    )

    diagnostic = ranking["diagnostics"]["candidate_diagnostics"][0]
    assert ranking["features"]["acute_watery_diarrhoea"] is True
    assert ranking["candidates"][0]["decision"] == "uncertain"
    assert any(
        "cholera_outbreak_declared" in rule["fields"]
        and rule["status"] == "unknown"
        for rule in diagnostic["source_criteria"]
    )
    assert "acute_watery_diarrhoea" in diagnostic["verified_facts"]


def test_overlapping_chikungunya_dengue_candidates_keep_supported_and_uncertain_status():
    patient = {
        "symptoms": "high fever, severe joint pain, headache, rash",
        "clinical_findings": "swollen and tender joints, difficulty walking",
    }
    def model(prompt):
        return json.dumps({"candidates": [
            {"card_id": "CHI-011", "decision": "supported",
             "patient_quotes": ["severe joint pain"], "missing": []},
            {"card_id": "DEN-DF-DX-001", "decision": "uncertain",
             "patient_quotes": ["high fever", "rash"],
             "missing": ["additional dengue evaluation"]},
        ], "facts": {}})

    ranking = triage(
        patient,
        CARDS,
        llm_generate=model,
        candidate_search=lambda _patient, _cards: ["CHI-011", "DEN-DF-DX-001"],
    )

    assert {item["disease"] for item in ranking["candidates"]} == {
        "Chikungunya", "Dengue"
    }
    assert {item["decision"] for item in ranking["candidates"]} == {
        "supported", "uncertain"
    }


def test_cyclic_classification_dependencies_are_unresolved():
    cards = []
    for card_id, label, other_card, other_label in (
        ("CHO-002", "alpha", "CHO-003", "beta"),
        ("CHO-003", "beta", "CHO-002", "alpha"),
    ):
        card = CARDS[card_id].model_copy(update={
            "conditions_decision_logic": [{
                "condition": {"field": "flag", "operator": "=", "value": True},
                "action": f"classify_{label}",
            }],
            "if_else_conditions_and_corresponding_actions": [{
                "if": {"field": other_label, "operator": "=", "value": True},
                "then": {"next_card": other_card},
            }],
        })
        cards.append(card)

    derived, trace = derive_classifications(cards, "Cholera", {"flag": True})

    assert derived == {}
    assert all(trace["labels"][label]["value"] == "UNKNOWN" for label in ("alpha", "beta"))
    assert all("cyclic classification dependency" in trace["labels"][label]["note"]
               for label in ("alpha", "beta"))
