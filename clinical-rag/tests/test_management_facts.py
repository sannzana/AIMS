"""Treatment-branch facts are collected for management (real cards, mocked LLM)."""
import json
from pathlib import Path

import retrieve
from build_index import load_cards
from management_facts import extract_management_facts, needed_fields
from recommend import recommend_management

CARDS = {c.card_id: c for c in load_cards(Path("data/cards.jsonl"))}
POOL = {"Medication": ["CHO-007", "CHO-009", "CHO-004", "CHO-006"],
        "Advice": ["CHO-011", "CHO-012"], "Investigation": []}
PATIENT = {"symptoms": "profuse watery diarrhoea and vomiting",
           "clinical_findings": "patient is lethargic with a weak pulse", "age": 30}


def _retrieve(patient, category, top_k=2, **kw):
    return [{"card_id": i} for i in POOL[category][:top_k]]


def _llm(extract, confirm="confirmed", calls=None):
    def llm(prompt):
        if calls is not None:
            calls.append(prompt[:40])
        if prompt.startswith("Extract patient facts"):
            if isinstance(extract, Exception):
                raise extract
            return json.dumps({"facts": extract})
        if "semantic fact verification" in prompt:
            fields = json.loads(prompt.split("PROPOSED FACTS:")[1].replace("\\n", "", 1))
            return json.dumps({"decisions": {f: confirm for f in fields}})
        return json.dumps({"missing_info": []})
    return llm


SEVERE = {"lethargic_or_unconscious": {"value": True, "quote": "lethargic"}}


def test_needed_fields_come_from_the_classification_rules():
    fields = needed_fields(CARDS.values(), "Cholera", {})
    assert "lethargic_or_unconscious" in fields and "sunken_eyes" in fields
    assert needed_fields(CARDS.values(), "Cholera", {"sunken_eyes": True}).count("sunken_eyes") == 0
    assert needed_fields(CARDS.values(), "Dengue", {}) == []  # no classification rules


def test_verified_severe_sign_settles_plan_c_for_management(monkeypatch):
    monkeypatch.setattr(retrieve, "retrieve_category", _retrieve)
    out = recommend_management(PATIENT, mode="improved", cards=CARDS, diseases=["Cholera"],
                               llm_generate=_llm(SEVERE))
    d = out["management_diagnostics"]
    assert d["management_fact_extraction"]["accepted"] == ["lethargic_or_unconscious"]
    assert "CHO-004" in d["prompt_card_ids"]
    assert not {"CHO-007", "CHO-009"} & set(d["prompt_card_ids"])
    assert not {"CHO-004", "CHO-005"} & set(d["blocked_card_ids"])


def test_quote_not_in_patient_text_is_rejected_and_plan_stays_withheld(monkeypatch):
    monkeypatch.setattr(retrieve, "retrieve_category", _retrieve)
    fake = {"lethargic_or_unconscious": {"value": True, "quote": "unconscious and pulseless"}}
    out = recommend_management(PATIENT, mode="improved", cards=CARDS, diseases=["Cholera"],
                               llm_generate=_llm(fake))
    d = out["management_diagnostics"]
    assert d["management_fact_extraction"]["accepted"] == []
    assert {"CHO-004", "CHO-007", "CHO-009"} <= set(d["blocked_card_ids"])
    assert not {"CHO-004", "CHO-007", "CHO-009"} & set(d["prompt_card_ids"])


def test_unconfirmed_polarity_is_not_accepted(monkeypatch):
    monkeypatch.setattr(retrieve, "retrieve_category", _retrieve)
    out = recommend_management(PATIENT, mode="improved", cards=CARDS, diseases=["Cholera"],
                               llm_generate=_llm(SEVERE, confirm="uncertain"))
    assert out["management_diagnostics"]["management_fact_extraction"]["accepted"] == []
    assert out["management_diagnostics"]["blocked_card_ids"]


def test_extraction_failure_keeps_everything_unknown_without_crashing(monkeypatch):
    monkeypatch.setattr(retrieve, "retrieve_category", _retrieve)
    out = recommend_management(PATIENT, mode="improved", cards=CARDS, diseases=["Cholera"],
                               llm_generate=_llm(RuntimeError("429")))
    assert "error" in out["management_diagnostics"]["management_fact_extraction"]
    assert out["management_diagnostics"]["blocked_card_ids"]
    assert any("Not shown until" in m for m in out["missing_info"])


def test_supplied_facts_are_not_overwritten(monkeypatch):
    monkeypatch.setattr(retrieve, "retrieve_category", _retrieve)
    facts, info = extract_management_facts(
        PATIENT, CARDS.values(), "Cholera", {"lethargic_or_unconscious": False},
        _llm(SEVERE))
    assert "lethargic_or_unconscious" not in info["requested_fields"]
    assert "lethargic_or_unconscious" not in facts


def test_baseline_makes_no_extra_call(monkeypatch):
    monkeypatch.setattr(retrieve, "retrieve_category", _retrieve)
    calls = []
    out = recommend_management(PATIENT, mode="baseline", cards=CARDS, diseases=["Cholera"],
                               llm_generate=_llm(SEVERE, calls=calls))
    assert not any(c.startswith("Extract patient facts") for c in calls)
    assert "management_fact_extraction" not in out["management_diagnostics"]


def test_disease_without_classification_rules_makes_no_extra_call(monkeypatch):
    monkeypatch.setattr(retrieve, "retrieve_category",
                        lambda p, c, top_k=2, **kw: [{"card_id": "DEN-MED-001"}] if c == "Medication" else [])
    calls = []
    recommend_management({"symptoms": "fever"}, mode="improved", cards=CARDS, diseases=["Dengue"],
                         llm_generate=_llm({}, calls=calls))
    assert not any(c.startswith("Extract patient facts") for c in calls)


def test_malformed_verification_response_fails_closed():
    def model(prompt):
        if prompt.startswith("Extract patient facts"):
            return json.dumps({"facts": SEVERE})
        return "[]"
    facts, info = extract_management_facts(PATIENT, CARDS.values(), "Cholera", {}, model)
    assert facts == {}
    assert "verification unavailable" in info["error"]


def test_unknown_and_negative_facts_do_not_select_severe_branch(monkeypatch):
    monkeypatch.setattr(retrieve, "retrieve_category", _retrieve)
    patient = {"clinical_findings": "No lethargy; pulse is normal."}
    proposals = {
        "lethargic_or_unconscious": {"value": False, "quote": "No lethargy"},
        "absent_or_weak_pulse": {"value": False, "quote": "pulse is normal"},
    }
    out = recommend_management(patient, cards=CARDS, diseases=["Cholera"],
                               llm_generate=_llm(proposals))
    assert "CHO-004" not in out["management_diagnostics"]["prompt_card_ids"]


def test_verified_facts_and_working_protocol_reach_generation(monkeypatch):
    monkeypatch.setattr(retrieve, "retrieve_category", _retrieve)
    helper = _llm(SEVERE)
    prompts = []
    def model(prompt):
        if (prompt.startswith("Extract patient facts")
                or "semantic fact verification" in prompt):
            return helper(prompt)
        prompts.append(prompt)
        return json.dumps({"medication": [{
            "text": "Source-guided pathway for the supported classification.",
            "reason": "The quoted patient finding meets the source rule.",
            "evidence": [{"card_id": "CHO-004", "pdf_page": 7}],
        }], "missing_info": []})
    out = recommend_management(PATIENT, cards=CARDS, diseases=["Cholera"],
                               llm_generate=model)
    assert out["medication"]
    assert len(prompts) == 1
    assert 'SELECTED WORKING PROTOCOL (probable, not confirmed): "Cholera"' in prompts[0]
    assert '"lethargic_or_unconscious": true' in prompts[0]
    assert '"dehydration_classification": "severe_dehydration_or_shock"' in prompts[0]


def test_unseen_protocol_and_field_use_source_rules_without_special_cases(monkeypatch):
    from schema import Card
    replacements = {
        "Cholera": "Synthetic protocol", "CHO-002": "SYN-102",
        "CHO-004": "SYN-204", "lethargic_or_unconscious": "observed_marker",
        "dehydration_classification": "source_class",
        "severe_dehydration_or_shock": "critical_branch",
    }
    cards = {}
    for cid in ("CHO-002", "CHO-004"):
        raw = json.dumps(CARDS[cid].model_dump())
        for old, new in replacements.items():
            raw = raw.replace(old, new)
        card = Card.model_validate_json(raw)
        cards[card.card_id] = card
    monkeypatch.setattr(retrieve, "retrieve_category", lambda p, c, **kw:
                        [{"card_id": "SYN-204"}] if c == "Medication" else [])
    out = recommend_management(
        {"clinical_findings": "marker explicitly observed"}, cards=cards,
        diseases=["Synthetic protocol"], llm_generate=_llm({
            "observed_marker": {"value": True, "quote": "marker explicitly observed"}}))
    d = out["management_diagnostics"]
    assert d["management_fact_extraction"]["accepted"] == ["observed_marker"]
    assert d["derived_classification"]["fields"]["source_class"]["value"] == "critical_branch"
    assert "SYN-204" in d["prompt_card_ids"]
