"""Classification gating on the real 108 cards (retrieval and LLM are mocked)."""
import json
from pathlib import Path

import retrieve
from build_index import load_cards
from derived_fields import blocked_cards, derive_classifications
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


def test_unknown_class_blocks_plan_cards_but_keeps_supportive_cards(monkeypatch):
    out = _run(monkeypatch, {"sunken_eyes": True}, cite="CHO-007")
    d = out["management_diagnostics"]
    assert set(d["blocked_card_ids"]) == {"CHO-004", "CHO-007", "CHO-009"}
    assert not set(d["blocked_card_ids"]) & set(d["prompt_card_ids"])
    # supportive cards that do not depend on the class remain available
    assert "CHO-006" in d["prompt_card_ids"] and "CHO-011" in d["prompt_card_ids"]
    # a definitive plan citing a blocked card cannot get through
    assert out["medication"] == []
    assert any("Not shown until" in m for m in out["missing_info"])
    assert any("lethargic_or_unconscious" in m for m in out["missing_info"])


def test_clinician_supplied_class_is_used_not_overwritten(monkeypatch):
    out = _run(monkeypatch, {"dehydration_classification": "some_dehydration"}, cite="CHO-007")
    d = out["management_diagnostics"]
    assert d["blocked_card_ids"] == []
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
    assert trace["dependencies"] == {}
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


def test_rule_based_missing_info_survives_llm_missing_info(monkeypatch):
    monkeypatch.setattr(retrieve, "retrieve_category", _fake_retrieve)
    out = recommend_management({"_structured_facts": {}, "age": 30}, mode="improved",
                               cards=CARDS, diseases=["Cholera"],
                               llm_generate=lambda p: json.dumps({"missing_info": ["model note"]}))
    assert any("Not shown until" in m for m in out["missing_info"])
    assert "model note" in out["missing_info"]
