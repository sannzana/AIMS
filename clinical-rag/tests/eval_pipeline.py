"""Phase 0 diagnostic retrieval baseline. Run: python -m pytest -q tests/eval_pipeline.py
Requires a locally built Chroma index and sentence-transformers; uses NO LLM.
The metrics are retrieval-only proxies, not clinical diagnostic accuracy.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from build_index import load_cards
from retrieve import retrieve_category

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "cases" / "cases.json"
INDEX = ROOT / "data" / "index"
CATEGORIES = ("Investigation", "Medication", "Advice")


def _fixtures():
    return json.loads(CASES.read_text(encoding="utf-8"))


def _cards():
    return {card.card_id: card for card in load_cards(ROOT / "data" / "cards.jsonl")}


def _index_ready():
    if not INDEX.is_dir():
        return False
    try:
        import chromadb
        collection = chromadb.PersistentClient(path=str(INDEX)).get_collection("clinical_cards")
        return collection.count() > 0
    except (ImportError, OSError, ValueError, Exception):
        return False


def evaluate(cases, cards, search=retrieve_category):
    """Search injectable for deterministic offline unit tests."""
    top1 = top2 = scoped_items = contaminated = covered = requested = 0
    reciprocal_rank = 0.0
    recall5 = recall10 = eligible_cases = 0
    detail = []
    for case in cases:
        patient = case["patient"]
        hits = search(patient, "Diagnosis", top_k=10, cards=cards, persist_dir=INDEX)
        # Simple card-order grouping measures CURRENT retriever, not an invented
        # probabilistic disease classifier.
        ranked = []
        for hit in hits:
            card = cards.get(hit["card_id"])
            if card is None:
                continue
            disease = card.disease_protocol
            if disease and disease not in ranked:
                ranked.append(disease)
        expected = case["expected_disease"]
        if expected != "none":
            eligible_cases += 1
            if expected in ranked:
                reciprocal_rank += 1 / (ranked.index(expected) + 1)
            recall5 += expected in ranked[:5]
            recall10 += expected in ranked[:10]
        top1 += bool(ranked and ranked[0] == expected)
        top2 += expected in ranked[:2]
        # Existing retrieval has no abstain gate; 'none' cases are expected
        # to fail, making that limitation visible rather than hiding it.
        primary = ranked[0] if ranked else None
        present = set()
        for category in CATEGORIES:
            requested += 1
            if not primary:
                continue
            section_hits = search(patient, category, diseases=[primary],
                                  top_k=2, cards=cards, persist_dir=INDEX)
            if section_hits:
                present.add(category)
            for hit in section_hits:
                card = cards.get(hit["card_id"])
                if card is None:
                    continue
                scoped_items += 1
                contaminated += card.disease_protocol != primary
        covered += len(present)
        detail.append({"id": case["id"], "expected": expected,
                       "top1": primary, "top2": ranked[:2],
                       "management_categories": sorted(present)})
    n = len(cases)
    return {
        "cases": n,
        "disease_recall_at_5": recall5 / eligible_cases if eligible_cases else None,
        "disease_recall_at_10": recall10 / eligible_cases if eligible_cases else None,
        "disease_mrr_at_10": reciprocal_rank / eligible_cases if eligible_cases else None,
        "top1_disease_acc": top1 / n if n else 0,
        "top2_disease_acc": top2 / n if n else 0,
        "cross_disease_contamination_rate": contaminated / scoped_items if scoped_items else None,
        "category_coverage": covered / requested if requested else 0,
        "details": detail,
        "limitations": "Retrieval-only proxy; not LLM output accuracy or clinical validity.",
    }


def test_case_dataset_shape():
    cases = _fixtures()
    assert len(cases) >= 30
    assert len({case["id"] for case in cases}) == len(cases)
    counts = Counter(case["expected_disease"] for case in cases)
    assert all(counts[disease] >= 5 for disease in ("Dengue", "Chikungunya", "Cholera", "Hypertension"))
    assert counts["none"] >= 3
    assert sum(case["id"].startswith("confusable-") for case in cases) >= 4
    for case in cases:
        assert all(key in case for key in ("id", "patient", "expected_disease",
                                          "must_include_categories", "forbidden_diseases_in_management"))


def test_evaluator_detects_cross_disease_contamination():
    from types import SimpleNamespace
    cards = {
        "A": SimpleNamespace(disease_protocol="Chikungunya"),
        "B": SimpleNamespace(disease_protocol="Dengue"),
    }
    def fake_search(patient, category, **kwargs):
        return [{"card_id": "A"}] if category == "Diagnosis" else [{"card_id": "B"}]
    cases = [{"id": "case", "patient": {}, "expected_disease": "Chikungunya"}]
    result = evaluate(cases, cards, search=fake_search)
    assert result["top1_disease_acc"] == 1
    assert result["cross_disease_contamination_rate"] == 1


def test_live_retrieval_baseline():
    if not _index_ready():
        pytest.skip("Local Chroma clinical_cards index unavailable; baseline not measured.")
    pytest.importorskip("sentence_transformers")
    result = evaluate(_fixtures(), _cards())
    print("\nPHASE_0_BASELINE=" + json.dumps(result, sort_keys=True))
    assert len(result["details"]) >= 30
