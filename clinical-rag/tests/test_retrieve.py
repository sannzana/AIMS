from __future__ import annotations

import json
from pathlib import Path

from build_index import load_cards
from applicability import card_rule_applicability
from retrieve import retrieve


class Embedder:
    def encode(self, texts):
        return [[1.0] for _ in texts]


class Collection:
    def __init__(self, rows):
        self.rows = rows

    def query(self, query_embeddings, n_results, include, where=None):
        rows = self.rows
        if where:
            rows = [
                row for row in rows
                if row["metadata"].get("clinical_category")
                == where["clinical_category"]
            ]
        rows = rows[:n_results]
        return {
            "metadatas": [[row["metadata"] for row in rows]],
            "distances": [[row["distance"] for row in rows]],
        }


class Client:
    def __init__(self, rows):
        self.collection = Collection(rows)

    def get_collection(self, name):
        return self.collection


def metadata(card_id, category, *, disease=None, pages=(1,), score=0.8):
    return {
        "card_id": card_id,
        "clinical_category": category,
        "disease_protocol": disease,
        "source_title": "Protocol",
        "source_version": "",
        "source_pdf_pages": json.dumps(list(pages)),
    }


def test_category_coverage_and_deduplication():
    rows = [
        {"metadata": metadata("DX", "Diagnosis"), "distance": 0.2},
        {"metadata": metadata("INV", "Investigation"), "distance": 0.3},
        {"metadata": metadata("MED", "Medication"), "distance": 0.4},
        {"metadata": metadata("ADV", "Advice"), "distance": 0.5},
        {"metadata": metadata("DX", "Diagnosis"), "distance": 0.1},
    ]
    result = retrieve(
        {"symptoms": "fever"},
        top_k=5,
        embedder=Embedder(),
        client=Client(rows),
        cards={},
    )
    assert len(result) == len({item["card_id"] for item in result})
    assert {item["metadata"]["clinical_category"] for item in result} == {
        "Diagnosis",
        "Investigation",
        "Medication",
        "Advice",
    }


def test_relevant_medication_and_safety_evidence_are_selected():
    cards = {
        card.card_id: card
        for card in load_cards(Path(__file__).parents[1] / "data" / "cards.jsonl")
    }
    rows = [
        {
            "metadata": metadata("CHI-026", "Diagnosis", pages=(17, 18), score=0.9),
            "distance": 0.1,
        },
        {
            "metadata": metadata("DEN-MED-001", "Medication", pages=(37,), score=0.85),
            "distance": 0.15,
        },
    ]
    result = retrieve(
        {"symptoms": "fever and severe joint pain"},
        top_k=5,
        embedder=Embedder(),
        client=Client(rows),
        cards=cards,
    )
    selected = {item["card_id"]: item for item in result}
    assert "CHI-026" in selected
    assert "DEN-MED-001" in selected
    assert selected["CHI-026"]["safety_critical"] is True
    assert selected["DEN-MED-001"]["metadata"]["clinical_category"] == "Medication"


def test_chi026_cross_disease_differential_regression():
    cards = {
        card.card_id: card
        for card in load_cards(Path(__file__).parents[1] / "data" / "cards.jsonl")
    }
    row = {
        "metadata": metadata("CHI-026", "Diagnosis", pages=(17, 18), score=0.8),
        "distance": 0.2,
    }
    result = retrieve(
        {"symptoms": "fever arthralgia rash"},
        top_k=5,
        embedder=Embedder(),
        client=Client([row]),
        cards=cards,
    )
    assert result[0]["card_id"] == "CHI-026"
    assert "safety-critical" in result[0]["selection_reason"]


def test_irrelevant_low_score_cards_are_rejected_and_missing_data_is_not_false():
    rows = [
        {
            "metadata": metadata("CHI-DX", "Diagnosis", disease="Chikungunya", score=0.1),
            "distance": 1.1,
        },
        {
            "metadata": metadata("IRRELEVANT", "Medication", disease="Hypertension", score=0.9),
            "distance": 0.1,
        },
    ]
    result = retrieve(
        {},
        top_k=5,
        embedder=Embedder(),
        client=Client(rows),
        cards={},
    )
    assert "IRRELEVANT" not in {item["card_id"] for item in result}


def test_two_stage_disease_filter_keeps_chikungunya_and_dengue_precaution():
    rows = [
        {"metadata": metadata("CHI-026", "Diagnosis", disease="Chikungunya", pages=(17, 18)), "distance": 0.2},
        {"metadata": metadata("DEN-MED-001", "Medication", disease="Dengue", pages=(37,)), "distance": 0.3},
        {"metadata": metadata("CHI-018", "Medication", disease="Chikungunya", pages=(18,)), "distance": 0.4},
        {"metadata": metadata("CHI-015", "Advice", disease="Chikungunya", pages=(19,)), "distance": 0.5},
        {"metadata": metadata("UNRELATED", "Medication", disease="Hypertension", pages=(1,)), "distance": 0.1},
    ]
    cards = {
        card.card_id: card
        for card in load_cards(Path(__file__).parents[1] / "data" / "cards.jsonl")
    }
    result = retrieve(
        {"symptoms": "fever severe polyarthralgia rash"},
        top_k=5,
        embedder=Embedder(),
        client=Client(rows),
        cards=cards,
    )
    ids = {item["card_id"] for item in result}
    assert "CHI-026" in ids
    assert "CHI-018" in ids
    assert "CHI-015" in ids
    assert "UNRELATED" not in ids


def test_supported_card_rules_keep_missing_prerequisites_unknown():
    cards = {
        card.card_id: card
        for card in load_cards(Path(__file__).parents[1] / "data" / "cards.jsonl")
    }
    result = card_rule_applicability(
        cards["CHI-014"],
        {"patient_has_fever": True},
    )
    assert result is not None
    assert result["status"] == "undetermined"
    assert "preexisting_liver_disease" in result["missing_inputs"]


def test_small_budget_reports_incomplete_category_coverage():
    rows = [
        {"metadata": metadata("DX", "Diagnosis", disease="Chikungunya"), "distance": 0.2},
        {"metadata": metadata("INV", "Investigation", disease="Chikungunya"), "distance": 0.3},
        {"metadata": metadata("MED", "Medication", disease="Chikungunya"), "distance": 0.4},
        {"metadata": metadata("ADV", "Advice", disease="Chikungunya"), "distance": 0.5},
    ]
    cards = {
        card.card_id: card
        for card in load_cards(Path(__file__).parents[1] / "data" / "cards.jsonl")
    }
    result = retrieve(
        {"symptoms": "fever"},
        top_k=2,
        embedder=Embedder(),
        client=Client(rows),
        cards=cards,
    )
    assert result[0]["coverage"]["complete"] is False
    assert result[0]["coverage"]["incomplete_reason"] == "selection budget"


def test_multiple_protocols_are_balanced_without_disease_specific_rules():
    rows = [
        {"metadata": metadata("DEN-DX", "Diagnosis", disease="Dengue"), "distance": 0.2},
        {"metadata": metadata("CHO-DX", "Diagnosis", disease="Cholera"), "distance": 0.3},
        {"metadata": metadata("HTN-DX", "Diagnosis", disease="Hypertension"), "distance": 0.4},
        {"metadata": metadata("DEN-MED", "Medication", disease="Dengue"), "distance": 0.2},
        {"metadata": metadata("CHO-MED", "Medication", disease="Cholera"), "distance": 0.3},
        {"metadata": metadata("HTN-MED", "Medication", disease="Hypertension"), "distance": 0.4},
    ]
    result = retrieve(
        {"symptoms": "fever and dehydration"},
        top_k=6,
        embedder=Embedder(),
        client=Client(rows),
        cards={},
    )
    assert {
        item["metadata"]["disease_protocol"] for item in result
    } == {"Dengue", "Cholera", "Hypertension"}
    assert len({item["card_id"] for item in result}) == len(result)
