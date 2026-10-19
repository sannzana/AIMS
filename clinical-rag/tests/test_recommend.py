import json
from pathlib import Path

import pytest

from llm import LLMError, generate
from recommend import RecommendationError, recommend
from schema import Card


def make_card(card_id="DEV-001", *, required=None, medication=None):
    return Card(
        card_id=card_id,
        disease="synthetic-condition",
        category="diagnosis",
        conditions={"required_inputs": required or []},
        action="DEV ONLY placeholder action; not clinical advice.",
        medication=medication,
        source={"title": "Synthetic DEV protocol", "pdf_pages": [3]},
        evidence=[{"pdf_page": 3, "passage": "DEV ONLY placeholder evidence."}],
        verified=False,
    )


def retrieved(card_id="DEV-001"):
    return [{"card_id": card_id, "similarity_score": 0.9}]


def valid_json(card_id="DEV-001", **extra):
    payload = {
        "probable_diagnosis": [
            {
                "text": "Probable synthetic condition",
                "reason": "Supported by the supplied DEV placeholder evidence.",
                "evidence": [{"card_id": card_id, "pdf_page": 3}],
            }
        ],
        "investigations": [],
        "medication": [],
        "advice": [],
        "missing_info": ["Clinical confirmation is missing."],
    }
    payload.update(extra)
    return json.dumps(payload)


def test_baseline_retrieval_and_generation_and_same_retrieval_for_improved():
    card = make_card()
    calls = []

    def retrieve_fn(patient, top_k, persist_dir):
        calls.append((patient, top_k, persist_dir))
        return retrieved()

    def mock_llm(prompt):
        assert "symptoms" in prompt
        assert "medicines" in prompt
        assert "allergies" in prompt
        return valid_json()

    baseline = recommend(
        {"symptoms": ["placeholder"], "medicines": ["x"], "allergies": ["y"]},
        retrieve_fn=retrieve_fn,
        llm_generate=mock_llm,
        cards={card.card_id: card},
        persist_dir=Path("data/dev_index"),
    )
    improved = recommend(
        {"symptoms": ["placeholder"]},
        mode="improved",
        retrieve_fn=retrieve_fn,
        llm_generate=mock_llm,
        cards={card.card_id: card},
        persist_dir=Path("data/dev_index"),
    )
    assert baseline["mode"] == "baseline"
    assert improved["mode"] == "improved"
    assert [call[1] for call in calls] == [5, 5]
    assert baseline["dev_only"] is True


def test_improved_filters_not_applicable_and_keeps_unknown_annotated():
    excluded = Card(
        card_id="DEV-FALSE",
        disease="synthetic",
        category="diagnosis",
        conditions={"all": [{"field": "age", "op": ">=", "value": 18}]},
        source={"title": "DEV", "pdf_pages": [3]},
        evidence=[{"pdf_page": 3, "passage": "DEV"}],
    )
    unknown = make_card("DEV-UNKNOWN", required=["history"])
    cards = {excluded.card_id: excluded, unknown.card_id: unknown}

    def retrieve_fn(patient, top_k, persist_dir):
        return retrieved("DEV-FALSE") + retrieved("DEV-UNKNOWN")

    captured = {}

    def mock_llm(prompt):
        captured["prompt"] = prompt
        return valid_json("DEV-UNKNOWN")

    result = recommend(
        {"age": 10},
        mode="improved",
        retrieve_fn=retrieve_fn,
        llm_generate=mock_llm,
        cards=cards,
    )
    assert "DEV-FALSE" not in captured["prompt"]
    assert "DEV-UNKNOWN" in captured["prompt"]
    assert "history" in captured["prompt"]
    assert result["mode"] == "improved"


def test_prompt_contains_full_conditions_and_exclusions():
    card = Card(
        card_id="DEV-CONDITIONS",
        disease="synthetic",
        category="advice",
        conditions={
            "all": [{"field": "age", "op": ">=", "value": 18}],
            "any": [{"field": "symptoms", "op": "has", "value": "placeholder"}],
            "exceptions": [{"field": "allergies", "op": "has", "value": "placeholder"}],
            "required_inputs": ["history"],
        },
        exclusions_or_missing_info="DEV ONLY missing history.",
        source={"title": "DEV", "pdf_pages": [3]},
        evidence=[{"pdf_page": 3, "passage": "DEV"}],
    )
    captured = {}

    def mock_llm(prompt):
        captured["prompt"] = prompt
        return valid_json("DEV-CONDITIONS")

    recommend(
        {},
        retrieve_fn=lambda **kwargs: retrieved("DEV-CONDITIONS"),
        llm_generate=mock_llm,
        cards={card.card_id: card},
    )
    assert '"conditions"' in captured["prompt"]
    assert '"required_inputs": ["history"]' in captured["prompt"]
    assert "DEV ONLY missing history." in captured["prompt"]


def test_baseline_and_improved_use_same_generation_instructions():
    card = make_card()
    prompts = []

    def mock_llm(prompt):
        prompts.append(prompt)
        return valid_json()

    retrieve_fn = lambda **kwargs: retrieved()
    recommend(
        {},
        retrieve_fn=retrieve_fn,
        llm_generate=mock_llm,
        cards={card.card_id: card},
        persist_dir=Path("data/dev_index"),
    )
    recommend(
        {},
        mode="improved",
        retrieve_fn=retrieve_fn,
        llm_generate=mock_llm,
        cards={card.card_id: card},
        persist_dir=Path("data/dev_index"),
    )
    for prompt in prompts:
        assert "Mode:" not in prompt
    assert prompts[0].split("Patient context:", 1)[0] == prompts[1].split(
        "Patient context:", 1
    )[0]


def test_recommendation_without_evidence_is_rejected():
    card = make_card()
    raw = json.loads(valid_json())
    raw["probable_diagnosis"][0]["evidence"] = []
    with pytest.raises(RecommendationError, match="at least one evidence"):
        recommend(
            {},
            retrieve_fn=lambda **kwargs: retrieved(),
            llm_generate=lambda prompt: json.dumps(raw),
            cards={card.card_id: card},
        )


@pytest.mark.parametrize(
    "raw",
    [
        '{"probable_diagnosis":[],"investigations":[],"medication":[],"advice":[],"missing_info":[]}',
        '{"probable_diagnosis":[{"text":"x","reason":"r","evidence":[{"card_id":"NOPE","pdf_page":3}]}],"investigations":[],"medication":[],"advice":[],"missing_info":[]}',
        '{"probable_diagnosis":[{"text":"x","reason":"r","evidence":[{"card_id":"DEV-001","pdf_page":99}]}],"investigations":[],"medication":[],"advice":[],"missing_info":[]}',
    ],
)
def test_invalid_or_unsupported_model_output(raw):
    card = make_card()
    with pytest.raises(RecommendationError):
        recommend(
            {},
            retrieve_fn=lambda **kwargs: retrieved(),
            llm_generate=lambda prompt: raw,
            cards={card.card_id: card},
        )


def test_invalid_model_json():
    card = make_card()
    with pytest.raises(RecommendationError, match="valid recommendation JSON"):
        recommend(
            {},
            retrieve_fn=lambda **kwargs: retrieved(),
            llm_generate=lambda prompt: "not json",
            cards={card.card_id: card},
        )


def test_missing_api_key():
    import os

    old = {key: os.environ.pop(key, None) for key in ("LLM_PROVIDER", "LLM_API_KEY")}
    os.environ["LLM_PROVIDER"] = "api"
    try:
        with pytest.raises(LLMError, match="LLM_API_KEY"):
            generate("ignored")
    finally:
        for key, value in old.items():
            if value is not None:
                os.environ[key] = value


def test_missing_clinical_information_is_preserved():
    card = make_card(required=["history"])
    result = recommend(
        {},
        retrieve_fn=lambda **kwargs: retrieved(),
        llm_generate=lambda prompt: valid_json(),
        cards={card.card_id: card},
    )
    assert result["missing_info"] == ["Clinical confirmation is missing."]


def test_missing_medication_field_cannot_be_invented():
    card = make_card(
        medication={"name": "DEV placeholder", "dose": None},
    )
    raw = json.loads(valid_json())
    raw["probable_diagnosis"] = []
    raw["medication"] = [
        {
            "text": "placeholder",
            "reason": "r",
            "dose": "invented dose",
            "evidence": [{"card_id": card.card_id, "pdf_page": 3}],
        }
    ]
    with pytest.raises(RecommendationError, match="dose"):
        recommend(
            {},
            retrieve_fn=lambda **kwargs: retrieved(),
            llm_generate=lambda prompt: json.dumps(raw),
            cards={card.card_id: card},
        )


def test_empty_retrieval_returns_safe_result_without_llm_call():
    called = False

    def fail(prompt):
        nonlocal called
        called = True
        raise AssertionError("LLM should not run with empty retrieval")

    result = recommend(
        {},
        retrieve_fn=lambda **kwargs: [],
        llm_generate=fail,
        cards={},
    )
    assert result["missing_info"]
    assert called is False
