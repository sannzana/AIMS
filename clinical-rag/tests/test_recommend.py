import json
from pathlib import Path

import pytest

import llm
from llm import LLMError, generate
from recommend import (
    MalformedRecommendationError,
    RecommendationError,
    RecommendationSchemaError,
    UnsupportedEvidenceError,
    recommend,
)
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
        assert "clinical_findings" in prompt
        assert "measurements" in prompt
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
    assert baseline["retrieval_summary"] == improved["retrieval_summary"]
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


def test_prompt_specifies_complete_output_schema_and_retrieved_references():
    card = make_card()
    captured = {}

    def mock_llm(prompt):
        captured["prompt"] = prompt
        return valid_json()

    recommend(
        {"clinical_findings": "alert", "measurements": "BP 120/80"},
        retrieve_fn=lambda **kwargs: retrieved(),
        llm_generate=mock_llm,
        cards={card.card_id: card},
    )
    prompt = captured["prompt"]
    assert "probable_diagnosis" in prompt
    assert '"text": "string"' in prompt
    assert '"reason": "string"' in prompt
    assert '"evidence"' in prompt
    assert '"missing_info": ["string"]' in prompt
    assert "DEV-001" in prompt
    assert "clinical_findings" in prompt
    assert "measurements" in prompt


def test_medication_schema_requires_text_reason_and_evidence():
    card = make_card()
    raw = json.loads(valid_json())
    raw["medication"] = [
        {
            "reason": "Supported by the supplied evidence.",
            "evidence": [{"card_id": card.card_id, "pdf_page": 3}],
        }
    ]
    result = recommend(
        {},
        retrieve_fn=lambda **kwargs: retrieved(),
        llm_generate=lambda prompt: json.dumps(raw),
        cards={card.card_id: card},
    )
    assert result["medication"] == []
    assert "medication.0.text (missing)" in result["rejected_recommendations"]


def test_valid_medication_output_has_traceable_source_passage():
    card = make_card(
        medication={
            "name": "DEV medicine",
            "dose": "one unit",
            "route": "oral",
            "frequency": "once",
            "duration": "one day",
            "instructions": "DEV instruction",
        }
    )
    raw = json.loads(valid_json())
    raw["medication"] = [
        {
            "text": "Use DEV medicine.",
            "reason": "The supplied card supports this.",
            "evidence": [{"card_id": card.card_id, "pdf_page": 3}],
            "name": "DEV medicine",
            "dose": "one unit",
            "route": "oral",
            "frequency": "once",
            "duration": "one day",
            "instructions": "DEV instruction",
        }
    ]
    result = recommend(
        {},
        retrieve_fn=lambda **kwargs: retrieved(),
        llm_generate=lambda prompt: json.dumps(raw),
        cards={card.card_id: card},
    )
    assert result["medication"][0]["evidence_details"][0]["passages"] == [
        "DEV ONLY placeholder evidence."
    ]


def test_nested_source_medication_values_are_supported_exactly():
    card = make_card()
    card.medication_details = {
        "paracetamol": {
            "adult_dose": "Up to 1000 mg four times daily",
            "route": "Oral",
        }
    }
    raw = json.loads(valid_json())
    raw["medication"] = [
        {
            "text": "Use paracetamol.",
            "reason": "The source provides this medication guidance.",
            "evidence": [{"card_id": card.card_id, "pdf_page": 3}],
            "name": "paracetamol",
            "dose": "Up to 1000 mg four times daily",
        }
    ]
    result = recommend(
        {},
        retrieve_fn=lambda **kwargs: retrieved(),
        llm_generate=lambda prompt: json.dumps(raw),
        cards={card.card_id: card},
    )
    assert result["medication"][0]["name"] == "paracetamol"


def test_invalid_medication_does_not_hide_valid_diagnosis():
    card = make_card()
    raw = json.loads(valid_json())
    raw["medication"] = [
        {
            "text": "Use unsupported medicine.",
            "reason": "Not supported by the card.",
            "evidence": [{"card_id": card.card_id, "pdf_page": 3}],
            "name": "unsupported medicine",
        }
    ]
    result = recommend(
        {},
        retrieve_fn=lambda **kwargs: retrieved(),
        llm_generate=lambda prompt: json.dumps(raw),
        cards={card.card_id: card},
    )
    assert result["probable_diagnosis"][0]["text"] == "Probable synthetic condition"
    assert result["medication"] == []
    assert result["incomplete_output"] is True
    assert any(
        "medication.0: unsupported medication field name" in item
        for item in result["rejected_recommendations"]
    )


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
    result = recommend(
        {},
        retrieve_fn=lambda **kwargs: retrieved(),
        llm_generate=lambda prompt: json.dumps(raw),
        cards={card.card_id: card},
    )
    assert result["probable_diagnosis"] == []
    assert "probable_diagnosis.0: missing evidence" in result["rejected_recommendations"]


def test_missing_required_item_field_is_schema_error():
    card = make_card()
    raw = json.loads(valid_json())
    del raw["probable_diagnosis"][0]["reason"]
    result = recommend(
        {},
        retrieve_fn=lambda **kwargs: retrieved(),
        llm_generate=lambda prompt: json.dumps(raw),
        cards={card.card_id: card},
    )
    assert result["probable_diagnosis"] == []
    assert result["incomplete_output"] is True
    assert any(
        "probable_diagnosis.0.reason (missing)" == item
        for item in result["rejected_recommendations"]
    )


def test_schema_error_reports_only_paths_and_error_types():
    card = make_card()
    raw = json.loads(valid_json())
    raw["investigations"] = {"text": "not a list"}
    raw["missing_info"] = [None]
    result = recommend(
        {},
        retrieve_fn=lambda **kwargs: retrieved(),
        llm_generate=lambda prompt: json.dumps(raw),
        cards={card.card_id: card},
    )
    assert "investigations (list_type)" in result["rejected_recommendations"]
    assert "missing_info (list/string_type)" in result["rejected_recommendations"]
    assert "not a list" not in json.dumps(result)


def test_prompt_includes_unknown_fields_but_excludes_images():
    card = make_card()
    card.future_clinical_field = "DEV future field"
    card.image = "binary-or-url"
    captured = {}

    def mock_llm(prompt):
        captured["prompt"] = prompt
        return valid_json()

    recommend(
        {},
        retrieve_fn=lambda **kwargs: retrieved(),
        llm_generate=mock_llm,
        cards={card.card_id: card},
    )
    assert "DEV future field" in captured["prompt"]
    assert "binary-or-url" not in captured["prompt"]


def test_card_without_structured_rules_is_retained_in_improved_mode():
    card = make_card()
    captured = {}

    def mock_llm(prompt):
        captured["prompt"] = prompt
        return valid_json()

    recommend(
        {},
        mode="improved",
        retrieve_fn=lambda **kwargs: retrieved(),
        llm_generate=mock_llm,
        cards={card.card_id: card},
    )
    assert "DEV-001" in captured["prompt"]


def test_structured_rules_use_applicability_in_improved_mode():
    card = make_card()
    card.structured_rules = {
        "all": [{"field": "age", "op": ">=", "value": 18}],
        "any": [],
        "exceptions": [],
        "required_inputs": [],
    }
    called = False

    def fail_if_model_called(prompt):
        nonlocal called
        called = True
        return valid_json()

    result = recommend(
        {"age": 10},
        mode="improved",
        retrieve_fn=lambda **kwargs: retrieved(),
        llm_generate=fail_if_model_called,
        cards={card.card_id: card},
    )
    assert called is False
    assert result["missing_info"] == ["All retrieved cards were not applicable."]


def test_groq_request_uses_json_and_low_reasoning_configuration(monkeypatch):
    monkeypatch.setattr(llm, "_load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "api")
    monkeypatch.setenv("LLM_API_KEY", "offline-test-key")
    monkeypatch.setenv("LLM_API_URL", "https://api.groq.com/openai/v1/chat/completions")
    captured = {}

    def mock_request(url, payload, headers):
        captured["payload"] = payload
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": '{"ok": true}'}}
            ]
        }

    monkeypatch.setattr(llm, "_json_request", mock_request)
    assert generate("Return JSON") == '{"ok": true}'
    assert captured["payload"]["reasoning_effort"] == "low"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["max_completion_tokens"] == 3000
    assert "max_tokens" not in captured["payload"]


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
    with pytest.raises((RecommendationSchemaError, UnsupportedEvidenceError)):
        recommend(
            {},
            retrieve_fn=lambda **kwargs: retrieved(),
            llm_generate=lambda prompt: raw,
            cards={card.card_id: card},
        )


def test_invalid_model_json():
    card = make_card()
    with pytest.raises(MalformedRecommendationError, match="valid JSON"):
        recommend(
            {},
            retrieve_fn=lambda **kwargs: retrieved(),
            llm_generate=lambda prompt: "not json",
            cards={card.card_id: card},
        )


def test_missing_api_key(monkeypatch):
    import os

    monkeypatch.setattr("llm._load_dotenv", lambda: None)
    old = {key: os.environ.pop(key, None) for key in ("LLM_PROVIDER", "LLM_API_KEY")}
    os.environ["LLM_PROVIDER"] = "api"
    try:
        with pytest.raises(LLMError, match="LLM_API_KEY"):
            generate("ignored")
    finally:
        for key, value in old.items():
            if value is not None:
                os.environ[key] = value


def test_api_request_uses_configured_key_and_user_agent(monkeypatch):
    monkeypatch.setattr(llm, "_load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "api")
    monkeypatch.setenv("LLM_API_KEY", "offline-test-key")
    captured = {}

    def mock_request(url, payload, headers):
        captured["url"] = url
        captured["headers"] = headers
        return {"choices": [{"message": {"content": "{}"}}]}

    monkeypatch.setattr(llm, "_json_request", mock_request)
    assert generate("Return an empty JSON object") == "{}"
    assert captured["headers"]["Authorization"] == "Bearer offline-test-key"
    assert captured["headers"]["User-Agent"] == "AIMScribe-ClinicalRAG/0.1"

@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "", "reasoning": "internal reasoning"},
                    }
                ]
            },
            "empty message content",
        ),
        (
            {
                "choices": [
                    {"finish_reason": "length", "message": {"content": '{"partial"'}}
                ]
            },
            "truncated",
        ),
        ({"choices": []}, "valid choice"),
        ({"choices": [{"finish_reason": "stop", "message": {}}]}, "message content"),
    ],
)
def test_groq_response_failures_are_explicit(monkeypatch, response, message):
    monkeypatch.setenv("LLM_CACHE", "0")  # exercise mocked transport, never cached output
    monkeypatch.setattr(llm, "_load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "api")
    monkeypatch.setenv("LLM_API_KEY", "offline-test-key")
    monkeypatch.setenv("LLM_API_URL", "https://api.groq.com/openai/v1/chat/completions")
    monkeypatch.setattr(llm, "_json_request", lambda *args: response)

    with pytest.raises(LLMError, match=message):
        generate("Return JSON")


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
    result = recommend(
        {},
        retrieve_fn=lambda **kwargs: retrieved(),
        llm_generate=lambda prompt: json.dumps(raw),
        cards={card.card_id: card},
    )
    assert result["medication"] == []
    assert "medication.0: unsupported medication field dose" in result[
        "rejected_recommendations"
    ]


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
