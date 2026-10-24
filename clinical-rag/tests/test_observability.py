"""Regression tests for privacy-safe LLM measurement and retrieval metrics."""
import json

import llm
from telemetry import record_llm_request


def test_telemetry_records_only_allowlisted_metadata(tmp_path, monkeypatch):
    target = tmp_path / "llm.jsonl"
    monkeypatch.setenv("AIMS_TELEMETRY", "1")
    monkeypatch.setenv("AIMS_TELEMETRY_PATH", str(target))
    record_llm_request("https://api.groq.com/openai/v1/chat/completions?secret=PHI",
                       15.0, "ok", 200,
                       {"prompt_tokens": 12, "completion_tokens": 6,
                        "unexpected_patient": "sensitive text"})
    data = json.loads(target.read_text())
    assert data == {"provider": "groq", "duration_ms": 15.0,
                    "outcome": "ok", "http_status": 200,
                    "prompt_tokens": 12, "completion_tokens": 6}
    assert "PHI" not in target.read_text()
    assert "sensitive" not in target.read_text()


def test_telemetry_disabled_by_default(tmp_path, monkeypatch):
    target = tmp_path / "disabled.jsonl"
    monkeypatch.delenv("AIMS_TELEMETRY", raising=False)
    monkeypatch.setenv("AIMS_TELEMETRY_PATH", str(target))
    record_llm_request("https://api.groq.com", 3, "ok")
    assert not target.exists()


def test_llm_transport_event_without_credentials(monkeypatch, tmp_path):
    target = tmp_path / "events.jsonl"
    monkeypatch.setenv("AIMS_TELEMETRY", "1")
    monkeypatch.setenv("AIMS_TELEMETRY_PATH", str(target))

    def failing_urlopen(*args, **kwargs):
        raise ConnectionResetError("sensitive hidden connection")
    monkeypatch.setattr(llm.urllib.request, "urlopen", failing_urlopen)
    try:
        llm._json_request("https://api.groq.com/openai/v1/chat/completions",
                          {"messages": [{"content": "patient private text"}]},
                          {"Authorization": "Bearer PRIVATE"})
    except llm.LLMTransportError:
        pass
    else:
        raise AssertionError("Must propagate transport failure")
    entry = json.loads(target.read_text().strip())
    assert entry["outcome"] == "connection_reset"
    assert "PRIVATE" not in target.read_text()
    assert "patient" not in target.read_text()


def test_retrieval_evaluator_reports_disease_ranks():
    import importlib.util
    from pathlib import Path
    script = Path(__file__).with_name("eval_pipeline.py")
    spec = importlib.util.spec_from_file_location("local_eval_pipeline", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluate = module.evaluate
    from types import SimpleNamespace
    cards = {"A": SimpleNamespace(disease_protocol="Alpha"),
             "B": SimpleNamespace(disease_protocol="Beta")}
    def search(patient, category, **kwargs):
        if category == "Diagnosis":
            return [{"card_id": "A"}, {"card_id": "B"}]
        return []
    result = evaluate([{"id": "x", "patient": {}, "expected_disease": "Beta"}],
                      cards, search=search)
    assert result["disease_mrr_at_10"] == 0.5
    assert result["disease_recall_at_5"] == 1
    assert result["disease_recall_at_10"] == 1


def test_llm_timeout_is_configurable_and_bounded(monkeypatch):
    import pytest
    from types import SimpleNamespace
    captured = []
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return b'{"choices":[]}'
    def fake_open(request, timeout):
        captured.append(timeout)
        return Response()
    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_open)
    monkeypatch.setenv("AIMS_LLM_TIMEOUT_SECONDS", "60")
    assert llm._json_request("https://api.groq.com", {}, {}) == {"choices":[]}
    assert captured == [60.0]
    monkeypatch.setenv("AIMS_LLM_TIMEOUT_SECONDS", "10000")
    llm._json_request("https://api.groq.com", {}, {})
    assert captured[-1] == 90.0


def test_telemetry_classifies_connection_reset(monkeypatch, tmp_path):
    target = tmp_path / "reset.jsonl"
    monkeypatch.setenv("AIMS_TELEMETRY", "1")
    monkeypatch.setenv("AIMS_TELEMETRY_PATH", str(target))
    def reset(*args, **kwargs):
        raise ConnectionResetError("SECRET PATIENT")
    monkeypatch.setattr(llm.urllib.request, "urlopen", reset)
    try:
        llm._json_request("https://api.groq.com", {}, {})
    except llm.LLMTransportError:
        pass
    else:
        raise AssertionError("Expected LLMTransportError")
    event = json.loads(target.read_text())
    assert event["outcome"] == "connection_reset"
    assert "SECRET PATIENT" not in target.read_text()
