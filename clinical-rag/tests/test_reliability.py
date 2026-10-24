"""Regression tests for shared baseline/improved LLM transport reliability."""
import json
import pytest

import llm
from recommend import _parse_response_partial, MalformedRecommendationError


def test_json_fence_is_parsed_without_unsafe_repair():
    payload = {"probable_diagnosis": [], "investigations": [], "medication": [],
               "advice": [], "missing_info": ["More information required"]}
    parsed, rejected = _parse_response_partial("```json\n" + json.dumps(payload) + "\n```")
    assert parsed.missing_info == ["More information required"]
    assert not rejected


def test_non_dict_json_still_rejected():
    with pytest.raises(Exception):
        _parse_response_partial("[1, 2, 3]")


def test_rate_limit_retry_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "_load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "api")
    monkeypatch.setenv("LLM_MODEL", "offline-model")
    monkeypatch.setenv("LLM_API_KEY", "offline-test-key")
    monkeypatch.setenv("LLM_API_URL", "https://api.groq.com/openai/v1/chat/completions")
    monkeypatch.setenv("LLM_CACHE", "1")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)
    calls = []
    def request(url, payload, headers):
        calls.append(1)
        if len(calls) == 1:
            raise llm.LLMRateLimitError("0")
        return {"choices": [{"finish_reason": "stop",
                             "message": {"content": '{"missing_info": ["x"]}'}}]}
    monkeypatch.setattr(llm, "_json_request", request)
    assert llm.generate("same fictional input") == '{"missing_info": ["x"]}'
    assert llm.generate("same fictional input") == '{"missing_info": ["x"]}'
    assert len(calls) == 2


def test_429_fails_explicitly_after_bounded_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "_load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "api")
    monkeypatch.setenv("LLM_MODEL", "offline-model")
    monkeypatch.setenv("LLM_API_KEY", "offline-test-key")
    monkeypatch.setenv("LLM_API_URL", "https://api.groq.com/openai/v1/chat/completions")
    monkeypatch.setenv("LLM_CACHE", "0")
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)
    calls = []
    def blocked(*args):
        calls.append(1)
        raise llm.LLMRateLimitError("0")
    monkeypatch.setattr(llm, "_json_request", blocked)
    with pytest.raises(llm.LLMRateLimitError):
        llm.generate("fictional")
    assert len(calls) == 3


def test_transport_backoff_recovers_after_short_resets(monkeypatch):
    monkeypatch.setattr(llm, "_load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "api")
    monkeypatch.setenv("LLM_API_KEY", "offline-test-key")
    monkeypatch.setenv("LLM_CACHE", "0")
    monkeypatch.setattr(llm.random, "uniform", lambda a, b: 0.25)
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    calls = []
    def request(*args):
        calls.append(1)
        if len(calls) < 3:
            raise llm.LLMTransportError("simulated reset")
        return {"choices": [{"finish_reason": "stop",
                             "message": {"content": '{"status":"ok"}'}}]}
    monkeypatch.setattr(llm, "_json_request", request)
    assert llm.generate("fictional") == '{"status":"ok"}'
    assert len(calls) == 3
    assert sleeps == [2.25, 5.25]


def test_transport_failure_bounded_to_three_attempts(monkeypatch):
    monkeypatch.setattr(llm, "_load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "api")
    monkeypatch.setenv("LLM_API_KEY", "offline-test-key")
    monkeypatch.setenv("LLM_CACHE", "0")
    monkeypatch.setattr(llm.random, "uniform", lambda a, b: 0.0)
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    calls = []
    def request(*args):
        calls.append(1)
        raise llm.LLMTransportError("simulated reset")
    monkeypatch.setattr(llm, "_json_request", request)
    with pytest.raises(llm.LLMTransportError, match="after 3 attempts"):
        llm.generate("fictional")
    assert len(calls) == 3
    assert sleeps == [2.0, 5.0]
