import types
import pytest

from RAG.RealEstate import agent


class DummyResp:
    def __init__(self, content=None, text=None):
        self.content = content
        self.text = text


def test_pick_response_mode_with_clean_json(monkeypatch):
    state = agent.RealEstateState()
    state.user_question = "How should I invest in residential property?"

    # Mock llm.invoke to return clean JSON in .content
    monkeypatch.setattr(agent.llm, "invoke", lambda prompt: DummyResp(content='{"needs_detailed_answer": true, "reason": "asked for a plan"}'))

    result = agent.pick_response_mode(state)
    assert result["needs_detailed_answer"] is True
    assert "plan" in result["answer_reason"]


def test_pick_response_mode_with_noisy_response_then_retry(monkeypatch):
    state = agent.RealEstateState()
    state.user_question = "Compare renting vs buying in 2026"

    # First response is noisy (text with explanation + JSON at end)
    noisy = 'Some explanation... {"needs_detailed_answer": false, "reason": "short answer ok"}'
    # Retry response will be strict JSON
    retry_json = '{"needs_detailed_answer": false, "reason": "short answer ok"}'

    calls = {"n": 0}

    def fake_invoke(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return DummyResp(content=noisy)
        return DummyResp(content=retry_json)

    monkeypatch.setattr(agent.llm, "invoke", fake_invoke)

    result = agent.pick_response_mode(state)
    assert result["needs_detailed_answer"] is False
    assert "short answer" in result["answer_reason"]


def test_pick_response_mode_fallback_when_unparseable(monkeypatch):
    state = agent.RealEstateState()
    state.user_question = "Give me a detailed investment plan"

    # Both primary and retry responses are unparseable
    monkeypatch.setattr(agent.llm, "invoke", lambda prompt: DummyResp(content="I cannot produce JSON"))

    result = agent.pick_response_mode(state)
    # Should fallback to quick mode
    assert result["needs_detailed_answer"] is False
    assert "defaulting to a quick answer" in result["answer_reason"].lower()
