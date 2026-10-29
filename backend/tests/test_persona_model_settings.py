"""Model migration checks; all SDK calls are mocked and never use paid APIs."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.api.persona_lab import _persona_llm
from app.utils.llm_client import LLMClient
from app.utils.openai_chat_compat import create_chat_completion


@pytest.mark.parametrize("model", ["gpt-5.6-luna", "gpt-6-luna"])
def test_reasoning_model_xhigh_request_has_supported_parameters(model):
    create = Mock()
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    create_chat_completion(client, model=model, messages=[], temperature=.3,
                           max_tokens=18000, reasoning_effort="xhigh",
                           response_format={"type": "json_object"})
    kwargs = create.call_args.kwargs
    assert kwargs["reasoning_effort"] == "xhigh"
    assert kwargs["max_completion_tokens"] == 18000
    assert "temperature" not in kwargs
    assert "max_tokens" not in kwargs


def test_persona_defaults_to_gpt61_high(monkeypatch):
    monkeypatch.setenv("PERSONA_LAB_API_KEY", "test-only")
    monkeypatch.delenv("PERSONA_LAB_MODEL", raising=False)
    monkeypatch.delenv("PERSONA_LAB_REASONING_EFFORT", raising=False)
    client = _persona_llm()
    assert client.model == "gpt-6.1-terra"
    assert client.reasoning_effort == "high"


@pytest.mark.parametrize("method", ["chat", "chat_json"])
def test_llm_client_forwards_effort_for_text_and_json(monkeypatch, method):
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content='{"ok":true}'))])
    create = Mock(return_value=response)
    monkeypatch.setattr("app.utils.llm_client.create_chat_completion", create)
    client = LLMClient(api_key="test-only", base_url="https://api.openai.com/v1",
                       model="gpt-5.6-luna", reasoning_effort="xhigh")
    getattr(client, method)(messages=[{"role": "user", "content": "Return JSON."}], max_tokens=None)
    assert create.call_args.kwargs["model"] == "gpt-5.6-luna"
    assert create.call_args.kwargs["reasoning_effort"] == "xhigh"
    assert create.call_args.kwargs["max_tokens"] is None


def test_local_ollama_request_is_unchanged():
    create = Mock()
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    create_chat_completion(client, model="qwen2.5:14b", messages=[], temperature=.3, max_tokens=123)
    assert create.call_args.kwargs["temperature"] == .3
    assert create.call_args.kwargs["max_tokens"] == 123
    assert "reasoning_effort" not in create.call_args.kwargs
