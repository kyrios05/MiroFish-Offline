from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from camel.agents import ChatAgent
from camel.toolkits import FunctionTool

from app.utils.responses_model import ResponsesModel


def _response(call_id=None, text="done"):
    output = []
    if call_id:
        output = [
            {"type": "reasoning", "id": "rs-" + call_id, "summary": [], "encrypted_content": "signed-" + call_id},
            {"type": "function_call", "id": "fc-" + call_id, "call_id": call_id, "name": "probe", "arguments": "{}", "status": "completed"},
        ]
    return SimpleNamespace(
        id="resp-" + (call_id or "text"), created_at=1, model="gpt-6.1-terra",
        status="completed", error=None, output_text="" if call_id else text,
        output=[],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
    ) if not output else _tool_response(output, call_id)


def _tool_response(output, call_id):
    class Item:
        def __init__(self, value):
            self.value = value

        def model_dump(self, **kwargs):
            return self.value.copy()

    return SimpleNamespace(
        id="resp-" + call_id, created_at=1, model="gpt-6.1-terra",
        status="completed", error=None, output_text="",
        output=[Item(item) for item in output],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
    )


@pytest.fixture
def model():
    return ResponsesModel("gpt-6.1-terra", "test-only", "https://api.openai.com/v1", "high")


def test_sync_requests_use_high_and_responses_tool_shape(model):
    create = Mock(return_value=_response("call-a"))
    model._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    tool = {"type": "function", "function": {"name": "probe", "parameters": {"type": "object", "properties": {}}}}
    result = model.run([{"role": "user", "content": "Use probe."}], tools=[tool])
    args = create.call_args.kwargs
    assert args["model"] == "gpt-6.1-terra"
    assert args["reasoning"] == {"effort": "high"}
    assert args["store"] is False
    assert args["tools"][0]["name"] == "probe"
    assert args["tools"][0]["strict"] is False
    assert "function" not in args["tools"][0]
    assert result.choices[0].message.tool_calls[0].id == "call-a"
    assert result.usage.total_tokens == 15


def test_concurrent_agent_histories_replay_only_their_own_reasoning(model):
    model._completion(_response("call-a"))
    model._completion(_response("call-b"))
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-b", "function": {"name": "probe", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call-b", "content": "pong"},
    ]
    items = model._input_items(messages)
    assert items[0]["encrypted_content"] == "signed-call-b"
    assert items[1]["call_id"] == "call-b"
    assert items[2] == {"type": "function_call_output", "call_id": "call-b", "output": "pong"}
    assert "call-a" not in str(items)


@pytest.mark.asyncio
async def test_camel_executes_a_tool_and_preserves_reasoning_on_followup(model):
    def probe() -> str:
        """Return the connectivity probe result."""
        return "pong"

    create = AsyncMock(side_effect=[_response("call-live"), _response(text="pong received")])
    model._async_client = SimpleNamespace(responses=SimpleNamespace(create=create))
    agent = ChatAgent(system_message="Use the probe tool.", model=model, tools=[FunctionTool(probe)])
    result = await agent.astep("Run the probe and report its result.")
    assert result.msgs[0].content == "pong received"
    assert create.await_count == 2
    second = create.await_args_list[1].kwargs
    assert second["reasoning"] == {"effort": "high"}
    assert any(item.get("encrypted_content") == "signed-call-live" for item in second["input"])
    assert any(item.get("type") == "function_call_output" and item["call_id"] == "call-live" for item in second["input"])


def test_multiple_function_calls_restore_the_response_only_once(model):
    response = _response("a")
    original = [item.model_dump() for item in response.output]
    original.append({"type": "function_call", "id": "fc-b", "call_id": "b", "name": "probe", "arguments": "{}"})
    model._completion(_tool_response(original, "multi"))
    calls = lambda name: {"role": "assistant", "tool_calls": [{"id": name, "function": {"name": "probe", "arguments": "{}"}}]}
    items = model._input_items([calls("a"), calls("b")])
    assert [item.get("call_id") for item in items if item["type"] == "function_call"] == ["a", "b"]
