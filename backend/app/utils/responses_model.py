"""CAMEL's completion interface backed by Responses, including reasoning tools."""

import json
from typing import Any

from camel.models import ModelFactory, OpenAIModel
from camel.types import ModelPlatformType
from openai.types.chat import ChatCompletion

from ..config import Config


class ResponsesModel(OpenAIModel):
    """Preserve CAMEL tool messages while retaining each call's reasoning items.

    OASIS shares a backend across agents. Cache by tool call ID, never by a
    global previous_response_id, so concurrent agents cannot share context.
    """

    def __init__(self, model: str, api_key: str, base_url: str, effort: str):
        super().__init__(
            model_type=model,
            api_key=api_key,
            url=base_url,
            model_config_dict={"reasoning_effort": effort},
        )
        self.reasoning_effort = effort
        self._tool_outputs: dict[str, list[dict[str, Any]]] = {}

    def _input_items(self, messages):
        items = []
        restored = set()
        for message in messages:
            role = message["role"]
            if role == "tool":
                output = message.get("content") or ""
                items.append({
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": output if isinstance(output, str) else json.dumps(output),
                })
                continue

            calls = message.get("tool_calls") or []
            cached = next((self._tool_outputs[c["id"]] for c in calls if c["id"] in self._tool_outputs), None)
            if cached is not None:
                key = next(c["id"] for c in calls if c["id"] in self._tool_outputs)
                if key not in restored:
                    items.extend(cached)
                    restored.update(item["call_id"] for item in cached if item["type"] == "function_call")
                continue

            if message.get("content") or not calls:
                items.append({"role": role, "content": message.get("content") or ""})
            for call in calls:
                items.append({
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                })
        return items

    def _request_args(self, messages, tools=None, response_format=None):
        args = {
            "model": str(self.model_type),
            "input": self._input_items(messages),
            "reasoning": {"effort": self.reasoning_effort},
            "include": ["reasoning.encrypted_content"],
            "store": False,
        }
        if tools:
            args["tools"] = [
                {"type": "function", **tool["function"], "strict": tool["function"].get("strict", False)}
                for tool in tools
            ]
        if response_format:
            if hasattr(response_format, "model_json_schema"):
                output_format = {"type": "json_schema", "name": response_format.__name__, "schema": response_format.model_json_schema(), "strict": False}
            else:
                output_format = response_format
            args["text"] = {"format": output_format}
        return args

    def _completion(self, response):
        if response.status not in {"completed", "incomplete"}:
            raise RuntimeError(f"Responses request {response.status}: {response.error}")
        output = [item.model_dump(mode="json", exclude_none=True) for item in response.output]
        calls = [item for item in output if item["type"] == "function_call"]
        for call in calls:
            self._tool_outputs[call["call_id"]] = output

        message = {"role": "assistant", "content": response.output_text or None}
        if calls:
            message["tool_calls"] = [
                {"id": call["call_id"], "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}
                for call in calls
            ]
        usage = response.usage
        return ChatCompletion(
            id=response.id,
            created=int(response.created_at),
            model=response.model,
            object="chat.completion",
            choices=[{"index": 0, "message": message, "finish_reason": "length" if response.status == "incomplete" else "tool_calls" if calls else "stop"}],
            usage={"prompt_tokens": usage.input_tokens, "completion_tokens": usage.output_tokens, "total_tokens": usage.total_tokens} if usage else None,
        )

    def _run(self, messages, response_format=None, tools=None):
        return self._completion(self._client.responses.create(**self._request_args(messages, tools, response_format)))

    async def _arun(self, messages, response_format=None, tools=None):
        response = await self._async_client.responses.create(**self._request_args(messages, tools, response_format))
        return self._completion(response)


def create_simulation_model(model: str, api_key: str, base_url: str):
    if model.lower().startswith(("gpt-5", "gpt-6")):
        return ResponsesModel(model, api_key, base_url, Config.LLM_REASONING_EFFORT)
    return ModelFactory.create(model_platform=ModelPlatformType.OPENAI, model_type=model)
