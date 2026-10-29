"""OpenAI-compatible Chat Completions request helpers.

Keep the local Ollama request shape while adapting model-specific parameters
for GPT-5/6 family endpoints and normalizing SDK response content.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..config import Config


def is_gpt5_family(model: Optional[str]) -> bool:
    """Return whether the configured model belongs to the GPT-5 family."""
    if not model:
        return False
    return model.strip().lower().startswith("gpt-5")


def create_chat_completion(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    reasoning_effort: Optional[str] = None,
) -> Any:
    """Create a completion using parameters supported by the model family."""
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    if response_format is not None:
        kwargs["response_format"] = response_format

    reasoning_model = is_gpt5_family(model) or model.strip().lower().startswith("gpt-6")
    if reasoning_model:
        reasoning_effort = reasoning_effort or Config.LLM_REASONING_EFFORT
        if reasoning_effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Unsupported reasoning effort")
        kwargs["reasoning_effort"] = reasoning_effort
    if temperature is not None and not reasoning_model:
        kwargs["temperature"] = temperature

    if max_tokens is not None:
        token_parameter = "max_completion_tokens" if reasoning_model else "max_tokens"
        kwargs[token_parameter] = max_tokens

    if extra_body is not None:
        kwargs["extra_body"] = extra_body

    return client.chat.completions.create(**kwargs)


def extract_chat_completion_text(response: Any) -> str:
    """Extract plain text across common OpenAI-compatible SDK content shapes."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text_obj = item.get("text")
                if isinstance(text_obj, dict):
                    text_obj = text_obj.get("value")
                if isinstance(text_obj, str):
                    chunks.append(text_obj)
                elif isinstance(item.get("content"), str):
                    chunks.append(item["content"])
                continue

            text_obj = getattr(item, "text", None)
            if isinstance(text_obj, dict):
                text_obj = text_obj.get("value")
            if isinstance(text_obj, str):
                chunks.append(text_obj)
                continue

            content_obj = getattr(item, "content", None)
            if isinstance(content_obj, str):
                chunks.append(content_obj)

        return "".join(chunks).strip()

    return str(content or "")
