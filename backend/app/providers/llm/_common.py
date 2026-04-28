"""Shared helpers for LLM providers."""

import re
from typing import Sequence

from ..types import ChatMessage


_THINK_BLOCK = re.compile(r"<think>[\s\S]*?</think>")


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks (e.g. MiniMax M2.5)."""
    if not text:
        return text
    return _THINK_BLOCK.sub("", text).strip()


def messages_to_openai_format(messages: Sequence[ChatMessage]) -> list[dict]:
    """Translate ChatMessage list to OpenAI {role, content} dicts."""
    return [{"role": m.role, "content": m.content} for m in messages]


def split_system_messages(messages: Sequence[ChatMessage]) -> tuple[str, list[dict]]:
    """Split out 'system' messages for Anthropic-style APIs.

    Anthropic Messages API takes the system prompt as a separate top-level
    field, not as a message. Multiple system messages are concatenated with
    blank lines.

    Returns:
        (system_prompt, non_system_messages_in_openai_format)
    """
    system_parts: list[str] = []
    rest: list[dict] = []
    for m in messages:
        if m.role == "system":
            system_parts.append(m.content)
        else:
            rest.append({"role": m.role, "content": m.content})
    return "\n\n".join(system_parts).strip(), rest


def is_ollama_base_url(base_url: str) -> bool:
    """Heuristic: detect Ollama servers by default port."""
    return ":11434" in (base_url or "")
