"""OpenAI-compatible chat-completions provider.

Works with:
- Ollama (`http://localhost:11434/v1`) — default
- OpenAI itself (`https://api.openai.com/v1`)
- Any third-party API speaking the OpenAI Chat Completions schema
"""

import logging
from typing import Optional, Sequence

from openai import OpenAI

from ..types import ChatMessage, LLMConfig, LLMProviderError
from ._common import (
    is_ollama_base_url,
    messages_to_openai_format,
    strip_think_blocks,
)

logger = logging.getLogger("mirofish.llm.openai_compatible")


class OpenAICompatibleLLMProvider:
    """Sends chat-completion requests via the OpenAI SDK."""

    def __init__(self, config: LLMConfig, client: Optional[OpenAI] = None):
        if not config.api_key:
            raise LLMProviderError(
                "openai_compatible provider requires LLM_API_KEY "
                "(use any non-empty value for Ollama, e.g. 'ollama')"
            )
        if not config.base_url:
            raise LLMProviderError("openai_compatible provider requires LLM_BASE_URL")

        self._config = config
        self._client = client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )
        self._is_ollama = is_ollama_base_url(config.base_url)

    def generate(
        self,
        messages: Sequence[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        kwargs: dict = {
            "model": model or self._config.model,
            "messages": messages_to_openai_format(messages),
            "temperature": (
                temperature if temperature is not None else self._config.temperature
            ),
            "max_tokens": max_tokens if max_tokens is not None else self._config.max_tokens,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # Ollama-specific: prevent silent prompt truncation by raising num_ctx
        # above the 2048 default. Routed via OpenAI SDK's `extra_body`.
        if self._is_ollama and self._config.ollama_num_ctx:
            kwargs["extra_body"] = {
                "options": {"num_ctx": self._config.ollama_num_ctx}
            }

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            # OpenAI SDK raises a variety of error types; normalize to ours.
            raise LLMProviderError(
                f"openai_compatible request failed: {type(e).__name__}: {e}"
            ) from e

        if not response.choices:
            raise LLMProviderError("openai_compatible response had no choices")

        content = response.choices[0].message.content or ""
        return strip_think_blocks(content)
