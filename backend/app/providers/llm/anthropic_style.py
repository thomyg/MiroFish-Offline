"""Anthropic-style Messages API provider.

Used for real Anthropic and providers that adopt the same wire format
(e.g. MiniMax M2.5).

Endpoint:
  POST {base_url}/v1/messages
Auth:
  Header `x-api-key: <key>` and `anthropic-version: <version>`
Mapping:
  - 'system' messages collapse into top-level `system` field
  - 'user' / 'assistant' messages stay in `messages` list
  - Response: `content[0].text`
"""

import logging
from typing import Optional, Sequence

import requests

from ..types import ChatMessage, LLMConfig, LLMProviderError
from ._common import split_system_messages, strip_think_blocks

logger = logging.getLogger("mirofish.llm.anthropic_style")


class AnthropicStyleLLMProvider:
    """Sends Messages-API style requests."""

    def __init__(self, config: LLMConfig, session: Optional[requests.Session] = None):
        if not config.api_key:
            raise LLMProviderError("anthropic_style provider requires LLM_API_KEY")
        if not config.base_url:
            raise LLMProviderError("anthropic_style provider requires LLM_BASE_URL")
        if not config.model:
            raise LLMProviderError("anthropic_style provider requires LLM_MODEL_NAME")

        self._config = config
        self._session = session or requests.Session()

    def _endpoint(self) -> str:
        base = self._config.base_url.rstrip("/")
        # Allow callers to point at full path (e.g. https://api.minimax.io/v1)
        # or at the host root.
        if base.endswith("/v1") or base.endswith("/messages"):
            if base.endswith("/v1"):
                return f"{base}/messages"
            return base
        return f"{base}/v1/messages"

    def generate(
        self,
        messages: Sequence[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        # json_mode is not part of the Anthropic Messages API. Callers must
        # rely on prompt-level JSON instructions; we still parse defensively
        # downstream (see LLMClient.chat_json).
        del json_mode

        system_prompt, message_list = split_system_messages(messages)

        body: dict = {
            "model": model or self._config.model,
            "messages": message_list,
            "max_tokens": max_tokens if max_tokens is not None else self._config.max_tokens,
        }
        if system_prompt:
            body["system"] = system_prompt
        if temperature is not None or self._config.temperature is not None:
            body["temperature"] = (
                temperature if temperature is not None else self._config.temperature
            )

        headers = {
            "x-api-key": self._config.api_key,
            "anthropic-version": self._config.anthropic_version,
            "content-type": "application/json",
        }

        try:
            response = self._session.post(
                self._endpoint(),
                json=body,
                headers=headers,
                timeout=self._config.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            raise LLMProviderError(f"anthropic_style HTTP {status}") from e
        except requests.exceptions.RequestException as e:
            raise LLMProviderError(
                f"anthropic_style request failed: {type(e).__name__}"
            ) from e
        except ValueError as e:
            raise LLMProviderError(f"anthropic_style response was not JSON: {e}") from e

        try:
            blocks = data["content"]
            if not blocks:
                raise LLMProviderError("anthropic_style response had empty content list")
            text = blocks[0].get("text", "")
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise LLMProviderError(
                "anthropic_style response missing content[0].text"
            ) from e

        return strip_think_blocks(text or "")
