"""
LLMClient — backwards-compatible facade over the provider layer.

All wire-format details live in `app.providers.llm.*`. This module exists
solely to keep existing call sites (`report_agent`, `ontology_generator`,
`graph_tools`, `ner_extractor`) working without modification.

Public API: `chat()` and `chat_json()`.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..providers import (
    ChatMessage,
    LLMConfig,
    LLMProvider,
    LLMProviderError,
    create_llm_provider,
)
from ..providers.factory import build_llm_config

logger = logging.getLogger("mirofish.llm_client")


class LLMClient:
    """Thin compatibility wrapper that delegates to a configured LLMProvider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 300.0,
        provider: Optional[LLMProvider] = None,
    ):
        if provider is not None:
            self._provider = provider
            return

        # Build a config from app Config, allow per-instance overrides for
        # legacy callers that pass api_key/base_url/model directly.
        base = build_llm_config()
        cfg = LLMConfig(
            provider=base.provider,
            api_key=api_key or base.api_key,
            base_url=base_url or base.base_url,
            model=model or base.model,
            deployment_name=base.deployment_name,
            api_version=base.api_version,
            max_tokens=base.max_tokens,
            temperature=base.temperature,
            timeout=timeout,
            anthropic_version=base.anthropic_version,
            ollama_num_ctx=base.ollama_num_ctx,
        )
        if not cfg.api_key:
            raise ValueError("LLM_API_KEY not configured")
        self._provider = create_llm_provider(cfg)

    # ------------------------------------------------------------------
    # Public API (legacy)
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> str:
        """Send a chat request and return text content.

        Mirrors the original signature: `messages` is a list of
        `{role, content}` dicts (callers were already producing this).
        """
        chat_messages = [
            ChatMessage(role=m["role"], content=m.get("content", ""))
            for m in messages
        ]
        json_mode = bool(response_format and response_format.get("type") == "json_object")
        try:
            return self._provider.generate(
                messages=chat_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
        except LLMProviderError:
            raise
        except Exception as e:
            raise LLMProviderError(f"LLMClient.chat failed: {type(e).__name__}") from e

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """Send a chat request and parse the response as JSON.

        Strips markdown code fences before parsing because some open
        models (and Anthropic-style providers ignoring `response_format`)
        wrap JSON in ```json blocks.
        """
        raw = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format from LLM: {cleaned}") from e

    # Convenience for new code that wants the raw provider.
    @property
    def provider(self) -> LLMProvider:
        return self._provider
