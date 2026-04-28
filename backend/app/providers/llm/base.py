"""LLMProvider Protocol — the only interface app code may rely on."""

from typing import Protocol, runtime_checkable, Optional, Sequence

from ..types import ChatMessage


@runtime_checkable
class LLMProvider(Protocol):
    """Provider-agnostic LLM interface.

    Implementations live under `providers.llm.*` and translate
    `messages` into the provider's wire format. Application code never
    constructs request bodies itself.
    """

    def generate(
        self,
        messages: Sequence[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        """Send a chat completion request and return the text content.

        Args:
            messages: Ordered chat messages. Implementations may project
                'system' messages into provider-specific fields.
            temperature: Sampling temperature; falls back to config default.
            max_tokens: Max output tokens; falls back to config default.
            model: Override model name; falls back to config default.
            json_mode: Hint that the caller expects strict JSON output.
                Providers that support it enable a JSON response format;
                others ignore it (callers must still parse defensively).

        Returns:
            Model output text. Provider-specific markers like
            `<think>...</think>` thinking blocks are stripped.

        Raises:
            LLMProviderError: On HTTP, parsing, or empty-response failures.
        """
        ...
