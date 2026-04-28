"""
Shared provider types: messages, configs, error hierarchy.

All values are immutable (frozen dataclasses) per project coding style.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """Provider-agnostic chat message."""
    role: Role
    content: str


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for an LLM provider instance."""
    provider: str
    api_key: str
    base_url: str
    model: str
    deployment_name: str = ""
    api_version: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: float = 300.0
    # Anthropic-style providers (real Anthropic, MiniMax) need this header.
    anthropic_version: str = "2023-06-01"
    # Ollama context window (only used by OpenAI-compatible provider when
    # base_url points to an Ollama server). Default 8192 because Ollama's
    # default of 2048 truncates prompts silently.
    ollama_num_ctx: int = 8192
    # Optional extra headers (rarely needed; useful for proxies).
    extra_headers: tuple = ()


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for an embedding provider instance."""
    provider: str
    api_key: str
    base_url: str
    model: str
    deployment_name: str = ""
    api_version: str = ""
    dimensions: int = 768
    timeout: int = 30
    max_retries: int = 3


# ----------------------------------------------------------------------
# Error hierarchy
# ----------------------------------------------------------------------


class ProviderError(Exception):
    """Base class for provider errors. Never includes secrets."""


class LLMProviderError(ProviderError):
    """Raised when an LLM provider call fails."""


class EmbeddingProviderError(ProviderError):
    """Raised when an embedding provider call fails."""


class EmbeddingDimensionMismatchError(EmbeddingProviderError):
    """Raised when configured dimensions disagree with an existing index/output."""

    def __init__(self, configured: int, actual: int, source: str = ""):
        self.configured = configured
        self.actual = actual
        self.source = source
        suffix = f" ({source})" if source else ""
        super().__init__(
            f"Embedding dimension mismatch: configured {configured}, "
            f"existing {source or 'value'} uses {actual}{suffix}."
        )
