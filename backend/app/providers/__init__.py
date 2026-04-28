"""
Provider abstraction layer.

Application code interacts with `LLMProvider` and `EmbeddingProvider`
Protocols. Concrete implementations live under `providers.llm.*` and
`providers.embedding.*`. Concrete providers are constructed via the
factories in `providers.factory`.

Provider-specific request/response logic MUST stay inside this package.
The rest of the codebase must remain provider-agnostic.
"""

from .types import (
    ChatMessage,
    LLMConfig,
    EmbeddingConfig,
    ProviderError,
    LLMProviderError,
    EmbeddingProviderError,
    EmbeddingDimensionMismatchError,
)
from .llm.base import LLMProvider
from .embedding.base import EmbeddingProvider
from .factory import (
    create_llm_provider,
    create_embedding_provider,
    build_llm_config,
    build_embedding_config,
)

__all__ = [
    "ChatMessage",
    "LLMConfig",
    "EmbeddingConfig",
    "LLMProvider",
    "EmbeddingProvider",
    "ProviderError",
    "LLMProviderError",
    "EmbeddingProviderError",
    "EmbeddingDimensionMismatchError",
    "create_llm_provider",
    "create_embedding_provider",
    "build_llm_config",
    "build_embedding_config",
]
