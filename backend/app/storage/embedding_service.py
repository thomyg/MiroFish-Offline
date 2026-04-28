"""
EmbeddingService — backwards-compatible facade over the provider layer.

All embedding wire-format details live in `app.providers.embedding.*`.
This module keeps the historical `EmbeddingService` API alive so
`Neo4jStorage` and `SearchService` don't need to change.

Public API: `embed()`, `embed_batch()`, `health_check()`, `dimensions`.
"""

import logging
from typing import List, Optional, Sequence

from ..providers import (
    EmbeddingProvider,
    create_embedding_provider,
)
# Re-export the canonical error so legacy callers that catch EmbeddingError
# continue to work.
from ..providers.types import (
    EmbeddingDimensionMismatchError,  # noqa: F401  re-exported
    EmbeddingProviderError as EmbeddingError,
)

logger = logging.getLogger("mirofish.embedding")


class EmbeddingService:
    """Compatibility wrapper that delegates to a configured EmbeddingProvider."""

    def __init__(
        self,
        model: Optional[str] = None,  # noqa: ARG002  retained for legacy signature
        base_url: Optional[str] = None,  # noqa: ARG002
        max_retries: int = 3,  # noqa: ARG002
        timeout: int = 30,  # noqa: ARG002
        provider: Optional[EmbeddingProvider] = None,
    ):
        # Legacy positional/keyword args (model/base_url/...) are retained
        # for backwards compat with old call sites and are honored via env
        # config; explicit overrides need a custom EmbeddingConfig today.
        # If callers really need overrides we can extend this constructor.
        if provider is not None:
            self._provider = provider
        else:
            self._provider = create_embedding_provider()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dimensions(self) -> int:
        return self._provider.dimensions

    def embed(self, text: str) -> List[float]:
        return self._provider.embed(text)

    def embed_batch(
        self, texts: Sequence[str], batch_size: int = 32
    ) -> List[List[float]]:
        # batch_size is informational here; the underlying provider
        # decides its own batch chunking. We forward when the impl
        # supports it (Ollama / OpenAI-compatible / Azure all do).
        try:
            return self._provider.embed_batch(texts, batch_size=batch_size)  # type: ignore[call-arg]
        except TypeError:
            return self._provider.embed_batch(texts)

    def health_check(self) -> bool:
        return self._provider.health_check()

    @property
    def provider(self) -> EmbeddingProvider:
        return self._provider


__all__ = ["EmbeddingService", "EmbeddingError", "EmbeddingDimensionMismatchError"]
