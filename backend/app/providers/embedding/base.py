"""EmbeddingProvider Protocol — the only interface app code may rely on."""

from typing import Protocol, runtime_checkable, Sequence


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Provider-agnostic embedding interface.

    Implementations live under `providers.embedding.*`. App code never
    constructs request bodies or knows the provider's wire format.
    """

    @property
    def dimensions(self) -> int:
        """Vector dimension produced by this provider."""
        ...

    def embed(self, text: str) -> list[float]:
        """Return a single embedding vector for `text`.

        Raises:
            EmbeddingProviderError: On request failure or empty input.
            EmbeddingDimensionMismatchError: If returned vector length
                disagrees with the configured dimension.
        """
        ...

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of texts.

        Empty strings yield zero-vectors of the configured dimension
        (preserves order so callers can zip with the input list).
        """
        ...

    def health_check(self) -> bool:
        """Best-effort liveness probe."""
        ...
