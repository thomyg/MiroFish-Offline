"""Ollama embedding provider — POST {base_url}/api/embed.

Preserves the original EmbeddingService behaviour:
- Batch requests (Ollama supports multi-input in one call)
- In-memory cache
- Empty-text → zero-vector fallback
- Exponential-backoff retry on connection / 5xx errors
"""

import logging
import time
from typing import List, Optional, Sequence

import requests

from ..types import (
    EmbeddingConfig,
    EmbeddingDimensionMismatchError,
    EmbeddingProviderError,
)
from ._cache import EmbeddingCache

logger = logging.getLogger("mirofish.embedding.ollama")


class OllamaEmbeddingProvider:
    """Calls Ollama's /api/embed endpoint."""

    def __init__(
        self,
        config: EmbeddingConfig,
        session: Optional[requests.Session] = None,
        cache: Optional[EmbeddingCache] = None,
    ):
        if not config.base_url:
            raise EmbeddingProviderError("ollama provider requires EMBEDDING_BASE_URL")
        if not config.model:
            raise EmbeddingProviderError("ollama provider requires EMBEDDING_MODEL_NAME")
        if config.dimensions <= 0:
            raise EmbeddingProviderError(
                f"ollama provider requires EMBEDDING_DIMENSIONS > 0 (got {config.dimensions})"
            )

        self._config = config
        self._session = session or requests.Session()
        self._embed_url = f"{config.base_url.rstrip('/')}/api/embed"
        self._cache = cache or EmbeddingCache()

    @property
    def dimensions(self) -> int:
        return self._config.dimensions

    def embed(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise EmbeddingProviderError("Cannot embed empty text")

        text = text.strip()
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        vectors = self._request([text])
        vector = vectors[0]
        self._validate_dimension(vector)
        self._cache.put(text, vector)
        return vector

    def embed_batch(self, texts: Sequence[str], batch_size: int = 32) -> List[List[float]]:
        if not texts:
            return []

        results: list[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, raw in enumerate(texts):
            text = raw.strip() if raw else ""
            cached = self._cache.get(text) if text else None
            if cached is not None:
                results[i] = cached
            elif text:
                uncached_indices.append(i)
                uncached_texts.append(text)
            else:
                results[i] = [0.0] * self._config.dimensions

        if uncached_texts:
            all_vectors: list[List[float]] = []
            for start in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[start:start + batch_size]
                vectors = self._request(batch)
                for v in vectors:
                    self._validate_dimension(v)
                all_vectors.extend(vectors)

            for idx, vec, text in zip(uncached_indices, all_vectors, uncached_texts):
                results[idx] = vec
                self._cache.put(text, vec)

        return results  # type: ignore[return-value]

    def health_check(self) -> bool:
        try:
            return len(self.embed("health check")) > 0
        except Exception:
            return False

    # ------------------------------------------------------------------

    def _validate_dimension(self, vector: List[float]) -> None:
        if len(vector) != self._config.dimensions:
            raise EmbeddingDimensionMismatchError(
                configured=self._config.dimensions,
                actual=len(vector),
                source=f"Ollama model '{self._config.model}'",
            )

    def _request(self, texts: List[str]) -> List[List[float]]:
        payload = {"model": self._config.model, "input": texts}

        last_error: Exception | None = None
        for attempt in range(self._config.max_retries):
            try:
                response = self._session.post(
                    self._embed_url, json=payload, timeout=self._config.timeout
                )
                response.raise_for_status()
                data = response.json()
                embeddings = data.get("embeddings", [])
                if len(embeddings) != len(texts):
                    raise EmbeddingProviderError(
                        f"Expected {len(texts)} embeddings, got {len(embeddings)}"
                    )
                return embeddings
            except requests.exceptions.ConnectionError as e:
                last_error = e
                logger.warning(
                    "Ollama connection failed (attempt %d/%d)",
                    attempt + 1,
                    self._config.max_retries,
                )
            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning(
                    "Ollama request timed out (attempt %d/%d)",
                    attempt + 1,
                    self._config.max_retries,
                )
            except requests.exceptions.HTTPError as e:
                last_error = e
                status = e.response.status_code if e.response is not None else 0
                logger.error("Ollama HTTP error: %s", status)
                # Client errors (4xx) won't recover by retrying — fail fast
                if status and status < 500:
                    raise EmbeddingProviderError(
                        f"Ollama embedding HTTP {status}"
                    ) from e
            except (KeyError, ValueError) as e:
                raise EmbeddingProviderError(f"Invalid Ollama response: {e}") from e

            if attempt < self._config.max_retries - 1:
                wait = 2 ** attempt
                logger.info("Retrying Ollama embed in %ds...", wait)
                time.sleep(wait)

        raise EmbeddingProviderError(
            f"Ollama embedding failed after {self._config.max_retries} retries"
        ) from last_error
