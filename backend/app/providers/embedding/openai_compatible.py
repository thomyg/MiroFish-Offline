"""OpenAI-compatible embedding provider — POST {base_url}/embeddings.

Works with hosted OpenAI, OpenAI-compatible third parties, and local
servers exposing the same shape (request: {model, input}; response:
{data: [{embedding}, ...]}).
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

logger = logging.getLogger("mirofish.embedding.openai_compatible")


class OpenAICompatibleEmbeddingProvider:
    """Calls /v1/embeddings (OpenAI schema)."""

    def __init__(
        self,
        config: EmbeddingConfig,
        session: Optional[requests.Session] = None,
        cache: Optional[EmbeddingCache] = None,
    ):
        if not config.base_url:
            raise EmbeddingProviderError(
                "openai_compatible embedding provider requires EMBEDDING_BASE_URL"
            )
        if not config.model:
            raise EmbeddingProviderError(
                "openai_compatible embedding provider requires EMBEDDING_MODEL_NAME"
            )
        if config.dimensions <= 0:
            raise EmbeddingProviderError(
                "openai_compatible embedding provider requires EMBEDDING_DIMENSIONS > 0"
            )

        self._config = config
        self._session = session or requests.Session()
        self._cache = cache or EmbeddingCache()

        # Base may already include /v1; normalize to ".../embeddings".
        base = config.base_url.rstrip("/")
        if base.endswith("/embeddings"):
            self._url = base
        elif base.endswith("/v1"):
            self._url = f"{base}/embeddings"
        else:
            self._url = f"{base}/v1/embeddings"

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
        vec = vectors[0]
        self._validate_dimension(vec)
        self._cache.put(text, vec)
        return vec

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
                source=f"OpenAI-compatible model '{self._config.model}'",
            )

    def _build_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers

    def _request(self, texts: List[str]) -> List[List[float]]:
        payload = {"model": self._config.model, "input": texts}
        last_error: Exception | None = None
        for attempt in range(self._config.max_retries):
            try:
                response = self._session.post(
                    self._url,
                    json=payload,
                    headers=self._build_headers(),
                    timeout=self._config.timeout,
                )
                response.raise_for_status()
                data = response.json()
                items = data.get("data", [])
                if len(items) != len(texts):
                    raise EmbeddingProviderError(
                        f"Expected {len(texts)} embeddings, got {len(items)}"
                    )
                return [item["embedding"] for item in items]
            except requests.exceptions.HTTPError as e:
                last_error = e
                status = e.response.status_code if e.response is not None else 0
                if status and status < 500:
                    raise EmbeddingProviderError(
                        f"openai_compatible embedding HTTP {status}"
                    ) from e
                logger.warning(
                    "openai_compatible embedding HTTP %s (attempt %d/%d)",
                    status, attempt + 1, self._config.max_retries,
                )
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_error = e
                logger.warning(
                    "openai_compatible embedding network error (attempt %d/%d)",
                    attempt + 1, self._config.max_retries,
                )
            except (KeyError, ValueError) as e:
                raise EmbeddingProviderError(
                    f"Invalid openai_compatible embedding response: {e}"
                ) from e

            if attempt < self._config.max_retries - 1:
                time.sleep(2 ** attempt)

        raise EmbeddingProviderError(
            f"openai_compatible embedding failed after {self._config.max_retries} retries"
        ) from last_error
