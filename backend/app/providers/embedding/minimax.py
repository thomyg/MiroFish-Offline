"""MiniMax embedding provider.

MiniMax's embedding endpoint is NOT OpenAI-compatible:
  Request:  {"model": "embo-01", "texts": [...], "type": "db" | "query"}
  Response: {"vectors": [[...], ...], "base_resp": {"status_code": 0, ...}}

The `type` field tells MiniMax whether the embedding is for storage
("db") or for similarity-search input ("query"). Since the storage path
embeds many texts and the search path embeds the user's query, we use
"db" for batches and "query" for single embeds. Override via env if
needed.

Endpoint: POST {base_url}/embeddings  (e.g. https://api.minimax.io/v1)
Auth:     Authorization: Bearer <key>
"""

import logging
import os
import time
from typing import List, Optional, Sequence

import requests

from ..types import (
    EmbeddingConfig,
    EmbeddingDimensionMismatchError,
    EmbeddingProviderError,
)
from ._cache import EmbeddingCache

logger = logging.getLogger("mirofish.embedding.minimax")

_DB_TYPE = os.environ.get("MINIMAX_EMBEDDING_DB_TYPE", "db")
_QUERY_TYPE = os.environ.get("MINIMAX_EMBEDDING_QUERY_TYPE", "query")


class MinimaxEmbeddingProvider:
    """Calls MiniMax's `/embeddings` endpoint."""

    def __init__(
        self,
        config: EmbeddingConfig,
        session: Optional[requests.Session] = None,
        cache: Optional[EmbeddingCache] = None,
    ):
        if not config.api_key:
            raise EmbeddingProviderError(
                "minimax embedding provider requires EMBEDDING_API_KEY"
            )
        if not config.base_url:
            raise EmbeddingProviderError(
                "minimax embedding provider requires EMBEDDING_BASE_URL"
            )
        if not config.model:
            raise EmbeddingProviderError(
                "minimax embedding provider requires EMBEDDING_MODEL_NAME"
            )
        if config.dimensions <= 0:
            raise EmbeddingProviderError(
                "minimax embedding provider requires EMBEDDING_DIMENSIONS > 0"
            )

        self._config = config
        self._session = session or requests.Session()
        self._cache = cache or EmbeddingCache()

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
        # Single calls are usually search queries.
        vectors = self._request([text], emb_type=_QUERY_TYPE)
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
                vectors = self._request(batch, emb_type=_DB_TYPE)
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
                source=f"MiniMax model '{self._config.model}'",
            )

    def _request(self, texts: List[str], emb_type: str) -> List[List[float]]:
        payload = {
            "model": self._config.model,
            "texts": texts,
            "type": emb_type,
        }
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self._config.max_retries):
            try:
                response = self._session.post(
                    self._url, json=payload, headers=headers, timeout=self._config.timeout
                )
                # MiniMax returns HTTP 200 with errors in body.base_resp.
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.HTTPError as e:
                last_error = e
                status = e.response.status_code if e.response is not None else 0
                if status and status < 500:
                    raise EmbeddingProviderError(
                        f"minimax embedding HTTP {status}"
                    ) from e
                logger.warning(
                    "minimax embedding HTTP %s (attempt %d/%d)",
                    status, attempt + 1, self._config.max_retries,
                )
                if attempt < self._config.max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_error = e
                logger.warning(
                    "minimax embedding network error (attempt %d/%d)",
                    attempt + 1, self._config.max_retries,
                )
                if attempt < self._config.max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
            except ValueError as e:
                raise EmbeddingProviderError(
                    f"minimax embedding response was not JSON: {e}"
                ) from e

            base_resp = data.get("base_resp") or {}
            status_code = base_resp.get("status_code", 0)

            if status_code == 1002:
                # Rate limit — wait long enough to clear a per-minute window.
                last_error = EmbeddingProviderError(
                    f"minimax rate limit: {base_resp.get('status_msg')}"
                )
                wait = min(60, 15 * (attempt + 1))
                logger.warning(
                    "minimax embedding rate-limited, retry in %ds (attempt %d/%d)",
                    wait, attempt + 1, self._config.max_retries,
                )
                if attempt < self._config.max_retries - 1:
                    time.sleep(wait)
                continue

            if status_code != 0:
                # Other API-level errors are not retryable.
                raise EmbeddingProviderError(
                    f"minimax embedding error {status_code}: {base_resp.get('status_msg')}"
                )

            vectors = data.get("vectors")
            if vectors is None:
                raise EmbeddingProviderError(
                    "minimax embedding response had null vectors with success status"
                )
            if len(vectors) != len(texts):
                raise EmbeddingProviderError(
                    f"minimax expected {len(texts)} vectors, got {len(vectors)}"
                )
            return vectors

        raise EmbeddingProviderError(
            f"minimax embedding failed after {self._config.max_retries} retries"
        ) from last_error
