"""In-memory LRU-ish cache shared by embedding providers.

Mirrors the original behaviour of EmbeddingService: bounded dict with
~10% eviction once full. Not thread-safe — embeddings happen in the
Flask request thread or a single simulation worker.
"""

from typing import List


class EmbeddingCache:
    def __init__(self, max_size: int = 2000):
        self._store: dict[str, List[float]] = {}
        self._max_size = max_size

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def get(self, key: str) -> List[float] | None:
        return self._store.get(key)

    def put(self, key: str, value: List[float]) -> None:
        if len(self._store) >= self._max_size:
            # Evict ~10% oldest entries (insertion order, dicts preserve it
            # since Python 3.7).
            evict_n = max(1, self._max_size // 10)
            for k in list(self._store.keys())[:evict_n]:
                del self._store[k]
        self._store[key] = value

    def clear(self) -> None:
        self._store.clear()
