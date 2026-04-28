"""MiniMax embedding provider: native texts/vectors shape, base_resp errors."""

import pytest

from app.providers import (
    EmbeddingConfig,
    EmbeddingDimensionMismatchError,
    EmbeddingProviderError,
)
from app.providers.embedding.minimax import MinimaxEmbeddingProvider

from tests.conftest import FakeResponse, FakeSession


def _cfg(dimensions=4):
    return EmbeddingConfig(
        provider="minimax",
        api_key="sk-test",
        base_url="https://api.minimax.io/v1",
        model="embo-01",
        dimensions=dimensions,
        max_retries=1,
    )


def _ok(vectors):
    return FakeResponse(
        200,
        json_body={"vectors": vectors, "base_resp": {"status_code": 0, "status_msg": "success"}},
    )


@pytest.mark.unit
def test_request_shape_uses_texts_not_input():
    session = FakeSession([_ok([[0.1, 0.2, 0.3, 0.4]])])
    provider = MinimaxEmbeddingProvider(_cfg(), session=session)

    provider.embed("hello")
    body = session.calls[0].json
    assert body == {"model": "embo-01", "texts": ["hello"], "type": "query"}


@pytest.mark.unit
def test_endpoint_and_bearer_auth():
    session = FakeSession([_ok([[0.0, 0.0, 0.0, 0.0]])])
    provider = MinimaxEmbeddingProvider(_cfg(), session=session)
    provider.embed("x")
    assert session.calls[0].url == "https://api.minimax.io/v1/embeddings"
    assert session.calls[0].headers["Authorization"] == "Bearer sk-test"


@pytest.mark.unit
def test_batch_uses_db_type_and_zero_vector_for_empty():
    session = FakeSession([_ok([[0.5] * 4, [0.7] * 4])])
    provider = MinimaxEmbeddingProvider(_cfg(), session=session)

    out = provider.embed_batch(["a", "", "b"])
    assert out[0] == [0.5] * 4
    assert out[1] == [0.0] * 4
    assert out[2] == [0.7] * 4
    body = session.calls[0].json
    assert body["texts"] == ["a", "b"]
    assert body["type"] == "db"


@pytest.mark.unit
def test_dimension_mismatch_raises():
    session = FakeSession([_ok([[0.1] * 8])])
    provider = MinimaxEmbeddingProvider(_cfg(dimensions=4), session=session)
    with pytest.raises(EmbeddingDimensionMismatchError):
        provider.embed("x")


@pytest.mark.unit
def test_base_resp_error_propagates():
    session = FakeSession(
        [
            FakeResponse(
                200,
                json_body={
                    "vectors": None,
                    "base_resp": {"status_code": 2013, "status_msg": "invalid params"},
                },
            )
        ]
    )
    provider = MinimaxEmbeddingProvider(_cfg(), session=session)
    with pytest.raises(EmbeddingProviderError) as exc:
        provider.embed("x")
    assert "2013" in str(exc.value)
    assert "invalid params" in str(exc.value)


@pytest.mark.unit
def test_rate_limit_retried_then_fails():
    rate_limited = FakeResponse(
        200,
        json_body={
            "vectors": None,
            "base_resp": {"status_code": 1002, "status_msg": "rate limit exceeded(RPM)"},
        },
    )
    # max_retries=1 → only 1 attempt → no retry
    session = FakeSession([rate_limited])
    provider = MinimaxEmbeddingProvider(_cfg(), session=session)
    with pytest.raises(EmbeddingProviderError):
        provider.embed("x")
