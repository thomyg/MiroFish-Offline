"""Embedding providers: parsing, dimension validation, batch zero-vector."""

import pytest

from app.providers import (
    EmbeddingConfig,
    EmbeddingDimensionMismatchError,
    EmbeddingProviderError,
)
from app.providers.embedding.azure_openai import AzureOpenAIEmbeddingProvider
from app.providers.embedding.ollama import OllamaEmbeddingProvider
from app.providers.embedding.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)

from tests.conftest import FakeResponse, FakeSession


# ---------------------------------------------------------------------- Ollama


def _ollama_cfg(dimensions=4):
    return EmbeddingConfig(
        provider="ollama",
        api_key="",
        base_url="http://localhost:11434",
        model="nomic-embed-text",
        dimensions=dimensions,
        max_retries=1,
    )


@pytest.mark.unit
def test_ollama_embed_returns_vector_and_caches():
    session = FakeSession(
        [FakeResponse(200, json_body={"embeddings": [[0.1, 0.2, 0.3, 0.4]]})]
    )
    provider = OllamaEmbeddingProvider(_ollama_cfg(), session=session)

    v1 = provider.embed("hello")
    v2 = provider.embed("hello")  # cache hit; no second request
    assert v1 == [0.1, 0.2, 0.3, 0.4]
    assert v1 == v2
    assert len(session.calls) == 1


@pytest.mark.unit
def test_ollama_embed_endpoint_and_payload():
    session = FakeSession(
        [FakeResponse(200, json_body={"embeddings": [[0.0, 0.0, 0.0, 0.0]]})]
    )
    provider = OllamaEmbeddingProvider(_ollama_cfg(), session=session)
    provider.embed("hi")
    assert session.calls[0].url == "http://localhost:11434/api/embed"
    assert session.calls[0].json == {"model": "nomic-embed-text", "input": ["hi"]}


@pytest.mark.unit
def test_ollama_dimension_mismatch_raises():
    session = FakeSession(
        [FakeResponse(200, json_body={"embeddings": [[0.1] * 8]})]
    )
    provider = OllamaEmbeddingProvider(_ollama_cfg(dimensions=4), session=session)
    with pytest.raises(EmbeddingDimensionMismatchError) as exc:
        provider.embed("hi")
    assert exc.value.configured == 4
    assert exc.value.actual == 8


@pytest.mark.unit
def test_ollama_batch_uses_zero_vector_for_empty_text():
    session = FakeSession(
        [FakeResponse(200, json_body={"embeddings": [[0.5] * 4, [0.7] * 4]})]
    )
    provider = OllamaEmbeddingProvider(_ollama_cfg(), session=session)
    out = provider.embed_batch(["a", "", "b"])
    assert out[0] == [0.5, 0.5, 0.5, 0.5]
    assert out[1] == [0.0, 0.0, 0.0, 0.0]   # zero-vector for empty input
    assert out[2] == [0.7, 0.7, 0.7, 0.7]
    # Empty string should not be sent to the server
    assert session.calls[0].json == {
        "model": "nomic-embed-text",
        "input": ["a", "b"],
    }


@pytest.mark.unit
def test_ollama_empty_string_rejected():
    provider = OllamaEmbeddingProvider(_ollama_cfg(), session=FakeSession([]))
    with pytest.raises(EmbeddingProviderError):
        provider.embed("")


# ------------------------------------------------------------ OpenAI-compatible


def _openai_cfg(base_url="https://api.openai.com/v1", dimensions=3):
    return EmbeddingConfig(
        provider="openai_compatible",
        api_key="oa-key",
        base_url=base_url,
        model="text-embedding-3-small",
        dimensions=dimensions,
        max_retries=1,
    )


def _oa_response(vectors):
    return FakeResponse(
        200,
        json_body={"data": [{"embedding": v} for v in vectors]},
    )


@pytest.mark.unit
def test_openai_compatible_url_and_auth_header():
    session = FakeSession([_oa_response([[0.1, 0.2, 0.3]])])
    provider = OpenAICompatibleEmbeddingProvider(_openai_cfg(), session=session)
    provider.embed("x")
    assert session.calls[0].url == "https://api.openai.com/v1/embeddings"
    assert session.calls[0].headers["Authorization"] == "Bearer oa-key"


@pytest.mark.unit
def test_openai_compatible_normalizes_url_without_v1():
    session = FakeSession([_oa_response([[0.1, 0.2, 0.3]])])
    provider = OpenAICompatibleEmbeddingProvider(
        _openai_cfg(base_url="https://example.com"), session=session
    )
    provider.embed("x")
    assert session.calls[0].url == "https://example.com/v1/embeddings"


@pytest.mark.unit
def test_openai_compatible_dimension_mismatch_raises():
    session = FakeSession([_oa_response([[0.1] * 5])])
    provider = OpenAICompatibleEmbeddingProvider(_openai_cfg(dimensions=3), session=session)
    with pytest.raises(EmbeddingDimensionMismatchError):
        provider.embed("x")


# ---------------------------------------------------------------- Azure OpenAI


def _azure_cfg(dimensions=3):
    return EmbeddingConfig(
        provider="azure_openai",
        api_key="azure-key",
        base_url="https://my.openai.azure.com",
        model="text-embedding-3-small",
        deployment_name="text-embedding-3-small",
        api_version="2024-02-01",
        dimensions=dimensions,
        max_retries=1,
    )


@pytest.mark.unit
def test_azure_url_and_api_key_header():
    session = FakeSession(
        [FakeResponse(200, json_body={"data": [{"embedding": [0.1, 0.2, 0.3]}]})]
    )
    provider = AzureOpenAIEmbeddingProvider(_azure_cfg(), session=session)
    provider.embed("x")
    assert session.calls[0].url == (
        "https://my.openai.azure.com/openai/deployments/"
        "text-embedding-3-small/embeddings?api-version=2024-02-01"
    )
    assert session.calls[0].headers["api-key"] == "azure-key"
    # Azure body has no model field (deployment is in URL)
    assert "model" not in session.calls[0].json


@pytest.mark.unit
def test_azure_dimension_mismatch_raises():
    session = FakeSession(
        [FakeResponse(200, json_body={"data": [{"embedding": [0.1] * 9}]})]
    )
    provider = AzureOpenAIEmbeddingProvider(_azure_cfg(dimensions=3), session=session)
    with pytest.raises(EmbeddingDimensionMismatchError):
        provider.embed("x")


@pytest.mark.unit
def test_azure_missing_deployment_rejected():
    cfg = EmbeddingConfig(
        provider="azure_openai",
        api_key="k",
        base_url="https://x.openai.azure.com",
        model="m",
        deployment_name="",
        api_version="2024-02-01",
        dimensions=3,
    )
    with pytest.raises(EmbeddingProviderError):
        AzureOpenAIEmbeddingProvider(cfg, session=FakeSession([]))
