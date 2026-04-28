"""Factory: provider selection by config + error on unknown values."""

from types import SimpleNamespace

import pytest

from app.providers import (
    EmbeddingConfig,
    EmbeddingProviderError,
    LLMConfig,
    LLMProviderError,
    create_embedding_provider,
    create_llm_provider,
)
from app.providers.embedding.azure_openai import AzureOpenAIEmbeddingProvider
from app.providers.embedding.minimax import MinimaxEmbeddingProvider
from app.providers.embedding.ollama import OllamaEmbeddingProvider
from app.providers.embedding.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)
from app.providers.llm.anthropic_style import AnthropicStyleLLMProvider
from app.providers.llm.azure_openai import AzureOpenAILLMProvider
from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider


def _llm_config(provider, **overrides):
    base = dict(
        provider=provider,
        api_key="key",
        base_url="http://x",
        model="m",
        deployment_name="dep",
        api_version="2024-02-01",
    )
    base.update(overrides)
    return LLMConfig(**base)


def _embed_config(provider, **overrides):
    base = dict(
        provider=provider,
        api_key="key",
        base_url="http://x",
        model="m",
        deployment_name="dep",
        api_version="2024-02-01",
        dimensions=128,
    )
    base.update(overrides)
    return EmbeddingConfig(**base)


@pytest.mark.unit
def test_factory_picks_openai_compatible_llm():
    p = create_llm_provider(_llm_config("openai_compatible", base_url="http://x/v1"))
    assert isinstance(p, OpenAICompatibleLLMProvider)


@pytest.mark.unit
def test_factory_picks_azure_openai_llm():
    p = create_llm_provider(_llm_config("azure_openai"))
    assert isinstance(p, AzureOpenAILLMProvider)


@pytest.mark.unit
def test_factory_picks_anthropic_style_llm():
    p = create_llm_provider(_llm_config("anthropic_style"))
    assert isinstance(p, AnthropicStyleLLMProvider)


@pytest.mark.unit
def test_factory_rejects_unknown_llm_provider():
    with pytest.raises(LLMProviderError):
        create_llm_provider(_llm_config("does_not_exist"))


@pytest.mark.unit
def test_factory_picks_ollama_embedding():
    p = create_embedding_provider(_embed_config("ollama"))
    assert isinstance(p, OllamaEmbeddingProvider)


@pytest.mark.unit
def test_factory_picks_openai_compatible_embedding():
    p = create_embedding_provider(_embed_config("openai_compatible"))
    assert isinstance(p, OpenAICompatibleEmbeddingProvider)


@pytest.mark.unit
def test_factory_picks_azure_openai_embedding():
    p = create_embedding_provider(_embed_config("azure_openai"))
    assert isinstance(p, AzureOpenAIEmbeddingProvider)


@pytest.mark.unit
def test_factory_picks_minimax_embedding():
    p = create_embedding_provider(_embed_config("minimax"))
    assert isinstance(p, MinimaxEmbeddingProvider)


@pytest.mark.unit
def test_factory_rejects_unknown_embedding_provider():
    with pytest.raises(EmbeddingProviderError):
        create_embedding_provider(_embed_config("imaginary"))


@pytest.mark.unit
def test_factory_reads_from_config_class():
    """The factory should accept a Config-like class for env-driven init."""
    fake_config = SimpleNamespace(
        LLM_PROVIDER="anthropic_style",
        LLM_API_KEY="k",
        LLM_BASE_URL="https://api.example/v1",
        LLM_MODEL_NAME="some-model",
        LLM_DEPLOYMENT_NAME="",
        LLM_API_VERSION="",
        LLM_MAX_TOKENS=2048,
        LLM_TEMPERATURE=0.5,
        LLM_ANTHROPIC_VERSION="2023-06-01",
    )
    p = create_llm_provider(config_cls=fake_config)
    assert isinstance(p, AnthropicStyleLLMProvider)
