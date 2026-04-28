"""Factories that materialise providers from `Config`.

Application code should depend on the `LLMProvider` / `EmbeddingProvider`
Protocols and let these factories pick the concrete implementation.
"""

from typing import Optional, Type

from ..config import Config
from .types import (
    EmbeddingConfig,
    EmbeddingProviderError,
    LLMConfig,
    LLMProviderError,
)
from .llm.base import LLMProvider
from .llm.openai_compatible import OpenAICompatibleLLMProvider
from .llm.azure_openai import AzureOpenAILLMProvider
from .llm.anthropic_style import AnthropicStyleLLMProvider
from .embedding.base import EmbeddingProvider
from .embedding.ollama import OllamaEmbeddingProvider
from .embedding.openai_compatible import OpenAICompatibleEmbeddingProvider
from .embedding.azure_openai import AzureOpenAIEmbeddingProvider
from .embedding.minimax import MinimaxEmbeddingProvider


_LLM_REGISTRY: dict[str, Type] = {
    "openai_compatible": OpenAICompatibleLLMProvider,
    "azure_openai": AzureOpenAILLMProvider,
    "anthropic_style": AnthropicStyleLLMProvider,
}

_EMBEDDING_REGISTRY: dict[str, Type] = {
    "ollama": OllamaEmbeddingProvider,
    "openai_compatible": OpenAICompatibleEmbeddingProvider,
    "azure_openai": AzureOpenAIEmbeddingProvider,
    "minimax": MinimaxEmbeddingProvider,
}


def build_llm_config(config_cls: Type[Config] = Config) -> LLMConfig:
    """Materialise an immutable LLMConfig from the app Config class."""
    return LLMConfig(
        provider=config_cls.LLM_PROVIDER,
        api_key=config_cls.LLM_API_KEY or "",
        base_url=config_cls.LLM_BASE_URL or "",
        model=config_cls.LLM_MODEL_NAME or "",
        deployment_name=config_cls.LLM_DEPLOYMENT_NAME or "",
        api_version=config_cls.LLM_API_VERSION or "",
        max_tokens=config_cls.LLM_MAX_TOKENS,
        temperature=config_cls.LLM_TEMPERATURE,
        anthropic_version=config_cls.LLM_ANTHROPIC_VERSION or "2023-06-01",
    )


def build_embedding_config(config_cls: Type[Config] = Config) -> EmbeddingConfig:
    """Materialise an immutable EmbeddingConfig from the app Config class."""
    # If the user didn't set EMBEDDING_API_KEY but did set LLM_API_KEY and
    # is on the same provider family, fall back. This preserves the
    # existing single-key Ollama/OpenAI setup.
    api_key = config_cls.EMBEDDING_API_KEY or config_cls.LLM_API_KEY or ""
    return EmbeddingConfig(
        provider=config_cls.EMBEDDING_PROVIDER,
        api_key=api_key,
        base_url=config_cls.EMBEDDING_BASE_URL or "",
        model=config_cls.EMBEDDING_MODEL or "",
        deployment_name=config_cls.EMBEDDING_DEPLOYMENT_NAME or "",
        api_version=config_cls.EMBEDDING_API_VERSION or "",
        dimensions=config_cls.EMBEDDING_DIMENSIONS,
    )


def create_llm_provider(
    config: Optional[LLMConfig] = None,
    config_cls: Type[Config] = Config,
) -> LLMProvider:
    """Construct the concrete LLM provider declared by config.

    Raises:
        LLMProviderError: If LLM_PROVIDER is unknown.
    """
    cfg = config or build_llm_config(config_cls)
    impl = _LLM_REGISTRY.get(cfg.provider)
    if impl is None:
        raise LLMProviderError(
            f"Unknown LLM_PROVIDER '{cfg.provider}'. "
            f"Expected one of: {sorted(_LLM_REGISTRY)}"
        )
    return impl(cfg)


def create_embedding_provider(
    config: Optional[EmbeddingConfig] = None,
    config_cls: Type[Config] = Config,
) -> EmbeddingProvider:
    """Construct the concrete embedding provider declared by config."""
    cfg = config or build_embedding_config(config_cls)
    impl = _EMBEDDING_REGISTRY.get(cfg.provider)
    if impl is None:
        raise EmbeddingProviderError(
            f"Unknown EMBEDDING_PROVIDER '{cfg.provider}'. "
            f"Expected one of: {sorted(_EMBEDDING_REGISTRY)}"
        )
    return impl(cfg)
