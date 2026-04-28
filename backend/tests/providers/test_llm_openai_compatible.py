"""OpenAI-compatible LLM mapping: messages, Ollama num_ctx, <think> stripping."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.providers import ChatMessage, LLMConfig, LLMProviderError
from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider


def _config(base_url="http://localhost:11434/v1"):
    return LLMConfig(
        provider="openai_compatible",
        api_key="ollama",
        base_url=base_url,
        model="qwen2.5:32b",
        max_tokens=4096,
        temperature=0.7,
    )


def _fake_openai_client(content: str):
    """Build a MagicMock that mimics openai.OpenAI's chat.completions.create."""
    client = MagicMock()
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    client.chat.completions.create.return_value = response
    return client


@pytest.mark.unit
def test_generate_passes_messages_and_returns_content():
    client = _fake_openai_client("hello")
    provider = OpenAICompatibleLLMProvider(_config(), client=client)

    out = provider.generate(
        [ChatMessage("system", "be helpful"), ChatMessage("user", "hi")]
    )
    assert out == "hello"

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "qwen2.5:32b"
    assert kwargs["messages"] == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
    ]


@pytest.mark.unit
def test_ollama_base_url_injects_num_ctx():
    client = _fake_openai_client("ok")
    provider = OpenAICompatibleLLMProvider(_config(), client=client)

    provider.generate([ChatMessage("user", "ping")])
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"] == {"options": {"num_ctx": 8192}}


@pytest.mark.unit
def test_non_ollama_base_url_does_not_inject_num_ctx():
    client = _fake_openai_client("ok")
    cfg = _config(base_url="https://api.openai.com/v1")
    provider = OpenAICompatibleLLMProvider(cfg, client=client)

    provider.generate([ChatMessage("user", "ping")])
    kwargs = client.chat.completions.create.call_args.kwargs
    assert "extra_body" not in kwargs


@pytest.mark.unit
def test_json_mode_sets_response_format():
    client = _fake_openai_client('{"ok": true}')
    provider = OpenAICompatibleLLMProvider(_config(), client=client)

    provider.generate([ChatMessage("user", "?")], json_mode=True)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.unit
def test_strips_think_blocks_from_response():
    client = _fake_openai_client(
        "<think>hidden reasoning</think>\nactual answer"
    )
    provider = OpenAICompatibleLLMProvider(_config(), client=client)
    assert provider.generate([ChatMessage("user", "x")]) == "actual answer"


@pytest.mark.unit
def test_missing_api_key_rejected():
    cfg = LLMConfig(
        provider="openai_compatible",
        api_key="",
        base_url="http://x",
        model="m",
    )
    with pytest.raises(LLMProviderError):
        OpenAICompatibleLLMProvider(cfg, client=MagicMock())
