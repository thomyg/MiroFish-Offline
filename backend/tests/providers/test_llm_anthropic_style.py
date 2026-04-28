"""Anthropic-style LLM: system split, content[0].text parsing, headers."""

import pytest

from app.providers import ChatMessage, LLMConfig, LLMProviderError
from app.providers.llm.anthropic_style import AnthropicStyleLLMProvider

from tests.conftest import FakeResponse, FakeSession


def _config(**overrides):
    base = dict(
        provider="anthropic_style",
        api_key="anthropic-key",
        base_url="https://api.minimax.io/v1",
        model="MiniMax-M2.5",
        max_tokens=2048,
        temperature=0.4,
        anthropic_version="2023-06-01",
    )
    base.update(overrides)
    return LLMConfig(**base)


def _ok_response(text="answer"):
    return FakeResponse(
        200,
        json_body={"content": [{"type": "text", "text": text}]},
    )


@pytest.mark.unit
def test_splits_system_message_into_top_level_field():
    session = FakeSession([_ok_response()])
    provider = AnthropicStyleLLMProvider(_config(), session=session)

    provider.generate(
        [
            ChatMessage("system", "be terse"),
            ChatMessage("user", "hello"),
            ChatMessage("assistant", "hi"),
            ChatMessage("user", "go on"),
        ]
    )

    body = session.calls[0].json
    assert body["system"] == "be terse"
    assert body["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "go on"},
    ]


@pytest.mark.unit
def test_concatenates_multiple_system_messages():
    session = FakeSession([_ok_response()])
    provider = AnthropicStyleLLMProvider(_config(), session=session)
    provider.generate(
        [
            ChatMessage("system", "rule one"),
            ChatMessage("system", "rule two"),
            ChatMessage("user", "?"),
        ]
    )
    body = session.calls[0].json
    assert body["system"] == "rule one\n\nrule two"


@pytest.mark.unit
def test_sets_anthropic_headers():
    session = FakeSession([_ok_response()])
    provider = AnthropicStyleLLMProvider(_config(), session=session)
    provider.generate([ChatMessage("user", "x")])

    headers = session.calls[0].headers
    assert headers["x-api-key"] == "anthropic-key"
    assert headers["anthropic-version"] == "2023-06-01"


@pytest.mark.unit
def test_parses_content_zero_text():
    session = FakeSession([_ok_response("hello world")])
    provider = AnthropicStyleLLMProvider(_config(), session=session)
    assert provider.generate([ChatMessage("user", "x")]) == "hello world"


@pytest.mark.unit
def test_strips_think_blocks_in_response():
    session = FakeSession([_ok_response("<think>secret</think>actual")])
    provider = AnthropicStyleLLMProvider(_config(), session=session)
    assert provider.generate([ChatMessage("user", "x")]) == "actual"


@pytest.mark.unit
def test_endpoint_appends_messages_when_base_ends_in_v1():
    session = FakeSession([_ok_response()])
    provider = AnthropicStyleLLMProvider(_config(), session=session)
    provider.generate([ChatMessage("user", "x")])
    assert session.calls[0].url == "https://api.minimax.io/v1/messages"


@pytest.mark.unit
def test_endpoint_appends_v1_messages_when_base_is_host():
    session = FakeSession([_ok_response()])
    provider = AnthropicStyleLLMProvider(
        _config(base_url="https://api.example.com"), session=session
    )
    provider.generate([ChatMessage("user", "x")])
    assert session.calls[0].url == "https://api.example.com/v1/messages"


@pytest.mark.unit
def test_empty_content_list_raises():
    session = FakeSession([FakeResponse(200, json_body={"content": []})])
    provider = AnthropicStyleLLMProvider(_config(), session=session)
    with pytest.raises(LLMProviderError):
        provider.generate([ChatMessage("user", "x")])
