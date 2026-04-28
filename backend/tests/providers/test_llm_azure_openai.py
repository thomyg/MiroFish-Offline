"""Azure OpenAI LLM: deployment URL, api-key header, response parsing."""

import pytest

from app.providers import ChatMessage, LLMConfig, LLMProviderError
from app.providers.llm.azure_openai import AzureOpenAILLMProvider

from tests.conftest import FakeResponse, FakeSession


def _config(**overrides):
    base = dict(
        provider="azure_openai",
        api_key="azure-secret",
        base_url="https://my-resource.openai.azure.com",
        model="gpt-4.1",
        deployment_name="gpt-4.1",
        api_version="2024-02-01",
    )
    base.update(overrides)
    return LLMConfig(**base)


def _ok_chat_response(text="hi"):
    return FakeResponse(
        200,
        json_body={
            "choices": [{"message": {"content": text}}],
        },
    )


@pytest.mark.unit
def test_builds_deployment_scoped_url():
    session = FakeSession([_ok_chat_response()])
    provider = AzureOpenAILLMProvider(_config(), session=session)

    provider.generate([ChatMessage("user", "ping")])

    call = session.calls[0]
    assert call.url == (
        "https://my-resource.openai.azure.com/openai/deployments/gpt-4.1/"
        "chat/completions?api-version=2024-02-01"
    )


@pytest.mark.unit
def test_uses_api_key_header_not_bearer():
    session = FakeSession([_ok_chat_response()])
    provider = AzureOpenAILLMProvider(_config(), session=session)

    provider.generate([ChatMessage("user", "x")])

    headers = session.calls[0].headers
    assert headers["api-key"] == "azure-secret"
    assert "Authorization" not in headers


@pytest.mark.unit
def test_returns_message_content_and_strips_think():
    session = FakeSession([_ok_chat_response("<think>x</think>real")])
    provider = AzureOpenAILLMProvider(_config(), session=session)
    assert provider.generate([ChatMessage("user", "x")]) == "real"


@pytest.mark.unit
def test_json_mode_in_body():
    session = FakeSession([_ok_chat_response('{"ok":true}')])
    provider = AzureOpenAILLMProvider(_config(), session=session)
    provider.generate([ChatMessage("user", "x")], json_mode=True)
    body = session.calls[0].json
    assert body["response_format"] == {"type": "json_object"}


@pytest.mark.unit
def test_missing_deployment_rejected():
    cfg = _config(deployment_name="")
    with pytest.raises(LLMProviderError):
        AzureOpenAILLMProvider(cfg, session=FakeSession([]))


@pytest.mark.unit
def test_4xx_raises_provider_error():
    session = FakeSession([FakeResponse(401, json_body={"error": "unauthorized"})])
    provider = AzureOpenAILLMProvider(_config(), session=session)
    with pytest.raises(LLMProviderError):
        provider.generate([ChatMessage("user", "x")])
