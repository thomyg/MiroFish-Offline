"""Azure OpenAI chat-completions provider.

Endpoint:
  {base_url}/openai/deployments/{deployment}/chat/completions?api-version={api_version}
Auth:
  Header `api-key: <key>`
"""

import logging
from typing import Optional, Sequence

import requests

from ..types import ChatMessage, LLMConfig, LLMProviderError
from ._common import messages_to_openai_format, strip_think_blocks

logger = logging.getLogger("mirofish.llm.azure_openai")


class AzureOpenAILLMProvider:
    """Sends chat-completion requests to an Azure OpenAI deployment."""

    def __init__(self, config: LLMConfig, session: Optional[requests.Session] = None):
        if not config.api_key:
            raise LLMProviderError("azure_openai provider requires LLM_API_KEY")
        if not config.base_url:
            raise LLMProviderError("azure_openai provider requires LLM_BASE_URL")
        if not config.deployment_name:
            raise LLMProviderError(
                "azure_openai provider requires LLM_DEPLOYMENT_NAME"
            )
        if not config.api_version:
            raise LLMProviderError("azure_openai provider requires LLM_API_VERSION")

        self._config = config
        self._session = session or requests.Session()

    def _build_url(self, model_override: Optional[str]) -> str:
        # Azure is deployment-scoped; `model_override` is largely ignored
        # for routing but we still allow callers to pin a model name in
        # request body for telemetry purposes.
        deployment = self._config.deployment_name
        base = self._config.base_url.rstrip("/")
        return (
            f"{base}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={self._config.api_version}"
        )

    def generate(
        self,
        messages: Sequence[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        url = self._build_url(model)
        body: dict = {
            "messages": messages_to_openai_format(messages),
            "temperature": (
                temperature if temperature is not None else self._config.temperature
            ),
            "max_tokens": max_tokens if max_tokens is not None else self._config.max_tokens,
        }
        # Some Azure deployments accept `model` in body, others ignore it.
        if model or self._config.model:
            body["model"] = model or self._config.model
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        headers = {
            "api-key": self._config.api_key,
            "Content-Type": "application/json",
        }

        try:
            response = self._session.post(
                url, json=body, headers=headers, timeout=self._config.timeout
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            raise LLMProviderError(
                f"azure_openai HTTP {status}: {_redact(str(e))}"
            ) from e
        except requests.exceptions.RequestException as e:
            raise LLMProviderError(
                f"azure_openai request failed: {type(e).__name__}: {_redact(str(e))}"
            ) from e
        except ValueError as e:
            raise LLMProviderError(f"azure_openai response was not JSON: {e}") from e

        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMProviderError(
                f"azure_openai response missing choices[0].message.content: {data}"
            ) from e

        return strip_think_blocks(content)


def _redact(text: str) -> str:
    """Best-effort scrub of secrets in error strings before logging."""
    if not text:
        return text
    # Strip common api-key patterns. Only used in error messages we raise.
    return text.replace("api-key", "api-key=***")
