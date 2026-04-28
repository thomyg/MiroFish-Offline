"""Shared pytest fixtures and helpers for the provider test suite."""

import pytest


@pytest.fixture
def fake_session_factory():
    """Build a `requests.Session`-like double that records calls and replies
    with a sequence of canned responses.

    Usage:
        session, calls = fake_session_factory([{"json": {"foo": 1}}])
        session.post(...)  # → first canned response
        assert calls[0].url == "..."
    """

    def _factory(responses):
        return _FakeSession(responses)

    return _factory


class _FakeResponse:
    def __init__(self, status_code=200, json_body=None, raise_http=None):
        self.status_code = status_code
        self._json = json_body if json_body is not None else {}
        self._raise_http = raise_http
        self.text = ""

    def raise_for_status(self):
        if self._raise_http is not None:
            raise self._raise_http
        if self.status_code >= 400:
            import requests
            err = requests.exceptions.HTTPError(f"HTTP {self.status_code}")
            err.response = self
            raise err

    def json(self):
        return self._json


class _Call:
    def __init__(self, url, json, headers, timeout):
        self.url = url
        self.json = json
        self.headers = headers
        self.timeout = timeout


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[_Call] = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(_Call(url, json, headers, timeout))
        if not self._responses:
            return _FakeResponse(200, {})
        nxt = self._responses.pop(0)
        if isinstance(nxt, _FakeResponse):
            return nxt
        return _FakeResponse(
            status_code=nxt.get("status", 200),
            json_body=nxt.get("json"),
            raise_http=nxt.get("raise_http"),
        )


# Re-export helpers so tests can `from conftest import _FakeResponse` if needed.
FakeResponse = _FakeResponse
FakeSession = _FakeSession
