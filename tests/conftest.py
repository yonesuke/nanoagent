# -*- coding: utf-8 -*-
"""Shared test fixtures and utilities for nanoagent.

Provides a FakeLLM that monkeypatches the OpenAI AsyncClient to return
pre-programmed responses, enabling deterministic testing without API calls.

Inspired by OpenAI Agents SDK's ``FakeModel`` pattern.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from openai import AsyncClient


# -- Recursive dict-to-SimpleNamespace converter --


def _ns(obj: Any) -> Any:
    """Recursively convert dicts/lists to SimpleNamespace for attribute access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(item) for item in obj]
    return obj


# -- Fake chat completion chunk (for streaming) --


def _make_chunk(
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    """Build a fake OpenAI streaming chunk with proper attribute access."""
    delta = SimpleNamespace(content=content, tool_calls=_ns(tool_calls) if tool_calls else None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason, index=0)
    return SimpleNamespace(choices=[choice])


def _tool_call_delta(index: int, id_: str = "", name: str = "", args: str = "") -> dict[str, Any]:
    """Build a tool call delta dict (OpenAI streaming format)."""
    tc: dict[str, Any] = {"index": index}
    if id_:
        tc["id"] = id_
    if name or args:
        func: dict[str, Any] = {}
        if name:
            func["name"] = name
        if args:
            func["arguments"] = args
        tc["function"] = func
    return tc


# -- FakeLLM --


@dataclass
class _QueuedResponse:
    """A pre-programmed LLM response for one turn."""

    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stream_chunks: list[tuple[str | None, list[dict[str, Any]] | None]] = field(default_factory=list)


class FakeLLM:
    """Mock the OpenAI chat completions API.

    Pre-program responses and monkeypatch an AsyncClient to return them.
    Supports both streaming and non-streaming modes.

    Usage::

        fake = FakeLLM()
        fake.add_text_response("Hello, world!")
        fake.add_tool_call("get_weather", {"city": "Tokyo"})
        fake.add_text_response("The weather in Tokyo is sunny.")

        client = fake.patch(AsyncClient(api_key="fake"))
        agent = Agent(name="test", instructions="...", client=client)

        async for event in agent.run("..."):
            ...
    """

    def __init__(self):
        self._queue: list[_QueuedResponse] = []
        self._call_count = 0
        self._last_messages: list[dict[str, Any]] = []
        self._last_tools: list[dict[str, Any]] = []

    # -- Queueing --

    def add_text_response(self, text: str) -> FakeLLM:
        """Queue a simple text response."""
        self._queue.append(_QueuedResponse(text=text, stream_chunks=[(text, None)]))
        return self

    def add_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        call_id: str | None = None,
    ) -> FakeLLM:
        """Queue a tool call response with proper nested function structure."""
        tc_id = call_id or f"fake_call_{len(self._queue)}"
        args_str = json.dumps(tool_args)

        self._queue.append(
            _QueuedResponse(
                tool_calls=[
                    {
                        "id": tc_id,
                        "function_name": tool_name,
                        "function_args": args_str,
                    }
                ],
                stream_chunks=[
                    (
                        None,
                        [_tool_call_delta(0, id_=tc_id, name=tool_name, args=args_str)],
                    )
                ],
            )
        )
        return self

    def add_multi_turn(
        self,
        turns: list[tuple[str | None, str | None, dict[str, Any] | None]],
    ) -> FakeLLM:
        """Queue multiple turns at once.

        Each turn is (text, tool_name, tool_args).
        """
        for text, tool_name, tool_args in turns:
            if tool_name:
                self.add_tool_call(tool_name, tool_args or {})
            elif text:
                self.add_text_response(text)
        return self

    # -- Properties --

    @property
    def last_messages(self) -> list[dict[str, Any]]:
        return self._last_messages

    @property
    def last_tools(self) -> list[dict[str, Any]]:
        return self._last_tools

    @property
    def call_count(self) -> int:
        return self._call_count

    # -- Patching --

    def patch(self, client: AsyncClient) -> AsyncClient:
        """Monkeypatch client to return fake responses."""
        fake = self

        async def fake_create(**kwargs: Any) -> Any:
            fake._call_count += 1
            fake._last_messages = kwargs.get("messages", [])
            fake._last_tools = kwargs.get("tools", [])

            if not fake._queue:
                fake._queue.append(_QueuedResponse(text=""))

            response = fake._queue.pop(0)
            is_stream = kwargs.get("stream", False)

            if is_stream:
                return _FakeStream(response.stream_chunks)
            else:
                return _FakeCompletion(response.text, response.tool_calls)

        client.chat.completions.create = fake_create  # type: ignore[method-assign]
        return client


# -- Fake stream --


class _FakeStream:
    """Simulates an OpenAI streaming response."""

    def __init__(self, chunks: list[tuple[str | None, list[dict[str, Any]] | None]]):
        self._chunks = chunks
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        content, tool_calls = self._chunks[self._index]
        self._index += 1

        finish = "stop" if self._index >= len(self._chunks) else None
        return _make_chunk(content=content, tool_calls=tool_calls, finish_reason=finish)


# -- Fake non-stream completion --


class _FakeCompletion:
    """Simulates an OpenAI non-streaming completion response."""

    def __init__(self, text: str, tool_calls: list[dict[str, Any]]):
        tc_list = None
        if tool_calls:
            tc_list = [
                SimpleNamespace(
                    id=tc["id"],
                    type="function",
                    function=SimpleNamespace(
                        name=tc["function_name"],
                        arguments=tc["function_args"],
                    ),
                )
                for tc in tool_calls
            ]

        message = SimpleNamespace(content=text or None, tool_calls=tc_list)
        self.choices = [SimpleNamespace(message=message, index=0)]


# -- Fixtures --


@pytest.fixture
def fake_llm() -> FakeLLM:
    """Fresh FakeLLM instance for each test."""
    return FakeLLM()


@pytest.fixture
def client(fake_llm: FakeLLM) -> AsyncClient:
    """AsyncClient patched with FakeLLM."""
    return fake_llm.patch(AsyncClient(api_key="fake-test-key", base_url="https://fake.test/v1"))