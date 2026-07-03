# -*- coding: utf-8 -*-
"""Tests for streaming events and SSE serialization."""

from __future__ import annotations

import json

from nanoagent.events import Event, EventType


class TestEvent:
    def test_create_text_delta(self):
        e = Event(
            type=EventType.TEXT_DELTA,
            node_id="abc123",
            node_path=["root"],
            depth=0,
            delta="Hello",
        )
        assert e.type == EventType.TEXT_DELTA
        assert e.node_id == "abc123"
        assert e.delta == "Hello"
        assert e.node_path == ["root"]
        assert e.depth == 0

    def test_create_tool_call(self):
        e = Event(
            type=EventType.TOOL_CALL,
            node_id="abc123",
            node_path=["root", "child"],
            depth=1,
            name="search",
            tool_call_id="call_1",
            tool_args={"query": "weather"},
        )
        assert e.type == EventType.TOOL_CALL
        assert e.name == "search"
        assert e.tool_args == {"query": "weather"}

    def test_create_error(self):
        e = Event(
            type=EventType.ERROR,
            node_id="abc123",
            node_path=["root"],
            depth=0,
            error="Something went wrong",
        )
        assert e.error == "Something went wrong"

    def test_default_values(self):
        e = Event(
            type=EventType.NODE_START,
            node_id="a",
            node_path=["a"],
            depth=0,
        )
        assert e.delta is None
        assert e.tool_call_id is None
        assert e.tool_args is None
        assert e.tool_result is None
        assert e.question is None
        assert e.error is None
        assert e.meta == {}


class TestEventToSSE:
    def test_text_delta_sse(self):
        e = Event(
            type=EventType.TEXT_DELTA,
            node_id="n1",
            node_path=["agent"],
            depth=0,
            name="agent",
            delta="Hello",
        )
        sse = e.to_sse()
        assert sse.startswith("data: ")
        data = json.loads(sse[6:].strip())
        assert data["type"] == "text_delta"
        assert data["delta"] == "Hello"
        assert data["node_id"] == "n1"
        assert data["depth"] == 0

    def test_tool_result_sse(self):
        e = Event(
            type=EventType.TOOL_RESULT,
            node_id="n1",
            node_path=["agent"],
            depth=0,
            name="search",
            tool_call_id="c1",
            tool_result="Found 3 results",
            meta={"latency_ms": 42},
        )
        sse = e.to_sse()
        data = json.loads(sse[6:].strip())
        assert data["type"] == "tool_result"
        assert data["tool_result"] == "Found 3 results"
        assert data["meta"]["latency_ms"] == 42

    def test_no_optional_fields_in_sse(self):
        e = Event(
            type=EventType.NODE_END,
            node_id="n1",
            node_path=["agent"],
            depth=0,
        )
        sse = e.to_sse()
        data = json.loads(sse[6:].strip())
        assert "delta" not in data
        assert "error" not in data
        assert "tool_call_id" not in data
        assert "finish_reason" not in data

    def test_reasoning_delta_sse(self):
        e = Event(
            type=EventType.REASONING_DELTA,
            node_id="n1",
            node_path=["agent"],
            depth=0,
            delta="Let me think about this...",
        )
        sse = e.to_sse()
        data = json.loads(sse[6:].strip())
        assert data["type"] == "reasoning_delta"
        assert data["delta"] == "Let me think about this..."

    def test_reasoning_start_end_sse(self):
        for etype in (EventType.REASONING_START, EventType.REASONING_END):
            e = Event(
                type=etype,
                node_id="n1",
                node_path=["agent"],
                depth=0,
            )
            sse = e.to_sse()
            data = json.loads(sse[6:].strip())
            assert data["type"] == etype.value
            assert "delta" not in data

    def test_text_start_end_sse(self):
        for etype in (EventType.TEXT_START, EventType.TEXT_END):
            e = Event(
                type=etype,
                node_id="n1",
                node_path=["agent"],
                depth=0,
            )
            sse = e.to_sse()
            data = json.loads(sse[6:].strip())
            assert data["type"] == etype.value

    def test_finish_sse(self):
        e = Event(
            type=EventType.RUN_FINISH,
            node_id="n1",
            node_path=["agent"],
            depth=0,
            finish_reason="stop",
            meta={"usage": {"prompt_tokens": 10, "completion_tokens": 20}},
        )
        sse = e.to_sse()
        data = json.loads(sse[6:].strip())
        assert data["type"] == "finish"
        assert data["finish_reason"] == "stop"
        assert data["meta"]["usage"]["prompt_tokens"] == 10


class TestEventTypeValues:
    def test_all_event_types(self):
        assert EventType.NODE_START.value == "node_start"
        assert EventType.NODE_END.value == "node_end"
        assert EventType.REASONING_START.value == "reasoning_start"
        assert EventType.REASONING_DELTA.value == "reasoning_delta"
        assert EventType.REASONING_END.value == "reasoning_end"
        assert EventType.TEXT_START.value == "text_start"
        assert EventType.TEXT_DELTA.value == "text_delta"
        assert EventType.TEXT_END.value == "text_end"
        assert EventType.TOOL_CALL.value == "tool_call"
        assert EventType.TOOL_RESULT.value == "tool_result"
        assert EventType.HUMAN_INPUT_REQUEST.value == "human_input_request"
        assert EventType.ERROR.value == "error"
        assert EventType.RUN_FINISH.value == "finish"