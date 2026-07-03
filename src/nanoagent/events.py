"""Stream events emitted by agents during execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """Types of events emitted during agent execution.

    Follows the Vercel AI SDK `start`/`delta`/`end` lifecycle pattern
    for frontend compatibility.
    """

    # Node lifecycle
    NODE_START = "node_start"
    NODE_END = "node_end"

    # Reasoning (thinking) block
    REASONING_START = "reasoning_start"
    REASONING_DELTA = "reasoning_delta"
    REASONING_END = "reasoning_end"

    # Text block
    TEXT_START = "text_start"
    TEXT_DELTA = "text_delta"
    TEXT_END = "text_end"

    # Tool calls
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

    # Human-in-the-loop
    HUMAN_INPUT_REQUEST = "human_input_request"

    # Error
    ERROR = "error"

    # Run lifecycle
    RUN_FINISH = "finish"


@dataclass
class Event:
    """A single event in the agent's streaming output.

    Each event carries full node identity (id, path, depth) so consumers
    can reconstruct the execution tree.
    """

    type: EventType
    node_id: str
    node_path: list[str]
    depth: int
    name: str | None = None
    delta: str | None = None
    tool_call_id: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: str | None = None
    question: str | None = None
    error: str | None = None
    finish_reason: str | None = None
    timestamp: float | None = None
    elapsed_ms: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        """Serialize to Server-Sent Events format."""
        data: dict[str, Any] = {
            "type": self.type.value,
            "node_id": self.node_id,
            "node_path": self.node_path,
            "depth": self.depth,
        }
        if self.name is not None:
            data["name"] = self.name
        if self.delta is not None:
            data["delta"] = self.delta
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_args is not None:
            data["tool_args"] = self.tool_args
        if self.tool_result is not None:
            data["tool_result"] = self.tool_result
        if self.question is not None:
            data["question"] = self.question
        if self.error is not None:
            data["error"] = self.error
        if self.finish_reason is not None:
            data["finish_reason"] = self.finish_reason
        if self.meta:
            data["meta"] = self.meta
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"