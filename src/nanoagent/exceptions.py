"""Exceptions for agent execution flow control."""

from __future__ import annotations


class NanoagentError(Exception):
    """Base exception for all nanoagent errors."""


class NeedsHumanInput(NanoagentError):
    """Raised by a tool to request human input mid-execution.

    The agent will pause and wait for a human response before continuing.
    """

    def __init__(self, question: str, tool_name: str = "ask_user"):
        super().__init__(question)
        self.question = question
        self.tool_name = tool_name


class GuardrailRejected(NanoagentError):
    """Raised when a guardrail blocks the agent's input or output."""

    def __init__(self, guardrail_name: str, message: str):
        super().__init__(f"[{guardrail_name}] {message}")
        self.guardrail_name = guardrail_name
        self.message = message


class ToolExecutionError(NanoagentError):
    """Raised when a tool execution fails."""

    def __init__(self, tool_name: str, original_error: Exception):
        super().__init__(f"Tool '{tool_name}' failed: {original_error}")
        self.tool_name = tool_name
        self.original_error = original_error


class MaxTurnsExceeded(NanoagentError):
    """Raised when the agent exceeds the maximum number of turns."""

    def __init__(self, max_turns: int):
        super().__init__(f"Agent exceeded maximum turns ({max_turns})")
        self.max_turns = max_turns