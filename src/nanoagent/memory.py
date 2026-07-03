"""Conversation memory for persistent agent sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from openai.types.chat import ChatCompletionMessageParam

MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCallRecord:
    """Record of a tool call within a message."""

    id: str
    name: str
    arguments: str


@dataclass
class Message:
    """A single message in the conversation history."""

    role: MessageRole
    content: str | None = None
    tool_calls: list[ToolCallRecord] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_openai(self) -> ChatCompletionMessageParam:
        """Convert to an OpenAI-compatible message dict."""
        msg: dict[str, Any] = {"role": self.role}

        if self.content is not None:
            msg["content"] = self.content

        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]

        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id

        if self.name is not None:
            msg["name"] = self.name

        return msg  # type: ignore[return-value]


@dataclass
class ConversationMemory:
    """Manages conversation history across multiple agent turns.

    Usage:
        memory = ConversationMemory()
        agent = Agent(name="bot", ..., memory=memory)

        await agent.run("My name is Alice")
        await agent.run("What's my name?")  # remembers "Alice"
    """

    messages: list[Message] = field(default_factory=list)
    max_messages: int = 100

    def add(self, msg: Message) -> None:
        """Add a message to the conversation history."""
        self.messages.append(msg)
        self._trim()

    def add_user(self, content: str) -> None:
        """Shorthand for adding a user message."""
        self.add(Message(role="user", content=content))

    def add_assistant(self, content: str | None = None, tool_calls: list[ToolCallRecord] | None = None) -> None:
        """Shorthand for adding an assistant message."""
        self.add(Message(role="assistant", content=content, tool_calls=tool_calls))

    def add_tool(self, content: str, tool_call_id: str) -> None:
        """Shorthand for adding a tool result message."""
        self.add(Message(role="tool", content=content, tool_call_id=tool_call_id))

    def to_openai_messages(self) -> list[ChatCompletionMessageParam]:
        """Convert all messages to OpenAI-compatible format."""
        return [m.to_openai() for m in self.messages]

    def clear(self) -> None:
        """Reset the conversation history."""
        self.messages.clear()

    def _trim(self) -> None:
        """Trim history to max_messages, preserving system messages."""
        if len(self.messages) <= self.max_messages:
            return
        # Keep system messages at the start
        non_system = [m for m in self.messages if m.role != "system"]
        system = [m for m in self.messages if m.role == "system"]
        overflow = len(self.messages) - self.max_messages
        self.messages = system + non_system[overflow:]