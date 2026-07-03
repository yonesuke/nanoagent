"""nanoagent — Nano-scale AI agent framework.

Tree-based agents with streaming, tools, guardrails, and memory.
Only one external dependency: ``openai``.

Quick start::

    from openai import AsyncClient
    from nanoagent import Agent, tool

    @tool(name="search")
    async def search(query: str) -> str:
        return f"Results for '{query}'"

    agent = Agent(
        name="assistant",
        instructions="You are a helpful assistant.",
        client=AsyncClient(),
        tools=[search],
    )

    async for event in agent.run("Hello!"):
        print(event.delta, end="")
"""

from .agent import Agent, HandoffTarget, RunConfig, RunResult
from .events import Event, EventType
from .exceptions import (
    GuardrailRejected,
    MaxTurnsExceeded,
    NanoagentError,
    NeedsHumanInput,
    ToolExecutionError,
)
from .guardrails import AgentHooks, Guardrail, GuardrailResult, guardrail
from .memory import ConversationMemory, Message, ToolCallRecord
from .node import Node
from .tools import Tool, ask_user, tool

__all__ = [
    # Core
    "Agent",
    "Node",
    # Events
    "Event",
    "EventType",
    # Tools
    "Tool",
    "tool",
    "ask_user",
    # Configuration
    "RunConfig",
    "RunResult",
    # Guardrails
    "Guardrail",
    "guardrail",
    "GuardrailResult",
    "AgentHooks",
    # Memory
    "ConversationMemory",
    "Message",
    "ToolCallRecord",
    # Exceptions
    "NanoagentError",
    "NeedsHumanInput",
    "GuardrailRejected",
    "ToolExecutionError",
    "MaxTurnsExceeded",
    # Handoff
    "HandoffTarget",
]