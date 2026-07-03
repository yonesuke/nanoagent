"""Agent — the primary LLM-powered node in the execution tree.

The :class:`Agent` class combines an LLM with tools, guardrails, memory,
and sub-agent delegation. It's the main entry point for building agent
workflows with nanoagent.

Quick start::

    from openai import AsyncClient
    from nanoagent import Agent, tool, RunConfig

    @tool(name="search")
    async def search(query: str) -> str:
        return f"Results for '{query}'"

    client = AsyncClient(api_key="...")
    agent = Agent(
        name="assistant",
        instructions="You are a helpful assistant.",
        client=client,
        tools=[search],
    )

    async for event in agent.run("What's the weather?"):
        if event.type == EventType.TEXT_DELTA:
            print(event.delta, end="")
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncClient
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from .events import Event, EventType
from .exceptions import GuardrailRejected, MaxTurnsExceeded, NeedsHumanInput
from .guardrails import AgentHooks, Guardrail, GuardrailResult
from .memory import ConversationMemory, ToolCallRecord
from .node import Node
from .tools import Tool


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class RunConfig:
    """Configuration for a single agent run.

    Attributes:
        model: Model name (default: ``"gpt-4o"``).
        temperature: Sampling temperature (0.0 to 2.0).
        max_turns: Maximum LLM-turn iterations before raising MaxTurnsExceeded.
        stream: Whether to stream text deltas.
        top_p: Nucleus sampling parameter.
    """

    model: str = "gpt-4o"
    temperature: float = 0.7
    max_turns: int = 20
    stream: bool = True
    top_p: float | None = None


# ── Run result ───────────────────────────────────────────────────────────


@dataclass
class RunResult:
    """Result of a completed agent run.

    Attributes:
        final_output: The agent's final text output.
        events: All events emitted during the run.
        tool_calls: List of tool call summaries (name + result).
    """

    final_output: str
    events: list[Event]
    tool_calls: list[dict[str, str]] = field(default_factory=list)


# ── Agent ─────────────────────────────────────────────────────────────────


class Agent(Node):
    """An LLM-powered agent that can use tools, delegate to sub-agents,
    request human input, and enforce guardrails.

    Parameters:
        name: Human-readable name for this agent.
        instructions: System prompt defining the agent's behavior.
        client: An openai ``AsyncClient`` instance.
        tools: List of :class:`Tool` instances the agent can call.
        config: :class:`RunConfig` for model, temperature, etc.
        parent: Parent node in the execution tree (set automatically for sub-agents).
        human_input_handler: Async callback for resolving ask_user requests.
        input_guardrails: List of :class:`Guardrail` checks run on agent input.
        output_guardrails: List of :class:`Guardrail` checks run on agent output.
        hooks: :class:`AgentHooks` for middleware interception.
        memory: Optional :class:`ConversationMemory` for persistent sessions.
    """

    def __init__(
        self,
        name: str,
        instructions: str,
        client: AsyncClient,
        *,
        tools: list[Tool | Agent] | None = None,
        config: RunConfig | None = None,
        parent: Node | None = None,
        human_input_handler: Callable[[str], Awaitable[str]] | None = None,
        input_guardrails: list[Guardrail] | None = None,
        output_guardrails: list[Guardrail] | None = None,
        hooks: AgentHooks | None = None,
        memory: ConversationMemory | None = None,
    ):
        super().__init__(name=name, parent=parent)
        self.instructions = instructions
        self.client = client
        # Auto-wrap Agent instances as tools
        self.tools: list[Tool] = [
            t.as_tool() if isinstance(t, Agent) else t for t in (tools or [])
        ]
        self.config = config or RunConfig()
        self.human_input_handler = human_input_handler
        self.input_guardrails = input_guardrails or []
        self.output_guardrails = output_guardrails or []
        self.hooks = hooks
        self.memory = memory

    # ── Public API ───────────────────────────────────────────────────────

    async def run(self, input_text: str) -> AsyncIterator[Event]:
        """Execute the agent with the given input, streaming events.

        This is the primary async API. Each event carries full node identity.

        Yields:
            :class:`Event` objects for each step of execution.
        """
        # Input guardrails
        for gr in self.input_guardrails:
            result = await gr.check(input_text, {"agent": self})
            if not result.allowed:
                yield self._make_event(
                    EventType.ERROR,
                    error=f"Guardrail [{gr.name}] rejected: {result.message}",
                )
                raise GuardrailRejected(gr.name, result.message or "rejected")

        # Hooks: before_agent
        if self.hooks and self.hooks.before_agent:
            await self.hooks.before_agent(input_text, {"agent": self})

        yield self._make_event(EventType.NODE_START)

        # Build initial messages
        history: list[ChatCompletionMessageParam] = (
            self.memory.to_openai_messages() if self.memory else []
        )
        system_msg: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": self.instructions,
        }
        user_msg: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": input_text,
        }
        messages: list[ChatCompletionMessageParam] = [system_msg, *history, user_msg]

        # Track memory if enabled
        if self.memory:
            self.memory.add_user(input_text)

        openai_tools = [t.to_openai_tool() for t in self.tools]
        tool_map = {t.name: t for t in self.tools}

        final_output = ""
        tool_call_summaries: list[dict[str, str]] = []

        # Main execution loop
        for turn in range(self.config.max_turns):
            # Hooks: before_model
            if self.hooks and self.hooks.before_model:
                await self.hooks.before_model(messages, {"agent": self, "turn": turn})

            # Run LLM
            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=openai_tools or None,
                temperature=self.config.temperature,
                stream=self.config.stream,
                top_p=self.config.top_p,
            )

            if self.config.stream:
                assistant_content = ""
                tool_calls_acc: dict[int, dict[str, str]] = {}
                async for maybe_event, stream_result in self._consume_stream(stream):
                    if maybe_event is not None:
                        yield maybe_event
                    else:
                        # Final result marker
                        assistant_content = stream_result.content
                        tool_calls_acc = stream_result.tool_calls
            else:
                choice = stream.choices[0]
                assistant_content = choice.message.content or ""
                tool_calls_acc = {}
                if choice.message.tool_calls:
                    for idx, tc in enumerate(choice.message.tool_calls):
                        tool_calls_acc[idx] = {
                            "id": tc.id or f"call_{idx}",
                            "function_name": tc.function.name,
                            "function_args": tc.function.arguments,
                        }
                # Emit TEXT_DELTA for non-streaming so consumers see the output
                if assistant_content:
                    yield self._make_event(EventType.TEXT_DELTA, delta=assistant_content)

            # Build assistant message
            assistant_msg: ChatCompletionAssistantMessageParam = {
                "role": "assistant",
                "content": assistant_content or None,
            }

            if tool_calls_acc:
                tc_parts: list[ChatCompletionMessageToolCallParam] = []
                for idx in sorted(tool_calls_acc):
                    acc = tool_calls_acc[idx]
                    tc_parts.append({
                        "id": acc["id"] or f"call_{idx}",
                        "type": "function",
                        "function": {
                            "name": acc["function_name"],
                            "arguments": acc["function_args"],
                        },
                    })
                assistant_msg["tool_calls"] = tc_parts

            messages.append(assistant_msg)
            final_output = assistant_content

            # Memory: record assistant response
            if self.memory and tool_calls_acc:
                self.memory.add_assistant(
                    content=assistant_content,
                    tool_calls=[
                        ToolCallRecord(
                            id=acc["id"] or f"call_{idx}",
                            name=acc["function_name"],
                            arguments=acc["function_args"],
                        )
                        for idx, acc in tool_calls_acc.items()
                    ],
                )
            elif self.memory:
                self.memory.add_assistant(content=assistant_content)

            if not tool_calls_acc:
                break  # Agent finished

            # Execute tool calls
            for idx in sorted(tool_calls_acc):
                acc = tool_calls_acc[idx]
                tool_name = acc["function_name"]
                call_id = acc["id"] or f"call_{idx}"

                try:
                    raw_args = json.loads(acc["function_args"])
                except json.JSONDecodeError:
                    raw_args = {}

                result_text, tool_events = await self._execute_tool(
                    tool_name=tool_name,
                    raw_args=raw_args,
                    call_id=call_id,
                    tool_map=tool_map,
                    tool_call_summaries=tool_call_summaries,
                )
                for ev in tool_events:
                    yield ev

                tool_msg: ChatCompletionToolMessageParam = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result_text,
                }
                messages.append(tool_msg)

                if self.memory:
                    self.memory.add_tool(content=result_text, tool_call_id=call_id)

        else:
            # Max turns exceeded
            yield self._make_event(
                EventType.ERROR,
                error=f"Max turns ({self.config.max_turns}) exceeded",
            )
            raise MaxTurnsExceeded(self.config.max_turns)

        # Output guardrails
        for gr in self.output_guardrails:
            result: GuardrailResult = await gr.check(
                final_output, {"agent": self}
            )
            if not result.allowed:
                yield self._make_event(
                    EventType.ERROR,
                    error=f"Guardrail [{gr.name}] rejected output: {result.message}",
                )
                raise GuardrailRejected(gr.name, result.message or "rejected")

        # Hooks: after_agent
        if self.hooks and self.hooks.after_agent:
            await self.hooks.after_agent(
                final_output, {"agent": self, "tool_calls": tool_call_summaries}
            )

        yield self._make_event(EventType.NODE_END)

    async def run_sync(self, input_text: str) -> RunResult:
        """Execute the agent synchronously and return a :class:`RunResult`.

        This collects all events into memory and returns a structured result.

        Returns:
            :class:`RunResult` with final_output, all events, and tool call summaries.
        """
        events: list[Event] = []
        final_output = ""
        tool_calls: list[dict[str, str]] = []

        async for event in self.run(input_text):
            events.append(event)
            if event.type == EventType.TEXT_DELTA and event.delta:
                final_output += event.delta
            if event.type == EventType.TOOL_RESULT:
                tool_calls.append({
                    "name": event.name or "unknown",
                    "result": event.tool_result or "",
                })

        return RunResult(
            final_output=final_output,
            events=events,
            tool_calls=tool_calls,
        )

    # ── Agent-as-Tool ────────────────────────────────────────────────────

    def as_tool(
        self,
        tool_name: str | None = None,
        tool_description: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> Tool:
        """Expose this agent as a :class:`Tool` that can be used by other agents.

        This is the "manager pattern" — the parent agent retains control
        and invokes this agent as a sub-task.

        Example::

            analyst = Agent(name="analyst", instructions="...", client=client)
            orchestrator = Agent(
                name="orchestrator",
                instructions="...",
                client=client,
                tools=[analyst.as_tool()],
            )

        Args:
            tool_name: Override the tool name (defaults to agent name).
            tool_description: Override the tool description.
            parameters: Explicit JSON Schema for the tool input.
        """
        name = tool_name or self.name
        desc = tool_description or f"Delegate to the '{self.name}' agent"

        if parameters is None:
            parameters = {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": f"Input to pass to the {self.name} agent",
                    }
                },
                "required": ["input"],
            }

        async def agent_fn(raw_args: dict[str, Any]) -> object:
            return self  # Return the agent itself — handled by _execute_tool

        return Tool(
            name=name,
            description=desc,
            parameters=parameters,
            fn=agent_fn,
        )

    # ── Handoff ──────────────────────────────────────────────────────────

    def handoff(
        self,
        target: Agent,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> Tool:
        """Create a handoff tool that delegates the conversation to another agent.

        Unlike ``as_tool()``, a handoff transfers the full conversation
        context to the target agent. The target becomes the active agent.

        Example::

            billing = Agent(name="billing", instructions="...", client=client)
            triage = Agent(
                name="triage",
                instructions="...",
                client=client,
                tools=[billing.handoff(billing)],
            )

        Args:
            target: The agent to hand off to.
            tool_name: Override the tool name.
            tool_description: Override the tool description.
        """
        name = tool_name or f"handoff_to_{target.name}"
        desc = tool_description or f"Hand off the conversation to the {target.name} agent"

        parameters: dict[str, Any] = {
            "type": "object",
            "properties": {
                "context": {
                    "type": "string",
                    "description": f"Summary or context to pass to {target.name}",
                }
            },
            "required": ["context"],
        }

        async def handoff_fn(raw_args: dict[str, Any]) -> object:
            # Return a special marker that the runner interprets
            return HandoffTarget(agent=target, context=raw_args.get("context", ""))

        return Tool(
            name=name,
            description=desc,
            parameters=parameters,
            fn=handoff_fn,
        )

    # ── Internals ────────────────────────────────────────────────────────

    @dataclass
    class _StreamResult:
        """Mutable holder for accumulated stream results."""
        content: str = ""
        tool_calls: dict[int, dict[str, str]] = field(default_factory=dict)

    async def _consume_stream(
        self,
        stream: Any,
    ) -> AsyncIterator[tuple[Event | None, _StreamResult]]:
        """Consume a streaming LLM response, yielding (event, result_so_far) tuples.

        The last yielded tuple has event=None and the final accumulated result.
        """
        result = self._StreamResult()

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            if delta.content:
                result.content += delta.content
                evt = self._make_event(EventType.TEXT_DELTA, delta=delta.content)
                yield (evt, result)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in result.tool_calls:
                        result.tool_calls[idx] = {
                            "id": "",
                            "function_name": "",
                            "function_args": "",
                        }
                    acc = result.tool_calls[idx]
                    if tc_delta.id:
                        acc["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            acc["function_name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            acc["function_args"] += tc_delta.function.arguments

        # Yield the final result marker
        yield (None, result)

    async def _execute_tool(
        self,
        tool_name: str,
        raw_args: dict[str, Any],
        call_id: str,
        tool_map: dict[str, Tool],
        tool_call_summaries: list[dict[str, str]],
    ) -> tuple[str, list[Event]]:
        """Execute a single tool call. Returns (result_text, events)."""
        events: list[Event] = []

        # Hooks: wrap_tool_call
        if self.hooks and self.hooks.wrap_tool_call:
            await self.hooks.wrap_tool_call(
                tool_name, raw_args, {"agent": self, "call_id": call_id}
            )

        events.append(self._make_event(
            EventType.TOOL_CALL,
            tool_call_id=call_id,
            name=tool_name,
            tool_args=raw_args,
        ))

        if tool_name not in tool_map:
            result_text = f"Error: unknown tool '{tool_name}'"
            events.append(self._make_event(
                EventType.ERROR,
                error=result_text,
                tool_call_id=call_id,
                name=tool_name,
            ))
            return result_text, events

        tool_def = tool_map[tool_name]

        try:
            tool_result = await tool_def.fn(raw_args)
        except NeedsHumanInput as nhi:
            events.append(self._make_event(
                EventType.HUMAN_INPUT_REQUEST,
                question=nhi.question,
                tool_call_id=call_id,
                name=tool_name,
            ))
            handler = self._resolve_human_input_handler()
            if handler is None:
                result_text = (
                    f"[Human input requested but no handler "
                    f"configured: {nhi.question}]"
                )
            else:
                result_text = await handler(nhi.question)
            events.append(self._make_event(
                EventType.TOOL_RESULT,
                tool_call_id=call_id,
                tool_result=result_text,
                name=tool_name,
            ))
            return result_text, events

        # Agent-as-tool: tool returned an Agent instance
        if isinstance(tool_result, Agent):
            sub_agent: Agent = tool_result
            sub_agent.parent = self
            sub_agent.depth = self.depth + 1
            sub_agent.reset_path_cache()

            accumulated = ""
            async for sub_event in sub_agent.run(json.dumps(raw_args)):
                events.append(sub_event)
                if sub_event.type == EventType.TEXT_DELTA and sub_event.delta:
                    accumulated += sub_event.delta

            result_text = accumulated or "[sub-agent completed]"
            preview = result_text if len(result_text) <= 200 else result_text[:200] + "..."
            events.append(self._make_event(
                EventType.TOOL_RESULT,
                tool_call_id=call_id,
                tool_result=preview,
                name=tool_name,
                meta={"is_subagent": True, "full_length": len(result_text)},
            ))
            return result_text, events

        # Handoff: tool returned a HandoffTarget
        if isinstance(tool_result, HandoffTarget):
            target: HandoffTarget = tool_result
            target.agent.parent = self
            target.agent.depth = self.depth + 1
            target.agent.reset_path_cache()

            accumulated = ""
            async for sub_event in target.agent.run(target.context):
                events.append(sub_event)
                if sub_event.type == EventType.TEXT_DELTA and sub_event.delta:
                    accumulated += sub_event.delta

            result_text = accumulated or f"[handed off to {target.agent.name}]"
            events.append(self._make_event(
                EventType.TOOL_RESULT,
                tool_call_id=call_id,
                tool_result=result_text,
                name=tool_name,
                meta={"is_handoff": True, "target": target.agent.name},
            ))
            return result_text, events

        # Regular tool result
        if isinstance(tool_result, str):
            result_text = tool_result
        else:
            result_text = str(tool_result)

        events.append(self._make_event(
            EventType.TOOL_RESULT,
            tool_call_id=call_id,
            tool_result=result_text,
            name=tool_name,
        ))
        tool_call_summaries.append({"name": tool_name, "result": result_text})
        return result_text, events

    def _resolve_human_input_handler(self) -> Callable[[str], Awaitable[str]] | None:
        """Walk up the tree to find a human_input_handler."""
        if self.human_input_handler is not None:
            return self.human_input_handler
        current: Node | None = self.parent
        while current is not None:
            if isinstance(current, Agent) and current.human_input_handler is not None:
                return current.human_input_handler
            current = current.parent
        return None


# ── Handoff target marker ────────────────────────────────────────────────


@dataclass
class HandoffTarget:
    """Internal marker returned by handoff tools.

    Carries the target agent and context to pass to it.
    """

    agent: Agent
    context: str