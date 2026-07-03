# -*- coding: utf-8 -*-
"""Integration tests for Agent using FakeLLM.

Covers: text streaming, tool calls, sub-agents, guardrails,
human-in-the-loop, handoffs, memory, RunConfig, and error handling.

Pattern: pre-program FakeLLM responses, run the agent, assert events.
Inspired by OpenAI Agents SDK's FakeModel testing pattern.
"""

from __future__ import annotations

import pytest

from nanoagent import Agent, EventType, RunConfig, RunResult
from nanoagent.events import Event
from nanoagent.exceptions import GuardrailRejected, MaxTurnsExceeded
from nanoagent.guardrails import Guardrail, GuardrailResult
from nanoagent.memory import ConversationMemory
from nanoagent.tools import Tool, ask_user, tool


# -- Helper: collect events --


async def _collect(agent: Agent, prompt: str) -> tuple[str, list[Event]]:
    """Run agent and collect all events + final text output."""
    events: list[Event] = []
    text = ""
    async for ev in agent.run(prompt):
        events.append(ev)
        if ev.type == EventType.TEXT_DELTA and ev.delta:
            text += ev.delta
    return text, events


# -- Text generation --


class TestAgentTextGeneration:
    async def test_simple_text_response(self, fake_llm, client):
        fake_llm.add_text_response("Hello, world!")

        agent = Agent(name="bot", instructions="Be helpful.", client=client)
        text, events = await _collect(agent, "Hi")

        assert text == "Hello, world!"
        assert events[0].type == EventType.NODE_START
        assert events[-1].type == EventType.NODE_END

    async def test_multi_turn_text(self, fake_llm, client):
        """Agent makes multiple LLM calls via tool-call loop."""
        fake_llm.add_tool_call("think", {"thought": "Let me analyze..."})
        fake_llm.add_text_response("Here is my final answer.")

        @tool(name="think")
        async def think(thought: str) -> str:
            return f"Thought: {thought}"

        agent = Agent(
            name="bot",
            instructions="Think before answering.",
            client=client,
            tools=[think],
        )
        text, _ = await _collect(agent, "Complex question")
        assert "Here is my final answer." in text

    async def test_node_identity_in_events(self, fake_llm, client):
        fake_llm.add_text_response("OK")

        agent = Agent(name="test_agent", instructions="...", client=client)
        _, events = await _collect(agent, "input")

        for ev in events:
            assert ev.node_id is not None
            assert len(ev.node_id) == 12
            assert ev.node_path == ["test_agent"]
            assert ev.depth == 0


# -- Tool calls --


class TestAgentToolCalls:
    async def test_single_tool_call(self, fake_llm, client):
        fake_llm.add_tool_call("echo", {"msg": "hello"})
        fake_llm.add_text_response("I called echo and got: hello echo")

        @tool(name="echo")
        async def echo(msg: str) -> str:
            return f"{msg} echo"

        agent = Agent(name="bot", instructions="Use echo tool.", client=client, tools=[echo])
        text, events = await _collect(agent, "Echo hello")

        tool_events = [e for e in events if e.type in (EventType.TOOL_CALL, EventType.TOOL_RESULT)]
        assert len(tool_events) == 2
        assert tool_events[0].type == EventType.TOOL_CALL
        assert tool_events[0].name == "echo"
        assert tool_events[1].type == EventType.TOOL_RESULT
        assert tool_events[1].tool_result == "hello echo"

    async def test_multiple_tool_calls(self, fake_llm, client):
        fake_llm.add_tool_call("add", {"a": 1, "b": 2})
        fake_llm.add_tool_call("add", {"a": 3, "b": 4})
        fake_llm.add_text_response("Results: 3 and 7")

        @tool(name="add")
        async def add(a: int, b: int) -> str:
            return str(a + b)

        agent = Agent(name="bot", instructions="Use add tool.", client=client, tools=[add])
        text, events = await _collect(agent, "Add 1+2 and 3+4")

        results = [e for e in events if e.type == EventType.TOOL_RESULT]
        assert len(results) == 2
        assert results[0].tool_result == "3"
        assert results[1].tool_result == "7"

    async def test_unknown_tool(self, fake_llm, client):
        fake_llm.add_tool_call("nonexistent", {"x": 1})
        fake_llm.add_text_response("I tried an unknown tool.")

        agent = Agent(name="bot", instructions="...", client=client, tools=[])
        _, events = await _collect(agent, "Use nonexistent tool")

        errors = [e for e in events if e.type == EventType.ERROR]
        assert len(errors) >= 1
        assert "unknown tool" in (errors[0].error or "").lower()


# -- Sub-agents (Agent-as-Tool) --


class TestAgentAsTool:
    async def test_sub_agent_delegation(self, fake_llm, client):
        """Parent delegates to sub-agent; sub-agent runs its own LLM calls."""
        # Queue: (1) parent decides to call analyst tool
        fake_llm.add_tool_call("analyst", {"input": "Analyze sales"})
        # (2) sub-agent's LLM response (inside _execute_tool)
        fake_llm.add_text_response("Sales: Q3 $1.2M, up 15%")
        # (3) parent receives result and does final turn
        fake_llm.add_text_response("Summary: strong Q3 with 15% growth")

        analyst = Agent(name="analyst", instructions="Analyze concisely.", client=client)
        parent = Agent(
            name="orchestrator",
            instructions="Delegate to analyst then summarize.",
            client=client,
            tools=[analyst],
        )
        text, events = await _collect(parent, "Analyze our sales")

        # Both depths present
        depths = {e.depth for e in events}
        assert 0 in depths
        assert 1 in depths

        # Sub-agent lifecycle events
        sub_starts = [e for e in events if e.type == EventType.NODE_START and e.name == "analyst"]
        assert len(sub_starts) == 1

    async def test_deep_nesting(self, fake_llm, client):
        """Three-level nesting: root -> mid -> leaf."""
        fake_llm.add_tool_call("mid", {"input": "task"})  # root calls mid
        fake_llm.add_tool_call("leaf", {"input": "subtask"})  # mid calls leaf
        fake_llm.add_text_response("Leaf analysis done")  # leaf
        fake_llm.add_text_response("Mid summary done")  # mid
        fake_llm.add_text_response("Final root summary")  # root

        leaf = Agent(name="leaf", instructions="Deep analysis.", client=client)
        mid = Agent(name="mid", instructions="Coordinate.", client=client, tools=[leaf])
        root = Agent(name="root", instructions="Orchestrate.", client=client, tools=[mid])

        text, events = await _collect(root, "Deep analysis task")

        depths = {e.depth for e in events}
        assert 0 in depths
        assert 1 in depths
        assert 2 in depths

        names_seen = {e.name for e in events if e.type == EventType.NODE_START}
        assert names_seen == {"root", "mid", "leaf"}

    async def test_as_tool_explicit(self, client):
        """Agent.as_tool() returns a valid Tool that returns the Agent."""
        sub = Agent(name="worker", instructions="Do work.", client=client)
        tool_def = sub.as_tool()

        assert isinstance(tool_def, Tool)
        assert tool_def.name == "worker"
        assert "input" in str(tool_def.parameters)

        result = await tool_def.fn({"input": "task"})
        assert isinstance(result, Agent)
        assert result.name == "worker"


# -- Human-in-the-loop --


class TestHumanInTheLoop:
    async def test_ask_user_with_handler(self, fake_llm, client):
        fake_llm.add_tool_call("ask_user", {"question": "What metrics?"})
        fake_llm.add_text_response("Based on sales metrics, here is the analysis.")

        async def handler(q: str) -> str:
            return "sales metrics"

        agent = Agent(
            name="bot",
            instructions="Ask user if unclear.",
            client=client,
            tools=[ask_user],
            human_input_handler=handler,
        )
        _, events = await _collect(agent, "Analyze performance")

        human_events = [e for e in events if e.type == EventType.HUMAN_INPUT_REQUEST]
        assert len(human_events) == 1
        assert human_events[0].question == "What metrics?"

    async def test_ask_user_without_handler(self, fake_llm, client):
        """Without handler, produces placeholder text without crashing."""
        fake_llm.add_tool_call("ask_user", {"question": "Clarify?"})
        fake_llm.add_text_response("OK, proceeding anyway.")

        agent = Agent(name="bot", instructions="Ask user if needed.", client=client, tools=[ask_user])
        _, events = await _collect(agent, "Do something")

        human_events = [e for e in events if e.type == EventType.HUMAN_INPUT_REQUEST]
        assert len(human_events) == 1

    async def test_handler_inheritance(self, fake_llm, client):
        """Sub-agent inherits parent's human_input_handler."""
        fake_llm.add_tool_call("worker", {"input": "task"})
        fake_llm.add_tool_call("ask_user", {"question": "Which format?"})
        fake_llm.add_text_response("Using JSON format")
        fake_llm.add_text_response("Summary: formatted as JSON")

        async def handler(q: str) -> str:
            return "JSON"

        worker = Agent(name="worker", instructions="Ask then format.", client=client, tools=[ask_user])
        parent = Agent(
            name="parent",
            instructions="Delegate to worker.",
            client=client,
            tools=[worker],
            human_input_handler=handler,
        )
        _, events = await _collect(parent, "Format my data")

        human = [e for e in events if e.type == EventType.HUMAN_INPUT_REQUEST]
        assert len(human) >= 1


# -- Guardrails --


class TestAgentGuardrails:
    async def test_input_guardrail_blocks(self, fake_llm, client):
        async def block_bad(text: str, ctx: dict) -> GuardrailResult:
            if "forbidden" in text.lower():
                return GuardrailResult.rejected("contains forbidden content")
            return GuardrailResult.allowed()

        gr = Guardrail(name="no_forbidden", check=block_bad, runs_on="input")
        agent = Agent(name="bot", instructions="Be helpful.", client=client, input_guardrails=[gr])

        with pytest.raises(GuardrailRejected) as exc:
            async for _ in agent.run("This is FORbidden content"):
                pass
        assert "no_forbidden" in str(exc.value)
        assert "forbidden" in str(exc.value).lower()

    async def test_input_guardrail_passes(self, fake_llm, client):
        fake_llm.add_text_response("All good!")

        async def always_ok(text: str, ctx: dict) -> GuardrailResult:
            return GuardrailResult.allowed()

        gr = Guardrail(name="check", check=always_ok, runs_on="input")
        agent = Agent(name="bot", instructions="Be helpful.", client=client, input_guardrails=[gr])

        text, _ = await _collect(agent, "Normal input")
        assert text == "All good!"

    async def test_output_guardrail_blocks(self, fake_llm, client):
        fake_llm.add_text_response("I will reveal SECRET information!")

        async def block_secret(text: str, ctx: dict) -> GuardrailResult:
            if "SECRET" in text:
                return GuardrailResult.rejected("output contains secret")
            return GuardrailResult.allowed()

        gr = Guardrail(name="no_secrets", check=block_secret, runs_on="output")
        agent = Agent(name="bot", instructions="Be helpful.", client=client, output_guardrails=[gr])

        with pytest.raises(GuardrailRejected):
            async for _ in agent.run("Tell me a secret"):
                pass


# -- Memory --


class TestAgentMemory:
    async def test_memory_across_turns(self, fake_llm, client):
        memory = ConversationMemory()

        # Turn 1
        fake_llm.add_text_response("Nice to meet you, Alice!")
        agent = Agent(name="bot", instructions="Remember user info.", client=client, memory=memory)
        text1, _ = await _collect(agent, "My name is Alice")
        assert "Alice" in text1

        # Turn 2 — memory carries previous context
        fake_llm.add_text_response("Your name is Alice!")
        text2, _ = await _collect(agent, "What's my name?")
        assert "Alice" in text2
        assert len(memory.messages) >= 2

    async def test_memory_with_tools(self, fake_llm, client):
        memory = ConversationMemory()
        fake_llm.add_tool_call("store", {"key": "name", "value": "Alice"})
        fake_llm.add_text_response("Stored!")

        @tool(name="store")
        async def store(key: str, value: str) -> str:
            return f"Stored {key}={value}"

        agent = Agent(name="bot", instructions="Store user data.", client=client, tools=[store], memory=memory)
        await _collect(agent, "My name is Alice")

        tool_msgs = [m for m in memory.messages if m.role == "tool"]
        assert len(tool_msgs) >= 1


# -- RunConfig --


class TestRunConfig:
    async def test_max_turns_exceeded(self, fake_llm, client):
        """Agent loops on tool calls until max_turns exceeded."""
        for _ in range(5):
            fake_llm.add_tool_call("noop", {})

        @tool(name="noop")
        async def noop() -> str:
            return "done"

        agent = Agent(
            name="bot",
            instructions="Call noop repeatedly.",
            client=client,
            config=RunConfig(max_turns=2),
            tools=[noop],
        )

        with pytest.raises(MaxTurnsExceeded):
            async for _ in agent.run("Start"):
                pass

    async def test_non_streaming_mode(self, fake_llm, client):
        fake_llm.add_text_response("Non-streamed response")

        agent = Agent(name="bot", instructions="...", client=client, config=RunConfig(stream=False))
        result = await agent.run_sync("Hi")
        assert result.final_output == "Non-streamed response"

    async def test_temperature_and_top_p(self, fake_llm, client):
        fake_llm.add_text_response("OK")
        agent = Agent(name="bot", instructions="...", client=client, config=RunConfig(temperature=0.0, top_p=0.5))
        await _collect(agent, "Hi")
        # Smoke test: just verify no crash


# -- RunResult (sync API) --


class TestRunSync:
    async def test_run_sync(self, fake_llm, client):
        fake_llm.add_tool_call("search", {"q": "test"})
        fake_llm.add_text_response("Found results.")

        @tool(name="search")
        async def search(q: str) -> str:
            return f"Results for {q}"

        agent = Agent(name="bot", instructions="Search and report.", client=client, tools=[search])
        result: RunResult = await agent.run_sync("Search for test")

        assert isinstance(result, RunResult)
        assert "Found results" in result.final_output
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "search"
        assert len(result.events) > 0


# -- Reasoning / thinking streaming --


class TestAgentReasoning:
    async def test_reasoning_before_text(self, fake_llm, client):
        """o-series pattern: reasoning_content streams before content."""
        fake_llm.add_reasoning_response(
            reasoning="Let me think... The answer is 42.",
            text="The answer is 42.",
        )

        agent = Agent(name="bot", instructions="Think before answering.", client=client)
        _, events = await _collect(agent, "Life universe everything?")

        types = [e.type for e in events if e.type in (
            EventType.REASONING_START, EventType.REASONING_DELTA,
            EventType.REASONING_END, EventType.TEXT_START,
            EventType.TEXT_DELTA, EventType.TEXT_END,
        )]
        # Should be: REASONING_START, REASONING_DELTA, REASONING_END,
        #            TEXT_START, TEXT_DELTA, TEXT_END
        assert EventType.REASONING_START in types
        assert EventType.REASONING_DELTA in types
        assert EventType.REASONING_END in types
        assert EventType.TEXT_START in types
        assert EventType.TEXT_DELTA in types
        assert EventType.TEXT_END in types
        # Reasoning before text
        reason_idx = types.index(EventType.REASONING_START)
        text_idx = types.index(EventType.TEXT_START)
        assert reason_idx < text_idx

    async def test_text_only_bookends(self, fake_llm, client):
        """Plain text response gets TEXT_START/TEXT_END bookends."""
        fake_llm.add_text_response("Hello, world!")

        agent = Agent(name="bot", instructions="Be helpful.", client=client)
        full_text, events = await _collect(agent, "Hi")

        types = [e.type for e in events]
        assert EventType.TEXT_START in types
        assert EventType.TEXT_DELTA in types
        assert EventType.TEXT_END in types
        assert full_text == "Hello, world!"

    async def test_run_finish_event(self, fake_llm, client):
        """Every run ends with a RUN_FINISH event before NODE_END."""
        fake_llm.add_text_response("OK")

        agent = Agent(name="bot", instructions="Be helpful.", client=client)
        _, events = await _collect(agent, "Hi")

        # Last two events: RUN_FINISH, NODE_END
        assert events[-2].type == EventType.RUN_FINISH
        assert events[-1].type == EventType.NODE_END
        assert events[-2].finish_reason == "stop"

    async def test_text_start_end_order(self, fake_llm, client):
        """TEXT_START → TEXT_DELTA → TEXT_END in order."""
        fake_llm.add_text_response("ABC")

        agent = Agent(name="bot", instructions="...", client=client)
        _, events = await _collect(agent, "Hi")

        text_types = [e.type for e in events
                      if e.type in (EventType.TEXT_START, EventType.TEXT_DELTA, EventType.TEXT_END)]
        assert text_types == [EventType.TEXT_START, EventType.TEXT_DELTA, EventType.TEXT_END]

    async def test_reasoning_with_tool_call(self, fake_llm, client):
        """Reasoning → tool call pattern: reasoning ends before tool call."""
        from conftest import _tool_call_delta

        # First turn: reasoning then tool call
        chunks = [
            (None, "Thinking about which tool to use...", None),  # reasoning
            (None, None, [_tool_call_delta(0, id_="c1", name="search", args='{"q":"test"}')]),
        ]
        fake_llm.add_interleaved_response(chunks)
        # Second turn: text response
        fake_llm.add_text_response("Found results.")

        @tool(name="search")
        async def search(q: str) -> str:
            return f"Results for {q}"

        agent = Agent(name="bot", instructions="Search and report.", client=client, tools=[search])
        _, events = await _collect(agent, "Search for test")

        reason_types = [e.type for e in events if e.type in (
            EventType.REASONING_START, EventType.REASONING_DELTA, EventType.REASONING_END
        )]
        assert len(reason_types) >= 3  # start, delta, end
        # REASONING_END comes before TOOL_CALL
        reason_end_indices = [i for i, e in enumerate(events) if e.type == EventType.REASONING_END]
        tool_call_idx = next(i for i, e in enumerate(events) if e.type == EventType.TOOL_CALL)
        assert reason_end_indices[0] < tool_call_idx

    async def test_non_streaming_bookends(self, fake_llm, client):
        """Non-streaming mode also emits TEXT_START/TEXT_END."""
        fake_llm.add_text_response("Non-streamed")

        agent = Agent(name="bot", instructions="...", client=client, config=RunConfig(stream=False))
        _, events = await _collect(agent, "Hi")

        types = [e.type for e in events]
        assert EventType.TEXT_START in types
        assert EventType.TEXT_END in types


# -- Error handling --


class TestAgentErrors:
    async def test_tool_raises_exception(self, fake_llm, client):
        fake_llm.add_tool_call("crash", {})
        fake_llm.add_text_response("Recovered.")  # won't be reached

        @tool(name="crash")
        async def crash() -> str:
            raise ValueError("Boom!")

        agent = Agent(name="bot", instructions="Call crash.", client=client, tools=[crash])

        with pytest.raises(ValueError, match="Boom!"):
            async for _ in agent.run("Crash it"):
                pass