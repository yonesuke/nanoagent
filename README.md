# nanoagent

> **Nano-scale AI agent framework** — tree-based agents with streaming, tools,
> guardrails, and memory. One external dependency: `openai`.

## Philosophy

Most AI agent SDKs pull in dozens of dependencies and force you into their
abstractions. **nanoagent** takes the opposite approach:

- **One external dependency** — just `openai`
- **Tree-based execution** — agents form a tree, each with full identity
  (node id, path, depth)
- **Streaming-first** — every action emits typed events, composable with
  any UI/backend
- **Composable** — guardrails, hooks, memory, and sub-agents are all opt-in
- **~500 lines of core logic** — easy to read, fork, and extend

## Installation

```bash
uv add nanoagent
# or
pip install nanoagent
```

## Quick Start

```python
import asyncio
from openai import AsyncClient
from nanoagent import Agent, tool

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

async def main():
    async for event in agent.run("What's the weather?"):
        if event.type.value == "text_delta":
            print(event.delta, end="")

asyncio.run(main())
```

## Core Concepts

### Agent Tree
Every agent is a **Node** in a tree. The root agent has depth=0, sub-agents
have depth=1, etc. Each event carries `node_id`, `node_path`, and `depth`
so consumers can reconstruct the full execution tree.

```
orchestrator (depth=0)
├── analyst (depth=1)
│   └── fetcher (depth=2)
└── summarizer (depth=1)
```

### Streaming Events
All agent activity is emitted as typed `Event` objects:

| Event Type | When |
|---|---|
| `NODE_START` / `NODE_END` | Agent lifecycle |
| `TEXT_DELTA` | LLM text streaming |
| `TOOL_CALL` / `TOOL_RESULT` | Tool execution |
| `HUMAN_INPUT_REQUEST` | Agent asks user |
| `ERROR` | Execution error |

### Tools
Define tools with the `@tool` decorator:

```python
from nanoagent import tool

@tool(name="get_weather", description="Get weather for a city")
async def get_weather(city: str, units: str = "celsius") -> str:
    return f"{city}: sunny, 22°C"
```

### Sub-Agents (Agent-as-Tool)
A parent agent can invoke child agents as tools:

```python
analyst = Agent(name="analyst", instructions="...", client=client)
orchestrator = Agent(
    name="orchestrator",
    instructions="...",
    client=client,
    tools=[analyst.as_tool()],
)
```

### Sub-Agents (Agent-as-Tool)
A parent agent can invoke child agents as tools, retaining control
and receiving the sub-agent's output:

```python
analyst = Agent(name="analyst", instructions="...", client=client)
orchestrator = Agent(
    name="orchestrator",
    instructions="...",
    client=client,
    tools=[analyst.as_tool()],
)
```

> **Note:** True one-way handoff (where the target becomes the active
> agent and the caller never regains control) is on the roadmap.
> See `TODO(handoff)` in the source.

### Guardrails
Input/output validation hooks that can block or modify content:

```python
from nanoagent import guardrail, GuardrailResult

@guardrail(name="no_pii", runs_on="input")
async def block_pii(text: str, ctx: dict) -> GuardrailResult:
    if "password" in text.lower():
        return GuardrailResult.rejected("PII detected")
    return GuardrailResult.allowed()

agent = Agent(..., input_guardrails=[block_pii])
```

### Conversation Memory
Persistent session memory across multiple turns:

```python
from nanoagent import ConversationMemory

memory = ConversationMemory()
agent = Agent(..., memory=memory)

await agent.run("My name is Alice")
await agent.run("What's my name?")  # remembers "Alice"
```

### Human-in-the-Loop
Use the built-in `ask_user` tool to prompt for human input:

```python
from nanoagent import ask_user

agent = Agent(
    ...,
    tools=[ask_user],
    human_input_handler=lambda q: input(f"Agent asks: {q}\n> "),
)
```

### Result Collection
For non-streaming use, `run_sync()` returns a `RunResult`:

```python
result = await agent.run_sync("Analyze sales data")
print(result.final_output)
print(result.tool_calls)
```

## vs. Other SDKs

| Feature | nanoagent | OpenAI Agents SDK | LangChain |
|---|---|---|---|
| External deps | **1** (`openai`) | 10+ | 20+ |
| Core LOC | ~500 | ~5000 | ~50000 |
| Tree-based execution | ✅ | ❌ | via subgraphs |
| Streaming events | ✅ typed | ✅ | ✅ |
| Guardrails | ✅ composable | ✅ built-in | via middleware |
| Memory/Sessions | ✅ | ✅ | ✅ |
| Delegation (agent-as-tool) | ✅ | ✅ | via subgraphs |
| Handoffs (true control transfer) | TODO | ✅ | ❌ |
| Tracing | via events | ✅ built-in | via callbacks |

## License

MIT