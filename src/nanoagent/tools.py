# -*- coding: utf-8 -*-
"""Tool system for defining agent-callable functions.

Tools are the primary way agents interact with the outside world:
fetching data, calling APIs, delegating to sub-agents, or requesting
human input.

Two patterns are supported:

1. Decorator-based (recommended)::

    from nanoagent import tool

    @tool(name="search", description="Search the web")
    async def search(query: str) -> str:
        return f"Results for '{query}'"

2. Explicit schema -- for complex parameter schemas::

    from nanoagent import Tool

    my_tool = Tool(
        name="complex_action",
        description="Do something complex",
        parameters={...},  # JSON Schema dict
        fn=my_async_handler,
    )
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from openai.types.chat import ChatCompletionToolParam

from .exceptions import NeedsHumanInput

# -- JSON Schema helpers for type inference --

_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


def _type_to_json_schema(py_type: type) -> dict[str, Any]:
    """Convert a Python type to a simple JSON Schema type dict."""
    if py_type in _TYPE_MAP:
        return {"type": _TYPE_MAP[py_type]}
    # Handle generic origins like list[str], dict[str, int]
    origin = getattr(py_type, "__origin__", None)
    if origin is list:
        args = getattr(py_type, "__args__", ())
        schema: dict[str, Any] = {"type": "array"}
        if args:
            schema["items"] = _type_to_json_schema(args[0])
        return schema
    if origin is dict:
        return {"type": "object"}
    # Fallback
    return {"type": "string"}


def _infer_schema_from_function(
    fn: Callable[..., Any], skip_first: bool = False
) -> dict[str, Any]:
    """Generate a JSON Schema from a function's type hints."""
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    if skip_first:
        params = params[1:]  # skip 'self' for methods

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param in params:
        if param.name in hints:
            py_type = hints[param.name]
        elif param.annotation is not inspect.Parameter.empty:
            py_type = param.annotation
        else:
            py_type = str  # default

        properties[param.name] = _type_to_json_schema(py_type)

        if param.default is inspect.Parameter.empty:
            required.append(param.name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


# -- Tool definition --


@dataclass
class Tool:
    """Definition of a tool callable by an agent.

    Attributes:
        name: Unique tool name (used in function calling).
        description: What the tool does (shown to the LLM).
        parameters: JSON Schema describing the tool's input.
        fn: Async callable that executes the tool.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[[dict[str, Any]], Awaitable[object]]

    def to_openai_tool(self) -> ChatCompletionToolParam:
        """Convert to an OpenAI tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# -- Decorator --


def tool(
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorator to create a :class:`Tool` from an async function.

    The function's type hints are used to auto-generate a JSON Schema.
    For complex schemas, pass ``parameters`` explicitly.

    Example::

        @tool(name="get_weather", description="Get current weather")
        async def get_weather(city: str, units: str = "celsius") -> str:
            return f"Weather in {city}: sunny, 22 deg C"

    Args:
        name: Tool name. Defaults to the function name.
        description: Tool description. Defaults to the first line of docstring.
        parameters: Explicit JSON Schema. If None, inferred from type hints.
    """
    def decorator(fn: Callable[..., Any]) -> Tool:
        tool_name = name or fn.__name__
        tool_desc = description or (fn.__doc__ or "").strip().split("\n")[0]
        schema = parameters or _infer_schema_from_function(fn)

        async def wrapped(raw_args: dict[str, Any]) -> object:
            # Map raw args dict to function kwargs
            sig = inspect.signature(fn)
            kwargs: dict[str, Any] = {}
            for param_name, param in sig.parameters.items():
                if param_name in raw_args:
                    kwargs[param_name] = raw_args[param_name]
                elif param.default is not inspect.Parameter.empty:
                    kwargs[param_name] = param.default
            return await fn(**kwargs)

        return Tool(name=tool_name, description=tool_desc, parameters=schema, fn=wrapped)

    return decorator


# -- Built-in: ask_user --


async def _ask_user_impl(question: str) -> str:
    raise NeedsHumanInput(question=question, tool_name="ask_user")


ask_user: Tool = Tool(
    name="ask_user",
    description=(
        "Ask the user a clarifying question when you need more information "
        "to proceed. Use this when the request is ambiguous or you need to "
        "confirm understanding before taking action."
    ),
    parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the user",
            }
        },
        "required": ["question"],
    },
    fn=lambda raw: _ask_user_impl(raw.get("question", "")),
)
"""Built-in tool for requesting human input mid-execution."""