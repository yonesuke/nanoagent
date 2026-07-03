# -*- coding: utf-8 -*-
"""Tests for the tool system: decorator, schema inference, execution."""

from __future__ import annotations


import pytest

from nanoagent.exceptions import NeedsHumanInput
from nanoagent.tools import Tool, _infer_schema_from_function, _type_to_json_schema, ask_user, tool


# ── Schema inference ─────────────────────────────────────────────────────


class TestTypeToJsonSchema:
    def test_primitive_types(self):
        assert _type_to_json_schema(str) == {"type": "string"}
        assert _type_to_json_schema(int) == {"type": "integer"}
        assert _type_to_json_schema(float) == {"type": "number"}
        assert _type_to_json_schema(bool) == {"type": "boolean"}

    def test_list_type(self):
        schema = _type_to_json_schema(list[str])
        assert schema["type"] == "array"
        assert schema["items"] == {"type": "string"}

    def test_dict_type(self):
        assert _type_to_json_schema(dict) == {"type": "object"}

    def test_fallback(self):
        assert _type_to_json_schema(bytes) == {"type": "string"}  # unknown -> string


class TestInferSchemaFromFunction:
    @staticmethod
    async def simple_fn(x: int, y: str = "default") -> str:
        return f"{x} {y}"

    def test_infers_properties(self):
        schema = _infer_schema_from_function(self.simple_fn)
        assert schema["type"] == "object"
        assert "x" in schema["properties"]
        assert "y" in schema["properties"]
        assert schema["properties"]["x"]["type"] == "integer"
        assert schema["properties"]["y"]["type"] == "string"

    def test_required_params(self):
        schema = _infer_schema_from_function(self.simple_fn)
        assert "x" in schema["required"]
        assert "y" not in schema["required"]  # has default


# ── Tool decorator ───────────────────────────────────────────────────────


class TestToolDecorator:
    def test_basic_decorator(self):
        @tool(name="add", description="Add two numbers")
        async def add(a: int, b: int) -> str:
            return str(a + b)

        assert isinstance(add, Tool)
        assert add.name == "add"
        assert add.description == "Add two numbers"
        assert add.parameters["type"] == "object"

    def test_name_from_function(self):
        @tool()
        async def my_func(x: str) -> str:
            """My docstring."""
            return x

        assert my_func.name == "my_func"
        assert my_func.description == "My docstring."

    def test_explicit_name_override(self):
        @tool(name="custom_name")
        async def original_name(x: str) -> str:
            return x

        assert original_name.name == "custom_name"

    async def test_tool_execution(self):
        @tool()
        async def greet(name: str) -> str:
            return f"Hello, {name}!"

        result = await greet.fn({"name": "World"})
        assert result == "Hello, World!"

    def test_to_openai_tool(self):
        @tool(name="search")
        async def search(query: str) -> str:
            return f"results for {query}"

        openai_def = search.to_openai_tool()
        assert openai_def["type"] == "function"
        assert openai_def["function"]["name"] == "search"
        assert "query" in openai_def["function"]["parameters"]["properties"]


# ── Explicit Tool creation ───────────────────────────────────────────────


class TestToolExplicit:
    async def test_explicit_schema(self):
        async def handler(raw: dict) -> str:
            return raw.get("msg", "")

        t = Tool(
            name="echo",
            description="Echo back",
            parameters={
                "type": "object",
                "properties": {"msg": {"type": "string"}},
            },
            fn=handler,
        )

        assert t.name == "echo"
        result = await t.fn({"msg": "hi"})
        assert result == "hi"


# ── ask_user ─────────────────────────────────────────────────────────────


class TestAskUser:
    def test_ask_user_is_tool(self):
        assert isinstance(ask_user, Tool)
        assert ask_user.name == "ask_user"

    async def test_ask_user_raises_needs_human_input(self):
        with pytest.raises(NeedsHumanInput) as exc:
            await ask_user.fn({"question": "What do you think?"})
        assert exc.value.question == "What do you think?"
        assert exc.value.tool_name == "ask_user"