# -*- coding: utf-8 -*-
"""Model Context Protocol (MCP) tool integration for nanoagent.

MCP is an open protocol that standardizes how AI models connect to
external tools and data sources. An MCP server exposes tools that
any MCP-compatible client (like nanoagent) can discover and use.

This module provides:
- ``MCPServer`` base class for connecting to MCP servers
- ``MCPServerStdio`` for subprocess-based servers (e.g., via npx)
- ``MCPServerSse`` for HTTP SSE-based servers
- ``MCPTool`` adapter that converts MCP tools to nanoagent Tools
- ``mcp_tools()`` helper to load tools from multiple servers

Inspired by OpenAI Agents SDK's ``agents.mcp`` module.

Prerequisites:
    ``pip install mcp``  (the official ``mcp`` Python package)

Usage::

    from nanoagent.contrib.mcp import MCPServerStdio, mcp_tools

    async with MCPServerStdio(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    ) as server:
        tools = await mcp_tools([server])
        agent = Agent(
            name="assistant",
            instructions="Use the filesystem tools.",
            client=client,
            tools=[*tools, *my_other_tools],
        )
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

# ── MCPTool -- adapter from MCP tool to nanoagent Tool ───────────────────


@dataclass
class MCPTool:
    """Wraps an MCP tool definition as a nanoagent-compatible Tool.

    This is a lightweight wrapper. The actual MCP client library
    handles the protocol details (JSON-RPC over stdio/SSE).
    """

    name: str
    description: str
    parameters: dict[str, Any]
    _invoke: Callable[[dict[str, Any]], Awaitable[object]]

    def to_nanoagent_tool(self):
        """Convert to a nanoagent :class:`Tool`."""
        from nanoagent.tools import Tool

        return Tool(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            fn=self._invoke,
        )


# ── MCPServer base ───────────────────────────────────────────────────────


class MCPServer:
    """Base class for MCP server connections.

    Subclasses implement ``connect()`` and ``cleanup()`` for
    specific transport protocols (stdio, SSE, streamable HTTP).

    Usage::

        server = MCPServerStdio(name="my-server", command="python", args=["server.py"])
        async with server:
            tools = await server.list_tools()
    """

    name: str

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args: object):
        await self.cleanup()

    async def connect(self) -> None:
        """Establish the connection to the MCP server."""
        raise NotImplementedError

    async def cleanup(self) -> None:
        """Close the connection and release resources."""
        raise NotImplementedError

    async def list_tools(self) -> list[MCPTool]:
        """Discover tools exposed by this MCP server."""
        raise NotImplementedError

    async def invoke_tool(self, tool_name: str, arguments: dict[str, Any]) -> object:
        """Invoke a specific MCP tool with arguments."""
        raise NotImplementedError


# ── MCPServerStdio ───────────────────────────────────────────────────────


class MCPServerStdio(MCPServer):
    """Connect to an MCP server via subprocess stdio.

    Uses the ``mcp`` library's stdio client under the hood.

    Example::

        async with MCPServerStdio(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/data"],
        ) as server:
            tools = await server.list_tools()
    """

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cache_tools: bool = True,
    ):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env
        self.cache_tools = cache_tools
        self._session: Any = None
        self._tools_cache: list[MCPTool] | None = None

    async def connect(self) -> None:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            raise ImportError(
                "MCP support requires the 'mcp' package. Install with: pip install mcp"
            )

        params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )

        self._stdio_context = stdio_client(params)
        transport = await self._stdio_context.__aenter__()
        self._session = ClientSession(transport[0], transport[1])
        await self._session.initialize()

    async def cleanup(self) -> None:
        if hasattr(self, "_stdio_context"):
            await self._stdio_context.__aexit__(None, None, None)
        self._session = None
        self._tools_cache = None

    async def list_tools(self) -> list[MCPTool]:
        if self.cache_tools and self._tools_cache is not None:
            return self._tools_cache

        if self._session is None:
            raise RuntimeError("Not connected. Use 'async with server:' first.")

        result = await self._session.list_tools()
        tools: list[MCPTool] = []

        for t in result.tools:
            mcp_tool = MCPTool(
                name=t.name,
                description=t.description or "",
                parameters=t.inputSchema or {"type": "object", "properties": {}},
                _invoke=lambda args, tn=t.name: self._invoke_inner(tn, args),
            )
            tools.append(mcp_tool)

        if self.cache_tools:
            self._tools_cache = tools
        return tools

    async def invoke_tool(self, tool_name: str, arguments: dict[str, Any]) -> object:
        return await self._invoke_inner(tool_name, arguments)

    async def _invoke_inner(self, tool_name: str, arguments: dict[str, Any]) -> object:
        if self._session is None:
            raise RuntimeError("Not connected")
        result = await self._session.call_tool(tool_name, arguments)
        # Extract text content from MCP result
        texts: list[str] = []
        for item in result.content:
            if hasattr(item, "text"):
                texts.append(item.text)
            elif isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
        return "\n".join(texts) if texts else json.dumps(result.content)


# ── MCPServerSse ─────────────────────────────────────────────────────────


class MCPServerSse(MCPServer):
    """Connect to an MCP server via HTTP SSE transport.

    Example::

        async with MCPServerSse(
            name="remote-tools",
            url="http://localhost:8000/sse",
            headers={"Authorization": "Bearer token123"},
        ) as server:
            tools = await server.list_tools()
    """

    def __init__(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
        cache_tools: bool = True,
    ):
        self.name = name
        self.url = url
        self.headers = headers or {}
        self.cache_tools = cache_tools
        self._session: Any = None
        self._tools_cache: list[MCPTool] | None = None

    async def connect(self) -> None:
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
        except ImportError:
            raise ImportError(
                "MCP support requires the 'mcp' package. Install with: pip install mcp"
            )

        self._sse_context = sse_client(self.url, headers=self.headers)
        transport = await self._sse_context.__aenter__()
        self._session = ClientSession(transport[0], transport[1])
        await self._session.initialize()

    async def cleanup(self) -> None:
        if hasattr(self, "_sse_context"):
            await self._sse_context.__aexit__(None, None, None)
        self._session = None
        self._tools_cache = None

    async def list_tools(self) -> list[MCPTool]:
        if self.cache_tools and self._tools_cache is not None:
            return self._tools_cache

        if self._session is None:
            raise RuntimeError("Not connected.")

        result = await self._session.list_tools()
        tools: list[MCPTool] = []

        for t in result.tools:
            mcp_tool = MCPTool(
                name=t.name,
                description=t.description or "",
                parameters=t.inputSchema or {"type": "object", "properties": {}},
                _invoke=lambda args, tn=t.name: self._invoke_inner(tn, args),
            )
            tools.append(mcp_tool)

        if self.cache_tools:
            self._tools_cache = tools
        return tools

    async def invoke_tool(self, tool_name: str, arguments: dict[str, Any]) -> object:
        return await self._invoke_inner(tool_name, arguments)

    async def _invoke_inner(self, tool_name: str, arguments: dict[str, Any]) -> object:
        if self._session is None:
            raise RuntimeError("Not connected")
        result = await self._session.call_tool(tool_name, arguments)
        texts: list[str] = []
        for item in result.content:
            if hasattr(item, "text"):
                texts.append(item.text)
            elif isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
        return "\n".join(texts) if texts else json.dumps(result.content)


# ── Helper: load tools from multiple MCP servers ─────────────────────────


async def mcp_tools(
    servers: list[MCPServer],
    *,
    tool_filter: Callable[[str], bool] | None = None,
) -> list[Any]:
    """Discover tools from a list of MCP servers.

    Each server's tools are prefixed with the server name to avoid
    name collisions: ``server_name__tool_name``.

    Args:
        servers: Connected MCP server instances.
        tool_filter: Optional predicate to filter tools by name.

    Returns:
        List of nanoagent ``Tool`` objects.
    """
    from nanoagent.tools import Tool

    all_tools: list[Tool] = []
    seen: set[str] = set()

    for server in servers:
        mcp_tool_list = await server.list_tools()
        for mt in mcp_tool_list:
            # Prefix with server name to avoid collisions
            full_name = f"{server.name}__{mt.name}"
            if tool_filter and not tool_filter(full_name):
                continue
            if full_name in seen:
                continue
            seen.add(full_name)

            nt = mt.to_nanoagent_tool()
            nt.name = full_name
            nt.description = f"[MCP:{server.name}] {mt.description}"
            all_tools.append(nt)

    return all_tools  # type: ignore[return-value]