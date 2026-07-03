# -*- coding: utf-8 -*-
"""Permission and approval system for tool execution.

Adds a safety layer: before executing a tool, the permission system
checks whether it should be allowed, blocked, or requires confirmation.

Inspired by pi's ``permission-gate`` and ``confirm-destructive`` extensions.

Usage::

    from nanoagent.contrib.permissions import (
        PermissionPolicy,
        AllowAll,
        BlockPatterns,
        ConfirmDestructive,
        permission_gate,
    )

    # Block .env files, confirm destructive commands
    policy = BlockPatterns(["rm ", "DROP TABLE", ".env"]) | ConfirmDestructive()

    @tool(name="bash")
    async def bash(command: str) -> str:
        ...

    # Wrap a tool with permission checking
    safe_tool = permission_gate(bash, policy)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


# ── Permission result ────────────────────────────────────────────────────


@dataclass
class PermissionDecision:
    """Result of a permission check.

    Attributes:
        allowed: ``True`` if the tool call is permitted.
        reason: Human-readable explanation.
        require_confirmation: If ``True``, the caller should prompt for
            user confirmation before proceeding.
    """

    allowed: bool
    reason: str = ""
    require_confirmation: bool = False

    @classmethod
    def allow(cls, reason: str = "") -> PermissionDecision:
        return cls(allowed=True, reason=reason)

    @classmethod
    def block(cls, reason: str) -> PermissionDecision:
        return cls(allowed=False, reason=reason)

    @classmethod
    def confirm(cls, reason: str = "This action requires confirmation") -> PermissionDecision:
        return cls(allowed=True, reason=reason, require_confirmation=True)


# ── Policy protocol ──────────────────────────────────────────────────────


@dataclass
class PermissionPolicy:
    """A composable permission policy.

    Policies can be combined with ``|`` (OR): the first blocking
    decision wins, and confirmations propagate.

    Example::

        policy = BlockPatterns(["rm -rf"]) | ConfirmDestructive()

    Attributes:
        check: Async function ``(tool_name, args) -> PermissionDecision``.
        name: Human-readable policy name.
    """

    name: str
    check: Callable[[str, dict[str, Any]], Awaitable[PermissionDecision]]

    def __or__(self, other: PermissionPolicy) -> PermissionPolicy:
        """Combine two policies. First block wins, then confirmations chain."""
        p1 = self
        p2 = other

        async def combined(tool_name: str, args: dict[str, Any]) -> PermissionDecision:
            r1 = await p1.check(tool_name, args)
            if not r1.allowed:
                return r1
            r2 = await p2.check(tool_name, args)
            if not r2.allowed:
                return r2
            if r1.require_confirmation or r2.require_confirmation:
                return PermissionDecision.confirm(
                    r2.reason or r1.reason or "Action requires confirmation"
                )
            return PermissionDecision.allow()

        return PermissionPolicy(
            name=f"{self.name} | {other.name}",
            check=combined,
        )


# ── Built-in policies ────────────────────────────────────────────────────


class AllowAll(PermissionPolicy):
    """Permit all tool calls unconditionally."""

    def __init__(self):
        async def allow(_name: str, _args: dict[str, Any]) -> PermissionDecision:
            return PermissionDecision.allow()

        super().__init__(name="AllowAll", check=allow)


class BlockPatterns(PermissionPolicy):
    """Block tool calls whose arguments match any of the given patterns.

    Checks all string values in the tool arguments.

    Example::

        BlockPatterns(["rm -rf", "DROP TABLE", ".env"])
    """

    def __init__(self, patterns: list[str], name: str = "BlockPatterns"):
        self.patterns = patterns

        async def check(_name: str, args: dict[str, Any]) -> PermissionDecision:
            for key, value in args.items():
                if isinstance(value, str):
                    for pat in patterns:
                        if pat in value:
                            return PermissionDecision.block(
                                f"Argument '{key}' matches blocked pattern '{pat}'"
                            )
            return PermissionDecision.allow()

        super().__init__(name=name, check=check)


class ConfirmDestructive(PermissionPolicy):
    """Require confirmation for potentially destructive operations.

    Matches common destructive patterns in tool arguments.
    """

    DESTRUCTIVE_PATTERNS = [
        "rm ", "rm -rf", "rmdir",
        "DROP ", "DELETE ", "TRUNCATE ",
        "format ", "mkfs.",
        "chmod 777",
        "> /dev/",
        "shutdown", "reboot",
    ]

    def __init__(self, extra_patterns: list[str] | None = None):
        patterns = list(self.DESTRUCTIVE_PATTERNS)
        if extra_patterns:
            patterns.extend(extra_patterns)

        async def check(_name: str, args: dict[str, Any]) -> PermissionDecision:
            for key, value in args.items():
                if isinstance(value, str):
                    for pat in patterns:
                        if pat.lower() in value.lower():
                            return PermissionDecision.confirm(
                                f"Destructive operation detected: '{value[:80]}'"
                            )
            return PermissionDecision.allow()

        super().__init__(name="ConfirmDestructive", check=check)


class RequireApproval(PermissionPolicy):
    """Require external approval (via callback) for all tool calls.

    Example::

        RequireApproval(lambda name, args: input(f"Allow {name}({args})? [y/N] ").lower() == "y")
    """

    def __init__(
        self,
        approver: Callable[[str, dict[str, Any]], Awaitable[bool]],
        name: str = "RequireApproval",
    ):
        async def check(tool_name: str, args: dict[str, Any]) -> PermissionDecision:
            approved = await approver(tool_name, args)
            if approved:
                return PermissionDecision.allow()
            return PermissionDecision.block("Rejected by approver")

        super().__init__(name=name, check=check)


# ── Tool wrapper ─────────────────────────────────────────────────────────


def permission_gate(tool: Any, policy: PermissionPolicy) -> Any:
    """Wrap a nanoagent Tool with permission checking.

    Returns a new Tool that checks permissions before executing.

    Args:
        tool: A nanoagent ``Tool`` instance.
        policy: The permission policy to apply.

    Returns:
        A new ``Tool`` with permission checking.
    """
    from nanoagent.tools import Tool as NanoTool

    if not isinstance(tool, NanoTool):
        raise TypeError("tool must be a nanoagent Tool")

    original_fn = tool.fn

    async def guarded(raw_args: dict[str, Any]) -> object:
        decision = await policy.check(tool.name, raw_args)
        if not decision.allowed:
            return f"BLOCKED: {decision.reason}"
        if decision.require_confirmation:
            return (
                f"CONFIRMATION REQUIRED: {decision.reason}\n"
                f"Tool '{tool.name}' was not executed. "
                f"Set up a human_input_handler to handle confirmations."
            )
        return await original_fn(raw_args)

    return NanoTool(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
        fn=guarded,
    )