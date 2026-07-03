"""Guardrails and middleware hooks for agent safety and control.

Guardrails let you intercept and validate agent inputs and outputs before
they're processed. They can block execution, modify content, or simply
log/trace.

Inspired by OpenAI Agents SDK's ``input_guardrails``/``output_guardrails``
and LangChain's ``AgentMiddleware`` hooks.

Usage::

    from nanoagent import Guardrail, GuardrailResult, guardrail

    @guardrail(name="no_pii")
    async def no_pii_check(input_text: str) -> GuardrailResult:
        if "password" in input_text.lower():
            return GuardrailResult.rejected("Input contains sensitive data")
        return GuardrailResult.allowed()

    agent = Agent(
        ...,
        input_guardrails=[no_pii_check],
    )
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

# ── Guardrail result ─────────────────────────────────────────────────────


@dataclass
class GuardrailResult:
    """Result of running a guardrail check.

    Attributes:
        allowed: Whether the content passes the guardrail.
        message: Optional explanation (shown if rejected).
        modified_content: If set, replaces the original content.
    """

    allowed: bool
    message: str | None = None
    modified_content: str | None = None

    @classmethod
    def allowed(cls, modified_content: str | None = None) -> GuardrailResult:
        """Create a passing result, optionally modifying the content."""
        return cls(allowed=True, modified_content=modified_content)

    @classmethod
    def rejected(cls, message: str) -> GuardrailResult:
        """Create a failing result with an explanation."""
        return cls(allowed=False, message=message)


# ── Guardrail definition ─────────────────────────────────────────────────


@dataclass
class Guardrail:
    """A guardrail that validates agent input or output.

    Attributes:
        name: Guardrail name (used in error messages and tracing).
        check: Async function taking (content, context) and returning a
            GuardrailResult.
        runs_on: When to run — ``"input"`` (before the agent) or
            ``"output"`` (after the agent).
    """

    name: str
    check: Callable[[str, dict[str, Any]], Awaitable[GuardrailResult]]
    runs_on: str = "input"  # "input" | "output"


# ── Decorator ────────────────────────────────────────────────────────────


def guardrail(
    name: str | None = None,
    runs_on: str = "input",
) -> Callable[[Callable[..., Awaitable[GuardrailResult]]], Guardrail]:
    """Decorator to create a :class:`Guardrail` from a check function.

    The decorated function receives ``(content: str, context: dict)`` and
    must return a :class:`GuardrailResult`.

    Example::

        @guardrail(name="no_pii", runs_on="input")
        async def check_no_pii(input_text: str, ctx: dict) -> GuardrailResult:
            if "ssn" in input_text.lower():
                return GuardrailResult.rejected("SSN detected")
            return GuardrailResult.allowed()

    Args:
        name: Guardrail name. Defaults to the function name.
        runs_on: ``"input"`` or ``"output"``.
    """
    def decorator(fn: Callable[..., Awaitable[GuardrailResult]]) -> Guardrail:
        return Guardrail(
            name=name or fn.__name__,
            check=fn,  # type: ignore[arg-type]
            runs_on=runs_on,
        )

    return decorator


# ── Middleware hooks ──────────────────────────────────────────────────────


@dataclass
class AgentHooks:
    """Middleware hooks for intercepting agent execution phases.

    All hooks are optional async callbacks. Return ``None`` or raise
    an exception to interrupt the flow.

    Attributes:
        before_agent: Called before the agent starts processing.
        after_agent: Called after the agent finishes (with final output).
        before_model: Called before each LLM call (with messages).
        after_model: Called after each LLM response (with response text).
        wrap_tool_call: Called before each tool execution (with tool name and args).
    """

    before_agent: Callable[[str, Any], Awaitable[None]] | None = None
    after_agent: Callable[[str, Any], Awaitable[None]] | None = None
    before_model: Callable[[list[dict[str, Any]], Any], Awaitable[None]] | None = None
    after_model: Callable[[str, Any], Awaitable[None]] | None = None
    wrap_tool_call: Callable[[str, dict[str, Any], Any], Awaitable[None]] | None = None  # (tool_name, args, ctx)