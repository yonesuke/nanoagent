# -*- coding: utf-8 -*-
"""Tests for guardrails and middleware hooks."""

from __future__ import annotations


from nanoagent.guardrails import AgentHooks, Guardrail, GuardrailResult, guardrail


class TestGuardrailResult:
    def test_allowed(self):
        r = GuardrailResult.allowed()
        assert r.allowed is True
        assert r.message is None
        assert r.modified_content is None

    def test_allowed_with_modification(self):
        r = GuardrailResult.allowed(modified_content="sanitized")
        assert r.allowed is True
        assert r.modified_content == "sanitized"

    def test_rejected(self):
        r = GuardrailResult.rejected("Bad content")
        assert r.allowed is False
        assert r.message == "Bad content"


class TestGuardrail:
    async def test_guardrail_allows(self):
        async def check(text: str, ctx: dict) -> GuardrailResult:
            return GuardrailResult.allowed()

        g = Guardrail(name="test_gr", check=check, runs_on="input")
        result = await g.check("hello", {})
        assert result.allowed is True

    async def test_guardrail_rejects(self):
        async def check(text: str, ctx: dict) -> GuardrailResult:
            if "bad" in text:
                return GuardrailResult.rejected("contains bad word")
            return GuardrailResult.allowed()

        g = Guardrail(name="no_bad", check=check)
        result = await g.check("this is bad stuff", {})
        assert result.allowed is False
        assert "bad word" in (result.message or "")

    def test_runs_on_default(self):
        g = Guardrail(name="g", check=lambda t, c: GuardrailResult.allowed())  # type: ignore[arg-type]
        assert g.runs_on == "input"


class TestGuardrailDecorator:
    def test_decorator_input_guardrail(self):
        @guardrail(name="no_pii", runs_on="input")
        async def check_pii(text: str, ctx: dict) -> GuardrailResult:
            if "ssn" in text.lower():
                return GuardrailResult.rejected("SSN detected")
            return GuardrailResult.allowed()

        assert isinstance(check_pii, Guardrail)
        assert check_pii.name == "no_pii"
        assert check_pii.runs_on == "input"

    def test_decorator_output_guardrail(self):
        @guardrail(name="clean_output", runs_on="output")
        async def check_output(text: str, ctx: dict) -> GuardrailResult:
            return GuardrailResult.allowed()

        assert isinstance(check_output, Guardrail)
        assert check_output.runs_on == "output"
        assert check_output.name == "clean_output"


class TestAgentHooks:
    def test_default_hooks(self):
        hooks = AgentHooks()
        assert hooks.before_agent is None
        assert hooks.after_agent is None
        assert hooks.before_model is None
        assert hooks.after_model is None
        assert hooks.wrap_tool_call is None

    def test_partial_hooks(self):
        async def my_hook(*args, **kwargs):
            pass

        hooks = AgentHooks(
            before_agent=my_hook,
            after_agent=my_hook,
        )
        assert hooks.before_agent is not None
        assert hooks.after_agent is not None
        assert hooks.before_model is None

    async def test_hook_execution(self):
        called = []

        async def before_agent(input_text, ctx):
            called.append(("before_agent", input_text))

        hooks = AgentHooks(before_agent=before_agent)
        await hooks.before_agent("test input", {})  # type: ignore[misc]
        assert len(called) == 1
        assert called[0] == ("before_agent", "test input")