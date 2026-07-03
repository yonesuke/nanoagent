# -*- coding: utf-8 -*-
"""Rich user input tools — questions, confirmations, multi-step forms.

Goes beyond the basic ``ask_user`` tool with structured question types:
- ``question_tool()`` — Single question with options + custom input
- ``questionnaire_tool()`` — Multi-question tabbed form
- ``confirm_tool()`` — Yes/no confirmation dialog
- ``clarify_tool()`` — Open-ended clarification (auto-summarizes)

All tools work with nanoagent's ``human_input_handler`` callback pattern —
they format the question, pass it to the handler, and return structured
results the LLM can understand.

Inspired by pi's ``question.ts`` and ``questionnaire.ts`` tools.

Usage::

    from nanoagent.contrib.input import question_tool, confirm_tool

    agent = Agent(
        name="assistant",
        ...,
        tools=[question_tool(), confirm_tool()],
        human_input_handler=my_handler,
    )
"""

from __future__ import annotations

from typing import Any

from nanoagent.exceptions import NeedsHumanInput


# ── Helpers ──────────────────────────────────────────────────────────────


def _format_options(options: list[dict[str, str]], allow_other: bool = True) -> str:
    """Format options for display to the user."""
    lines = []
    for i, opt in enumerate(options, 1):
        desc = f" — {opt['description']}" if opt.get("description") else ""
        lines.append(f"  {i}. {opt['label']}{desc}")
    if allow_other:
        lines.append(f"  {len(options) + 1}. (Type your own answer)")
    return "\n".join(lines)


def _parse_answer(
    raw: str, options: list[dict[str, str]], allow_other: bool = True
) -> dict[str, Any]:
    """Parse user's raw answer against option list.

    Returns a dict with: ``selected_value``, ``selected_label``,
    ``was_custom`` (bool), ``raw_input``.
    """
    raw = raw.strip()

    # Try numeric selection
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(options):
            return {
                "selected_value": options[idx]["value"],
                "selected_label": options[idx]["label"],
                "was_custom": False,
                "raw_input": raw,
            }
    except (ValueError, TypeError):
        pass

    # Try matching by label or value
    for opt in options:
        if raw.lower() == opt["label"].lower() or raw.lower() == opt["value"].lower():
            return {
                "selected_value": opt["value"],
                "selected_label": opt["label"],
                "was_custom": False,
                "raw_input": raw,
            }

    # Custom/other
    if allow_other:
        return {
            "selected_value": raw,
            "selected_label": raw,
            "was_custom": True,
            "raw_input": raw,
        }

    # Fallback: return first option
    return {
        "selected_value": options[0]["value"],
        "selected_label": options[0]["label"],
        "was_custom": False,
        "raw_input": raw,
    }


# ── Question tool ────────────────────────────────────────────────────────


def question_tool(
    *,
    tool_name: str = "question",
) -> Any:
    """Create a tool for asking the user a single question with options.

    The LLM provides the question text and options. The tool formats
    everything for the human_input_handler and parses the response.

    Example LLM usage::

        question(
            question="Which database should we use?",
            options=[
                {"value": "postgres", "label": "PostgreSQL"},
                {"value": "sqlite", "label": "SQLite"},
            ]
        )

    Returns:
        A nanoagent Tool instance.
    """
    from nanoagent.tools import Tool

    async def ask(raw_args: dict[str, Any]) -> str:
        question = raw_args.get("question", "Please choose:")
        options: list[dict[str, str]] = [
            {"value": o["value"], "label": o["label"], "description": o.get("description", "")}
            for o in raw_args.get("options", [])
        ]
        if not options:
            return "Error: question tool requires at least one option."

        formatted = f"{question}\n\n{_format_options(options)}"
        raise NeedsHumanInput(question=formatted, tool_name=tool_name)

    return Tool(
        name=tool_name,
        description=(
            "Ask the user a single question with predefined options. "
            "Use when you need the user to choose from specific choices. "
            "The user can pick an option by number, label, or type a "
            "custom answer. Returns the selected value."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user.",
                },
                "options": {
                    "type": "array",
                    "description": "Available options to choose from.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "value": {
                                "type": "string",
                                "description": "The value returned when this option is selected.",
                            },
                            "label": {
                                "type": "string",
                                "description": "Display label for the option.",
                            },
                            "description": {
                                "type": "string",
                                "description": "Optional description shown below the label.",
                            },
                        },
                        "required": ["value", "label"],
                    },
                },
            },
            "required": ["question", "options"],
        },
        fn=ask,
    )


# ── Confirm tool ─────────────────────────────────────────────────────────


def confirm_tool(
    *,
    tool_name: str = "confirm",
) -> Any:
    """Create a tool for yes/no confirmations.

    Example LLM usage::

        confirm(
            message="This will delete all records. Continue?",
            default="no",
        )

    Returns:
        A nanoagent Tool instance.
    """
    from nanoagent.tools import Tool

    async def check(raw_args: dict[str, Any]) -> str:
        message = raw_args.get("message", "Are you sure?")
        default = raw_args.get("default", "no")
        default_hint = " [Y/n]" if default == "yes" else " [y/N]"

        formatted = f"{message}{default_hint}"
        raise NeedsHumanInput(question=formatted, tool_name=tool_name)

    return Tool(
        name=tool_name,
        description=(
            "Ask the user for a yes/no confirmation. "
            "Use before destructive or irreversible actions. "
            "Returns 'yes' or 'no' based on the user's response."
        ),
        parameters={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The confirmation message to show.",
                },
                "default": {
                    "type": "string",
                    "description": "Default answer: 'yes' or 'no'.",
                    "enum": ["yes", "no"],
                },
            },
            "required": ["message"],
        },
        fn=check,
    )


# ── Clarify tool ─────────────────────────────────────────────────────────


def clarify_tool(
    *,
    tool_name: str = "clarify",
) -> Any:
    """Create a tool for open-ended clarification.

    Unlike ``question_tool``, this doesn't constrain the user to
    predefined options. Useful when the LLM needs to understand
    the user's intent.

    Example LLM usage::

        clarify(
            topic="data export format",
            context="User asked to export their data but didn't specify format.",
            suggestions=["CSV", "JSON", "Parquet"],
        )

    Returns:
        A nanoagent Tool instance.
    """
    from nanoagent.tools import Tool

    async def ask(raw_args: dict[str, Any]) -> str:
        topic = raw_args.get("topic", "clarification")
        context = raw_args.get("context", "")
        suggestions = raw_args.get("suggestions", [])

        parts = [f"I need to clarify: {topic}"]
        if context:
            parts.append(f"\nContext: {context}")
        if suggestions:
            parts.append(f"\nSuggestions: {', '.join(suggestions)}")
        parts.append("\n\nYour response:")

        formatted = "".join(parts)
        raise NeedsHumanInput(question=formatted, tool_name=tool_name)

    return Tool(
        name=tool_name,
        description=(
            "Ask the user for clarification on a topic. "
            "Use when the user's request is ambiguous and you need "
            "more context before proceeding. You can optionally "
            "suggest possible answers."
        ),
        parameters={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic needing clarification.",
                },
                "context": {
                    "type": "string",
                    "description": "What you already know about the situation.",
                },
                "suggestions": {
                    "type": "array",
                    "description": "Suggested answers to guide the user.",
                    "items": {"type": "string"},
                },
            },
            "required": ["topic"],
        },
        fn=ask,
    )


# ── Questionnaire tool ───────────────────────────────────────────────────


def questionnaire_tool(
    *,
    tool_name: str = "questionnaire",
) -> Any:
    """Create a tool for multi-step questionnaires.

    Each question has its own id, prompt, and options. The tool
    formats everything and returns structured results.

    Example LLM usage::

        questionnaire(questions=[
            {"id": "db", "prompt": "Which database?", "options": [
                {"value": "pg", "label": "PostgreSQL"},
                {"value": "sqlite", "label": "SQLite"},
            ]},
            {"id": "lang", "prompt": "Which language?", "options": [
                {"value": "py", "label": "Python"},
                {"value": "ts", "label": "TypeScript"},
            ]},
        ])

    Returns:
        A nanoagent Tool instance.
    """
    from nanoagent.tools import Tool

    async def ask(raw_args: dict[str, Any]) -> str:
        questions = raw_args.get("questions", [])
        if not questions:
            return "Error: questionnaire requires at least one question."

        parts = ["Please answer the following questions:\n"]
        for i, q in enumerate(questions, 1):
            qid = q.get("id", f"q{i}")
            prompt = q.get("prompt", qid)
            options: list[dict[str, str]] = [
                {"value": o["value"], "label": o["label"], "description": o.get("description", "")}
                for o in q.get("options", [])
            ]

            parts.append(f"## {i}. {prompt}")
            parts.append(f"   ID: {qid}")
            if options:
                parts.append(_format_options(options))
            else:
                parts.append("   (Open-ended)")
            parts.append("")

        parts.append("Respond with: <question_id>: <your answer>")
        parts.append("Example: db: 1, lang: custom answer")

        formatted = "\n".join(parts)
        raise NeedsHumanInput(question=formatted, tool_name=tool_name)

    return Tool(
        name=tool_name,
        description=(
            "Ask the user multiple questions at once. "
            "Use for gathering requirements, preferences, or structured "
            "feedback. Each question has an id, prompt, and optional "
            "options list. The user answers all questions in one response."
        ),
        parameters={
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "description": "Questions to ask the user.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Unique identifier (e.g. 'db', 'lang').",
                            },
                            "prompt": {
                                "type": "string",
                                "description": "The question text.",
                            },
                            "options": {
                                "type": "array",
                                "description": "Available options (omit for open-ended).",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "value": {"type": "string"},
                                        "label": {"type": "string"},
                                    },
                                    "required": ["value", "label"],
                                },
                            },
                        },
                        "required": ["id", "prompt"],
                    },
                },
            },
            "required": ["questions"],
        },
        fn=ask,
    )