# -*- coding: utf-8 -*-
"""Tests for ConversationMemory."""

from __future__ import annotations


from nanoagent.memory import ConversationMemory, Message, ToolCallRecord


class TestMessage:
    def test_user_message_to_openai(self):
        msg = Message(role="user", content="hello")
        result = msg.to_openai()
        assert result == {"role": "user", "content": "hello"}

    def test_assistant_with_tool_calls(self):
        msg = Message(
            role="assistant",
            content="Let me check.",
            tool_calls=[
                ToolCallRecord(id="c1", name="search", arguments='{"q":"x"}'),
            ],
        )
        result = msg.to_openai()
        assert result["role"] == "assistant"
        assert result["content"] == "Let me check."
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "c1"
        assert result["tool_calls"][0]["function"]["name"] == "search"

    def test_tool_message(self):
        msg = Message(role="tool", content="result data", tool_call_id="c1")
        result = msg.to_openai()
        assert result["role"] == "tool"
        assert result["content"] == "result data"
        assert result["tool_call_id"] == "c1"

    def test_assistant_no_tool_calls(self):
        msg = Message(role="assistant", content="Done.")
        result = msg.to_openai()
        assert result["role"] == "assistant"
        assert result["content"] == "Done."
        assert "tool_calls" not in result


class TestConversationMemory:
    def test_add_and_retrieve(self):
        mem = ConversationMemory()
        mem.add_user("Hi")
        mem.add_assistant("Hello!")
        mem.add_tool("result", "call_1")

        msgs = mem.to_openai_messages()
        assert len(msgs) == 3
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "tool"

    def test_clear(self):
        mem = ConversationMemory()
        mem.add_user("Hi")
        mem.add_assistant("Hello!")
        mem.clear()
        assert len(mem.messages) == 0
        assert len(mem.to_openai_messages()) == 0

    def test_trim_preserves_system(self):
        mem = ConversationMemory(max_messages=3)
        mem.add(Message(role="system", content="You are helpful."))
        for i in range(5):
            mem.add_user(f"msg {i}")
            mem.add_assistant(f"reply {i}")

        # Should have max_messages (3) total
        assert len(mem.messages) == 3
        # System message preserved
        assert mem.messages[0].role == "system"
        assert mem.messages[0].content == "You are helpful."

    def test_shorthand_methods(self):
        mem = ConversationMemory()
        mem.add_user("question")
        mem.add_assistant("answer!")
        mem.add_tool("data", "t1")

        assert mem.messages[0].role == "user"
        assert mem.messages[0].content == "question"
        assert mem.messages[1].role == "assistant"
        assert mem.messages[1].content == "answer!"
        assert mem.messages[2].role == "tool"
        assert mem.messages[2].tool_call_id == "t1"

    def test_empty_memory(self):
        mem = ConversationMemory()
        assert len(mem.messages) == 0
        assert mem.to_openai_messages() == []

    def test_messages_type(self):
        mem = ConversationMemory()
        mem.add_user("test")
        raw = mem.to_openai_messages()
        assert isinstance(raw, list)
        assert isinstance(raw[0], dict)