# -*- coding: utf-8 -*-
"""Tests for the Node tree structure."""

from __future__ import annotations

import pytest

from nanoagent.node import Node


class TestNodeBasics:
    def test_node_creation(self):
        n = Node(name="root")
        assert n.name == "root"
        assert len(n.id) == 12
        assert n.parent is None
        assert n.depth == 0
        assert n.root is n

    def test_node_with_parent(self):
        root = Node(name="root")
        child = Node(name="child", parent=root)
        assert child.parent is root
        assert child.root is root
        assert child.depth == 1

    def test_deep_nesting(self):
        root = Node(name="r")
        a = Node(name="a", parent=root)
        b = Node(name="b", parent=a)
        c = Node(name="c", parent=b)
        assert c.depth == 3
        assert c.root is root


class TestNodePath:
    def test_root_path(self):
        root = Node(name="root")
        assert root.node_path == ["root"]

    def test_child_path(self):
        root = Node(name="orchestrator")
        child = Node(name="analyst", parent=root)
        assert child.node_path == ["orchestrator", "analyst"]

    def test_deep_path(self):
        root = Node(name="a")
        b = Node(name="b", parent=root)
        c = Node(name="c", parent=b)
        assert c.node_path == ["a", "b", "c"]

    def test_path_caching(self):
        root = Node(name="root")
        child = Node(name="child", parent=root)
        path1 = child.node_path
        path2 = child.node_path
        assert path1 is path2  # cached
        assert path1 == ["root", "child"]

    def test_reset_path_cache(self):
        root = Node(name="root")
        child = Node(name="child", parent=root)
        old = child.node_path
        child.reset_path_cache()
        new = child.node_path
        assert new == old
        # After reset they should be equal but not necessarily same object
        assert new == old


class TestNodeRun:
    async def test_default_run_raises(self):
        n = Node(name="test")
        with pytest.raises(NotImplementedError):
            async for _ in n.run("input"):
                pass

    async def test_custom_node(self):
        class EchoNode(Node):
            async def run(self, input_text: str):
                from nanoagent.events import Event, EventType

                yield Event(
                    type=EventType.TEXT_DELTA,
                    node_id=self.id,
                    node_path=self.node_path,
                    depth=self.depth,
                    delta=input_text,
                )

        n = EchoNode(name="echo")
        results = []
        async for ev in n.run("hello"):
            results.append(ev)
        assert len(results) == 1
        assert results[0].delta == "hello"


class TestNodeMakeEvent:
    def test_make_event_basics(self):
        from nanoagent.events import EventType

        n = Node(name="test_node")
        ev = n._make_event(EventType.NODE_START)
        assert ev.type == EventType.NODE_START
        assert ev.node_id == n.id
        assert ev.name == "test_node"
        assert ev.depth == 0

    def test_make_event_with_kwargs(self):
        from nanoagent.events import EventType

        n = Node(name="test_node")
        ev = n._make_event(
            EventType.TEXT_DELTA,
            delta="hi",
            name="override",
        )
        assert ev.delta == "hi"
        assert ev.name == "override"