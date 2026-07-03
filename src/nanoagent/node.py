"""Base Node class for the agent execution tree.

Every agent is a Node in a tree. The root agent has depth=0, its sub-agents
have depth=1, and so on. Nodes carry identity (id, name, path) and can emit
streaming events.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator

from .events import Event, EventType


class Node:
    """Base class for all nodes in the agent execution tree.

    Subclass this to create custom node types, or use the built-in
    :class:`Agent` which is an LLM-powered node.
    """

    def __init__(self, name: str, parent: Node | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.parent = parent
        self.root: Node = self if parent is None else parent.root
        self.depth: int = 0 if parent is None else parent.depth + 1
        self._node_path: list[str] | None = None
        self._last_event_time: float = 0.0

    @property
    def node_path(self) -> list[str]:
        """Full path from root to this node, e.g. ['orchestrator', 'analyst']."""
        if self._node_path is None:
            path: list[str] = []
            current: Node | None = self
            while current is not None:
                path.append(current.name)
                current = current.parent
            path.reverse()
            self._node_path = path
        return self._node_path

    def reset_path_cache(self) -> None:
        """Clear cached path so it's recomputed next access."""
        self._node_path = None

    def _make_event(self, type_: EventType, **kwargs: object) -> Event:
        name = kwargs.pop("name", self.name)
        now = time.time()
        elapsed = (
            (now - self._last_event_time) * 1000 if self._last_event_time > 0 else 0.0
        )
        self._last_event_time = now
        return Event(
            type=type_,
            node_id=self.id,
            node_path=self.node_path,
            depth=self.depth,
            name=name,
            timestamp=now,
            elapsed_ms=elapsed,
            **kwargs,  # type: ignore[arg-type]
        )

    async def run(self, input_text: str) -> AsyncIterator[Event]:
        """Execute this node with the given input.

        Override in subclasses. The default yields nothing.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement run()"
        )
        yield  # type: ignore[unreachable]