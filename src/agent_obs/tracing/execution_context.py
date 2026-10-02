"""Execution context propagated through a single agent run."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any


class ExecutionContext:

    def __init__(
        self,
        run_id: str,
        metadata: dict[str, Any] | None = None,
    ):
        self.run_id = run_id
        self.metadata = metadata or {}

        self._node_stack: ContextVar[list] = ContextVar(
            f"trace_stack_{self.run_id}",
            default=[],
        )

    def get_current_node(self):
        stack = self._node_stack.get()
        return stack[-1] if stack else None

    def push_node(self, node):
        stack = self._node_stack.get().copy()
        current = self.get_current_node()

        if current is not None and node.parent_id is None:
            node.parent_id = current.node_id

        stack.append(node)
        self._node_stack.set(stack)

    def pop_node(self):
        stack = self._node_stack.get().copy()

        if stack:
            stack.pop()

        self._node_stack.set(stack)

    def clear(self):
        self._node_stack.set([])

    def child_context(self, **metadata):
        return ExecutionContext(
            run_id=self.run_id,
            metadata={
                **self.metadata,
                **metadata,
            },
        )