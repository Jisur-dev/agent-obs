"""Deterministic replay of a previously recorded execution graph."""

from __future__ import annotations

from dataclasses import dataclass

from .trace_graph.execution_graph import ExecutionTree
from .trace_graph.trace_node import TraceNode
from ..tracing.trace_graph.enums import NodeType


@dataclass
class ReplayOverride:
    """A single field to override when replaying a node (e.g. temperature, prompt)."""

    node_id: str
    field: str
    value: object


class ReplayEngine:
    """Re-executes a recorded run.

    Tool/model calls are replayed from stored snapshots unless a node is
    explicitly overridden, in which case execution branches from that point
    forward -- letting a developer change one variable and see exactly where
    behavior diverges.
    """

    def __init__(self, graph: ExecutionTree):
        pass

    def replay(self, overrides: list[ReplayOverride] | None = None) -> ExecutionTree:
        """Produce a new ExecutionGraph representing the replayed run."""
        pass

    def _resolve_node(self, node: TraceNode, overrides: list[ReplayOverride]) -> TraceNode:
        """Apply any matching overrides to a single node before re-executing it."""
        pass
