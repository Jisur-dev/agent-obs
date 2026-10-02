"""Structural diffing between two execution graphs.

Compares run structure (which node did what, in what order) rather than raw
text output, so a divergence report reads like "Node 18 expected Calculator,
observed Search -- confidence below threshold" instead of "output changed".
"""

from __future__ import annotations

from dataclasses import dataclass

from .trace_graph.execution_graph import ExecutionTree


@dataclass
class GraphDivergence:
    """The first point where two execution graphs structurally diverge."""

    node_id: str
    expected: str
    observed: str
    reason: str | None = None


class GraphDiffEngine:
    """Compares two ExecutionGraphs structurally rather than by raw text output."""

    def __init__(self, graph_a: ExecutionTree, graph_b: ExecutionTree):
        pass

    def diff(self) -> list[GraphDivergence]:
        """Return the ordered list of divergences between the two graphs."""
        pass

    def first_divergence(self) -> GraphDivergence | None:
        """Return only the earliest point of divergence."""
        pass
