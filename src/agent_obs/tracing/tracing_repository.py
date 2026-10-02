from __future__ import annotations

from abc import ABC

from .trace_graph.execution_graph import ExecutionTree


class TracingRepository(ABC):
    """Interface for persisting execution graphs."""

    def save(self, graph: ExecutionTree) -> None:
        """Persist an execution graph."""
        raise NotImplementedError

    def get(self, run_id: str) -> ExecutionTree | None:
        """Retrieve an execution graph by run ID."""
        raise NotImplementedError


class InMemoryTracingRepository:
    """Temporary in-memory implementation for testing."""

    def __init__(self) -> None:
        self.graphs: dict[str, ExecutionTree] = {}

    def save(self, graph: ExecutionTree) -> None:
        self.graphs[graph.run_id] = graph

    def get(self, run_id: str) -> ExecutionTree | None:
        return self.graphs.get(run_id)