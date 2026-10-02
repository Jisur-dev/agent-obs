from __future__ import annotations
from typing import Any
from ..trace_graph.trace_node import TraceNode, NodeType


class TraceNodeHelper:
    """
    Helper class responsible for data manipulation and serialization/deserialization 
    of TraceNode objects, keeping the core dataclass clean and decoupled.
    """

    @staticmethod
    def to_dict(node: TraceNode) -> dict[str, Any]:
        """
        Role:
            Serializes a TraceNode instance into a standard dictionary format.

        When & Why:
            Used when exporting execution state, persisting trace data, or sending 
            node details across the network/logs.
        """
        return {
            "node_id": node.node_id,
            "node_type": (
                node.node_type.value
                if isinstance(node.node_type, NodeType)
                else node.node_type
            ),
            "parent_id": node.parent_id,
            "timestamp": node.timestamp,
            "prompt_hash": node.prompt_hash,
            "model": node.model,
            "temperature": node.temperature,
            "seed": node.seed,
            "latency_ms": node.latency_ms,
            "cost_usd": node.cost_usd,
            "input_data": node.input_data,
            "output_data": node.output_data,
            "memory_snapshot": node.memory_snapshot,
            "tool_parameters": node.tool_parameters,
            "tool_result": node.tool_result,
            "exception": node.exception,
            "retry_count": node.retry_count,
            "tags": node.tags,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> TraceNode:
        """
        Role:
            Reconstructs a TraceNode instance from a raw dictionary.

        When & Why:
            Used during state restoration, loading saved traces, or parsing 
            incoming serialized payloads back into active objects.
        """
        data_copy = data.copy()
        if "node_type" in data_copy and isinstance(data_copy["node_type"], str):
            data_copy["node_type"] = NodeType(data_copy["node_type"])
        return TraceNode(**data_copy)