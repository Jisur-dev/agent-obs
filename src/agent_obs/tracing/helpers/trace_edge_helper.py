
from __future__ import annotations
from typing import Any
from ..trace_graph.trace_edge import TraceEdge, EdgeType


class TraceEdgeHelper:
    """
    Helper class responsible for data manipulation and serialization/deserialization 
    of TraceEdge objects, decoupling data conversion logic from the edge model.
    """

    @staticmethod
    def to_dict(edge: TraceEdge) -> dict[str, Any]:
        """
        Role:
            Serializes a TraceEdge instance into a standard dictionary format.

        When & Why:
            Used when exporting the graph's structural edges or persisting relationship 
            data for tracking agent workflows.
        """
        return {
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "edge_type": (
                edge.edge_type.value
                if isinstance(edge.edge_type, EdgeType)
                else edge.edge_type
            ),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> TraceEdge:
        """
        Role:
            Reconstructs a TraceEdge instance from a raw dictionary.

        When & Why:
            Used during graph restoration and parsing serialized connection data 
            back into valid edge objects.
        """
        data_copy = data.copy()
        if "edge_type" in data_copy and isinstance(data_copy["edge_type"], str):
            data_copy["edge_type"] = EdgeType(data_copy["edge_type"])
        return TraceEdge(**data_copy)