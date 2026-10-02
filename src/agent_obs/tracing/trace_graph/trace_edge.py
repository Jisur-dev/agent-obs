from __future__ import annotations

from dataclasses import dataclass, field


from .enums import EdgeType

@dataclass
class TraceEdge:
    """
    Role:
        Represents a directed connection (edge) establishing a causal or logical link between two trace nodes.

    Type:
        Classifies the relationship using EdgeType (e.g., PARENT_CHILD, TOOL_CALL, DELEGATED_TO).

    When & Why:
        Instantiated whenever two steps or components are connected in the workflow, 
        allowing the system to map execution flow, dependencies, and data movement.

    Output:
        Exposes a clean structure linking source and target nodes with custom metadata, ready for graph assembly.
    """

    source_id: str
    target_id: str
    edge_type: EdgeType = EdgeType.PARENT_CHILD
    metadata: dict = field(default_factory=dict)
