"""Core execution graph primitives: Nodes, Edges, and the ExecutionGraph.

An agent run is represented as a Tree rather than a flat log, with 
serialization logic cleanly decoupled into dedicated helper classes.
This makes the run queryable, diffable, and replayable from any point.
"""

from __future__ import annotations

from .trace_edge import TraceEdge
from .trace_node import TraceNode
from .enums import EdgeType

import uuid

class ExecutionTree:
    """
    Role:
        Represents one complete, self-contained execution workflow as a directed graph.

    Type:
        Graph container holding nodes (`TraceNode`) and directed edges (`TraceEdge`) with fast adjacency lookups.

    When & Why:
        Instantiated at the start of an agent run to track all steps, dependencies, tool calls, 
        and state changes, enabling holistic graph traversal, analysis, and cost/latency aggregation.

    Output:
        Maintains an in-memory graph structure equipped with nodes, edges, adjacency maps, 
        and a designated root node ID for full execution introspection.
    """

    def __init__(self, run_id: str | None = None):
        """
        Role:
            Initializes a new empty execution graph with a unique run identifier.

        When & Why:
            Called upon starting a new agent workflow to set up tracking structures, dictionaries for nodes and edges, 
            and adjacency maps required for fast graph navigation.

        Output:
            Sets up an empty graph instance with initialized collections (`nodes`, `edges`, `_adjacency`) 
            and a unique `run_id`.
        """
        self.run_id = run_id or str(uuid.uuid4())
        self.nodes: dict[str, TraceNode] = {}
        self.edges: list[TraceEdge] = []
        self._adjacency: dict[str, list[str]] = {}
        self.root_node_id: str | None = None

    def add_node(self, node: TraceNode) -> None:
        """
        Role:
            Inserts a new TraceNode into the execution graph and manages its internal structure.

        When & Why:
            Called during agent execution whenever a new step, tool call, or sub-operation occurs, 
            ensuring uniqueness, validating parent references, and automatically setting up adjacency lists 
            and parent-child edges.

        Output:
            Updates the graph state by registering the node, initializing its adjacency entries, 
            setting it as root if applicable, and automatically creating an edge if a parent ID exists.
        """

        if node.node_id in self.nodes:
            return

        # If the node has a parent, the parent must already exist.
        if node.parent_id is not None and node.parent_id not in self.nodes:
            raise ValueError(
                f"Parent node not found: {node.parent_id}"
            )

        self.nodes[node.node_id] = node

        self._adjacency.setdefault(node.node_id, [])

        if node.parent_id is not None:
            self._adjacency.setdefault(node.parent_id, [])

            if node.node_id not in self._adjacency[node.parent_id]:
                self._adjacency[node.parent_id].append(node.node_id)

            # Automatically create and append a PARENT_CHILD trace edge
            parent_edge = TraceEdge(
                source_id=node.parent_id,
                target_id=node.node_id,
                edge_type=EdgeType.PARENT_CHILD
            )
            self.edges.append(parent_edge)

        elif self.root_node_id is None:
            self.root_node_id = node.node_id

    def add_edge(self, edge: TraceEdge) -> None:
        """
        Role:
            Connects two existing trace nodes within the execution graph by registering a directed edge.

        When & Why:
            Called to explicitly define relationships (like tool calls, delegation, or custom flows) 
            between nodes after they have been created, validating that both source and target nodes exist.

        Output:
            Appends the edge to the graph's edge list and updates forward mappings.
        """

        if edge.source_id not in self.nodes:
            raise ValueError(
                f"Source node not found: {edge.source_id}"
            )

        if edge.target_id not in self.nodes:
            raise ValueError(
                f"Target node not found: {edge.target_id}"
            )

        self.edges.append(edge)

        self._adjacency.setdefault(edge.source_id, [])

        if edge.target_id not in self._adjacency[edge.source_id]:
            self._adjacency[edge.source_id].append(edge.target_id)

    def get_node(self, node_id: str) -> TraceNode:
        """
        Role:
            Retrieves a specific TraceNode instance using its unique identifier.

        When & Why:
            Called during graph inspection, traversal, or analysis whenever direct access 
            to a node's metrics, inputs, or outputs is required, validating that the node exists.

        Output:
            Returns the corresponding `TraceNode` object or raises a KeyError if it is missing.
        """
        return self.nodes[node_id]

    def children(self, node_id: str) -> list[TraceNode]:
        """
        Role:
            Retrieves all direct child nodes connected to a specified parent node.

        When & Why:
            Called during depth-first or breadth-first graph traversals to explore 
            subsequent execution steps, sub-agent delegations, or nested tool calls.

        Output:
            Returns a list of direct child `TraceNode` objects based on the forward adjacency map.
        """

        if node_id not in self.nodes:
            raise KeyError(f"Node not found: {node_id}")

        return [
            self.nodes[child_id]
            for child_id in self._adjacency.get(node_id, [])
        ]

    def get_ancestors(self, node_id: str) -> list[TraceNode]:
        """
        Role:
            Retrieves all upstream ancestor nodes (parents, grandparents, etc.) for a given node.

        When & Why:
            Called during causal analysis, error root-cause tracing, or lineage tracking 
            to understand the complete execution path that led up to a specific step.

        Output:
            Returns a list of `TraceNode` objects representing all preceding steps connected upstream.
        """

        if node_id not in self.nodes:
            raise KeyError(f"Node not found: {node_id}")

        ancestors: list[TraceNode] = []

        node = self.nodes[node_id]
        ancestor_id = node.parent_id
        while (ancestor_id is not None):
            ancestor = self.nodes[ancestor_id]
            ancestors.append(ancestor)
            ancestor_id = ancestor.parent_id

        return ancestors

    def get_descendants(self, node_id: str) -> list[TraceNode]:
        """
        Role:
            Retrieves all downstream descendant nodes (children, grandchildren, etc.) for a given node.

        When & Why:
            Called during impact analysis, sub-workflow auditing, or recursive cleanup 
            to explore all subsequent steps triggered by a specific action.

        Output:
            Returns a list of `TraceNode` objects representing all succeeding steps connected downstream.
        """

        if node_id not in self.nodes:
            raise KeyError(f"Node not found: {node_id}")

        seen: set[str] = set()
        stack = list(self._adjacency.get(node_id, []))
        descendants: list[TraceNode] = []

        while stack:
            current = stack.pop()

            if current in seen:
                continue

            seen.add(current)

            if current in self.nodes:
                descendants.append(self.nodes[current])
                stack.extend(
                    self._adjacency.get(current, [])
                )

        return descendants

    def validate_consistency(self) -> bool:
        """
        Role:
            Validates that the execution graph is structurally sound and consistent.

        When & Why:
            Invoked before serialization, storage, or final analysis to ensure that root references, 
            nodes, and bidirectional adjacency maps match correctly without orphaned references.

        Output:
            Returns True if the graph is fully consistent, or False otherwise.
        """

        # Empty graph is valid.
        if not self.nodes:
            return self.root_node_id is None

        # A non-empty graph must have a root.
        if self.root_node_id is None:
            return False

        # Root must exist.
        if self.root_node_id not in self.nodes:
            return False

        # Root must not have a parent.
        root = self.nodes[self.root_node_id]

        if root.parent_id is not None:
            return False

        # Check every node.
        for node_id, node in self.nodes.items():

            # Every node should have adjacency entries.
            if node_id not in self._adjacency:
                return False

            # Check parent relationship.
            if node.parent_id is not None:

                # Parent must exist.
                if node.parent_id not in self.nodes:
                    return False

                # Parent must contain this node as a child.
                if node_id not in self._adjacency.get(
                    node.parent_id, []
                ):
                    return False

        # Check forward adjacency.
        for parent_id, child_ids in self._adjacency.items():

            if parent_id not in self.nodes:
                return False

            for child_id in child_ids:

                if child_id not in self.nodes:
                    return False

                # Child's parent must point back to parent.
                if self.nodes[child_id].parent_id != parent_id:
                    return False

        return True