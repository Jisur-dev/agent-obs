
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Graphs, Run, Nodes, Edges

from ..trace_graph.execution_graph import ExecutionTree
from ..trace_graph.trace_node import TraceNode
from ..trace_graph.trace_edge import TraceEdge


class TracingDAO:

    def __init__(self, session: Session):
        self.session = session

    def save(self, graph):
        """Save or update an execution graph."""

        try:
            # Check whether this run already exists.
            db_run = self.session.get(Run, graph.run_id)

            if db_run is None:
                # Create a new graph and run.
                db_graph = Graphs()
                self.session.add(db_graph)
                self.session.flush()

                db_run = Run(
                    run_id=graph.run_id,
                    graph_id=db_graph.graph_id
                )
                self.session.add(db_run)
                self.session.flush()

            # Save or update nodes.

            for node in graph.nodes.values():

                db_node = self.session.get(Nodes, node.node_id)

                if db_node is None:
                    db_node = Nodes(
                        node_id=node.node_id,
                        run_id=graph.run_id
                    )
                    self.session.add(db_node)

                elif db_node.run_id != graph.run_id:
                    raise ValueError(
                        f"Node {node.node_id} belongs to another run."
                    )

                db_node.node_type = node.node_type
                db_node.parent_id = node.parent_id
                db_node.timestamp = node.timestamp
                db_node.prompt_hash = node.prompt_hash
                db_node.model = node.model
                db_node.temperature = node.temperature
                db_node.seed = node.seed
                db_node.latency_ms = node.latency_ms
                db_node.cost_usd = node.cost_usd
                db_node.input_data = node.input_data
                db_node.output_data = node.output_data
                db_node.memory_snapshot = node.memory_snapshot
                db_node.tool_parameters = node.tool_parameters
                db_node.tool_result = node.tool_result
                db_node.exception = node.exception
                db_node.retry_count = node.retry_count
                db_node.tags = node.tags

           # Ensure nodes exist before inserting edges.

            self.session.flush()

            # Load existing edges to avoid inserting duplicates.
            existing_edges = self.session.scalars(
                select(Edges).where(Edges.run_id == graph.run_id)
            ).all()

            existing_edge_map = {
                (
                    edge.source_id,
                    edge.target_id,
                    edge.edge_type
                ): edge
                for edge in existing_edges
            }

            # Save or update edges.
            for edge in graph.edges:
                edge_key = (
                    edge.source_id,
                    edge.target_id,
                    edge.edge_type
                )

                db_edge = existing_edge_map.get(edge_key)

                if db_edge is None:
                    db_edge = Edges(
                        run_id=graph.run_id,
                        source_id=edge.source_id,
                        target_id=edge.target_id,
                        edge_type=edge.edge_type,
                        edge_metadata=edge.metadata
                    )
                    self.session.add(db_edge)
                    existing_edge_map[edge_key] = db_edge

                else:
                    db_edge.edge_metadata = edge.metadata
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
    def get(self, run_id: str) -> ExecutionTree | None:
        """Retrieve an execution graph by its run ID."""

        # Check whether the run exists.
        db_run = self.session.get(Run, run_id)

        if db_run is None:
            return None

        # Create an empty execution graph.
        graph = ExecutionTree(run_id=run_id)

        # Retrieve all nodes belonging to this run.
        db_nodes = self.session.scalars(
            select(Nodes).where(Nodes.run_id == run_id)
        ).all()

        # Convert database nodes into TraceNode objects.
        pending_nodes = {}

        for db_node in db_nodes:
            node = TraceNode(
                node_id=db_node.node_id,
                node_type=db_node.node_type,
                parent_id=db_node.parent_id,
                timestamp=db_node.timestamp,
                prompt_hash=db_node.prompt_hash,
                model=db_node.model,
                temperature=db_node.temperature,
                seed=db_node.seed,
                latency_ms=db_node.latency_ms,
                cost_usd=db_node.cost_usd,
                input_data=db_node.input_data,
                output_data=db_node.output_data,
                memory_snapshot=db_node.memory_snapshot,
                tool_parameters=db_node.tool_parameters,
                tool_result=db_node.tool_result,
                exception=db_node.exception,
                retry_count=db_node.retry_count,
                tags=db_node.tags or []
            )

            pending_nodes[node.node_id] = node

        # Add nodes in parent-first order.
        while pending_nodes:
            added_nodes = []

            for node_id, node in pending_nodes.items():
                if (
                    node.parent_id is None
                    or node.parent_id in graph.nodes
                ):
                    graph.add_node(node)
                    added_nodes.append(node_id)

            if not added_nodes:
                raise ValueError(
                    "Invalid graph: missing parent or cyclic hierarchy."
                )

            for node_id in added_nodes:
                del pending_nodes[node_id]

        # Retrieve all edges belonging to this run.
        db_edges = self.session.scalars(
            select(Edges).where(Edges.run_id == run_id)
        ).all()

        # Add edges without duplicating automatic parent-child edges.
        existing_edge_keys = {
            (
                edge.source_id,
                edge.target_id,
                edge.edge_type
            )
            for edge in graph.edges
        }

        for db_edge in db_edges:
            edge_key = (
                db_edge.source_id,
                db_edge.target_id,
                db_edge.edge_type
            )

            if edge_key in existing_edge_keys:
                continue

            edge = TraceEdge(
                source_id=db_edge.source_id,
                target_id=db_edge.target_id,
                edge_type=db_edge.edge_type,
                metadata=db_edge.edge_metadata or {}
            )

            graph.add_edge(edge)
            existing_edge_keys.add(edge_key)

        return graph
