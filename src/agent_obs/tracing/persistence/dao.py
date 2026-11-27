
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Graphs, Run, Nodes, Edges


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
