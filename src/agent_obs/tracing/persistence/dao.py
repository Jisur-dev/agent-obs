from sqlalchemy.orm import Session
from .models import Graphs, Run ,Nodes, Edges

class TracingDAO:

    def __init__(self, session: Session):
        self.session = session


    def save(self, graph):

        """save graph to database"""
        db_graph = Graphs()
        self.session.add(db_graph)
        self.session.flush()
        """save run to database"""
        db_run = Run(
            run_id=graph.run_id,
            graph_id=db_graph.graph_id
        )
        self.session.add(db_run)

        """save node to database"""
        # create all nodes
        db_nodes = {}

        for node in graph.nodes.values():
            db_node = Nodes(
                node_id=node.node_id,
                run_id=graph.run_id,
                node_type=node.node_type,
                parent_id=node.parent_id,
                timestamp=node.timestamp,
                prompt_hash=node.prompt_hash,
                model=node.model,
                temperature=node.temperature,
                seed=node.seed,
                latency_ms=node.latency_ms,
                cost_usd=node.cost_usd,
                input_data=node.input_data,
                output_data=node.output_data,
                memory_snapshot=node.memory_snapshot,
                tool_parameters=node.tool_parameters,
                tool_result=node.tool_result,
                exception=node.exception,
                retry_count=node.retry_count,
                tags=node.tags,
            )

            db_nodes[node.node_id] = db_node
            self.session.add(db_node)

        # connect parent nodes
        for node in graph.nodes.values():
            if node.parent_id is not None:
                db_nodes[node.node_id].parent = db_nodes[node.parent_id]


        """save edge to database"""
        for edge in graph.edges:
            db_edge=Edges(
                run_id=graph.run_id,
                source_id=edge.source_id,
                target_id=edge.target_id,
                edge_type=edge.edge_type,
                edge_metadata=edge.metadata
            )
            self.session.add(db_edge)
        self.session.commit()