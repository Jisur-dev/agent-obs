
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from agent_obs.tracing.persistence.base import Base
from agent_obs.tracing.persistence.dao import TracingDAO
from agent_obs.tracing.persistence.models import Graphs, Run, Nodes, Edges

from agent_obs.tracing.trace_graph.execution_graph import ExecutionTree
from agent_obs.tracing.trace_graph.trace_node import TraceNode
from agent_obs.tracing.trace_graph.enums import NodeType


def test_save_graph():
    # Create an in-memory database
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # Create an execution graph
        graph = ExecutionTree()

        # Create root node
        root = TraceNode(
            node_id="node-1",
            node_type=NodeType.AGENT,
            parent_id=None,
            timestamp=1.0,
            prompt_hash=None,
            model=None,
            temperature=None,
            seed=None,
            latency_ms=None,
            cost_usd=None,
            input_data=None,
            output_data=None,
            memory_snapshot=None,
            tool_parameters=None,
            tool_result=None,
            exception=None,
            retry_count=0,
            tags=[]
        )

        graph.add_node(root)

        # Create child node
        child = TraceNode(
            node_id="node-2",
            node_type=NodeType.AGENT,
            parent_id="node-1",
            timestamp=2.0,
            prompt_hash=None,
            model=None,
            temperature=None,
            seed=None,
            latency_ms=None,
            cost_usd=None,
            input_data=None,
            output_data=None,
            memory_snapshot=None,
            tool_parameters=None,
            tool_result=None,
            exception=None,
            retry_count=0,
            tags=[]
        )

        graph.add_node(child)

        # Save graph using DAO
        dao = TracingDAO(session)
        dao.save(graph)

        # Verify database records
        assert session.scalar(select(Graphs)) is not None
        assert session.scalar(select(Run)) is not None

        saved_nodes = session.scalars(select(Nodes)).all()
        assert len(saved_nodes) == 2

        saved_edges = session.scalars(select(Edges)).all()
        assert len(saved_edges) == 1

        # Verify parent-child relationship
        saved_child = session.get(Nodes, "node-2")
        assert saved_child.parent_id == "node-1"
