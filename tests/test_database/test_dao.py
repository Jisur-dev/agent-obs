
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from agent_obs.tracing.persistence.base import Base
from agent_obs.tracing.persistence.dao import TracingDAO
from agent_obs.tracing.persistence.models import Graphs, Run, Nodes, Edges

from agent_obs.tracing.trace_graph.execution_graph import ExecutionTree
from agent_obs.tracing.trace_graph.trace_node import TraceNode
from agent_obs.tracing.trace_graph.enums import NodeType

import pytest

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

def test_save_graph_rollback(monkeypatch):
    # Create an in-memory database
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        graph = ExecutionTree()
        dao = TracingDAO(session)

        # Simulate a failure during commit
        def fail_commit():
            raise RuntimeError("Simulated database error")

        monkeypatch.setattr(session, "commit", fail_commit)

        # Verify that the error is raised
        with pytest.raises(RuntimeError, match="Simulated database error"):
            dao.save(graph)

        # Verify that all changes were rolled back
        assert session.scalar(select(Graphs)) is None
        assert session.scalar(select(Run)) is None
        assert session.scalars(select(Nodes)).all() == []
        assert session.scalars(select(Edges)).all() == []


def test_save_graph_updates_edge_metadata():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        graph = ExecutionTree()

        root = TraceNode(
            node_id="node-1",
            node_type=NodeType.AGENT
        )

        child = TraceNode(
            node_id="node-2",
            node_type=NodeType.AGENT,
            parent_id="node-1"
        )

        graph.add_node(root)
        graph.add_node(child)

        dao = TracingDAO(session)

        # First save.
        dao.save(graph)

        # Update the metadata of the existing edge.
        edge = graph.edges[0]
        edge.metadata = {"updated": True}

        # Second save.
        dao.save(graph)

        # Verify that metadata was updated.
        saved_edges = session.scalars(select(Edges)).all()

        assert len(saved_edges) == 1
        assert saved_edges[0].edge_metadata == {"updated": True}

