from sqlalchemy import create_engine, inspect
from agent_obs.tracing.persistence.models import Base
from sqlalchemy.orm import Session
from agent_obs.tracing.persistence.models import Base, Graphs, Run , Nodes, Edges
from agent_obs.tracing.trace_graph.enums import NodeType , EdgeType



def test_create_tables():
    engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(engine)

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    assert "graphs" in tables
    assert "runs" in tables
    assert "nodes" in tables
    assert "edges" in tables


def test_graph_run_relationship():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        graph = Graphs()
        run1 = Run()
        run2 = Run()

        graph.runs.append(run1)
        graph.runs.append(run2)

        session.add(graph)
        session.commit()

        assert len(graph.runs) == 2
        assert run1.graph is graph
        assert run2.graph is graph
        assert run1.graph_id == graph.graph_id
        assert run2.graph_id == graph.graph_id
        assert run1.run_id != run2.run_id

def test_run_nodes_relationship():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        graph = Graphs()
        run = Run(graph=graph)

        node1 = Nodes(timestamp=1.0, node_type=NodeType.AGENT)
        node2 = Nodes(timestamp=2.0, node_type=NodeType.AGENT)
        node3 = Nodes(timestamp=3.0, node_type=NodeType.AGENT)

        run.nodes.extend([node1, node2, node3])

        session.add(run)
        session.commit()

        assert len(run.nodes) == 3
        assert node1.run is run
        assert node2.run is run
        assert node3.run is run
        assert node1.run_id == run.run_id
        assert node2.run_id == run.run_id
        assert node3.run_id == run.run_id





def test_node_parent_children_relationship():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        graph = Graphs()
        run = Run(graph=graph)

        parent = Nodes(timestamp=1.0, node_type=NodeType.AGENT)
        child1 = Nodes(timestamp=2.0, node_type=NodeType.LLM_CALL)
        child2 = Nodes(timestamp=3.0, node_type=NodeType.TOOL_CALL)

        parent.children.extend([child1, child2])

        run.nodes.extend([parent, child1, child2])

        session.add(run)
        session.commit()

        assert len(parent.children) == 2
        assert child1.parent is parent
        assert child2.parent is parent

        assert child1.parent_id == parent.node_id
        assert child2.parent_id == parent.node_id

        assert parent.run is run
        assert child1.run is run
        assert child2.run is run

        assert parent.run_id == run.run_id
        assert child1.run_id == run.run_id
        assert child2.run_id == run.run_id


def test_run_edges_relationship():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        graph = Graphs()
        run = Run(graph=graph)

        node1 = Nodes(timestamp=1.0, node_type=NodeType.AGENT)
        node2 = Nodes(timestamp=2.0, node_type=NodeType.LLM_CALL)

        run.nodes.extend([node1, node2])

        edge1 = Edges(
            source=node1,
            target=node2,
            edge_type=EdgeType.PARENT_CHILD
        )

        edge2 = Edges(
            source=node2,
            target=node1,
            edge_type=EdgeType.DELEGATED_TO
        )

        run.edges.extend([edge1, edge2])

        session.add(run)
        session.commit()

        assert len(run.edges) == 2
        assert edge1.run is run
        assert edge2.run is run

        assert edge1.run_id == run.run_id
        assert edge2.run_id == run.run_id


def test_edge_source_target_relationship():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        graph = Graphs()
        run = Run(graph=graph)

        node1 = Nodes(timestamp=1.0, node_type=NodeType.AGENT)
        node2 = Nodes(timestamp=2.0, node_type=NodeType.LLM_CALL)

        run.nodes.extend([node1, node2])

        edge = Edges(
            source=node1,
            target=node2,
            edge_type=EdgeType.PARENT_CHILD
        )

        run.edges.append(edge)

        session.add(run)
        session.commit()

        assert edge.source is node1
        assert edge.target is node2

        assert edge.source_id == node1.node_id
        assert edge.target_id == node2.node_id
