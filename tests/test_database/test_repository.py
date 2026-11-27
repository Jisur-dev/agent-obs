
from sqlalchemy import create_engine

from sqlalchemy.orm import Session

from agent_obs.tracing.persistence.base import Base
from agent_obs.tracing.persistence.repository import SQLAlchemyTracingRepository
from agent_obs.tracing.trace_graph.enums import NodeType
from agent_obs.tracing.tracer import Tracer


def test_sqlalchemy_repository_save_and_get():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        repository = SQLAlchemyTracingRepository(session)
        tracer = Tracer(tracing_repository=repository)

        with tracer.span(NodeType.AGENT) as node:
            node.tags = ["database-test"]

        restored_graph = repository.get(tracer.graph.run_id)

        assert restored_graph is not None
        assert restored_graph.run_id == tracer.graph.run_id
        assert len(restored_graph.nodes) == 1

        restored_node = restored_graph.get_node(node.node_id)
        assert restored_node.node_type == NodeType.AGENT
        assert restored_node.tags == ["database-test"]


def test_sqlalchemy_repository_returns_none_for_unknown_run():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        repository = SQLAlchemyTracingRepository(session)

        assert repository.get("unknown-run") is None
