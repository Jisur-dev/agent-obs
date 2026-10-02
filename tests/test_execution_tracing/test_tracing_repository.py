from agent_obs.tracing.tracer import Tracer
from agent_obs.tracing.trace_graph.enums import NodeType
from agent_obs.tracing.tracing_repository import InMemoryTracingRepository


def test_repository_saves_complete_graph_after_execution():
    repository = InMemoryTracingRepository()
    tracer = Tracer(tracing_repository=repository)

    with tracer.span(NodeType.AGENT):
        with tracer.span(NodeType.TOOL_CALL):
            pass

        # The outer AGENT is still running,
        # so the graph should not be saved yet.
        assert repository.get(tracer.graph.run_id) is None

    # The complete execution has now finished.
    saved_graph = repository.get(tracer.graph.run_id)

    assert saved_graph is not None
    assert saved_graph.run_id == tracer.graph.run_id
    assert len(saved_graph.nodes) == 2

    node_types = {
        node.node_type
        for node in saved_graph.nodes.values()
    }

    assert NodeType.AGENT in node_types
    assert NodeType.TOOL_CALL in node_types

