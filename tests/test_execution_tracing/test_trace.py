from agent_obs.tracing.tracer import Tracer
from agent_obs.tracing.trace_graph.trace_node import TraceNode
from agent_obs.tracing.trace_graph.enums import NodeType


def test_trace_node_methods():
    node = TraceNode(node_type=NodeType.LLM_CALL)

    node.set_output("Test Output")
    assert node.output_data == "Test Output"

    node.set_latency()
    assert node.latency_ms is not None

    try:
        raise ValueError("Test Error")
    except Exception as e:
        node.set_exception(e)

    assert "ValueError: Test Error" in node.exception


def test_execution_context_stack():
    tracer = Tracer()
    context = tracer.context

    assert context.get_current_node() is None

    node1 = TraceNode(node_type=NodeType.AGENT)
    context.push_node(node1)

    assert context.get_current_node() == node1

    node2 = TraceNode(node_type=NodeType.LLM_CALL)
    context.push_node(node2)

    assert context.get_current_node() == node2

    context.pop_node()
    assert context.get_current_node() == node1

    context.pop_node()
    assert context.get_current_node() is None


def test_tracer_span_basic_flow():
    tracer = Tracer()

    with tracer.span(NodeType.AGENT) as node:
        node.set_output("Agent result")

    assert node.output_data == "Agent result"
    assert node.node_type == NodeType.AGENT
    assert node.latency_ms is not None

    # The complete graph is saved after the outermost span finishes.
    saved_graph = tracer.tracing_repository.get(tracer.graph.run_id)

    assert saved_graph is not None
    assert node.node_id in saved_graph.nodes
    assert saved_graph.nodes[node.node_id] == node
    assert tracer.context.get_current_node() is None


def test_tracer_span_nested_parent_child():
    tracer = Tracer()

    with tracer.span(NodeType.AGENT) as parent:
        assert tracer.context.get_current_node() == parent

        with tracer.span(NodeType.TOOL_CALL) as tool:
            assert tracer.context.get_current_node() == tool

            with tracer.span(NodeType.LLM_CALL) as llm:
                assert tracer.context.get_current_node() == llm

    saved_graph = tracer.tracing_repository.get(tracer.graph.run_id)

    assert saved_graph is not None
    assert len(saved_graph.nodes) == 3

    assert tool.parent_id == parent.node_id
    assert llm.parent_id == tool.node_id

    assert tracer.context.get_current_node() is None


def test_tracer_span_exception_handling():
    tracer = Tracer()

    try:
        with tracer.span(NodeType.LLM_CALL) as node:
            raise RuntimeError("LLM API Rate Limit Exceeded")
    except RuntimeError:
        pass

    saved_graph = tracer.tracing_repository.get(tracer.graph.run_id)

    assert saved_graph is not None
    assert len(saved_graph.nodes) == 1

    failed_node = saved_graph.nodes[node.node_id]

    assert failed_node.exception is not None
    assert "RuntimeError: LLM API Rate Limit Exceeded" in failed_node.exception
    assert failed_node.latency_ms is not None
    assert failed_node.latency_ms >= 0
    assert tracer.context.get_current_node() is None


def test_tracer_trace_named_span():
    tracer = Tracer()

    with tracer.trace("my_agent", NodeType.AGENT) as node:
        node.set_output("Agent finished")

    saved_graph = tracer.tracing_repository.get(tracer.graph.run_id)

    assert saved_graph is not None
    assert len(saved_graph.nodes) == 1

    saved_node = saved_graph.nodes[node.node_id]

    assert saved_node.node_type == NodeType.AGENT
    assert saved_node.output_data == "Agent finished"
    assert "my_agent" in saved_node.tags
    assert tracer.context.get_current_node() is None


def test_tracer_metadata():
    tracer = Tracer()

    with tracer.span(
        NodeType.LLM_CALL,
        model="gpt-4",
        temperature=0.7,
        seed=42,
        cost_usd=0.005,
        tags=["production", "test"],
    ) as node:
        node.input_data = {"prompt": "Hello"}
        node.set_output({"response": "World"})

    saved_graph = tracer.tracing_repository.get(tracer.graph.run_id)

    assert saved_graph is not None
    assert len(saved_graph.nodes) == 1

    saved_node = saved_graph.nodes[node.node_id]

    assert saved_node.model == "gpt-4"
    assert saved_node.temperature == 0.7
    assert saved_node.seed == 42
    assert saved_node.cost_usd == 0.005
    assert "production" in saved_node.tags
    assert "test" in saved_node.tags
    assert saved_node.input_data["prompt"] == "Hello"
    assert saved_node.output_data["response"] == "World"


def test_tracer_default_repository():
    tracer = Tracer()

    with tracer.span(NodeType.AGENT) as node:
        node.set_output("Default repository")

    saved_graph = tracer.tracing_repository.get(tracer.graph.run_id)

    assert saved_graph is not None
    assert node.node_id in saved_graph.nodes
    assert saved_graph.nodes[node.node_id].output_data == "Default repository"
    assert tracer.context.get_current_node() is None