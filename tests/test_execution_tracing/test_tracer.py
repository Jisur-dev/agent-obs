import sys
from pathlib import Path

import pytest

# Ensure project root is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_obs.tracing.trace_graph.enums import NodeType, EdgeType
from agent_obs.tracing.tracer import Tracer
from agent_obs.tracing.helpers.trace_node_helper import TraceNodeHelper
from agent_obs.tracing.helpers.trace_edge_helper import TraceEdgeHelper


@pytest.fixture
def tracer_instance():
    """Provides a Tracer using the real in-memory repository."""
    return Tracer()


def test_trace_agent_execution_flow(tracer_instance):
    """Test that @trace_agent creates a root node."""
    @tracer_instance.trace_agent
    def run_agent(query: str):
        return f"Processed: {query}"

    result = run_agent("Test Query")

    assert result == "Processed: Test Query"

    graph = tracer_instance.get_graph()

    assert len(graph.nodes) == 1

    saved_node = next(iter(graph.nodes.values()))

    assert saved_node.node_type == NodeType.AGENT
    assert saved_node.parent_id is None
    assert saved_node.latency_ms >= 0

    assert saved_node.node_id in graph.nodes
    assert graph.root_node_id == saved_node.node_id


def test_trace_tool_call_metadata(tracer_instance):
    """Test tool parameters, result, output, and node type."""

    @tracer_instance.trace_tool_call
    def multiply(a: int, b: int = 2):
        return a * b

    result = multiply(5, b=3)

    assert result == 15

    graph = tracer_instance.get_graph()

    assert len(graph.nodes) == 1

    saved_node = next(iter(graph.nodes.values()))

    assert saved_node.node_type == NodeType.TOOL_CALL

    tool_params = saved_node.tool_parameters

    assert tool_params["a"] == 5
    assert tool_params["b"] == 3

    assert saved_node.tool_result == 15
    assert saved_node.output_data == 15
    assert saved_node.latency_ms >= 0


def test_trace_llm_call_metadata_extraction(tracer_instance):
    """Test extracting model and temperature from decorator and kwargs."""

    @tracer_instance.trace_llm_call(model="gpt-4o-mini")
    def generate_text(
        prompt: str,
        temperature: float = 0.7,
    ):
        return "Generated response"

    result = generate_text(
        "Hello",
        temperature=0.2,
    )

    assert result == "Generated response"

    graph = tracer_instance.get_graph()

    assert len(graph.nodes) == 1

    saved_node = next(iter(graph.nodes.values()))

    assert saved_node.node_type == NodeType.LLM_CALL
    assert saved_node.model == "gpt-4o-mini"
    assert saved_node.temperature == 0.2
    assert saved_node.output_data == "Generated response"
    assert saved_node.latency_ms >= 0


def test_nested_execution_tree(tracer_instance):
    """Test automatic parent-child relationships."""

    @tracer_instance.trace_tool_call
    def fetch_data(item_id: str):
        return f"Item {item_id}"

    @tracer_instance.trace_llm_call
    def summarize(text: str):
        return f"Summary of {text}"

    @tracer_instance.trace_agent
    def agent_workflow(item_id: str):
        data = fetch_data(item_id)
        summary = summarize(data)
        return summary

    result = agent_workflow("123")

    assert result == "Summary of Item 123"

    graph = tracer_instance.get_graph()

    assert len(graph.nodes) == 3

    agent_node = next(
        node for node in graph.nodes.values()
        if node.node_type == NodeType.AGENT
    )

    tool_node = next(
        node for node in graph.nodes.values()
        if node.node_type == NodeType.TOOL_CALL
    )

    llm_node = next(
        node for node in graph.nodes.values()
        if node.node_type == NodeType.LLM_CALL
    )

    assert agent_node.parent_id is None
    assert tool_node.parent_id == agent_node.node_id
    assert llm_node.parent_id == agent_node.node_id

    assert graph.root_node_id == agent_node.node_id

    children = graph.children(agent_node.node_id)
    child_ids = {node.node_id for node in children}

    assert tool_node.node_id in child_ids
    assert llm_node.node_id in child_ids

    assert graph.validate_consistency()


def test_exception_handling(tracer_instance):
    """Test that exceptions are recorded and re-raised."""

    @tracer_instance.trace_tool_call
    def failing_tool():
        raise ValueError("Database connection failed")

    with pytest.raises(
        ValueError,
        match="Database connection failed",
    ):
        failing_tool()

    graph = tracer_instance.get_graph()

    assert len(graph.nodes) == 1

    saved_node = next(iter(graph.nodes.values()))

    assert saved_node.exception is not None
    assert "ValueError" in saved_node.exception
    assert "Database connection failed" in saved_node.exception
    assert saved_node.latency_ms >= 0


def test_tool_exception_is_not_swallowed(tracer_instance):
    """Test that the original tool exception reaches the caller."""

    @tracer_instance.trace_tool_call
    def divide_by_zero():
        return 10 / 0

    with pytest.raises(ZeroDivisionError):
        divide_by_zero()

    graph = tracer_instance.get_graph()

    assert len(graph.nodes) == 1

    saved_node = next(iter(graph.nodes.values()))

    assert saved_node.exception is not None
    assert "ZeroDivisionError" in saved_node.exception


def test_llm_runtime_parameters(tracer_instance):
    """Test that LLM parameters can be captured from runtime kwargs."""

    @tracer_instance.trace_llm_call
    def call_llm(
        prompt: str,
        model: str = "default-model",
        temperature: float = 0.5,
        seed: int = 42,
    ):
        return "response"

    result = call_llm(
        "Hello",
        model="test-model",
        temperature=0.3,
        seed=123,
    )

    assert result == "response"

    graph = tracer_instance.get_graph()

    saved_node = next(iter(graph.nodes.values()))

    assert saved_node.node_type == NodeType.LLM_CALL
    assert saved_node.model == "test-model"
    assert saved_node.temperature == 0.3
    assert saved_node.seed == 123


def test_edge_serialization(tracer_instance):
    """Test serialization and restoration of a parent-child edge."""

    @tracer_instance.trace_tool_call
    def tool():
        return "tool result"

    @tracer_instance.trace_agent
    def workflow():
        return tool()

    workflow()

    graph = tracer_instance.get_graph()

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1

    edge = graph.edges[0]

    data = TraceEdgeHelper.to_dict(edge)

    assert data["source_id"] == edge.source_id
    assert data["target_id"] == edge.target_id
    assert data["edge_type"] == EdgeType.PARENT_CHILD.value

    restored_edge = TraceEdgeHelper.from_dict(data)

    assert restored_edge.source_id == edge.source_id
    assert restored_edge.target_id == edge.target_id
    assert restored_edge.edge_type == EdgeType.PARENT_CHILD


def test_multiple_independent_executions(tracer_instance):
    """Test that separate top-level calls create separate root nodes."""

    @tracer_instance.trace_tool_call
    def tool(value):
        return value * 2

    first_result = tool(5)
    second_result = tool(10)

    assert first_result == 10
    assert second_result == 20

    graph = tracer_instance.get_graph()

    assert len(graph.nodes) == 2

    saved_nodes = list(graph.nodes.values())

    assert saved_nodes[0].parent_id is None
    assert saved_nodes[1].parent_id is None

    assert graph.validate_consistency()