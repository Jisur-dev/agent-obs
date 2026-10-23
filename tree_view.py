from agent_obs.tracing.trace_graph.execution_graph import ExecutionTree
from agent_obs.tracing.trace_graph.enums import NodeType
from agent_obs.tracing.trace_graph.trace_node import TraceNode
from rich.tree import Tree
from rich import print
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def render_tree(execution_tree):
    if not execution_tree.root_node_id:
        return
    root_node = execution_tree.get_node(execution_tree.root_node_id)
    tree = Tree(root_node.node_type.value)

    def build_tree(node_id, parent_branch):
        child_nodes = execution_tree.children(node_id)
        for child_node in child_nodes:
            branch = parent_branch.add(child_node.node_type.value)
            build_tree(child_node.node_id, branch)
    build_tree(execution_tree.root_node_id, tree)
    print(tree)


if __name__ == "__main__":
    execution_tree = ExecutionTree()

    n1 = TraceNode(node_id="1", node_type=NodeType.USER_MESSAGE)
    execution_tree.add_node(n1)

    n2 = TraceNode(node_id="2", node_type=NodeType.AGENT, parent_id="1")
    execution_tree.add_node(n2)

    n3 = TraceNode(node_id="3", node_type=NodeType.TOOL_CALL, parent_id="2")
    execution_tree.add_node(n3)

    n4 = TraceNode(node_id="4", node_type=NodeType.LLM_CALL, parent_id="2")
    execution_tree.add_node(n4)

    n5 = TraceNode(node_id="5", node_type=NodeType.TOOL_CALL, parent_id="3")
    execution_tree.add_node(n5)

    render_tree(execution_tree)
