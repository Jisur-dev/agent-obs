from __future__ import annotations

import enum

class NodeType(str, enum.Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    AGENT = "agent"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    USER_MESSAGE = "user_message"
    RETRY = "retry"
    HUMAN_APPROVAL = "human_approval"

class EdgeType(str, enum.Enum):
    PARENT_CHILD = "parent_child"
    DELEGATED_TO = "delegated_to"
    TOOL_CALL = "tool_call"
    MEMORY_READ = "memory_read"
    RETURNED_DATA = "returned_data"