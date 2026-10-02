from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time
import uuid
from .enums import NodeType

@dataclass
class TraceNode:
    """
    Role:
        Represents a single atomic unit of agent execution.

    Type:
        Classifies each step using NodeType (e.g., LLM_CALL, TOOL_CALL).

    When & Why:
        Instantiated and tracked during workflow execution to capture 
        comprehensive telemetry, inputs, outputs, latency, and cost per step.

    Exception Tracking:
        Records runtime errors and exceptions natively to enable robust 
        debugging, root-cause analysis, and system auditing.

    Output:
        Exposes a raw telemetry data structure ready for serialization 
        and persistence.
    """

    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType = NodeType.LLM_CALL
    parent_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    prompt_hash: str | None = None
    model: str | None = None
    temperature: float | None = None
    seed: int | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    input_data: Any = None
    output_data: Any = None
    memory_snapshot: dict | None = None
    tool_parameters: dict | None = None
    tool_result: Any = None
    exception: str | None = None
    retry_count: int = 0
    tags: list[str] = field(default_factory=list)

    def set_output(self, output):
        """
        Role:
            Stores the resulting data produced by this execution step.

        When & Why:
            Called upon successful completion of an operation (e.g., LLM text output 
            or tool result) to capture and persist the final output payload.

        Output:
            Updates the internal `output_data` field of the node.
        """
        self.output_data = output


    def set_latency(self, latency_ms: float | None = None) -> None:
        """
        Role:
            Finalizes the lifespan of the node and tracks execution duration.

        When & Why:
            Invoked at the end of a step to measure performance. If explicit 
            latency is provided, it uses it; otherwise, it computes the exact 
            elapsed time in milliseconds from the node's creation timestamp.

        Output:
            Updates the internal `latency_ms` field with the total execution time.
        """
        if latency_ms is not None:
            self.latency_ms = latency_ms
        else:
            self.latency_ms = (time.time() - self.timestamp) * 1000

    def set_exception(self, exc):
        """
        Role:
            Captures, formats, and stores runtime errors and exceptions.

        When & Why:
            Triggered automatically when an unexpected error or failure occurs 
            during a workflow step, ensuring the exception type and message 
            are safely recorded for root-cause analysis and debugging.

        Output:
            Updates the internal `exception` field with a formatted string 
            containing both the error name and its message.
        """
        error_name = type(exc).__name__
        error_message = str(exc)
        self.exception = error_name + ": " + error_message