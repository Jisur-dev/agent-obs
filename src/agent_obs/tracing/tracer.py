"""Instrumentation entry points: core context managers for tracing."""

from __future__ import annotations
import functools    
import inspect
import time
from contextlib import contextmanager
from typing import Any, Callable
from .execution_context import ExecutionContext
from .trace_graph.execution_graph import ExecutionTree
from .tracing_repository import InMemoryTracingRepository
from .trace_graph.enums import NodeType
from .trace_graph.trace_node import TraceNode


class Tracer:
    """Primary user-facing API for instrumenting agent execution traces.

    Handles the core tracing lifecycle, managing node contexts, execution duration,
    and persisting completed trace nodes to the designated backend storage.
    """

    def __init__(self,tracing_repository: InMemoryTracingRepository | None = None):
        self.graph = ExecutionTree()

        self.tracing_repository = (
            tracing_repository
            if tracing_repository is not None
            else InMemoryTracingRepository()
        )

        self.context = ExecutionContext(
            run_id=self.graph.run_id
        )

    @contextmanager
    def span(self, node_type: NodeType, **metadata: Any):
        """Core context manager that opens, manages, and closes a trace node lifecycle."""

        node = TraceNode(node_type=node_type)

        for key, value in metadata.items():
            if hasattr(node, key):
                setattr(node, key, value)

        self.context.push_node(node)

        try:
            self.graph.add_node(node)
        except Exception:
            self.context.pop_node()
            raise

        start_counter = time.perf_counter()

        try:
            yield node

        except Exception as exc:
            node.set_exception(exc)
            raise

        finally:
            latency_ms = (time.perf_counter() - start_counter) * 1000

            node.set_latency(latency_ms=latency_ms)

            self.context.pop_node()

            # Persist the complete graph when the outermost span has finished.
            if (self.tracing_repository is not None and self.context.get_current_node() is None):
                self.tracing_repository.save(self.graph)
                
    @contextmanager
    def trace(
        self,
        name: str,
        node_type: NodeType = NodeType.AGENT,
        **metadata: Any,
    ):
        """Named entry-point context manager for structuring trace phases."""

        with self.span(
            node_type,
            **metadata,
        ) as node:

            if name and hasattr(node, "tags"):
                node.tags.append(name)

            yield node

    def trace_tool_call(self,func: Callable | None = None,**metadata: Any,) -> Callable:
        """Decorator that auto-instruments a tool execution."""

        def decorator(f: Callable) -> Callable:

            @functools.wraps(f)
            def wrapper(
                *args: Any,
                **kwargs: Any,
            ) -> Any:

                inputs = {
                    "args": args,
                    "kwargs": kwargs,
                }

                span_meta = {
                    **metadata,
                    "input_data": inputs,
                }

                with self.span(
                    NodeType.TOOL_CALL,
                    **span_meta,
                ) as node:

                    # Capture tool parameters cleanly across positional & keyword args
                    try:
                        signature = inspect.signature(f)

                        bound_arguments = signature.bind_partial(
                            *args,
                            **kwargs,
                        )

                        bound_arguments.apply_defaults()

                        node.tool_parameters = dict(
                            bound_arguments.arguments
                        )

                    except (TypeError, ValueError):
                        node.tool_parameters = {
                            "args": args,
                            "kwargs": kwargs,
                        }

                    # Execute the actual tool
                    result = f(
                        *args,
                        **kwargs,
                    )

                    # Capture tool results & outputs
                    node.tool_result = result

                    node.set_output(result)

                    return result

            return wrapper

        if func is None:
            return decorator

        return decorator(func)

    def trace_agent(
        self,
        func: Callable | None = None,
        **metadata: Any,
    ) -> Callable:
        """Decorator that auto-instruments an agent execution."""

        def decorator(f: Callable) -> Callable:

            @functools.wraps(f)
            def wrapper(
                *args: Any,
                **kwargs: Any,
            ) -> Any:

                inputs = {
                    "args": args,
                    "kwargs": kwargs,
                }

                span_meta = {
                    **metadata,
                    "input_data": inputs,
                }

                with self.span(
                    NodeType.AGENT,
                    **span_meta,
                ) as node:

                    result = f(
                        *args,
                        **kwargs,
                    )

                    node.set_output(result)

                    return result

            return wrapper

        if func is None:
            return decorator

        return decorator(func)

    def trace_llm_call(
        self,
        func: Callable | None = None,
        **metadata: Any,
    ) -> Callable:
        """Decorator that auto-instruments an LLM execution."""

        def decorator(f: Callable) -> Callable:

            @functools.wraps(f)
            def wrapper(
                *args: Any,
                **kwargs: Any,
            ) -> Any:

                inputs = {
                    "args": args,
                    "kwargs": kwargs,
                }

                # Capture common LLM parameters
                model = metadata.get(
                    "model",
                    kwargs.get("model"),
                )

                temperature = metadata.get(
                    "temperature",
                    kwargs.get("temperature"),
                )

                seed = metadata.get(
                    "seed",
                    kwargs.get("seed"),
                )

                retry_count = metadata.get(
                    "retry_count",
                    0,
                )

                span_meta = {
                    **metadata,
                    "input_data": inputs,
                    "model": model,
                    "temperature": temperature,
                    "seed": seed,
                    "retry_count": retry_count,
                }

                with self.span(
                    NodeType.LLM_CALL,
                    **span_meta,
                ) as node:

                    # Execute the actual LLM call
                    result = f(
                        *args,
                        **kwargs,
                    )

                    # Capture LLM output
                    node.set_output(result)

                    return result

            return wrapper

        if func is None:
            return decorator

        return decorator(func)

    def get_graph(self) -> ExecutionTree:
        """Return the execution graph."""

        return self.graph