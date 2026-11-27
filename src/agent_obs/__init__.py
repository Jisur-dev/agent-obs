"""agent-obs: observability, prompt/agent configuration versioning, and
reliability middleware for AI agents.

The core idea: an agent execution is a first-class object that can be
inspected, replayed, diffed, versioned, and evaluated -- not just logged.
"""

__version__ = "0.1.0"


from .tracing import Tracer

__all__ = [
    "Tracer",
    "__version__",
]
