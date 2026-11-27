"""Saga-style compensation middleware for agent tool execution.

When an agent performs a multi-step task, a failure during a later step can
leave earlier actions incomplete or unhandled in external systems. This
middleware tracks completed actions and automatically executes the matching
compensations if the overall process fails.
"""

from __future__ import annotations

from .registry import ActionRegistry, CompensationRegistry


class CompensationMiddleware:
    """Wraps agent tool execution with automatic saga-style rollback.

    Core responsibilities: action tracking, automatic rollback in reverse
    order on failure, state continuity across pauses/restarts, and
    configurable reversal logic kept out of the primary action code.
    """

    def __init__(
        self,
        action_registry: ActionRegistry | None = None,
        compensation_registry: CompensationRegistry | None = None,
        state_store=None,
        logger=None,
    ):
        pass

    def execute(self, action_name: str, **parameters):
        """Execute an action, recording it for potential compensation."""
        pass

    def handle_failure(self, exc: Exception) -> None:
        """Entry point invoked on unhandled failure: triggers compensation."""
        pass
