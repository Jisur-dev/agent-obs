"""Executes compensations in reverse (LIFO) order when a run fails.

E.g. Book hotel -> Buy flight -> Rent car FAILS, so compensation runs
Cancel flight -> Cancel hotel, in reverse completion order.
"""

from __future__ import annotations

from dataclasses import dataclass

from .registry import CompensationRegistry


@dataclass
class CompletedAction:
    """A record of one successfully completed action, kept for rollback."""

    name: str
    parameters: dict
    result: object


class CompensationExecutor:
    """Given the list of actions completed so far, executes their
    compensations in LIFO order with retry and idempotency handling.
    """

    def __init__(
        self,
        compensation_registry: CompensationRegistry,
        retry_policy: dict | None = None,
    ):
        pass

    def compensate(self, completed_actions: list[CompletedAction]) -> None:
        """Roll back completed actions in reverse order."""
        pass

    def _execute_with_retry(self, compensation, action: CompletedAction) -> None:
        """Run a single compensation with retry/idempotency handling."""
        pass
