
"""Tests for the public tracing API."""

from agent_obs import Tracer
from agent_obs.tracing import (
    TracingRepository,
    InMemoryTracingRepository,
)
from agent_obs.tracing.persistence import (
    SQLAlchemyTracingRepository,
    get_session,
    create_tables,
)


def test_root_public_api():
    assert Tracer is not None


def test_tracing_public_api():
    assert TracingRepository is not None
    assert InMemoryTracingRepository is not None
    assert issubclass(InMemoryTracingRepository, TracingRepository)


def test_persistence_public_api():
    assert SQLAlchemyTracingRepository is not None
    assert callable(get_session)
    assert callable(create_tables)
