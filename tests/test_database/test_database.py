
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from agent_obs.tracing.persistence.database import (
    engine,
    create_tables,
    get_session,
)


def test_create_tables():
    create_tables()

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    assert "graphs" in tables
    assert "runs" in tables
    assert "nodes" in tables
    assert "edges" in tables


def test_get_session_returns_session():
    session = get_session()

    try:
        assert isinstance(session, Session)
        assert session.execute(text("SELECT 1")).scalar() == 1
    finally:
        session.close()
