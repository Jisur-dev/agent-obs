from sqlalchemy import inspect

from agent_obs.tracing.persistence.database import engine, create_tables


def test_create_tables():
    create_tables()

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    assert "graphs" in tables
    assert "runs" in tables
    assert "nodes" in tables
    assert "edges" in tables
    