import pytest
from agent_obs.versioning.db import init_db, make_engine, make_session_factory
from agent_obs.versioning.repository import InMemoryVersionRepository, SqliteVersionRepository


@pytest.fixture
def factory():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    return make_session_factory(engine)


@pytest.fixture(params=["memory", "sqlite"])
def repo(request, factory):
    if request.param == "memory":
        return InMemoryVersionRepository("agent-1")
    return SqliteVersionRepository("agent-1", factory)