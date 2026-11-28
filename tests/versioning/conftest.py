
import pytest

from agent_obs.versioning.db import (
    init_db,
    make_engine,
    make_session_factory,
)
from agent_obs.versioning.repository import SqliteVersionRepository


@pytest.fixture
def factory():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)

    yield make_session_factory(engine)

    engine.dispose()


@pytest.fixture
def repo(factory):
    return SqliteVersionRepository("agent-1", factory)
