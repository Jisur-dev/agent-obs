
import pytest

from agent_obs.versioning.db import (
    init_db,
    make_engine,
    make_session_factory,
)
from agent_obs.versioning.repository import SqliteVersionRepository


@pytest.fixture
def repository():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)

    factory = make_session_factory(engine)
    repo = SqliteVersionRepository("agent-1", factory)

    yield repo

    engine.dispose()
