import pytest
from agent_obs.versioning.repository import Version, SqliteVersionRepository
from agent_obs.versioning.artifact import AgentConfiguration, PromptArtifact


def make_config() -> AgentConfiguration:
    return AgentConfiguration(
        prompt=PromptArtifact(
            system_prompt="You are a test agent.",
            examples=[{"input": "hi", "output": "hello"}],
            model="test-model",
        ),
        tool_versions={"search": "1.0"},
        guardrails=["no-pii"],
    )

def make_version(vid, parent=None, fp=None):
    return Version(
        version_id=vid,
        parent_version_id=parent,
        configuration=make_config(),       # build a minimal AgentConfiguration here
        author="alice",
        message=f"commit {vid}",
        created_at=1_700_000_000.0,
        fingerprint=fp or f"fp-{vid}",
    )


def test_save_and_get_roundtrip(repo):
    v = make_version("v1")
    repo.save(v)
    assert repo.get("v1") == v


def test_get_unknown_raises_keyerror(repo):
    with pytest.raises(KeyError):
        repo.get("nope")


def test_duplicate_id_rejected(repo):
    repo.save(make_version("v1"))
    assert repo.check_version(make_version("v1")) is False
    with pytest.raises(ValueError):
        repo.save(make_version("v1"))


def test_unknown_parent_rejected(repo):
    with pytest.raises(ValueError):
        repo.save(make_version("v2", parent="ghost"))


def test_list_preserves_save_order(repo):
    repo.save(make_version("b"))
    repo.save(make_version("a", parent="b"))
    repo.save(make_version("c", parent="a"))
    assert [v.version_id for v in repo.list_versions()] == ["b", "a", "c"]


def test_get_latest(repo):
    assert repo.get_latest() is None
    repo.save(make_version("v1"))
    repo.save(make_version("v2", parent="v1"))
    assert repo.get_latest().version_id == "v2"


# --- SQLite-only ---

def test_agents_are_isolated(factory):
    a = SqliteVersionRepository("agent-a", factory)
    b = SqliteVersionRepository("agent-b", factory)
    a.save(make_version("v1"))
    assert b.list_versions() == []
    with pytest.raises(KeyError):
        b.get("v1")


def test_data_survives_new_engine(tmp_path):
    from agent_obs.versioning.db import init_db, make_engine, make_session_factory
    url = f"sqlite:///{tmp_path / 'v.db'}"
    for i in range(2):                       # two "process lifetimes"
        engine = make_engine(url)
        init_db(engine)
        repo = SqliteVersionRepository("agent-1", make_session_factory(engine))
        if i == 0:
            repo.save(make_version("v1"))
        else:
            assert repo.get("v1").author == "alice"
        engine.dispose()