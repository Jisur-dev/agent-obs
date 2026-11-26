
from agent_obs.versioning.dao import ConfigurationDAO, VersionDAO
from agent_obs.versioning.models import ConfigurationRow, VersionRow


def create_test_configuration():
    return ConfigurationRow(
        tool_versions={"python": "3.14"},
        guardrails=["no harmful output"],
        prompt_system_prompt="You are a helpful assistant.",
        prompt_examples=[],
        prompt_tools=[],
        prompt_model="test-model",
        prompt_model_parameters={"temperature": 0.7},
    )


def create_test_version(configuration):
    return VersionRow(
        version_id="v1",
        agent_id="agent-1",
        parent_version_id=None,
        configuration=configuration,
        author="test-author",
        message="Initial version",
        created_at=1.0,
        fingerprint="a" * 64,
    )


# ConfigurationDAO tests

def test_configuration_dao_create_and_get(factory):
    dao = ConfigurationDAO(factory)
    configuration = create_test_configuration()

    configuration_id = dao.create(configuration)
    result = dao.get(configuration_id)

    assert result is not None
    assert result.prompt_model == "test-model"


def test_configuration_dao_update(factory):
    dao = ConfigurationDAO(factory)
    configuration = create_test_configuration()

    configuration_id = dao.create(configuration)

    configuration.seq = configuration_id
    configuration.prompt_model = "updated-model"
    dao.update(configuration)

    result = dao.get(configuration_id)

    assert result.prompt_model == "updated-model"


def test_configuration_dao_delete(factory):
    dao = ConfigurationDAO(factory)
    configuration = create_test_configuration()

    configuration_id = dao.create(configuration)

    assert dao.delete(configuration_id) is True
    assert dao.get(configuration_id) is None
    assert dao.delete(configuration_id) is False


def test_configuration_dao_exists(factory):
    dao = ConfigurationDAO(factory)
    configuration = create_test_configuration()

    configuration_id = dao.create(configuration)

    assert dao.exists(configuration_id) is True
    assert dao.exists(9999) is False


# VersionDAO tests

def test_version_dao_create_and_get(factory):
    dao = VersionDAO(factory)
    configuration = create_test_configuration()
    version = create_test_version(configuration)

    dao.create(version)
    result = dao.get("agent-1", "v1")

    assert result is not None
    assert result.version_id == "v1"
    assert result.configuration.prompt_model == "test-model"


def test_version_dao_update(factory):
    dao = VersionDAO(factory)
    version = create_test_version(create_test_configuration())

    dao.create(version)

    version.message = "Updated message"
    dao.update(version)

    result = dao.get("agent-1", "v1")

    assert result.message == "Updated message"


def test_version_dao_delete(factory):
    dao = VersionDAO(factory)
    version = create_test_version(create_test_configuration())

    dao.create(version)

    assert dao.delete("agent-1", "v1") is True
    assert dao.get("agent-1", "v1") is None
    assert dao.delete("agent-1", "v1") is False


def test_version_dao_exists(factory):
    dao = VersionDAO(factory)
    version = create_test_version(create_test_configuration())

    dao.create(version)

    assert dao.exists("agent-1", "v1") is True
    assert dao.exists("agent-1", "unknown") is False
