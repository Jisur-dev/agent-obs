import pytest

from agent_obs.versioning.repository import VersionRepository
from agent_obs.versioning.service import VersioningService
from tests.test_helpers import make_configuration


@pytest.fixture
def service() -> VersioningService:
    """Return a versioning service with an empty repository.

    If the repository class changes (PR #30), this is the only place to edit.
    """
    return VersioningService(VersionRepository(agent_id="agent-1"))


def test_first_configuration_is_saved(service):
    """Verify that an empty repository always saves the first configuration."""
    version = service.commit_if_changed(
        make_configuration(), author="Dor3i", message="first"
    )

    assert version is not None
    assert len(service.repository.list_versions()) == 1


def test_identical_configuration_is_skipped(service):
    """Verify that the same configuration twice returns None and saves nothing."""
    service.commit_if_changed(make_configuration(), author="Dor3i", message="first")

    second = service.commit_if_changed(
        make_configuration(), author="Dor3i", message="again"
    )

    assert second is None
    assert len(service.repository.list_versions()) == 1


def test_changed_configuration_creates_child_version(service):
    """Verify that a changed configuration is saved with the previous one as parent."""
    first = service.commit_if_changed(
        make_configuration(), author="Dor3i", message="first"
    )

    second = service.commit_if_changed(
        make_configuration(prompt_overrides={"system_prompt": "Different."}),
        author="Dor3i",
        message="changed",
    )

    assert second is not None
    assert second.parent_version_id == first.version_id
    assert len(service.repository.list_versions()) == 2
