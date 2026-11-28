
import pytest

from .test_helpers import create_test_version


def test_save_version(repository):
    """Test that a valid version can be saved to the repository."""
    version = create_test_version("v1")

    repository.save(version)

    assert repository.get("v1") == version


def test_save_duplicate_version(repository):
    """Test that saving a version with an existing ID raises ValueError."""
    version = create_test_version("v1")

    repository.save(version)

    with pytest.raises(ValueError):
        repository.save(version)
