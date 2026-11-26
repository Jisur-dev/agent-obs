
import pytest

from .test_helpers import create_test_version


def test_get_existing_version(repository):
    """Test that an existing version can be retrieved by its ID."""
    version = create_test_version("v1")
    repository.save(version)

    result = repository.get("v1")

    assert result == version


def test_get_nonexistent_version(repository):
    """Test that getting a nonexistent version raises KeyError."""
    with pytest.raises(KeyError):
        repository.get("does-not-exist")
