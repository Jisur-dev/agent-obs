from .test_helpers import create_test_version


def test_list_versions_empty(repository):
    """Test that listing an empty repository returns an empty list."""
    result = repository.list_versions()

    assert result == []


def test_list_versions(repository):
    """Test that listing the repository returns saved versions."""
    version = create_test_version("v1")
    repository.save(version)

    result = repository.list_versions()

    assert result == [version]


def test_list_versions_preserves_save_order(repository):
    """Test that versions are returned in their save order."""
    v1 = create_test_version("v1")
    v2 = create_test_version("v2", "v1")
    v3 = create_test_version("v3", "v2")

    repository.save(v1)
    repository.save(v2)
    repository.save(v3)

    result = repository.list_versions()

    assert result == [v1, v2, v3]
