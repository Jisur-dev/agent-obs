
# dataclass allows us to create a small configuration object
# with fields that our VersionDifferenceCalculator can compare.
from dataclasses import dataclass

# pytest is used for running tests and testing multiple cases.
import pytest

# Import the class we want to test.
from agent_obs.versioning.version_difference import VersionDifferenceCalculator

# Import the real project classes.
from agent_obs.versioning.repository import Version, ConfigurationDiff


# ---------------------------------------------------------
# 1. A simplified configuration for our tests
# ---------------------------------------------------------

@dataclass
class DummyConfiguration:
    """
    A simplified configuration used only for testing.

    It contains three fields so we can test different changes.
    """

    prompt: str | None
    tool_versions: dict
    retrieval_strategy: str


# ---------------------------------------------------------
# 2. Helper function to create Version objects
# ---------------------------------------------------------

def make_version(version_id: str, configuration: DummyConfiguration) -> Version:
    """
    Create a Version object for testing.

    Args:
        version_id (str): The unique ID of the version.
        configuration (DummyCon+figuration): Its configuration.

    Returns:
        Version: A Version object containing the configuration.
    """

    return Version(
        version_id=version_id,
        parent_version_id=None,
        configuration=configuration,
        author="test_user",
        message="Test version",
        created_at=1000.0,
        fingerprint=f"fingerprint_{version_id}",
    )


# ---------------------------------------------------------
# 3. Test identical configurations
# ---------------------------------------------------------

def test_identical_configurations():
    """Two identical configurations should produce no differences."""

    # Create two separate configuration objects with identical values.
    config_a = DummyConfiguration(
        prompt="Hello",
        tool_versions={"search": "1.0"},
        retrieval_strategy="basic",
    )

    config_b = DummyConfiguration(
        prompt="Hello",
        tool_versions={"search": "1.0"},
        retrieval_strategy="basic",
    )

    # Put each configuration inside a Version object.
    version_a = make_version("v1", config_a)
    version_b = make_version("v2", config_b)

    # Compare the two versions.
    result = VersionDifferenceCalculator.compare(version_a, version_b)

    # The result must be a ConfigurationDiff object.
    assert isinstance(result, ConfigurationDiff)

    # Check that the correct version IDs are returned.
    assert result.version_a == "v1"
    assert result.version_b == "v2"

    # No fields changed, so the dictionary must be empty.
    assert result.changed_fields == {}


# ---------------------------------------------------------
# 4. Test when only one field changes
# ---------------------------------------------------------

def test_one_changed_field():
    """Only the field that changed should appear in changed_fields."""

    config_a = DummyConfiguration("Hello", {"search": "1.0"}, "basic")
    config_b = DummyConfiguration("Goodbye", {"search": "1.0"}, "basic")

    version_a = make_version("v1", config_a)
    version_b = make_version("v2", config_b)

    result = VersionDifferenceCalculator.compare(version_a, version_b)

    # The prompt changed from "Hello" to "Goodbye".
    # The other two fields stayed the same.
    assert result.changed_fields == {
        "prompt": ("Hello", "Goodbye")
    }


# ---------------------------------------------------------
# 5. Test when all fields change
# ---------------------------------------------------------

def test_all_fields_changed():
    """All modified fields should be detected."""

    config_a = DummyConfiguration("Hello", {"search": "1.0"}, "basic")
    config_b = DummyConfiguration("Goodbye", {"search": "2.0"}, "advanced")

    version_a = make_version("v1", config_a)
    version_b = make_version("v2", config_b)

    result = VersionDifferenceCalculator.compare(version_a, version_b)

    # Every field has a different value.
    # Each tuple contains (old_value, new_value).
    assert result.changed_fields == {
        "prompt": ("Hello", "Goodbye"),
        "tool_versions": ({"search": "1.0"}, {"search": "2.0"}),
        "retrieval_strategy": ("basic", "advanced"),
    }


# ---------------------------------------------------------
# 6. Test None values
# ---------------------------------------------------------

def test_none_to_value():
    """A change from None to a normal value should be detected."""

    config_a = DummyConfiguration(None, {}, "basic")
    config_b = DummyConfiguration("Hello", {}, "basic")

    version_a = make_version("v1", config_a)
    version_b = make_version("v2", config_b)

    result = VersionDifferenceCalculator.compare(version_a, version_b)

    assert result.changed_fields == {
        "prompt": (None, "Hello")
    }


def test_value_to_none():
    """A change from a normal value to None should be detected."""

    config_a = DummyConfiguration("Hello", {}, "basic")
    config_b = DummyConfiguration(None, {}, "basic")

    version_a = make_version("v1", config_a)
    version_b = make_version("v2", config_b)

    result = VersionDifferenceCalculator.compare(version_a, version_b)

    assert result.changed_fields == {
        "prompt": ("Hello", None)
    }


# ---------------------------------------------------------
# 7. Test empty values
# ---------------------------------------------------------

def test_empty_values_to_non_empty_values():
    """Changes from empty strings and dictionaries should be detected."""

    config_a = DummyConfiguration("", {}, "")
    config_b = DummyConfiguration("Hello", {"search": "1.0"}, "basic")

    version_a = make_version("v1", config_a)
    version_b = make_version("v2", config_b)

    result = VersionDifferenceCalculator.compare(version_a, version_b)

    assert result.changed_fields == {
        "prompt": ("", "Hello"),
        "tool_versions": ({}, {"search": "1.0"}),
        "retrieval_strategy": ("", "basic"),
    }


# ---------------------------------------------------------
# 8. Test dictionary changes
# ---------------------------------------------------------

def test_dictionary_value_changed():
    """A change inside a dictionary should be detected."""

    config_a = DummyConfiguration(
        "Hello",
        {"search": "1.0", "calculator": "1.0"},
        "basic",
    )

    config_b = DummyConfiguration(
        "Hello",
        {"search": "1.0", "calculator": "2.0"},
        "basic",
    )

    version_a = make_version("v1", config_a)
    version_b = make_version("v2", config_b)

    result = VersionDifferenceCalculator.compare(version_a, version_b)

    # The calculator version changed, so the whole dictionary field
    # is reported as changed.
    assert result.changed_fields == {
        "tool_versions": (
            {"search": "1.0", "calculator": "1.0"},
            {"search": "1.0", "calculator": "2.0"},
        )
    }


def test_dictionary_key_added():
    """Adding a dictionary key should be detected."""

    config_a = DummyConfiguration(
        "Hello",
        {"search": "1.0"},
        "basic",
    )

    config_b = DummyConfiguration(
        "Hello",
        {"search": "1.0", "calculator": "1.0"},
        "basic",
    )

    version_a = make_version("v1", config_a)
    version_b = make_version("v2", config_b)

    result = VersionDifferenceCalculator.compare(version_a, version_b)

    assert result.changed_fields == {
        "tool_versions": (
            {"search": "1.0"},
            {"search": "1.0", "calculator": "1.0"},
        )
    }


def test_dictionary_key_removed():
    """Removing a dictionary key should be detected."""

    config_a = DummyConfiguration(
        "Hello",
        {"search": "1.0", "calculator": "1.0"},
        "basic",
    )

    config_b = DummyConfiguration(
        "Hello",
        {"search": "1.0"},
        "basic",
    )

    version_a = make_version("v1", config_a)
    version_b = make_version("v2", config_b)

    result = VersionDifferenceCalculator.compare(version_a, version_b)

    assert result.changed_fields == {
        "tool_versions": (
            {"search": "1.0", "calculator": "1.0"},
            {"search": "1.0"},
        )
    }


# ---------------------------------------------------------
# 9. Test reversed versions
# ---------------------------------------------------------

def test_reversed_versions():
    """Reversing the versions should reverse old and new values."""

    config_a = DummyConfiguration("Hello", {}, "basic")
    config_b = DummyConfiguration("Goodbye", {}, "basic")

    version_a = make_version("v1", config_a)
    version_b = make_version("v2", config_b)

    # Notice that we pass version_b first this time.
    result = VersionDifferenceCalculator.compare(version_b, version_a)

    assert result.version_a == "v2"
    assert result.version_b == "v1"

    # The old and new values should also be reversed.
    assert result.changed_fields == {
        "prompt": ("Goodbye", "Hello")
    }


# ---------------------------------------------------------
# 10. Test comparing a version with itself
# ---------------------------------------------------------

def test_compare_version_with_itself():
    """Comparing a version with itself should produce no differences."""

    config = DummyConfiguration("Hello", {"search": "1.0"}, "basic")

    version = make_version("v1", config)

    result = VersionDifferenceCalculator.compare(version, version)

    assert result.version_a == "v1"
    assert result.version_b == "v1"
    assert result.changed_fields == {}
