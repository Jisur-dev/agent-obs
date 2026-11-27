from agent_obs.versioning.artifact import (
    AgentConfiguration,
    PromptArtifact,
)
from agent_obs.versioning.fingerprint import FingerprintComputer
from agent_obs.versioning.repository import Version


def create_test_version(
    version_id: str,
    parent_version_id: str | None = None,
    system_prompt: str = "You are a helpful assistant.",
) -> Version:
    """Create a Version object for repository tests.

    Args:
        version_id: The unique identifier of the test version.
        parent_version_id: The identifier of the parent version, if any.
        system_prompt: The system prompt used by the test artifact.

    Returns:
        A Version object configured for testing.
    """
    artifact = PromptArtifact(
        system_prompt=system_prompt,
        model="test-model",
        model_parameters={
            "temperature": 0.7,
            "top_p": 1.0,
            "max_tokens": 500,
        },
    )

    configuration = AgentConfiguration(
        prompt=artifact,
    )

    return Version(
        version_id=version_id,
        parent_version_id=parent_version_id,
        configuration=configuration,
        author="test-author",
        message=f"Create {version_id}",
        created_at=1.0,
        fingerprint=FingerprintComputer.compute(configuration),
    )
