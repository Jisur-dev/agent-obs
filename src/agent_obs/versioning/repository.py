from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .artifact import AgentConfiguration
from .dao import VersionDAO
from .models import ConfigurationRow, VersionRow


@dataclass(frozen=True)
class Version:
    """A single committed snapshot of an agent configuration."""

    version_id: str
    parent_version_id: str | None
    configuration: AgentConfiguration
    author: str
    message: str
    created_at: float
    fingerprint: str


@dataclass
class ConfigurationDiff:
    """The set of field-level changes between two configuration versions."""

    version_a: str
    version_b: str
    changed_fields: dict[str, tuple[object, object]]


class VersionStore(Protocol):
    """Interface for persistent version repositories."""

    agent_id: str

    def save(self, version: Version) -> None: ...

    def get(self, version_id: str) -> Version: ...

    def list_versions(self) -> list[Version]: ...

    def get_latest(self) -> Version | None: ...

    def check_version(self, version: Version) -> bool: ...


def _configuration_to_row(
    configuration: AgentConfiguration,
) -> ConfigurationRow:
    """Convert an AgentConfiguration into a database configuration row."""

    prompt = configuration.prompt

    return ConfigurationRow(
        tool_versions=configuration.tool_versions,
        retrieval_strategy=configuration.retrieval_strategy,
        memory_configuration=configuration.memory_configuration,
        guardrails=configuration.guardrails,
        retry_policy=configuration.retry_policy,
        planner_configuration=configuration.planner_configuration,
        prompt_system_prompt=prompt.system_prompt,
        prompt_examples=list(prompt.examples),
        prompt_tools=list(prompt.tools),
        prompt_model=prompt.model,
        prompt_model_parameters=prompt.model_parameters,
        prompt_response_schema=prompt.response_schema,
        prompt_safety_instructions=prompt.safety_instructions,
        prompt_memory_template=prompt.memory_template,
        prompt_retrieval_template=prompt.retrieval_template,
    )


def _configuration_to_domain(row: ConfigurationRow) -> AgentConfiguration:
    """Convert a database configuration row into a domain object."""

    return AgentConfiguration.from_dict(
        {
            "tool_versions": row.tool_versions,
            "retrieval_strategy": row.retrieval_strategy,
            "memory_configuration": row.memory_configuration,
            "guardrails": row.guardrails,
            "retry_policy": row.retry_policy,
            "planner_configuration": row.planner_configuration,
            "prompt": {
                "system_prompt": row.prompt_system_prompt,
                "examples": row.prompt_examples,
                "tools": row.prompt_tools,
                "model": row.prompt_model,
                "model_parameters": row.prompt_model_parameters,
                "response_schema": row.prompt_response_schema,
                "safety_instructions": row.prompt_safety_instructions,
                "memory_template": row.prompt_memory_template,
                "retrieval_template": row.prompt_retrieval_template,
            },
        }
    )


def _to_row(agent_id: str, version: Version) -> VersionRow:
    """Convert a domain Version into a database row."""

    return VersionRow(
        version_id=version.version_id,
        agent_id=agent_id,
        parent_version_id=version.parent_version_id,
        configuration=_configuration_to_row(version.configuration),
        author=version.author,
        message=version.message,
        created_at=version.created_at,
        fingerprint=version.fingerprint,
    )


def _to_domain(row: VersionRow) -> Version:
    """Convert a database row into a domain Version."""

    return Version(
        version_id=row.version_id,
        parent_version_id=row.parent_version_id,
        configuration=_configuration_to_domain(row.configuration),
        author=row.author,
        message=row.message,
        created_at=row.created_at,
        fingerprint=row.fingerprint,
    )


class SqliteVersionRepository:
    """Repository for persistent agent configuration versions."""

    def __init__(self, agent_id: str, session_factory):
        self.agent_id = agent_id
        self.version_dao = VersionDAO(session_factory)

    def save(self, version: Version) -> None:
        """Save a version, rejecting duplicate IDs and unknown parents."""

        if not self.check_version(version):
            raise ValueError(f"Version '{version.version_id}' already exists")

        if version.parent_version_id is not None:
            parent = self.version_dao.get(
                self.agent_id,
                version.parent_version_id,
            )

            if parent is None:
                raise ValueError(
                    f"Parent version '{version.parent_version_id}' does not exist"
                )

        self.version_dao.create(_to_row(self.agent_id, version))

    def get(self, version_id: str) -> Version:
        """Retrieve a version belonging to this agent."""

        row = self.version_dao.get(self.agent_id, version_id)

        if row is None:
            raise KeyError(version_id)

        return _to_domain(row)

    def list_versions(self) -> list[Version]:
        """Return this agent's versions in save order."""

        rows = self.version_dao.list_by_agent(self.agent_id)
        return [_to_domain(row) for row in rows]

    def get_latest(self) -> Version | None:
        """Return this agent's latest saved version, or None."""

        row = self.version_dao.get_latest(self.agent_id)
        return _to_domain(row) if row is not None else None

    def check_version(self, version: Version) -> bool:
        """Return whether this version ID is available for this agent."""

        return not self.version_dao.exists(
            self.agent_id,
            version.version_id,
        )
