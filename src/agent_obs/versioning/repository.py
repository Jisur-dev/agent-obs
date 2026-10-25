
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .artifact import AgentConfiguration
from .blame import FieldChange
from .db import session_scope
from .models import VersionRow


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
    """Interface shared by in-memory and persistent version repositories."""

    agent_id: str

    def save(self, version: Version) -> None: ...

    def get(self, version_id: str) -> Version: ...

    def list_versions(self) -> list[Version]: ...

    def get_latest(self) -> Version | None: ...

    def check_version(self, version: Version) -> bool: ...


class InMemoryVersionRepository:
    """Store versions in memory, preserving insertion order."""

    def __init__(self, agent_id: str, storage=None):
        self.agent_id = agent_id
        self._versions: dict[str, Version] = {}

    def save(self, version: Version) -> None:
        """Save a version, rejecting duplicate IDs and unknown parents."""
        if not self.check_version(version):
            raise ValueError(
                f"Version '{version.version_id}' already exists"
            )

        if (
            version.parent_version_id is not None
            and version.parent_version_id not in self._versions
        ):
            raise ValueError(
                f"Parent version '{version.parent_version_id}' does not exist"
            )

        self._versions[version.version_id] = version

    def get(self, version_id: str) -> Version:
        """Retrieve a version or raise KeyError if it does not exist."""
        return self._versions[version_id]

    def list_versions(self) -> list[Version]:
        """Return versions in insertion order."""
        return list(self._versions.values())

    def get_latest(self) -> Version | None:
        """Return the most recently saved version, or None."""
        if not self._versions:
            return None
        return next(reversed(self._versions.values()))

    def check_version(self, version: Version) -> bool:
        """Return whether the version ID is not already present."""
        return version.version_id not in self._versions


class VersionRepository(InMemoryVersionRepository):
    """Backward-compatible repository using in-memory storage.

    Keeps the original constructor and methods. Use
    SqliteVersionRepository explicitly when persistence is required.
    """

    def __init__(self, agent_id: str, storage=None):
        super().__init__(agent_id=agent_id, storage=storage)


def _to_row(agent_id: str, version: Version) -> VersionRow:
    """Convert a domain Version to its SQLAlchemy row."""
    return VersionRow(
        version_id=version.version_id,
        agent_id=agent_id,
        parent_version_id=version.parent_version_id,
        configuration=version.configuration.to_dict(),
        author=version.author,
        message=version.message,
        created_at=version.created_at,
        fingerprint=version.fingerprint,
    )


def _to_domain(row: VersionRow) -> Version:
    """Convert a SQLAlchemy row to a domain Version."""
    return Version(
        version_id=row.version_id,
        parent_version_id=row.parent_version_id,
        configuration=AgentConfiguration.from_dict(row.configuration),
        author=row.author,
        message=row.message,
        created_at=row.created_at,
        fingerprint=row.fingerprint,
    )


class SqliteVersionRepository:
    """Persist and retrieve agent configuration versions using SQLAlchemy."""

    def __init__(
        self,
        agent_id: str,
        session_factory: sessionmaker[Session],
    ):
        self.agent_id = agent_id
        self.session_factory = session_factory

    def _exists(self, version_id: str) -> bool:
        with session_scope(self.session_factory) as session:
            statement = select(VersionRow.version_id).where(
                VersionRow.agent_id == self.agent_id,
                VersionRow.version_id == version_id,
            )
            return session.execute(statement).first() is not None

    def save(self, version: Version) -> None:
        """Save a version, rejecting duplicates and unknown parents."""
        with session_scope(self.session_factory) as session:
            existing = session.execute(
                select(VersionRow.version_id).where(
                    VersionRow.agent_id == self.agent_id,
                    VersionRow.version_id == version.version_id,
                )
            ).first()

            if existing is not None:
                raise ValueError(
                    f"Version '{version.version_id}' already exists"
                )

            if version.parent_version_id is not None:
                parent = session.execute(
                    select(VersionRow.version_id).where(
                        VersionRow.agent_id == self.agent_id,
                        VersionRow.version_id == version.parent_version_id,
                    )
                ).first()

                if parent is None:
                    raise ValueError(
                        f"Parent version '{version.parent_version_id}' "
                        "does not exist"
                    )

            session.add(_to_row(self.agent_id, version))

    def get(self, version_id: str) -> Version:
        """Retrieve a version belonging to this agent."""
        with session_scope(self.session_factory) as session:
            row = session.execute(
                select(VersionRow).where(
                    VersionRow.agent_id == self.agent_id,
                    VersionRow.version_id == version_id,
                )
            ).scalar_one_or_none()

            if row is None:
                raise KeyError(version_id)

            return _to_domain(row)

    def list_versions(self) -> list[Version]:
        """Return this agent's versions in save order."""
        with session_scope(self.session_factory) as session:
            rows = session.execute(
                select(VersionRow)
                .where(VersionRow.agent_id == self.agent_id)
                .order_by(VersionRow.seq)
            ).scalars().all()

            return [_to_domain(row) for row in rows]

    def get_latest(self) -> Version | None:
        """Return this agent's latest saved version, or None."""
        with session_scope(self.session_factory) as session:
            row = session.execute(
                select(VersionRow)
                .where(VersionRow.agent_id == self.agent_id)
                .order_by(VersionRow.seq.desc())
                .limit(1)
            ).scalar_one_or_none()

            return _to_domain(row) if row is not None else None

    def check_version(self, version: Version) -> bool:
        """Return whether this version ID is available for this agent."""
        return not self._exists(version.version_id)
