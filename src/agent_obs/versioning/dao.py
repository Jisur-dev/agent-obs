
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload, sessionmaker

from .db import session_scope
from .models import ConfigurationRow, VersionRow


class ConfigurationDAO:
    """Data access operations for agent configurations."""

    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    def create(self, configuration: ConfigurationRow) -> int:
        with session_scope(self.session_factory) as session:
            session.add(configuration)
            session.flush()
            return configuration.seq

    def get(self, configuration_id: int) -> ConfigurationRow | None:
        with session_scope(self.session_factory) as session:
            return session.get(ConfigurationRow, configuration_id)

    def update(self, configuration: ConfigurationRow) -> None:
        with session_scope(self.session_factory) as session:
            session.merge(configuration)

    def delete(self, configuration_id: int) -> bool:
        with session_scope(self.session_factory) as session:
            configuration = session.get(ConfigurationRow, configuration_id)

            if configuration is None:
                return False

            session.delete(configuration)
            return True

    def exists(self, configuration_id: int) -> bool:
        with session_scope(self.session_factory) as session:
            return session.get(ConfigurationRow, configuration_id) is not None


class VersionDAO:
    """Data access operations for agent versions."""

    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    def create(self, version: VersionRow) -> int:
        with session_scope(self.session_factory) as session:
            session.add(version)
            session.flush()
            return version.seq

    def get(self, agent_id: str, version_id: str) -> VersionRow | None:
        with session_scope(self.session_factory) as session:
            statement = (
                select(VersionRow)
                .options(selectinload(VersionRow.configuration))
                .where(
                    VersionRow.agent_id == agent_id,
                    VersionRow.version_id == version_id,
                )
            )
            return session.execute(statement).scalar_one_or_none()

    def list_by_agent(self, agent_id: str) -> list[VersionRow]:
        with session_scope(self.session_factory) as session:
            statement = (
                select(VersionRow)
                .options(selectinload(VersionRow.configuration))
                .where(VersionRow.agent_id == agent_id)
                .order_by(VersionRow.seq)
            )
            return list(session.execute(statement).scalars().all())

    def get_latest(self, agent_id: str) -> VersionRow | None:
        with session_scope(self.session_factory) as session:
            statement = (
                select(VersionRow)
                .options(selectinload(VersionRow.configuration))
                .where(VersionRow.agent_id == agent_id)
                .order_by(VersionRow.seq.desc())
                .limit(1)
            )
            return session.execute(statement).scalar_one_or_none()

    def exists(self, agent_id: str, version_id: str) -> bool:
        with session_scope(self.session_factory) as session:
            statement = select(VersionRow.seq).where(
                VersionRow.agent_id == agent_id,
                VersionRow.version_id == version_id,
            )
            return session.execute(statement).first() is not None

    def update(self, version: VersionRow) -> None:
        with session_scope(self.session_factory) as session:
            session.merge(version)

    def delete(self, agent_id: str, version_id: str) -> bool:
        with session_scope(self.session_factory) as session:
            statement = select(VersionRow).where(
                VersionRow.agent_id == agent_id,
                VersionRow.version_id == version_id,
            )
            version = session.execute(statement).scalar_one_or_none()

            if version is None:
                return False

            session.delete(version)
            return True
