from __future__ import annotations

import time
import uuid

from .artifact import AgentConfiguration
from .repository import Version, VersionRepository
from .fingerprint import FingerprintComputer



class VersioningService:
    """Apply versioning rules and persist configuration versions."""

    def __init__(self, repository: VersionRepository):
        self.repository = repository

    def commit(
        self,
        configuration: AgentConfiguration,
        author: str,
        message: str,
    ) -> Version:
        """Create and persist a new version."""

        latest = self.repository.get_latest()

        parent_version_id = (
            latest.version_id
            if latest is not None
            else None
        )

        version = Version(
            version_id=str(uuid.uuid4()),
            parent_version_id=parent_version_id,
            configuration=configuration,
            author=author,
            message=message,
            created_at=round(time.time(), 3),
            fingerprint=FingerprintComputer.compute(configuration),
        )

        self.repository.save(version)

        return version

    def commit_if_changed(
        self,
        configuration: AgentConfiguration,
        author: str,
        message: str,
    ) -> Version | None:
        """Save a new version only if the configuration changed.

        Args:
            configuration (AgentConfiguration): The configuration to save.
            author (str): Recorded as the version's author.
            message (str): Recorded as the version's message.

        Returns:
            Version | None: The new version, or None if the fingerprint
                matches the latest saved version.
        """
        fingerprint = FingerprintComputer.compute(configuration)
        latest = self.repository.get_latest()

        if latest is not None and latest.fingerprint == fingerprint:
            return None

        return self.commit(configuration, author=author, message=message)

   
    