from __future__ import annotations

import time
import uuid

from .artifact import AgentConfiguration
from .repository import Version, VersionStore
from .fingerprint import FingerprintComputer



class VersioningService:
    """Apply versioning rules and persist configuration versions."""

    def __init__(self, repository: VersionStore):
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

   
    