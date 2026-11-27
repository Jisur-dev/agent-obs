"""Git-like versioning for prompts and agent configurations.

Git concept -> library concept: commit -> save version, branch -> experiment,
tag -> label, diff -> configuration changes, blame -> change history,
revert -> rollback, commit hash -> configuration fingerprint.
"""
from .temp_agent import TempAgent

__all__ = [
    "TempAgent",
]