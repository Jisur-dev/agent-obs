from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TempAgent:
    """
    Framework-independent representation of an agent.

    Integrations such as LangChain convert their native agent
    representation into this structure before versioning.
    """

    name: str | None = None

    system_prompt: str | None = None

    examples: list[dict[str, Any]] = field(default_factory=list)

    tools: list[dict[str, Any]] = field(default_factory=list)

    model: dict[str, Any] = field(default_factory=dict)

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None

    response_schema: dict[str, Any] | None = None

    safety_instructions: list[str] = field(default_factory=list)

    memory_template: str | None = None
    retrieval_template: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)