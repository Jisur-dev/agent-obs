"""Public entry point: build a CrewAI Agent and version its configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .adapter import CrewAIAdapter

if TYPE_CHECKING:
    from crewai import Agent

    from ...service import VersioningService


def create_agent(
    service: VersioningService,
    author: str = "crewai-integration",
    message: str = "Agent created",
    **agent_kwargs: Any,
) -> Agent:
    """Build a CrewAI Agent and save its configuration as a version.

    Args:
        service (VersioningService): Service bound to this agent's repository.
        author (str): Recorded as the version's author.
        message (str): Recorded as the version's message.
        **agent_kwargs (Any): Passed unchanged to crewai.Agent.

    Returns:
        Agent: The real CrewAI Agent, not a wrapper.
    """
    from crewai import Agent

    agent = Agent(**agent_kwargs)
    configuration = CrewAIAdapter.to_agent_configuration(agent)
    service.commit_if_changed(configuration, author=author, message=message)
    return agent
