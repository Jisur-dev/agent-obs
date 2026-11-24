"""LangChain integration for Jisur agent versioning."""

from time import time
from uuid import uuid4

from langchain.agents import create_agent

from agent_obs.versioning.artifact import (
    AgentConfiguration,
    PromptArtifact,
)
from agent_obs.versioning.extractor import AgentConfigurationExtractor
from agent_obs.versioning.fingerprint import FingerprintComputer
from agent_obs.versioning.repository import Version, VersionRepository


class LangChainConfigurationAdapter:
    """Adapt LangChain factory arguments to Jisur's configuration schema."""

    @staticmethod
    def extract(model, tools=None, system_prompt=None) -> dict:
        """Extract configuration fields for versioning."""

        class AgentView:
            pass

        agent = AgentView()
        agent.model = model
        agent.tools = list(tools) if tools is not None else []
        agent.system_prompt = system_prompt or ""
        agent.examples = []

        return AgentConfigurationExtractor.extract(agent)


def create_agent_obs(model, tools=None, system_prompt=None, **kwargs):
    """Create a LangChain agent and save its configuration version."""

    tools = list(tools) if tools is not None else []

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        **kwargs,
    )

    configuration = LangChainConfigurationAdapter.extract(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )

    prompt = PromptArtifact(
        system_prompt=configuration["system_prompt"],
        examples=configuration.get("examples", []),
        tools=configuration.get("tools", []),
        model=configuration.get("model", ""),
        model_parameters=configuration.get("model_parameters", {}),
    )

    agent_configuration = AgentConfiguration(prompt=prompt)

    fingerprint = FingerprintComputer.compute(agent_configuration)

    version = Version(
        version_id=str(uuid4()),
        parent_version_id=None,
        configuration=agent_configuration,
        author="unknown",
        message="Automatic configuration snapshot",
        created_at=time(),
        fingerprint=fingerprint,
    )

    repository = VersionRepository()
    repository.save(version)

    return agent