"""LangChain integration for Jisur agent versioning."""

from langchain.agents import create_agent
from agent_obs.integrations.langchain import (
    LangChainConfigurationAdapter,
)
from agent_obs.versioning.artifact import (
    AgentConfiguration,
    PromptArtifact,
)
from agent_obs.versioning.extractor import AgentConfigurationExtractor
from agent_obs.versioning.fingerprint import FingerprintComputer


class LangChainConfigurationAdapter:
    """Adapt LangChain factory arguments to Jisur's configuration schema."""

    @staticmethod
    def extract(
        model,
        tools=None,
        system_prompt=None,
        examples=None,
    ) -> dict:
        """Extract configuration fields for versioning."""

        class AgentView:
            pass

        agent = AgentView()
        agent.model = model
        agent.tools = list(tools) if tools is not None else []
        agent.system_prompt = (
            system_prompt if system_prompt is not None else ""
        )
        agent.examples = list(examples) if examples is not None else []

        return AgentConfigurationExtractor.extract(agent)


def create_obs_agent(
    model,
    tools=None,
    system_prompt=None,
    examples=None,
    *,
    return_version_info=False,
    **kwargs,
):
    """Create a LangChain agent and compute its configuration fingerprint."""

    tools = list(tools) if tools is not None else []
    examples = list(examples) if examples is not None else []

    configuration = LangChainConfigurationAdapter.extract(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        examples=examples,
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

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        **kwargs,
    )

    if return_version_info:
        return agent, agent_configuration, fingerprint

    return agent
