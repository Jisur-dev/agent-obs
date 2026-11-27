from __future__ import annotations

from agent_obs.integrations.langchain.adapter import LangChainAdapter
from agent_obs.integrations.langchain.classic.executor import (
    AgentExecutorAdapter,
)
from agent_obs.integrations.langchain.classic.react import (
    ReactAgentAdapter,
)
from agent_obs.integrations.langchain.modern.create_agent import (
    CreateAgentAdapter,
)
from agent_obs.versioning.temp_agent import TempAgent


_ADAPTERS: tuple[LangChainAdapter, ...] = (
    AgentExecutorAdapter(),
    ReactAgentAdapter(),
    CreateAgentAdapter(),
)


def get_adapter(agent: object) -> LangChainAdapter:
    """
    Find the first LangChain adapter capable of handling the object.
    """

    for adapter in _ADAPTERS:
        if adapter.can_handle(agent):
            return adapter

    raise TypeError(
        "Unsupported LangChain agent type: "
        f"{type(agent).__module__}.{type(agent).__name__}"
    )


def adapt(agent: object) -> TempAgent:
    """
    Convert a LangChain agent into a TempAgent.
    """

    adapter = get_adapter(agent)

    return adapter.extract(agent)