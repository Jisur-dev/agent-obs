from __future__ import annotations

from typing import Any

from agent_obs.integrations.langchain.adapter import LangChainAdapter
from agent_obs.integrations.langchain.classic.react import ReactAgentAdapter
from agent_obs.integrations.langchain.shared.tools import extract_tools
from agent_obs.versioning.temp_agent import TempAgent


class AgentExecutorAdapter(LangChainAdapter):
    """Adapter for classic LangChain AgentExecutor."""

    def can_handle(self, agent: object) -> bool:
        module = type(agent).__module__
        class_name = type(agent).__name__

        if "langchain_classic.agents" not in module:
            return False

        if class_name == "AgentExecutor":
            return True

        # Support subclasses/wrappers with the same public structure.
        return (
            hasattr(agent, "agent")
            and hasattr(agent, "tools")
        )

    def extract(self, agent: object) -> TempAgent:
        underlying_agent = getattr(agent, "agent", None)

        if underlying_agent is None:
            return TempAgent(
                metadata={
                    "adapter": "langchain_classic.executor",
                    "type": type(agent).__name__,
                    "module": type(agent).__module__,
                }
            )

        react_adapter = ReactAgentAdapter()

        if react_adapter.can_handle(underlying_agent):
            result = react_adapter.extract(underlying_agent)

            executor_tools = getattr(agent, "tools", None)

            if executor_tools is not None:
                result.tools = extract_tools(executor_tools)

            result.metadata["adapter"] = "langchain_classic.executor"

            return result

        return TempAgent(
            name=getattr(underlying_agent, "name", None),
            tools=extract_tools(getattr(agent, "tools", None)),
            metadata={
                "adapter": "langchain_classic.executor",
                "underlying_agent": {
                    "type": type(underlying_agent).__name__,
                    "module": type(underlying_agent).__module__,
                },
            },
        )