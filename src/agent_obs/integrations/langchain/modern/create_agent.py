from __future__ import annotations

from typing import Any

from agent_obs.integrations.langchain.adapter import LangChainAdapter
from agent_obs.integrations.langchain.shared.model import (
    extract_model,
    extract_model_parameters,
)
from agent_obs.integrations.langchain.shared.tools import extract_tools
from agent_obs.versioning.temp_agent import TempAgent


class CreateAgentAdapter(LangChainAdapter):
    """Adapter for agents created with LangChain's create_agent()."""

    def can_handle(self, agent: object) -> bool:
        module = type(agent).__module__

        if not module.startswith("langgraph."):
            return False

        class_name = type(agent).__name__

        if class_name == "CompiledStateGraph":
            return True

        return hasattr(agent, "nodes") and isinstance(
            getattr(agent, "nodes", None),
            dict,
        )

    def extract(self, agent: object) -> TempAgent:
        model = self._extract_model(agent)
        tools = self._extract_tools(agent)

        parameters = extract_model_parameters(model)

        return TempAgent(
            name=getattr(agent, "name", None),
            system_prompt=self._extract_system_prompt(agent),
            tools=tools,
            model=extract_model(model),
            temperature=parameters.get("temperature"),
            top_p=parameters.get("top_p"),
            max_tokens=parameters.get("max_tokens"),
            response_schema=self._extract_response_schema(agent),
            metadata={
                "adapter": "langchain.create_agent",
                "type": type(agent).__name__,
                "module": type(agent).__module__,
            },
        )

    def _extract_model(self, agent: object) -> object | None:
        """Try to locate the model inside the compiled graph."""

        nodes = getattr(agent, "nodes", {})

        if not isinstance(nodes, dict):
            return None

        agent_node = nodes.get("agent")

        if agent_node is None:
            return None

        runnable = getattr(agent_node, "runnable", None)

        if runnable is None:
            return None

        bound = getattr(runnable, "bound", None)

        if bound is not None and self._looks_like_model(bound):
            return bound

        if self._looks_like_model(runnable):
            return runnable

        return None

    def _extract_tools(self, agent: object) -> list[dict[str, Any]]:
        """Try to locate tools inside the compiled graph."""

        nodes = getattr(agent, "nodes", {})

        if not isinstance(nodes, dict):
            return []

        tools_node = nodes.get("tools")

        if tools_node is None:
            return []

        runnable = getattr(tools_node, "runnable", None)

        if runnable is None:
            return []

        tools_by_name = getattr(runnable, "tools_by_name", None)

        if tools_by_name is not None:
            return extract_tools(tools_by_name)

        tools = getattr(runnable, "tools", None)

        if tools is not None:
            return extract_tools(tools)

        return []

    def _extract_system_prompt(
        self,
        agent: object,
    ) -> str | None:
        config = getattr(agent, "config", None)

        if isinstance(config, dict):
            system_prompt = config.get("system_prompt")

            if isinstance(system_prompt, str):
                return system_prompt

        return None

    def _extract_response_schema(
        self,
        agent: object,
    ) -> dict[str, Any] | None:
        config = getattr(agent, "config", None)

        if not isinstance(config, dict):
            return None

        response_format = config.get("response_format")

        if response_format is None:
            return None

        if isinstance(response_format, dict):
            return response_format

        return {
            "type": type(response_format).__name__,
            "module": type(response_format).__module__,
        }

    @staticmethod
    def _looks_like_model(value: object) -> bool:
        module = type(value).__module__

        if "langchain" not in module:
            return False

        return any(
            callable(getattr(value, attribute, None))
            for attribute in (
                "invoke",
                "generate",
                "bind_tools",
            )
        )