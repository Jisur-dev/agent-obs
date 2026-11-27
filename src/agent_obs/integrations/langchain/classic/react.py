from __future__ import annotations

from typing import Any

from agent_obs.integrations.langchain.adapter import LangChainAdapter
from agent_obs.integrations.langchain.shared.model import (
    extract_model,
    extract_model_parameters,
)
from agent_obs.integrations.langchain.shared.prompts import extract_prompt
from agent_obs.integrations.langchain.shared.tools import extract_tools
from agent_obs.versioning.temp_agent import TempAgent


class ReactAgentAdapter(LangChainAdapter):
    """Adapter for classic LangChain ReAct agents."""

    def can_handle(self, agent: object) -> bool:
        module = type(agent).__module__
        class_name = type(agent).__name__

        if "langchain_classic.agents" not in module:
            return False

        if class_name in {"Agent", "ReActChain"}:
            return True

        # Useful for tests and compatible subclasses/wrappers.
        return (
            hasattr(agent, "tools")
            and (
                hasattr(agent, "llm")
                or hasattr(agent, "llm_chain")
            )
        )
    def extract(self, agent: object) -> TempAgent:
        model = self._extract_model(agent)
        tools = self._extract_tools(agent)
        prompt = self._extract_prompt(agent)

        parameters = extract_model_parameters(model)
        prompt_data = extract_prompt(prompt)

        return TempAgent(
            name=getattr(agent, "name", None),
            system_prompt=self._extract_system_prompt(prompt),
            tools=tools,
            model=extract_model(model),
            temperature=parameters.get("temperature"),
            top_p=parameters.get("top_p"),
            max_tokens=parameters.get("max_tokens"),
            metadata={
                "adapter": "langchain_classic.react",
                "type": type(agent).__name__,
                "module": type(agent).__module__,
                "prompt": prompt_data,
            },
        )

    def _extract_model(self, agent: object) -> object | None:
        """Extract the LLM from a classic ReAct agent."""

        llm_chain = getattr(agent, "llm_chain", None)

        if llm_chain is not None:
            model = getattr(llm_chain, "llm", None)

            if model is not None:
                return model

        return getattr(agent, "llm", None)

    def _extract_tools(self, agent: object) -> list[dict[str, Any]]:
        """Extract tools from a classic ReAct agent."""

        tools = getattr(agent, "tools", None)

        return extract_tools(tools)

    def _extract_prompt(self, agent: object) -> object | None:
        """Extract the prompt used by the ReAct agent."""

        prompt = getattr(agent, "prompt", None)

        if prompt is not None:
            return prompt

        llm_chain = getattr(agent, "llm_chain", None)

        if llm_chain is not None:
            return getattr(llm_chain, "prompt", None)

        return None

    def _extract_system_prompt(
        self,
        prompt: object | None,
    ) -> str | None:
        if prompt is None:
            return None

        template = getattr(prompt, "template", None)

        if isinstance(template, str):
            return template

        return None