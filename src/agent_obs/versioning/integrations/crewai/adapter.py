"""Map a CrewAI Agent onto the shared AgentConfiguration schema."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...artifact import AgentConfiguration, PromptArtifact

if TYPE_CHECKING:
    from crewai import Agent
    from crewai.llms.base_llm import BaseLLM
    from crewai.tools import BaseTool


class CrewAIAdapter:
    """Map a CrewAI Agent onto the shared AgentConfiguration schema.

    This is the only class in Agent-Obs versioning that knows CrewAI field
    names. It turns live CrewAI objects (LLMs, tools, guardrails) into plain,
    JSON-safe data so the shared core can fingerprint and version them.
    All methods are static: the adapter keeps no state.
    """

    @staticmethod
    def _describe_llm(llm: BaseLLM | str | None) -> tuple[str, dict[str, float | int]]:
        """Turn a CrewAI LLM into plain data that can be fingerprinted.

        CrewAI replaces the ``llm=`` value with a provider object (for example
        ``OpenAICompletion``) when the Agent is built. This function reads only
        the model name and the generation parameters from that object, so the
        result is JSON-safe and never includes API keys or client objects.

        Args:
            llm (BaseLLM | str | None): The agent's LLM. Usually a ``BaseLLM``
                subclass. A raw model string or ``None`` is also accepted.

        Returns:
            tuple[str, dict[str, float | int]]: The model name (an empty string
                if unknown), and a dict holding only the parameters that are
                set, among ``temperature``, ``top_p`` and ``max_tokens``.
        """
        if llm is None:
            return "", {}

        if isinstance(llm, str):
            return llm, {}

        model = getattr(llm, "model", "") or ""
        params: dict[str, float | int] = {}

        for name in ("temperature", "top_p", "max_tokens"):
            value = getattr(llm, name, None)

            if value is not None:
                params[name] = value

        return model, params

    @staticmethod
    def _describe_tool(tool: BaseTool) -> dict[str, Any]:
        """Turn a CrewAI tool into plain data that can be fingerprinted.

        A tool is a live object with a function inside it. Its code can't be
        hashed reliably, so we keep what the LLM actually sees: the name, the
        description, and the JSON schema of its arguments.

        Args:
            tool (BaseTool): A tool attached to the agent.

        Returns:
            dict[str, Any]: A dict with ``name`` (str), ``description`` (str)
                and ``args_schema`` (dict, or None if the tool has no schema).
        """
        schema = None
        args_schema = getattr(tool, "args_schema", None)

        if args_schema is not None and hasattr(args_schema, "model_json_schema"):
            schema = args_schema.model_json_schema()

        return {
            "name": getattr(tool, "name", type(tool).__name__),
            "description": getattr(tool, "description", ""),
            "args_schema": schema,
        }

    @staticmethod
    def _describe_guardrail(guardrail: Any) -> str | None:
        """Turn a CrewAI guardrail into a stable string.

        CrewAI accepts a guardrail either as text (checked by an LLM) or as a
        Python function. A function object can't go into JSON, so we store its
        full import path instead.

        Args:
            guardrail (Any): A callable, a text guardrail, or None.

        Returns:
            str | None: The text itself, the function's "module.name", or None
                when the agent has no guardrail.
        """
        if guardrail is None:
            return None

        if isinstance(guardrail, str):
            return guardrail

        module = getattr(guardrail, "__module__", "")
        name = getattr(guardrail, "__qualname__", type(guardrail).__name__)

        return f"{module}.{name}"

    @staticmethod
    def _build_system_prompt(agent: Agent) -> str:
        """Combine the agent's role, goal and backstory into one system prompt.

        CrewAI has no single system prompt field. These three fields are what
        CrewAI puts into the prompt, so together they define the agent's
        instructions.

        Args:
            agent (Agent): The CrewAI agent.

        Returns:
            str: The three fields on separate lines, each with a label.
        """
        return (
            f"Role: {agent.role}\n"
            f"Goal: {agent.goal}\n"
            f"Backstory: {agent.backstory}"
        )

    @staticmethod
    def to_agent_configuration(agent: Agent) -> AgentConfiguration:
        """Map a CrewAI Agent onto the shared AgentConfiguration schema.

        Only fields that change the agent's behavior are kept. Runtime state
        (ids, callbacks, caches, executors) is left out so the fingerprint
        stays stable. CrewAI-only settings such as ``max_iter`` are not mapped
        yet: they wait for the ``framework_config`` decision.

        Args:
            agent (Agent): A fully built CrewAI agent.

        Returns:
            AgentConfiguration: The agent's versionable configuration.
        """
        model, model_parameters = CrewAIAdapter._describe_llm(agent.llm)

        prompt = PromptArtifact(
            system_prompt=CrewAIAdapter._build_system_prompt(agent),
            tools=[CrewAIAdapter._describe_tool(tool) for tool in agent.tools or []],
            model=model,
            model_parameters=model_parameters,
            response_schema=None,
        )

        guardrail = CrewAIAdapter._describe_guardrail(agent.guardrail)

        return AgentConfiguration(
            prompt=prompt,
            guardrails=[guardrail] if guardrail else [],
            retry_policy={
                "max_retry_limit": agent.max_retry_limit,
                "guardrail_max_retries": agent.guardrail_max_retries,
            },
            memory_configuration={"enabled": bool(agent.memory)},
            planner_configuration={
                "planning": agent.planning,
                "reasoning": agent.reasoning,
                "max_reasoning_attempts": agent.max_reasoning_attempts,
            },
        )
