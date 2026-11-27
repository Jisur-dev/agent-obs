"""LangChain integration for Jisur agent versioning."""

from uuid import uuid4

from langchain.agents import create_agent

from agent_obs.versioning.artifact import (
    AgentConfiguration,
    PromptArtifact,
)
from agent_obs.versioning.db import (
    init_db,
    make_engine,
    make_session_factory,
)
from agent_obs.versioning.repository import SqliteVersionRepository
from agent_obs.versioning.service import VersioningService


class LangChainConfigurationAdapter:
    """Adapt LangChain objects to Jisur's configuration schema."""

    @staticmethod
    def _extract_model(model: object) -> tuple[str, dict]:
        """Extract serializable model information."""

        if model is None:
            return "", {}

        if isinstance(model, str):
            return model, {}

        model_name = (
            getattr(model, "model_name", None)
            or getattr(model, "model", None)
            or getattr(model, "model_id", None)
            or type(model).__name__
        )

        parameter_names = (
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "seed",
        )

        model_parameters = {}

        for name in parameter_names:
            value = getattr(model, name, None)

            if isinstance(value, (str, int, float, bool, list, dict)):
                if value is not None:
                    model_parameters[name] = value

        return str(model_name), model_parameters

    @staticmethod
    def _extract_tool(tool: object) -> dict:
        """Convert a LangChain tool into serializable metadata."""

        if isinstance(tool, dict):
            return tool

        to_dict = getattr(tool, "to_dict", None)
        if callable(to_dict):
            return to_dict()

        return {
            "name": getattr(tool, "name", type(tool).__name__),
            "description": getattr(tool, "description", ""),
            "args": getattr(tool, "args", {}),
        }

    @staticmethod
    def extract(model, tools=None, system_prompt=None) -> dict:
        """Extract LangChain configuration into Jisur's schema."""

        model_name, model_parameters = LangChainConfigurationAdapter._extract_model(
            model
        )

        return {
            "system_prompt": system_prompt or "",
            "examples": [],
            "tools": [
                LangChainConfigurationAdapter._extract_tool(tool)
                for tool in (tools or [])
            ],
            "model": model_name,
            "model_parameters": model_parameters,
        }


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
        examples=configuration["examples"],
        tools=configuration["tools"],
        model=configuration["model"],
        model_parameters=configuration["model_parameters"],
    )

    agent_configuration = AgentConfiguration(prompt=prompt)

    engine = make_engine()
    init_db(engine)
    session_factory = make_session_factory(engine)

    # Internal identifier: it is not part of the public API.
    agent_id = str(uuid4())

    repository = SqliteVersionRepository(agent_id, session_factory)
    service = VersioningService(repository)

    service.commit(
        configuration=agent_configuration,
        author="unknown",
        message="Automatic configuration snapshot",
    )

    return agent
