"""Tests for real LangChain agent construction and execution."""

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from agent_obs.versioning.db import make_engine
from agent_obs.versioning.integrations import langchain as langchain_integration
from agent_obs.versioning.integrations.langchain import create_agent_obs
from agent_obs.versioning.repository import SqliteVersionRepository

SYSTEM_PROMPT = "You are a helpful weather assistant."


# ---------------------------------------------------------------------
# Real LangChain test objects
# ---------------------------------------------------------------------


class FakeChatModel(BaseChatModel):
    """Fake chat model used with a real LangChain agent."""

    model_name: str = "jisur-test-model"
    temperature: float = 0.2

    @property
    def _llm_type(self) -> str:
        return "jisur-test-model"

    def bind_tools(self, tools, **kwargs):
        """Allow LangChain to bind tools to the model."""
        return self

    def _generate(
        self,
        messages,
        stop=None,
        run_manager=None,
        **kwargs,
    ):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="Test response"))]
        )


class ToolCallingFakeChatModel(FakeChatModel):
    """Fake model that requests a tool call before responding."""

    def _generate(
        self,
        messages,
        stop=None,
        run_manager=None,
        **kwargs,
    ):
        tool_message = next(
            (message for message in messages if isinstance(message, ToolMessage)),
            None,
        )

        if tool_message is None:
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                        "id": "weather_call_1",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            response = AIMessage(content="The weather in Paris is sunny.")

        return ChatResult(generations=[ChatGeneration(message=response)])


class MultiToolCallingFakeChatModel(FakeChatModel):
    """Fake model that requests multiple tools in one turn."""

    def _generate(
        self,
        messages,
        stop=None,
        run_manager=None,
        **kwargs,
    ):
        tool_messages = [
            message for message in messages if isinstance(message, ToolMessage)
        ]

        if not tool_messages:
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                        "id": "weather_call_1",
                        "type": "tool_call",
                    },
                    {
                        "name": "get_time",
                        "args": {"city": "Paris"},
                        "id": "time_call_1",
                        "type": "tool_call",
                    },
                ],
            )
        else:
            response = AIMessage(content="Paris is sunny and the local time is 14:30.")

        return ChatResult(generations=[ChatGeneration(message=response)])


@tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is sunny."


@tool
def get_time(city: str) -> str:
    """Get the local time for a city."""
    return f"The local time in {city} is 14:30."


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def saved_versions(monkeypatch):
    """Capture saved versions using an in-memory database."""

    versions = []
    engine = make_engine("sqlite://")

    monkeypatch.setattr(
        langchain_integration,
        "make_engine",
        lambda: engine,
    )

    def capture_save(self, version):
        versions.append(version)

    monkeypatch.setattr(
        SqliteVersionRepository,
        "save",
        capture_save,
    )

    return versions


# ---------------------------------------------------------------------
# Real agent construction and execution
# ---------------------------------------------------------------------


def test_create_agent_obs_returns_real_langchain_agent(saved_versions):
    """The wrapper creates and returns an actual LangChain agent."""

    agent = create_agent_obs(
        model=FakeChatModel(),
        tools=[get_weather],
        system_prompt=SYSTEM_PROMPT,
    )

    assert agent is not None
    assert callable(agent.invoke)
    assert len(saved_versions) == 1


def test_real_langchain_agent_can_invoke_model(saved_versions):
    """A real LangChain agent can execute a model invocation."""

    agent = create_agent_obs(
        model=FakeChatModel(),
        tools=[],
        system_prompt=SYSTEM_PROMPT,
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello!",
                }
            ]
        }
    )

    assert "messages" in result
    assert result["messages"][-1].content == "Test response"


def test_real_langchain_agent_executes_tool(saved_versions):
    """A real LangChain agent can execute a tool requested by its model."""

    agent = create_agent_obs(
        model=ToolCallingFakeChatModel(),
        tools=[get_weather],
        system_prompt=SYSTEM_PROMPT,
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather in Paris?",
                }
            ]
        }
    )

    messages = result["messages"]

    tool_messages = [
        message for message in messages if isinstance(message, ToolMessage)
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0].name == "get_weather"
    assert tool_messages[0].content == ("The weather in Paris is sunny.")

    assert messages[-1].content == ("The weather in Paris is sunny.")


def test_real_agent_configuration_is_saved(saved_versions):
    """The real LangChain agent configuration is captured in a version."""

    agent = create_agent_obs(
        model=FakeChatModel(),
        tools=[get_weather],
        system_prompt=SYSTEM_PROMPT,
    )

    assert agent is not None
    assert len(saved_versions) == 1

    version = saved_versions[0]
    prompt = version.configuration.prompt

    assert prompt.system_prompt == SYSTEM_PROMPT

    assert len(prompt.tools) == 1
    assert prompt.tools[0]["name"] == "get_weather"
    assert prompt.tools[0]["description"] == ("Get the weather for a given city.")

    assert prompt.model == "jisur-test-model"
    assert prompt.model_parameters["temperature"] == 0.2

    assert isinstance(version.fingerprint, str)
    assert version.fingerprint

    assert version.author == "unknown"
    assert version.message == "Automatic configuration snapshot"
    assert version.parent_version_id is None


# ---------------------------------------------------------------------
# Configuration edge cases
# ---------------------------------------------------------------------


def test_agent_with_no_tools(saved_versions):
    """An agent can be created and invoked without tools."""

    agent = create_agent_obs(
        model=FakeChatModel(),
        tools=[],
        system_prompt="You are a simple assistant.",
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})

    assert result["messages"][-1].content == "Test response"
    assert len(saved_versions) == 1

    prompt = saved_versions[0].configuration.prompt

    assert prompt.tools == []
    assert prompt.system_prompt == "You are a simple assistant."


def test_agent_with_multiple_tools(saved_versions):
    """An agent can be configured with multiple tools."""

    agent = create_agent_obs(
        model=FakeChatModel(),
        tools=[get_weather, get_time],
        system_prompt=SYSTEM_PROMPT,
    )

    assert agent is not None
    assert callable(agent.invoke)
    assert len(saved_versions) == 1

    prompt = saved_versions[0].configuration.prompt

    tool_names = [tool["name"] for tool in prompt.tools]

    assert tool_names == ["get_weather", "get_time"]


def test_agent_with_custom_model_configuration(saved_versions):
    """Different model parameters are captured in the configuration."""

    model = FakeChatModel(
        model_name="custom-model",
        temperature=0.8,
    )

    create_agent_obs(
        model=model,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
    )

    prompt = saved_versions[0].configuration.prompt

    assert prompt.model == "custom-model"
    assert prompt.model_parameters["temperature"] == 0.8


def test_agent_with_different_system_prompt(saved_versions):
    """A custom system prompt is preserved in the version."""

    custom_prompt = (
        "You are an expert financial assistant. "
        "Always explain your calculations clearly."
    )

    create_agent_obs(
        model=FakeChatModel(),
        tools=[],
        system_prompt=custom_prompt,
    )

    prompt = saved_versions[0].configuration.prompt

    assert prompt.system_prompt == custom_prompt


def test_agent_with_empty_system_prompt(saved_versions):
    """An empty system prompt is accepted and preserved."""

    create_agent_obs(
        model=FakeChatModel(),
        tools=[],
        system_prompt="",
    )

    prompt = saved_versions[0].configuration.prompt

    assert prompt.system_prompt == ""


def test_agent_with_long_system_prompt(saved_versions):
    """A long system prompt is preserved without truncation."""

    long_prompt = (
        "You are a specialized assistant. Follow the instructions carefully. "
    ) * 200

    create_agent_obs(
        model=FakeChatModel(),
        tools=[],
        system_prompt=long_prompt,
    )

    prompt = saved_versions[0].configuration.prompt

    assert prompt.system_prompt == long_prompt
    assert len(prompt.system_prompt) == len(long_prompt)


# ---------------------------------------------------------------------
# Complex tool execution
# ---------------------------------------------------------------------


def test_agent_executes_multiple_tools(saved_versions):
    """An agent can execute multiple tools in one model turn."""

    agent = create_agent_obs(
        model=MultiToolCallingFakeChatModel(),
        tools=[get_weather, get_time],
        system_prompt=SYSTEM_PROMPT,
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Give me the weather and local time in Paris.",
                }
            ]
        }
    )

    messages = result["messages"]

    tool_messages = [
        message for message in messages if isinstance(message, ToolMessage)
    ]

    assert len(tool_messages) == 2

    tool_names = {message.name for message in tool_messages}

    assert tool_names == {"get_weather", "get_time"}

    tool_results = {message.name: message.content for message in tool_messages}

    assert tool_results["get_weather"] == ("The weather in Paris is sunny.")
    assert tool_results["get_time"] == ("The local time in Paris is 14:30.")

    assert messages[-1].content == ("Paris is sunny and the local time is 14:30.")


# ---------------------------------------------------------------------
# Fingerprint consistency and configuration changes
# ---------------------------------------------------------------------


def test_identical_configurations_produce_same_fingerprint(
    saved_versions,
):
    """Identical configurations produce identical fingerprints."""

    for _ in range(2):
        create_agent_obs(
            model=FakeChatModel(
                model_name="jisur-test-model",
                temperature=0.2,
            ),
            tools=[get_weather],
            system_prompt=SYSTEM_PROMPT,
        )

    assert len(saved_versions) == 2

    first_fingerprint = saved_versions[0].fingerprint
    second_fingerprint = saved_versions[1].fingerprint

    assert first_fingerprint == second_fingerprint


def test_different_system_prompts_produce_different_fingerprints(
    saved_versions,
):
    """Changing the system prompt changes the fingerprint."""

    create_agent_obs(
        model=FakeChatModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
    )

    create_agent_obs(
        model=FakeChatModel(),
        tools=[],
        system_prompt="You are a financial assistant.",
    )

    assert len(saved_versions) == 2

    assert saved_versions[0].fingerprint != saved_versions[1].fingerprint


def test_different_model_parameters_produce_different_fingerprints(
    saved_versions,
):
    """Changing model parameters changes the fingerprint."""

    create_agent_obs(
        model=FakeChatModel(temperature=0.2),
        tools=[],
        system_prompt=SYSTEM_PROMPT,
    )

    create_agent_obs(
        model=FakeChatModel(temperature=0.8),
        tools=[],
        system_prompt=SYSTEM_PROMPT,
    )

    assert len(saved_versions) == 2

    assert saved_versions[0].fingerprint != saved_versions[1].fingerprint


def test_different_tools_produce_different_fingerprints(
    saved_versions,
):
    """Changing the available tools changes the fingerprint."""

    create_agent_obs(
        model=FakeChatModel(),
        tools=[get_weather],
        system_prompt=SYSTEM_PROMPT,
    )

    create_agent_obs(
        model=FakeChatModel(),
        tools=[get_weather, get_time],
        system_prompt=SYSTEM_PROMPT,
    )

    assert len(saved_versions) == 2

    assert saved_versions[0].fingerprint != saved_versions[1].fingerprint
