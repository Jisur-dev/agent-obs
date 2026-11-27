from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from agent_obs.versioning.integrations.langchain import (
    LangChainConfigurationAdapter,
    create_agent_obs,
)

# ---------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------


class FakeChatModel(BaseChatModel):
    """A lightweight LangChain chat model for integration testing."""

    model_name: str = "jisur-test-model"
    temperature: float = 0.2

    @property
    def _llm_type(self) -> str:
        return "jisur-test-model"

    def bind_tools(self, tools, **kwargs):
        """Allow LangChain create_agent to bind tools."""
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


@tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is sunny."


@pytest.fixture
def configuration():
    return {
        "system_prompt": "You are a helpful assistant.",
        "examples": [],
        "tools": [
            {"name": "search", "description": "Search the web"},
        ],
        "model": "gpt-4.1",
        "model_parameters": {"temperature": 0.2},
    }


@pytest.fixture
def mocked_integration(configuration):
    """Mock LangChain and versioning dependencies."""

    with (
        patch(
            "agent_obs.versioning.integrations.langchain.create_agent"
        ) as mock_create_agent,
        patch(
            "agent_obs.versioning.integrations.langchain."
            "LangChainConfigurationAdapter.extract"
        ) as mock_extract,
        patch(
            "agent_obs.versioning.integrations.langchain.make_engine"
        ) as mock_make_engine,
        patch("agent_obs.versioning.integrations.langchain.init_db"),
        patch(
            "agent_obs.versioning.integrations.langchain.make_session_factory"
        ) as mock_make_session_factory,
        patch(
            "agent_obs.versioning.integrations.langchain.SqliteVersionRepository"
        ) as mock_repository_class,
        patch(
            "agent_obs.versioning.integrations.langchain.VersioningService"
        ) as mock_service_class,
    ):
        expected_agent = MagicMock(name="langchain_agent")
        mock_repository = mock_repository_class.return_value

        mock_create_agent.return_value = expected_agent
        mock_extract.return_value = configuration
        mock_make_engine.return_value = MagicMock()
        mock_make_session_factory.return_value = MagicMock()
        mock_service_class.return_value = MagicMock()

        yield {
            "agent": expected_agent,
            "create_agent": mock_create_agent,
            "extract": mock_extract,
            "repository_class": mock_repository_class,
            "repository": mock_repository,
            "service_class": mock_service_class,
            "service": mock_service_class.return_value,
            "configuration": configuration,
        }


# ---------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------


def test_create_agent_obs_returns_langchain_agent(mocked_integration):
    """The wrapper returns the agent created by LangChain."""

    mocks = mocked_integration

    result = create_agent_obs(
        model="gpt-4.1",
        tools=["search_tool"],
        system_prompt="You are a helpful assistant.",
    )

    assert result is mocks["agent"]


def test_create_agent_obs_forwards_arguments(mocked_integration):
    """The wrapper forwards factory arguments to LangChain."""

    mocks = mocked_integration
    checkpointer = MagicMock()

    create_agent_obs(
        model="gpt-4.1",
        tools=["search_tool"],
        system_prompt="Test prompt",
        checkpointer=checkpointer,
        debug=True,
    )

    mocks["create_agent"].assert_called_once_with(
        model="gpt-4.1",
        tools=["search_tool"],
        system_prompt="Test prompt",
        checkpointer=checkpointer,
        debug=True,
    )


@pytest.mark.parametrize(
    "tools",
    [
        pytest.param(None, id="none"),
        pytest.param([], id="empty-list"),
    ],
)
def test_create_agent_obs_without_tools(mocked_integration, tools):
    """None and empty tools are normalized to an empty list."""

    mocks = mocked_integration

    create_agent_obs(
        model="gpt-4.1",
        tools=tools,
        system_prompt="Test prompt",
    )

    assert mocks["create_agent"].call_args.kwargs["tools"] == []


def test_create_agent_obs_with_generator_tools(mocked_integration):
    """Generator tools are consumed and forwarded as a list."""

    mocks = mocked_integration
    tools = (tool for tool in ["tool_a", "tool_b"])

    create_agent_obs(
        model="gpt-4.1",
        tools=tools,
        system_prompt="Test prompt",
    )

    assert mocks["create_agent"].call_args.kwargs["tools"] == [
        "tool_a",
        "tool_b",
    ]


def test_create_agent_obs_with_none_system_prompt(mocked_integration):
    """A None system prompt is normalized when passed to the adapter."""

    mocks = mocked_integration

    create_agent_obs(
        model="gpt-4.1",
        tools=[],
        system_prompt=None,
    )

    mocks["extract"].assert_called_once()
    assert mocks["extract"].call_args.kwargs["system_prompt"] is None


# ---------------------------------------------------------------------
# Version creation and persistence
# ---------------------------------------------------------------------
def test_create_agent_obs_saves_expected_version(mocked_integration):
    """The wrapper commits a version with the expected configuration."""

    mocks = mocked_integration

    create_agent_obs(
        model="gpt-4.1",
        tools=["search_tool"],
        system_prompt="Test prompt",
    )

    mocks["repository_class"].assert_called_once()

    generated_agent_id = mocks["repository_class"].call_args.args[0]
    UUID(generated_agent_id)

    mocks["service"].commit.assert_called_once()

    kwargs = mocks["service"].commit.call_args.kwargs
    assert kwargs["author"] == "unknown"
    assert kwargs["message"] == "Automatic configuration snapshot"

    agent_configuration = kwargs["configuration"]

    assert (
        agent_configuration.prompt.system_prompt
        == (mocks["configuration"]["system_prompt"])
    )
    assert agent_configuration.prompt.model == "gpt-4.1"


# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------


def test_does_not_save_if_agent_creation_fails(mocked_integration):
    """No version is saved if LangChain agent creation fails."""

    mocks = mocked_integration
    mocks["create_agent"].side_effect = RuntimeError("Agent creation failed")

    with pytest.raises(RuntimeError, match="Agent creation failed"):
        create_agent_obs(
            model="gpt-4.1",
            tools=[],
            system_prompt="Test prompt",
        )

    mocks["extract"].assert_not_called()
    mocks["service"].commit.assert_not_called()
    mocks["repository"].save.assert_not_called()


def test_does_not_save_if_extraction_fails(mocked_integration):
    """No version is saved if configuration extraction fails."""

    mocks = mocked_integration
    mocks["extract"].side_effect = ValueError("Extraction failed")

    with pytest.raises(ValueError, match="Extraction failed"):
        create_agent_obs(
            model="gpt-4.1",
            tools=[],
            system_prompt="Test prompt",
        )

    mocks["service"].commit.assert_not_called()
    mocks["repository"].save.assert_not_called()


# ---------------------------------------------------------------------
# LangChain configuration extraction
# ---------------------------------------------------------------------


def test_adapter_extracts_real_langchain_tool():
    """Adapter extracts metadata from a real LangChain tool."""

    result = LangChainConfigurationAdapter.extract(
        model="gpt-4.1",
        tools=[get_weather],
        system_prompt="You are a weather assistant.",
    )

    assert result["system_prompt"] == "You are a weather assistant."
    assert result["model"] == "gpt-4.1"
    assert len(result["tools"]) == 1
    assert result["tools"][0]["name"] == "get_weather"
    assert result["tools"][0]["description"] == ("Get the weather for a given city.")
    assert result["tools"][0]["args"]["city"]["type"] == "string"


def test_adapter_extracts_real_langchain_model():
    """Adapter extracts model name and parameters from a chat model."""

    model = FakeChatModel(
        model_name="jisur-test-model",
        temperature=0.2,
    )

    result = LangChainConfigurationAdapter.extract(
        model=model,
        tools=[],
        system_prompt="You are a helpful assistant.",
    )

    assert "jisur-test-model" in str(result["model"])
    assert result["model_parameters"]["temperature"] == 0.2
