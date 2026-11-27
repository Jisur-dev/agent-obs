import pytest

from agent_obs.versioning.artifact import PromptArtifact
from agent_obs.versioning.extractor import AgentConfigurationExtractor


class FakeTool:
    def __init__(self, name="sql_tool", version="1.0"):
        self.name = name
        self.version = version

    def to_dict(self):
        return {
            "name": self.name,
            "version": self.version,
        }


def test_from_agent_creates_equivalent_prompt_artifact():
    class FakeAgent:
        system_prompt = "You are a SQL expert."
        examples = [{"input": "hello", "output": "Hi"}]
        tools = [FakeTool()]
        model = "GPT-5"
        temperature = 0.2
        top_p = 1.0
        max_tokens = 500
        response_schema = {"type": "object"}
        safety_instructions = "Be safe."
        memory_template = "Remember context."
        retrieval_template = "Use retrieval."

    artifact = AgentConfigurationExtractor.from_agent(FakeAgent())

    assert isinstance(artifact, PromptArtifact)
    assert artifact.system_prompt == FakeAgent.system_prompt
    assert artifact.examples == FakeAgent.examples
    assert artifact.tools == [FakeAgent.tools[0].to_dict()]
    assert artifact.model == FakeAgent.model
    assert artifact.model_parameters == {
        "temperature": 0.2,
        "top_p": 1.0,
        "max_tokens": 500,
    }
    assert artifact.response_schema == FakeAgent.response_schema
    assert artifact.safety_instructions == FakeAgent.safety_instructions
    assert artifact.memory_template == FakeAgent.memory_template
    assert artifact.retrieval_template == FakeAgent.retrieval_template


def test_from_agent_uses_documented_defaults():
    class FakeAgent:
        pass

    artifact = AgentConfigurationExtractor.from_agent(FakeAgent())

    assert artifact.system_prompt == "you are an agent"
    assert artifact.examples == []
    assert artifact.tools == []
    assert artifact.model == "GPT-5"
    assert artifact.model_parameters == {
        "temperature": 0.2,
        "top_p": 1.0,
        "max_tokens": 500,
    }
    assert artifact.response_schema is None
    assert artifact.safety_instructions is None
    assert artifact.memory_template is None
    assert artifact.retrieval_template is None


def test_from_agent_handles_partially_defined_agent():
    class FakeAgent:
        system_prompt = "You are a SQL expert."
        model = "GPT-5"

    artifact = AgentConfigurationExtractor.from_agent(FakeAgent())

    assert artifact.system_prompt == "You are a SQL expert."
    assert artifact.model == "GPT-5"
    assert artifact.examples == []
    assert artifact.tools == []
    assert artifact.model_parameters == {
        "temperature": 0.2,
        "top_p": 1.0,
        "max_tokens": 500,
    }


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("system_prompt", None),
        ("model", None),
        ("response_schema", None),
        ("safety_instructions", None),
        ("memory_template", None),
        ("retrieval_template", None),
    ],
)
def test_explicit_none_values_are_preserved(attribute, value):
    class FakeAgent:
        pass

    setattr(FakeAgent, attribute, value)

    artifact = AgentConfigurationExtractor.from_agent(FakeAgent())

    assert getattr(artifact, attribute) is None


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("system_prompt", ""),
        ("examples", []),
        ("tools", []),
        ("model", ""),
        ("response_schema", {}),
        ("safety_instructions", ""),
        ("memory_template", ""),
        ("retrieval_template", ""),
    ],
)
def test_explicit_falsy_values_override_defaults(attribute, value):
    class FakeAgent:
        pass

    setattr(FakeAgent, attribute, value)

    artifact = AgentConfigurationExtractor.from_agent(FakeAgent())

    assert getattr(artifact, attribute) == value


def test_tools_are_converted_using_to_dict():
    class FakeAgent:
        tools = [FakeTool("search", "2.0")]

    artifact = AgentConfigurationExtractor.from_agent(FakeAgent())

    assert artifact.tools == [
        {
            "name": "search",
            "version": "2.0",
        }
    ]


@pytest.mark.parametrize(
    "invalid_tool",
    [
        object(),
        "not-a-tool",
        123,
        None,
    ],
)
def test_tool_without_to_dict_raises_attribute_error(invalid_tool):
    class FakeAgent:
        tools = [invalid_tool]

    with pytest.raises(AttributeError):
        AgentConfigurationExtractor.from_agent(FakeAgent())


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("system_prompt", "Changed"),
        ("model", "GPT-4"),
        ("response_schema", {"type": "string"}),
        ("safety_instructions", "Changed"),
        ("memory_template", "Changed"),
        ("retrieval_template", "Changed"),
    ],
)
def test_prompt_artifact_is_immutable(attribute, value):
    artifact = PromptArtifact(
        system_prompt="You are helpful.",
        model="GPT-5",
    )

    with pytest.raises(AttributeError):
        setattr(artifact, attribute, value)
