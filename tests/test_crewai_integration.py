import subprocess
import sys

import pytest

crewai = pytest.importorskip("crewai")

from crewai.tools import tool  # noqa: E402

from agent_obs.versioning.integrations.crewai import (  # noqa: E402
    CrewAIAdapter,
    create_agent,
)
from agent_obs.versioning.repository import VersionRepository  # noqa: E402
from agent_obs.versioning.service import VersioningService  # noqa: E402

AGENT_KWARGS = {
    "role": "Researcher",
    "goal": "Find facts",
    "backstory": "Careful analyst",
    "llm": "gpt-4o-mini",
}


@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch):
    """Give CrewAI a dummy key so building an LLM object needs no network."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture
def service() -> VersioningService:
    """Return a versioning service with an empty repository.

    If the repository class changes (PR #30), this is the only place to edit.
    """
    return VersioningService(VersionRepository(agent_id="researcher"))


def build_agent(**overrides) -> crewai.Agent:
    """Build a real CrewAI agent, overriding any default argument."""
    return crewai.Agent(**{**AGENT_KWARGS, **overrides})


# --- create_agent -----------------------------------------------------------


def test_create_agent_returns_real_agent(service):
    """Verify that create_agent returns a genuine CrewAI Agent, not a wrapper."""
    agent = create_agent(service, **AGENT_KWARGS)

    assert type(agent) is crewai.Agent


def test_create_agent_saves_first_version(service):
    """Verify that creating an agent saves exactly one version."""
    create_agent(service, **AGENT_KWARGS)

    assert len(service.repository.list_versions()) == 1


def test_same_configuration_saves_one_version(service):
    """Verify that creating the same agent twice does not add a duplicate."""
    create_agent(service, **AGENT_KWARGS)
    create_agent(service, **AGENT_KWARGS)

    assert len(service.repository.list_versions()) == 1


def test_changed_configuration_saves_new_version(service):
    """Verify that a changed goal creates a second version linked to the first."""
    create_agent(service, **AGENT_KWARGS)
    create_agent(service, **{**AGENT_KWARGS, "goal": "Find better facts"})

    versions = service.repository.list_versions()

    assert len(versions) == 2
    assert versions[1].parent_version_id == versions[0].version_id


def test_agent_works_in_crew(service):
    """Verify that the returned agent can be used in a Task and a Crew."""
    agent = create_agent(service, **AGENT_KWARGS)
    task = crewai.Task(description="d", expected_output="o", agent=agent)
    crew = crewai.Crew(agents=[agent], tasks=[task])

    assert crew.agents[0] is agent


# --- adapter ----------------------------------------------------------------


def test_adapter_reads_real_values():
    """Verify that the adapter captures the prompt text and the model name."""
    configuration = CrewAIAdapter.to_agent_configuration(build_agent())

    assert configuration.prompt.model == "gpt-4o-mini"
    assert "Find facts" in configuration.prompt.system_prompt


def test_adapter_does_not_leak_prompt_artifact_defaults():
    """Verify that the made-up default schema and parameters are not captured."""
    configuration = CrewAIAdapter.to_agent_configuration(build_agent())

    assert configuration.prompt.response_schema is None
    assert configuration.prompt.model_parameters == {}


def test_adapter_captures_llm_parameters():
    """Verify that temperature set on the LLM object reaches the configuration."""
    agent = build_agent(llm=crewai.LLM(model="gpt-4o-mini", temperature=0.3))

    configuration = CrewAIAdapter.to_agent_configuration(agent)

    assert configuration.prompt.model_parameters == {"temperature": 0.3}


def test_adapter_describes_tools_as_plain_data():
    """Verify that a tool becomes a dict with its name, description and schema."""

    @tool("Search")
    def search(query: str) -> str:
        """Search the web."""
        return query

    configuration = CrewAIAdapter.to_agent_configuration(build_agent(tools=[search]))

    (described,) = configuration.prompt.tools
    assert described["name"] == "Search"
    assert described["description"] == "Search the web."
    assert "query" in described["args_schema"]["properties"]


def test_adapter_stores_function_guardrail_by_name():
    """Verify that a guardrail function is stored as its import path."""

    def no_pii(output):
        return True, output

    configuration = CrewAIAdapter.to_agent_configuration(
        build_agent(guardrail=no_pii)
    )

    assert configuration.guardrails[0].endswith("no_pii")


def test_fingerprint_is_stable_and_sensitive():
    """Verify same agent gives the same fingerprint and a changed one differs."""
    from agent_obs.versioning.fingerprint import FingerprintComputer

    def fingerprint(**overrides):
        agent = build_agent(**overrides)
        return FingerprintComputer.compute(CrewAIAdapter.to_agent_configuration(agent))

    assert fingerprint() == fingerprint()
    assert fingerprint() != fingerprint(goal="Something else")


# --- optional dependency ----------------------------------------------------


def test_module_imports_without_crewai():
    """Verify that importing the integration does not require CrewAI."""
    code = (
        "import sys; sys.modules['crewai'] = None; "
        "import agent_obs.versioning.integrations.crewai"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True)

    assert result.returncode == 0, result.stderr.decode()
