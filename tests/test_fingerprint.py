from agent_obs.versioning.artifact import (
    AgentConfiguration,
    PromptArtifact,
)
from agent_obs.versioning.fingerprint import FingerprintComputer


def make_configuration(prompt_overrides=None, **configuration_overrides):
    prompt_values = {
        "system_prompt": "You are a helpful SQL assistant.",
        "model": "GPT-5",
        "model_parameters": {
            "temperature": 0.2,
            "top_p": 1.0,
            "max_tokens": 500,
        },
    }

    if prompt_overrides:
        prompt_values.update(prompt_overrides)

    return AgentConfiguration(
        prompt=PromptArtifact(**prompt_values),
        **configuration_overrides,
    )


def test_canonicalization_is_deterministic():
    value = {
        "b": 2,
        "a": 1,
        "nested": {
            "z": 3,
            "y": 4,
        },
    }

    assert FingerprintComputer.canonicalize(value) == FingerprintComputer.canonicalize(
        value
    )


def test_nested_configuration_is_canonicalized_deterministically():
    first = {
        "outer": {
            "b": 2,
            "a": 1,
        }
    }

    second = {
        "outer": {
            "a": 1,
            "b": 2,
        }
    }

    assert FingerprintComputer.canonicalize(first) == FingerprintComputer.canonicalize(
        second
    )


def test_canonicalization_preserves_list_order():
    first = FingerprintComputer.canonicalize({"items": ["a", "b", "c"]})
    second = FingerprintComputer.canonicalize({"items": ["c", "b", "a"]})

    assert first != second


def test_fingerprint_is_deterministic():
    configuration = make_configuration()

    assert FingerprintComputer.compute(configuration) == FingerprintComputer.compute(
        configuration
    )


def test_from_agent_fingerprint_is_reproducible():
    from agent_obs.versioning.extractor import AgentConfigurationExtractor

    class FakeAgent:
        system_prompt = "You are a SQL expert."
        model = "GPT-5"
        temperature = 0.2
        top_p = 1.0
        max_tokens = 500

    first = AgentConfiguration(
        prompt=AgentConfigurationExtractor.from_agent(FakeAgent())
    )
    second = AgentConfiguration(
        prompt=AgentConfigurationExtractor.from_agent(FakeAgent())
    )

    assert FingerprintComputer.compute(first) == FingerprintComputer.compute(second)


def test_fingerprint_changes_when_configuration_changes():
    first = make_configuration(prompt_overrides={"system_prompt": "Prompt A"})
    second = make_configuration(prompt_overrides={"system_prompt": "Prompt B"})

    assert FingerprintComputer.compute(first) != FingerprintComputer.compute(second)


def test_fingerprint_changes_when_nested_configuration_changes():
    first = make_configuration(
        prompt_overrides={
            "model_parameters": {
                "temperature": 0.2,
                "top_p": 1.0,
                "max_tokens": 500,
            }
        }
    )
    second = make_configuration(
        prompt_overrides={
            "model_parameters": {
                "temperature": 0.9,
                "top_p": 1.0,
                "max_tokens": 500,
            }
        }
    )

    assert FingerprintComputer.compute(first) != FingerprintComputer.compute(second)


def test_agent_configuration_fingerprint_changes_when_tool_version_changes():
    first = make_configuration(tool_versions={"search": "1.0"})

    second = make_configuration(tool_versions={"search": "2.0"})

    assert FingerprintComputer.compute(first) != FingerprintComputer.compute(second)


def test_agent_configuration_dictionary_key_order_is_ignored():
    first = AgentConfiguration(
        prompt=PromptArtifact(
            model_parameters={
                "temperature": 0.2,
                "top_p": 1.0,
            }
        ),
        tool_versions={
            "search": "1.0",
            "calculator": "1.0",
        },
    )

    second = AgentConfiguration(
        prompt=PromptArtifact(
            model_parameters={
                "top_p": 1.0,
                "temperature": 0.2,
            }
        ),
        tool_versions={
            "calculator": "1.0",
            "search": "1.0",
        },
    )

    assert FingerprintComputer.compute(first) == FingerprintComputer.compute(second)


def test_agent_configuration_guardrail_order_is_preserved():
    first = AgentConfiguration(
        prompt=PromptArtifact(),
        guardrails=["a", "b"],
    )

    second = AgentConfiguration(
        prompt=PromptArtifact(),
        guardrails=["b", "a"],
    )

    assert FingerprintComputer.compute(first) != FingerprintComputer.compute(second)


def test_every_configuration_field_changes_fingerprint():
    base = make_configuration()

    variations = [
        AgentConfiguration(
            prompt=base.prompt,
            tool_versions={"search": "1.0"},
        ),
        AgentConfiguration(
            prompt=base.prompt,
            retrieval_strategy={"strategy": "semantic"},
        ),
        AgentConfiguration(
            prompt=base.prompt,
            memory_configuration={"enabled": True},
        ),
        AgentConfiguration(
            prompt=base.prompt,
            guardrails=["strict"],
        ),
        AgentConfiguration(
            prompt=base.prompt,
            retry_policy={"max_retries": 3},
        ),
        AgentConfiguration(
            prompt=base.prompt,
            planner_configuration={"enabled": True},
        ),
    ]

    base_fingerprint = FingerprintComputer.compute(base)

    for variation in variations:
        assert base_fingerprint != FingerprintComputer.compute(variation)
