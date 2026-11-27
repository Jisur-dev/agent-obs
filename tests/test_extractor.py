from agent_obs.versioning.extractor import AgentConfigurationExtractor


def test_extract_agent_configuration_captures_all_supported_fields():
    class FakeTool:
        def to_dict(self):
            return {
                "name": "sql_tool",
                "version": "1.0",
            }

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

    configuration = AgentConfigurationExtractor.extract(FakeAgent())

    assert configuration == {
        "system_prompt": "You are a SQL expert.",
        "examples": [{"input": "hello", "output": "Hi"}],
        "tools": [
            {
                "name": "sql_tool",
                "version": "1.0",
            }
        ],
        "model": "GPT-5",
        "model_parameters": {
            "temperature": 0.2,
            "top_p": 1.0,
            "max_tokens": 500,
        },
        "response_schema": {"type": "object"},
        "safety_instructions": "Be safe.",
        "memory_template": "Remember context.",
        "retrieval_template": "Use retrieval.",
    }


def test_from_agent_returns_prompt_artifact():
    class FakeAgent:
        system_prompt = "You are helpful."

    artifact = AgentConfigurationExtractor.from_agent(FakeAgent())

    assert artifact.system_prompt == "You are helpful."
