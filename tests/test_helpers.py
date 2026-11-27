from agent_obs.versioning.artifact import AgentConfiguration, PromptArtifact


def make_configuration(
    prompt_overrides=None,
    **configuration_overrides,
):
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
