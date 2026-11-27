from .artifact import PromptArtifact


class AgentConfigurationExtractor:
    """Extract versionable configuration from an agent."""

    @staticmethod
    def extract(agent: object) -> dict:
        """Extract configuration fields from an agent."""
        return {
            "system_prompt": getattr(agent, "system_prompt", "you are an agent"),
            "examples": getattr(agent, "examples", []),
            "tools": [tool.to_dict() for tool in getattr(agent, "tools", [])],
            "model": getattr(agent, "model", "GPT-5"),
            "model_parameters": {
                "temperature": getattr(agent, "temperature", 0.2),
                "top_p": getattr(agent, "top_p", 1.0),
                "max_tokens": getattr(agent, "max_tokens", 500),
            },
            "response_schema": getattr(agent, "response_schema", None),
            "safety_instructions": getattr(agent, "safety_instructions", None),
            "memory_template": getattr(agent, "memory_template", None),
            "retrieval_template": getattr(agent, "retrieval_template", None),
        }

    @staticmethod
    def from_agent(agent: object) -> PromptArtifact:
        """Create a PromptArtifact from an agent configuration."""
        config = AgentConfigurationExtractor.extract(agent)
        return PromptArtifact(**config)
