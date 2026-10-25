"""The versioned units: PromptArtifact and the broader AgentConfiguration.

A "prompt" is rarely just a system string -- it's the system prompt, examples,
tool schemas, model config, and response schema compiled together. That whole
compiled artifact is what gets versioned, not just one field of it. The
end-user message is deliberately excluded: it belongs to the execution trace,
not the prompt definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from typing import Any


from agent_obs.shared.immutable_list import ImmutableList

 
@dataclass(frozen=True)
class PromptArtifact:
    """Everything that actually goes into an LLM call, versioned as one unit."""

    system_prompt: str = ""
    examples: list[dict] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)

    model: str = ""

    model_parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 500,
        }
    )

    response_schema: dict[str, Any] | None = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["answer"],
        }
    )

    safety_instructions: Any | None = None
    memory_template: Any | None = None
    retrieval_template: Any | None = None


    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "examples",
            ImmutableList(self.examples),
        )
        object.__setattr__(
            self,
            "tools",
            ImmutableList(self.tools),
        )

        temperature = self.model_parameters.get("temperature")
        top_p = self.model_parameters.get("top_p")

        if temperature is not None:
            if not isinstance(temperature, (int, float)):
                raise ValueError("temperature must be a number")

            if not 0 <= temperature <= 2:
                raise ValueError(
                    "temperature must be between 0 and 2"
                )

        if top_p is not None:
            if not isinstance(top_p, (int, float)):
                raise ValueError("top_p must be a number")

            if not 0 <= top_p <= 1:
                raise ValueError(
                    "top_p must be between 0 and 1"
                )
    def to_dict(self) -> dict:
        """JSON-friendly form: ImmutableLists become plain lists."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["examples"] = list(self.examples)
        data["tools"] = list(self.tools)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "PromptArtifact":
        # __post_init__ wraps examples/tools back into ImmutableList
        return cls(**data)
       

    
    
@dataclass
class AgentConfiguration:
    """The full reproducible definition of an agent, beyond just its prompt.

    Recreating this gives full reproducibility: "what exact configuration
    produced this execution three months ago?" becomes answerable.
    """

    prompt: PromptArtifact
    tool_versions: dict[str, str] = field(default_factory=dict)
    retrieval_strategy: dict | None = None
    memory_configuration: dict | None = None
    guardrails: list[str] = field(default_factory=list)
    retry_policy: dict | None = None
    planner_configuration: dict | None = None
    def to_dict(self) -> dict:
        data = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "prompt"}
        data["prompt"] = self.prompt.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "AgentConfiguration":
        data = dict(data)
        data["prompt"] = PromptArtifact.from_dict(data["prompt"])
        return cls(**data)

