"""Content-addressing for prompt artifacts and agent configurations.

Two identical compiled artifacts always produce the same fingerprint, which
gives free deduplication and lets a run declare exactly which configuration
produced it -- SHA256(system_prompt + examples + tools + model_config + ...).
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_obs.versioning.artifact import AgentConfiguration


class FingerprintComputer:
    """Compute deterministic fingerprints for agent configurations."""

    @staticmethod
    def compute(configuration: AgentConfiguration) -> str:
        """Compute the deterministic fingerprint of the full agent configuration."""
        payload = {
            "prompt": {
                "system_prompt": configuration.prompt.system_prompt,
                "examples": configuration.prompt.examples,
                "tools": configuration.prompt.tools,
                "model": configuration.prompt.model,
                "model_parameters": configuration.prompt.model_parameters,
                "response_schema": configuration.prompt.response_schema,
                "safety_instructions": configuration.prompt.safety_instructions,
                "memory_template": configuration.prompt.memory_template,
                "retrieval_template": configuration.prompt.retrieval_template,
            },
            "tool_versions": configuration.tool_versions,
            "retrieval_strategy": configuration.retrieval_strategy,
            "memory_configuration": configuration.memory_configuration,
            "guardrails": configuration.guardrails,
            "retry_policy": configuration.retry_policy,
            "planner_configuration": configuration.planner_configuration,
        }

        canonical = FingerprintComputer.canonicalize(payload)

        return hashlib.sha256(canonical).hexdigest()

    @staticmethod
    def canonicalize(payload: dict) -> bytes:
        """Deterministically serialize a payload (stable key order, no whitespace)."""
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
