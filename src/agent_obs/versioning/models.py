from __future__ import annotations

from typing import Any

from sqlalchemy import JSON, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class ConfigurationRow(Base):
    __tablename__ = "configurations"

    seq: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # AgentConfiguration fields
    tool_versions: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False)
    retrieval_strategy: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    memory_configuration: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    guardrails: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    retry_policy: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    planner_configuration: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # PromptArtifact fields
    prompt_system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_examples: Mapped[list[dict]] = mapped_column(JSON, nullable=False)
    prompt_tools: Mapped[list[dict]] = mapped_column(JSON, nullable=False)
    prompt_model: Mapped[str] = mapped_column(String(200), nullable=False)
    prompt_model_parameters: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False
    )
    prompt_response_schema: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )
    prompt_safety_instructions: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    prompt_memory_template: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    prompt_retrieval_template: Mapped[Any | None] = mapped_column(JSON, nullable=True)

    versions: Mapped[list[VersionRow]] = relationship(back_populates="configuration")


class VersionRow(Base):
    __tablename__ = "versions"

    seq: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    version_id: Mapped[str] = mapped_column(String(64), unique=True)
    agent_id: Mapped[str] = mapped_column(String(200), index=True)

    parent_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("versions.version_id", ondelete="SET NULL"),
        nullable=True,
    )

    configuration_id: Mapped[int] = mapped_column(
        ForeignKey("configurations.seq"),
        nullable=False,
        index=True,
    )

    configuration: Mapped[ConfigurationRow] = relationship(back_populates="versions")

    author: Mapped[str] = mapped_column(String(200))
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[float] = mapped_column(Float)
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)
