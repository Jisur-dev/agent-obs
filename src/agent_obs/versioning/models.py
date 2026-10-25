from __future__ import annotations

from sqlalchemy import JSON, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class VersionRow(Base):
    __tablename__ = "versions"

    seq: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)  # save order
    version_id: Mapped[str] = mapped_column(String(64), unique=True)
    agent_id: Mapped[str] = mapped_column(String(200), index=True)
    parent_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("versions.version_id", ondelete="SET NULL"), nullable=True
    )
    configuration: Mapped[dict] = mapped_column(JSON)
    author: Mapped[str] = mapped_column(String(200))
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[float] = mapped_column(Float)   # same type as your dataclass
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)