from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey,JSON,Enum
from typing import List ,Any
from ..trace_graph.enums import NodeType, EdgeType
from .base import Base
import uuid

class Graphs(Base):
    __tablename__ = "graphs"
    graph_id : Mapped[int] = mapped_column(primary_key=True)


    runs : Mapped[List["Run"]] = relationship(back_populates="graph")

class Run(Base):
    __tablename__="runs"
    run_id : Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    graph_id : Mapped[int] = mapped_column(ForeignKey("graphs.graph_id"))


    graph : Mapped["Graphs"] = relationship(back_populates="runs")
    nodes : Mapped[List["Nodes"]] = relationship(back_populates="run")
    edges : Mapped[List["Edges"]] = relationship(back_populates="run")
class Nodes(Base):
    __tablename__ = "nodes"
    node_id : Mapped[str] = mapped_column(primary_key=True , default=lambda: str(uuid.uuid4()))
    run_id : Mapped[str] = mapped_column(ForeignKey("runs.run_id"))
    node_type : Mapped[NodeType] = mapped_column(Enum(NodeType), nullable=False)
    parent_id : Mapped[str | None ] = mapped_column(ForeignKey("nodes.node_id"), nullable=True)
    timestamp : Mapped[float] = mapped_column(nullable=False)
    prompt_hash : Mapped[str | None ] = mapped_column(nullable=True)
    model : Mapped[str | None ] = mapped_column(nullable=True)
    temperature : Mapped[float | None ] = mapped_column(nullable=True)
    seed : Mapped[int | None ] = mapped_column(nullable=True)
    latency_ms : Mapped[float | None ] = mapped_column(nullable=True)
    cost_usd : Mapped[float | None ] = mapped_column(nullable=True)
    input_data : Mapped[Any | None ] = mapped_column(JSON , nullable=True)
    output_data : Mapped[Any | None ] = mapped_column(JSON , nullable=True)
    memory_snapshot : Mapped[dict | None ] = mapped_column(JSON , nullable=True)
    tool_parameters : Mapped[dict | None ] = mapped_column(JSON , nullable=True)
    tool_result : Mapped[Any | None ] = mapped_column(JSON , nullable=True)
    exception : Mapped[str | None ] = mapped_column(nullable=True)
    retry_count : Mapped[int] = mapped_column(default=0,nullable=False)
    tags : Mapped[List[str] | None] = mapped_column(JSON , nullable=True)
 

    run : Mapped["Run"] = relationship(back_populates="nodes")
    parent: Mapped["Nodes | None"] = relationship("Nodes",remote_side="Nodes.node_id",back_populates="children")
    children: Mapped[List["Nodes"]] = relationship("Nodes",back_populates="parent")

class Edges(Base):
    __tablename__ = "edges"
    edge_id : Mapped[int] = mapped_column(primary_key=True)
    run_id : Mapped[str] = mapped_column(ForeignKey("runs.run_id"))
    source_id : Mapped[str] = mapped_column(ForeignKey("nodes.node_id"))
    target_id : Mapped[str] = mapped_column(ForeignKey("nodes.node_id"))
    edge_type : Mapped[EdgeType] = mapped_column(Enum(EdgeType), nullable=False)
    edge_metadata : Mapped[dict|None] = mapped_column(JSON, nullable=True)

    run : Mapped["Run"] = relationship(back_populates="edges")
    source : Mapped["Nodes"] = relationship("Nodes", foreign_keys=[source_id])
    target : Mapped["Nodes"] = relationship("Nodes", foreign_keys=[target_id])