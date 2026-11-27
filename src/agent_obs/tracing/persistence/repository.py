"""SQLAlchemy implementation of the tracing repository."""

from sqlalchemy.orm import Session

from ..tracing_repository import TracingRepository
from ..trace_graph.execution_graph import ExecutionTree
from .dao import TracingDAO


class SQLAlchemyTracingRepository(TracingRepository):
    """Persist execution graphs using SQLAlchemy."""

    def __init__(self, session: Session) -> None:
        self.dao = TracingDAO(session)

    def save(self, graph: ExecutionTree) -> None:
        self.dao.save(graph)

    def get(self, run_id: str) -> ExecutionTree | None:
        return self.dao.get(run_id)
