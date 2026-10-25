from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

DEFAULT_URL = "sqlite:///agent_obs.db"


class Base(DeclarativeBase):
    pass


def make_engine(url: str = DEFAULT_URL, *, echo: bool = False) -> Engine:
    kwargs: dict = {}
    if url in ("sqlite://", "sqlite:///:memory:"):
        # one shared connection so the in-memory DB survives across sessions (tests)
        kwargs = {"poolclass": StaticPool, "connect_args": {"check_same_thread": False}}

    engine = create_engine(url, echo=echo, **kwargs)

    if engine.dialect.name == "sqlite":
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record) -> None:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")   # SQLite ignores FKs by default!
            cur.execute("PRAGMA journal_mode=WAL")  # better concurrent read/write
            cur.close()

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    from . import models  # noqa: F401  (registers tables on Base.metadata)
    Base.metadata.create_all(engine)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()