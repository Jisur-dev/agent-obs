from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Load environment variables from .env
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


class Base(DeclarativeBase):
    pass


def make_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    database_url = url if url is not None else DATABASE_URL

    if not database_url:
        raise ValueError("DATABASE_URL is not configured in .env")

    kwargs: dict = {}

    if database_url in ("sqlite://", "sqlite:///:memory:"):
        # Share one connection for in-memory databases (tests)
        kwargs = {
            "poolclass": StaticPool,
            "connect_args": {"check_same_thread": False},
        }

    engine = create_engine(database_url, echo=echo, **kwargs)

    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record) -> None:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    from . import models  # noqa: F401

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
