import logging
import time
from typing import Optional
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

logger = logging.getLogger(__name__)

encoded_password = quote_plus(settings.postgres_password)

postgres_dsn = (
    f"postgresql+psycopg://{settings.postgres_user}:{encoded_password}"
    f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_database}"
)

engine = create_engine(
    postgres_dsn,
    pool_pre_ping=True,
    pool_size=settings.postgres_pool_size,
    max_overflow=settings.postgres_max_overflow,
    echo=settings.postgres_echo,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def wait_for_postgres(max_wait_seconds: int = 90, initial_interval: float = 0.5) -> None:
    """Wait until Postgres accepts connections (Docker may start after uvicorn)."""
    deadline = time.monotonic() + max_wait_seconds
    interval = initial_interval
    last_exc: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return
        except OperationalError as exc:
            last_exc = exc
            logger.warning(
                "PostgreSQL not reachable at %s:%s (%s). Retrying in %.1fs…",
                settings.postgres_host,
                settings.postgres_port,
                exc.orig.__class__.__name__ if getattr(exc, "orig", None) else type(exc).__name__,
                interval,
            )
            time.sleep(interval)
            interval = min(interval * 1.4, 4.0)

    hint = (
        f"Could not connect to PostgreSQL at {settings.postgres_host}:{settings.postgres_port} "
        "within the startup wait window.\n\n"
        "Fix:\n"
        "  1) Open Docker Desktop and wait until it is running.\n"
        "  2) From the project repository root, run:  docker compose up -d\n"
        "  3) Confirm Postgres is listening (default port 5432).\n"
    )
    raise RuntimeError(hint + f"\nLast error: {last_exc}") from last_exc


def init_db() -> None:
    wait_for_postgres()
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)
    with engine.begin() as connection:
        column_length = connection.execute(
            text(
                """
                SELECT character_maximum_length
                FROM information_schema.columns
                WHERE table_name = 'memory_units'
                  AND column_name = 'source_id'
                """
            )
        ).scalar()
        if column_length and int(column_length) < 255:
            connection.execute(text("ALTER TABLE memory_units ALTER COLUMN source_id TYPE VARCHAR(255)"))
