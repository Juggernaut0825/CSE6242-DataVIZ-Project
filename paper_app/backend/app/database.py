from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

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


def init_db() -> None:
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
