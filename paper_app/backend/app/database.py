from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

encoded_password = quote_plus(settings.mysql_password)

mysql_dsn = (
    f"mysql+pymysql://{settings.mysql_user}:{encoded_password}"
    f"@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}"
    f"?charset={settings.mysql_charset}"
)

engine = create_engine(
    mysql_dsn,
    pool_pre_ping=True,
    pool_size=settings.mysql_pool_size,
    max_overflow=settings.mysql_max_overflow,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
