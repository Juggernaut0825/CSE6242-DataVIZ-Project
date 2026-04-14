import uuid
from datetime import datetime, timedelta, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, Integer, String, Text

from app.database import Base
from app.config import settings


def beijing_now() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=8)


class Project(Base):
    __tablename__ = "projects"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, index=True)
    type = Column(String(100), index=True)
    description = Column(Text)

    message_count = Column(Integer, default=0)
    file_count = Column(Integer, default=0)
    last_message_preview = Column(String(200))

    created_at = Column(DateTime, default=beijing_now, index=True)
    updated_at = Column(DateTime, default=beijing_now, onupdate=beijing_now)
    last_active_at = Column(DateTime, default=beijing_now, index=True)

    status = Column(String(20), default="active", index=True)


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False, index=True)

    title = Column(String(255))
    message_count = Column(Integer, default=0)
    is_current = Column(Boolean, default=True, index=True)

    created_at = Column(DateTime, default=beijing_now, index=True)
    last_message_at = Column(DateTime, default=beijing_now, index=True)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), nullable=False, index=True)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False, index=True)

    role = Column(String(50), nullable=False, index=True)
    content = Column(Text, nullable=False)
    reasoning_trace = Column(Text)
    search_results = Column(Text)
    has_thinking = Column(Boolean, default=False)
    message_index = Column(Integer, default=0)

    created_at = Column(DateTime, default=beijing_now, index=True)


class SourceFile(Base):
    __tablename__ = "source_files"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False, index=True)

    filename = Column(String(512), nullable=False)
    file_path = Column(Text)
    file_hash = Column(String(128), nullable=False, index=True)
    media_type = Column(String(64), nullable=False, index=True)
    parser = Column(String(100), default="local")
    page_count = Column(Integer, default=0)
    status = Column(String(32), default="processed", index=True)
    metadata_json = Column(JSON)

    created_at = Column(DateTime, default=beijing_now, index=True)


class MemoryUnit(Base):
    __tablename__ = "memory_units"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False, index=True)
    source_type = Column(String(64), nullable=False, index=True)
    source_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), index=True)
    file_id = Column(String(36), ForeignKey("source_files.id"), index=True)

    text = Column(Text, nullable=False)
    summary = Column(Text)
    metadata_json = Column(JSON)
    semantic_labels = Column(JSON)
    embedding = Column(Vector(settings.embedding_dimensions), nullable=False)

    created_at = Column(DateTime, default=beijing_now, index=True)


class RetrievalEvent(Base):
    __tablename__ = "retrieval_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False, index=True)
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), index=True)
    query_text = Column(Text, nullable=False)
    selected_unit_ids = Column(JSON)
    metadata_json = Column(JSON)

    created_at = Column(DateTime, default=beijing_now, index=True)


class ReasoningTrace(Base):
    __tablename__ = "reasoning_traces"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False, index=True)
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), index=True)
    assistant_message_id = Column(String(36), ForeignKey("chat_messages.id"), index=True)

    query_text = Column(Text, nullable=False)
    answer_text = Column(Text)
    semantic_nodes = Column(JSON)
    reasoning_edges = Column(JSON)
    citations = Column(JSON)
    trace_steps = Column(JSON)

    created_at = Column(DateTime, default=beijing_now, index=True)
