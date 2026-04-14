from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ProjectCreate(BaseModel):
    name: str
    type: Optional[str] = None
    description: Optional[str] = None


class ProjectRead(BaseModel):
    id: str
    name: str
    type: Optional[str] = None
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ChatSessionCreate(BaseModel):
    project_id: str
    title: Optional[str] = None


class ChatSessionRead(BaseModel):
    id: str
    project_id: str
    title: Optional[str] = None
    message_count: int
    is_current: bool
    created_at: datetime
    last_message_at: datetime

    class Config:
        from_attributes = True


class ChatMessageRead(BaseModel):
    id: str
    session_id: str
    project_id: str
    role: str
    content: str
    reasoning_trace: Optional[str] = None
    search_results: Optional[str] = None
    has_thinking: bool
    message_index: int
    created_at: datetime

    class Config:
        from_attributes = True


class FileIngestRequest(BaseModel):
    file_path: str
    project_id: str


class ExtractMemoryRequest(BaseModel):
    text: str
    project_id: str
    source_name: Optional[str] = None
    content_type: str = "conversation"
    replace: bool = False
    metadata: Optional[Dict[str, Any]] = None


class GraphNodeRead(BaseModel):
    id: str
    label: str
    kind: str
    score: float = 0.0
    detail: Optional[Dict[str, Any]] = None


class GraphEdgeRead(BaseModel):
    id: str
    source: str
    target: str
    type: str
    weight: float = 1.0


class GraphPayloadRead(BaseModel):
    nodes: List[GraphNodeRead]
    edges: List[GraphEdgeRead]
    meta: Dict[str, Any]
