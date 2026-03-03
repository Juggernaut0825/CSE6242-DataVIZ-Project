from datetime import datetime
from typing import Optional

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
