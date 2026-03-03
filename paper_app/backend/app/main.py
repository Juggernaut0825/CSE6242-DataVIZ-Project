import json
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal, init_db
from app.memory_client import post_agentic_search, post_extract
from app.models import ChatMessage, ChatSession, Project, beijing_now
from app.graph_query import get_project_graph
from app.reasoner_agent import ReasonerAgent
from app.schemas import (
    ChatSessionCreate,
    ChatSessionRead,
    ChatMessageRead,
    ProjectCreate,
    ProjectRead,
)

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_allow_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/projects", response_model=List[ProjectRead])
def list_projects(db: Session = Depends(get_db)) -> List[Project]:
    return db.query(Project).order_by(Project.created_at.desc()).all()


@app.post("/projects", response_model=ProjectRead)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)) -> Project:
    project = Project(
        name=payload.name,
        type=payload.type,
        description=payload.description,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@app.post("/chat_sessions", response_model=ChatSessionRead)
def create_chat_session(
    payload: ChatSessionCreate, db: Session = Depends(get_db)
) -> ChatSession:
    session = ChatSession(project_id=payload.project_id, title=payload.title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_or_create_current_session(db: Session, project_id: str) -> ChatSession:
    session = (
        db.query(ChatSession)
        .filter(ChatSession.project_id == project_id, ChatSession.is_current.is_(True))
        .order_by(ChatSession.created_at.desc())
        .first()
    )
    if session:
        return session
    session = ChatSession(project_id=project_id, title="Main")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def save_message(
    db: Session,
    project_id: str,
    session_id: str,
    role: str,
    content: str,
    reasoning_trace: Optional[str] = None,
    search_results: Optional[str] = None,
) -> ChatMessage:
    message_count = (
        db.query(func.count(ChatMessage.id))
        .filter(ChatMessage.session_id == session_id)
        .scalar()
        or 0
    )
    message = ChatMessage(
        session_id=session_id,
        project_id=project_id,
        role=role,
        content=content,
        reasoning_trace=reasoning_trace,
        search_results=search_results,
        has_thinking=bool(reasoning_trace),
        message_index=message_count + 1,
    )
    db.add(message)

    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session:
        session.message_count = message_count + 1
        session.last_message_at = beijing_now()

    project = db.query(Project).filter(Project.id == project_id).first()
    if project:
        project.message_count = (project.message_count or 0) + 1
        project.last_message_preview = (content or "")[:200]
        project.last_active_at = beijing_now()

    db.commit()
    db.refresh(message)
    return message


@app.post("/extract")
async def extract_memory(payload: Dict[str, Any]) -> Dict[str, Any]:
    return await post_extract(payload)


@app.post("/agenticSearch")
async def agentic_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    return await post_agentic_search(payload)


@app.get("/graph/{project_id}")
def graph(project_id: str, limit: int = 200) -> Dict[str, Any]:
    return get_project_graph(project_id, limit=limit)


@app.delete("/projects/{project_id}")
def delete_project(project_id: str, db: Session = Depends(get_db)) -> Dict[str, bool]:
    db.query(ChatMessage).filter(ChatMessage.project_id == project_id).delete()
    db.query(ChatSession).filter(ChatSession.project_id == project_id).delete()
    deleted = db.query(Project).filter(Project.id == project_id).delete()
    db.commit()
    return {"ok": deleted > 0}


@app.get("/projects/{project_id}/messages", response_model=List[ChatMessageRead])
def get_project_messages(project_id: str, db: Session = Depends(get_db)) -> List[ChatMessage]:
    session = get_or_create_current_session(db, project_id)
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.message_index.asc())
        .all()
    )
    return messages


@app.post("/chat/stream")
async def chat_stream(
    payload: Dict[str, Any], db: Session = Depends(get_db)
) -> StreamingResponse:
    query = payload.get("query")
    project_id = payload.get("project_id")
    top_k = payload.get("top_k", 5)

    if not query or not project_id:
        raise HTTPException(status_code=400, detail="query 和 project_id 为必填")

    search_payload = {"query": query, "project_id": project_id, "top_k": top_k}
    search_results = await post_agentic_search(search_payload)

    context_text = json.dumps(search_results, ensure_ascii=False)
    system_prompt = (
        "你是一个仿人类对话助手，可以跟用户进行任何从日常到专业的对话。你大脑里的知识来自于提供的检索结果，请基于提供的检索结果回答用户问题。"
    )

    agent = ReasonerAgent(system_prompt=system_prompt)

    async def event_stream() -> AsyncGenerator[str, None]:
        full_content = ""
        full_reasoning = ""
        session = get_or_create_current_session(db, project_id)
        yield f"data: {json.dumps({'type': 'search', 'payload': search_results}, ensure_ascii=False)}\n\n"
        async for chunk in agent.stream(f"你大脑里检索到的知识有:\n{context_text}\n\nUser:\n{query}"):
            if chunk.get("type") == "content_chunk":
                full_content += chunk.get("content", "")
            elif chunk.get("type") == "reasoning":
                full_reasoning += chunk.get("content", "")
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        if full_content:
            try:
                save_message(db, project_id, session.id, "user", query)
                save_message(
                    db,
                    project_id,
                    session.id,
                    "assistant",
                    full_content,
                    reasoning_trace=full_reasoning,
                    search_results=json.dumps(search_results, ensure_ascii=False),
                )
                await post_extract(
                    {
                        "text": query,
                        "project_id": project_id,
                        "source_name": "Chat User",
                        "content_type": "conversation",
                        "replace": False,
                        "metadata": {"role": "user"},
                    }
                )
                await post_extract(
                    {
                        "text": full_content,
                        "project_id": project_id,
                        "source_name": "Chat Assistant",
                        "content_type": "conversation",
                        "replace": False,
                        "metadata": {"role": "assistant", "reasoning": full_reasoning},
                    }
                )
            except Exception:
                pass

    return StreamingResponse(event_stream(), media_type="text/event-stream")
