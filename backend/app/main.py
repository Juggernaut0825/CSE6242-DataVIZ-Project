import json
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import APIError, AuthenticationError
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal, init_db
from app.models import (
    ChatMessage,
    ChatSession,
    MemoryUnit,
    Project,
    ReasoningTrace,
    RetrievalEvent,
    SourceFile,
    beijing_now,
)
from app.reasoner_agent import ReasonerAgent
from app.schemas import (
    ChatMessageRead,
    ChatSessionCreate,
    ChatSessionRead,
    ExtractMemoryRequest,
    FileIngestRequest,
    ProjectCreate,
    ProjectRead,
)
from app.services.graph_service import GraphService
from app.services.memory_service import MemoryService

app = FastAPI(title=settings.app_name)
graph_service = GraphService()
memory_service = MemoryService(graph_service)

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
    graph_service.wait_for_neo4j()
    graph_service.ensure_schema()


@app.on_event("shutdown")
def on_shutdown() -> None:
    graph_service.close()


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
        project.last_message_preview = (content or "")[:200]
        project.last_active_at = beijing_now()

    db.commit()
    db.refresh(message)
    return message


def build_answer_context(
    retrieved_units: List[Dict[str, Any]],
    char_budget: int,
    chars_per_unit: int,
) -> str:
    sections: List[str] = []
    used = 0
    for index, item in enumerate(retrieved_units, start=1):
        metadata = item.get("metadata") or {}
        source = metadata.get("filename") or item.get("source_id") or item.get("source_type")
        page = metadata.get("page")
        chunk_index = metadata.get("chunk_index")
        source_parts = [str(source)]
        if page is not None:
            source_parts.append(f"page {page}")
        if chunk_index is not None:
            source_parts.append(f"chunk {chunk_index}")
        source_label = ", ".join(source_parts)
        header = (
            f"[{index}] {source_label}; "
            f"retrieval={item.get('retrieval_source', 'vector')}; "
            f"score={item.get('score', 0)}\n"
            f"Summary: {item.get('summary', '')}\n"
            "Evidence: "
        )
        remaining = char_budget - used - len(header) - 2
        if remaining <= 0:
            break
        evidence = (item.get("text") or "").strip()
        evidence = evidence[: min(chars_per_unit, remaining)]
        if not evidence:
            continue
        section = header + evidence
        sections.append(section)
        used += len(section) + 2
    return "\n\n".join(sections)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "stack": {
            "postgres": settings.postgres_database,
            "neo4j": settings.neo4j_uri,
            "neo4j_available": graph_service.available,
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
        },
    }


@app.get("/projects", response_model=List[ProjectRead])
def list_projects(db: Session = Depends(get_db)) -> List[Project]:
    return db.query(Project).order_by(Project.created_at.desc()).all()


@app.post("/projects", response_model=ProjectRead)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)) -> Project:
    project = Project(name=payload.name, type=payload.type, description=payload.description)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@app.delete("/projects/{project_id}")
def delete_project(project_id: str, db: Session = Depends(get_db)) -> Dict[str, bool]:
    db.query(ReasoningTrace).filter(ReasoningTrace.project_id == project_id).delete()
    db.query(RetrievalEvent).filter(RetrievalEvent.project_id == project_id).delete()
    db.query(MemoryUnit).filter(MemoryUnit.project_id == project_id).delete()
    db.query(SourceFile).filter(SourceFile.project_id == project_id).delete()
    db.query(ChatMessage).filter(ChatMessage.project_id == project_id).delete()
    db.query(ChatSession).filter(ChatSession.project_id == project_id).delete()
    deleted = db.query(Project).filter(Project.id == project_id).delete()
    db.commit()
    graph_service.delete_project_graph(project_id)
    return {"ok": deleted > 0}


@app.post("/chat_sessions", response_model=ChatSessionRead)
def create_chat_session(payload: ChatSessionCreate, db: Session = Depends(get_db)) -> ChatSession:
    existing_count = (
        db.query(func.count(ChatSession.id))
        .filter(ChatSession.project_id == payload.project_id)
        .scalar()
        or 0
    )
    db.query(ChatSession).filter(ChatSession.project_id == payload.project_id).update(
        {ChatSession.is_current: False},
        synchronize_session=False,
    )
    session = ChatSession(
        project_id=payload.project_id,
        title=payload.title or f"Chat {existing_count + 1}",
        is_current=True,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@app.get("/projects/{project_id}/chat_sessions", response_model=List[ChatSessionRead])
def list_chat_sessions(project_id: str, db: Session = Depends(get_db)) -> List[ChatSession]:
    return (
        db.query(ChatSession)
        .filter(ChatSession.project_id == project_id)
        .order_by(ChatSession.is_current.desc(), ChatSession.last_message_at.desc())
        .all()
    )


@app.post("/chat_sessions/{session_id}/activate", response_model=ChatSessionRead)
def activate_chat_session(session_id: str, db: Session = Depends(get_db)) -> ChatSession:
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="chat session not found")
    project_id = session.project_id
    db.query(ChatSession).filter(ChatSession.project_id == project_id).update(
        {ChatSession.is_current: False},
        synchronize_session=False,
    )
    updated = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.project_id == project_id)
        .update({ChatSession.is_current: True}, synchronize_session=False)
    )
    if not updated:
        raise HTTPException(status_code=404, detail="chat session not found")
    db.commit()
    refreshed = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not refreshed:
        raise HTTPException(status_code=404, detail="chat session not found")
    return refreshed


@app.delete("/chat_sessions/{session_id}")
def delete_chat_session(session_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="chat session not found")

    project_id = session.project_id
    was_current = bool(session.is_current)

    db.query(ReasoningTrace).filter(ReasoningTrace.session_id == session_id).delete()
    db.query(RetrievalEvent).filter(RetrievalEvent.session_id == session_id).delete()
    db.query(MemoryUnit).filter(MemoryUnit.session_id == session_id).delete()
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
    deleted = db.query(ChatSession).filter(ChatSession.id == session_id).delete()

    remaining_sessions = (
        db.query(ChatSession)
        .filter(ChatSession.project_id == project_id)
        .order_by(ChatSession.last_message_at.desc(), ChatSession.created_at.desc())
        .all()
    )
    next_current_id = None
    if remaining_sessions:
        next_current_id = remaining_sessions[0].id
        if was_current or not any(item.is_current for item in remaining_sessions):
            db.query(ChatSession).filter(ChatSession.project_id == project_id).update(
                {ChatSession.is_current: False},
                synchronize_session=False,
            )
            db.query(ChatSession).filter(ChatSession.id == next_current_id).update(
                {ChatSession.is_current: True},
                synchronize_session=False,
            )

    db.commit()
    graph_service.delete_session_graph(project_id, session_id)
    memory_service._refresh_project_stats(db, project_id)
    return {"ok": deleted > 0, "next_session_id": next_current_id}


@app.get("/projects/{project_id}/messages", response_model=List[ChatMessageRead])
def get_project_messages(
    project_id: str,
    session_id: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
) -> List[ChatMessage]:
    if session_id:
        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id, ChatSession.project_id == project_id)
            .first()
        )
        if not session:
            raise HTTPException(status_code=404, detail="chat session not found")
    else:
        session = get_or_create_current_session(db, project_id)
    return (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.message_index.asc())
        .all()
    )


@app.post("/extract")
async def extract_memory(
    payload: ExtractMemoryRequest, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    source_type = "conversation_turn"
    if (payload.source_name or "").lower() == "quick capture":
        source_type = "quick_capture"
    return await memory_service.ingest_capture(
        db=db,
        project_id=payload.project_id,
        text=payload.text,
        source_type=source_type,
        source_name=payload.source_name,
        metadata=payload.metadata,
    )


@app.post("/files/ingest")
async def ingest_file(
    payload: FileIngestRequest, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    return await memory_service.ingest_file(
        db=db,
        project_id=payload.project_id,
        file_path=payload.file_path,
    )


@app.post("/files/ingest_upload")
async def ingest_file_upload(
    project_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")
    filename = file.filename or "upload"
    return await memory_service.ingest_file_bytes(
        db=db,
        project_id=project_id,
        filename=filename,
        data=raw,
    )


@app.post("/agenticSearch")
async def agentic_search(payload: Dict[str, Any], db: Session = Depends(get_db)) -> Dict[str, Any]:
    query = (payload.get("query") or "").strip()
    project_id = payload.get("project_id")
    top_k = int(payload.get("top_k") or settings.retrieval_top_k)
    session_id = payload.get("session_id")
    source_name = payload.get("source_name") or payload.get("filename")
    if not query or not project_id:
        raise HTTPException(status_code=400, detail="query and project_id are required")
    if session_id:
        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id, ChatSession.project_id == project_id)
            .first()
        )
        if not session:
            raise HTTPException(status_code=404, detail="chat session not found")
    else:
        session = get_or_create_current_session(db, project_id)
    return await memory_service.search(
        db=db,
        project_id=project_id,
        query=query,
        top_k=top_k,
        session_id=session.id,
        source_name=source_name,
    )


@app.get("/graph/{project_id}")
def graph(
    project_id: str,
    view: str = Query(default="macro"),
    limit: int = Query(default=80, ge=5, le=300),
    focus_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    return graph_service.get_graph(
        project_id=project_id,
        view=view,
        limit=limit,
        focus_id=focus_id,
    )


@app.post("/chat/stream")
async def chat_stream(payload: Dict[str, Any], db: Session = Depends(get_db)) -> StreamingResponse:
    query = (payload.get("query") or "").strip()
    project_id = payload.get("project_id")
    top_k = int(payload.get("top_k") or settings.retrieval_top_k)
    requested_session_id = payload.get("session_id")
    skip_memory_ingest = bool(payload.get("skip_memory_ingest"))
    source_name = payload.get("source_name") or payload.get("filename")
    context_budget = int(payload.get("context_budget") or settings.answer_context_char_budget)
    chars_per_unit = int(payload.get("evidence_chars_per_unit") or settings.answer_evidence_chars_per_unit)
    if not query or not project_id:
        raise HTTPException(status_code=400, detail="query and project_id are required")

    if requested_session_id:
        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == requested_session_id, ChatSession.project_id == project_id)
            .first()
        )
        if not session:
            raise HTTPException(status_code=404, detail="chat session not found")
    else:
        session = get_or_create_current_session(db, project_id)
    session_id = session.id
    search_results = await memory_service.search(
        db=db,
        project_id=project_id,
        query=query,
        top_k=top_k,
        session_id=session_id,
        source_name=source_name,
    )
    context_text = build_answer_context(
        search_results["retrieved_units"],
        char_budget=context_budget,
        chars_per_unit=chars_per_unit,
    )
    system_prompt = (
        "You are PaperMem Copilot, an explainable memory assistant. "
        "Answer only from the retrieved evidence. "
        "When the question asks for numbers, benchmark scores, definitions, datasets, or named methods, "
        "copy those details exactly from the evidence. "
        "For broad entity questions, if the evidence describes the entity's role, actions, outputs, or relationships, "
        "answer with how the entity is presented in the evidence instead of requiring a dictionary-style definition. "
        "If the evidence gives partial but relevant context, answer the supported part and explicitly say what is missing. "
        "Only say that information is not found when the retrieved evidence contains no relevant facts for the question. "
        "Keep the answer concise and cite evidence bracket numbers when useful."
    )
    agent = ReasonerAgent(system_prompt=system_prompt, model_provider=settings.llm_model)

    async def event_stream() -> AsyncGenerator[str, None]:
        full_content = ""
        stream_error: Optional[str] = None
        user_message = save_message(db, project_id, session_id, "user", query)
        user_message_id = user_message.id
        yield f"data: {json.dumps({'type': 'search', 'payload': search_results}, ensure_ascii=False)}\n\n"
        for step in search_results["trace_steps"]:
            reasoning_text = f"{step['title']}: {step['detail']}\n"
            yield f"data: {json.dumps({'type': 'reasoning', 'content': reasoning_text}, ensure_ascii=False)}\n\n"
        try:
            async for chunk in agent.stream(
                f"User query:\n{query}\n\nRetrieved evidence:\n{context_text}\n\n"
                "Give the final answer. Do not add claims that are not supported by the retrieved evidence. "
                "Prefer a partial grounded answer over a blanket not-found response when the evidence is relevant but incomplete."
            ):
                if chunk.get("type") == "content_chunk":
                    full_content += chunk.get("content", "")
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except AuthenticationError:
            stream_error = (
                "LLM authentication failed (401). Set a valid LLM_API_KEY from "
                "https://openrouter.ai/keys in backend/.env and restart the server."
            )
            yield f"data: {json.dumps({'type': 'error', 'message': stream_error}, ensure_ascii=False)}\n\n"
        except APIError as exc:
            stream_error = getattr(exc, "message", None) or str(exc)
            yield f"data: {json.dumps({'type': 'error', 'message': stream_error}, ensure_ascii=False)}\n\n"

        trace_json = json.dumps(search_results["trace_steps"], ensure_ascii=False)
        search_json = json.dumps(search_results, ensure_ascii=False)

        if full_content:
            assistant_message = save_message(
                db,
                project_id,
                session_id,
                "assistant",
                full_content,
                reasoning_trace=trace_json,
                search_results=search_json,
            )
            assistant_message_id = assistant_message.id
            db.add(
                ReasoningTrace(
                    project_id=project_id,
                    session_id=session_id,
                    assistant_message_id=assistant_message_id,
                    query_text=query,
                    answer_text=full_content,
                    semantic_nodes=search_results["semantic_nodes"],
                    reasoning_edges=search_results["reasoning_edges"],
                    citations=search_results["citations"],
                    trace_steps=search_results["trace_steps"],
                )
            )
            db.commit()
            if not skip_memory_ingest:
                await memory_service.ingest_capture(
                    db=db,
                    project_id=project_id,
                    text=query,
                    source_type="conversation_turn",
                    source_name="Chat User",
                    session_id=session_id,
                    metadata={"role": "user", "message_id": user_message_id},
                )
                await memory_service.ingest_capture(
                    db=db,
                    project_id=project_id,
                    text=full_content,
                    source_type="conversation_turn",
                    source_name="Chat Assistant",
                    session_id=session_id,
                    metadata={"role": "assistant", "message_id": assistant_message_id},
                )
        else:
            fallback = stream_error or (
                "No reply text was generated. If your model only streams internal reasoning, "
                "set LLM_MODEL to a general chat model (e.g. openai/gpt-4o-mini) in backend .env."
            )
            save_message(
                db,
                project_id,
                session_id,
                "assistant",
                fallback,
                reasoning_trace=trace_json,
                search_results=search_json,
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")
