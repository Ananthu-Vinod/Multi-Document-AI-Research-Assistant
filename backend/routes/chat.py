"""Chat / RAG query routes with PostgreSQL logging and rate limiting."""

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from auth import get_current_user
from database import get_db
from middleware.rate_limiter import limiter
from models import ChatMessageModel, User
from routes.schemas import ChatRequest, ChatResponse, ChunkOut
from services.rag_service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


def _to_chat_response(response, session_id: str) -> ChatResponse:
    return ChatResponse(
        answer=response.answer,
        chunks=[ChunkOut(**c) for c in response.chunks],
        citations=response.citations,
        latency_ms=response.latency_ms,
        search_mode=response.search_mode,
        session_id=session_id,
    )


def _log_chat_history(
    db: Session,
    user_id: Optional[int],
    session_id: str,
    question: str,
    answer: Optional[str],
    citations: list[str],
    search_mode: str,
    latency_ms: float,
):
    try:
        citations_json = json.dumps(citations) if citations else None
        # User message
        db.add(ChatMessageModel(
            user_id=user_id,
            session_id=session_id,
            role="user",
            content=question,
        ))
        # Assistant message
        if answer:
            db.add(ChatMessageModel(
                user_id=user_id,
                session_id=session_id,
                role="assistant",
                content=answer,
                citations=citations_json,
                search_mode=search_mode,
                latency_ms=int(latency_ms),
            ))
        db.commit()
    except Exception as exc:
        logger.warning("Failed to store chat history in DB: %s", exc)


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("60 per minute")
def chat(
    request: Request,
    payload: ChatRequest,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatResponse:
    """Run RAG pipeline and return answer with citations."""
    service = RAGService.get_instance(payload.session_id)
    try:
        service.initialize_llm()
    except ValueError as exc:
        logger.warning("LLM not configured: %s", exc)

    try:
        if payload.stream:
            raise HTTPException(
                status_code=400,
                detail="Use POST /chat/stream for streaming responses",
            )
        result = service.ask(
            payload.question,
            use_hybrid=payload.use_hybrid,
            source_filter=payload.source_filter,
            stream=False,
            remember=payload.remember,
        )

        _log_chat_history(
            db=db,
            user_id=current_user.id if current_user else None,
            session_id=service.session_id,
            question=payload.question,
            answer=result.answer,
            citations=result.citations,
            search_mode=result.search_mode,
            latency_ms=result.latency_ms,
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _to_chat_response(result, service.session_id)


@router.post("/chat/stream")
@limiter.limit("60 per minute")
def chat_stream(
    request: Request,
    payload: ChatRequest,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Stream LLM tokens as Server-Sent Events."""
    service = RAGService.get_instance(payload.session_id)
    try:
        service.initialize_llm()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not service.llm:
        raise HTTPException(status_code=503, detail="LLM not configured")

    try:
        result = service.ask(
            payload.question,
            use_hybrid=payload.use_hybrid,
            source_filter=payload.source_filter,
            stream=True,
            remember=False,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if result.stream_generator is None:
        raise HTTPException(status_code=500, detail="Streaming unavailable")

    def event_stream():
        meta = {
            "citations": result.citations,
            "chunks": result.chunks,
            "search_mode": result.search_mode,
            "latency_ms": result.latency_ms,
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"
        full_tokens = []
        for token in result.stream_generator:
            full_tokens.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "event: done\ndata: {}\n\n"

        full_answer = "".join(full_tokens)
        _log_chat_history(
            db=db,
            user_id=current_user.id if current_user else None,
            session_id=service.session_id,
            question=payload.question,
            answer=full_answer,
            citations=result.citations,
            search_mode=result.search_mode,
            latency_ms=result.latency_ms,
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )
