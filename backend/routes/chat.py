"""Chat / RAG query routes."""

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

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


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
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
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _to_chat_response(result, service.session_id)


@router.post("/chat/stream")
def chat_stream(payload: ChatRequest):
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
        for token in result.stream_generator:
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )
