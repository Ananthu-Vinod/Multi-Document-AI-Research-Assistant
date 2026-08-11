"""
Health, readiness, and metrics endpoints for production monitoring.
"""

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session
from sqlalchemy import text

from config import Config
from database import get_db
from models import DocumentModel, User, ChatMessageModel
from routes.schemas import HealthResponse, StatsResponse
from services.rag_service import RAGService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
@router.get("/healthz", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Liveness probe returning application operational status."""
    service = RAGService.get_instance()
    stats = service.stats()
    return HealthResponse(
        status="ok",
        chunk_count=stats["chunk_count"],
        llm_configured=stats["llm_configured"],
    )


@router.get("/readyz")
def readiness_check(db: Session = Depends(get_db)):
    """Readiness probe checking database and vector store connectivity."""
    db_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    service = RAGService.get_instance()
    vector_ok = service.vector_store.collection_exists() or True

    if db_ok and vector_ok:
        return {"status": "ready", "database": "connected", "vector_store": "ok"}
    return Response(content='{"status":"unready"}', status_code=503, media_type="application/json")


@router.get("/stats", response_model=StatsResponse)
def stats(session_id: str | None = None) -> StatsResponse:
    service = RAGService.get_instance(session_id)
    data = service.stats()
    return StatsResponse(**data)


@router.get("/metrics")
def metrics(db: Session = Depends(get_db)):
    """Metrics endpoint for monitoring system usage."""
    service = RAGService.get_instance()
    rag_stats = service.stats()
    
    total_users = db.query(User).count()
    total_docs = db.query(DocumentModel).count()
    total_messages = db.query(ChatMessageModel).count()

    return {
        "status": "ok",
        "llm_provider": Config.LLM_PROVIDER,
        "embedding_model": Config.EMBEDDING_MODEL,
        "total_users": total_users,
        "total_documents": total_docs,
        "total_chat_messages": total_messages,
        "vector_chunk_count": rag_stats.get("chunk_count", 0),
        "reranking_enabled": Config.ENABLE_RERANKING,
    }
