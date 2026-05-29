"""Health and status routes."""

from fastapi import APIRouter

from config import Config
from routes.schemas import HealthResponse, StatsResponse
from services.rag_service import RAGService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    service = RAGService.get_instance()
    stats = service.stats()
    return HealthResponse(
        status="ok",
        chunk_count=stats["chunk_count"],
        llm_configured=stats["llm_configured"],
    )


@router.get("/stats", response_model=StatsResponse)
def stats(session_id: str | None = None) -> StatsResponse:
    service = RAGService.get_instance(session_id)
    data = service.stats()
    return StatsResponse(**data)
