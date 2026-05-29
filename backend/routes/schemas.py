"""Pydantic request/response models for the API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    use_hybrid: bool = Field(False, description="Force BM25 + vector hybrid search")
    source_filter: Optional[str] = Field(
        None, description="Optional metadata source filter"
    )
    stream: bool = Field(False, description="Stream tokens (use /chat/stream)")
    session_id: Optional[str] = Field(None, description="Conversation session id")
    remember: bool = Field(True, description="Store turn in session memory")


class ChunkOut(BaseModel):
    content: str
    preview: str
    score: float
    source: Optional[str] = None
    page: Optional[int] = None
    citation: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: Optional[str]
    chunks: List[ChunkOut]
    citations: List[str]
    latency_ms: float
    search_mode: str
    session_id: str


class UploadResponse(BaseModel):
    chunks_added: int
    files_processed: int
    duplicates_skipped: int = 0
    total_chunks: int
    message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    chunk_count: int
    llm_configured: bool


class StatsResponse(BaseModel):
    session_id: str
    collection_exists: bool
    chunk_count: int
    llm_configured: bool
    reranking_enabled: bool
