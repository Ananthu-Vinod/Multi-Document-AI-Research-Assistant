"""
Configuration module for RAG application.
Handles environment variables, paths, and tunable pipeline settings.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root (parent of backend/)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

class Config:
    """Application configuration settings."""

    # Paths
    BASE_DIR: Path = BASE_DIR
    UPLOADS_DIR: Path = BASE_DIR / "uploads"
    CHROMA_PERSIST_DIR: Path = BASE_DIR / "chroma_db"
    DATA_DIR: Path = BASE_DIR / "data"
    DOCUMENT_REGISTRY_PATH: Path = DATA_DIR / "document_registry.json"

    # API Keys
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

    # LLM Provider Selection
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    # Tried in order if primary model returns 404/429 (Google retires or quotas models)
    GEMINI_MODEL_FALLBACKS: str = os.getenv(
        "GEMINI_MODEL_FALLBACKS",
        "gemini-2.5-flash,gemini-2.0-flash,gemini-2.0-flash-lite",
    )
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Embedding Settings
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))

    # Chunking Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Vector Database Settings
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "documents")

    # Retrieval Settings
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "4"))
    RETRIEVAL_CANDIDATE_MULTIPLIER: int = int(
        os.getenv("RETRIEVAL_CANDIDATE_MULTIPLIER", "3")
    )
    HYBRID_ALPHA: float = float(os.getenv("HYBRID_ALPHA", "0.5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.25"))
    MIN_RETRIEVAL_RESULTS: int = int(os.getenv("MIN_RETRIEVAL_RESULTS", "1"))
    MAX_RETRIEVAL_RESULTS: int = int(os.getenv("MAX_RETRIEVAL_RESULTS", "12"))

    # Context / token budgeting
    MAX_CONTEXT_CHARS: int = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
    MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "3000"))
    CHARS_PER_TOKEN_ESTIMATE: int = int(os.getenv("CHARS_PER_TOKEN_ESTIMATE", "4"))

    # Reranking (off by default — first query loads ~100MB model and can take 20s+)
    ENABLE_RERANKING: bool = os.getenv("ENABLE_RERANKING", "false").lower() == "true"
    RERANKER_MODEL: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "4"))
    RERANK_TIMEOUT_SECONDS: float = float(os.getenv("RERANK_TIMEOUT_SECONDS", "30"))

    # LLM API
    LLM_REQUEST_TIMEOUT_SECONDS: float = float(
        os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "120")
    )

    # BM25
    BM25_K1: float = float(os.getenv("BM25_K1", "1.5"))
    BM25_B: float = float(os.getenv("BM25_B", "0.75"))

    # Session
    SESSION_ID_METADATA_KEY: str = "session_id"

    # API / deployment (Render sets PORT)
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
    CORS_ORIGINS: str = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501,*",
    )
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")

    @classmethod
    def cors_origin_list(cls) -> list[str]:
        """Parse comma-separated CORS origins."""
        return [o.strip() for o in cls.CORS_ORIGINS.split(",") if o.strip()]

    @classmethod
    def ensure_directories(cls) -> None:
        """Create required runtime directories."""
        cls.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration for LLM usage."""
        if cls.LLM_PROVIDER == "gemini" and not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required when using Gemini")
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI")
