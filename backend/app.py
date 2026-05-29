"""
FastAPI backend for Ask My Docs RAG application.

Run locally:
    cd backend && uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend package root is on sys.path
_BACKEND_ROOT = Path(__file__).resolve().parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from config import Config  # noqa: E402
from logger import setup_logger  # noqa: E402
from routes import chat, health, upload  # noqa: E402
from services.rag_service import RAGService  # noqa: E402

setup_logger(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure dirs and warm RAG service."""
    Config.ensure_directories()
    RAGService.get_instance()
    logger.info("RAG backend started (port=%s)", Config.API_PORT)
    yield
    logger.info("RAG backend shutdown")


app = FastAPI(
    title="Ask My Docs API",
    description="Production RAG API with hybrid retrieval, reranking, and citations",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.cors_origin_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(upload.router)


@app.get("/")
def root():
    return {
        "name": "Ask My Docs API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "chat": "POST /chat",
            "upload": "POST /upload",
        },
    }


@app.delete("/reset")
def reset_index(session_id: str | None = None):
    """Clear vector index and document registry for a session."""
    service = RAGService.get_instance(session_id)
    service.reset()
    return {"status": "reset", "session_id": service.session_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=Config.LOG_LEVEL == "DEBUG",
    )
