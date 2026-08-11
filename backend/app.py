"""
FastAPI backend for Ask My Docs RAG application with database persistence, rate limiting, and JWT authentication.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Ensure backend package root is on sys.path
_BACKEND_ROOT = Path(__file__).resolve().parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from config import Config  # noqa: E402
from database import init_db  # noqa: E402
from logger import setup_logger  # noqa: E402
from middleware.rate_limiter import limiter  # noqa: E402
from routes import auth, chat, health, upload  # noqa: E402
from services.rag_service import RAGService  # noqa: E402

setup_logger(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialize database tables and directories instantly."""
    Config.ensure_directories()
    init_db()
    logger.info("RAG backend started (port=%s)", Config.API_PORT)
    yield
    logger.info("RAG backend shutdown")


app = FastAPI(
    title="Ask My Docs API",
    description="Production RAG API with hybrid retrieval, JWT authentication, and citations",
    version="2.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration supporting localhost & production deployment domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(upload.router)


@app.get("/")
def root():
    return {
        "name": "Ask My Docs API",
        "version": "2.1.0",
        "docs": "/docs",
        "health": "/healthz",
        "metrics": "/metrics",
        "endpoints": {
            "register": "POST /auth/register",
            "login": "POST /auth/login",
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
