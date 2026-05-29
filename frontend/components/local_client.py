"""In-process RAG client — no FastAPI server required."""

import sys
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from services.rag_service import RAGService  # noqa: E402

from components.api_client import APIError  # noqa: E402


class LocalRAGClient:
    """Same interface as RAGApiClient but calls RAGService directly."""

    mode = "local"

    def __init__(self, session_id: str | None = None):
        self.base_url = "local://embedded"
        self.session_id = session_id or "default"
        self.last_stream_meta: Dict[str, Any] = {}
        self._service: RAGService | None = None

    def _svc(self) -> RAGService:
        if self._service is None:
            self._service = RAGService.get_instance(self.session_id)
        return self._service

    def health(self) -> Dict[str, Any]:
        stats = self._svc().stats()
        return {
            "status": "ok",
            "chunk_count": stats["chunk_count"],
        }

    def stats(self, session_id: str | None = None) -> Dict[str, Any]:
        if session_id and session_id != self.session_id:
            return RAGService.get_instance(session_id).stats()
        return self._svc().stats()

    def upload(
        self,
        files: List[Tuple[str, bytes]],
        session_id: str | None = None,
        reindex: bool = False,
    ) -> Dict[str, Any]:
        svc = self._svc()
        if session_id and session_id != self.session_id:
            svc = RAGService.get_instance(session_id)
        if reindex:
            svc.reset()
            svc = RAGService.get_instance(session_id or self.session_id)
            self._service = svc
        return svc.process_upload_bytes(files)

    def chat(
        self,
        question: str,
        *,
        use_hybrid: bool = False,
        source_filter: str | None = None,
        session_id: str | None = None,
    ) -> Dict[str, Any]:
        svc = self._svc()
        if session_id and session_id != self.session_id:
            svc = RAGService.get_instance(session_id)
        try:
            svc.initialize_llm()
        except ValueError as exc:
            raise APIError(str(exc)) from exc

        try:
            result = svc.ask(
                question,
                use_hybrid=use_hybrid,
                source_filter=source_filter,
                stream=False,
            )
        except ValueError as exc:
            raise APIError(str(exc)) from exc

        return {
            "answer": result.answer,
            "chunks": result.chunks,
            "citations": result.citations,
            "latency_ms": result.latency_ms,
            "search_mode": result.search_mode,
            "session_id": svc.session_id,
        }

    def chat_stream_tokens(
        self,
        question: str,
        *,
        use_hybrid: bool = False,
        source_filter: str | None = None,
        session_id: str | None = None,
    ) -> Generator[str, None, None]:
        svc = self._svc()
        if session_id and session_id != self.session_id:
            svc = RAGService.get_instance(session_id)
        try:
            svc.initialize_llm()
        except ValueError as exc:
            raise APIError(str(exc)) from exc

        try:
            result = svc.ask(
                question,
                use_hybrid=use_hybrid,
                source_filter=source_filter,
                stream=True,
            )
        except ValueError as exc:
            raise APIError(str(exc)) from exc

        self.last_stream_meta = {
            "citations": result.citations,
            "chunks": result.chunks,
            "search_mode": result.search_mode,
            "latency_ms": result.latency_ms,
        }
        if result.stream_generator:
            yield from result.stream_generator
        elif result.answer:
            yield result.answer

    def reset(self, session_id: str | None = None) -> None:
        svc = self._svc()
        if session_id and session_id != self.session_id:
            RAGService.get_instance(session_id).reset()
        else:
            svc.reset()
            self._service = None
