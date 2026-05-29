"""RAG service orchestrating ingestion, retrieval, and generation."""

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from langchain_core.documents import Document

from config import Config
from ingestion.deduplication import DocumentDeduplicator
from ingestion.processor import DocumentProcessor
from llm.generator import LLMGenerator
from logger import get_logger, setup_logger
from retrieval.hybrid import HybridRetriever
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_expansion import should_use_hybrid
from retrieval.vector_store import VectorStore
from utils.citations import chunks_from_results, unique_citations
from utils.files import compute_file_hash, safe_upload_path

logger = get_logger(__name__)

# In-memory conversation history per session (last N turns)
_MAX_HISTORY_TURNS = 6


@dataclass
class RAGResponse:
    """Structured response from ask()."""

    answer: str | None
    chunks: List[Dict[str, Any]]
    citations: List[str]
    latency_ms: float
    search_mode: str
    stream_generator: Generator[str, None, None] | None = None
    conversation_id: str | None = None


class RAGService:
    """Production-oriented facade for the RAG application."""

    _instances: Dict[str, "RAGService"] = {}

    def __init__(self, session_id: str | None = None):
        setup_logger(level=Config.LOG_LEVEL)
        Config.ensure_directories()

        self.session_id = session_id or str(uuid.uuid4())
        self.processor = DocumentProcessor()
        self.deduplicator = DocumentDeduplicator()
        self.vector_store = VectorStore()
        self.hybrid_retriever: HybridRetriever | None = None
        self.retrieval_pipeline: RetrievalPipeline | None = None
        self.llm: LLMGenerator | None = None
        self._chat_history: List[Dict[str, str]] = []

        if self.vector_store.load_existing():
            self._initialize_retrievers()
            t_bm25 = time.perf_counter()
            self._rebuild_bm25_index()
            logger.info(
                "Restored existing collection (%d chunks, BM25 rebuild %.2fs)",
                self.vector_store.get_collection_stats(),
                time.perf_counter() - t_bm25,
            )

    @classmethod
    def get_instance(cls, session_id: str | None = None) -> "RAGService":
        """Singleton-style accessor (one service per session id)."""
        sid = session_id or "default"
        if sid not in cls._instances:
            cls._instances[sid] = cls(session_id=sid)
        return cls._instances[sid]

    def _initialize_retrievers(self) -> None:
        self.hybrid_retriever = HybridRetriever(self.vector_store)
        self.retrieval_pipeline = RetrievalPipeline(
            self.vector_store,
            hybrid_retriever=self.hybrid_retriever,
        )

    def _rebuild_bm25_index(self) -> None:
        if not self.hybrid_retriever:
            return
        docs = self.vector_store.get_all_documents()
        if docs:
            self.hybrid_retriever.index_documents(docs)
            logger.info("Rebuilt BM25 index from %d persisted chunks", len(docs))

    def _search_mode(self, query: str, use_hybrid: bool) -> str:
        if use_hybrid or (
            self.hybrid_retriever and should_use_hybrid(query)
        ):
            base = "hybrid"
        else:
            base = "vector"
        if Config.ENABLE_RERANKING:
            return f"{base}+rerank"
        return base

    def process_upload_paths(self, paths: List[Path]) -> dict:
        """Index PDF files already saved on disk."""
        new_paths, duplicate_paths = self.deduplicator.filter_new_files(paths)

        if not new_paths and duplicate_paths:
            return {
                "chunks_added": 0,
                "files_processed": 0,
                "duplicates_skipped": len(duplicate_paths),
                "total_chunks": self.vector_store.get_collection_stats(),
                "message": "All uploaded files were already indexed.",
            }

        self.vector_store.create_collection()
        all_chunks: List[Document] = []

        for path in new_paths:
            file_hash = compute_file_hash(path)
            chunks = self.processor.process_pdf(
                path,
                session_id=self.session_id,
                document_hash=file_hash,
                display_name=path.name,
            )
            if chunks:
                all_chunks.extend(chunks)
                self.deduplicator.register(
                    path,
                    original_name=path.name,
                    chunk_count=len(chunks),
                    session_id=self.session_id,
                )

        if all_chunks:
            self.vector_store.add_documents(all_chunks)
            self.hybrid_retriever = HybridRetriever(self.vector_store)
            self.hybrid_retriever.index_documents(all_chunks)
            self.retrieval_pipeline = RetrievalPipeline(
                self.vector_store,
                hybrid_retriever=self.hybrid_retriever,
            )

        return {
            "chunks_added": len(all_chunks),
            "files_processed": len(new_paths),
            "duplicates_skipped": len(duplicate_paths),
            "total_chunks": self.vector_store.get_collection_stats(),
        }

    def process_upload_bytes(
        self,
        files: List[Tuple[str, bytes]],
    ) -> dict:
        """
        Save and index uploaded PDFs from API multipart uploads.

        Args:
            files: List of (filename, raw_bytes) tuples
        """
        saved: List[Path] = []
        for filename, data in files:
            dest = safe_upload_path(filename)
            dest.write_bytes(data)
            saved.append(dest)
        return self.process_upload_paths(saved)

    def process_uploads(self, uploaded_files: list) -> dict:
        """Process Streamlit UploadedFile objects."""
        saved_paths: List[Path] = []
        for uploaded in uploaded_files:
            dest = safe_upload_path(uploaded.name)
            dest.write_bytes(uploaded.getbuffer())
            saved_paths.append(dest)
        return self.process_upload_paths(saved_paths)

    def initialize_llm(self, force: bool = False) -> LLMGenerator:
        expected_model = (
            Config.GEMINI_MODEL
            if Config.LLM_PROVIDER == "gemini"
            else Config.OPENAI_MODEL
        )
        stale = (
            self.llm is not None
            and getattr(self.llm, "_gemini_model_name", None)
            in {"gemini-1.5-pro", "gemini-1.5-pro-latest"}
        )
        if force or self.llm is None or stale:
            Config.validate()
            self.llm = LLMGenerator()
            logger.info(
                "LLM ready (provider=%s, model=%s)",
                Config.LLM_PROVIDER,
                expected_model,
            )
        return self.llm

    def ask(
        self,
        question: str,
        *,
        use_hybrid: bool = False,
        source_filter: str | None = None,
        stream: bool = False,
        remember: bool = True,
    ) -> RAGResponse:
        """
        Run full RAG query and return a structured API-friendly response.

        Example:
            response = rag_service.ask("What is RAG?")
            print(response.answer, response.citations)
        """
        t0 = time.perf_counter()
        answer, results, stream_gen = self.answer(
            question,
            use_hybrid=use_hybrid,
            source_filter=source_filter,
            stream=stream,
        )
        chunk_dicts = chunks_from_results(results)
        citations = unique_citations(chunk_dicts)
        latency_ms = (time.perf_counter() - t0) * 1000

        if remember and answer:
            self._append_history(question, answer)

        return RAGResponse(
            answer=answer,
            chunks=chunk_dicts,
            citations=citations,
            latency_ms=round(latency_ms, 2),
            search_mode=self._search_mode(question, use_hybrid),
            stream_generator=stream_gen,
            conversation_id=self.session_id,
        )

    def get_conversation_history(self) -> List[Dict[str, str]]:
        return list(self._chat_history)

    def clear_conversation(self) -> None:
        self._chat_history.clear()

    def _append_history(self, question: str, answer: str) -> None:
        self._chat_history.append({"role": "user", "content": question})
        self._chat_history.append({"role": "assistant", "content": answer})
        # Keep last N user+assistant pairs
        max_messages = _MAX_HISTORY_TURNS * 2
        if len(self._chat_history) > max_messages:
            self._chat_history = self._chat_history[-max_messages:]

    def answer(
        self,
        question: str,
        *,
        use_hybrid: bool = False,
        source_filter: str | None = None,
        stream: bool = False,
    ) -> Tuple[Optional[str], List[Tuple[Document, float]], Generator[str, None, None] | None]:
        """Run full RAG query (original interface preserved)."""
        if self.retrieval_pipeline is None:
            raise ValueError("No documents indexed. Upload and process PDFs first.")

        filter_metadata: dict = {}
        if source_filter:
            filter_metadata["source"] = source_filter

        t0 = time.perf_counter()
        results = self.retrieval_pipeline.retrieve(
            question,
            use_hybrid=use_hybrid,
            filter_metadata=filter_metadata or None,
        )
        context_chunks, metadata, packed = self.retrieval_pipeline.build_context(results)
        logger.info(
            "Retrieval+context: %d chunks in %.2fs",
            len(packed),
            time.perf_counter() - t0,
        )

        stream_gen = None
        answer = None

        if self.llm:
            t_llm = time.perf_counter()
            if stream:
                stream_gen = self.llm.generate_answer_stream(
                    question, context_chunks, metadata
                )
                logger.info("LLM stream started in %.2fs", time.perf_counter() - t_llm)
            else:
                answer = self.llm.generate_answer(question, context_chunks, metadata)
                logger.info("LLM answer in %.2fs", time.perf_counter() - t_llm)

        logger.info("answer() total: %.2fs", time.perf_counter() - t0)
        return answer, packed, stream_gen

    def reset(self) -> None:
        """Reset vector store and deduplication registry."""
        if self.vector_store:
            self.vector_store.delete_collection()
        self.deduplicator.clear()
        self.hybrid_retriever = None
        self.retrieval_pipeline = None
        self._chat_history.clear()
        RAGService._instances.pop(self.session_id, None)
        logger.info("RAG service reset for session %s", self.session_id)

    def stats(self) -> dict:
        """Database / index statistics for health and UI."""
        exists = self.vector_store.collection_exists()
        return {
            "session_id": self.session_id,
            "collection_exists": exists,
            "chunk_count": self.vector_store.get_collection_stats() if exists else 0,
            "llm_configured": bool(
                Config.GEMINI_API_KEY or Config.OPENAI_API_KEY
            ),
            "reranking_enabled": Config.ENABLE_RERANKING,
        }
