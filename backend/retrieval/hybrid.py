"""Hybrid retrieval combining BM25 and vector similarity."""

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from config import Config
from logger import get_logger
from retrieval.bm25 import BM25Retriever
from retrieval.scoring import (
    apply_similarity_threshold,
    distance_to_similarity,
    document_key,
    normalize_scores,
)

logger = get_logger(__name__)


class HybridRetriever:
    """Combines BM25 and vector search with corrected similarity scoring."""

    def __init__(self, vector_store: Any, alpha: float | None = None):
        self.vector_store = vector_store
        self.alpha = alpha if alpha is not None else Config.HYBRID_ALPHA
        self.bm25 = BM25Retriever()
        self.documents_indexed = False
        self._doc_lookup: Dict[str, Document] = {}

    def index_documents(self, documents: List[Document]) -> None:
        """Index documents for BM25 (vector store indexed separately)."""
        self.bm25.index_documents(documents)
        self._doc_lookup = {document_key(doc): doc for doc in documents}
        self.documents_indexed = True
        logger.info("Hybrid retriever indexed %d documents", len(documents))

    def search(
        self,
        query: str,
        k: int | None = None,
        *,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Hybrid search with score fusion on normalized similarities.

        Vector distances from Chroma are converted via 1/(1+distance).
        """
        if not self.documents_indexed:
            raise ValueError("Documents not indexed. Call index_documents() first.")

        k = k or Config.TOP_K_RESULTS
        candidate_k = min(
            k * Config.RETRIEVAL_CANDIDATE_MULTIPLIER,
            Config.MAX_RETRIEVAL_RESULTS,
        )

        bm25_results = self.bm25.search(query, k=candidate_k)
        vector_results = self.vector_store.similarity_search_with_scores(
            query,
            k=candidate_k,
            filter_metadata=filter_metadata,
            convert_distance=True,
        )

        return self._combine_scores(bm25_results, vector_results, k)

    def _combine_scores(
        self,
        bm25_results: List[Tuple[Document, float]],
        vector_results: List[Tuple[Document, float]],
        k: int,
    ) -> List[Tuple[Document, float]]:
        """Fuse BM25 and vector scores (both higher = better)."""
        bm25_scores = {document_key(doc): score for doc, score in bm25_results}
        vector_scores = {document_key(doc): score for doc, score in vector_results}

        bm25_norm = normalize_scores(bm25_scores)
        vector_norm = normalize_scores(vector_scores)

        combined: Dict[str, float] = {}
        all_keys = set(bm25_norm) | set(vector_norm)

        for key in all_keys:
            bm25_score = bm25_norm.get(key, 0.0)
            vector_score = vector_norm.get(key, 0.0)
            combined[key] = self.alpha * vector_score + (1.0 - self.alpha) * bm25_score

        ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)

        final: List[Tuple[Document, float]] = []
        for key, score in ranked:
            doc = self._doc_lookup.get(key)
            if doc is None:
                for candidate, _ in bm25_results + vector_results:
                    if document_key(candidate) == key:
                        doc = candidate
                        self._doc_lookup[key] = doc
                        break
            if doc is not None:
                final.append((doc, score))

        final = apply_similarity_threshold(
            final,
            Config.SIMILARITY_THRESHOLD,
            min_results=Config.MIN_RETRIEVAL_RESULTS,
        )
        return final[:k]
