"""End-to-end retrieval pipeline: retrieve → rerank → budget."""

import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from config import Config
from logger import get_logger
from retrieval.hybrid import HybridRetriever
from retrieval.query_expansion import (
    build_alias_map,
    expand_query,
    should_use_hybrid,
)
from retrieval.scoring import apply_similarity_threshold
from retrieval.vector_store import VectorStore
from reranking.reranker import CrossEncoderReranker
from utils.tokens import pack_chunks_by_budget

logger = get_logger(__name__)


class RetrievalPipeline:
    """
    Production retrieval orchestrator.

    Pipeline: Retriever → (optional) Reranker → context budget packing
    """

    def __init__(
        self,
        vector_store: VectorStore,
        hybrid_retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
    ):
        self.vector_store = vector_store
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker or (
            CrossEncoderReranker() if Config.ENABLE_RERANKING else None
        )
        self._alias_map: dict | None = None

    def _get_alias_map(self) -> dict:
        if self._alias_map is None:
            docs = self.vector_store.get_all_documents()
            self._alias_map = build_alias_map(docs)
        return self._alias_map

    def retrieve(
        self,
        query: str,
        *,
        use_hybrid: bool = False,
        k: int | None = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant chunks with dynamic candidate pool and thresholding.
        """
        t0 = time.perf_counter()
        k = k or Config.TOP_K_RESULTS
        candidate_k = min(
            max(k * Config.RETRIEVAL_CANDIDATE_MULTIPLIER, k),
            Config.MAX_RETRIEVAL_RESULTS,
        )

        search_query = expand_query(query, self._get_alias_map())
        if search_query != query:
            logger.info("Query expanded: %r -> %r", query, search_query)

        use_hybrid = use_hybrid or (
            self.hybrid_retriever is not None and should_use_hybrid(query)
        )

        mode = "hybrid" if use_hybrid and self.hybrid_retriever else "vector"
        if use_hybrid and self.hybrid_retriever:
            results = self.hybrid_retriever.search(
                search_query, k=candidate_k, filter_metadata=filter_metadata
            )
        else:
            results = self.vector_store.similarity_search_with_scores(
                search_query,
                k=candidate_k,
                filter_metadata=filter_metadata,
                convert_distance=True,
            )
            results = apply_similarity_threshold(
                results,
                Config.SIMILARITY_THRESHOLD,
                min_results=Config.MIN_RETRIEVAL_RESULTS,
            )
        logger.info(
            "%s search: %d candidates in %.2fs",
            mode,
            len(results),
            time.perf_counter() - t0,
        )

        if self.reranker and results:
            t_rerank = time.perf_counter()
            results = self.reranker.rerank(query, results, top_k=Config.RERANK_TOP_K)
            logger.info(
                "Reranking: %d results in %.2fs",
                len(results),
                time.perf_counter() - t_rerank,
            )

        logger.info("Retrieve total: %.2fs", time.perf_counter() - t0)
        return results[:k]

    def build_context(
        self,
        results: List[Tuple[Document, float]],
    ) -> Tuple[List[str], List[Dict], List[Tuple[Document, float]]]:
        """
        Apply token/character budgeting to retrieved chunks.

        Returns:
            (context_texts, metadata_list, packed_results)
        """
        ranked = [
            (doc.page_content, score, doc.metadata)
            for doc, score in sorted(results, key=lambda x: x[1], reverse=True)
        ]
        packed = pack_chunks_by_budget(ranked)

        context_chunks = [item[0] for item in packed]
        metadata = [item[2] for item in packed]
        packed_docs = []
        for content, score, meta in packed:
            packed_docs.append((Document(page_content=content, metadata=meta), score))

        logger.info(
            "Context packed: %d/%d chunks (chars<=%d)",
            len(packed),
            len(results),
            Config.MAX_CONTEXT_CHARS,
        )
        return context_chunks, metadata, packed_docs
