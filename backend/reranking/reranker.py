"""Cross-encoder reranking for improved retrieval precision."""

import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from config import Config
from logger import get_logger

logger = get_logger(__name__)

_model_lock = threading.Lock()
_reranker_instance = None


def _get_cross_encoder():
    """Lazy-load cross-encoder model singleton."""
    global _reranker_instance
    if _reranker_instance is None:
        with _model_lock:
            if _reranker_instance is None:
                from sentence_transformers import CrossEncoder

                logger.info("Loading reranker model: %s", Config.RERANKER_MODEL)
                _reranker_instance = CrossEncoder(Config.RERANKER_MODEL)
    return _reranker_instance


class CrossEncoderReranker:
    """Reranks retrieved documents using a cross-encoder model."""

    def __init__(self, enabled: bool | None = None):
        self.enabled = Config.ENABLE_RERANKING if enabled is None else enabled
        self._model = None

    @property
    def model(self):
        if self._model is None and self.enabled:
            self._model = _get_cross_encoder()
        return self._model

    def rerank(
        self,
        query: str,
        results: List[Tuple[Document, float]],
        top_k: int | None = None,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents by cross-encoder relevance scores.

        Args:
            query: User query
            results: Initial retrieval results
            top_k: Number of results to return after reranking

        Returns:
            Reranked (Document, score) tuples (higher = better)
        """
        if not self.enabled or not results:
            return results

        top_k = top_k or Config.RERANK_TOP_K

        try:
            pairs = [[query, doc.page_content] for doc, _ in results]
            scores = self._predict_with_timeout(pairs)

            reranked = list(zip([doc for doc, _ in results], scores))
            reranked.sort(key=lambda item: item[1], reverse=True)

            output = [(doc, float(score)) for doc, score in reranked[:top_k]]
            logger.info("Reranked %d → %d chunks", len(results), len(output))
            return output

        except FuturesTimeoutError:
            logger.warning(
                "Reranking timed out after %.0fs, using original order",
                Config.RERANK_TIMEOUT_SECONDS,
            )
            return results[:top_k]
        except Exception as exc:
            logger.warning("Reranking failed, using original order: %s", exc)
            return results[:top_k]

    def _predict_with_timeout(self, pairs: list) -> list:
        """Run cross-encoder predict with a wall-clock timeout."""
        timeout = Config.RERANK_TIMEOUT_SECONDS
        if timeout <= 0:
            return self.model.predict(pairs)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.model.predict, pairs)
            return future.result(timeout=timeout)
