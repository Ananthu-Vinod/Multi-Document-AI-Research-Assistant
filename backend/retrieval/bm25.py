"""BM25 keyword retriever with correct document length normalization."""

import math
import re
from collections import Counter
from typing import List, Tuple

from langchain_core.documents import Document

from config import Config
from logger import get_logger

logger = get_logger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)


class BM25Retriever:
    """BM25 keyword-based retriever for hybrid search."""

    def __init__(self, k1: float | None = None, b: float | None = None):
        self.k1 = k1 if k1 is not None else Config.BM25_K1
        self.b = b if b is not None else Config.BM25_B
        self.documents: List[Document] = []
        self.doc_freqs: List[Counter] = []
        self.doc_lengths: List[int] = []
        self.idf: dict[str, float] = {}
        self.avg_doc_len: float = 0.0

    def index_documents(self, documents: List[Document]) -> None:
        """Index documents for BM25 retrieval."""
        self.documents = documents
        self.doc_freqs = []
        self.doc_lengths = []

        for doc in documents:
            tokens = self._tokenize(doc.page_content)
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
            # Total token count (not unique vocabulary size)
            self.doc_lengths.append(sum(freq.values()))

        total_tokens = sum(self.doc_lengths)
        self.avg_doc_len = total_tokens / len(self.doc_lengths) if self.doc_lengths else 0.0
        self._calculate_idf()
        logger.info("BM25 indexed %d documents (avg_len=%.1f)", len(documents), self.avg_doc_len)

    def _tokenize(self, text: str) -> List[str]:
        """Robust lowercase tokenization preserving alphanumeric tokens."""
        return _TOKEN_PATTERN.findall(text.lower())

    def _calculate_idf(self) -> None:
        """Calculate smoothed IDF for all terms."""
        n_docs = len(self.documents)
        all_terms: set[str] = set()
        for freq in self.doc_freqs:
            all_terms.update(freq.keys())

        self.idf = {}
        for term in all_terms:
            doc_count = sum(1 for freq in self.doc_freqs if term in freq)
            self.idf[term] = math.log((n_docs - doc_count + 0.5) / (doc_count + 0.5) + 1.0)

    def search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Search using BM25; higher scores are better."""
        if not self.documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: List[Tuple[Document, float]] = []
        for i, doc_freq in enumerate(self.doc_freqs):
            doc_len = self.doc_lengths[i]
            score = 0.0

            for token in query_tokens:
                if token not in doc_freq:
                    continue
                tf = doc_freq[token]
                idf = self.idf.get(token, 0.0)
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * doc_len / max(self.avg_doc_len, 1.0)
                )
                score += idf * (numerator / denominator)

            scores.append((self.documents[i], score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:k]
