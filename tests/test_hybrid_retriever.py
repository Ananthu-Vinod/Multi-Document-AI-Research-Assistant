"""Test script for Phase 6: Hybrid Retrieval."""

from langchain_core.documents import Document
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from retrieval.vector_store import VectorStore


def test_bm25_retriever():
    """Test BM25 keyword-based retrieval."""
    docs = [
        Document(
            page_content="Python is a high-level programming language created by Guido van Rossum.",
            metadata={"source": "python.pdf", "page": 1}
        ),
        Document(
            page_content="Machine learning algorithms enable computers to learn from data patterns.",
            metadata={"source": "ml.pdf", "page": 1}
        ),
    ]
    retriever = BM25Retriever()
    retriever.index_documents(docs)
    results = retriever.search("Python programming", k=1)
    assert len(results) > 0
    assert "Python" in results[0][0].page_content


def test_hybrid_retriever():
    """Test hybrid retrieval (requires vector store)."""
    docs = [
        Document(
            page_content="Python is a high-level programming language created by Guido van Rossum.",
            metadata={"source": "test.pdf", "page": 1}
        ),
        Document(
            page_content="Machine learning algorithms enable computers to learn from data patterns.",
            metadata={"source": "test.pdf", "page": 2}
        )
    ]
    vector_store = VectorStore()
    vector_store.create_collection("test_hybrid")
    try:
        vector_store.add_documents(docs)
        hybrid = HybridRetriever(vector_store, alpha=0.5)
        hybrid.index_documents(docs)
        results = hybrid.search("Python programming", k=1)
        assert len(results) > 0
    finally:
        vector_store.delete_collection()
