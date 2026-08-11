"""Test script for Phase 3: ChromaDB Vector Storage."""

from langchain_core.documents import Document
from retrieval.vector_store import VectorStore


def test_vector_store():
    """Test vector store operations."""
    vector_store = VectorStore()
    vector_store.create_collection("test_collection")
    try:
        sample_docs = [
            Document(
                page_content="Python is a high-level programming language known for simplicity.",
                metadata={"source": "test_doc.pdf", "page": 1, "chunk_id": 0}
            ),
            Document(
                page_content="Machine learning algorithms enable computers to learn from data.",
                metadata={"source": "test_doc.pdf", "page": 2, "chunk_id": 1}
            )
        ]
        vector_store.add_documents(sample_docs)
        results = vector_store.similarity_search("What is Python?", k=1)
        assert len(results) > 0
        assert "Python" in results[0].page_content
    finally:
        vector_store.delete_collection()
