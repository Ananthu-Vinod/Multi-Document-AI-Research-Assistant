"""
Test script for Phase 6: Hybrid Retrieval
Run this to verify hybrid retrieval (BM25 + vector search) works correctly.
"""

from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from langchain_core.documents import Document


def test_bm25_retriever():
    """Test BM25 keyword-based retrieval."""
    print("=" * 60)
    print("PHASE 6 TEST: BM25 Retrieval")
    print("=" * 60)
    
    try:
        # Create sample documents
        docs = [
            Document(
                page_content="Python is a high-level programming language created by Guido van Rossum.",
                metadata={"source": "python.pdf", "page": 1}
            ),
            Document(
                page_content="Machine learning algorithms enable computers to learn from data patterns.",
                metadata={"source": "ml.pdf", "page": 1}
            ),
            Document(
                page_content="Vector databases store embeddings for semantic search applications.",
                metadata={"source": "vector_db.pdf", "page": 1}
            ),
            Document(
                page_content="Python supports multiple programming paradigms including object-oriented programming.",
                metadata={"source": "python.pdf", "page": 2}
            ),
            Document(
                page_content="Deep learning is a subset of machine learning using neural networks.",
                metadata={"source": "ml.pdf", "page": 2}
            )
        ]
        
        # Initialize BM25 retriever
        print("\n🔄 Initializing BM25 retriever...")
        retriever = BM25Retriever()
        
        # Index documents
        retriever.index_documents(docs)
        
        # Test queries
        test_queries = [
            "Python programming language",
            "machine learning algorithms",
            "vector database"
        ]
        
        print(f"\n🔍 Testing {len(test_queries)} queries...")
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = retriever.search(query, k=2)
            
            print(f"Top {len(results)} results:")
            for i, (doc, score) in enumerate(results, 1):
                print(f"  {i}. Score: {score:.4f}")
                print(f"     Content: {doc.page_content[:70]}...")
                print(f"     Source: {doc.metadata['source']}")
        
        print("\n✓ BM25 retrieval test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_retriever():
    """Test hybrid retrieval (requires vector store)."""
    print("\n" + "=" * 60)
    print("PHASE 6 TEST: Hybrid Retrieval")
    print("=" * 60)
    
    try:
        from vector_store import VectorStore
        
        # Create sample documents
        docs = [
            Document(
                page_content="Python is a high-level programming language created by Guido van Rossum.",
                metadata={"source": "test.pdf", "page": 1}
            ),
            Document(
                page_content="Machine learning algorithms enable computers to learn from data patterns.",
                metadata={"source": "test.pdf", "page": 2}
            ),
            Document(
                page_content="Vector databases store embeddings for semantic search applications.",
                metadata={"source": "test.pdf", "page": 3}
            )
        ]
        
        # Initialize vector store
        print("\n🔄 Initializing vector store...")
        vector_store = VectorStore()
        vector_store.create_collection("test_hybrid")
        vector_store.add_documents(docs)
        
        # Initialize hybrid retriever
        print("\n🔄 Initializing hybrid retriever...")
        hybrid = HybridRetriever(vector_store, alpha=0.5)
        hybrid.index_documents(docs)
        
        # Test query
        query = "Python programming"
        print(f"\n🔍 Testing hybrid search for: '{query}'")
        
        results = hybrid.search(query, k=2)
        
        print(f"\nTop {len(results)} hybrid results:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"  {i}. Combined Score: {score:.4f}")
            print(f"     Content: {doc.page_content[:70]}...")
        
        # Clean up
        vector_store.delete_collection()
        
        print("\n✓ Hybrid retrieval test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🧪 Testing Phase 6: Hybrid Retrieval...\n")
    
    # Test BM25 (doesn't require vector store)
    bm25_test = test_bm25_retriever()
    
    # Test Hybrid (requires vector store)
    hybrid_test = test_hybrid_retriever()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"BM25 Retrieval: {'✓ PASS' if bm25_test else '✗ FAIL'}")
    print(f"Hybrid Retrieval: {'✓ PASS' if hybrid_test else '✗ FAIL'}")
    print("=" * 60)
