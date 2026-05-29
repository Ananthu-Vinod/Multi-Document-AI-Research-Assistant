"""
Test script for Phase 3: ChromaDB Vector Storage
Run this to verify vector database operations work correctly.
"""

from retrieval.vector_store import VectorStore
from langchain_core.documents import Document


def test_vector_store():
    """Test vector store operations."""
    print("=" * 60)
    print("PHASE 3 TEST: Vector Database Operations")
    print("=" * 60)
    
    try:
        # Initialize vector store
        print("\n🔄 Initializing vector store...")
        vector_store = VectorStore()
        
        # Create collection
        print("\n🔄 Creating collection...")
        vector_store.create_collection("test_collection")
        
        # Create sample documents
        sample_docs = [
            Document(
                page_content="Python is a high-level programming language known for its simplicity and readability.",
                metadata={"source": "test_doc.pdf", "page": 1, "chunk_id": 0}
            ),
            Document(
                page_content="Machine learning algorithms enable computers to learn from data and make predictions.",
                metadata={"source": "test_doc.pdf", "page": 2, "chunk_id": 1}
            ),
            Document(
                page_content="Vector databases store embeddings for fast semantic search and retrieval.",
                metadata={"source": "test_doc.pdf", "page": 3, "chunk_id": 2}
            ),
            Document(
                page_content="RAG combines retrieval systems with generative AI for accurate, context-aware responses.",
                metadata={"source": "test_doc.pdf", "page": 4, "chunk_id": 3}
            )
        ]
        
        # Add documents
        print(f"\n🔄 Adding {len(sample_docs)} documents to vector store...")
        vector_store.add_documents(sample_docs)
        
        # Get collection stats
        print("\n📊 Collection Statistics:")
        vector_store.get_collection_stats()
        
        # Test similarity search
        print("\n🔍 Testing similarity search...")
        query = "What is Python?"
        print(f"Query: '{query}'")
        results = vector_store.similarity_search(query, k=2)
        
        print(f"\nTop {len(results)} results:")
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. Content: {doc.page_content[:100]}...")
            print(f"   Source: {doc.metadata.get('source', 'N/A')}")
            print(f"   Page: {doc.metadata.get('page', 'N/A')}")
        
        # Test search with scores
        print("\n🔍 Testing similarity search with scores...")
        results_with_scores = vector_store.similarity_search_with_scores(query, k=2)
        
        print(f"\nTop {len(results_with_scores)} results with scores:")
        for i, (doc, score) in enumerate(results_with_scores, 1):
            print(f"\n{i}. Score: {score:.4f}")
            print(f"   Content: {doc.page_content[:100]}...")
        
        # Clean up test collection
        print("\n🧹 Cleaning up test collection...")
        vector_store.delete_collection()
        
        print("\n✓ All vector store tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🧪 Testing Phase 3: Vector Database...\n")
    success = test_vector_store()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Vector Store Operations: {'✓ PASS' if success else '✗ FAIL'}")
    print("=" * 60)
