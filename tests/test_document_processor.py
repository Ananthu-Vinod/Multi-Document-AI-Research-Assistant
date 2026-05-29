"""
Test script for Phase 2: PDF Ingestion and Chunking
Run this to verify document processing works correctly.
"""

import os

from embeddings.generator import EmbeddingGenerator
from ingestion.processor import DocumentProcessor


def test_document_processing():
    """Test PDF loading and chunking."""
    print("=" * 60)
    print("PHASE 2 TEST: Document Processing")
    print("=" * 60)
    
    # Check if we have a test PDF
    pdf_path = "uploads/test.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"\n⚠ No test PDF found at {pdf_path}")
        print("Please add a PDF file to the uploads/ folder and update the path.")
        print("\nYou can download a sample PDF from:")
        print("https://www.africau.edu/images/default/sample.pdf")
        return False
    
    try:
        # Initialize processor
        processor = DocumentProcessor()
        
        # Process PDF
        chunks = processor.process_pdf(pdf_path)
        
        print(f"\n✓ Successfully processed PDF")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  First chunk preview: {chunks[0].page_content[:150]}...")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        return False


def test_embedding_generation():
    """Test embedding generation."""
    print("\n" + "=" * 60)
    print("PHASE 2 TEST: Embedding Generation")
    print("=" * 60)
    
    try:
        # Initialize embedding generator
        generator = EmbeddingGenerator()
        
        # Test with sample text
        test_texts = [
            "This is a test sentence for embedding generation.",
            "RAG systems combine retrieval with generation for better answers."
        ]
        
        embeddings = generator.generate_embeddings(test_texts)
        
        print(f"\n✓ Successfully generated embeddings")
        print(f"  Number of embeddings: {len(embeddings)}")
        print(f"  Embedding dimension: {len(embeddings[0])}")
        print(f"  Sample embedding values (first 5): {embeddings[0][:5]}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        return False


if __name__ == "__main__":
    print("\n🧪 Testing Phase 2 Components...\n")
    
    # Test embedding generation (doesn't require PDF)
    embedding_test = test_embedding_generation()
    
    # Test document processing (requires PDF)
    doc_test = test_document_processing()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Embedding Generation: {'✓ PASS' if embedding_test else '✗ FAIL'}")
    print(f"Document Processing: {'✓ PASS' if doc_test else '✗ FAIL (needs PDF)'}")
    print("=" * 60)
