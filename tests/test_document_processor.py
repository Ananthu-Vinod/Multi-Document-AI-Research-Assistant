"""Test script for PDF Ingestion and Chunking."""

import os
import pytest

from embeddings.generator import EmbeddingGenerator
from ingestion.processor import DocumentProcessor


def test_document_processing():
    """Test PDF loading and chunking."""
    pdf_path = "uploads/test.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip("No test PDF found at uploads/test.pdf (skipping PDF load test in headless CI)")

    processor = DocumentProcessor()
    chunks = processor.process_pdf(pdf_path)
    assert len(chunks) > 0


def test_embedding_generation():
    """Test embedding generation."""
    generator = EmbeddingGenerator()
    test_texts = [
        "This is a test sentence for embedding generation.",
        "RAG systems combine retrieval with generation for better answers."
    ]
    embeddings = generator.generate_embeddings(test_texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0
