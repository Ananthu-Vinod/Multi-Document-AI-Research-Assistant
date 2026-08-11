"""Test script for LLM Integration."""

import os
import pytest
from llm.generator import LLMGenerator


def test_llm_generator():
    """Test LLM answer generation."""
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        pytest.skip("No GEMINI_API_KEY or OPENAI_API_KEY set in environment for live LLM test.")

    generator = LLMGenerator()
    sample_context = [
        "Python is a high-level programming language created by Guido van Rossum in 1991.",
    ]
    sample_metadata = [
        {"source": "python_intro.pdf", "page": 1, "chunk_id": 0},
    ]
    query = "Who created Python?"
    answer = generator.generate_answer(query, sample_context, sample_metadata)
    assert answer is not None
    assert len(answer) > 0
