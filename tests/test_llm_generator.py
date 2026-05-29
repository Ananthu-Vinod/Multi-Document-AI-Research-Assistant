"""
Test script for Phase 4: LLM Integration
Run this to verify LLM answer generation works correctly.
"""

from llm.generator import LLMGenerator


def test_llm_generator():
    """Test LLM answer generation."""
    print("=" * 60)
    print("PHASE 4 TEST: LLM Answer Generation")
    print("=" * 60)
    
    try:
        # Initialize LLM generator
        print("\n🔄 Initializing LLM generator...")
        generator = LLMGenerator()
        
        # Sample context (simulating retrieved chunks)
        sample_context = [
            "Python is a high-level programming language created by Guido van Rossum in 1991. It was designed with an emphasis on code readability.",
            "Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
            "The language's name is inspired by Monty Python's Flying Circus, not the snake."
        ]
        
        # Sample metadata for citations
        sample_metadata = [
            {"source": "python_intro.pdf", "page": 1, "chunk_id": 0},
            {"source": "python_intro.pdf", "page": 2, "chunk_id": 1},
            {"source": "python_intro.pdf", "page": 3, "chunk_id": 2}
        ]
        
        # Test queries
        test_queries = [
            "Who created Python and when?",
            "What programming paradigms does Python support?",
            "Why is Python named after Monty Python?"
        ]
        
        print(f"\n📝 Testing with {len(test_queries)} queries...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*60}")
            print(f"Query {i}: {query}")
            print(f"{'='*60}")
            
            # Generate answer
            answer = generator.generate_answer(query, sample_context, sample_metadata)
            
            print(f"\nAnswer:\n{answer}\n")
        
        print("\n✓ All LLM tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🧪 Testing Phase 4: LLM Integration...\n")
    
    # Check if API key is configured
    from dotenv import load_dotenv
    load_dotenv()
    
    import os
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("⚠ No API key found in .env file")
        print("Please add your GEMINI_API_KEY or OPENAI_API_KEY to .env file")
        print("Gemini API is free: https://makersuite.google.com/app/apikey")
    else:
        success = test_llm_generator()
        
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"LLM Integration: {'✓ PASS' if success else '✗ FAIL'}")
        print("=" * 60)
