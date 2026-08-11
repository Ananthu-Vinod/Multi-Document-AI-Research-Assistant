"""
Comprehensive RAG System Evaluation Script
Evaluates Retrieval, Generation, RAG-specific, and Performance metrics.
"""

import json
import os
import sys
import time
from pathlib import Path

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from config import Config
from services.rag_service import RAGService

def run_evaluation():
    service = RAGService.get_instance("eval-suite")
    try:
        service.initialize_llm(force=True)
    except Exception as e:
        print("LLM Init notice:", e)

    test_cases = [
        {
            "id": "TC-01",
            "category": "Direct Factual Query",
            "query": "What is parsing and what is the main difference between top-down and bottom-up parsing?",
            "expected": "Parsing is analyzing a sentence structure according to a grammar. Top-down starts from the start symbol (S) and rewrites to words. Bottom-up starts from words and builds up to the start symbol.",
            "test_type": "factual"
        },
        {
            "id": "TC-02",
            "category": "No-Answer / Out-of-Scope",
            "query": "What is the warranty policy for quantum gravity propulsion engines in spacecraft?",
            "expected": "The system should acknowledge that no information about quantum gravity propulsion engines exists in the document collection.",
            "test_type": "no_answer"
        },
        {
            "id": "TC-03",
            "category": "Typo & Paraphrase Robustness",
            "query": "whats botom up chart parsen and how does arc extension work?",
            "expected": "Bottom-up chart parsing builds completed constituents from words and extends active arcs using an agenda.",
            "test_type": "robustness"
        },
        {
            "id": "TC-04",
            "category": "Scenario / Rubric Mapping",
            "query": "If a student performed an experiment on schedule but submitted one week late, what score scale applies on the rubric?",
            "expected": "According to the rubric, submitting one week late receives a scale score of 4 out of 5 for regularity and punctuality.",
            "test_type": "scenario"
        },
        {
            "id": "TC-05",
            "category": "Acronym & Keyword Fusion",
            "query": "What is Finite State Transducer used for in morphological processing?",
            "expected": "Finite State Transducers (FSTs) are processing mechanisms introduced primarily for morphological parsing and word form processing.",
            "test_type": "keyword"
        }
    ]

    results = []

    print("=== STARTING COMPREHENSIVE RAG EVALUATION BENCHMARK ===\n")

    for tc in test_cases:
        print(f"Executing {tc['id']}: {tc['query']} ...")
        t_start = time.perf_counter()
        
        # Measure retrieval time
        t_ret_0 = time.perf_counter()
        if service.retrieval_pipeline:
            results_ret = service.retrieval_pipeline.retrieve(tc["query"], use_hybrid=True)
            context_chunks, metadata, packed = service.retrieval_pipeline.build_context(results_ret)
        else:
            packed = []
            context_chunks = []
        t_ret_ms = (time.perf_counter() - t_ret_0) * 1000

        # Measure end-to-end RAG response
        rag_resp = service.ask(tc["query"], use_hybrid=True, stream=False, remember=False)
        t_e2e_ms = (time.perf_counter() - t_start) * 1000

        res_entry = {
            "id": tc["id"],
            "category": tc["category"],
            "query": tc["query"],
            "expected": tc["expected"],
            "generated_answer": rag_resp.answer,
            "citations": rag_resp.citations,
            "chunks_retrieved": len(rag_resp.chunks),
            "retrieval_latency_ms": round(t_ret_ms, 2),
            "e2e_latency_ms": round(t_e2e_ms, 2),
            "search_mode": rag_resp.search_mode,
            "context_previews": [c["preview"] for c in rag_resp.chunks[:3]]
        }
        results.append(res_entry)
        print(f" -> Completed in {t_e2e_ms:.2f} ms (Retrieval: {t_ret_ms:.2f} ms)\n")

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Evaluation benchmark finished. Saved to eval_results.json")

if __name__ == "__main__":
    run_evaluation()
