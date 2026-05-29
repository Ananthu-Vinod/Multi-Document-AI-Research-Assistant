"""
RAG Application Package
Production-grade Retrieval-Augmented Generation system for document intelligence.
"""

from .config import Config
from .document_processor import DocumentProcessor
from .embeddings import EmbeddingGenerator, get_embedding_generator
from .evaluation import (
    answer_relevancy_score,
    evaluate_retrieval,
    faithfulness_score,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from .hybrid_retriever import BM25Retriever, HybridRetriever, distance_to_similarity
from .llm_generator import LLMGenerator
from .logger import get_logger, setup_logger
from .pipeline import RAGService
from .vector_store import SentenceTransformerEmbeddings, VectorStore

__all__ = [
    "Config",
    "DocumentProcessor",
    "EmbeddingGenerator",
    "get_embedding_generator",
    "VectorStore",
    "SentenceTransformerEmbeddings",
    "LLMGenerator",
    "BM25Retriever",
    "HybridRetriever",
    "distance_to_similarity",
    "RAGService",
    "setup_logger",
    "get_logger",
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "faithfulness_score",
    "answer_relevancy_score",
    "evaluate_retrieval",
]
