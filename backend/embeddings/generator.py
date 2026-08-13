"""
Embedding generator supporting Google Gemini API Embeddings (Zero Local Memory, ~75MB RAM)
and SentenceTransformers fallback.
"""

import os
import threading
from typing import List, Optional

from config import Config
from logger import get_logger

logger = get_logger(__name__)

_model_lock = threading.Lock()
_local_model_instance = None


class EmbeddingGenerator:
    """Generates embeddings using Gemini API (75MB RAM) or local SentenceTransformers fallback."""

    def __init__(self):
        self.api_key = Config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        self.use_api = Config.LLM_PROVIDER == "gemini" and bool(self.api_key)
        if self.use_api:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            logger.info("Using Gemini API Embeddings (models/gemini-embedding-001, ~75MB RAM)")
        else:
            logger.info("Using local SentenceTransformers embedding model: %s", Config.EMBEDDING_MODEL)

    def _get_local_model(self):
        global _local_model_instance
        if _local_model_instance is None:
            with _model_lock:
                if _local_model_instance is None:
                    from sentence_transformers import SentenceTransformer
                    _local_model_instance = SentenceTransformer(Config.EMBEDDING_MODEL)
        return _local_model_instance

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        if not texts:
            return []

        if self.use_api:
            try:
                import google.generativeai as genai
                res = genai.embed_content(
                    model="models/gemini-embedding-001",
                    content=texts,
                    task_type="retrieval_document"
                )
                embeddings = res["embedding"]
                if embeddings and isinstance(embeddings[0], float):
                    return [embeddings]
                return embeddings
            except Exception as exc:
                logger.warning("Gemini API batch embedding failed (%s); trying fallback", exc)

        model = self._get_local_model()
        embeddings = model.encode(
            texts,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def generate_single_embedding(self, text: str) -> List[float]:
        """Embed a single query string."""
        if not text:
            return []

        if self.use_api:
            try:
                import google.generativeai as genai
                res = genai.embed_content(
                    model="models/gemini-embedding-001",
                    content=text,
                    task_type="retrieval_query"
                )
                return res["embedding"]
            except Exception as exc:
                logger.warning("Gemini API single embedding failed (%s); trying fallback", exc)

        model = self._get_local_model()
        embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()


def get_embedding_generator() -> EmbeddingGenerator:
    """Factory for embedding generator."""
    return EmbeddingGenerator()
