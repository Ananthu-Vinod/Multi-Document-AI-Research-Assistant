"""Singleton embedding model for memory-efficient reuse."""

import threading
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from config import Config
from logger import get_logger

logger = get_logger(__name__)

_model_lock = threading.Lock()
_model_instance: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """
    Return a process-wide singleton SentenceTransformer instance.

    Prevents reloading the model on every VectorStore / embedding call.
    """
    global _model_instance
    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:
                logger.info("Loading embedding model: %s", Config.EMBEDDING_MODEL)
                _model_instance = SentenceTransformer(Config.EMBEDDING_MODEL)
                dim = getattr(
                    _model_instance,
                    "get_embedding_dimension",
                    _model_instance.get_sentence_embedding_dimension,
                )()
                logger.info("Embedding model ready (dim=%d)", dim)
    return _model_instance


class EmbeddingGenerator:
    """Generates embeddings using the shared singleton model."""

    def __init__(self):
        self.model = get_embedding_model()

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        if not texts:
            return []
        try:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=len(texts) > 10,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings.tolist()
        except Exception as exc:
            logger.error("Batch embedding failed: %s", exc)
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc

    def generate_single_embedding(self, text: str) -> List[float]:
        """Embed a single query string."""
        try:
            embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as exc:
            logger.error("Query embedding failed: %s", exc)
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc


def get_embedding_generator() -> EmbeddingGenerator:
    """Factory for a lightweight wrapper around the singleton model."""
    return EmbeddingGenerator()
