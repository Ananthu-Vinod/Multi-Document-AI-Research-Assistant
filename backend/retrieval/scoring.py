"""Score conversion utilities for retrieval."""

from typing import Dict, List, Tuple

from langchain_core.documents import Document


def distance_to_similarity(distance: float) -> float:
    """
    Convert Chroma/LangChain distance to similarity (higher = better).

    Chroma returns distance where lower is more similar.
    """
    return 1.0 / (1.0 + max(distance, 0.0))


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize scores to [0, 1]."""
    if not scores:
        return {}
    values = list(scores.values())
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return {key: 0.5 for key in scores}
    return {key: (value - min_val) / (max_val - min_val) for key, value in scores.items()}


def document_key(doc: Document) -> str:
    """Stable key for score fusion across retrievers."""
    chunk_id = doc.metadata.get("chunk_id", "")
    source = doc.metadata.get("source", "")
    return f"{source}::{chunk_id}::{hash(doc.page_content)}"


def apply_similarity_threshold(
    results: List[Tuple[Document, float]],
    threshold: float,
    min_results: int = 1,
) -> List[Tuple[Document, float]]:
    """Filter results below similarity threshold while keeping minimum count."""
    filtered = [(doc, score) for doc, score in results if score >= threshold]
    if len(filtered) >= min_results:
        return filtered
    return results[:max(min_results, len(results))]
