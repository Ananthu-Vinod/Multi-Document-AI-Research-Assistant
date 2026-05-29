"""
Evaluation metrics for RAG retrieval and generation quality.

Suitable for offline benchmarking with labeled QA datasets.
"""

from typing import Dict, Iterable, List, Sequence, Set


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    """Fraction of top-k retrieved items that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    """Fraction of relevant items found in top-k."""
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(
    retrieved_lists: Iterable[Sequence[str]],
    relevant_sets: Iterable[Set[str]],
) -> float:
    """Mean reciprocal rank across multiple queries."""
    reciprocal_ranks: List[float] = []
    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        rr = 0.0
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def faithfulness_score(answer: str, context_chunks: Sequence[str]) -> float:
    """
    Lightweight lexical faithfulness proxy.

    Measures overlap between answer tokens and context tokens.
    For production, replace with LLM-as-judge or Ragas.
    """
    if not answer or not context_chunks:
        return 0.0

    answer_tokens = set(_tokenize(answer))
    context_tokens = set(_tokenize(" ".join(context_chunks)))
    if not answer_tokens:
        return 0.0

    overlap = answer_tokens & context_tokens
    return len(overlap) / len(answer_tokens)


def answer_relevancy_score(answer: str, question: str) -> float:
    """Lightweight lexical relevancy proxy between answer and question."""
    if not answer or not question:
        return 0.0

    answer_tokens = set(_tokenize(answer))
    question_tokens = set(_tokenize(question))
    if not question_tokens:
        return 0.0

    overlap = answer_tokens & question_tokens
    return len(overlap) / len(question_tokens)


def evaluate_retrieval(
    queries: List[dict],
    retrieve_fn,
    k: int = 5,
) -> Dict[str, float]:
    """
    Evaluate a retrieval function over labeled queries.

    Each query dict expects:
      - "query": str
      - "relevant_ids": set/list of relevant document IDs

    retrieve_fn(query, k) -> list of document IDs
    """
    precisions: List[float] = []
    recalls: List[float] = []
    retrieved_lists: List[List[str]] = []
    relevant_sets: List[Set[str]] = []

    for item in queries:
        query = item["query"]
        relevant = set(item["relevant_ids"])
        retrieved = retrieve_fn(query, k)

        precisions.append(precision_at_k(retrieved, relevant, k))
        recalls.append(recall_at_k(retrieved, relevant, k))
        retrieved_lists.append(retrieved)
        relevant_sets.append(relevant)

    return {
        f"precision@{k}": sum(precisions) / len(precisions) if precisions else 0.0,
        f"recall@{k}": sum(recalls) / len(recalls) if recalls else 0.0,
        "mrr": mean_reciprocal_rank(retrieved_lists, relevant_sets),
    }


def _tokenize(text: str) -> List[str]:
    return [token for token in text.lower().split() if token.isalnum()]
