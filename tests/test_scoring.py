"""Unit tests for retrieval scoring utilities."""

from retrieval.scoring import distance_to_similarity, normalize_scores


def test_distance_to_similarity_ordering():
    """Lower distance must map to higher similarity."""
    close = distance_to_similarity(0.1)
    far = distance_to_similarity(2.0)
    assert close > far
    assert 0.0 < close <= 1.0


def test_normalize_scores_range():
    scores = {"a": 1.0, "b": 3.0, "c": 2.0}
    normalized = normalize_scores(scores)
    assert normalized["a"] == 0.0
    assert normalized["b"] == 1.0
    assert normalized["c"] == 0.5


if __name__ == "__main__":
    test_distance_to_similarity_ordering()
    test_normalize_scores_range()
    print("test_scoring: PASS")
